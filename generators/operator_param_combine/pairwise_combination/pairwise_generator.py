"""
Pairwise 组合测试核心算法。

设计背景
--------
旧方案对每个算子参数的 7 个属性（dtype, format, dim_count, dim_value_profile,
allowed_range_value, is_optional, is_operator_param）独立随机取值，组合空间呈
指数爆炸，无法保证关键的 2-wise 组合覆盖率。

新方案采用 Pairwise (All-Pairs) 贪婪算法：假设绝大多数缺陷由某参数某值与另一参数
某值的交互触发，覆盖所有 2-pair 即可在测试成本与缺陷检出率之间取得最佳平衡。

核心流程
--------
1. 域提取（AttributeDomain 类）：
   从 OperatorRule 提取每个参数每个属性的所有合法取值。若存在
   dtype_support_description（多参数 dtype 固定组合列表），从中随机选一组。

2. 约束域剪裁（ConstraintProcessor 类）：
   - Equality 约束（type_equality / shape_equality / format_equality）：
     对组内所有参数取值域求交集，缩减域范围。
   - 非 Equality 约束（value_dependency / presence_dependency）：
     留待后验验证，不做域裁剪。

3. 共享属性合并（_build_shared_groups）：
   - 为每个 equality group 创建一个 shared attribute key
     （如 "dtype_eq_14016708.dtype"）
   - 在整个 pairwise 算法中，shared key 替代原始属性名参与 2-pair 收集、
     贪心选择和 coverage 计数
   - 输出阶段通过 shared_to_raw 映射回原始属性名
   - 效果：equality 组内所有参数在该属性上永远取相同值，无需后处理修正

4. 2-pair 集合收集（collect_pairs）：
   枚举 (param1, attr1, value1, param2, attr2, value2) 所有组合去重，
   即为待覆盖的全部 2-pair。

5. 贪婪迭代构建（generate）：
   每轮生成多个候选 case：
   - 随机打乱参数顺序
   - 对每个参数逐一选择能覆盖最多未覆盖 pair 的取值
   - 取覆盖数最多的候选加入结果集，标记已覆盖 pair
   - 兜底：若无法找到能覆盖新 pair 的候选，则随机生成一个 case 并继续
   - 终止：所有 pair 均被覆盖，或已达 max_cases 上限

6. 输出转换（_case_effective_to_raw）：
   - shared key → raw attr name 映射还原
   - 确保下游 OperatorParameterCombination 能正确消费

算法特性
--------
- 终止性：每轮至少覆盖 1 个新 pair，pair 集有限，必然终止。
- 近似最优：局部贪心 + 多候选搜索，实践中可达 95%+ 覆盖率。
- 约束后验：value_dependency 等复杂约束由下游 Z3 求解器处理。
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Any, Dict, List, Set, Tuple

from agent.generators.common_utils.logger_util import LazyLogger
from agent.generators.data_definition.constants import ParamModelConfig
from agent.generators.operator_param_combine.pairwise_combination.attribute_domain import (
    AttributeDomain,
    ATTR_DTYPE, ATTR_FORMAT, ATTR_DIMENSIONS,
    ATTR_RANGE_VALUE, ATTR_ARRAY_LENGTH, ATTR_IS_OPTIONAL, ATTR_IS_OPERATOR_PARAM,
)
from agent.generators.operator_param_combine.pairwise_combination.constraint_filter import ConstraintProcessor

logger = LazyLogger()


class PairwiseCombinationGenerator:
    def __init__(self, attr_domain: AttributeDomain, constraint_processor: ConstraintProcessor):
        self.attr_domain = attr_domain
        self.constraint_processor = constraint_processor
        self.params: List[str] = []

        self.shared_attrs: Dict[str, Dict[str, List[Any]]] = {}
        self.shared_to_raw: Dict[str, str] = {}
        self.param_to_shared: Dict[str, Dict[str, str]] = {}
        self.param_private_attrs: Dict[str, Dict[str, List[Any]]] = {}

        self._build_param_attributes()

    def _build_param_attributes(self):
        self.constraint_processor.process()
        self.params = self.attr_domain.get_effective_params()

        dtype_domains = {}
        format_domains = {}
        dim_domains = {}
        profile_domains = {}

        for p in self.params:
            domain = self.attr_domain.param_domains[p]
            dtype_domains[p] = domain.get(ATTR_DTYPE, [ParamModelConfig.DEFAULT_PARAM_DTYPE_DTYPE_IN_ORIGINAL_DOC])
            format_domains[p] = domain.get(ATTR_FORMAT, [])
            dim_domains[p] = domain.get(ATTR_DIMENSIONS, [ParamModelConfig.DEFAULT_TENSOR_SHAPE_DIM])
            profile_domains[p] = self.attr_domain.get_dim_value_profile_domain()

        dtype_domains = self.constraint_processor.apply_type_equality(dtype_domains)
        format_domains = self.constraint_processor.apply_format_equality(format_domains)
        dim_domains, profile_domains = self.constraint_processor.apply_shape_equality(dim_domains, profile_domains)

        dtype_combos = self.attr_domain.get_dtype_combinations()
        use_dtype_map = None
        if dtype_combos:
            picked = random.choice(dtype_combos)
            use_dtype_map = {p: [picked.get(p, dtype_domains[p][0])] for p in self.params}

        for p in self.params:
            domain = self.attr_domain.param_domains[p]
            is_tensor = self.attr_domain.is_tensor_param(p)
            is_list = self.attr_domain.is_list_param(p)
            ptype = self.attr_domain.get_param_type(p)

            attrs: Dict[str, List[Any]] = {}
            attrs["param_type"] = [ptype]
            attrs[ATTR_DTYPE] = use_dtype_map.get(p, dtype_domains[p]) if use_dtype_map else dtype_domains[p]

            if format_domains[p]:
                attrs[ATTR_FORMAT] = format_domains[p]

            if is_tensor:
                attrs["dim_count"] = dim_domains[p]
                attrs["dim_value_profile"] = profile_domains[p]

            range_vals = domain.get(ATTR_RANGE_VALUE, [])
            if range_vals:
                expanded = self._expand_range_values(range_vals, p)
                if expanded:
                    attrs["range_value_profile"] = expanded
                else:
                    rd = list(ParamModelConfig.FLOAT_TENSOR_DATA_PROFILE + ParamModelConfig.INT_TENSOR_DATA_PROFILE)
                    attrs["range_value_profile"] = rd
            else:
                rd = list(ParamModelConfig.FLOAT_TENSOR_DATA_PROFILE + ParamModelConfig.INT_TENSOR_DATA_PROFILE)
                attrs["range_value_profile"] = rd

            if is_list:
                attrs["length"] = domain.get(ATTR_ARRAY_LENGTH, [ParamModelConfig.DEFAULT_LIST_LENGTH])

            self.param_private_attrs[p] = attrs

        self._build_shared_groups()

    def _build_shared_groups(self):
        """
        构建共享属性组，将 equality 约束转换为 shared attribute key。

        策略说明：
        假设 param_A 和 param_B 之间存在 type_equality 约束（dtype 必须相等），
        传统做法是先枚举再过滤冲突 case，但在 pairwise 场景中这会导致 coverage
        计数虚高——因为 A.dtype=FLOAT32 和 B.dtype=BFLOAT16 的 pair 被计入
        "已覆盖" 后又被 V&V 阶段删除。

        本方法的核心思路：在 pair 收集和 coverage 计数层面就将 A.dtype 和 B.dtype
        视为同一个因子。具体做法：
        1. 为每个 equality group 创建一个 shared key，如 "dtype_eq_14016708.dtype"
        2. collect_pairs() 时使用 shared key 而非原始属性名收集 pair
        3. _build_case_from_uncovered() 时共享 key 在所有相关参数上取同一值
        4. 最终输出时通过 self.shared_to_raw 映射回原始名

        效果：equality 约束无需后验修正，且在 pairwise 层面天然成立。
        """
        groups = {
            "dtype": self.constraint_processor.type_equal_groups,
            "format": self.constraint_processor.format_equal_groups,
            "shape": self.constraint_processor.shape_equal_groups,
        }
        eq_attr_names = {
            "dtype": [ATTR_DTYPE],
            "format": [ATTR_FORMAT],
            "shape": ["dim_count", "dim_value_profile"],
        }

        for prefix, group_list in groups.items():
            for g in group_list:
                members = [p for p in g if p in self.param_private_attrs]
                if len(members) < 2:
                    continue
                shared_key = f"{prefix}_eq_{id(g)}"
                attr_names = eq_attr_names[prefix]
                for attr_name in attr_names:
                    all_vals: List[Any] = []
                    seen = set()
                    for m in members:
                        vals = self.param_private_attrs[m].get(attr_name, [])
                        for v in vals:
                            key = str(v)
                            if key not in seen:
                                seen.add(key)
                                all_vals.append(v)
                    if all_vals:
                        effective_key = f"{shared_key}.{attr_name}"
                        self.shared_attrs[effective_key] = all_vals
                        self.shared_to_raw[effective_key] = attr_name
                        for m in members:
                            if attr_name not in self.param_to_shared.setdefault(m, {}):
                                self.param_to_shared[m][attr_name] = effective_key

    @staticmethod
    def _expand_range_values(raw_values: List, param_name: str) -> List[Any]:
        expanded = []
        for v in raw_values:
            if isinstance(v, list) and len(v) == 2:
                expanded.append(v[0])
                expanded.append(v[1])
            elif isinstance(v, (int, float, bool)):
                expanded.append(v)
        return expanded

    def get_effective_name(self, param: str, raw_attr: str) -> str:
        return self.param_to_shared.get(param, {}).get(raw_attr, raw_attr)

    def get_effective_values(self, param: str, raw_attr: str) -> List[Any]:
        shared_key = self.param_to_shared.get(param, {}).get(raw_attr)
        if shared_key and shared_key in self.shared_attrs:
            return self.shared_attrs[shared_key]
        return self.param_private_attrs.get(param, {}).get(raw_attr, [])

    def get_all_effective_attrs(self, param: str) -> Dict[str, List[Any]]:
        result: Dict[str, List[Any]] = {}
        for raw_name in self.param_private_attrs.get(param, {}):
            eff_name = self.get_effective_name(param, raw_name)
            result[eff_name] = self.get_effective_values(param, raw_name)
        return result

    def collect_pairs(self) -> Set[Tuple]:
        """
        枚举所有待覆盖的 2-pair 组合。

        一个 2-pair 定义为 (param_a, attr_a, value_a, param_b, attr_b, value_b) ——
        即参数 A 的某属性的某取值 与 参数 B 的某属性的某取值 的配对。

        枚举范围：
        - 不同参数的所有属性之间（如 x.dtype × y.dim_count）
        - 同一参数的不同属性之间（如 x.dtype × x.format）
        - 不包含同一参数的同一属性自配对（如 x.dtype × x.dtype）

        当 equality group 存在时，attr_a/attr_b 使用 shared key（如 "dtype_eq_xx.dtype"）
        而非原始属性名，保证约束组内参数在该属性上视为同一因子。
        """
        pairs: Set[Tuple] = set()
        all_eff: Dict[str, Dict[str, List[Any]]] = {}
        for p in self.params:
            all_eff[p] = self.get_all_effective_attrs(p)

        for i in range(len(self.params)):
            p1 = self.params[i]
            attrs1 = all_eff[p1]
            for eff1, values1 in attrs1.items():
                for v1 in values1:
                    for j in range(i, len(self.params)):
                        p2 = self.params[j]
                        attrs2 = all_eff[p2]
                        for eff2, values2 in attrs2.items():
                            for v2 in values2:
                                if p1 == p2 and eff1 == eff2:
                                    continue
                                pair = (p1, eff1, v1, p2, eff2, v2)
                                pairs.add(pair)
        return pairs

    def generate(self, max_cases: int | None = None) -> List[Dict[str, Dict[str, Any]]]:
        """
        贪婪迭代生成覆盖全部 2-pair 的测试用例集。

        算法步骤：
        1. 调用 collect_pairs() 获取所有待覆盖的 2-pair
        2. 循环直到 uncovered 集合为空：
           a. 候选搜索：尝试 k 个候选，k = clamp(50, total_pairs // remaining)
              即剩余工作量越大，搜索越充分；越靠近收尾搜索越收敛。
           b. 每个候选由 _build_case_from_uncovered 构建：
              - 随机打乱参数顺序
              - 对每个参数，选择覆盖最多未覆盖 pair 的取值
              - 同一 shared key 在所有参数间取同一个值
           c. 选覆盖最多的候选加入结果集，标记其 pair 为已覆盖
           d. 若所有候选均无法覆盖新 pair（degenerate case），
              用 _build_fallback_case 随机兜底并继续
        3. 遍历所有 raw_case，通过 _case_effective_to_raw 将 shared key
           映射回原始属性名后返回

        参数：
            max_cases：可选上限，防止生成过多 case（通常不限）

        返回：
            List[Dict[str, Dict[str, Any]]]，每个元素为
            { param_name: { raw_attr_name: value } }
        """
        uncovered = self.collect_pairs()
        total_pairs = len(uncovered)
        logger.info(f"Total 2-pair combinations to cover: {total_pairs}")

        raw_cases: List[Dict[str, Dict[str, Any]]] = []

        while uncovered:
            best_case = None
            best_coverage = 0

            for _ in range(min(50, max(1, total_pairs // max(1, len(uncovered))))):
                candidate = self._build_case_from_uncovered(uncovered)
                if candidate is None:
                    continue
                coverage = self._count_coverage(candidate, uncovered)
                if coverage > best_coverage:
                    best_coverage = coverage
                    best_case = candidate

            if best_case is None or best_coverage == 0:
                fallback = self._build_fallback_case()
                if fallback is None:
                    break
                best_case = fallback
                best_coverage = self._count_coverage(best_case, uncovered)

            if best_case is None or best_coverage == 0:
                break

            self._mark_covered(best_case, uncovered)
            raw_cases.append(best_case)

            if max_cases is not None and len(raw_cases) >= max_cases:
                break

        covered_count = total_pairs - len(uncovered)
        if total_pairs > 0:
            logger.info(f"Generated {len(raw_cases)} test cases, covering {covered_count}/{total_pairs} "
                        f"2-pair combinations ({covered_count / total_pairs * 100:.1f}%)")

        converted = []
        for raw_case in raw_cases:
            converted.append(self._case_effective_to_raw(raw_case))
        return converted

    def _case_effective_to_raw(self, eff_case: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        raw_case: Dict[str, Dict[str, Any]] = {}
        for p, eff_attrs in eff_case.items():
            raw_case[p] = {}
            for eff_name, v in eff_attrs.items():
                raw_name = self.shared_to_raw.get(eff_name, eff_name)
                raw_case[p][raw_name] = v
        return raw_case

    def _build_case_from_uncovered(self, uncovered: Set[Tuple]) -> Dict[str, Dict[str, Any]] | None:
        """
        贪婪构建单个候选 case。

        策略：逐一为参数的所有属性选择取值，使该 case 覆盖尽可能多的未覆盖 2-pair。

        步骤：
        1. 随机打乱参数顺序（引入随机性，不同候选覆盖不同 pair）
        2. 对每个参数：
           a. 遍历其所有属性（若属性已被 shared_picked 固定则跳过）
           b. 对该属性的每个候选值，计算其与已赋值参数之间能产生的
              未覆盖 pair 数量（双向：a×b 和 b×a 均计数）
           c. 选取得分最高的 (属性, 取值) pair
           d. 若该属性是 shared_attr，将其值记入 shared_picked（同一 group 的其他参数继承此值）
        3. 补全：对尚未赋值的属性，按优先级填充：
           a. 已在 shared_picked 中的值（equality group 继承）
           b. 属于 shared_attr 但尚未 pick 的组（从候选集中随机选）
           c. 参数私有属性（从候选集中随机选）

        此方法不检查 value_dependency / presence_dependency 等复杂约束，
        因为这些约束需要具体数值上下文，留待下游 Z3 求解器处理。
        """
        assigned: Dict[str, Dict[str, Any]] = {}
        shared_picked: Dict[str, Any] = {}
        assigned_params: Set[str] = set()

        param_order = list(self.params)
        random.shuffle(param_order)

        for p in param_order:
            eff_attrs = self.get_all_effective_attrs(p)
            if not eff_attrs:
                continue

            best_eff = None
            best_val = None
            best_score = -1

            for eff_name, values in eff_attrs.items():
                if eff_name in shared_picked:
                    continue
                for v in values:
                    score = 0
                    for other_p in assigned_params:
                        for other_eff, other_v in assigned[other_p].items():
                            pair = (p, eff_name, v, other_p, other_eff, other_v)
                            if pair in uncovered:
                                score += 1
                            pair_rev = (other_p, other_eff, other_v, p, eff_name, v)
                            if pair_rev in uncovered:
                                score += 1
                    if score > best_score:
                        best_score = score
                        best_eff = eff_name
                        best_val = v

            if best_eff is not None:
                if best_eff in self.shared_attrs:
                    shared_picked[best_eff] = best_val
                if p not in assigned:
                    assigned[p] = {}
                assigned[p][best_eff] = best_val
                assigned_params.add(p)

        for p in self.params:
            if p not in assigned:
                assigned[p] = {}
            eff_attrs = self.get_all_effective_attrs(p)
            for eff_name in eff_attrs:
                if eff_name not in assigned[p]:
                    if eff_name in shared_picked:
                        assigned[p][eff_name] = shared_picked[eff_name]
                    elif eff_name in self.shared_attrs:
                        vals = self.shared_attrs[eff_name]
                        if vals:
                            v = random.choice(vals)
                            assigned[p][eff_name] = v
                            shared_picked[eff_name] = v
                    else:
                        vals = eff_attrs[eff_name]
                        if vals:
                            assigned[p][eff_name] = random.choice(vals)

        return assigned

    def _build_fallback_case(self) -> Dict[str, Dict[str, Any]] | None:
        assigned: Dict[str, Dict[str, Any]] = {}
        shared_picked: Dict[str, Any] = {}

        for p in self.params:
            eff_attrs = self.get_all_effective_attrs(p)
            for eff_name in eff_attrs:
                if eff_name in self.shared_attrs and eff_name not in shared_picked:
                    vals = self.shared_attrs[eff_name]
                    if vals:
                        shared_picked[eff_name] = random.choice(vals)

        for p in self.params:
            assigned[p] = {}
            eff_attrs = self.get_all_effective_attrs(p)
            for eff_name, values in eff_attrs.items():
                if eff_name in shared_picked:
                    assigned[p][eff_name] = shared_picked[eff_name]
                elif values:
                    assigned[p][eff_name] = random.choice(values)

        return assigned

    @staticmethod
    def _count_coverage(case: Dict[str, Dict[str, Any]], uncovered: Set[Tuple]) -> int:
        count = 0
        params = list(case.keys())
        for i in range(len(params)):
            p1 = params[i]
            attrs1 = case[p1]
            for eff1, v1 in attrs1.items():
                for j in range(i, len(params)):
                    p2 = params[j]
                    attrs2 = case[p2]
                    for eff2, v2 in attrs2.items():
                        if p1 == p2 and eff1 == eff2:
                            continue
                        pair = (p1, eff1, v1, p2, eff2, v2)
                        if pair in uncovered:
                            count += 1
        return count

    @staticmethod
    def _mark_covered(case: Dict[str, Dict[str, Any]], uncovered: Set[Tuple]):
        params = list(case.keys())
        for i in range(len(params)):
            p1 = params[i]
            attrs1 = case[p1]
            for eff1, v1 in attrs1.items():
                for j in range(i, len(params)):
                    p2 = params[j]
                    attrs2 = case[p2]
                    for eff2, v2 in attrs2.items():
                        if p1 == p2 and eff1 == eff2:
                            continue
                        pair = (p1, eff1, v1, p2, eff2, v2)
                        uncovered.discard(pair)
