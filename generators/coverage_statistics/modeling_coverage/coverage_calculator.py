"""
覆盖覆盖率计算模块

功能：基于参数属性域定义（可能取值全集）和实际用例的属性值记录，
      计算 1-pair 和 2-pair 覆盖率。

覆盖定义：
  1. 1-pair 覆盖（Each-Choice 覆盖）：
     每个参数的每个属性的每种可能取值，是否至少被一个用例覆盖。

  2. 2-pair 覆盖（Pairwise 覆盖）：
     两个不同参数的同一属性类型的所有可能取值组合，是否至少被一个用例覆盖。
     例如：(x1.dtype=fp16, x2.dtype=fp32) 是一个 2-pair 组合。

计算公式：
  覆盖率 = 已覆盖的取值数（或组合数） / 理论上可能的取值数（或组合数）
"""

import itertools
from typing import Dict, List

from agent.generators.common_utils.logger_util import LazyLogger
from .attribute_defs import OperatorAttributeDomain, ParamAttributeDomain
from .case_parser import CaseAttributeRecord

logger = LazyLogger()


# ============================================================
# 参与覆盖率统计的属性列表
# ============================================================

# 1-pair 覆盖涉及的属性
# dim_value_profile 仅对 tensor 参数有效
# length 仅对 list 类型参数有效
ATTR_NAMES_1PAIR = ["dtype", "format", "dim_count", "dim_value_profile", "range_value_profile", "length"]

# 2-pair 覆盖涉及的属性（仅统计可枚举的离散化属性）
# dim_count 和 length 未包含在内，原因是：
#   - dim_count 的取值范围通常在 1~8，配对空间巨大但实际约束限制严格
#   - length 仅对少量 list 参数有效，跨参数时多数不存在
ATTR_NAMES_2PAIR = ["dtype", "format", "dim_value_profile", "range_value_profile"]


class CoverageResult:
    """
    覆盖率计算结果容器。

    Attributes:
        one_pair: Dict[参数名, Dict[属性名, dict]]
          示例: {"x1": {"dtype": {"count": 3, "total": 4, "rate": 0.75, ...}}}
        two_pair: Dict[属性对标识, dict]
          示例: {"x1.dtype × x2.dtype": {"count": 9, "total": 16, "rate": 0.5625, ...}}
    """

    def __init__(self):
        self.one_pair: Dict[str, Dict[str, dict]] = {}
        self.two_pair: Dict[str, Dict[str, dict]] = {}


# ============================================================
# 内部辅助函数
# ============================================================

def _get_domain_values(domain: ParamAttributeDomain, attr_name: str) -> list:
    """
    获取参数指定属性的域定义（可能取值列表）。

    通过属性名分发到 ParamAttributeDomain 的对应字段。
    """
    mapping = {
        "dtype": domain.dtype_values,
        "format": domain.format_values,
        "dim_count": domain.dim_count_values,
        "dim_value_profile": domain.dim_value_profile_values,
        "range_value_profile": domain.range_value_profile_values,
        "length": domain.length_values,
    }
    return mapping.get(attr_name, [])


def _get_case_value(record: CaseAttributeRecord, attr_name: str):
    """
    从用例属性记录中获取指定属性的实际取值。
    """
    return getattr(record, attr_name, None)


# ============================================================
# 1-pair 覆盖率计算
# ============================================================

def compute_one_pair_coverage(
    case_records_list: List[Dict[str, CaseAttributeRecord]],
    domain: OperatorAttributeDomain,
) -> Dict[str, Dict[str, dict]]:
    """
    计算 1-pair 覆盖率。

    处理逻辑：
      1. 遍历每个参数及其每个属性
      2. 跳过不适用的属性（如非 tensor 参数不统计 dim_value_profile）
      3. 从域定义中获取可能取值列表（分母）
      4. 在所有用例中收集该参数该属性的实际已覆盖取值（分子）
      5. 计算覆盖率 = 已覆盖取值数 / 可能取值总数

    Args:
        case_records_list: 所有用例的参数属性记录列表
        domain: 算子属性域定义

    Returns:
        Dict[参数名, Dict[属性名, {覆盖统计详情}]]
    """
    logger.info("Computing 1-pair coverage...")
    result = {}

    for param_name, param_domain in domain.params.items():
        param_result = {}
        logger.debug(f"  Processing param: {param_name} (type={param_domain.param_type})")

        for attr_name in ATTR_NAMES_1PAIR:
            # --- 过滤不适用的属性 ---
            # dim_value_profile 只有 tensor 参数有
            if attr_name == "dim_value_profile" and not param_domain.is_tensor():
                continue
            # length 只有 list 类型参数有
            if attr_name == "length" and not param_domain.is_list_type():
                continue

            # --- 获取可能取值列表（分母） ---
            domain_vals = _get_domain_values(param_domain, attr_name)
            if not domain_vals:
                logger.debug(f"    [{attr_name}] no domain values defined, skipped")
                continue

            # --- 在所有用例中收集已覆盖取值（分子） ---
            covered = set()
            for records in case_records_list:
                rec = records.get(param_name)
                if rec is None:
                    # 该参数在 case 中不存在（可能是 optional 参数被跳过）
                    continue
                val = _get_case_value(rec, attr_name)
                if val is not None:
                    covered.add(val)

            # --- 计算覆盖率 ---
            total = len(domain_vals)
            cov_count = sum(1 for v in domain_vals if v in covered)
            rate = cov_count / total if total > 0 else 0.0

            param_result[attr_name] = {
                "covered": sorted(c if not isinstance(c, bool) else c for c in covered if c in domain_vals),
                "missed": sorted(m if not isinstance(m, bool) else m for m in domain_vals if m not in covered),
                "count": cov_count,
                "total": total,
                "rate": rate,
                "domain_list": domain_vals,
            }

            logger.debug(f"    [{attr_name}] {cov_count}/{total} covered ({rate:.1%})")

        result[param_name] = param_result

    # 汇总
    all_rates = [v["rate"] for p in result.values() for v in p.values()]
    if all_rates:
        avg = sum(all_rates) / len(all_rates)
        logger.info(f"1-pair coverage done. Avg rate: {avg:.2%}")

    return result


# ============================================================
# 2-pair 覆盖率计算
# ============================================================

def compute_two_pair_coverage(
    case_records_list: List[Dict[str, CaseAttributeRecord]],
    domain: OperatorAttributeDomain,
) -> Dict[str, dict]:
    """
    计算 2-pair 覆盖率（跨参数同属性类型配对覆盖）。

    处理逻辑：
      1. 对所有参数做两两组合 C(n, 2)
      2. 对于每个参数对，遍历 ATTR_NAMES_2PAIR 中的属性
      3. 从两个参数的域定义中分别获取可能取值列表，计算笛卡尔积（分母）
      4. 在所有用例中收集实际出现的取值对（分子）
      5. 计算覆盖率 = 已覆盖组合数 / 可能组合总数

    Args:
        case_records_list: 所有用例的参数属性记录列表
        domain: 算子属性域定义

    Returns:
        Dict[属性对标识, {覆盖统计详情}]
    """
    logger.info("Computing 2-pair coverage...")
    result = {}
    param_names = list(domain.params.keys())
    total_pairs = len(param_names) * (len(param_names) - 1) // 2

    logger.info(f"Total param pairs to process: {total_pairs}")

    # --- 对所有参数做两两组合 ---
    pair_count = 0
    for p1_name, p2_name in itertools.combinations(param_names, 2):
        d1, d2 = domain.params[p1_name], domain.params[p2_name]
        pair_count += 1

        for attr_name in ATTR_NAMES_2PAIR:
            # --- 构建属性对标识 ---
            key = f"{p1_name}.{attr_name} × {p2_name}.{attr_name}"

            # --- 获取两个参数的域定义 ---
            dom1 = _get_domain_values(d1, attr_name)
            dom2 = _get_domain_values(d2, attr_name)
            if not dom1 or not dom2:
                continue

            # --- 过滤：dim_value_profile 仅对 tensor 参数有意义 ---
            if attr_name == "dim_value_profile" and not (d1.is_tensor() and d2.is_tensor()):
                continue

            # --- 计算全组合空间（分母） ---
            total = len(dom1) * len(dom2)

            # --- 在所有用例中收集实际出现的取值对（分子） ---
            covered_pairs = set()
            for records in case_records_list:
                r1 = records.get(p1_name)
                r2 = records.get(p2_name)
                if r1 is None or r2 is None:
                    # 某个参数可能因 optional 被跳过
                    continue
                v1 = _get_case_value(r1, attr_name)
                v2 = _get_case_value(r2, attr_name)
                if v1 is not None and v2 is not None:
                    covered_pairs.add((v1, v2))

            # --- 统计在域范围内的已覆盖组合数 ---
            cov_count = sum(1 for a in dom1 for b in dom2 if (a, b) in covered_pairs)
            rate = cov_count / total if total > 0 else 0.0

            result[key] = {
                "covered_pairs": sorted((a, b) for a, b in covered_pairs if a in dom1 and b in dom2),
                "count": cov_count,
                "total": total,
                "rate": rate,
            }

            logger.debug(f"  [{key}] {cov_count}/{total} covered ({rate:.1%})")

        if pair_count % 10 == 0:
            logger.info(f"  2-pair progress: {pair_count}/{total_pairs} param pairs processed")

    # 汇总
    all_rates = [v["rate"] for v in result.values()]
    if all_rates:
        avg = sum(all_rates) / len(all_rates)
        logger.info(f"2-pair coverage done. Total attr-pairs: {len(result)}, Avg rate: {avg:.2%}")

    return result


# ============================================================
# 总入口
# ============================================================

def compute_coverage(
    case_records_list: List[Dict[str, CaseAttributeRecord]],
    domain: OperatorAttributeDomain,
) -> CoverageResult:
    """
    计算完整的覆盖率结果（1-pair + 2-pair）。

    这是 coverage_calculator 模块的唯一外部接口。
    """
    logger.info("=" * 60)
    logger.info(f"Starting coverage computation for operator: {domain.operator_name}")
    logger.info(f"  Params count: {len(domain.params)}")
    logger.info(f"  Cases count: {len(case_records_list)}")
    logger.info("=" * 60)

    result = CoverageResult()
    result.one_pair = compute_one_pair_coverage(case_records_list, domain)
    result.two_pair = compute_two_pair_coverage(case_records_list, domain)

    logger.info("=" * 60)
    logger.info("Coverage computation completed")
    logger.info("=" * 60)
    return result
