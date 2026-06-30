"""
最终用例属性解析模块

功能：解析生成的最终 case JSON 文件，提取每个参数的属性值。
      其中 dtype / format / dim_count / length 可直接提取，
      dim_value_profile 和 range_value_profile 需要通过具体取值反向推断。

推断必要性：
  参数组合生成阶段使用的是离散化模型名称（如 "Typical", "Has_Odd_Size"），
  但经过 case_generate 展开后，最终 JSON 中只保留了具体数值
  （如 shape=[3,7,13], range_values=[0.0, 1.0]），模型名称已丢失。
  因此需要通过启发式规则从具体数值反推其所属的离散化模型。
"""

import json
import math
from typing import Dict, List, Any, Optional

from generators.common_utils.logger_util import LazyLogger

logger = LazyLogger()


# ============================================================
# dim_value_profile 推断阈值
# 当一个维度 >= 2048 时，认为该 shape 属于 "Has_Large_Size" 策略
# ============================================================
LARGE_SIZE_THRESHOLD = 2048


def _is_prime(n: int) -> bool:
    """
    判断整数是否为质数，用于 Has_Odd_Size 策略推断。
    质数/奇数维度用于测试非对齐内存访问的正确性。
    """
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(math.isqrt(n)) + 1, 2):
        if n % i == 0:
            return False
    return True


def _infer_dim_value_profile(shape: List[int]) -> str:
    """
    从具体的 shape 值推断其所属的 dim_value_profile 离散化模型。

    推断优先级（从高到低）：
      1. Has_Size_1:     shape 包含维度值为 1  -> 测试广播机制
      2. Has_Large_Size:  shape 包含维度 >= 2048 -> 测试大块内存
      3. Has_Odd_Size:    shape 包含奇数/质数   -> 测试非对齐访问
      4. Typical:         以上均不满足           -> 典型 shape

    注意：一个 shape 可能同时满足多个条件（如 [1, 2048, 7]），
          此时按优先级取第一个匹配的模型。
    """
    if not shape:
        return "Typical"

    # 优先级 1: 包含维度值为 1
    if any(d == 1 for d in shape):
        return "Has_Size_1"

    # 优先级 2: 包含超大维度
    if any(d >= LARGE_SIZE_THRESHOLD for d in shape):
        return "Has_Large_Size"

    # 优先级 3: 包含奇数或质数
    if any(d % 2 == 1 or _is_prime(d) for d in shape):
        return "Has_Odd_Size"

    # 默认: 典型策略
    return "Typical"


def _infer_range_value_profile(range_values: Any, dtype: Optional[str] = None) -> str:
    """
    从具体的 range_values 值推断其所属的 range_value_profile 离散化模型。

    range_values 在最终 JSON 中的格式：
      - 标量参数: 单个数值（如 1e-5, 0, 1）
      - tensor 参数: [min, max] 区间列表（如 [0.0, 1.0]）
      - 异常值: 字符串 "NaN", "Infinity", "-Infinity"（经 abnormal_float_transfer 转换）

    推断规则：
      - 标量值:
        0    -> Zero
        1    -> One
        NaN  -> NaN
        +Inf -> PosInf
        -Inf -> NegInf
        极大值   -> Max
        极小值   -> Min
        负数     -> Neg
        正数     -> Pos
      - [min, max] 区间:
        [0, 0]      -> Zero
        [1, 1]      -> One
        含 NaN      -> NaN
        min >= 0    -> PosNormal（正数区间）
        max <= 0    -> NegNormal（负数区间）
        跨 0        -> Typical（混合符号）
    """
    if range_values is None:
        return "Typical"

    # ----------------------------------------------------------
    # Case 1: 字符串类型（NaN / Inf / -Inf）
    # 这些是由 abnormal_float_transfer 转换后的特殊值
    # ----------------------------------------------------------
    if isinstance(range_values, str):
        lower = range_values.lower()
        if lower in ("nan", "infinity", "-infinity"):
            mapping = {"nan": "NaN", "infinity": "PosInf", "-infinity": "NegInf"}
            return mapping.get(lower, "Typical")
        return "Typical"

    # ----------------------------------------------------------
    # Case 2: 布尔类型
    # ----------------------------------------------------------
    if isinstance(range_values, bool):
        return str(range_values)

    # ----------------------------------------------------------
    # Case 3: 数值类型（int / float）— 标量参数
    # ----------------------------------------------------------
    if isinstance(range_values, (int, float)):
        if range_values == 0:
            return "Zero"
        elif range_values == 1:
            return "One"
        elif math.isnan(range_values):
            return "NaN"
        elif math.isinf(range_values) and range_values > 0:
            return "PosInf"
        elif math.isinf(range_values) and range_values < 0:
            return "NegInf"
        elif range_values > 1e10:
            return "Max"
        elif range_values < -1e10:
            return "Min"
        elif range_values < 0:
            return "Neg"
        elif range_values > 0:
            return "Pos"
        else:
            return "Typical"

    # ----------------------------------------------------------
    # Case 4: 列表类型（[min, max] 或具体值列表）— tensor 参数
    # ----------------------------------------------------------
    if isinstance(range_values, (list, tuple)):
        if len(range_values) == 0:
            return "Typical"

        # 单元素列表递归处理
        if len(range_values) == 1:
            return _infer_range_value_profile(range_values[0], dtype)

        # 多元素列表：取 min/max 判断区间性质
        min_val, max_val = min(range_values), max(range_values)

        if min_val == 0 and max_val == 0:
            return "Zero"
        if min_val == 1 and max_val == 1:
            return "One"
        if any(math.isnan(v) if isinstance(v, (int, float)) else False for v in range_values):
            return "NaN"

        # 根据区间符号分类
        if min_val >= 0:
            return "PosNormal"   # 区间完全在正半轴
        if max_val <= 0:
            return "NegNormal"   # 区间完全在负半轴
        return "Typical"         # 区间跨 0

    return "Typical"


class CaseAttributeRecord:
    """
    单个参数在一个用例中的属性值记录。

    注意：dim_value_profile 和 range_value_profile 是推断值，
    并非最终 JSON 中的原始字段，仅供覆盖率统计使用。
    """

    def __init__(self, param_name: str, dtype: Optional[str], format: Optional[str],
                 dim_count: Optional[int], dim_value_profile: Optional[str],
                 range_value_profile: Optional[str], length: Optional[int]):
        self.param_name = param_name
        self.dtype = dtype                        # 直接提取
        self.format = format                      # 直接提取
        self.dim_count = dim_count                # 从 shape 长度提取
        self.dim_value_profile = dim_value_profile      # 从 shape 值推断
        self.range_value_profile = range_value_profile  # 从 range_values 推断
        self.length = length                      # 直接提取


def _extract_one(inp: dict, records: Dict[str, CaseAttributeRecord]):
    """
    从一个 input 字典中提取单个参数的属性值。

    处理步骤：
      1. 获取参数名（唯一标识）
      2. 直接提取 dtype / format / length
      3. 从 shape 列表长度得到 dim_count
      4. 从 shape 具体值推断 dim_value_profile
      5. 从 range_values 具体值推断 range_value_profile
    """
    name = inp.get("name")
    if not name:
        return

    dtype = inp.get("dtype")
    fmt = inp.get("format")
    shape = inp.get("shape")
    range_values = inp.get("range_values")
    length = inp.get("length")

    # dim_count: shape 列表的长度（仅 tensor 参数有）
    dim_count = len(shape) if isinstance(shape, list) and shape is not None else None

    # dim_value_profile: 从 shape 具体数值推断
    dim_value_profile = _infer_dim_value_profile(shape) if isinstance(shape, list) else None

    # range_value_profile: 从 range_values 具体数值推断
    range_value_profile = _infer_range_value_profile(range_values, dtype) if range_values is not None else None

    records[name] = CaseAttributeRecord(
        param_name=name, dtype=dtype, format=fmt,
        dim_count=dim_count, dim_value_profile=dim_value_profile,
        range_value_profile=range_value_profile, length=length,
    )

    logger.debug(f"  Extracted param '{name}': dtype={dtype}, format={fmt}, "
                 f"dim_count={dim_count}, dim_profile={dim_value_profile}, "
                 f"range_profile={range_value_profile}, length={length}")


def parse_single_case(case_obj: dict) -> Dict[str, CaseAttributeRecord]:
    """
    解析单个 case 的所有 input 参数，返回 {参数名: CaseAttributeRecord} 字典。

    CaseConfig.inputs 的结构可能是：
      - [InputCaseConfig, ...]          — 普通列表
      - [[InputCaseConfig], ...]        — 嵌套列表（如 tensor_list 类型参数）
    本函数递归展开所有层级。
    """
    records = {}
    inputs = case_obj.get("inputs") or []
    for inp in inputs:
        if isinstance(inp, list):
            # 嵌套列表：递归处理每个子元素
            for sub in inp:
                _extract_one(sub, records)
        elif isinstance(inp, dict):
            _extract_one(inp, records)
    return records


def parse_case_file(case_json_path: str) -> List[Dict[str, CaseAttributeRecord]]:
    """
    解析生成的最终 case JSON 文件，返回所有 case 的参数属性记录列表。

    Args:
        case_json_path: 最终生成的 case JSON 文件路径

    Returns:
        List[Dict[str, CaseAttributeRecord]]:
          每个元素是一个 case 的 {参数名: 属性记录} 字典
    """
    logger.info(f"Parsing case file: {case_json_path}")
    with open(case_json_path, "r", encoding="utf-8") as f:
        cases = json.load(f)

    # 单个 case 或 case 列表的统一处理
    if not isinstance(cases, list):
        cases = [cases]

    logger.info(f"Total cases loaded: {len(cases)}")

    parsed = []
    for idx, case_obj in enumerate(cases):
        records = parse_single_case(case_obj)
        logger.debug(f"  Case[{idx}]: extracted {len(records)} params")
        parsed.append(records)

    if parsed:
        sample_params = list(parsed[0].keys())
        logger.info(f"Case parsing done. Params found in first case: {sample_params}")
    else:
        logger.warning("Case parsing done. No valid params extracted from any case.")

    return parsed
