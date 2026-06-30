"""
参数属性可能取值定义模块

功能：从算子约束 JSON 文件中提取每个参数的属性可能取值全集（Domain），
      为后续覆盖率计算提供"分母"（可能取值/可能组合的总空间）。

关键逻辑：
  1. 读取算子约束 JSON（OperatorRule 的原始格式）
  2. 针对每个参数，提取 dtype / format / dim_count / length 的允许取值
  3. dim_value_profile 使用全局离散化模型列表（固定 4 种策略）
  4. range_value_profile 根据参数 dtype 选择对应的浮点/整数/布尔离散化模型
"""

import json
import os
from typing import Dict, List, Any, Optional

from generators.common_utils.logger_util import LazyLogger

logger = LazyLogger()


# ============================================================
# 全局离散化模型定义
# 对应 data_definition/constants.py 中 ParamModelConfig 的取值
# ============================================================

# shape 取值策略的离散化模型（4 种可枚举策略）
DIM_VALUE_PROFILE_LIST = ["Has_Large_Size", "Has_Size_1", "Has_Odd_Size", "Typical"]

# 浮点类型参数的数值范围离散化模型（9 种）
FLOAT_TENSOR_DATA_PROFILE = ["Typical", "PosNormal", "NegNormal", "Zero", "One", "NaN", "PosInf", "NegInf", "SubNormal"]

# 整数类型参数的数值范围离散化模型（6 种）
INT_TENSOR_DATA_PROFILE = ["Pos", "Neg", "Zero", "One", "Max", "Min"]

# bool 类型参数的取值
BOOL_DATA_PROFILE = [True, False]

# format 属性全局可能取值（来自 ParamModelConfig.FORMAT_VALUE_LIST）
FORMAT_VALUE_LIST = ["nchw", "nhwc", "nc", "cn", "fractal_nz", "nchw16", "nchw8", "chwn8", "nhwc8", "nhwc16"]

# 当算子约束 JSON 中未定义 dimensions 时，dim_count 的默认范围
DEFAULT_DIM_COUNT_MIN = 1
DEFAULT_DIM_COUNT_MAX = 8


def _extract_value(raw_value: Any) -> Any:
    """
    从算子约束 JSON 的字段中提取实际值。
    算子约束 JSON 中字段格式可能为 {"value": ...} 的包装结构，
    也可能是直接量，此函数做统一解包。
    """
    if isinstance(raw_value, dict) and "value" in raw_value:
        return raw_value["value"]
    return raw_value


class ParamAttributeDomain:
    """
    单个参数的所有属性可能取值定义。

    每个参数有 6 个可枚举属性（部分属性因参数类型不同而存在/不存在）：
      - dtype: 数据类型（如 fp16, fp32, int8...）
      - format: 数据格式（如 ND, fractal_nz...）
      - dim_count: shape 维度数量（仅 tensor 参数）
      - dim_value_profile: shape 取值策略（仅 tensor 参数，固定 4 种离散模型）
      - range_value_profile: 数值范围模型（根据 dtype 确定浮点/整数/布尔模型）
      - length: 数组长度（仅 list 类型参数）
    """

    def __init__(self, param_name: str, param_type: str):
        self.param_name = param_name
        # param_type 取值为 "tensor" / "scalar" / "list" / "attr" 等（ATK 类型）
        self.param_type = param_type

        # 以下列表将从算子约束 JSON 中填充
        self.dtype_values: List[str] = []
        self.format_values: List[str] = []
        self.dim_count_values: List[int] = []
        # dim_value_profile 固定为 4 种离散化模型，不依赖算子约束 JSON
        self.dim_value_profile_values: List[str] = list(DIM_VALUE_PROFILE_LIST)
        # range_value_profile 后续根据 dtype 选择对应的离散化模型
        self.range_value_profile_values: List[str] = []
        self.length_values: List[int] = []

    def is_tensor(self) -> bool:
        """判断是否为 tensor 类型参数（只有 tensor 才有 shape 相关属性）"""
        return self.param_type in ("tensor", "tensors")

    def is_list_type(self) -> bool:
        """判断是否为 list 类型参数（只有 list 才有 length 属性）"""
        return self.param_type in ("tensors", "scalars", "attrs")


class OperatorAttributeDomain:
    """
    单个算子的所有参数属性可能取值的全集。

    作为覆盖率计算的"分母"，记录该算子下每个参数的每个属性
    理论上可能取到的所有值范围。
    """

    def __init__(self, operator_name: str):
        self.operator_name = operator_name
        # key: 参数名, value: ParamAttributeDomain
        self.params: Dict[str, ParamAttributeDomain] = {}

    def get_param(self, name: str) -> Optional[ParamAttributeDomain]:
        return self.params.get(name)


def _get_range_profile_by_dtype(dtype: str) -> List[str]:
    """
    根据 dtype 选择对应的数值范围离散化模型。

    核心逻辑：
      1. 将 ACL dtype（如 FLOAT16, INT8）映射为 ATK dtype（如 fp16, int8）
      2. 判断 ATK dtype 属于浮点/整数/布尔中的哪一类
      3. 返回对应的离散化模型列表

    对应关系（来自 ParamModelConfig）：
      - 浮点类型 -> FLOAT_TENSOR_DATA_PROFILE（9 种）
      - 整数类型 -> INT_TENSOR_DATA_PROFILE（6 种）
      - 布尔类型 -> [True, False]
    """
    # ACL dtype -> ATK dtype 映射表
    acl_to_atk = {
        "INT4": "int", "INT8": "int8", "INT16": "int16", "INT32": "int32",
        "UINT8": "uint8", "UINT16": "uint16", "UINT32": "uint32", "UINT64": "uint64",
        "INT64": "int64", "BFLOAT16": "bf16", "FLOAT16": "fp16", "FLOAT32": "fp32",
        "FLOAT64": "fp64", "float32": "fp32", "float16": "fp16", "float64": "fp64",
        "FLOAT": "fp32", "DOUBLE": "double", "BOOL": "bool", "bool": "bool",
        "float": "fp32", "int64_t": "int",
    }
    atk_dtype = acl_to_atk.get(dtype, dtype)

    # 数据类型分类判断
    float_types = {"fp16", "fp32", "fp64", "bfp16", "bf16", "fp", "double"}
    int_types = {"int", "int16", "int8", "int32", "int64", "uint8", "uint16", "uint32", "uint64"}
    bool_types = {"bool"}

    if atk_dtype in float_types:
        return list(FLOAT_TENSOR_DATA_PROFILE)
    elif atk_dtype in int_types:
        return list(INT_TENSOR_DATA_PROFILE)
    elif atk_dtype in bool_types:
        return [True, False]
    return []


def _resolve_dim_list(raw_dims: Any) -> List[int]:
    """
    解析算子约束 JSON 中的 dimensions 字段为维度取值列表。

    dimensions 字段的格式可能是：
      - [1, 2, 3, 4]        -> 直接枚举
      - [[1, 4], [2, 3]]    -> 区间 [min, max] 展开
      - None                 -> 使用默认范围 1~8
    """
    if raw_dims is None:
        # 未定义时使用默认范围 1~8
        return list(range(DEFAULT_DIM_COUNT_MIN, DEFAULT_DIM_COUNT_MAX + 1))

    if isinstance(raw_dims, list):
        result = []
        for d in raw_dims:
            if isinstance(d, list) and len(d) >= 2:
                # [min, max] 区间展开为连续整数列表
                result.extend(range(d[0], d[1] + 1))
            elif isinstance(d, (int, float)):
                result.append(int(d))
        return sorted(set(result)) if result else [DEFAULT_DIM_COUNT_MIN]

    return [DEFAULT_DIM_COUNT_MIN]


def _resolve_list(raw: Any) -> List:
    """
    将算子约束 JSON 中可能以多种格式存储的列表字段统一解析为 Python list。

    处理三种情况：
      - None -> []
      - 已经是 list -> 原样返回
      - {"value": [...]} 包装结构 -> 解包
    """
    if raw is None:
        return []
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict) and "value" in raw:
        val = raw["value"]
        return val if isinstance(val, list) else [val]
    return [raw]


# ACL 类型到 ATK 类型的映射（来自 DataMatchMap.ACL_TYPE_TRANSFER_ATK_MAP）
ATK_TYPE_MAP = {
    "aclTensor": "tensor", "aclScalar": "scalar", "aclIntArray": "attrs",
    "aclFloatArray": "attrs", "aclBoolArray": "attrs", "aclTensorList": "tensors",
    "aclScalarList": "scalars",
}


def build_domain_from_constraint(constraint_json_path: str) -> OperatorAttributeDomain:
    """
    从算子约束 JSON 文件构建完整的属性域定义。

    处理流程：
      1. 读取 JSON 文件
      2. 遍历每个 input 参数
      3. 提取 type/dtype/format/dimensions/array_length 字段
      4. 根据 dtype 确定 range_value_profile 的离散化模型
      5. 组装为 ParamAttributeDomain 并注册到 OperatorAttributeDomain

    Args:
        constraint_json_path: 算子约束 JSON 文件路径

    Returns:
        OperatorAttributeDomain: 包含所有参数属性可能取值的域定义
    """
    logger.info(f"Building attribute domain from constraint file: {constraint_json_path}")
    with open(constraint_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    operator_name = data.get(
        "operator_name",
        os.path.splitext(os.path.basename(constraint_json_path))[0]
    )
    domain = OperatorAttributeDomain(operator_name)

    inputs = data.get("inputs", {})
    logger.info(f"Operator: {operator_name}, total input params in constraint: {len(inputs)}")

    for param_name, raw_attr in inputs.items():
        # --- 步骤 1: 提取参数类型并映射为 ATK 类型 ---
        raw_type = _extract_value(raw_attr.get("type"))
        param_type = ATK_TYPE_MAP.get(raw_type, "attr")

        # --- 步骤 2: 为当前参数创建域定义 ---
        p = ParamAttributeDomain(param_name, param_type)

        # --- 步骤 3: 提取各个属性的允许取值 ---
        # dtype: 可能取列表（如 ["FLOAT16", "BFLOAT16", "FLOAT32"]）
        p.dtype_values = _resolve_list(_extract_value(raw_attr.get("dtype")))
        logger.debug(f"  [{param_name}] dtype domain: {p.dtype_values}")

        # format: 可能取列表（如 ["ND", "NCHW"]）
        p.format_values = _resolve_list(_extract_value(raw_attr.get("format")))
        logger.debug(f"  [{param_name}] format domain: {p.format_values}")

        # dim_count: 可能取列表或区间（如 [1, 2, 3, 4] 或 [[1, 4]]）
        p.dim_count_values = _resolve_dim_list(_extract_value(raw_attr.get("dimensions")))
        logger.debug(f"  [{param_name}] dim_count domain: {p.dim_count_values}")

        # length: 仅 list 类型参数有
        p.length_values = _resolve_list(_extract_value(raw_attr.get("array_length")))
        if p.length_values:
            logger.debug(f"  [{param_name}] length domain: {p.length_values}")

        # --- 步骤 4: 根据 dtype 确定 range_value 的离散化模型 ---
        # 取第一个 dtype 作为代表（所有 dtype 同属一类：浮点/整数/布尔）
        if p.dtype_values:
            p.range_value_profile_values = _get_range_profile_by_dtype(p.dtype_values[0])
            logger.debug(f"  [{param_name}] range_value_profile domain: {p.range_value_profile_values} "
                         f"(derived from dtype: {p.dtype_values[0]})")

        # --- 步骤 5: 注册参数 ---
        domain.params[param_name] = p

    logger.info(f"Domain built: operator={operator_name}, params={len(domain.params)}")
    return domain
