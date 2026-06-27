# -*- coding: UTF-8 -*-
"""
功能：定义数据模型结构
版权信息：华为技术有限公司，版本所有(C) 2025-2025
修改记录：2025/12/23 14:35
"""
from dataclasses import dataclass
from enum import Enum
from typing import Union, List, Dict, Optional, Any

from pydantic import BaseModel

from agent.generators.common_utils.common_dispatcher import CommonDispatcher
from agent.generators.data_definition.common_models import DispatcherTargetType
from agent.generators.data_definition.constants import ParamModelConfig


class ParamRangeRoleRules(Enum):
    STATIC = "Static"
    NORMAL = "Normal"
    UNIFORM = "Uniform"
    LOGUNIFORM = "LogUniform"
    INTUNIFORM = "IntUniform"
    INTUNIFORMODD = "IntUniformOdd"
    CHOICE = "Choice"
    DEFAULT = "Normal"


class ParamShapeRoleRules(Enum):
    TYPICAL = "Typical"
    HAS_ODD_SIZE = "Has_Odd_Size"
    HAS_SIZE_1 = "Has_Size_1"
    HAS_LARGE_SIZE = "Has_Large_Size"
    DEFAULT = "Typical"


class ParamAttrName(Enum):
    DTYPE = "dtype"
    RANGE = "data_profile"
    SHAPE_DIM_COUNT = "dim_count"
    SHAPE_DIM_PROPERTY = "bim_property"
    MEMORY = "memory"
    PARAM_TYPE = "param_type"
    PLATFORM = "platform"
    VALUE = "value"
    IS_OPTIONAL = "is_optional"
    PARAM_FORMAT = "format"

class ParamRangeValueType(Enum):
    RANGE = "range"
    ENUM = "enum"


class RunPlatform(Enum):
    ATLAS_A3_TRAIN_AND_INFER_SERIES = "Atlas A3 训练系列产品/Atlas A3 推理系列产品"
    ATLAS_A2_TRAIN_AND_INFER_SERIES = "Atlas A2 训练系列产品/Atlas A2 推理系列产品"
    ATLAS_200I_OR_500_A2_INFER_PRODUCT = "Atlas 200I/500 A2 推理产品"
    ATLAS_INFER_SERIES = "Atlas 推理系列产品"
    ALL_SERIES = "All"
    ATLAS_350_ACCELERATOR_CARD = "Accelerator Card"
    DEFAULT_PLATFORM = "Atlas A3 训练系列产品/Atlas A3 推理系列产品"


class ShapePropertyConstraintStructure(Enum):
    DIMS = "dims"
    AXIS_VALUE = "axis_value"


class BaseRuleModel(BaseModel):
    type: str
    weight: float = 1.0


@CommonDispatcher.register(ParamRangeRoleRules.STATIC.value, target_type=DispatcherTargetType.CLASS.value)
class StaticModel(BaseRuleModel):
    value: Union[str, float]
    comment: str = None
    post_process: str = None
    clip_min: float = None
    clip_max: float = None


@CommonDispatcher.register(ParamRangeRoleRules.NORMAL.value, target_type=DispatcherTargetType.CLASS.value)
class NormalModel(BaseRuleModel):
    mean: float
    std: float
    post_process: str = None
    clip_min: float = None
    clip_max: float = None
    comment: str = None


@CommonDispatcher.register(ParamRangeRoleRules.UNIFORM.value, target_type=DispatcherTargetType.CLASS.value)
class UniformModel(BaseRuleModel):
    min: float
    max: float
    post_process: str = None
    clip_min: float = None
    clip_max: float = None
    comment: str = None


@CommonDispatcher.register(ParamRangeRoleRules.LOGUNIFORM.value, target_type=DispatcherTargetType.CLASS.value)
class LogUniformModel(UniformModel):
    pass


@CommonDispatcher.register(ParamRangeRoleRules.INTUNIFORM.value, target_type=DispatcherTargetType.CLASS.value)
class IntUniformModel(UniformModel):
    pass


@CommonDispatcher.register(ParamRangeRoleRules.INTUNIFORMODD.value, target_type=DispatcherTargetType.CLASS.value)
class IntUniformOddModel(UniformModel):
    pass


@CommonDispatcher.register(ParamRangeRoleRules.CHOICE.value, target_type=DispatcherTargetType.CLASS.value)
class ChoiceModel(BaseRuleModel):
    options: List
    post_process: str = None
    clip_min: float = None
    clip_max: float = None
    comment: str = None


@dataclass
class ShapeRoleRule:
    # 参数关键字匹配列表
    keywords: List[str]
    # 参数取值池名称，通过此名称在参数取值池中获取对应的取值列表
    pool: str
    # 规则描述
    desc: str = None


class SingleShapeStrategy(BaseModel):
    # strategy名称
    strategy_name: str = None
    # 默认shape取值池
    default_pool: str = None
    # 基础的shape取值池
    base_pool: str = None
    # shape取值上限
    fixed_large_dim: int = ParamModelConfig.TENSOR_SHAPE_MAX_VALUE
    # shape取值规则，通过参数名称和关键字匹配，获取参数shape取值规则
    role_rules: List[ShapeRoleRule] = None
    # 特定参数覆盖
    param_overrides: Dict[str, str] = None


class CaseInputRangeModel(BaseModel):
    param_role: Optional[str] = ParamModelConfig.DEFAULT_PARAM_ROLE
    param_range_model_name: Optional[str | int | float]
    param_range_rule: Optional[BaseRuleModel]


class ParameterShapeProperty(BaseModel):
    dim_count: int
    dim_value_profile: str


class ParameterPropertyData(BaseModel):
    param_name: str
    param_type: str
    shape_property: ParameterShapeProperty = None
    dtype: str = ParamModelConfig.DEFAULT_PARAM_DTYPE
    format: str | None = None
    range_value_profile: Any
    memory_continuity: bool = False
    length: int | None = None
    is_optional: bool = True
    is_operator_param : bool = True


class OperatorParameterCombination(BaseModel):
    operator_name: str
    parameter_property: List[ParameterPropertyData] = []
