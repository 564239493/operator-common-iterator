"""
算子规则 JSON 的 Pydantic 数据模型定义（新版）
用于严格校验 JSON 数据结构和类型
"""
from enum import Enum
from typing import Any, Dict, List

from pydantic import BaseModel, Field


class InterConstraintsRuleType(str, Enum):
    """参数见约护士类型枚举"""
    SHAPE_BROADCAST = "shape_broadcast"
    SHAPE_CHOICE = "shape_choice"
    SHAPE_EQUALITY = "shape_equality"
    SHAPE_DEPENDENCY = "shape_dependency"
    SHAPE_VALUE_DEPENDENCY = "shape_value_dependency"
    TYPE_DEPENDENCY = "type_dependency"
    TYPE_EQUALITY = "type_equality"
    VALUE_DEPENDENCY = "value_dependency"
    FORMAT_EQUALITY = "format_equality"
    PRESENCE_DEPENDENCY = "presence_dependency"

# ==================== 通用值模型 ====================

class ValueWithSrcText(BaseModel):
    """带 src_text 来源信息的通用值字段"""
    value: bool | str | List[str] | List[List[int]] | List[Any] | int | float = Field(..., description="字段值")
    src_text: str = Field(default="", description="来源文本")
    type: str | None = Field(default=None, description="range_value的type说明，可取值：enum(表示枚举), range(表示范围)")

    model_config = {"extra": "forbid"}


class ParamAttributes(BaseModel):
    """参数信息模型（按平台区分，通用结构）"""
    description: str = Field(default="", description="参数描述")
    type: ValueWithSrcText | str = Field(..., description="参数类型")
    format: ValueWithSrcText | str = Field(..., description="参数格式")
    is_optional: ValueWithSrcText | str = Field(..., description="是否可选")
    is_support_discontinuous: ValueWithSrcText | str = Field(..., description="是否支持非连续")
    is_operator_param: ValueWithSrcText | str = Field(..., description="是否为算子参数")
    array_length: ValueWithSrcText | str = Field(default="N/A", description="数组长度（可能为字符串或对象）")
    dtype: ValueWithSrcText | str = Field(..., description="支持的数据类型")
    dimensions: ValueWithSrcText | str = Field(..., description="维度信息")
    allowed_range_value: ValueWithSrcText | str = Field(default_factory=lambda : ValueWithSrcText(value=[], src_text=""), description="允许的取值范围")

    model_config = {"extra": "forbid"}


# ==================== 参数间约束 ====================

class InterParamConstraint(BaseModel):
    """参数约束条目"""
    expr_type: str = Field(..., description="约束表达式类型")
    expr: str = Field(..., description="约束表达式")
    relation_params: List[str] = Field(..., description="涉及的参数列表")
    src_text: str = Field(default="", description="来源文本")

    model_config = {"extra": "forbid"}


# ==================== 返回值信息 ====================

class ReturnInfoItem(BaseModel):
    """返回值信息"""
    return_value: str = Field(..., description="返回值标识")
    error_code: int = Field(..., description="错误码")
    description: List[str] = Field(default_factory=list, description="错误描述列表")

    model_config = {"extra": "forbid"}


# ==================== 顶层模型 ====================

class OperatorRule(BaseModel):
    """算子规则顶层模型（通用）"""
    operator_name: str = Field(..., description="算子名称")
    function_explanation: str = Field(..., description="功能说明")
    product_support: List[str] = Field(..., description="支持的产品列表")
    function_signature: str = Field(..., description="函数签名")
    deterministic_computing: Dict[str, ValueWithSrcText] | str | ValueWithSrcText = Field(
        default_factory=dict, description="确定性计算信息（按平台）"
    )
    inputs: Dict[str, Dict[str, ParamAttributes]] | Dict[str, ParamAttributes]= Field(
        default_factory=dict, description="输入参数信息（参数名 -> 平台 -> 参数详情）"
    )
    outputs: Dict[str, Dict[str, ParamAttributes]] | Dict[str, ParamAttributes] = Field(
        default_factory=dict, description="输出参数信息（参数名 -> 平台 -> 参数详情）"
    )
    constraints_in_parameters: Dict[str, List[InterParamConstraint]] | List[InterParamConstraint] = Field(
        default_factory=dict, description="参数内约束（按平台）"
    )
    return_info: List[ReturnInfoItem] = Field(
        default_factory=list, description="返回值信息"
    )
    dtype_support_description: Dict[str, List[Dict[str, str]]] | List[Dict[str, str]] = Field(
        default_factory=dict, description="数据类型支持描述(按平台区分)"
    )
    format_support_description: Dict[str, List[Dict[str, str]]] | List[Dict[str, str]] = Field(
        default_factory=dict, description="数据格式支持描述(按平台区分)"
    )

    model_config = {"extra": "forbid"}
