# -*- coding: UTF-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
"""
版权信息：华为技术有限公司，版本所有(C) 2025-2025
修改记录：2026/3/12 16:01
功能：基于结构化数据以及参数语义角色确定参数具体取值或使用模型名称
"""
import copy
import random
from typing import List

from agent.generators.common_utils.data_handle_utils import DataHandleUtil
from agent.generators.common_utils.logger_util import LazyLogger
from agent.generators.data_definition.constants import DataMatchMap, ParamModelConfig
from agent.generators.data_definition.param_models_def import OperatorParameterCombination, ParameterPropertyData, \
    ParameterShapeProperty, ParamRangeValueType
from agent.generators.param_constraint_solve.z3_expression_solver_utils import ExpressionPreprocessor
from agent.generators.common_model_definition import OperatorRule

logger = LazyLogger()


class ParamCombinationGenerator:
    def __init__(self, operator_rule_data: OperatorRule, case_num: int = 1):
        self.operator_rule_data = operator_rule_data
        self.case_num = case_num
        self.choose_dtype_map_combination = None

    def get_param_combination_input(self) -> List[OperatorParameterCombination] | None:
        """
        根据提取的算子参数约束数据(JSON)，提取参数具体信息，设置参数属性的具体设置值或属性对应模型的名称
        :param operator_constrain_data: 算子约束数据,已筛选过有效参数({opName}GetWorkspace方法下role为input的参数)
        :param case_num: 用例数量
        class ParameterPropertyData(BaseModel):
        param_name: str
        param_type: str
        shape_property: ParameterShapeProperty = None
        dtype: str
        format : str = None
        range_value_profile: str | int | float | bool
        memory_continuity: bool = False
        :return: 算子参数属性组合
        """
        if self.operator_rule_data is None:
            logger.error(f"Get param combination failed, input operator constraint data is None")
            return None
        logger.info(
            f"Start generate parameter combinations, operator name : '{self.operator_rule_data.operator_name}'")
        param_combination_list = []
        all_param_info = copy.deepcopy(self.operator_rule_data.inputs)
        all_param_info.update(self.operator_rule_data.outputs)
        for _ in range(self.case_num):
            operator_parameter_combination = OperatorParameterCombination(
                operator_name=self.operator_rule_data.operator_name)
            for input_name, input_attribute in all_param_info.items():
                logger.debug(f"Start generate param : '{input_name}' attribute")
                if input_attribute is None:
                    logger.error(
                        f"Operator: '{self.operator_rule_data.operator_name}', param: '{input_name}', attribute is None")
                    continue
                param_type_ori, _ = DataHandleUtil.get_relevant_attribute_value(input_name,
                                                                                input_attribute.type, "type")
                param_type = DataMatchMap.ACL_TYPE_TRANSFER_ATK_MAP.get(param_type_ori,
                                                                        ParamModelConfig.DEFAULT_ATK_TYPE)
                param_format, _ = DataHandleUtil.get_relevant_attribute_value(input_name,
                                                                              input_attribute.format, "format")
                param_format_value = random.choice(param_format) if isinstance(param_format, list) else param_format
                param_dtype = self.generate_dtype_property(input_name)
                param_length = self.generate_length_property(input_name, param_type)
                param_range_value_profile = self.generate_range_value_property(input_name, param_dtype)
                param_is_optional, _ = DataHandleUtil.get_relevant_attribute_value(input_name,
                                                                                   input_attribute.is_optional,
                                                                                   "is_optional")
                is_operator_param, _ = DataHandleUtil.get_relevant_attribute_value(input_name,
                                                                                   input_attribute.is_operator_param,
                                                                                   "is_operator_param")
                parameter_property_data = ParameterPropertyData(param_name=input_name, dtype=param_dtype,
                                                                param_type=param_type, format=param_format_value,
                                                                range_value_profile=param_range_value_profile,
                                                                length=param_length,
                                                                is_operator_param=is_operator_param,
                                                                is_optional=param_is_optional)
                if param_type in ParamModelConfig.TENSOR_ATK_TYPE:
                    param_shape_property = self.generate_shape_property(input_name)
                    parameter_property_data.shape_property = param_shape_property
                    range_value_profile = DataMatchMap.PARAM_VALUE_TO_ROLE_MODEL.get(param_range_value_profile,
                                                                                     param_range_value_profile)
                    parameter_property_data.range_value_profile = range_value_profile
                operator_parameter_combination.parameter_property.append(parameter_property_data)
            param_combination_list.append(operator_parameter_combination)
        logger.info(f"End generate parameter combinations, operator name : '{self.operator_rule_data.operator_name}'")
        return param_combination_list

    def generate_length_property(self, param_name: str, param_type: str) -> int | None:
        """
        生成数组参数的length属性
        :param param_name: 参数名称
        :param param_type: 参数类型，如果参数是数组类型，才有length属性，否则length为None
        :return: 参数的length值
        """
        logger.debug(f"Start generate parameter length, "
                     f"operator name: '{self.operator_rule_data.operator_name}', param name: '{param_name}'")
        param_attribute = self.operator_rule_data.inputs.get(param_name)
        if param_attribute is None:
            param_attribute = self.operator_rule_data.outputs.get(param_name)
        if param_type not in ParamModelConfig.LIST_ATK_TYPE:
            return None
        length_value, _ = DataHandleUtil.get_relevant_attribute_value(param_name, param_attribute.array_length,
                                                                      "array_length")
        if length_value is None:
            logger.warning(
                f"Generate parameter length, param name : '{param_name}', length value is None, "
                f"use default length: '{ParamModelConfig.DEFAULT_LIST_LENGTH}'")
            return ParamModelConfig.DEFAULT_LIST_LENGTH
        choose_length_data = random.choice(length_value)
        if isinstance(choose_length_data, list):
            length_range_data = DataHandleUtil.get_range_data_boundary(ParamModelConfig.INT_DTYPE[0],
                                                                       choose_length_data)
            if length_range_data is None:
                logger.error(f"Generate parameter length, param name : '{param_name}', "
                             f"length value : '{length_value}' solve failed, use default length: "
                             f"'{ParamModelConfig.DEFAULT_LIST_LENGTH}'")
                return ParamModelConfig.DEFAULT_LIST_LENGTH
            length_value = random.randint(length_range_data[0], length_range_data[1])
        elif isinstance(choose_length_data, int):
            length_value = choose_length_data
        else:
            logger.error(
                f"Generate parameter length, param name: '{param_name}', length data is not int or not a list, "
                f"use default length: '{ParamModelConfig.DEFAULT_LIST_LENGTH}'")
            return ParamModelConfig.DEFAULT_LIST_LENGTH
        logger.debug(f"End generate parameter length, "
                     f"operator name : '{self.operator_rule_data.operator_name}', param name : '{param_name}', "
                     f"length value : '{length_value}'")
        return length_value

    def generate_shape_property(self, param_name) -> ParameterShapeProperty | None:
        """
        生成参数的shape描述属性，包含shape的维度以及生成其中取值的模型名称：Has_Large_Size，Has_Size_1，Has_Odd_Size，Typical
        :return: shape属性取值，dim_count, dim_value_profile
        """
        logger.debug(
            f"Start generate parameter shape property, "
            f"operator name : '{self.operator_rule_data.operator_name}', param name : '{param_name}'")
        param_attribute = self.operator_rule_data.inputs.get(param_name)
        if param_attribute is None:
            param_attribute = self.operator_rule_data.outputs.get(param_name)
        dim_value, _ = DataHandleUtil.get_relevant_attribute_value(param_name, param_attribute.dimensions, "dimensions")
        if dim_value is None:
            dim_count = ParamModelConfig.DEFAULT_TENSOR_SHAPE_DIM
        else:
            dim_count = random.choice(dim_value)
        if isinstance(dim_count, list):
            dim_count_value = random.randint(dim_count[0], dim_count[1])
        else:
            dim_count_value = dim_count
        dim_count_value = dim_count_value if dim_count_value is not None else ParamModelConfig.DEFAULT_TENSOR_SHAPE_DIM
        dim_value_profile = random.choice(ParamModelConfig.DIM_VALUE_PROFILE_LIST)
        shape_property = ParameterShapeProperty(dim_count=dim_count_value, dim_value_profile=dim_value_profile)
        logger.debug(
            f"End generate parameter shape property, operator name : '{self.operator_rule_data.operator_name}', "
            f"param name : '{param_name}', shape property : '{shape_property}'")
        return shape_property

    def generate_dtype_property(self, param_name: str) -> str:
        """
        选择参数的数据类型,如果dtype_map不为空，则在dtype_map中选择一组数据类型作为参数的数据类型，
        否则从parameter_constraints的合法值随机选择
        :param param_name: 参数名称
        :return: 数据类型
        """
        logger.debug(
            f"Start generate dtype property,"
            f"operator name : '{self.operator_rule_data.operator_name}',param name : '{param_name}'")
        input_attribute = self.operator_rule_data.inputs.get(param_name)
        if input_attribute is None:
            input_attribute = self.operator_rule_data.outputs.get(param_name)
        dtype_set, _ = DataHandleUtil.get_relevant_attribute_value(param_name, input_attribute.dtype, "dtype")
        if not dtype_set:
            logger.error(
                f"Generate dtype property, param name : '{param_name}', dtype set is empty, use default data dtype")
            return ParamModelConfig.DEFAULT_PARAM_DTYPE_DTYPE_IN_ORIGINAL_DOC
        valid_dtype_set = [each for each in dtype_set if each not in ParamModelConfig.UNSUPPORT_DTYPE]
        param_dtype = random.choice(valid_dtype_set)
        if self.operator_rule_data.dtype_support_description:
            if self.choose_dtype_map_combination is None:
                self.choose_dtype_map_combination = random.choice(self.operator_rule_data.dtype_support_description)
            # 部分参数可能不在dtype_map中，如果不在dtype_map中，需要取inputs.dtype.value中获取
            if param_name in self.choose_dtype_map_combination:
                param_dtype = self.choose_dtype_map_combination.get(param_name)
        logger.debug(
            f"End generate dtype property, "
            f"operator name: '{self.operator_rule_data.operator_name}', param name: '{param_name}', dtype: '{param_dtype}'")
        return param_dtype

    @staticmethod
    def get_default_range_by_dtype(dtype: str):
        """
        如果无法根据allowed_value确定数据range模型，就根据数据类型选择默认模型，如果没有任何一项匹配上，则返回None
        :param dtype: 数据类型
        :return: 返回值
        """
        if DataMatchMap.ACL_DTYPE_TRANSFER_TENSOR_MAP.get(dtype) in ParamModelConfig.FLOAT_DTYPE:
            range_value_profiles = ParamModelConfig.FLOAT_TENSOR_DATA_PROFILE
        elif DataMatchMap.ACL_DTYPE_TRANSFER_TENSOR_MAP.get(dtype) in ParamModelConfig.INT_DTYPE:
            range_value_profiles = ParamModelConfig.INT_TENSOR_DATA_PROFILE
        elif DataMatchMap.ACL_DTYPE_TRANSFER_TENSOR_MAP.get(dtype) in ParamModelConfig.BOOL_DTYPE:
            range_value_profiles = ParamModelConfig.BOOL_DATA_PROFILE
        else:
            logger.warning(
                f"Get default range value profile failed, dtype : '{dtype}' is not in dtype map, range model is None")
            range_value_profiles = [None]
        return range_value_profiles

    def generate_range_value_property(self, param_name: str, dtype: str) -> str | int | float | bool:
        """
        生成参数的取值范围属性,检查parameter_constraint.allowed_values和parameter_constraint.not_allowed_values，
        1. 如果合法取值指定的固定取值，则设置为该值，如allowed_values = [0.01]
        2. 如果合法取值指定的是取值范围，则离散化为：[ min_val ], [ max_val ], [ mid_val ], [ near_min_val ],
        [ near_max_val ], Normal. (Also include NaN if the type is float)
        3. 如果未指定任何信息：则离散化为：(Float): PosNormal, NegNormal, Zero, NaN, PosInf, NegInf, SubNormal
        (Integer): Pos, Neg, Zero, Max, Min
        :param param_name: 参数名称
        :param dtype: 数据类型
        :return: 数据取值模型名称或具体值
        """
        logger.debug(f"Start generate param range_value_property, "
                     f"operator name : '{self.operator_rule_data.operator_name}', param name : '{param_name}'...")
        param_attribute = self.operator_rule_data.inputs.get(param_name)
        if param_attribute is None:
            param_attribute = self.operator_rule_data.outputs.get(param_name)
        default_data_profile = random.choice(ParamCombinationGenerator.get_default_range_by_dtype(dtype))
        allowed_values, value_type = DataHandleUtil.get_relevant_attribute_value(param_name,
                                                                                 param_attribute.allowed_range_value,
                                                                                 "allowed_range_value")
        if allowed_values is None or len(allowed_values) == 0:
            logger.info(
                f"Generate range value property, param name : '{param_name}', allowed range value set is None or empty")
            return default_data_profile

        select_allowed_value = random.choice(allowed_values)
        if value_type == ParamRangeValueType.ENUM.value:
            logger.debug(
                f"param : '{param_name}', range value type : '{value_type}', range value : '{select_allowed_value}'")
            return select_allowed_value
        if isinstance(select_allowed_value, list):
            allowed_value_boundary = DataHandleUtil.get_range_data_boundary(dtype, select_allowed_value)
            if allowed_value_boundary is None:
                logger.error(
                    f"Operator: '{self.operator_rule_data.operator_name}', param: '{param_name}', "
                    f"dtype : '{dtype}', range value: '{select_allowed_value}'. solve failed, "
                    f"use default data profile : {default_data_profile}")
                return default_data_profile
            range_value_profile_list = [allowed_value_boundary[0], allowed_value_boundary[1],
                                        (allowed_value_boundary[0] + allowed_value_boundary[1]) / 2,
                                        allowed_value_boundary[0] + 0.01, allowed_value_boundary[1] - 0.01]
        elif isinstance(select_allowed_value, str) and ExpressionPreprocessor.validate_expression_without_bool(
                str(select_allowed_value)):
            range_value_profile_list = [select_allowed_value]
        elif isinstance(select_allowed_value, (int, float, bool)):
            range_value_profile_list = [select_allowed_value]
        else:
            logger.error(
                f"Can't match allowed values, use default function, operator name : '{self.operator_rule_data.operator_name}', "
                f"param name : '{param_name}', allowed_values : '{allowed_values}'")
            return default_data_profile
        range_value_profile = random.choice(range_value_profile_list)
        logger.debug(
            f"End generate range value property, operator name : '{self.operator_rule_data.operator_name}', "
            f"param name : '{param_name}', value profile : '{range_value_profile}'")
        return range_value_profile
