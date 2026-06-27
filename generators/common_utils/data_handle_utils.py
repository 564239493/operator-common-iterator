# -*- coding: UTF-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
"""
版权信息：华为技术有限公司，版本所有(C) 2025-2025
修改记录：2025/12/26 10:28
功能：数据处理相关功能
"""
import json
import math
import os

from typing import List, Dict

from pydantic import ValidationError

from agent.generators.atk_common_utils.case_config import CaseConfig
from agent.generators.common_utils.logger_util import LazyLogger
from agent.generators.data_definition.constants import DataMatchMap, GlobalConfig
from agent.generators.data_definition.param_models_def import RunPlatform
from agent.generators.common_model_definition import OperatorRule, ParamAttributes, ValueWithSrcText

logger = LazyLogger()


class DataHandleUtil:

    @staticmethod
    def save_cases_to_json(api_name, generate_case_list: List[CaseConfig], json_save_path):
        """
        保存生成的case数据为JSON文件
        :param api_name: api名称，用来确认json文件名称
        :param generate_case_list: 将生成的case数据保存为json
        :param json_save_path: json保存路径
        :return: None
        """
        logger.info(f"Start save case json, api name : '{api_name}'")
        case_config_json_list = []
        for case_config in generate_case_list:
            case_config_json = case_config.model_dump()
            case_config_json_list.append(case_config_json)
        if not os.path.exists(json_save_path):
            os.makedirs(json_save_path)
        json_save_file = os.path.join(json_save_path, api_name + ".json")
        with open(json_save_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(case_config_json_list, ensure_ascii=False, indent=4))
        logger.info(f"End save case json, api name : '{api_name}'")

    @staticmethod
    def handle_operator_rule_data(operator_rule_file_path: str) -> OperatorRule | None:
        """
        从约束数据中获取所有约束信息，即inter_parameter_constraints中的数据
        :param operator_rule_file_path: 约束数据rule.json路径
        :return: 所有的参数约束关系数据对象: OperatorRule
        """
        if not os.path.exists(operator_rule_file_path):
            logger.error(f"Operator constraint data file is not find, file path : {operator_rule_file_path}")
            return None
        with open(operator_rule_file_path, "r", encoding="utf-8") as f:
            operator_rule_data = json.load(f)
        try:
            operator_rule_instance = OperatorRule(**operator_rule_data)
        except ValidationError as e:
            logger.error(
                f"Operator constraint data type validation, operator rule file : {operator_rule_file_path}, err msg : {str(e)}")
            operator_rule_instance = None
        return operator_rule_instance

    @staticmethod
    def select_effective_parameters(operator_constraint_data: OperatorRule,
                                    target_platform: str = RunPlatform.DEFAULT_PLATFORM.value) -> OperatorRule | None:
        """
        筛选有效的参数进行处理，1. 只处理{OpName}GetWorkspaceSize方法中，role为input的参数; 2. 根据执行平台选择对应的约束信息;
        :param operator_constraint_data: 原始结构化数据
        :param target_platform: 算子运行平台设备类型
        :return: 筛选有效数据之后约束数据
        """
        logger.debug("Start select effective parameters")
        if not operator_constraint_data:
            logger.error("Operator constraint data is None")
            return None
        if target_platform not in operator_constraint_data.deterministic_computing:
            logger.warning(f"Platform '{target_platform}' not in deterministic_computing")
        operator_constraint_data.deterministic_computing = operator_constraint_data.deterministic_computing.get(
            RunPlatform.DEFAULT_PLATFORM.value, {})
        operator_constraint_data.inputs = DataHandleUtil.select_constraint_by_target(operator_constraint_data.inputs,
                                                                                     target_platform)
        if operator_constraint_data.inputs is None:
            return None
        operator_constraint_data.outputs = DataHandleUtil.select_constraint_by_target(operator_constraint_data.outputs,
                                                                                      target_platform)
        if operator_constraint_data.outputs is None:
            return None
        if (target_platform not in operator_constraint_data.constraints_in_parameters and
                GlobalConfig.COMMON_PLATFORM not in operator_constraint_data.constraints_in_parameters):
            logger.warning(
                f"Platform '{target_platform}' and '{GlobalConfig.COMMON_PLATFORM}' not in constraints_in_parameters")
        target_platform_constraint_data = operator_constraint_data.constraints_in_parameters.get(
            target_platform, [])
        common_constraint_data = operator_constraint_data.constraints_in_parameters.get(GlobalConfig.COMMON_PLATFORM,
                                                                                        [])
        target_platform_constraint_data.extend(common_constraint_data)
        operator_constraint_data.constraints_in_parameters = target_platform_constraint_data
        if (target_platform not in operator_constraint_data.dtype_support_description and
                GlobalConfig.COMMON_PLATFORM not in operator_constraint_data.dtype_support_description):
            logger.warning(
                f"Platform '{target_platform}' and '{GlobalConfig.COMMON_PLATFORM}' not in dtype_support_description")
        target_platform_dtype_support_data = operator_constraint_data.dtype_support_description.get(
            target_platform, [])
        common_dtype_support_data = operator_constraint_data.dtype_support_description.get(GlobalConfig.COMMON_PLATFORM,
                                                                                           [])
        target_platform_dtype_support_data.extend(common_dtype_support_data)
        operator_constraint_data.dtype_support_description = target_platform_constraint_data
        if (target_platform not in operator_constraint_data.format_support_description and
                GlobalConfig.COMMON_PLATFORM not in operator_constraint_data.format_support_description):
            logger.warning(
                f"Platform '{target_platform}' and '{GlobalConfig.COMMON_PLATFORM}' not in format_support_description")
        target_platform_format_support_data = operator_constraint_data.format_support_description.get(
            target_platform, [])
        common_format_support_data = operator_constraint_data.format_support_description.get(
            GlobalConfig.COMMON_PLATFORM, [])
        target_platform_format_support_data.extend(common_format_support_data)
        operator_constraint_data.format_support_description = target_platform_format_support_data
        return operator_constraint_data

    @staticmethod
    def select_constraint_by_target(param_dict: Dict[str, Dict[str, ParamAttributes]], target_platform: str) -> Dict[
                                                                                                                    str, ParamAttributes] | None:
        """
        根据platform的类型筛选
        :param param_dict: 约束数据字典
        :param target_platform: 算子运行平台设备类型
        :return: 符合运行平台的数据
        """
        effective_param_dict = {}
        if param_dict is None:
            return None
        for param_name, param_attribute in param_dict.items():
            if target_platform not in param_attribute and GlobalConfig.COMMON_PLATFORM not in param_attribute:
                logger.error(
                    f"Platform '{target_platform}' not in param : {param_name} attribute and {GlobalConfig.COMMON_PLATFORM} not in param")
                return None
            effective_param_data = param_attribute.get(target_platform, None)
            effective_param_data = effective_param_data if effective_param_data is not None else param_attribute.get(
                GlobalConfig.COMMON_PLATFORM, None)
            effective_param_dict[param_name] = effective_param_data

        return effective_param_dict

    @staticmethod
    def abnormal_float_transfer(resolved_range):
        if isinstance(resolved_range, float):
            if math.isnan(resolved_range) or math.isinf(resolved_range):
                return str(resolved_range)
            else:
                return resolved_range
        elif isinstance(resolved_range, (list, tuple)):
            return [DataHandleUtil.abnormal_float_transfer(v) for v in resolved_range]
        return resolved_range

    @staticmethod
    def get_relevant_attribute_value(param_name, param_attribute: str | ValueWithSrcText, param_attribute_name: str):
        """
        判断参数的属性是不是不涉及，即param_attribute == “N/A”，涉及返回value，不涉及返回None
        """
        if isinstance(param_attribute, ValueWithSrcText):
            if isinstance(param_attribute.value, list):
                if len(param_attribute.value) == 0:
                    return None, None
            return param_attribute.value, param_attribute.type
        else:
            logger.info(
                f"Param : '{param_name}', attribute : '{param_attribute_name}', is not relevant : '{param_attribute}'")
            return None, None

    @staticmethod
    def get_range_data_boundary(dtype: str, range_data) -> List | None:
        """
        解析allowed_value list格式的数据：[0,1] -> [0,1]; [0,null] -> None; [0, inf] -> [0, 对应数据类型的最大值];
        [-inf, 0] -> []
        :param dtype: 数据类型，用于当边界包含inf, -inf时，确认该数据类型的最大值
        :param range_data: 原始的allowed_range_value
        """
        if len(range_data) < 2:
            return None
        value_boundary = []
        dtype_value = DataMatchMap.ACL_DTYPE_TRANSFER_TENSOR_MAP.get(dtype)
        # 处理下限
        low_boundary = range_data[0]
        if low_boundary == "null":
            return None
        if isinstance(low_boundary, (int, float)):
            value_boundary.append(low_boundary)
        elif low_boundary == "-inf":
            value_boundary.append(DataMatchMap.DTYPE_SPECS.get(dtype_value)[0])
        else:
            return None
        # 处理上限
        high_boundary = range_data[1]
        if high_boundary == "null":
            return None
        if isinstance(high_boundary, (int, float)):
            value_boundary.append(high_boundary)
        elif high_boundary == "inf":
            value_boundary.append(DataMatchMap.DTYPE_SPECS.get(dtype_value)[1])
        else:
            return None
        return value_boundary
