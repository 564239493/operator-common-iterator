# -*- coding: UTF-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
"""
版权信息：华为技术有限公司，版本所有(C) 2025-2025
修改记录：2025/12/25 15:19
功能：根据pict组合数据，生成用例数据，输入为pict结果，即operator_name_combinations.tsv文件
"""
import json
import os
import random
from typing import Dict, List

import numpy

from generators.atk_common_utils.case_config import CaseConfig, InputCaseConfig
from generators.common_utils.common_dispatcher import CommonDispatcher
from generators.common_utils.logger_util import LazyLogger
from generators.data_definition.constants import ParamModelConfig, GlobalConfig
from generators.data_definition.param_models_def import BaseRuleModel, \
    ParamShapeRoleRules, ParamRangeRoleRules, DispatcherTargetType, ParameterPropertyData
from generators.operator_param_models.param_dtype_models import ParamDtypeModel
from generators.operator_param_models.param_range_models import ParamRangeValueModelStatic
from generators.operator_param_models.param_shape_models import ParamShapeModel

logger = LazyLogger()


class CaseGenerate:
    def __init__(self, operator_name, params_role: Dict[str, str], global_role_definitions: Dict = None):
        self.operator_name = operator_name
        # 参数的角色，确定每个参数的角色，示例：
        # {
        #   "x1": "role_data_generic",
        #   "x2": "role_data_generic",
        # }
        self.params_role = params_role
        self.default_param_role = ParamModelConfig.DEFAULT_PARAM_ROLE
        self.default_rule_model = CaseGenerate.parse_role_rules(ParamModelConfig.DEFAULT_PARAM_RANGE_RULE)
        # 用于根据数据模型字段选择对应的处理模型，global_role_definitions为预定义的数据模型全集，
        # 从配置文件global_role_definitions.json中加载
        if global_role_definitions is None:
            self.global_role_definitions = CaseGenerate.get_global_role_definitions()
        else:
            self.global_role_definitions = global_role_definitions

    @staticmethod
    def parse_role_rules(role_rules: List[Dict]):
        """
        解析角色模型规则数据，示例：
        :param role_rules: 角色模型的规则列表
        :return: List[BaseRuleModel]
        """
        dispatcher = CommonDispatcher()
        rule_model_list = []
        for rule in role_rules:
            rule_type = rule.get(GlobalConfig.ROLE_RULE_TYPE_KEY)
            if rule_type is not None:
                try:
                    rule_model = dispatcher.dispatch(rule_type, target_type=DispatcherTargetType.CLASS.value,
                                                     init_dict=rule)
                    rule_model_list.append(rule_model)
                except ValueError as e:
                    logger.error("Param role instance failed, err msg: %s", str(e))
        return rule_model_list

    @staticmethod
    def get_global_role_definitions(global_role_definition_path=None):
        """
        加载参数的角色模型定义, normal，loguniform等要转换为已定义的数据结构
        :param global_role_definition_path: 定义json文件路径，若为None，则使用默认配置：configs/global_role_definitions.json
        :return: Dict，角色定义数据
        """
        if global_role_definition_path is None:
            global_role_definition_path = os.path.join(GlobalConfig.CONFIG_FILES_BASE_PATH,
                                                       ParamModelConfig.GLOBAL_ROLE_DEFINITIONS_PATH)
        with open(global_role_definition_path, "r", encoding="utf-8") as f:
            global_role_definitions = json.load(f)
        roles_name_list = global_role_definitions.get("__ROLE_ONTOLOGY_DOC__", [])
        global_role_definitions_parse = {}
        for role_name in roles_name_list:
            role_models_data = global_role_definitions.get(role_name, {})
            role_models_parse = {}
            for role_model_name, role_rules in role_models_data.items():
                role_rule_models = CaseGenerate.parse_role_rules(role_rules)
                role_models_parse[role_model_name] = role_rule_models
            global_role_definitions_parse[role_name] = role_models_parse
        return global_role_definitions_parse

    def default_return(self, task_name, error_msg, default_return_value, param_name):
        """
        定义参数错误时默认的返回内容，以及打印的日志
        :param task_name: 任务名称，用于标志哪个过程发生错误
        :param error_msg: 错误信息
        :param default_return_value: 默认返回值
        :param param_name: 参数名称
        :return: default_return_value
        """
        logger.error("Task: %s, operator name: %s, param name: %s, err msg : %s, use default value %s",
                     task_name, self.operator_name, param_name, error_msg, default_return_value)
        return default_return_value

    def get_param_rule(self, param_name: str, role_rules: List[BaseRuleModel]) -> BaseRuleModel:
        """
        根据参数组合中设置的range生成模型名称，如SubNormal，选择参数对应的rule
        :param operator_name: 算子名称
        :param param_name: 参数名称
        :param role_rules: 该参数的所有rules
        :return: rule模型，如Normal，或LogUniform，选择其中一个
        """
        if role_rules is None:
            role_rules = self.default_return("Get param rules", "Role rules is None",
                                             self.default_rule_model,
                                             param_name)
        rule_weights = [rule.weight for rule in role_rules]
        if numpy.sum(numpy.abs(rule_weights)) <= 0:
            rule_weights = [1] * len(rule_weights)

        choose_rule = random.choices(role_rules, weights=rule_weights, k=1)[0]
        return choose_rule

    def generate_case(self, param_combination: Dict[str, ParameterPropertyData],
                      case_config_template: CaseConfig = None):
        """
        根据输入的参数取值，生成case data，格式为CaseConfig
        :param param_combination: 一组参数组合数据，参数组合后生成的每一组参数用例取值，用dict表示，第一层key为param_name,
        第二层key为参数属性名称，如Dtype， DataProfile取值属性等,value为参数属性的取值，如：BFLOAT16, SubNormal
        :param case_config_template: case_config模板，如果未提供，初始化生成，
        此时case_config中除inputs和name属性外，其他属性未进行赋值
        :return: CaseConfig
        """
        if case_config_template is None:
            case_config = CaseConfig()
        else:
            case_config = case_config_template
        case_config.name = self.operator_name
        input_case_list = []
        for param_name, param_attributes in param_combination.items():
            param_type = param_attributes.param_type
            param_dtype = self.generate_param_dtype(param_name, param_attributes.dtype)
            param_length = param_attributes.length
            # 这里不过滤非算子参数，避免约束表达式中包含非算子参数求解失败
            # if not param_attributes.is_operator_param:
            #     continue
            # 用于确定可选参数是否设置取值，但是部分可选参数会与必选参数在同一个约束表达式中出现，为了避免表达式求解失败，
            # 这里不对可选参数做概率性选择，全部加入参数集合
            # optional_param_probability = random.random()
            # if param_attributes.is_optional and optional_param_probability > GlobalConfig.OPTIONAL_PARAM_PROBABILITY:
            #     continue
            if param_type in ParamModelConfig.TENSOR_ATK_TYPE:
                param_shape = self.generate_param_shape(param_name, param_attributes.shape_property.dim_count,
                                                        param_attributes.shape_property.dim_value_profile)
                param_range = self.generate_param_range(param_name, param_attributes.range_value_profile, param_dtype,
                                                        param_shape)
                input_case_config = InputCaseConfig(name=param_name, type=param_type, dtype=param_dtype,
                                                    shape=param_shape, range_values=param_range,
                                                    format=param_attributes.format)
            else:
                param_range = self.generate_param_range(param_name, param_attributes.range_value_profile,
                                                        param_dtype)
                input_case_config = InputCaseConfig(name=param_name, type=param_type, dtype=param_dtype,
                                                    range_values=param_range, length=param_length)
            input_case_list.append(input_case_config)
        case_config.inputs = input_case_list
        return case_config

    def generate_param_dtype(self, param_name, param_dtype_str: str):
        """
        生成参数的的type
        :param param_name: 参数名称
        :param param_dtype_str: 参数数据类型，即Dict[str, str]:
        {"Dtype":FLOAT32, "DataProfile": "NAN", "DimCount":2, "DimProperty":"Has_Large_Size", "Memory":"NonCountiguous"}
        中的Dtype
        :return: 参数的dtype
        """
        task_name = "generate param dtype"
        if param_dtype_str is None:
            return self.default_return(task_name, "dtype is None", ParamModelConfig.DEFAULT_PARAM_DTYPE, param_name)
        param_dtype_instance = ParamDtypeModel(operator_name=self.operator_name, param_name=param_name)
        param_dtype = param_dtype_instance.generate_param_dtype(param_dtype_str)
        return param_dtype

    def generate_param_shape(self, param_name: str, param_shape_dim_count: int, param_shape_model: str):
        """
        生成参数的shape属性
        :param param_name: 参数名称
        :param param_shape_dim_count: 参数shape的维度参数，即Dict[str, str]:
        {"Dtype":FLOAT32, "DataProfile": "NAN", "DimCount":2, "DimProperty":"Has_Large_Size", "Memory":"NonCountiguous"}
        中的DimCount
        :param param_shape_model: 参数shape模型，即即Dict[str, str]:
        {"Dtype":FLOAT32, "DataProfile": "NAN", "DimCount":2, "DimProperty":"Has_Large_Size", "Memory":"NonCountiguous"}
        中的DimProperty
        :return: shape数据：List
        """
        task_name = "generate param shape"
        logger.debug("Start %s, operator name: %s, param name: %s, param shape model : %s", task_name,
                     self.operator_name, param_name, param_shape_model)
        if param_shape_dim_count is None:
            param_shape_dim_count = self.default_return("generate param shape", "param shape dim count is invalid",
                                                        ParamModelConfig.DEFAULT_TENSOR_SHAPE_DIM, param_name)
        if param_shape_model is None:
            param_shape_model = self.default_return(task_name, "param shape model is None",
                                                    ParamShapeRoleRules.TYPICAL.value, param_name)
        param_shape_instance = ParamShapeModel(self.operator_name)
        try:
            param_shape_data = param_shape_instance.dispatch(param_shape_model, param_shape_dim_count, param_name,
                                                             target_type=DispatcherTargetType.METHOD.value)
        except ValueError as e:
            logger.error("Task name: %s, operator name: %s, param name: %s, err msg : %s", task_name,
                         self.operator_name, param_name, str(e))
            param_shape_model = ParamShapeRoleRules.DEFAULT.value
            param_shape_data = param_shape_instance.dispatch(param_shape_model, param_shape_dim_count, param_name,
                                                             target_type=DispatcherTargetType.METHOD.value)
        logger.debug("End %s, operator name: %s, param name: %s, param shape: %s", task_name, self.operator_name,
                     param_name, param_shape_data)
        return param_shape_data

    def generate_param_range(self, param_name: str, param_range_model: str, data_dtype: str, data_size=None,
                             is_generate_real_data=False):
        """
        生成参数的具体取值
        :param param_name: 参数名称
        :param param_range_model: 参数属性数据中range模型名字，即Dict[str, str]:
        {"Dtype":FLOAT32, "DataProfile": "NAN", "DimCount":2, "DimProperty":"Has_Large_Size", "Memory":"NonCountiguous"}
        中的DataProfile
        :param data_size: 数据的shape大小，用tuple表示,若为None，则返回数据模型的信息，不返回实际数据
        :param data_dtype: 数据类型
        :param is_generate_real_data: 是否生成真实数据，若为FALSE只保留生成数据模型的名称，用于后续实际执行的时候调用生成真实数据
        :return: input_case, 在输入参数中修改range属性
        """
        task_name = "generate param range"
        logger.debug("Start %s, operator name: %s, param name: %s, is generate real data: %s",
                     task_name, self.operator_name, param_name, is_generate_real_data)
        if (param_range_model not in ParamModelConfig.FLOAT_TENSOR_DATA_PROFILE and
                param_range_model not in ParamModelConfig.INT_TENSOR_DATA_PROFILE):
            logger.debug(
                f"Task: {task_name}, operator name : {self.operator_name}, param name : {param_name}, "
                f"is_generate real data : {is_generate_real_data}, "
                f"range model not in float data models and int data model, use ori model : {param_range_model}")
            return param_range_model
        if param_range_model is None:
            return self.default_return(task_name, "param combination range model is invalid", [],
                                       param_name)
        param_role_name = self.params_role.get(param_name)
        if param_role_name is None:
            param_role_name = self.default_return(task_name, "role is invalid", self.default_param_role,
                                                  param_name)
        param_role_model = self.global_role_definitions.get(param_role_name)
        if param_role_model is None:
            param_role_model = self.default_return(task_name, "role model is None",
                                                   self.global_role_definitions.get(self.default_param_role),
                                                   param_name)
        param_role_rules = param_role_model.get(param_range_model)
        param_rule = self.get_param_rule(param_name, param_role_rules)
        # if is_generate_real_data:
        rule_name = param_rule.type
        logger.debug(
            f"Get param range model rule name, operator name : {self.operator_name}, param name : {param_name}, rule name : {rule_name}")
        rule_instance = ParamRangeValueModelStatic(operator_name=self.operator_name, param_name=param_name)
        try:
            param_range_data = rule_instance.dispatch(rule_name, data_size, data_dtype, param_rule,
                                                      target_type=DispatcherTargetType.METHOD.value)
        except ValueError as e:
            logger.error("Task name: %s, operator name: %s, param name: %s, err msg : %s", task_name,
                         self.operator_name, param_name, str(e))
            rule_name = ParamRangeRoleRules.DEFAULT.value
            param_range_data = rule_instance.dispatch(rule_name, data_size, data_dtype, param_rule,
                                                      target_type=DispatcherTargetType.METHOD.value)
        # else:
        # 保存具体的模型名称以及模型参数，用于后续基于此信息生成具体的tensor填充值，暂时修改为ATK模式
        # param_range_instance = CaseInputRangeModel(param_role=param_role_name,
        #                                            param_range_model_name=param_range_model,
        #                                            param_range_rule=param_rule)
        # param_range_data = param_range_instance.model_dump()

        logger.debug(
            "End generate param range, operator name: %s, param name: %s, is generate real data: %s, "
            "param range data: %s", self.operator_name, param_name, is_generate_real_data, param_range_data)
        return param_range_data
