# -*- coding: UTF-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
"""
版权信息：华为技术有限公司，版本所有(C) 2025-2025
修改记录：2025/12/27 10:11
功能：读取参数组合tsv文件，批量生成算子用例
"""
import json
import os.path
import time
from typing import Dict, List

from agent.generators.atk_common_utils.case_config import CaseConfig
from agent.generators.common_utils.data_handle_utils import DataHandleUtil
from agent.generators.common_utils.logger_util import LazyLogger
from agent.generators.data_definition.constants import GlobalConfig
from agent.generators.data_definition.param_models_def import OperatorParameterCombination, RunPlatform, ParameterPropertyData
# [PAIRWISE] 替换旧随机生成器为 Pairwise 策略生成器
from agent.generators.operator_param_combine.param_combination_generate import ParamCombinationGenerator
# from operator_param_combine.pairwise_combination import PairwiseParamCombinationGenerator
from agent.generators.operator_param_models.case_generate import CaseGenerate
from agent.generators.param_constraint_solve.param_constraint_utils import ParamConstraintUtils
from agent.generators.common_model_definition import OperatorRule, InterParamConstraint

logger = LazyLogger()


class OperatorCaseGenerator:

    def get_params_roles(self, operator_name):
        """
        获取每个算子的参数角色，为json文件， 默认存放于代码同级目录下的configs文件夹下
        :param operator_name: 算子名称
        :return: params_role: Dict
        """
        param_roles_file_path = os.path.join(GlobalConfig.PARAM_ROLE_RESULT_SAVE_PATH, operator_name + ".json")
        if not os.path.exists(param_roles_file_path):
            logger.error(f"Operator: {operator_name}, params semantic role file not find")
            return {}
        with open(param_roles_file_path, "r", encoding="utf-8") as f:
            params_role = json.load(f)
            return params_role

    @staticmethod
    def get_output_param(operator_constraint_data: OperatorRule):
        """
        获取算子的output参数，多个参数用,相连接，如"output,xOut,yOut"
        :param operator_constraint_data: 参数约束数据
        :return: 所有的输出参数，如"output,xOut,yOut"
        """
        output_params = list(operator_constraint_data.outputs.keys())
        return ",".join(output_params)

    def handle_single_operator(self, operator_constraint_data: OperatorRule,
                               param_combination_list: List[OperatorParameterCombination],
                               target_platform=RunPlatform.ATLAS_A3_TRAIN_AND_INFER_SERIES.value, case_num: int = 1):
        """
        读取参数组合文件，xxx.tsv，解析数据，并生成对用的用例
        :param operator_constraint_data: 算子约束结构化数据
        :param param_combination_list: 算子参数组合用例数据
        :param target_platform: 执行机环境
        :param case_num: 生成用例的个数，默认为1
        :return: List[CaseConfig]
        """
        if operator_constraint_data is None:
            logger.error("Operator constraint data is None")
        logger.info(
            f"Start generate cases, operator name : {operator_constraint_data.operator_name}, "
            f"case num : {case_num}, target platform : {target_platform}")
        if param_combination_list is None:
            logger.error(f"Target platform: {target_platform}, no param combinations match")
            return []
        params_role = self.get_params_roles(operator_constraint_data.operator_name)
        case_generate_instance = CaseGenerate(operator_name=operator_constraint_data.operator_name,
                                              params_role=params_role)
        final_case_list = []
        case_index = 0
        solve_time = 0
        while case_index < case_num:
            logger.info(f"###### Start generate case data, case id : '{case_index}/{case_num}' ######")
            t_start = time.time()
            param_combination = param_combination_list[case_index % len(param_combination_list)]
            param_combination_dict = {each.param_name: each for each in param_combination.parameter_property}
            case_config = case_generate_instance.generate_case(param_combination_dict)
            correct_status, correct_case = self.correct_case(case_config, operator_constraint_data,
                                                             param_combination_dict)
            t_end = time.time()
            solve_time += 1
            logger.debug(
                f"Operator solve constraint running, solve time : {solve_time}, solve status : {correct_status}")
            if correct_status:
                correct_case.outputs = OperatorCaseGenerator.get_output_param(operator_constraint_data)
                final_case_list.append(correct_case)
                correct_case.id = case_index
                case_index += 1
                solve_time = 0
                logger.debug(
                    f"###### Operator constraint solve running, correct case index : '{case_index}/{case_num}' "
                    f"| use time : {t_end - t_start}s ######")
            if solve_time >= GlobalConfig.Z3_SOLVE_TIME_LIMIT:
                logger.error(
                    f"Operator : {operator_constraint_data.operator_name} has run Z3 solve {solve_time} times, "
                    f"and all result is failed or unsat")
                break
        logger.info(
            f"End generate cases, operator name : {operator_constraint_data.operator_name}, "
            f"case num : {case_num}, target platform : {target_platform}, actual case num : {len(final_case_list)}")
        return final_case_list

    def correct_case(self, case: CaseConfig, operator_rule_instance: OperatorRule,
                     param_combinations: Dict[str, ParameterPropertyData] = None):
        """
        根据算子参数的约束条件修正参数取值
        self, case: CaseConfig, case_generate_instance: CaseGenerate,
                     inter_param_constraints: List[InterParamConstraint], param_combinations: Dict,
                     is_generate_real_data: bool = False
        :param case: 算子用例对象
        :param param_combinations: 此用例生成时使用的参数组合信息，即pict输出的组合数据
        :param operator_rule_instance: 算子结构化数据，由模型辅助生成的结构化数据,，已转换为数据的实例
        :return: 修正后的算子用例
        """
        operator_name = case.name
        params_role = self.get_params_roles(operator_name)
        inter_param_constraints = operator_rule_instance.constraints_in_parameters
        case_generate_instance = CaseGenerate(operator_name=operator_name, params_role=params_role)
        param_constraint_patch = ParamConstraintUtils(case=case, case_generate_instance=case_generate_instance,
                                                      inter_param_constraints=inter_param_constraints,
                                                      param_combinations=param_combinations,
                                                      operator_rule_data=operator_rule_instance)
        correct_status = param_constraint_patch.correct_operator_param()
        for input_data in case.inputs:
            input_data_range_value = DataHandleUtil.abnormal_float_transfer(input_data.range_values)
            input_data.range_values = input_data_range_value
        return correct_status, case

    @staticmethod
    def construct_param_value(constraint_data: InterParamConstraint, case_config: CaseConfig):
        """
        根据参数名称，获取参数的取值，并构建参数和参数取值的联合数据：
        :param constraint_data: 约束数据，包含约束条件表达式字符串、表达式参与参数
        :param case_config: 用例取值
        :return: 结果分两部分，
                 第一部分，参数取值：
                'parameters': {
                    'x1': {'shape': [10, 20, 30], 'dtype': 'float32'},
                    'x2': {'shape': [10, 20, 30], 'dtype': 'float32'},
                },
                第二部分：表达式
                'expression': 'x1.shape[-1] == x2.shape[-1]'
        """
        params_dict = {}
        for param_name in constraint_data.params:
            param_value = case_config.get_input_data_config(name=param_name)
            if param_value is None:
                logger.error(f"Param value is None, param name: {param_name}")
                continue
            params_dict[param_name] = param_value.__dict__
        return params_dict, constraint_data.expr

    def handle_operators_batch(self, param_combination_file_directory: str, operator_constraint_data_directory,
                               case_save_path: str = None, case_num: int = 1,
                               target_platform=RunPlatform.ATLAS_A3_TRAIN_AND_INFER_SERIES.value) -> None:
        """
        从参数组合存储目录下读取所有待生成的组合文件，并处理生成用例，将用例以json形式保存,默认文件名称为算子名称
        :param param_combination_file_directory: 参数组合文件所在目录
        :param operator_constraint_data_directory: 算子结构化数据所在目录,用于获取算子参数的type
        :param case_save_path: 用例保存路径
        :param case_num: 单个算子用例个数
        :param target_platform: 算子用例执行平台设备类型
        :return: None
        """
        if not os.path.exists(param_combination_file_directory):
            raise FileNotFoundError("Param combinations directory not existed")
        if not os.path.exists(operator_constraint_data_directory):
            raise FileNotFoundError("Operator rule path not existed")
        file_list = os.listdir(param_combination_file_directory)
        tsv_file_list = [each.endswith(".tsv") for each in file_list]
        tsv_file_num = len(tsv_file_list)
        logger.info(f"Start handle param combination, operator file num : {tsv_file_num}")
        if case_save_path is None:
            case_save_path = GlobalConfig.CASE_RESULT_SAVE_PATH
        if not os.path.exists(case_save_path):
            os.makedirs(case_save_path)
        data_handle_util = DataHandleUtil()
        for index, file in enumerate(tsv_file_list):
            logger.info(f"Start handle operator data, file index : {index}/{tsv_file_num}, file name : {file}")
            operator_name, _ = os.path.splitext(file)
            operator_rule_data_path = os.path.join(operator_constraint_data_directory, operator_name + ".json")
            operator_rule_data = data_handle_util.handle_operator_rule_data(operator_rule_data_path)
            effective_operator_constraint_data = DataHandleUtil.select_effective_parameters(operator_rule_data)
            operator_name, _ = os.path.splitext(file)
            param_combination_generator = ParamCombinationGenerator(operator_rule_data=operator_rule_data,
                                                                            case_num=case_num)
            # param_combination_generator = PairwiseParamCombinationGenerator(operator_rule_data=operator_rule_data,
            #                                                                 case_num=case_num)
            param_combination_list = param_combination_generator.get_param_combination_input()
            case_list = self.handle_single_operator(operator_constraint_data=effective_operator_constraint_data,
                                                    target_platform=target_platform,
                                                    case_num=case_num,
                                                    param_combination_list=param_combination_list)
            data_handle_util.save_cases_to_json(api_name=operator_name, generate_case_list=case_list,
                                                json_save_path=case_save_path)
            logger.info(f"End handle operator data, file index : {index}/{tsv_file_num}, file name : {file}")
        logger.info(f"End handle param combination, operator file num : {len(tsv_file_list)}")
