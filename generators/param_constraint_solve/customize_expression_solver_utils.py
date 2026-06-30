# -*- coding: UTF-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""
版权信息：华为技术有限公司，版本所有(C) 2026-2026
修改记录：2026/3/30 11:16
功能：需要定制化实现的约束处理逻辑，包含shape_equality和type_equality
"""
import copy
import random
from collections import defaultdict
from typing import List, Dict, Optional

import torch

from generators.atk_common_utils.case_config import CaseConfig
from generators.common_utils.common_dispatcher import CommonDispatcher
from generators.common_utils.logger_util import LazyLogger
from generators.data_definition.common_models import DispatcherTargetType
from generators.data_definition.param_models_def import ParameterPropertyData, CaseInputRangeModel
from generators.operator_param_models.case_generate import CaseGenerate
from generators.common_model_definition import InterParamConstraint, OperatorRule, InterConstraintsRuleType

logger = LazyLogger()


class CustomizeConstraintPatch(CommonDispatcher):
    """
    需要定制化实现的约束处理逻辑，包含shape_equality和type_equality
    """

    def __init__(self, case: CaseConfig, case_generate_instance: CaseGenerate,
                 inter_param_constraints: List[InterParamConstraint], operator_rule_data: OperatorRule,
                 param_combinations: Dict[str, ParameterPropertyData] = None,
                 is_generate_real_data: bool = False):
        self.inter_param_constraints = inter_param_constraints
        # 形如{'x1': {'DType': 'BFLOAT16', 'DataProfile': 'NaN', 'DimCount': '2', 'DimProperty': 'Has_Large_Size',
        # 'DataProfile': 'SubNormal', 'Memory': 'Contiguous'}, 'epsilon': {'Value': '1e-5'},
        # 'additionalOutput': {'Mode': 'True'}}
        self.param_combinations = param_combinations
        # 是否生成真实数据，若为FALSE只保留生成数据模型的名称，用于后续实际执行的时候调用生成真实数据
        self.is_generate_real_data = is_generate_real_data
        # 数据生成实例，用于在不满足约束条件时，重新生成参数的数据
        self.case_generate_instance = case_generate_instance
        self.operator_name = case.name
        self.case = case
        self.case_input_map = {case_input.name: case_input for case_input in case.inputs}
        self.broadcast_master_params = defaultdict(List[InterParamConstraint])

    def get_param_range_model(self, param_name):
        """
        获取参数的range生成model，如果self.param_combinations为None,则从case.inputs中每个input.range_values中获取
        :return: 生成range数据的model的名字
        """
        if self.param_combinations:
            param_range_model = self.param_combinations.get(param_name).range_value_profile
        else:
            input_data = self.case_input_map.get(param_name)
            range_model_data = input_data.range_values
            if isinstance(range_model_data, dict):
                logger.warning("Input.range_values invalid, should be dict, but is: %s, use default range model",
                               type(range_model_data))
                global_role_definitions = CaseGenerate.get_global_role_definitions()
                param_role_model = global_role_definitions.get(CaseInputRangeModel.param_role)
                param_range_model = random.choice(list(param_role_model.keys()))
            else:
                param_range_model = CaseInputRangeModel(**range_model_data).param_range_model_name
        return param_range_model

    @CommonDispatcher.register(InterConstraintsRuleType.SHAPE_EQUALITY.value,
                               target_type=DispatcherTargetType.METHOD.value)
    def shape_equality(self, constraint_expr: InterParamConstraint) -> tuple[Optional[bool], List]:
        """
        实现shape_equality约束
        :param constraint_expr:约束关系
        :return: case数据直接在源数据上修改，即self.case，True/False表示执行过程是否成功
        """
        logger.debug("Start correct param attribute by constraint : shape_equality")
        relation_params = constraint_expr.relation_params
        relation_params = [param for param in relation_params if param in self.case_input_map]
        param_expr = constraint_expr.expr
        if len(relation_params) < 2 or "==" not in param_expr:
            logger.warning(f"Shape equality param constraint invalid, operator name: {self.operator_name}, "
                           f"constraint content: {constraint_expr.__dict__}, relation params : {relation_params}")
            return True, relation_params
        master_param_name = relation_params[0]
        master_param_input = self.case_input_map.get(master_param_name)
        if not master_param_input:
            logger.error("Shape equality, master param not existed in case.inputs, operator name: %s, param name:%s",
                         self.operator_name, master_param_name)
            return False, relation_params
        for slave_param_name in relation_params[1:]:
            slave_param_input = self.case_input_map.get(slave_param_name)
            if not slave_param_input:
                logger.error("Shape equality, slave param not existed in case.inputs, operator name: %s, param name:%s",
                             self.operator_name, slave_param_name)
                continue
            slave_original_shape = copy.deepcopy(slave_param_input.shape)
            slave_param_input.shape = copy.deepcopy(master_param_input.shape)
            param_range_model = self.get_param_range_model(slave_param_name)
            if self.is_generate_real_data:
                slave_param_input_range = (
                    self.case_generate_instance.generate_param_range(param_name=slave_param_name,
                                                                     param_range_model=param_range_model,
                                                                     data_size=slave_param_input.shape,
                                                                     data_dtype=slave_param_input.dtype,
                                                                     is_generate_real_data=self.is_generate_real_data))
                slave_param_input.range_values = slave_param_input_range
            logger.debug(
                "Shape equality end, param attribute correct, operator name: %s, param name: %s, attribute: shape, "
                "original value: %s, correct value: %s", self.operator_name, slave_param_name,
                slave_original_shape, slave_param_input.shape)
        return True, relation_params

    @CommonDispatcher.register(InterConstraintsRuleType.SHAPE_CHOICE.value,
                               target_type=DispatcherTargetType.METHOD.value)
    def shape_choice(self, constraint_expr: InterParamConstraint) -> tuple[Optional[bool], List]:
        """
        实现shape_choice约束
        :param constraint_expr: 约束关系对象
        :return: case数据直接在源数据上修改，即self.case，True/False表示执行过程是否成功
        """
        logger.debug("Start correct param attribute by constraint : shape_choice")
        relation_params = constraint_expr.relation_params
        relation_params = [param for param in relation_params if param in self.case_input_map]
        if len(relation_params) < 2:
            logger.warning(f"Shape choice param constraint invalid, operator name: {self.operator_name}, "
                           f"constraint content: {constraint_expr.__dict__}, relation params : {relation_params}")
            return True, relation_params
        target_param_name = relation_params[0]
        target_data = self.case_input_map.get(target_param_name)
        target_data_original_shape = copy.deepcopy(target_data.shape)
        if not target_data:
            logger.error(
                "Shape choice, param not existed in case.inputs, operator name: %s, param name:%s",
                self.operator_name, target_param_name)
            return False, relation_params
        src_param_data = [self.case_input_map.get(param_name) for param_name in relation_params[1:] if
                          self.case_input_map.get(param_name)]
        choose_param_data = random.choice(src_param_data)
        target_data.shape = choose_param_data.shape
        param_range_model = self.get_param_range_model(target_param_name)
        if self.is_generate_real_data:
            target_data_input_range = (
                self.case_generate_instance.generate_param_range(param_name=target_param_name,
                                                                 param_range_model=param_range_model,
                                                                 data_size=target_data.shape,
                                                                 data_dtype=target_data.dtype,
                                                                 is_generate_real_data=self.is_generate_real_data))
            target_data.range_values = target_data_input_range
        logger.debug(
            "Shape choice end, param attribute correct, operator name: %s, param name: %s, attribute: shape, "
            "original value: %s, correct value: %s", self.operator_name, target_param_name,
            target_data_original_shape, target_data.shape)
        return True, relation_params

    @CommonDispatcher.register(InterConstraintsRuleType.SHAPE_BROADCAST.value,
                               target_type=DispatcherTargetType.METHOD.value)
    def shape_broadcast(self, constraint_expr: InterParamConstraint) -> tuple[Optional[bool], List]:
        """
        实现broadcast约束关系，以x1和x2为例
        场景1：len(x1.shape) == len(x2.shape)
        (1) 相同轴的维度相等
        (2) 若相同轴的维度不相等(假设为第i轴)，则其中一个tensor该轴的shape值需为1，即x1.shape[i] == 1 or x2.shape[i] == 1
        场景2：len(x1.shape) != len(x2.shape)
        (1) len(x1.shape) > len(x2.shape): 从有往左遍历shape的每个轴，即i = len(x2.shape) - 1 : 0,
                                           需满足x1.shape[i] == x2.shape[i]，或x1.shape[i] == 1 or x2.shape[i] == 1
        (2) 对于len(x1.shape) < len(x2.shape) 同理
        (3) 实际执行时，当有多个参数需要满足broadcast关系时，为了避免前后矛盾，统一以第一个参数的shape作为基准，
        即若len(x2.shape) > len(x1.shape)，将x2.shape裁剪至和x1.shape的dim维度一致，且若修改的话，不能修改第一个参数的shape值
        场景3：shape确定之后，使用torch.broadcast_shapes进行校验，如果触发broadcast的异常，则将shape设置为相同值
        :param constraint_expr: 约束关系对象, 默认以params中的第一个param作为base进行broadcast shape 校验和生成
        :return: case数据直接在源数据上修改，即self.case，True/False表示执行过程是否成功
        """
        logger.debug("Start correct param attribute by constraint : shape_broadcast")
        relation_params = constraint_expr.relation_params
        relation_params = [param for param in relation_params if param in self.case_input_map]
        if len(relation_params) < 2:
            logger.warning(f"Shape broadcast param constraint invalid, operator name: {self.operator_name}, "
                           f"constraint content: {constraint_expr.__dict__}, relation params : {relation_params}")
            return True, relation_params
        master_param_name, slave_params = self.choose_broadcast_master(relation_params)
        if not master_param_name:
            logger.error(
                f"Shape broadcast, choose master param failed, operator name: {self.operator_name}, param name: {master_param_name}",
                self.operator_name, master_param_name)
            return False, relation_params
        master_param_input = self.case_input_map.get(master_param_name)
        if not master_param_input:
            logger.error(
                "Shape broadcast, param not existed in case.inputs, operator name: %s, param name:%s",
                self.operator_name, master_param_name)
            return False, relation_params
        master_shape_dim = len(master_param_input.shape)
        slave_shape_dict = {}
        for slave_param in slave_params:
            slave_param_input = self.case_input_map.get(slave_param)
            if not slave_param_input:
                logger.error("Shape broadcast, param not existed in case.inputs, operator name: %s, param name:%s",
                             self.operator_name, slave_param)
                continue
            slave_shape_dim = len(slave_param_input.shape)
            if slave_shape_dim > master_shape_dim:
                slave_param_input.shape = slave_param_input.shape[:master_shape_dim]
            slave_correct_shape = CustomizeConstraintPatch.correct_broadcast_shape(master_param_input.shape,
                                                                                   slave_param_input.shape)
            slave_shape_dict[slave_param] = slave_correct_shape
        all_broadcast_shapes = list(slave_shape_dict.values())
        all_broadcast_shapes.append(master_param_input.shape)
        is_broadcast_all = CustomizeConstraintPatch.is_shapes_match_broadcast(all_broadcast_shapes)
        if is_broadcast_all:
            correct_slave_dict = slave_shape_dict
        else:
            correct_slave_dict = {param_name: master_param_input.shape for param_name in slave_shape_dict.keys()}
        for param_name, correct_shape in correct_slave_dict.items():
            slave_param_input = self.case_input_map.get(param_name)
            slave_shape_original = copy.deepcopy(slave_param_input.shape)
            slave_param_input.shape = correct_shape
            param_range_model = self.get_param_range_model(param_name)
            if self.is_generate_real_data:
                slave_param_input_range = (
                    self.case_generate_instance.generate_param_range(param_name=param_name,
                                                                     param_range_model=param_range_model,
                                                                     data_size=slave_param_input.shape,
                                                                     data_dtype=slave_param_input.dtype,
                                                                     is_generate_real_data=self.is_generate_real_data))
                slave_param_input.range_values = slave_param_input_range
            logger.debug(
                "Shape equality end, param attribute correct, operator name: %s, param name: %s, attribute: shape, "
                "original value: %s, correct value: %s", self.operator_name, param_name,
                slave_shape_original, slave_param_input.shape)
        return True, relation_params

    def choose_broadcast_master(self, relation_params: List[str]) -> tuple[str | None, List]:
        """
        对于某个关系涉及到的所有参数，判断其中的参数是否已作为其他relation的master，如果是，则也将其作为master，否则选择第一个参数作为master
        :param relation_params: 约束关系涉及到的所有参数
        :return: str, 参数名称
        """
        if not relation_params:
            return None, []
        master_param = None
        slave_params = []
        for param in relation_params:
            if param in self.broadcast_master_params:
                master_param = param
            else:
                slave_params.append(param)
        if master_param is None:
            master_param = relation_params[0]
            slave_params.remove(master_param)
        return master_param, slave_params

    @staticmethod
    def correct_broadcast_shape(master_shape, slave_shape) -> List:
        """
        校验两个shape是否满足broadcast，若不满足，则修正，修正逻辑见shape_broadcast方法注释
        :param master_shape: base shape，作为基准shape，不能修改该shape中的值
        :param slave_shape: 修改slave shape中的值，与master_broadcast满足broadcast关系
        :return: 修正之后的slave shape
        """
        slave_shape_dim = len(slave_shape)
        for axis in range(slave_shape_dim - 1, -1, -1):
            if slave_shape[axis] == 1:
                continue
            if slave_shape[axis] != master_shape[axis]:
                if random.random() < 0.5:
                    slave_shape[axis] = 1
                else:
                    slave_shape[axis] = master_shape[axis]
        return slave_shape

    @staticmethod
    def is_shapes_match_broadcast(shape_list: List[List]) -> bool:
        """
        判断list中的多个shape是否满足broadcast关系，满足返回True，不满足返回False
        :param shape_list: 需要校验的shape
        :return: True/False
        """
        try:
            torch.broadcast_shapes(*shape_list)
            return True
        except RuntimeError:
            return False

    @CommonDispatcher.register(InterConstraintsRuleType.TYPE_EQUALITY.value,
                               target_type=DispatcherTargetType.METHOD.value)
    def type_dependency(self, constraint_expr: InterParamConstraint) -> tuple[Optional[bool], List]:
        """
        实现type_dependency约束关系
        :param constraint_expr: 约束关系对象，以第一个参数的dtype为基准
        :return: 修正之后满足约束关系的case
        """
        logger.debug("Start correct param attribute by constraint : type_equality")
        relation_params = constraint_expr.relation_params
        relation_params = [param for param in relation_params if param in self.case_input_map]
        if len(relation_params) < 2:
            logger.warning(
                f"Type equality, param constraint invalid, operator name: {self.operator_name}, "
                f"constraint content: {constraint_expr.__dict__}, relation param : {relation_params}")
            return True, relation_params
        master_param_name = relation_params[0]
        master_param_input = self.case_input_map.get(master_param_name)
        if not master_param_input:
            logger.error("Type equality, param not existed in case.inputs, operator name: %s, param name:%s",
                         self.operator_name, master_param_name)
            return False, relation_params
        for slave_param_name in relation_params[1:]:
            if slave_param_name not in self.case_input_map:
                logger.error("Type equality, param not existed in case.inputs, operator name: "
                             "%s, param name:%s", self.operator_name, slave_param_name)
                continue
            slave_param_input = self.case_input_map.get(slave_param_name)
            if not slave_param_input:
                logger.error("Type equality, param not existed in case.inputs, operator name: %s, param name:%s",
                             self.operator_name, slave_param_name)
                continue
            slave_original_dtype = slave_param_input.dtype
            slave_param_input.dtype = master_param_input.dtype
            logger.debug(
                "Type equality end, param attribute correct, operator name: %s, param name: %s, attribute: dtype, "
                "original value: %s, correct value: %s", self.operator_name, slave_param_name,
                slave_original_dtype, slave_param_input.dtype)
        return True, relation_params

    @CommonDispatcher.register(InterConstraintsRuleType.VALUE_DEPENDENCY.value,
                               target_type=DispatcherTargetType.METHOD.value)
    def value_dependency(self, constraint_expr: InterParamConstraint) -> Optional[bool]:
        """
        实现value_dependency的约束
        :param constraint_expr: 参数间约束数据对象
        :return: case数据直接在源数据上修改，即self.case，True/False表示执行过程是否成功
        """
        logger.debug("Start correct param attribute by constraint : value_dependency")
        relation_params = constraint_expr.relation_params
        relation_params = [param for param in relation_params if param in self.case_input_map]
        master_param_name = relation_params[0]
        slave_params = relation_params[1:]
        master_param_input = self.case_input_map.get(master_param_name)
        if not master_param_input:
            logger.error("Value dependency, param not existed in case.inputs, operator name: %s, param name:%s",
                         self.operator_name, master_param_name)
            return False
        for slave_param_name in slave_params:
            slave_param_input = self.case_input_map.get(slave_param_name)
            if not slave_param_input:
                logger.error(
                    "Value dependency, param not existed in case.inputs, operator name: %s, param name:%s",
                    self.operator_name, slave_param_name)
                continue
            slave_original_value = copy.deepcopy(slave_param_input.value)
            slave_param_input.value = master_param_input.value
            param_range_model = self.get_param_range_model(slave_param_name)
            if self.is_generate_real_data:
                slave_param_input_range = (
                    self.case_generate_instance.generate_param_range(param_name=slave_param_name,
                                                                     param_range_model=param_range_model,
                                                                     data_size=slave_param_input.shape,
                                                                     data_dtype=slave_param_input.dtype,
                                                                     is_generate_real_data=self.is_generate_real_data))
                slave_param_input.range_values = slave_param_input_range
            logger.debug(
                "Value dependency end, param attribute correct, operator name: %s, param name: %s, attribute: value, "
                "original value: %s, correct value: %s", self.operator_name, slave_param_name,
                slave_original_value, slave_param_input.value)
        return True

    @CommonDispatcher.register(InterConstraintsRuleType.FORMAT_EQUALITY.value,
                               target_type=DispatcherTargetType.METHOD.value)
    def format_equality(self, constraint_expr: InterParamConstraint) -> tuple[Optional[bool], List]:
        logger.debug("Start correct param attribute by constraint : format_equality")
        relation_params = constraint_expr.relation_params
        relation_params = [param for param in relation_params if param in self.case_input_map]
        if len(relation_params) < 2:
            logger.warning(
                f"Format equality, param constraint invalid, operator name: {self.operator_name}, "
                f"constraint content: {constraint_expr.__dict__}, relation param : {relation_params}")
            return True, relation_params
        master_param_name = relation_params[0]
        master_param_input = self.case_input_map.get(master_param_name)
        if not master_param_input:
            logger.error("Format equality, param not existed in case.inputs, operator name: %s, param name:%s",
                         self.operator_name, master_param_name)
            return False, relation_params
        for slave_param_name in relation_params[1:]:
            if slave_param_name not in self.case_input_map:
                logger.error("Format equality, param not existed in case.inputs, operator name: "
                             "%s, param name:%s", self.operator_name, slave_param_name)
                continue
            slave_param_input = self.case_input_map.get(slave_param_name)
            if not slave_param_input:
                logger.error("Format equality, param not existed in case.inputs, operator name: %s, param name:%s",
                             self.operator_name, slave_param_name)
                continue
            slave_original_format = slave_param_input.format
            slave_param_input.format = master_param_input.format
            logger.debug(
                "Format equality end, param attribute correct, operator name: %s, param name: %s, attribute: format, "
                "original value: %s, correct value: %s", self.operator_name, slave_param_name,
                slave_original_format, slave_param_input.format)
        return True, relation_params
