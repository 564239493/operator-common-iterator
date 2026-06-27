# -*- coding: UTF-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
"""
版权信息：华为技术有限公司，版本所有(C) 2025-2025
修改记录：2026/1/5 19:44
功能：参数约束关系实现
"""
import re
from collections import defaultdict
from typing import List, Dict

import z3
from pydantic import BaseModel

from agent.generators.atk_common_utils.case_config import CaseConfig
from agent.generators.operator_param_models.case_generate import CaseGenerate
from agent.generators.param_constraint_solve.customize_expression_solver_utils import CustomizeConstraintPatch
from agent.generators.param_constraint_solve.z3_expression_solver_utils import Z3ConstraintBuilder, ExpressionPreprocessor
from agent.generators.common_model_definition import InterParamConstraint, InterConstraintsRuleType, OperatorRule
from agent.generators.common_utils.common_dispatcher import CommonDispatcher
from agent.generators.common_utils.data_handle_utils import DataHandleUtil
from agent.generators.common_utils.logger_util import LazyLogger
from agent.generators.data_definition.constants import ParamModelConfig, DataMatchMap
from agent.generators.data_definition.param_models_def import ParameterPropertyData

logger = LazyLogger()


class ParamSetValueFlag(BaseModel):
    dtype: bool = False
    shape: bool = False
    range_value: bool = False
    format: bool = False
    length: bool = False


class ParamConstraintUtils(CommonDispatcher):
    def __init__(self, case: CaseConfig, case_generate_instance: CaseGenerate,
                 inter_param_constraints: List[InterParamConstraint], operator_rule_data: OperatorRule,
                 param_combinations: Dict[str, ParameterPropertyData] = None,
                 is_generate_real_data: bool = False):
        self.inter_param_constraints = inter_param_constraints
        # 形如{'x1': {'DType': 'BFLOAT16', 'DataProfile': 'NaN', 'DimCount': '2', 'DimProperty': 'Has_Large_Size',
        # 'DataProfile': 'SubNormal', 'Memory': 'Contiguous'}, 'epsilon': {'Value': '1e-5'},
        # 'additionalOutput': {'Mode': 'True'}}
        self.param_combinations = param_combinations
        self.operator_rule_data = operator_rule_data
        # 是否生成真实数据，若为FALSE只保留生成数据模型的名称，用于后续实际执行的时候调用生成真实数据
        self.is_generate_real_data = is_generate_real_data
        # 数据生成实例，用于在不满足约束条件时，重新生成参数的数据
        self.case_generate_instance = case_generate_instance
        self.operator_name = case.name
        self.case = case
        self.case_input_map = {case_input.name: case_input for case_input in case.inputs}
        self.broadcast_master_params = defaultdict(List[InterParamConstraint])
        self.has_set_value_param = defaultdict(ParamSetValueFlag)
        self.relation_params = list(operator_rule_data.inputs.keys())
        self.customize_constraint_patch = CustomizeConstraintPatch(case=case,
                                                                   case_generate_instance=case_generate_instance,
                                                                   inter_param_constraints=inter_param_constraints,
                                                                   operator_rule_data=operator_rule_data,
                                                                   param_combinations=param_combinations,
                                                                   is_generate_real_data=is_generate_real_data)
        self.dtype_domain_data, self.format_domain_data = self.get_param_domain_value()

    def get_param_domain_value(self) -> tuple[Dict[str, List], Dict[str, List]]:
        """
        获取所有参数的dtype和format的值域，用于构建求解条件
        :return: 每个参数dtype的值域数据和format值域数据
        """
        dtype_domain_data = {}
        format_domain_data = {}
        for param_name in self.case_input_map.keys():
            dtype_domain = self.generate_dtype_string_domain(param_name)
            format_domain = self.generate_format_string_domain(param_name)
            if dtype_domain:
                dtype_domain_data[param_name] = dtype_domain
            if format_domain:
                format_domain_data[param_name] = format_domain
        return dtype_domain_data, format_domain_data

    def is_param_all_input(self, relation_params: List[str]):
        """
        判断参数的角色是否都为输入
        :param relation_params: 约束相关的参数名称
        """
        for param_name in relation_params:
            if param_name not in self.case_input_map:
                logger.error(f"Can't match this parameter in input params, param name : {param_name}")
                return False
        return True

    def set_has_value_param_status(self, param_name, constraint_type: str):
        """
        将已确定的参数的属性的状态在self.has_set_value_param中设置为True，
        :param param_name: 参数名称
        :param constraint_type: 约束的类型
        :return: None
        """
        param_set_value_flag = self.has_set_value_param.get(param_name, ParamSetValueFlag())
        for type_key, type_list in DataMatchMap.CONSTRAINT_TYPE_MAP.items():
            if constraint_type in type_list:
                param_set_value_flag.__setattr__(type_key, True)
        self.has_set_value_param[param_name] = param_set_value_flag

    def correct_operator_param(self):
        """
        修复case的参数，在源数据上修改
        :return: None
        """
        logger.info(f"Start correct case param, operator name : {self.operator_name}")
        all_constraint_relations = [relation.value for relation in InterConstraintsRuleType]
        match_relations = []
        for relation in all_constraint_relations:
            match_relation = [each for each in self.inter_param_constraints if each.expr_type == relation]
            if match_relation:
                match_relations.extend(match_relation)
        customize_constraints = []
        z3_constraints = []

        for constraint_relation in match_relations:
            relation_type = constraint_relation.expr_type
            if relation_type in ParamModelConfig.STRICT_CONSTRAINT_TYPE:
                customize_constraints.append(constraint_relation)
            else:
                # 这里用于筛选输出参数，如果约束表达式中包含输出参数，此表达式不进行处理，但有些情况下，表达式中除了关于输出参数的表达式，
                # 还可能有只包含输入参数的子表达式，如果不处理可能会导致部分表达式不生效
                # if not self.is_param_all_input(constraint_relation.relation_params):
                #     continue
                logger.debug(f"Relation type : {relation_type}, use z3 solver")
                z3_constraints.append(constraint_relation)
        self.solve_z3_constraints(z3_constraints)
        for customize_constraint in customize_constraints:
            relation_type = customize_constraint.expr_type
            logger.debug(f"Relation type : {relation_type}, use strict constraint logical")
            strict_check_result, relation_params = self.customize_constraint_patch.dispatch(relation_type,
                                                                                            customize_constraint)
            if not strict_check_result:
                logger.debug(
                    f"Relation type : {relation_type}, use strict constraint logical failed, "
                    f"check result : {strict_check_result}")
                return False
            for param_name in relation_params:
                self.set_has_value_param_status(param_name, relation_type)
            logger.debug(f"Relation type : {relation_type}, use strict constraint logical success")
        logger.info(f"End correct case param, operator name : {self.operator_name}")
        return True

    def generate_dtype_string_domain(self, param_name: str) -> List[str]:
        """
        获取数据类型dtype可取值，用列表表示
        :param param_name: 参数名称
        :return: List[dtype]
        """
        param_attribute = self.operator_rule_data.inputs.get(param_name)
        if param_attribute is None:
            param_attribute = self.operator_rule_data.outputs.get(param_name)
        if param_attribute is None:
            logger.error(f"Param : {param_name}, dtype domain is None")
            return []
        dtype_domain, _ = DataHandleUtil.get_relevant_attribute_value(param_name, param_attribute.dtype, "dtype")
        if dtype_domain is None:
            logger.error(f"Param : {param_name}, dtype domain is not relevant")
            return []
        return dtype_domain

    def generate_format_string_domain(self, param_name: str) -> List[str]:
        """
        获取数据格式format的可取值，用列表表示
        :param param_name: 参数名称
        :return : List[format]
        """
        param_attribute = self.operator_rule_data.inputs.get(param_name)
        if param_attribute is None:
            param_attribute = self.operator_rule_data.outputs.get(param_name)
        if param_attribute is None:
            logger.warning(f"Param : {param_name}, format domain is None")
            return []
        format_domain, _ = DataHandleUtil.get_relevant_attribute_value(param_name, param_attribute.format, "format")
        if format_domain is None:
            logger.warning(f"Param : '{param_name}', format domain is not relevant")
            return []
        return format_domain

    @staticmethod
    def adapter_dtype_in_expr(expr: str) -> str:
        """
        适配表达式中的数据类型，如原始表达式中为INT4, 统一替换为int, INT8统一替换为int8,
        需要替换的值位于DataMatchMap.ACL_DTYPE_TRANSFER_TENSOR_MAP
        :param expr: 需要被替换的表达式
        :return: 替换之后的表达式
        """
        if not isinstance(expr, str):
            logger.info(f"Expr type is not string, can't be replace, expr : {expr}")
            return expr
        for key, value in DataMatchMap.ACL_DTYPE_TRANSFER_TENSOR_MAP.items():
            if isinstance(expr, str):
                expr = re.sub(rf"\b{re.escape(key)}\b", value, expr)
        return expr

    def check_expr_conflicts(self, expr_list: List[str], param_attr="") -> bool:
        """
        判断列表中的表达式是否冲突
        :param expr_list: 表达式列表
        :param param_attr: 参数属性名称，日志记录
        :return: True(不冲突)/False(冲突)
        """
        try:
            builder = Z3ConstraintBuilder()
            self.declare_param_in_z3(builder=builder)
            expr_dict = {f"expr : {expr}": expr for index, expr in enumerate(expr_list)}
            builder.add_constraints(expr_dict)
            check_status = builder.solver.check()
            if check_status == z3.sat:
                return True
            logger.debug("Check param value expr, result : conflicts, conflicting expression : ")
            unsat_cores = builder.solver.unsat_core()
            for core in unsat_cores:
                logger.error(f"   {core}   ")
            return False
        except Exception as e:
            logger.error(f"Expr check conflicts, param attr : {param_attr}, err msg : {str(e)}")
            return False

    def choice_no_conflicts_expr(self, param_union_expr: List[str], param_static_expr_list: List[str]) -> List[str]:
        """
        筛选和参数静态表达式(每个参数的静态表达式，parameters_constraint)
        以及关联表达式(多个参数间的约束表达式，inter_parameter_constraints)不冲突的初始值表达式，避免因为参数初始值设置不合理导致约束求解失败
        :param param_union_expr: 关联表达式(多个参数间的约束表达式，inter_parameter_constraints)
        :param param_static_expr_list: 参数静态表达式(每个参数的静态表达式，parameters_constraint)
        :return: 所有不发生冲突的表达式
        """
        no_conflicts_expr_list = param_union_expr
        for static_expr in param_static_expr_list:
            logger.debug(f"Check param value expr : {static_expr}")
            no_conflicts_expr_list.append(static_expr)
            if not self.check_expr_conflicts(no_conflicts_expr_list, param_attr="shape"):
                no_conflicts_expr_list.remove(static_expr)
            else:
                logger.debug("Check param value expr, result : no conflicts")
        logger.debug(f"No conflicts expr list : {no_conflicts_expr_list}")
        return no_conflicts_expr_list

    def build_param_dtype_constraint(self, constraint_exprs: List[str]) -> \
            List[str]:
        """
        查找是否有参数的dtype属性取值已确定，如果有，是否和已有的规则冲突，如不冲突，则添加为求解条件，避免每次求解结构都相同，如冲突，则不添加
        :param constraint_exprs: 约束表达式对象
        :return: dtype的约束表达式列表
        """
        logger.debug("Start build param dtype constraint")
        static_value_exprs = []
        relation_param = list(self.case_input_map.keys())
        for param in relation_param:
            static_value_exprs.append(f"{param}.dtype == '{self.case_input_map.get(param).dtype}'")
        no_conflicts_dtype_expr = self.choice_no_conflicts_expr(param_union_expr=constraint_exprs,
                                                                param_static_expr_list=static_value_exprs)
        logger.debug("End build param dtype constraint")
        return no_conflicts_dtype_expr

    def build_param_range_value_constraint(self, constraint_exprs: List[str]) -> List[str]:
        """
        对于range_value，如果range_value是浮点数或整数，则直接加入求解条件表达式，否则不加入求解表达式
        :param constraint_exprs: 约束条件数据
        :return: range的约束表达式列表
        """
        logger.debug("Start build param range value constraint")
        static_range_value_expr_list = []
        relation_params = list(self.case_input_map.keys())
        for param_name in relation_params:
            param_attr = self.operator_rule_data.inputs.get(param_name)
            if param_attr is None:
                param_attr = self.operator_rule_data.outputs.get(param_name)
            param_range_value_constraint, _ = DataHandleUtil.get_relevant_attribute_value(
                param_name, param_attr.allowed_range_value if param_attr else None, "allowed_range_value")
            if param_range_value_constraint is None:
                continue
            for value_rule in param_range_value_constraint:
                param_range_value_expr_list = []
                if isinstance(value_rule, list) and len(value_rule) >= 2:
                    param_range_value_expr_list.append(
                        f"({param_name}.range_value > {value_rule[0]} and {param_name}.range_value < {value_rule[1]})")
                elif isinstance(value_rule, (int, float, bool)):
                    param_range_value_expr_list.append(f"{param_name}.range_value == {value_rule}")
                elif isinstance(value_rule, str):
                    if value_rule.strip() in ("None", ""):
                        logger.warning(f"Skip invalid range value expression: '{value_rule}'")
                        continue
                    value_rule = ParamConstraintUtils.adapter_dtype_in_expr(value_rule)
                    if ExpressionPreprocessor.validate_expression(value_rule):
                        param_range_value_expr_list.append(value_rule)
                    else:
                        param_range_value_expr_list.append(f"{param_name}.range_value == {value_rule}")
                else:
                    logger.warning(f"Can't match range value expression : '{value_rule}'")
                param_range_value_expr = " or ".join(param_range_value_expr_list)
                static_range_value_expr_list.append(param_range_value_expr)
        no_conflicts_range_value_expr = self.choice_no_conflicts_expr(param_union_expr=constraint_exprs,
                                                                      param_static_expr_list=static_range_value_expr_list)
        logger.debug("End build param range value constraint")
        return no_conflicts_range_value_expr

    def build_param_shape_constraint(self, constraint_exprs: List[str]) -> \
            List[str]:
        """
        对于shape,如果该参数的parameter_constraints中shape约束乜有axis_value的约束表达式，则不添加确定值的约束，避免约束冲突，
        此时将axis_value中的约束表达式添加至列表中，如果没有axis_value的约束表达式，则将shape当前取值添加至表达式列表
        :param constraint_exprs: 约束表达式对象
        :return: shape的约束表达式列表
        """
        logger.debug("Start build param shape constraint")
        shape_static_value_expr_list = []
        relation_params = list(self.case_input_map.keys())
        for param_name in relation_params:
            if param_name not in self.case_input_map:
                continue
            case_param_shape = self.case_input_map.get(param_name).shape
            # 如果参数的shape属性为None，表明该参数不是tensor，没有shape属性，不需要添加shape约束
            if case_param_shape is None:
                continue
            shape_static_value_expr_list.append(f"{param_name}.shape == {case_param_shape}")

        no_conflicts_shape_expr = self.choice_no_conflicts_expr(param_union_expr=constraint_exprs,
                                                                param_static_expr_list=shape_static_value_expr_list)
        logger.debug("End build param shape constraint")
        return no_conflicts_shape_expr

    def build_param_format_constraint(self, constraint_exprs: List[str]) -> List[str]:
        """
        查找是否有参数的format属性取值已确定，如果有，是否和已有的规则冲突，如不冲突，则添加为求解条件，避免每次求解结构都相同，如冲突，则不添加
        :param constraint_exprs: 约束表达式对象
        :return: format的约束表达式列表
        """
        logger.debug("Start build param format constraint")
        format_static_value_expr_list = []
        relation_param = list(self.case_input_map.keys())
        for param in relation_param:
            if self.case_input_map.get(param).format is not None:
                format_static_value_expr_list.append(f"{param}.format == '{self.case_input_map.get(param).format}'")
        no_conflicts_format_expr = self.choice_no_conflicts_expr(param_union_expr=constraint_exprs,
                                                                 param_static_expr_list=format_static_value_expr_list)
        logger.debug("End build param format constraint")
        return no_conflicts_format_expr

    def build_param_length_constraint(self, constraint_exprs: List[str]) -> List[str]:
        logger.debug("Start build param length constraint")
        length_static_value_expr_list = []
        relation_param = list(self.case_input_map.keys())
        for param in relation_param:
            if self.case_input_map.get(param).type in ParamModelConfig.LIST_ATK_TYPE and self.case_input_map.get(
                    param).length is not None:
                length_static_value_expr_list.append(f"len({param}) == {self.case_input_map.get(param).length}")
        no_conflicts_length_expr = self.choice_no_conflicts_expr(param_union_expr=constraint_exprs,
                                                                 param_static_expr_list=length_static_value_expr_list)
        logger.debug("End build param length constraint")
        return no_conflicts_length_expr

    def declare_param_in_z3(self, builder: Z3ConstraintBuilder):
        """
        声明每个变量，指定变量的type(Tensor, scalar，list)，以及数据类型(float, int, string, bool)
        :param relation_parameters: 约束相关的参数名称
        :param builder: z3求解器实例
        :return: None
        """
        for param_name in self.case_input_map.keys():
            param_info = self.case_input_map.get(param_name)
            if not param_info:
                continue
            param_type = param_info.type
            z3_param_type = DataMatchMap.Z3_VAR_TYPE_MAP.get(param_type, "tensor")
            param_dtype = param_info.dtype
            param_length = param_info.length
            range_values = self.case_input_map.get(param_name).range_values
            if param_type in ParamModelConfig.TENSOR_ATK_TYPE:
                dtype_domain = self.dtype_domain_data.get(param_name)
                format_domain = self.format_domain_data.get(param_name)
                builder.declare_var(param_name, type_hint=z3_param_type, dtype=param_dtype, allowed_dtypes=dtype_domain,
                                    allowed_formats=format_domain, range_value=range_values)
            else:
                builder.declare_var(param_name, z3_param_type, dtype=param_dtype, range_value=range_values,
                                    length=param_length)

    def solve_z3_constraints(self, z3_constraints: List[InterParamConstraint]):
        """
        针对所有的参数关联表达式，使用z3求解器求解
        :param: z3_constraints: 表达式列表，包含所有需要求解的表达式
        """
        expr_list = [constraint_expr.expr for constraint_expr in z3_constraints if
                     constraint_expr.expr is not None and constraint_expr.expr]
        logger.info(f"Start solving solution of constraints by Z3, operator name : {self.operator_name}")
        builder = Z3ConstraintBuilder()
        self.declare_param_in_z3(builder=builder)
        dtype_expr_list = self.build_param_dtype_constraint(expr_list)
        format_expr_list = self.build_param_format_constraint(expr_list)
        shape_expr_list = self.build_param_shape_constraint(expr_list)
        range_value_expr_list = self.build_param_range_value_constraint(expr_list)
        length_value_expr_list = self.build_param_length_constraint(expr_list)
        expr_list.extend(dtype_expr_list)
        expr_list.extend(format_expr_list)
        expr_list.extend(shape_expr_list)
        expr_list.extend(range_value_expr_list)
        expr_list.extend(length_value_expr_list)
        expr_dict = {f"expr : {expr}": expr for index, expr in enumerate(expr_list)}
        builder.add_constraints(expr_str_dict=expr_dict)
        logger.info("Start whole expr solve")
        solver_result = builder.solve()
        if not solver_result:
            logger.error(
                f"Z3 solver error, no solution can satisfy constraints, operator name : {self.operator_name}")
            return False
        logger.info(f"End solving solution of constraints by Z3, operator name : {self.operator_name}")
        for param_name in self.case_input_map.keys():
            property_dict = solver_result.get(param_name, {})
            for field_name in ParamSetValueFlag.model_fields.keys():
                # 参数必须在case_input_map中，且数据值不为None，如非tensor参数没有shape属性，但是Z3求解器会对表达式涉及到的所有参数所有属性都会求解
                param_attr_ori_status = param_name in self.case_input_map and getattr(
                    self.case_input_map.get(param_name), field_name, None) is not None
                attr_value = property_dict.get(field_name, None)
                if param_attr_ori_status and attr_value is not None:
                    self.case_input_map[param_name].__setattr__(field_name, attr_value)
        return True
