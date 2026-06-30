# -*- coding: UTF-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""
版权信息：华为技术有限公司，版本所有(C) 2026-2026
修改记录：2026/3/26 10:05
功能：基于z3求解器求解SMT问题，约束表达式必须为标准的python合法表达式
使用示例：
    builder = Z3ConstraintBuilder()

    # 1. 声明变量 x，限制其 dtype 只能是 int8 或 uint8
    builder.declare_var("x", allowed_dtypes=["int8", "uint8"])

    # 2. 声明变量 y，限制其 dtype 只能是 float32
    builder.declare_var("y", allowed_dtypes=["float32"])

    # 3. 变量 z 未显式声明，将在使用时自动创建，默认支持所有类型

    # 添加约束
    builder.add_constraint('x.dtype == "int8"')  # 有效约束
    # builder.add_constraint('x.dtype == "float32"') # 这将导致 UNSAT，因为 float32 不在 x 的允许列表中

    builder.add_constraint('len(x.shape) == 2')
    builder.add_constraint('x[0] > 100')  # 触发 x 的数据约束，由于 x 是 int8，范围是 [-128, 127]，所以 x[0] 最大为 127

    builder.add_constraint('y[0] < 0.5')  # 触发 y 的数据约束

    # 求解
    builder.solve()
"""
import ast
import re
from typing import List, Dict

import z3

from generators.common_utils.logger_util import LazyLogger
from generators.data_definition.constants import DataMatchMap
from generators.param_constraint_solve.expression_preprocess_utils import ASTtoZ3Converter, TensorVar, ScalarVar, ListVar

logger = LazyLogger()


class ExpressionPreprocessor:
    """Preprocesses expressions before solving."""

    @staticmethod
    def apply_keyword_replace(expr: str) -> str:
        for keyword, replacement in DataMatchMap.EXPR_KEYWORD_REPLACE.items():
            if replacement is None:
                expr = expr.replace(keyword, 'None')
            elif isinstance(replacement, str):
                expr = expr.replace(keyword, f"'{replacement}'")
            else:
                expr = expr.replace(keyword, str(replacement))
        for keyword, replacement in DataMatchMap.ACL_DTYPE_TRANSFER_TENSOR_MAP.items():
            replacement_str = f"{replacement}" if isinstance(replacement, str) else str(replacement)
            expr = re.sub(rf"\b{re.escape(keyword)}\b", replacement_str, expr)
        return expr

    @staticmethod
    def preprocess_expressions(expressions: List[str]) -> List[str]:
        processed = []
        for expr in expressions:
            expr = ExpressionPreprocessor.apply_keyword_replace(expr)
            processed.append(expr)
        return processed

    @staticmethod
    def validate_expression(expr: str) -> bool:
        try:
            ast.parse(expr, mode='eval')
            return True
        except SyntaxError as e:
            logger.error(f"Expression '{expr}' is invalid by ast validation, err msg : {str(e)}")
            return False

    @staticmethod
    def validate_expression_without_bool(expr: str) -> bool:
        """
        判断expr是否为合法表达式，且本身不为True/False
        """
        try:
            tree = ast.parse(expr, mode='eval')
            # tree.body 是表达式节点
            node = tree.body

            # 布尔字面量是 ast.Constant 且值为 bool
            if isinstance(node, ast.Constant) and isinstance(node.value, bool):
                return False
            # 其他情况（包括其他类型的表达式）都认为是有效的
            return True
        except SyntaxError as e:
            # 解析失败，说明不是合法的表达式
            logger.error(f"Validate expression without bool failed, err msg : {str(e)}")
            return False


class Z3ConstraintBuilder:
    _VAR_FACTORY = {
        'tensor': (TensorVar, lambda self, kwargs: (self.solver, kwargs.get("dtype"), kwargs.get('allowed_dtypes'),
                                                    kwargs.get('allowed_formats'), kwargs.get('range_value'))),
        'scalar': (ScalarVar, lambda self, kwargs: (self.solver, kwargs.get('dtype'), kwargs.get('range_value'))),
        'list': (ListVar, lambda self, kwargs: (self.solver, kwargs.get('dtype'), kwargs.get('range_value'),
                                                kwargs.get("length"))),
    }

    def __init__(self):
        self.solver = z3.Solver()
        self.var_map = {}
        self._slice_counter = 0

    def get_next_slice_id(self):
        self._slice_counter += 1
        return self._slice_counter

    def declare_var(self, var_name, type_hint="scalar", dtype=None, allowed_dtypes=None, allowed_formats=None,
                    range_value=None, length=None):
        if var_name in self.var_map:
            logger.warning(f"[Warn] var {var_name} already declared")
            return

        if type_hint not in self._VAR_FACTORY:
            logger.error(f"[Declare] Unsupported type_hint '{type_hint}' for var '{var_name}'")
            return

        kwargs = {
            'dtype': dtype, 'allowed_dtypes': allowed_dtypes,
            'allowed_formats': allowed_formats, 'range_value': range_value,
            'length': length
        }
        cls, param_fn = self._VAR_FACTORY[type_hint]

        if type_hint in ['scalar', 'list'] and not dtype:
            # List 现在可以通过 range_value 推断类型，所以不强制报错，但 scalar 必须有
            if type_hint == 'scalar':
                logger.error(f"[Declare] Type '{type_hint}' requires 'dtype' argument for var '{var_name}'")
                return

        try:
            var_obj = cls(var_name, *param_fn(self, kwargs))
            self.var_map[var_name] = var_obj
            logger.info(f"[Declare] {var_name} -> {type_hint} (dtype: {dtype}, range_value: {range_value})")
        except Exception as e:
            logger.error(f"[Declare] Failed to create var '{var_name}', err msg : {e}")

    def get_or_create_var(self, var_name):
        if var_name not in self.var_map:
            self.declare_var(var_name, type_hint="tensor")
        return self.var_map[var_name]

    def add_constraints(self, expr_str_dict: Dict[str, str]):
        for expr_str_name, expr_str in expr_str_dict.items():
            replace_expr = ExpressionPreprocessor.apply_keyword_replace(expr_str)
            if ExpressionPreprocessor.validate_expression(replace_expr):
                self.add_constraint(expr_str_name, replace_expr)

    def add_constraint(self, expr_name, expr_str):
        try:
            tree = ast.parse(expr_str, mode='eval')
            converter = ASTtoZ3Converter(self)
            z3_constraint = converter.visit(tree.body)
            if z3_constraint is not None:
                self.solver.assert_and_track(z3_constraint, expr_name)
                logger.info(f"[OK] {expr_str}")
            else:
                logger.debug(f"[SKIP] {expr_str}: converter returned None, ignored")
        except Exception as e:
            logger.error(f"[FAIL] {expr_str}: {e}")

    def solve(self):
        check_res = self.solver.check()
        logger.info(f"Solve result: {check_res}")
        results = {}
        if check_res == z3.sat:
            m = self.solver.model()
            for name, var_obj in self.var_map.items():
                logger.info(f"Param name : {name}")
                results[name] = var_obj.resolve_model(m)
                logger.info(f"  {name}: {results[name]}")
        elif check_res == z3.unknown:
            logger.error(f"Solve result unknown, unknown reason : {self.solver.reason_unknown()}")
        else:
            logger.error("Causes of Constraint Conflicts")
            unsat_cores = self.solver.unsat_core()
            for core in unsat_cores:
                logger.error(f"   {core}   ")
        return results
