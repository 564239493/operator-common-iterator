# -*- coding: UTF-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""
版权信息：华为技术有限公司，版本所有(C) 2026-2026
修改记录：2026/3/26 9:56
功能：基于AST转换器，对输入的表达式进行处理，转换为Z3可接受的数据格式
"""
import ast
import operator

import z3

from agent.generators.common_utils.logger_util import LazyLogger
from agent.generators.param_constraint_solve.param_var_definition import TensorVar, ListVar, ScalarVar, DTYPE_MAP

logger = LazyLogger()


# ==========================================
# AST 转换器
# ==========================================
class ASTtoZ3Converter(ast.NodeVisitor):
    """
    将 Python AST 节点转换为 Z3 表达式
    支持的节点类型：
    - 常量（int/float/str/list/tuple）
    - 变量（通过 VariableBuilder 管理）
    - 属性访问（如 .dtype/.shape）
    - 条件表达式（if-else）
    - 布尔运算（and/or）
    - 一元运算（not/-）
    - 比较运算（==/!=/< etc.）
    - 二元运算（+/-/* etc.）
    - 下标与切片
    - 函数调用（len/all/any/max/min）
    """
    # --- 二元运算 ---
    _BIN_OP_MAP = {
        ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
    }
    # --- 函数调用 ---
    _CALL_DISPATCH_TABLE = {
        'len': '_handle_len', 'all': '_handle_all', 'any': '_handle_any',
        'max': '_handle_max_min', 'min': '_handle_max_min',
        'sum': '_handle_sum',
    }
    def __init__(self, builder):
        self.builder = builder

    def visit(self, node):
        method = 'visit_' + node.__class__.__name__
        visitor = getattr(self, method, self.generic_visit)
        return visitor(node)

    def generic_visit(self, node):
        raise NotImplementedError(f"Unsupported syntax node: {type(node).__name__}")

    # --- 辅助方法 ---
    @staticmethod
    def _promote_numeric_types(left, right):
        if isinstance(left, int):
            left = z3.IntVal(left)
        elif isinstance(left, float):
            left = z3.RealVal(left)
        if isinstance(right, int):
            right = z3.IntVal(right)
        elif isinstance(right, float):
            right = z3.RealVal(right)
        if not z3.is_expr(left) or not z3.is_expr(right): return left, right
        try:
            if z3.is_real(left) and z3.is_int(right):
                return left, z3.ToReal(right)
            elif z3.is_int(left) and z3.is_real(right):
                return z3.ToReal(left), right
        except:
            pass
        return left, right

    # --- 节点访问 ---
    def visit_Constant(self, node):
        return node.value

    def visit_Name(self, node):
        return self.builder.get_or_create_var(node.id).get_z3_expr()

    def visit_List(self, node):
        return [self.visit(e) for e in node.elts]

    def visit_Tuple(self, node):
        return [self.visit(e) for e in node.elts]

    def visit_Attribute(self, node):
        var_name = node.value.id
        t_var = self.builder.get_or_create_var(var_name)
        if isinstance(t_var, TensorVar):
            if node.attr == 'dtype':
                return t_var.dtype
            elif node.attr == 'shape':
                return t_var.shape
            elif node.attr == 'format':
                return t_var.format
            elif node.attr == 'range_value':
                return t_var.range_value
        elif isinstance(t_var, ListVar):
            if node.attr == 'dtype':
                return t_var.dtype
            elif node.attr == 'range_value':
                return t_var.range_value
        elif isinstance(t_var, ScalarVar):
            if node.attr == 'dtype':
                return t_var.dtype
            # 对于 ScalarVar，range_value 即为其值本身
            elif node.attr == 'range_value':
                return t_var.z3_var
        raise AttributeError(f"Attribute '{node.attr}' not supported for var '{var_name}'")

    def visit_IfExp(self, node):
        return z3.If(self.visit(node.test), self.visit(node.body), self.visit(node.orelse))

    def _is_pure_guard(self, node):
        """检查 AST 节点是否完全由 is None / is not None 检查组成（守卫条件）"""
        if isinstance(node, ast.Compare):
            if len(node.ops) == 1 and isinstance(node.ops[0], (ast.Is, ast.IsNot)):
                if len(node.comparators) == 1 and isinstance(node.comparators[0], ast.Constant) and node.comparators[0].value is None:
                    return True
        if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.And):
            return all(self._is_pure_guard(v) for v in node.values)
        return False

    def visit_BoolOp(self, node):
        if isinstance(node.op, ast.And):
            guards = [v for v in node.values if self._is_pure_guard(v)]
            rests = [v for v in node.values if not self._is_pure_guard(v)]
            if guards and rests:
                guard_exprs = [self.visit(g) for g in guards]
                rest_exprs = [self.visit(r) for r in rests]
                return z3.Implies(z3.And(*guard_exprs), z3.And(*rest_exprs))
            vals = [self.visit(v) for v in node.values]
            return z3.And(*vals)
        elif isinstance(node.op, ast.Or):
            vals = [self.visit(v) for v in node.values]
            return z3.Or(*vals)

    def visit_UnaryOp(self, node):
        op = self.visit(node.operand)
        if isinstance(node.op, ast.Not): return z3.Not(op)
        if isinstance(node.op, ast.USub): return -op
        raise NotImplementedError(f"Unsupported unary operator: {type(node.op).__name__}")

    def visit_Compare(self, node):
        left = self.visit(node.left)
        # 【修改】移除对 left 为 None 的直接报错，因为 None 可能是合法值

        ops = node.ops
        comps = [self.visit(c) for c in node.comparators]
        # 【修改】移除对 comps 包含 None 的直接报错

        res = []
        cur_left = left
        for op, right in zip(ops, comps):
            # 处理类型转换（仅针对非 None 值）
            def convert_operand(l, r):
                if r is None: return r  # 不转换 None
                if z3.is_expr(l) and l.sort().name() == 'DType' and isinstance(r, str):
                    if r in DTYPE_MAP: return DTYPE_MAP[r]
                    raise ValueError(f"Unknown dtype string: {r}")
                elif z3.is_string(l) and isinstance(r, str):
                    return z3.StringVal(r)
                return r

            right = convert_operand(cur_left, right)

            # --- 处理 is / is not ---
            if isinstance(op, (ast.Is, ast.IsNot)):
                if right is None:
                    # 优先使用变量的 is_present 标志（使 None 检查成为 Z3 约束）
                    if isinstance(node.left, ast.Name):
                        var_name = node.left.id
                        if var_name in self.builder.var_map:
                            var_obj = self.builder.var_map[var_name]
                            if hasattr(var_obj, 'is_present'):
                                result = z3.Not(var_obj.is_present) if isinstance(op, ast.Is) else var_obj.is_present
                                res.append(result)
                                cur_left = right
                                continue
                    # 回退：Z3 表达式永远不是 None，Python 常量直接比较
                    if z3.is_expr(cur_left):
                        result = z3.BoolVal(False) if isinstance(op, ast.Is) else z3.BoolVal(True)
                    else:
                        result = (cur_left is None) if isinstance(op, ast.Is) else (cur_left is not None)
                    res.append(result)
                else:
                    raise NotImplementedError(f"'is' operator only supports None comparison, got {right}")
                cur_left = right
                continue

            # --- 处理其他操作符 ---
            # 如果操作数包含 None 且不是 is/is not，则报错
            if cur_left is None or right is None:
                raise TypeError(f"Operator {type(op).__name__} does not support None operands")

            # 【新增】处理 Python List 与 Z3 Seq 的比较
            if z3.is_seq(cur_left) and isinstance(right, (list, tuple)):
                if isinstance(op, ast.Eq):
                    len_c = z3.Length(cur_left) == len(right)
                    elem_cs = [cur_left[i] == v for i, v in enumerate(right)]
                    res.append(z3.And(len_c, *elem_cs))
                elif isinstance(op, ast.NotEq):
                    len_diff = z3.Length(cur_left) != len(right)
                    elem_diffs = [cur_left[i] != v for i, v in enumerate(right)]
                    res.append(z3.Or(len_diff, *elem_diffs))
                else:
                    raise TypeError(f"Cannot compare Seq and List with op {type(op).__name__}")

            # 【新增】处理 Python List 与 Python List 的比较 (全常量情况)
            elif isinstance(cur_left, (list, tuple)) and isinstance(right, (list, tuple)):
                if isinstance(op, ast.Eq):
                    res.append(cur_left == right)
                elif isinstance(op, ast.NotEq):
                    res.append(cur_left != right)
                else:
                    raise NotImplementedError

            elif z3.is_array(cur_left) and not z3.is_array(right):
                res.append(self._handle_array_scalar_compare(cur_left, op, right))
            else:
                cur_left, right = self._promote_numeric_types(cur_left, right)
                if isinstance(op, ast.Eq):
                    if isinstance(right, (list, tuple)) and z3.is_seq(cur_left):
                        len_constraint = z3.Length(cur_left) == len(right)
                        elem_constraints = [cur_left[i] == val for i, val in enumerate(right)]
                        res.append(z3.And(len_constraint, *elem_constraints))
                    else:
                        res.append(cur_left == right)
                elif isinstance(op, ast.NotEq):
                    if isinstance(right, (list, tuple)) and z3.is_seq(cur_left):
                        res.append(z3.Length(cur_left) != len(right))
                    else:
                        res.append(cur_left != right)
                elif isinstance(op, ast.Lt):
                    res.append(cur_left < right)
                elif isinstance(op, ast.LtE):
                    res.append(cur_left <= right)
                elif isinstance(op, ast.Gt):
                    res.append(cur_left > right)
                elif isinstance(op, ast.GtE):
                    res.append(cur_left >= right)
                elif isinstance(op, ast.In):
                    if isinstance(right, list):
                        res.append(z3.Or([cur_left == v for v in right]))
                    elif z3.is_seq(right):
                        length = z3.Length(right)
                        idx = z3.Int(f"__in_{self.builder.get_next_slice_id()}")
                        res.append(z3.Exists([idx], z3.And(idx >= 0, idx < length, right[idx] == cur_left)))
                    else:
                        raise TypeError("'in' right operand must be a list or sequence")
                elif isinstance(op, ast.NotIn):
                    if isinstance(right, list):
                        res.append(z3.And([cur_left != v for v in right]))
                    elif z3.is_seq(right):
                        length = z3.Length(right)
                        idx = z3.Int(f"__notin_{self.builder.get_next_slice_id()}")
                        res.append(z3.ForAll([idx], z3.Implies(z3.And(idx >= 0, idx < length), right[idx] != cur_left)))
                    else:
                        raise TypeError("'not in' right operand must be a list or sequence")
                else:
                    raise NotImplementedError(f"Unsupported op: {type(op).__name__}")
            cur_left = right

        if not res: return z3.BoolVal(True)
        return z3.And(*res) if len(res) > 1 else res[0]

    def _handle_array_scalar_compare(self, arr, op, scalar):
        idx = z3.Int('idx')
        elem = z3.Select(arr, idx) if z3.is_array(arr) else arr[idx]
        bounds = idx >= 0 if z3.is_array(arr) else z3.And(idx >= 0, idx < z3.Length(arr))

        elem, scalar = self._promote_numeric_types(elem, scalar)
        if isinstance(op, ast.Gt):
            cond = elem > scalar
        elif isinstance(op, ast.GtE):
            cond = elem >= scalar
        elif isinstance(op, ast.Lt):
            cond = elem < scalar
        elif isinstance(op, ast.LtE):
            cond = elem <= scalar
        elif isinstance(op, ast.Eq):
            cond = elem == scalar
        elif isinstance(op, ast.NotEq):
            cond = elem != scalar
        else:
            raise NotImplementedError
        return z3.ForAll([idx], z3.Implies(bounds, cond))

    def visit_BinOp(self, node):
        left = self.visit(node.left)
        right = self.visit(node.right)
        if left is None or right is None: raise ValueError(f"Binary op failed: {ast.dump(node)}")

        # 常量折叠
        if not z3.is_expr(left) and not z3.is_expr(right):
            if isinstance(node.op, ast.Pow): return left ** right
            op_func = self._BIN_OP_MAP.get(type(node.op))
            if op_func: return op_func(left, right)
            raise NotImplementedError(f"Unsupported constant op: {type(node.op).__name__}")

        left, right = self._promote_numeric_types(left, right)
        if isinstance(node.op, ast.Pow): return self._handle_pow(left, right)

        op_func = self._BIN_OP_MAP.get(type(node.op))
        if op_func: return op_func(left, right)
        raise NotImplementedError(f"Unsupported binary op: {type(node.op).__name__}")

    def _handle_pow(self, left, right):
        if not z3.is_expr(right) and isinstance(right, (int, float)):
            exp_val = int(right)
            if exp_val < 0: raise NotImplementedError("Negative exponents not supported")
            if exp_val == 0: return 1
            if exp_val == 1: return left
            if exp_val > 8: raise NotImplementedError(f"Exponent {exp_val} too large")
            res = left
            for _ in range(exp_val - 1): res = res * left
            return res
        raise NotImplementedError("Only small integer constant exponents supported")

    # --- 下标与切片 ---
    def visit_Subscript(self, node):
        value = self.visit(node.value)
        if value is None: raise ValueError(f"Subscript on None")
        if isinstance(node.slice, ast.Slice):
            seq = value if z3.is_seq(value) else (value.z3_var if hasattr(value, 'z3_var') else None)
            if seq is None: raise TypeError("Slice on non-sequence")
            return self._handle_slice(seq, node.slice)

        idx = self.visit(node.slice)
        if idx is None: raise ValueError("Subscript index failed")

        actual_idx = idx
        if hasattr(value, 'z3_var') and z3.is_seq(value.z3_var):
            actual_idx = z3.If(idx < 0, z3.Length(value.z3_var) + idx, idx)
        elif z3.is_seq(value):
            actual_idx = z3.If(idx < 0, z3.Length(value) + idx, idx)

        if hasattr(value, 'get_element_at'):
            return value.get_element_at(actual_idx)
        elif z3.is_seq(value):
            return value[actual_idx]
        raise TypeError(f"Invalid subscript target: {type(value).__name__}")

    def _handle_slice(self, seq, slice_node):
        len_seq = z3.Length(seq)
        start = z3.IntVal(0) if slice_node.lower is None else self.visit(slice_node.lower)
        start = z3.If(start < 0, len_seq + start, start)
        stop = len_seq if slice_node.upper is None else self.visit(slice_node.upper)
        stop = z3.If(stop < 0, len_seq + stop, stop)
        slice_id = self.builder.get_next_slice_id()
        slice_var = z3.Const(f"__slice_{slice_id}", z3.SeqSort(z3.IntSort()))
        slice_len = z3.If(stop > start, stop - start, 0)
        self.builder.solver.add(z3.Length(slice_var) == slice_len)
        k = z3.Int(f"__k_{slice_id}")
        body = z3.Implies(z3.And(k >= 0, k < slice_len),
                          z3.And(start + k < len_seq, slice_var[k] == seq[start + k]))
        self.builder.solver.add(z3.ForAll([k], body))
        return slice_var

    def visit_Call(self, node):
        func_name = node.func.id
        handler_name = self._CALL_DISPATCH_TABLE.get(func_name)
        if handler_name:
            return getattr(self, handler_name)(node, func_name)
        raise NotImplementedError(f"Unsupported function: {func_name}")

    def _handle_len(self, node, func_name):
        if len(node.args) != 1: raise ValueError("len() expects 1 argument")
        arg = self.visit(node.args[0])
        if arg is None: raise ValueError("len() argument failed")
        if hasattr(arg, 'shape'): return z3.Length(arg.shape)
        if hasattr(arg, 'z3_var') and z3.is_seq(arg.z3_var): return z3.Length(arg.z3_var)
        if z3.is_seq(arg): return z3.Length(arg)
        raise TypeError(f"len() on unsupported type: {type(arg).__name__}")

    def _handle_all(self, node, func_name):
        if len(node.args) == 1 and isinstance(node.args[0], ast.GeneratorExp):
            return self._handle_quantifier(node.args[0], z3.ForAll)
        raise ValueError("all() argument must be a generator expression")

    def _handle_any(self, node, func_name):
        if len(node.args) == 1 and isinstance(node.args[0], ast.GeneratorExp):
            return self._handle_quantifier(node.args[0], z3.Exists)
        raise ValueError("any() argument must be a generator expression")

    def _handle_max_min(self, node, func_name):
        if len(node.args) == 0: raise ValueError(f"{func_name}() requires at least 1 argument")
        args_z3 = [self.visit(arg) for arg in node.args]
        if any(a is None for a in args_z3): raise ValueError(f"{func_name}() argument failed")
        is_max = (func_name == 'max')
        if len(node.args) == 1:
            return self._handle_max_min_seq(args_z3[0], is_max)
        else:
            return self._handle_max_min_scalars(args_z3, is_max)

    def _handle_range_quantifier(self, comprehension, condition_ast, quantifier_op):
        range_call = comprehension.iter
        if len(range_call.args) not in (1, 2, 3):
            raise ValueError("range() expects 1-3 arguments")

        args = [self.visit(a) for a in range_call.args]
        if any(a is None for a in args):
            raise ValueError("range() argument failed")

        if len(args) == 1:
            start, stop, step = z3.IntVal(0), args[0], z3.IntVal(1)
        elif len(args) == 2:
            start, stop, step = args[0], args[1], z3.IntVal(1)
        else:
            start, stop, step = args[0], args[1], args[2]

        if isinstance(step, int):
            step = z3.IntVal(step)
        if isinstance(start, int):
            start = z3.IntVal(start)

        loop_var_name = comprehension.target.id
        slice_id = self.builder.get_next_slice_id()
        k = z3.Int(f"__k_q_{slice_id}")

        class TempVisitor(ASTtoZ3Converter):
            def __init__(self, builder, vn, ph):
                super().__init__(builder)
                self.vn, self.ph = vn, ph
            def visit_Name(self, node):
                if node.id == self.vn: return self.ph
                return super().visit_Name(node)

        cond_expr = TempVisitor(self.builder, loop_var_name, k).visit(condition_ast)
        bounds = z3.And(k >= start, k < stop)
        if not (isinstance(step, z3.IntNumRef) and step.as_long() == 1):
            bounds = z3.And(bounds, (k - start) % step == 0)

        if quantifier_op == z3.ForAll:
            return z3.ForAll([k], z3.Implies(bounds, cond_expr))
        else:
            return z3.Exists([k], z3.And(bounds, cond_expr))

    # --- 量词与 Max/Min 内部实现 ---
    def _handle_quantifier(self, gen_node, quantifier_op):
        try:
            comprehension = gen_node.generators[0]

            # --- range() special case ---
            if isinstance(comprehension.iter, ast.Call) and isinstance(comprehension.iter.func, ast.Name) and comprehension.iter.func.id == 'range':
                return self._handle_range_quantifier(comprehension, gen_node.elt, quantifier_op)

            iter_target = self.visit(comprehension.iter)
            if iter_target is None: raise ValueError("Quantifier iter target is None")

            if hasattr(iter_target, 'get_element_sort'):
                element_sort = iter_target.get_element_sort()
                is_sequence = not isinstance(iter_target, TensorVar)
                if isinstance(iter_target, TensorVar):
                    _ = iter_target.range_value
                    target_obj = iter_target.range_value
                    is_sequence = False
                else:
                    target_obj = iter_target.get_z3_expr()
            elif z3.is_seq(iter_target):
                target_obj = iter_target
                is_sequence = True
                element_sort = z3.IntSort()
            elif z3.is_array(iter_target):
                target_obj = iter_target
                is_sequence = False
                element_sort = iter_target.sort().range()  # Array的值域sort
            else:
                raise TypeError(f"Cannot iterate over type: {type(iter_target).__name__}")

            loop_var_name = comprehension.target.id
            condition_ast = gen_node.elt
            slice_id = self.builder.get_next_slice_id()
            k = z3.Int(f"__k_q_{slice_id}")
            placeholder = z3.Const(f"__ph_{loop_var_name}", element_sort)

            class TempVisitor(ASTtoZ3Converter):
                def __init__(self, builder, vn, ph):
                    super().__init__(builder)
                    self.vn, self.ph = vn, ph

                def visit_Name(self, node):
                    if node.id == self.vn: return self.ph
                    return super().visit_Name(node)

            cond_expr = TempVisitor(self.builder, loop_var_name, placeholder).visit(condition_ast)
            actual_val = target_obj[k] if is_sequence else z3.Select(target_obj, k)
            final_cond = z3.substitute(cond_expr, (placeholder, actual_val))

            if quantifier_op == z3.ForAll:
                bounds = z3.And(k >= 0, k < z3.Length(target_obj)) if is_sequence else (k >= 0)
                return z3.ForAll([k], z3.Implies(bounds, final_cond))
            else:
                bounds = z3.And(k >= 0, k < z3.Length(target_obj)) if is_sequence else (k >= 0)
                return z3.Exists([k], z3.And(bounds, final_cond))
        except Exception as e:
            logger.error(f"[AST] Quantifier handling failed: {e}", exc_info=True)
            raise e

    def _handle_max_min_scalars(self, values, is_max):
        has_real = any(z3.is_real(v) or isinstance(v, float) for v in values)
        converted_values = []
        for v in values:
            if has_real:
                if z3.is_int(v):
                    converted_values.append(z3.ToReal(v))
                elif isinstance(v, int):
                    converted_values.append(z3.RealVal(v))
                elif isinstance(v, float):
                    converted_values.append(z3.RealVal(v))
                else:
                    converted_values.append(v)
            else:
                if isinstance(v, int):
                    converted_values.append(z3.IntVal(v))
                else:
                    converted_values.append(v)
        result = converted_values[0]
        for i in range(1, len(converted_values)):
            next_val = converted_values[i]
            if result.sort() != next_val.sort(): raise TypeError(f"Type mismatch in max/min")
            cond = result >= next_val if is_max else result <= next_val
            result = z3.If(cond, result, next_val)
        return result

    def _handle_max_min_seq(self, target, is_max):
        # 优化：如果是具体单值，直接返回
        spec = getattr(target, '_range_spec', None)
        if spec is not None and not isinstance(spec, (list, tuple)):
            logger.info(f"[Optimize] max/min on fixed value tensor {target.name}: {spec}")
            if isinstance(spec, bool):
                return z3.BoolVal(spec)
            elif isinstance(spec, int):
                return z3.IntVal(spec)
            elif isinstance(spec, float):
                return z3.RealVal(spec)
            elif isinstance(spec, str):
                return z3.StringVal(spec)

        if isinstance(target, TensorVar):
            arr = target.range_value
        elif isinstance(target, ListVar):
            arr = target.z3_var
        else:
            raise TypeError("max/min requires Tensor or List")

        element_sort = target.get_element_sort()
        res_name = f"__{'max' if is_max else 'min'}_{target.name}_{self.builder.get_next_slice_id()}"
        result_var = z3.Const(res_name, element_sort)

        length = self._get_sequence_length_for_maxmin(target)
        k = z3.Int(f"k_{res_name}")
        bounds = z3.And(k >= 0, k < length)

        if isinstance(target, TensorVar):
            elem = z3.Select(arr, k)
        else:
            elem = arr[k]

        self.builder.solver.add(length > 0)
        self.builder.solver.add(z3.Exists([k], z3.And(bounds, result_var == elem)))
        if is_max:
            self.builder.solver.add(z3.ForAll([k], z3.Implies(bounds, elem <= result_var)))
        else:
            self.builder.solver.add(z3.ForAll([k], z3.Implies(bounds, elem >= result_var)))

        # 优化：如果是范围，添加边界提示
        if isinstance(spec, (list, tuple)) and len(spec) == 2:
            min_val, max_val = spec
            if is_max:
                self.builder.solver.add(result_var <= max_val)
                logger.info(f"[Optimize] Bound hint: max({target.name}) <= {max_val}")
            else:
                self.builder.solver.add(result_var >= min_val)
                logger.info(f"[Optimize] Bound hint: min({target.name}) >= {min_val}")

        return result_var

    def _get_sequence_length_for_maxmin(self, target):
        if isinstance(target, ListVar):
            return z3.Length(target.z3_var)
        elif isinstance(target, TensorVar):
            _ = target.range_value
            rank = z3.Length(target.shape)
            self.builder.solver.add(rank == 1)
            return target.shape[0]
        raise NotImplementedError(f"Cannot determine length for type {type(target).__name__}")

    def _handle_sum(self, node, func_name):
        if len(node.args) == 0:
            raise ValueError("sum() requires at least 1 argument")

        if len(node.args) == 1 and isinstance(node.args[0], ast.GeneratorExp):
            return self._handle_sum_generator(node.args[0])

        if len(node.args) == 1 and isinstance(node.args[0], ast.Name):
            var = self.builder.get_or_create_var(node.args[0].id)
            if isinstance(var, TensorVar):
                return self._sum_tensor_elements(var)
            if isinstance(var, ListVar):
                return self._sum_z3_sequence(var.z3_var, var.get_element_sort())

        args = [self.visit(arg) for arg in node.args]
        if any(a is None for a in args):
            raise ValueError("sum() argument failed")

        if len(args) == 1:
            arg = args[0]
            if isinstance(arg, TensorVar):
                return self._sum_tensor_elements(arg)
            if z3.is_seq(arg):
                return self._sum_z3_sequence(arg, z3.IntSort())
            if isinstance(arg, (list, tuple)):
                nums = [v for v in arg if isinstance(v, (int, float))]
                return sum(nums) if nums else 0
            if z3.is_expr(arg) or isinstance(arg, (int, float, bool)):
                return arg
            raise TypeError(f"sum() on unsupported type: {type(arg).__name__}")

        return self._sum_scalars(args)

    def _sum_z3_sequence(self, seq, element_sort=None):
        if element_sort is None:
            element_sort = z3.IntSort()

        seq_sort = seq.sort()
        slice_id = self.builder.get_next_slice_id()
        func_name = f"__sum_seq_{slice_id}"

        SumSeq = z3.RecFunction(func_name, seq_sort, element_sort)
        seq_var = z3.Const(f"{func_name}_arg", seq_sort)

        if element_sort == z3.IntSort():
            zero = z3.IntVal(0)
        elif element_sort == z3.RealSort():
            zero = z3.RealVal(0)
        else:
            zero = z3.IntVal(0)

        z3.RecAddDefinition(SumSeq, [seq_var],
            z3.If(z3.Length(seq_var) == 0,
                   zero,
                   seq_var[0] + SumSeq(z3.SubSeq(seq_var, 1, z3.Length(seq_var) - 1))))

        return SumSeq(seq)

    def _sum_tensor_elements(self, tensor_var):
        arr = tensor_var.range_value
        element_sort = tensor_var.get_element_sort()

        slice_id = self.builder.get_next_slice_id()
        func_name = f"__sum_tensor_{tensor_var.name}_{slice_id}"

        array_sort = z3.ArraySort(z3.IntSort(), element_sort)

        SumArr = z3.RecFunction(func_name, array_sort, z3.IntSort(), z3.IntSort(), element_sort)
        arr_var = z3.Const(f"{func_name}_arr", array_sort)
        start_var = z3.Int(f"{func_name}_start")
        end_var = z3.Int(f"{func_name}_end")

        if element_sort == z3.IntSort():
            zero = z3.IntVal(0)
        elif element_sort == z3.RealSort():
            zero = z3.RealVal(0)
        else:
            zero = z3.IntVal(0)

        z3.RecAddDefinition(SumArr, [arr_var, start_var, end_var],
            z3.If(start_var >= end_var,
                   zero,
                   z3.Select(arr_var, start_var) + SumArr(arr_var, start_var + 1, end_var)))

        rank = z3.Length(tensor_var.shape)
        self.builder.solver.add(rank == 1)
        n_elements = tensor_var.shape[0]
        return SumArr(arr, z3.IntVal(0), n_elements)

    def _sum_scalars(self, values):
        result = values[0]
        for v in values[1:]:
            result, v = self._promote_numeric_types(result, v)
            result = result + v
        return result

    def _handle_sum_generator(self, gen_node):
        raise NotImplementedError("sum() with generator expression is not yet supported")
