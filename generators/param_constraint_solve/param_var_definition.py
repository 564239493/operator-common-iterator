# -*- coding: UTF-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""
版权信息：华为技术有限公司，版本所有(C) 2026-2026
修改记录：2026/5/11 14:28
功能：根据参数类型(Tensor, Scalar, List)构建Z3变量，支持Z3表达式根据属性获取取值，实现python标准变量到Z3变量的编码，
以及Z3变量到python标准变量的解码
"""
import math
import random

import z3
from z3 import FPSort

from agent.generators.common_utils.data_handle_utils import DataHandleUtil
from agent.generators.common_utils.logger_util import LazyLogger
from agent.generators.data_definition.constants import DataMatchMap

logger = LazyLogger()


# 辅助解析函数
def _parse_int_value(v):
    """健壮的整数解析"""
    # 1. 检查是否为具体的整数值
    if z3.is_int_value(v):
        return v.as_long()
    # 2. 检查是否为具体的实数值 (兼容类型提升)
    if z3.is_rational_value(v):
        return int(float(v.as_fraction()))
    # 3. 【新增】检查是否为未赋值的常量/变量
    if z3.is_const(v):
        # 如果是未赋值的变量，返回其字符串形式，避免崩溃
        logger.info(f"<Unassigned: {v}>")
        return None
    # 4. 兜底解析
    try:
        return int(str(v))
    except:
        raise TypeError(f"Cannot parse {v} as int")


def _parse_float_value(v):
    """从 Z3 浮点模型中提取 Python float，正确处理 inf / -inf / nan"""
    # 检查是否为特殊的浮点值
    if z3.is_fp(v):
        # 方法1：使用 Z3 提供的测试谓词
        if z3.fpIsNaN(v):
            return float('nan')
        if z3.fpIsInf(v):
            # 判断正负：fpIsPositive 或者检查符号位
            if z3.fpIsPositive(v):  # 或者 v.is_positive_infinity()
                return float('inf')
            else:
                return float('-inf')
        # 普通有限值：转为有理数再转 float
        # 注意：z3.fpToReal(v) 返回实数表达式，再 as_fraction()
        real_val = z3.fpToReal(v)
        if z3.is_rational_value(real_val):
            return float(real_val.as_fraction())
    # 如果传入的是实数或整数
    if z3.is_rational_value(v):
        return float(v.as_fraction())
    if z3.is_int_value(v):
        return float(v.as_long())
    # 未赋值等情况
    if z3.is_const(v):
        return None
    return float(str(v))


ABNORMAL_VALUE_CONFIG = {
    "inf": {
        "sort_fn": lambda: FPSort(8, 24),
        "parse_fn": _parse_float_value
    },
    "-inf": {
        "sort_fn": lambda: FPSort(8, 24),
        "parse_fn": _parse_float_value
    },
    "nan": {
        "sort_fn": lambda: FPSort(8, 24),
        "parse_fn": _parse_float_value
    }
}

TYPE_CONFIG = {
    'int': {
        'sort_fn': z3.IntSort,
        'parse_fn': _parse_int_value
    },
    'int8': {
        'sort_fn': z3.IntSort,
        'parse_fn': _parse_int_value
    },
    'uint8': {
        'sort_fn': z3.IntSort,
        'parse_fn': _parse_int_value
    },
    'int16': {
        'sort_fn': z3.IntSort,
        'parse_fn': _parse_int_value
    },
    'int32': {
        'sort_fn': z3.IntSort,
        'parse_fn': _parse_int_value
    },
    'uint16': {
        'sort_fn': z3.IntSort,
        'parse_fn': _parse_int_value
    },
    'uint32': {
        'sort_fn': z3.IntSort,
        'parse_fn': _parse_int_value
    },
    'uint64': {
        'sort_fn': z3.IntSort,
        'parse_fn': _parse_int_value
    },
    'int64': {
        'sort_fn': z3.IntSort,
        'parse_fn': _parse_int_value
    },
    'float': {
        'sort_fn': z3.RealSort,
        'parse_fn': _parse_float_value
    },
    'bfp16': {
        'sort_fn': z3.RealSort,
        'parse_fn': _parse_float_value
    },
    'bf16': {
        'sort_fn': z3.RealSort,
        'parse_fn': _parse_float_value
    },
    'fp16': {
        'sort_fn': z3.RealSort,
        'parse_fn': _parse_float_value
    },
    'fp32': {
        'sort_fn': z3.RealSort,
        'parse_fn': _parse_float_value
    },
    'fp64': {
        'sort_fn': z3.RealSort,
        'parse_fn': _parse_float_value
    },
    "double": {
        'sort_fn': z3.RealSort,
        'parse_fn': _parse_float_value
    },
    'bool': {
        'sort_fn': z3.BoolSort,
        'parse_fn': z3.is_true
    },
    'string': {
        'sort_fn': z3.StringSort,
        'parse_fn': lambda v: v.as_string()
    }
}

DType, DT_ENUMS = z3.EnumSort('DType', list(DataMatchMap.DTYPE_SPECS.keys()))
DTYPE_MAP = {name: const for name, const in zip(DataMatchMap.DTYPE_SPECS.keys(), DT_ENUMS)}


class BaseVar:
    def __init__(self, name, solver=None):
        self.name = name
        self._element_sort = None  # 由子类设置
        self.range_value = None
        self.solver = solver
        self._dtype_arg = None
        self.is_present = z3.Bool(f"{name}.is_present")

    def get_z3_expr(self):
        raise NotImplementedError

    def resolve_model(self, model):
        raise NotImplementedError

    def get_element_at(self, idx):
        return None

    def get_element_sort(self):
        return self._element_sort

    @staticmethod
    def _parse_input_spec(spec):
        """解析输入的 range_value 规格为 min, max"""
        if spec is None:
            return None, None
        if isinstance(spec, (list, tuple)):
            return (spec[0], spec[1]) if len(spec) == 2 else (None, None)
        return spec, spec  # 具体值

    @staticmethod
    def _z3_val_to_py(val):
        if z3.is_int_value(val):
            return val.as_long()
        elif z3.is_rational_value(val):
            return float(val.as_fraction())
        elif z3.is_true(val):
            return True
        elif z3.is_false(val):
            return False
        elif z3.is_string_value(val):
            return val.as_string()
        return None

    @staticmethod
    def _resolve_range_smartly(solver, z3_var, actual_val, input_spec):
        """
        标量专用：通过 Z3 探测决定返回格式。
        逻辑：先看实际值是否在输入范围内 -> 若在，探测边界合法性 -> 否则返回实际值范围
        """
        in_min, in_max = BaseVar._parse_input_spec(input_spec)

        # 1. 检查实际值是否越界
        out_of_bounds = False
        if in_min is not None and in_max is not None:
            if actual_val < in_min or actual_val > in_max:
                out_of_bounds = True

        # 2. 如果越界，直接基于实际值挖掘范围（不信任输入范围）
        if out_of_bounds:
            # 挖掘实际解附近的合法值
            vals = BaseVar._mine_distinct_values(solver, z3_var, actual_val, limit=5)
            # 此处返回的是基于实际值的范围，可能比输入范围小，或者完全不同
            if not vals: return actual_val
            unique_vals = sorted(list(set(vals)))
            if len(unique_vals) == 1: return unique_vals[0]
            # 简单判断连续性
            if all(isinstance(v, int) for v in unique_vals):
                is_cont = True
                for i in range(len(unique_vals) - 1):
                    if unique_vals[i + 1] - unique_vals[i] != 1: is_cont = False; break
                if is_cont: return (unique_vals[0], unique_vals[-1])
            return unique_vals

        # 3. 如果未越界，尝试保留输入范围（探测边界）
        if in_min is None or in_max is None:
            return actual_val  # 无输入参考

        sort = z3_var.sort()
        is_float = sort == z3.RealSort()
        is_int = sort == z3.IntSort()

        left_valid = BaseVar._check_value_validity(solver, z3_var, in_min)
        right_valid = BaseVar._check_value_validity(solver, z3_var, in_max)

        # 边界均合法 -> 返回输入区间 (元组)
        if (is_float or is_int) and left_valid and right_valid:
            return (in_min, in_max) if in_min != in_max else in_min

        # 边界非法 -> 挖掘离散解
        limit = 8 if is_int else 5
        found_vals = BaseVar._mine_distinct_values(solver, z3_var, actual_val, limit=limit)

        if not found_vals: return actual_val
        unique_vals = sorted(list(set(found_vals)))
        if len(unique_vals) == 1: return unique_vals[0]

        if is_int:
            is_continuous = True
            for i in range(len(unique_vals) - 1):
                if unique_vals[i + 1] - unique_vals[i] != 1: is_continuous = False; break
            if is_continuous: return (unique_vals[0], unique_vals[-1])

        return unique_vals

    @staticmethod
    def _resolve_range_from_set(values_set, input_spec):
        """
        张量/列表专用：基于提取出的实际值集合决定返回格式。
        """
        # 1. 提取失败时的回退逻辑
        if not values_set:
            in_min, in_max = BaseVar._parse_input_spec(input_spec)
            if in_min is not None and in_max is not None:
                if in_min == in_max:
                    return in_min  # 单值
                # 【关键】假设输入 [min, max] 代表区间，返回元组
                return (in_min, in_max)
            return input_spec  # 其他情况原样返回

        sorted_vals = sorted(list(values_set))
        actual_min, actual_max = sorted_vals[0], sorted_vals[-1]

        in_min, in_max = BaseVar._parse_input_spec(input_spec)

        # 2. 检查是否越界
        out_of_bounds = False
        if in_min is not None and in_max is not None:
            if actual_min < in_min or actual_max > in_max:
                out_of_bounds = True

        # 3. 如果越界，完全基于实际值集合生成结果
        if out_of_bounds:
            if len(sorted_vals) == 1: return sorted_vals[0]
            # 检查整数连续性
            if all(isinstance(v, int) for v in sorted_vals):
                is_cont = True
                for i in range(len(sorted_vals) - 1):
                    if sorted_vals[i + 1] - sorted_vals[i] != 1: is_cont = False; break
                if is_cont: return (sorted_vals[0], sorted_vals[-1])  # 返回元组
            return sorted_vals  # 返回列表

        # 4. 如果未越界，尝试匹配输入范围
        if len(sorted_vals) == 1: return sorted_vals[0]

        # 检查整数连续性
        if all(isinstance(v, int) for v in sorted_vals):
            is_cont = True
            for i in range(len(sorted_vals) - 1):
                if sorted_vals[i + 1] - sorted_vals[i] != 1: is_cont = False; break
            if is_cont: return (sorted_vals[0], sorted_vals[-1])  # 返回元组

        # 检查是否覆盖输入边界
        if in_min is not None and in_max is not None:
            if abs(sorted_vals[0] - in_min) < 1e-9 and abs(sorted_vals[-1] - in_max) < 1e-9:
                return (in_min, in_max)  # 返回元组

        return sorted_vals  # 返回列表

    @staticmethod
    def _check_value_validity(solver, z3_var, val):
        try:
            if z3_var.sort() == z3.IntSort() and isinstance(val, float): val = int(val)
            solver.push()
            solver.add(z3_var == val)
            res = solver.check() == z3.sat
            solver.pop()
            return res
        except:
            return False

    @staticmethod
    def _mine_distinct_values(solver, z3_var, initial_val, limit=5):
        results = []
        if initial_val is not None:
            results.append(initial_val)
        solver.push()
        if results:
            solver.add(z3_var != results[0])

        while len(results) < limit:
            if solver.check() == z3.sat:
                m = solver.model()
                val = m.eval(z3_var)
                py_val = BaseVar._z3_val_to_py(val)
                if py_val is not None:
                    results.append(py_val)
                    solver.add(z3_var != val)
                else:
                    break
            else:
                break
        solver.pop()
        return results

    def infer_range_constraint_by_dtype(self, dtype, range_value):
        """
        基于数据类型的隐式约束 (防止无 range_value 时出现负数长度等风险)
        """
        min_val, max_val, _ = DataMatchMap.DTYPE_SPECS.get(dtype)
        if min_val is None or max_val is None:
            return
        idx = z3.Int('idx')
        val = z3.Select(range_value, idx)
        bounds = idx >= 0
        constraint = z3.And(val >= min_val, val <= max_val)
        logger.info(
            f"[Init] {self.name} inferred range from dtype '{dtype}': [{min_val}, {max_val}]")
        return z3.ForAll([idx], z3.Implies(bounds, constraint))

    @staticmethod
    def _to_z3_const(value, expected_sort):
        """将 Python 字面量或特殊字符串转换为对应的 Z3 常量，适配目标排序"""
        # 处理浮点特殊字符串
        if isinstance(value, str):
            lower_val = value.lower()
            # 检查目标排序是否为浮点数
            if isinstance(expected_sort, z3.FPSortRef):
                if lower_val in ('inf', '+inf'):
                    return z3.fpPlusInfinity(expected_sort)
                elif lower_val == '-inf':
                    return z3.fpMinusInfinity(expected_sort)
                elif lower_val == 'nan':
                    return z3.fpNaN(expected_sort)
            # 普通字符串（非浮点特殊值）
            if lower_val not in ('inf', '+inf', '-inf', 'nan'):
                return z3.StringVal(value)
            else:
                raise TypeError(f"Special value '{value}' only allowed for floating-point arrays")
        # 布尔值
        if isinstance(value, bool):
            return z3.BoolVal(value)
        # 整数
        if isinstance(value, int):
            # 如果期望浮点排序，转为浮点常量
            if isinstance(expected_sort, z3.FPSortRef):
                return z3.FPVal(float(value), expected_sort)
            return z3.IntVal(value)
        # 浮点数
        if isinstance(value, float):
            if isinstance(expected_sort, z3.FPSortRef):
                return z3.FPVal(value, expected_sort)
            return z3.RealVal(value)
        # 其他无法转换的情况
        raise TypeError(f"Cannot convert {value} to Z3 constant with sort {expected_sort}")

    @staticmethod
    def infer_sort_from_value(val):
        """根据 Python 值推断 Z3 Sort"""
        if isinstance(val, bool): return z3.BoolSort()
        if isinstance(val, int): return z3.IntSort()
        if isinstance(val, float): return z3.RealSort()
        if isinstance(val, str): return z3.StringSort()
        return z3.RealSort()

    @staticmethod
    def _infer_element_sort(dtype, range_value, default_sort=z3.RealSort()):
        """
        全局类型推断逻辑 (合并自 TensorVar 和 ListVar)
        根据 dtype 和 range_value 推断 Z3 Sort
        """
        if isinstance(range_value, str) and range_value in ABNORMAL_VALUE_CONFIG:
            logger.info(f"Match range value in abnormal value, range value : {range_value}")
            return ABNORMAL_VALUE_CONFIG.get(range_value)['sort_fn']()
        if dtype and dtype in TYPE_CONFIG:
            return TYPE_CONFIG.get(dtype)['sort_fn']()
        if dtype not in TYPE_CONFIG:
            raise KeyError(f"Unsupported dtype : '{dtype}', not in _infer_sort', can't sort and parse")
        if range_value is not None:
            if isinstance(range_value, (list, tuple)):
                if len(range_value) == 2:
                    min_v, max_v = range_value
                    if isinstance(min_v, float) or isinstance(max_v, float): return z3.RealSort()
                    return z3.IntSort()
                else:
                    raise ValueError(f"range_value list must be [min, max], got {range_value}")
            else:
                return BaseVar.infer_sort_from_value(range_value)
        return default_sort

    @staticmethod
    def _extract_values_from_array_expr(expr):
        logger.info(f"_extract_values_from_array_expr, input expr : {expr}")
        values = set()
        if not z3.is_expr(expr):
            logger.info("expr is not expr")
            return values
        # 情况 1: K (Constant Array) - 所有元素都是同一个值
        # Z3 AST 中，K 只有 1 个参数：默认值。Sort 在声明中。
        if z3.is_K(expr):
            logger.info("expr is K")
            values.add(expr.arg(0))
            logger.info(f"expr is K , value is {values}")
            return values
        # 情况 2: Store - 修改某个索引的值
        # Z3 AST 中，Store 有 3 个参数：arg(0)=原数组, arg(1)=索引, arg(2)=新值
        if z3.is_store(expr):
            logger.info("expr is store")
            stored_val = expr.arg(2)
            values.add(stored_val)
            # 递归处理基础数组
            base_array = expr.arg(0)
            values.update(BaseVar._extract_values_from_array_expr(base_array))
            logger.info(f"expr is store, values : {values}")
            return values
        # 情况 3: As_Array - 由未解释函数表示的数组 (通常无法静态提取)
        if z3.is_as_array(expr):
            logger.warning(f"expr: {expr} is As_Array (uninterpreted function), cannot extract values statically.")
            return values
        # 如果没有上述结构，且是一个常量符号，说明未赋值，无法提取
        if z3.is_const(expr):
            return None
        # 其他未知情况
        logger.warning(f"expr: {expr} is unknown array expression type.")
        return values

    def _add_initial_range_constraints(self, spec):
        # 策略 1: 显式 range_value 约束
        if spec is not None:
            idx = z3.Int('idx')
            val = z3.Select(self.range_value, idx)
            bounds = idx >= 0

            if isinstance(spec, (list, tuple)):
                if len(spec) != 2:
                    raise ValueError(f"range_value list must be [min, max]")
                min_val, max_val = spec
                min_expr = self._to_z3_const(min_val, self.range_value.sort().range())
                max_expr = self._to_z3_const(max_val, self.range_value.sort().range())
                constraint = z3.And(val >= min_expr, val <= max_expr)
                self.solver.add(z3.ForAll([idx], z3.Implies(bounds, constraint)))
                logger.info(f"[Init] {self.name} range constraint: [{min_val}, {max_val}]")
            elif isinstance(spec, (int, float, bool, str)):
                z3_val = self._to_z3_const(spec, self.range_value.sort().range())
                self.solver.add(z3.ForAll([idx], z3.Implies(bounds, val == z3_val)))

        # 策略 2: 基于类型的隐式约束 (防止无 range_value 时出现负数长度等风险)
        elif self._dtype_arg and self._dtype_arg in DataMatchMap.DTYPE_SPECS:
            range_constraint = self.infer_range_constraint_by_dtype(self._dtype_arg, self.range_value)
            self.solver.add(range_constraint)

    @staticmethod
    def _resolve_range_fast(actual_val, input_spec):
        """
        轻量级解析逻辑 (不再调用 solver.check)
        逻辑：
        1. 如果 actual_val 在 input_spec 范围内，返回 input_spec (保留用户建议)。
        2. 如果 actual_val 越界，返回 actual_val (反映真实求解结果)。
        """
        in_min, in_max = BaseVar._parse_input_spec(input_spec)

        # 如果没有输入建议，直接返回实际值
        if in_min is None or in_max is None:
            actual_val = DataHandleUtil.abnormal_float_transfer(actual_val)
            return actual_val

        # 检查 actual_val 是否在 [in_min, in_max] 内
        # 兼容 actual_val 为单值或元组的情况
        if isinstance(actual_val, (tuple, list)):
            v_min, v_max = actual_val[0], actual_val[-1]
        else:
            v_min, v_max = actual_val, actual_val

        if all(isinstance(v, (int, float)) for v in (v_min, v_max, in_min, in_max)) and v_min >= in_min and v_max <= in_max:
            # 情况 A: 实际解在建议范围内 -> 返回建议范围
            input_spec = DataHandleUtil.abnormal_float_transfer(input_spec)
            return input_spec
        else:
            # 情况 B: 实际解越界 -> 返回实际解
            actual_val = DataHandleUtil.abnormal_float_transfer(actual_val)
            return actual_val

    @staticmethod
    def _resolve_range_from_set_fast(values_set, input_spec):
        """
        张量/列表专用：基于提取出的实际值集合决定返回格式 (快速版)。
        不再进行复杂的连续性检查或离散挖掘。
        """
        if not values_set:
            input_spec = DataHandleUtil.abnormal_float_transfer(input_spec)
            return input_spec

        sorted_vals = sorted(list(values_set))
        actual_min, actual_max = sorted_vals[0], sorted_vals[-1]

        # 1. 如果只有一个值，返回单值
        if actual_min == actual_max:
            actual_min = DataHandleUtil.abnormal_float_transfer(actual_min)
            return actual_min

        # 2. 检查是否与输入范围一致
        in_min, in_max = BaseVar._parse_input_spec(input_spec)
        if in_min is not None and in_max is not None:
            # 仅当所有值均为数值类型时才进行算术比较，避免 int - str 报错
            if all(isinstance(v, (int, float)) for v in (actual_min, actual_max, in_min, in_max)):
                if abs(actual_min - in_min) < 1e-9 and abs(actual_max - in_max) < 1e-9:
                    input_spec = DataHandleUtil.abnormal_float_transfer(input_spec)
                    return input_spec

        # 3. 否则返回实际值范围 (元组)
        resolve_range = (actual_min, actual_max)
        resolve_range = DataHandleUtil.abnormal_float_transfer(resolve_range)
        return resolve_range

    @staticmethod
    def _resolve_range_value(actual_min, actual_max, input_spec):
        in_min, in_max = BaseVar._parse_input_spec(input_spec)
        if in_min is not None and in_max is not None and actual_min is not None and actual_max is not None:
            intersect_min = max(in_min, actual_min)
            intersect_max = min(in_max, actual_max)
            if intersect_min <= intersect_max:
                return intersect_min if intersect_min == intersect_max else [intersect_min, intersect_max]
            else:
                return actual_min if actual_min == actual_max else [actual_min, actual_max]
        elif actual_min is not None and actual_max is not None:
            return actual_min if actual_min == actual_max else [actual_min, actual_max]
        return input_spec

    @staticmethod
    def select_from_solve_range(solve_range):
        if isinstance(solve_range, list):
            return random.choice(solve_range)
        elif isinstance(solve_range, tuple):
            return list(solve_range)
        return solve_range


class TensorVar(BaseVar):
    def __init__(self, name, solver, dtype=None, allowed_dtypes=None, allowed_formats=None, range_value=None):
        super().__init__(name, solver)
        self.name = name
        self.type = "tensor"
        self.solver = solver

        # 保存原始 spec 供优化逻辑使用
        self._range_spec = range_value
        self._dtype_arg = dtype

        # 1. 推断元素类型
        self._element_sort = self._infer_element_sort(dtype, range_value, z3.RealSort())

        # 2. 定义 Z3 变量
        self.dtype = z3.Const(f"{name}.dtype", DType)
        self.shape = z3.Const(f"{name}.shape", z3.SeqSort(z3.IntSort()))
        self.format = z3.Const(f"{name}.format", z3.StringSort())
        self.range_value = z3.Array(f"{name}.range_value", z3.IntSort(), self._element_sort)
        self.solver.add(z3.Length(self.shape) >= 0)

        # 3. 添加约束
        self._add_dtype_constraints(dtype, allowed_dtypes)
        self._add_format_constraints(allowed_formats)
        # 不再添加 range_value 约束，仅作为建议保存
        # self._add_initial_range_constraints(range_value)

    def _add_dtype_constraints(self, dtype, allowed_dtypes):
        # A. 定义域约束
        if allowed_dtypes:
            valid_dtypes = [dt for dt in allowed_dtypes if dt in DTYPE_MAP]
            if valid_dtypes:
                self.solver.add(z3.Or([self.dtype == DTYPE_MAP.get(dt) for dt in valid_dtypes]))

        # # B. 初始值约束
        # if dtype:
        #     if dtype in DTYPE_MAP:
        #         self.solver.add(self.dtype == DTYPE_MAP.get(dtype))
        #     else:
        #         raise ValueError(f"Unknown dtype '{dtype}'")

    def _add_format_constraints(self, allowed_formats):
        if allowed_formats:
            self.solver.add(z3.Or([self.format == z3.StringVal(fmt) for fmt in allowed_formats]))

    def get_z3_expr(self):
        return self

    def get_element_sort(self):
        return self._element_sort

    def get_element_at(self, idx):
        return z3.Select(self.range_value, idx)

    def resolve_model(self, model):
        # 解析基础属性
        # 如果变量或属性为无关变量，即约束表达式中不涉及该变量和属性,则不要将该属性或变量添加至结果中
        logger.info("Start analysis tensor resolve result")
        decls = model.decls()
        result = {'type': self.type}
        # 1. 解析 dtype
        if self.dtype.decl() in decls:
            logger.info("Get tensor dtype")
            dtype_val = str(model.eval(self.dtype))
            result["dtype"] = dtype_val

        # 2. 解析 format
        if self.format.decl() in decls:
            logger.info("Get tensor format")
            format_val = model.eval(self.format).as_string()
            result["format"] = format_val

        # 3. 解析 shape
        if self.shape.decl() in decls:
            logger.info("Get Tensor shape")
            shape_len = model.eval(z3.Length(self.shape)).as_long()
            if shape_len > 0:
                try:
                    shape_vals = [model.eval(self.shape[i]).as_long() for i in range(shape_len)]
                    result["shape"] = shape_vals
                except Exception as e:
                    logger.warning(f"Could not resolve shape get actual values for {self.name}: {str(e)}")

        # 4. 提取实际值集合
        if self.range_value.decl() in decls:
            logger.info("Get Tensor range")
            actual_values = set()
            try:
                arr_expr = model.eval(self.range_value)
                actual_values = self._extract_values_from_array_expr(arr_expr)
            except Exception as e:
                logger.warning(f"Could not resolve range_value get actual values for {self.name}: {str(e)}")
            py_vals = set()
            if actual_values is None:
                py_vals = None
            else:
                for v in actual_values:
                    pv = self._z3_val_to_py(v)
                    if pv is not None: py_vals.add(pv)
            # 快速解析 Range,效率高，避免多次调用solver.check()
            resolved_range = BaseVar._resolve_range_from_set_fast(py_vals, self._range_spec)
            # 智能解析
            # resolved_range = BaseVar._resolve_range_from_set(py_vals, self._range_spec)
            result["range_values"] = resolved_range
        return result


class ListVar(BaseVar):
    def __init__(self, name, solver, dtype=None, range_value=None, length=None):
        super().__init__(name, solver)
        self.name = name
        self.type = "list"
        self.solver = solver
        self.dtype_arg = dtype
        self._range_spec = range_value

        self._element_sort = BaseVar._infer_element_sort(dtype, range_value, z3.RealSort())
        self.z3_var = z3.Const(name, z3.SeqSort(self._element_sort))
        self.dtype = z3.Const(f"{name}.dtype", DType)
        self.range_value = z3.Array(f"{name}.range_value", z3.IntSort(), self._element_sort)
        if length is not None:
            if isinstance(length, int):
                self.solver.add(z3.Length(self.z3_var) == length)
            elif isinstance(length, (list, tuple)) and len(length) == 2:
                self.solver.add(z3.Length(self.z3_var) >= length[0])
                self.solver.add(z3.Length(self.z3_var) <= length[1])
        # 不再添加约束，仅作为建议保存
        # self._add_initial_range_constraints(range_value)

    def get_z3_expr(self):
        return self.z3_var

    def get_element_sort(self):
        return self._element_sort

    def get_element_at(self, idx):
        return self.z3_var[idx]

    def resolve_model(self, model):
        # 如果变量或属性为无关变量，即约束表达式中不涉及该变量和属性，则不要将该属性和变量添加至结果中
        decls = model.decls()
        result = {'type': self.type}

        if self.z3_var.decl() not in decls:
            return result
        len_expr = model.eval(z3.Length(self.z3_var))
        seq_len = self._z3_val_to_py(len_expr)
        if seq_len is None:
            logger.warning(f"ListVar '{self.name}' length is not constrained, treating as empty list.")
            # 无法确定长度，返回空列表或根据 input_spec 返回
            return {'type': self.type, 'dtype': self.dtype_arg, 'length': 0, 'range_values': self._range_spec}

        values = []
        for i in range(seq_len):
            elem = model.eval(self.z3_var[i])
            py_v = self._z3_val_to_py(elem)
            values.append(py_v if py_v is not None else str(elem))

        # 1. 提取实际值集合
        py_vals = set(values)
        # 快速解析 Range, 避免多次调用solver.check(),加快效率
        resolved_range = BaseVar._resolve_range_from_set_fast(py_vals, self._range_spec)
        # 智能解析 Range
        # resolved_range = BaseVar._resolve_range_from_set(values, self._range_spec)
        return {'type': self.type, 'dtype': self.dtype_arg, 'length': seq_len, 'range_values': resolved_range}


class ScalarVar(BaseVar):
    def __init__(self, name, solver, dtype, range_value=None):
        super().__init__(name, solver)
        if dtype not in TYPE_CONFIG:
            logger.error(f"Unsupported dtype '{dtype}' for var '{name}', fallback to 'int'")
            dtype = 'int'
        self.type = "scalar"
        self.dtype_arg = dtype
        self.solver = solver
        self.config = TYPE_CONFIG.get(dtype)
        self.dtype = z3.Const(f"{name}.dtype", DType)
        self.z3_var = z3.Const(name, self.config['sort_fn']())
        self._range_spec = range_value
        self._element_sort = self.config['sort_fn']()

    def get_z3_expr(self):
        return self.z3_var

    def resolve_model(self, model):
        result = {'type': self.type}
        decls = model.decls()
        if self.z3_var.decl() not in decls:
            return result

        val = model.eval(self.z3_var)
        py_val = self.config['parse_fn'](val)
        if py_val is None:
            logger.warning(
                f"Failed to parse Z3 value : '{val}' for var '{self.name}', dtype '{self.dtype}', input range value : '{self._range_spec}'")
            py_val = self._range_spec
        # 使用快速解析逻辑，避免多次调用solver.check()
        resolved_range = BaseVar._resolve_range_fast(py_val, self._range_spec)
        # 智能解析range
        # resolved_range = self._resolve_range_smartly(
        #     self.solver,
        #     self.z3_var,
        #     py_val,
        #     self._range_spec
        # )
        return {'type': self.type, 'dtype': self.dtype_arg, 'range_values': resolved_range}
