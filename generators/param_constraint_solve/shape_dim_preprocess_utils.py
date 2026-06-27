# -*- coding: UTF-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""
版权信息：华为技术有限公司，版本所有(C) 2026-2026
修改记录：2026/4/28 21:07
功能：解析结构化数据中对于shape dim的定义，获取shape dim的上下界，将其转换为可以处理的数组，如1<=len(x1.shape)<=8转换为[1,8]
"""
import ast
from typing import List, Dict

from agent.generators.common_utils.logger_util import LazyLogger

logger = LazyLogger()

class ShapeDimValueExtractor:
    """
    从字符串表达式中提取 shape 维度的上下限。
    严格限制：仅处理 len(xxx.shape) 形式的表达式。
    """
    # 内部标记，用于区分“解析失败”和“解析结果为None”
    _PARSE_ERROR = object()
    _NO_CONSTRAINT = object()  # 标记 is not None (无约束)

    def extract(self, expr_str) -> List[Dict] | None:
        """
        入口方法：解析字符串并返回结果。

        Args:
            expr_str (str): 输入的约束字符串，如 'len(x.shape) <= 5'

        Returns:
            dict or None: 包含 min/max 的字典，如果格式不符则返回 None。
        """
        if not isinstance(expr_str, str):
            logger.error("Shape dim expr must be string")
            return None

        try:
            # 解析字符串为 AST 树
            tree = ast.parse(expr_str.strip(), mode='eval')
        except SyntaxError as e:
            logger.error(f"Parse expr by ast failed, err msg : {str(e)}")
            return None

        node = tree.body
        # 1. 解析节点
        constraint_list = self._dispatch_node(node)

        if constraint_list is self._PARSE_ERROR:
            return None

            # 2. 应用默认最小值规则
        return self._apply_default_min(constraint_list)

    @staticmethod
    def _apply_default_min(constraint_list):
        """
        后处理：补全默认 min=0
        :param constraint_list: 如果有复合表达式，拆解为多个，放入列表中
        """
        final_list = []
        for item in constraint_list:
            if item is None:
                continue

            if isinstance(item, dict):
                if item.get('min') is None:
                    item['min'] = 0
                elif item['min'] < 0:
                    item['min'] = 0
                final_list.append(item)

        return final_list if final_list else None

    def _dispatch_node(self, node) -> List[Dict] | None | object:
        """分发逻辑，统一返回列表"""
        # 处理 or 连接
        if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.And):
            return self._process_and(node)

            # 处理 or 连接 (并集)
        if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.Or):
            return self._process_or(node)

            # 处理单个比较
        if isinstance(node, ast.Compare):
            return self._process_compare(node)

        return self._PARSE_ERROR

    def _process_or(self, node) -> List[Dict] | object:
        """处理 Or：收集所有结果"""
        collected_list = []
        for value_node in node.values:
            sub_res = self._dispatch_node(value_node)
            if sub_res is self._PARSE_ERROR:
                return self._PARSE_ERROR

            # 如果分支是 is not None (无约束)，则整个 or 表达式实际上无约束（或忽略）
            # 但在 shape 上下文中，通常 is not None 不产生数值约束
            if sub_res is self._NO_CONSTRAINT:
                continue

            if isinstance(sub_res, list):
                collected_list.extend(sub_res)
            else:
                collected_list.append(sub_res)

        return collected_list if collected_list else self._PARSE_ERROR

    def _process_and(self, node) -> List[Dict] | object:
        """处理 And：计算交集"""
        # 假设初始区间为全集
        current_ranges = [{'min': None, 'max': None}]

        for value_node in node.values:
            sub_res = self._dispatch_node(value_node)

            if sub_res is self._PARSE_ERROR:
                return self._PARSE_ERROR

            if sub_res is self._NO_CONSTRAINT:
                continue

            # 处理 is None 的情况
            if None in sub_res:
                return [None]

            # sub_res 现在必然是 List[Dict]
            current_ranges = self._intersect_lists(current_ranges, sub_res)
            if not current_ranges:
                return self._PARSE_ERROR

        return current_ranges

    def _intersect_lists(self, list_a, list_b):
        """计算两个约束列表的交集"""
        result = []
        for a in list_a:
            for b in list_b:
                inter = self._intersect_intervals(a, b)
                if inter:
                    result.append(inter)
        return result

    @staticmethod
    def _intersect_intervals(a, b):
        """计算两个区间的交集"""
        # 计算下界：取最大值
        # None 表示负无穷
        if a['min'] is None:
            new_min = b['min']
        elif b['min'] is None:
            new_min = a['min']
        else:
            new_min = max(a['min'], b['min'])

        # 计算上界：取最小值
        # None 表示正无穷
        if a['max'] is None:
            new_max = b['max']
        elif b['max'] is None:
            new_max = a['max']
        else:
            new_max = min(a['max'], b['max'])

        # 检查区间有效性
        # 如果 new_min 或 new_max 为 None，说明有一边无界，有效
        if new_min is None or new_max is None:
            return {'min': new_min, 'max': new_max}

        if new_min <= new_max:
            return {'min': new_min, 'max': new_max}

        return None  # 交集为空

    @staticmethod
    def _is_valid_shape_len_expr(node):
        """
        校验节点是否为 len(xxx.shape)
        """
        # 1. 必须是函数调用
        if not isinstance(node, ast.Call):
            logger.error("Shape dim expr must be function call")
            return False

        # 2. 函数名必须是 'len'
        if not (isinstance(node.func, ast.Name) and node.func.id == 'len'):
            logger.error("Shape dmi expr must be function call of 'len'")
            return False

        # 3. 参数必须只有一个
        if len(node.args) != 1:
            logger.error("Shape dim expr must contain only one parameter")
            return False

        arg = node.args[0]

        # 4. 参数必须是属性访问
        if isinstance(arg, ast.Attribute):
            # 5. 属性名必须是 'shape'
            if arg.attr == 'shape':
                return True
            logger.error("Shape dim expr's parameter attribute must be 'shape'")

        return False

    @staticmethod
    def _get_num(node):
        """
        从 AST 节点获取数字常量
        """
        if isinstance(node, ast.Constant):  # Python 3.8+
            return node.value
        return None

    @staticmethod
    def _check_is_none(node):
        """检查 node 是否为 'x is None'"""
        if len(node.ops) == 1 and isinstance(node.ops[0], ast.Is):
            right = node.comparators[0]
            if isinstance(right, ast.Constant) and right.value is None:
                return True
        return False

    @staticmethod
    def _check_is_not_none(node):
        """检查 node 是否为 'x is not None'"""
        if len(node.ops) == 1 and isinstance(node.ops[0], ast.IsNot):
            right = node.comparators[0]
            if isinstance(right, ast.Constant) and right.value is None:
                return True
        return False

    def _process_compare(self, node) -> Dict | None | object:
        """
        处理比较操作节点，提取上下限
        :param node: ast解析树中的节点
        """
        # 1. 检查 is None -> 返回包含 None 的列表
        if self._check_is_none(node):
            return [None]

            # 2. 检查 is not None -> 返回哨兵
        if self._check_is_not_none(node):
            return self._NO_CONSTRAINT

        # 3. 检查 len(shape) 比较
        results = {'min': None, 'max': None}

        # 链式比较
        if len(node.ops) == 2:
            left_val = self._get_num(node.left)
            mid_var = node.comparators[0]
            right_val = self._get_num(node.comparators[1])

            if self._is_valid_shape_len_expr(mid_var) and left_val is not None and right_val is not None:
                op1 = type(node.ops[0])
                op2 = type(node.ops[1])

                if op1 == ast.LtE and op2 == ast.LtE:
                    results['min'] = left_val
                    results['max'] = right_val
                    return [results]  # 包装成列表返回
                elif op1 == ast.GtE and op2 == ast.GtE:
                    results['max'] = left_val
                    results['min'] = right_val
                    return [results]  # 包装成列表返回

        # 单个比较
        if len(node.ops) == 1:
            op_type = type(node.ops[0])
            left = node.left
            right = node.comparators[0]

            if self._is_valid_shape_len_expr(left):
                val = self._get_num(right)
                if val is None: return self._PARSE_ERROR
                if op_type == ast.LtE:
                    results['max'] = val
                elif op_type == ast.Lt:
                    results['max'] = val - 1
                elif op_type == ast.GtE:
                    results['min'] = val
                elif op_type == ast.Gt:
                    results['min'] = val + 1
                elif op_type == ast.Eq:
                    results['min'] = results['max'] = val
                return [results]  # 包装成列表返回

            elif self._is_valid_shape_len_expr(right):
                val = self._get_num(left)
                if val is None: return self._PARSE_ERROR
                if op_type == ast.GtE:
                    results['max'] = val
                elif op_type == ast.Gt:
                    results['max'] = val - 1
                elif op_type == ast.LtE:
                    results['min'] = val
                elif op_type == ast.Lt:
                    results['min'] = val + 1
                elif op_type == ast.Eq:
                    results['min'] = results['max'] = val
                return [results]  # 包装成列表返回

        return self._PARSE_ERROR
