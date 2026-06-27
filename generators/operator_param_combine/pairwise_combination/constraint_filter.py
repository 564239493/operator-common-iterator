from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Dict, List, Set, Tuple

from agent.generators.common_utils.logger_util import LazyLogger
from agent.generators.data_definition.constants import ParamModelConfig
from agent.generators.common_model_definition import OperatorRule, InterParamConstraint, InterConstraintsRuleType

logger = LazyLogger()


class ConstraintProcessor:
    def __init__(self, operator_rule: OperatorRule):
        self.operator_rule = operator_rule
        raw = operator_rule.constraints_in_parameters
        if isinstance(raw, list):
            self.constraints: List[InterParamConstraint] = raw
        elif isinstance(raw, dict):
            flat = []
            for vals in raw.values():
                if isinstance(vals, list):
                    flat.extend(vals)
            self.constraints = flat
        else:
            self.constraints = []
        self._processed: bool = False

        self.type_equal_groups: List[Set[str]] = []
        self.shape_equal_groups: List[Set[str]] = []
        self.format_equal_groups: List[Set[str]] = []
        self.presence_deps: List[Tuple[str, str, bool]] = []
        self.fixed_values: Dict[str, Any] = {}
        self.value_deps: List[Dict] = []

    def process(self):
        if self._processed:
            return
        for c in self.constraints:
            expr_type = c.expr_type
            params = c.relation_params
            expr = c.expr
            if expr_type == InterConstraintsRuleType.TYPE_EQUALITY.value:
                self.type_equal_groups.append(set(params))
            elif expr_type == InterConstraintsRuleType.SHAPE_EQUALITY.value:
                self.shape_equal_groups.append(set(params))
            elif expr_type == InterConstraintsRuleType.FORMAT_EQUALITY.value:
                self.format_equal_groups.append(set(params))
            elif expr_type == InterConstraintsRuleType.PRESENCE_DEPENDENCY.value:
                self._process_presence_dep(params, expr)
            elif expr_type == InterConstraintsRuleType.VALUE_DEPENDENCY.value:
                self._process_value_dep(params, expr)
            elif expr_type == InterConstraintsRuleType.TYPE_DEPENDENCY.value:
                self._process_type_dep(params, expr)
        self._merge_overlapping_groups()
        self._processed = True

    def _process_presence_dep(self, params: List[str], expr: str):
        if len(params) >= 2:
            for i in range(1, len(params)):
                self.presence_deps.append((params[0], params[i], True))

    def _process_value_dep(self, params: List[str], expr: str):
        self.value_deps.append({"params": params, "expr": expr})

    def _process_type_dep(self, params: List[str], expr: str):
        pass

    def _merge_overlapping_groups(self):
        def merge(groups: List[Set[str]]) -> List[Set[str]]:
            merged = True
            while merged:
                merged = False
                new_groups = []
                for g in groups:
                    for ng in new_groups:
                        if g & ng:
                            ng |= g
                            merged = True
                            break
                    else:
                        new_groups.append(g)
                groups = new_groups
            return groups

        self.type_equal_groups = merge(self.type_equal_groups)
        self.shape_equal_groups = merge(self.shape_equal_groups)
        self.format_equal_groups = merge(self.format_equal_groups)

    def apply_type_equality(self, dtype_domains: Dict[str, List[str]]) -> Dict[str, List[str]]:
        self.process()
        result = dict(dtype_domains)
        for group in self.type_equal_groups:
            valid_params = [p for p in group if p in result]
            if len(valid_params) < 2:
                continue
            common = self._intersect_domains([result[p] for p in valid_params])
            if not common:
                logger.warning(f"Type equality group {valid_params} has no common dtype")
                continue
            for p in valid_params:
                result[p] = list(common)
        return result

    def apply_format_equality(self, format_domains: Dict[str, List[str]]) -> Dict[str, List[str]]:
        self.process()
        result = dict(format_domains)
        for group in self.format_equal_groups:
            valid_params = [p for p in group if p in result and result[p]]
            if len(valid_params) < 2:
                continue
            common = self._intersect_domains([result[p] for p in valid_params])
            if not common:
                continue
            for p in valid_params:
                result[p] = list(common)
        return result

    def apply_shape_equality(self, dim_domains: Dict[str, List[int]],
                             profile_domains: Dict[str, List[str]]) -> Tuple[Dict[str, List[int]], Dict[str, List[str]]]:
        self.process()
        dims = dict(dim_domains)
        profiles = dict(profile_domains)
        for group in self.shape_equal_groups:
            valid_params = [p for p in group if p in dims]
            if len(valid_params) < 2:
                continue
            common_dims = self._intersect_domains([dims[p] for p in valid_params])
            if common_dims:
                for p in valid_params:
                    dims[p] = list(common_dims)
            common_profiles = self._intersect_domains([profiles.get(p, []) for p in valid_params])
            if common_profiles:
                for p in valid_params:
                    profiles[p] = list(common_profiles)
        return dims, profiles

    @staticmethod
    def _intersect_domains(domain_list: List[List]) -> Set:
        if not domain_list:
            return set()
        result = set(domain_list[0])
        for d in domain_list[1:]:
            result &= set(d)
        return result

    def is_valid_combination(self, assignment: Dict[str, Dict[str, Any]]) -> bool:
        self.process()
        for dep in self.presence_deps:
            master, slave, _ = dep
            if master in assignment and slave not in assignment:
                return False
        for fv_name, fv_value in self.fixed_values.items():
            if fv_name in assignment:
                range_val = assignment[fv_name].get("range_value_profile")
                if range_val is not None and range_val != fv_value:
                    return False
        for vd in self.value_deps:
            if not self._check_value_dep(vd, assignment):
                return False
        return True

    def _check_value_dep(self, vd: Dict, assignment: Dict) -> bool:
        params = vd["params"]
        if not all(p in assignment for p in params):
            return True
        expr = vd["expr"]
        try:
            safe_expr = expr
            for p in params:
                info = assignment.get(p, {})
                for attr_name in ["dtype", "format"]:
                    placeholder = f"{{{p}.{attr_name}}}"
                    val = info.get(attr_name)
                    if val is not None:
                        safe_expr = safe_expr.replace(placeholder, str(val))
            return True
        except Exception:
            return True

    def get_dtype_common_map(self, dtype_combinations: List[Dict[str, str]] | None) -> Dict[str, str] | None:
        if dtype_combinations is None:
            return None
        return None
