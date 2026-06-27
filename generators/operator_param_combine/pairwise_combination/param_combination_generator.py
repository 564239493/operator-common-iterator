from __future__ import annotations

import random
from typing import Any, Dict, List

from agent.generators.common_utils.logger_util import LazyLogger
from agent.generators.data_definition.constants import DataMatchMap, ParamModelConfig
from agent.generators.data_definition.param_models_def import (
    OperatorParameterCombination,
    ParameterPropertyData,
    ParameterShapeProperty,
)
from agent.generators.operator_param_combine.pairwise_combination.attribute_domain import (
    AttributeDomain, ATTR_DTYPE, ATTR_FORMAT, ATTR_DIMENSIONS,
    ATTR_RANGE_VALUE, ATTR_ARRAY_LENGTH, ATTR_IS_OPTIONAL, ATTR_IS_OPERATOR_PARAM,
)
from agent.generators.operator_param_combine.pairwise_combination.constraint_filter import ConstraintProcessor
from agent.generators.operator_param_combine.pairwise_combination.pairwise_generator import PairwiseCombinationGenerator
from agent.generators.common_model_definition import OperatorRule

logger = LazyLogger()


class PairwiseParamCombinationGenerator:
    def __init__(self, operator_rule_data: OperatorRule, case_num: int = 1):
        self.operator_rule_data = operator_rule_data
        self.case_num = case_num
        self.generated_combinations: List[OperatorParameterCombination] | None = None

    def get_param_combination_input(self) -> List[OperatorParameterCombination] | None:
        if self.operator_rule_data is None:
            logger.error("Get param combination failed, input operator constraint data is None")
            return None

        logger.info(
            f"Start pairwise parameter combination generation, "
            f"operator name: '{self.operator_rule_data.operator_name}'"
        )

        attr_domain = AttributeDomain(self.operator_rule_data)
        constraint_processor = ConstraintProcessor(self.operator_rule_data)
        pairwise_gen = PairwiseCombinationGenerator(attr_domain, constraint_processor)
        raw_combinations = pairwise_gen.generate()

        if not raw_combinations:
            logger.error("Pairwise generation returned no combinations")
            return None

        result = []
        for raw_case in raw_combinations:
            operator_param_comb = OperatorParameterCombination(
                operator_name=self.operator_rule_data.operator_name
            )
            for input_name in attr_domain.get_effective_params():
                param_attr = self.operator_rule_data.inputs.get(input_name)
                if param_attr is None:
                    param_attr = self.operator_rule_data.outputs.get(input_name)
                if param_attr is None:
                    continue

                param_type_ori = self._get_attr_value(input_name, param_attr.type, "type")
                param_type = DataMatchMap.ACL_TYPE_TRANSFER_ATK_MAP.get(
                    param_type_ori, ParamModelConfig.DEFAULT_ATK_TYPE
                )
                param_format = self._get_format_value(raw_case, input_name)
                param_dtype = self._get_dtype_value(raw_case, input_name, param_attr)
                param_length = self._get_length_value(raw_case, input_name, param_attr, param_type)
                param_range = self._get_range_value(raw_case, input_name, param_attr, param_dtype)
                param_is_optional = self._get_bool_attr(
                    input_name, param_attr.is_optional, "is_optional"
                )
                is_operator_param = self._get_bool_attr(
                    input_name, param_attr.is_operator_param, "is_operator_param"
                )

                prop = ParameterPropertyData(
                    param_name=input_name,
                    dtype=param_dtype,
                    param_type=param_type,
                    format=param_format,
                    range_value_profile=param_range,
                    length=param_length,
                    is_operator_param=is_operator_param,
                    is_optional=param_is_optional,
                )

                if param_type in ParamModelConfig.TENSOR_ATK_TYPE:
                    dim_count, dim_profile = self._get_shape_property(raw_case, input_name, param_attr)
                    prop.shape_property = ParameterShapeProperty(
                        dim_count=dim_count, dim_value_profile=dim_profile
                    )
                    mapped = DataMatchMap.PARAM_VALUE_TO_ROLE_MODEL.get(param_range, param_range)
                    prop.range_value_profile = mapped

                operator_param_comb.parameter_property.append(prop)

            result.append(operator_param_comb)

        while len(result) < self.case_num:
            result.append(result[len(result) % len(raw_combinations)])

        result = result[:self.case_num] if self.case_num < len(result) else result
        self.generated_combinations = result

        logger.info(
            f"End pairwise parameter combination generation, "
            f"operator name: '{self.operator_rule_data.operator_name}', "
            f"generated {len(result)} combinations"
        )
        return result

    def _get_attr_value(self, param_name: str, attr, attr_name: str):
        from common_utils.data_handle_utils import DataHandleUtil
        raw, _ = DataHandleUtil.get_relevant_attribute_value(param_name, attr, attr_name)
        return raw

    def _get_format_value(self, raw_case: Dict[str, Dict[str, Any]], param_name: str) -> str | None:
        pf = raw_case.get(param_name, {}).get(ATTR_FORMAT)
        if pf is not None:
            return str(pf) if not isinstance(pf, str) else pf
        return None

    def _get_dtype_value(self, raw_case: Dict[str, Dict[str, Any]],
                          param_name: str, param_attr) -> str:
        from common_utils.data_handle_utils import DataHandleUtil
        dtype_vals = raw_case.get(param_name, {}).get(ATTR_DTYPE)
        if dtype_vals is not None:
            return str(dtype_vals)

        dtype_set, _ = DataHandleUtil.get_relevant_attribute_value(
            param_name, param_attr.dtype, "dtype"
        )
        if dtype_set:
            return random.choice(dtype_set)
        return ParamModelConfig.DEFAULT_PARAM_DTYPE_DTYPE_IN_ORIGINAL_DOC

    def _get_length_value(self, raw_case: Dict[str, Dict[str, Any]],
                           param_name: str, param_attr, param_type: str) -> int | None:
        from common_utils.data_handle_utils import DataHandleUtil
        if param_type not in ParamModelConfig.LIST_ATK_TYPE:
            return None
        lv = raw_case.get(param_name, {}).get(ATTR_ARRAY_LENGTH)
        if lv is not None:
            return int(lv)
        length_value, _ = DataHandleUtil.get_relevant_attribute_value(
            param_name, param_attr.array_length, "array_length"
        )
        if length_value is None:
            return ParamModelConfig.DEFAULT_LIST_LENGTH
        return random.choice(length_value) if isinstance(length_value, list) else length_value

    def _get_range_value(self, raw_case: Dict[str, Dict[str, Any]],
                          param_name: str, param_attr, dtype: str) -> str | int | float | bool | None:
        from operator_param_combine.param_combination_generate import ParamCombinationGenerator
        default_profile = random.choice(ParamCombinationGenerator.get_default_range_by_dtype(dtype))
        rv = raw_case.get(param_name, {}).get("range_value_profile")
        if rv is not None:
            return rv
        return default_profile

    def _get_bool_attr(self, param_name: str, attr, attr_name: str) -> bool:
        from common_utils.data_handle_utils import DataHandleUtil
        raw, _ = DataHandleUtil.get_relevant_attribute_value(param_name, attr, attr_name)
        if raw is None:
            return True
        return bool(raw)

    def _get_shape_property(self, raw_case: Dict[str, Dict[str, Any]],
                             param_name: str, param_attr) -> tuple:
        dim_count = raw_case.get(param_name, {}).get("dim_count")
        if dim_count is None:
            from common_utils.data_handle_utils import DataHandleUtil
            dim_value, _ = DataHandleUtil.get_relevant_attribute_value(
                param_name, param_attr.dimensions, "dimensions"
            )
            if dim_value is None:
                dim_count = ParamModelConfig.DEFAULT_TENSOR_SHAPE_DIM
            else:
                dim_count = random.choice(dim_value) if isinstance(dim_value, list) else dim_value
        dim_count = int(dim_count) if dim_count else ParamModelConfig.DEFAULT_TENSOR_SHAPE_DIM

        dim_profile = raw_case.get(param_name, {}).get("dim_value_profile")
        if dim_profile is None:
            dim_profile = random.choice(ParamModelConfig.DIM_VALUE_PROFILE_LIST)

        return dim_count, str(dim_profile)
