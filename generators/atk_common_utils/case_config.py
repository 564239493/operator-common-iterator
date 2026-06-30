# Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
import hashlib
import json
from functools import reduce
from typing import Optional, Union, List

from pydantic import BaseModel

from generators.atk_common_utils.design_config import StandardConfig, MAX_NUMBER_OF_ELEMENTS
from generators.atk_common_utils.logger_utils_atk import Logger

logging = Logger()


class MyBaseModel(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def get_id(self):
        json_str = json.dumps(self.model_dump())
        hash_obj = hashlib.sha1(json_str.encode())
        return int(hash_obj.hexdigest(), 16)


class InputCaseConfig(MyBaseModel):
    name: Optional[str]
    type: Optional[str]
    required: bool = True
    dtype: str
    shape: Optional[List] = None
    range_values: object
    # 对于ArrayList类型的参数，length用来描述List的长度信息
    length: int | None = None
    format: str | None = None
    backward: Optional[bool] = False
    align_32B: Optional[bool] = None
    outlier_values: Optional[List[float]] = None

    def __hash__(self):
        dtype_hash = hash(self.dtype)
        shape_hash = hash(tuple(self.shape)) if self.shape else 0
        if self.range_values and isinstance(self.range_values, list):
            range_values_hash = hash(tuple(self.range_values))
        elif self.range_values and not isinstance(self.range_values, dict):
            range_values_hash = hash(self.range_values)
        else:
            range_values_hash = 0
        return dtype_hash + shape_hash + range_values_hash

    def is_range_null(self):
        if isinstance(self.range_values, list):
            return "null" in self.range_values
        else:
            return self.range_values == "null"

    def numel(self):
        if not self.shape:
            return 1
        if self.is_range_null():
            return 0
        return reduce(lambda x, y: x * y, self.shape)


class CaseConfig(MyBaseModel):
    id: Optional[int] = 0
    name: Optional[str] = None
    aclnn_name: Optional[str] = None
    triton_name: Optional[str] = None
    version: Optional[str] = None
    expected_error_msg: Optional[str] = None
    api: Optional[str] = "pytorch"
    api_type: Optional[str] = "function"
    aclnn_api_type: Optional[str] = "aclnn_function"
    triton_api_type: Optional[str] = "triton_function"
    fusion_api_type: Optional[str] = "fusion_function"
    fusion_mode: Optional[str] = None
    dist_api_type: Optional[str] = "dist_function"
    backward: bool = False
    standard: Optional[StandardConfig] = StandardConfig()
    outputs: Optional[Union[int, str]] = None
    inputs: List[Union[List[InputCaseConfig], InputCaseConfig]] = None
    acl_json: Optional[str] = ""
    method_inputs: Optional[List[Union[List[InputCaseConfig], InputCaseConfig]]] = None
    tensor_input: Optional[InputCaseConfig] = None
    save_name: Optional[str] = None
    uuid: Optional[str] = None
    downloaded: Optional[bool] = False
    is_boundary: Optional[bool] = False
    xrun_cs_name: Optional[str] = None
    xrun_data: Optional[dict] = None

    def __eq__(self, other):
        all_input = []
        other_all_input = []

        inputs_match = (self.inputs is None) == (other.inputs is None) or len(self.inputs) == len(other.inputs)
        method_inputs_match = ((self.method_inputs is None) == (other.method_inputs is None)
                               or len(self.method_inputs) == len(other.method_inputs))
        tensor_input_match = (self.tensor_input is None) == (other.tensor_input is None)
        if not (inputs_match and method_inputs_match and tensor_input_match):
            logging.error(f"the number of parameters in case {self.name} and {other.name} is inconsistent", )
            return False
        if self.inputs is not None:
            all_input.extend(self.flatten_list(self.inputs))
            other_all_input.extend(self.flatten_list(other.inputs))
        if self.method_inputs is not None:
            all_input.extend(self.flatten_list(self.method_inputs))
            other_all_input.extend(self.flatten_list(other.method_inputs))
        if self.tensor_input is not None:
            all_input.append(self.tensor_input)
            other_all_input.append(other.tensor_input)
        for cur_input, other_input in zip(all_input, other_all_input):
            if (
                    cur_input.dtype != other_input.dtype
                    or cur_input.shape != other_input.shape
                    or cur_input.range_values != other_input.range_values
            ):
                return False
        return True

    def __hash__(self):
        all_input = []
        if self.inputs is not None:
            all_input.extend(self.flatten_list(self.inputs))
        if self.method_inputs is not None:
            all_input.extend(self.flatten_list(self.method_inputs))
        if self.tensor_input is not None:
            all_input.append(self.tensor_input)
        return sum(hash(case_input) for case_input in all_input)

    @property
    def opp_perf(self):
        return self.standard.perf

    @staticmethod
    def flatten_list(old_list):
        new_list = []
        for item in old_list:
            if isinstance(item, list):
                new_list.extend(CaseConfig.flatten_list(item))
            else:
                new_list.append(item)
        return new_list

    def is_backward(self):
        for input_info in self.inputs:
            if isinstance(input_info, list) and input_info[0].backward:
                return True
            elif not isinstance(input_info, list) and input_info.backward:
                return True
        return False

    def get_input_data_config(self, index=None, name=None):
        if name is None:
            input_index = 0
            for input_case in self.inputs:
                if isinstance(input_case, (list, tuple)):
                    input_case = input_case[0]

                if not input_case.name:
                    input_index += 1

                if input_index == index + 1:
                    return input_case
        else:
            for input_case in self.inputs:
                if isinstance(input_case, (list, tuple)):
                    input_case = input_case[0]

                if input_case.name == name:
                    return input_case
        return None


class CasesMetrics(BaseModel):
    oversize_case: bool = False
    shape_intervals: List[List[Union[int, float]]] = []
    case_repetition_rate: float = 0.0
    distribution_threshold: float = 0.03
    repetition_threshold: float = 0.05
    result: bool = False

    def get_cases_metrics(self, cases, design_config):
        cases_gens = []
        for i, case in enumerate(cases):
            case.case.id = i
            cases_gens.append(case.case)
            if not self.oversize_case and self.judgment_tensor_size(case):
                self.oversize_case = True
            self.calculate_cases_shape_intervals(case)
        for shape_interval in self.shape_intervals:
            shape_interval[1] /= len(cases)
        self.case_repetition_rate = (len(cases) - len(set(cases_gens))) / len(cases)
        self.get_metrics_result(design_config)

    def judgment_tensor_size(self, case):
        dtype = case.dtype
        max_length = MAX_NUMBER_OF_ELEMENTS
        if dtype in ["int16", "fp16"]:
            max_length = int(max_length / 2)
        elif dtype in ["fp32", "int32"]:
            max_length = int(max_length / 4)
        elif dtype in ["fp64", "int64", "complex64"]:
            max_length = int(max_length / 8)
        return case.numel >= max_length

    def calculate_cases_shape_intervals(self, case):
        for interval in self.shape_intervals:
            if case.numel >= interval[0]:
                interval[1] += 1

    def get_metrics_result(self, design_config):
        if self.oversize_case:
            return
        for reality, standard in zip(self.shape_intervals, design_config.shape_distributions):
            if abs(reality[1] - standard[1]) > self.distribution_threshold:
                return
        if self.case_repetition_rate > self.repetition_threshold:
            return
        self.result = True

    def model_dump(self):
        dumped_data = super().model_dump()
        exclude_attributes = ['distribution_threshold', "repetition_threshold"]
        for attr in exclude_attributes:
            dumped_data.pop(attr, None)
        return dumped_data
