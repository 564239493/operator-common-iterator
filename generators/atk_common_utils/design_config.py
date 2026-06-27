# Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
import random
from enum import Enum
from typing import Optional, List
from typing import Union
from typing_extensions import Literal

import yaml
from pydantic import BaseModel, ConfigDict, model_validator, PrivateAttr

from agent.generators.atk_common_utils.enums import SingBenchmarkType

DEFAULT_TYPES = ["tensor", "tensors", "tensor_tuple",
                 "scalar", "scalars", "scalar_tuple",
                 "attr", "attrs", "attr_tuple"]

DEFAULT_DTYPES = [
    "fp32",
    "bf16",
    "fp16",
    "fp64",
    "hf32",
    "int8",
    "uint8",
    "int16",
    'uint16',
    "int32",
    "uint32",
    "int64",
    "uint64",
    "bool",
]
ATTR_DTYPES = ["float", "int", "string", "attr_bool"]
NEW_DTYPES = ["complex128", "complex64", "complex32"]

DEFAULT_DIM_NUMBERS = [1, 2, 3, 4, 5, 6, 7, 8]
DEFAULT_DIM_WEIGHTS = [0.05, 0.2, 0.2, 0.2, 0.2, 0.01, 0.01, 0.01]

DEFAULT_DIM_VALUES = [
    1,
    [7, 9],
    [15, 17],
    [19, 21],
    [255, 257],
    131073,
    2147483648,
    [1, 1024],
    [1025, 10240],
    [10241, 102400],
    [102401, 1024000],
    [1024001, 10240000],
    [10240001, 102400000],
    [102400001, 1024000000],
    [1024000001, 2147483648],
]
DEFAULT_DIM_VALUES_WEIGHTS = [
    0.05,
    0.1,
    0.1,
    0.1,
    0.05,
    0.05,
    0.001,
    0.001,
    0.001,
    0.001,
    0.001,
    0.001,
    0.001,
    0.001,
    0.001,
]

DEFAULT_RANGES_VALUES = [[-1, 1], [-7, 7], ["-inf", "inf"]]

DEFAULT_TENSOR_RANGES_INVALID_VALUES = [["-inf"], ["inf"], ["nan"], ["null"]]
DEFAULT_SCALAR_RANGES_INVALID_VALUES = [["-inf"], ["inf"], ["nan"]]
DEFAULT_ATTR_RANGES_INVALID_VALUES = [["-inf"], ["inf"]]

VALID_WEIGHTS = 0.95
MAX_NUMBER_OF_ELEMENTS = 2 ** 32 - 1
DEFAULT_DTYPE_NUMBERS = 700


class RandomTypes(Enum):
    DEFAULT = "default"  # 均匀分布
    ND = "nd"  # 正态分布


class RandomTypesConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')
    name: Optional[RandomTypes] = RandomTypes.DEFAULT
    mean: Optional[List[float]] = [-100, 100]
    std: Optional[List[float]] = [1, 25]

    @model_validator(mode="after")
    def check(self):
        if self.name == RandomTypes.ND:
            if not self.mean:
                raise ValueError("not set nd mean, please check!")
            if not self.std:
                raise ValueError(" not set nd std, please check!")
        return self

    def model_dump(self):
        data = super(RandomTypesConfig, self).model_dump()
        if data.get("name"):
            data["name"] = data["name"].value
        return data


class RandomConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')
    type: Optional[Literal["choices"]] = "choices"
    values: List = None
    weights: Optional[List[float]] = None
    invalid_values: List[Union[str, int]] = []
    random_types: Optional[List[RandomTypesConfig]] = None

    @model_validator(mode="after")
    def weight_len_must_same_values(self):
        if self.weights and len(self.weights) != len(self.values):
            raise ValueError("length of weights and values is not same.")
        if self.random_types and len(self.random_types) != len(self.values):
            raise ValueError("length of random_types and values is not same.")
        return self

    def get_values(self, k=1, clc_interval=True):
        fun = getattr(self, self.type)
        if k == 1:
            return fun(k, clc_interval)[0]
        else:
            return fun(k, clc_interval)

    def choices(self, k, clc_interval):
        ret = []
        for _ in range(k):
            index = random.choices(range(len(self.values)), weights=self.weights, k=k)[0]
            if self.random_types and self.random_types[index].name != RandomTypes.DEFAULT:
                ret.append(self.random_types[index].model_dump())
                continue
            value = self.values[index]
            double_int = (
                    clc_interval
                    and isinstance(value, list)
                    and len(value) == 2
                    and isinstance(value[0], int)
                    and isinstance(value[1], int)
            )
            if double_int:
                ret.append(random.randint(value[0], value[1]))
            else:
                ret.append(value)
        return ret

    def get_actual_values(self):
        # 获得真实values
        ret = []
        for index, _ in enumerate(self.values):
            if self.random_types and self.random_types[index].name != RandomTypes.DEFAULT:
                ret.append(self.random_types[index].model_dump())
            else:
                ret.append(self.values[index])
        return ret


class InputsShape(BaseModel):
    model_config = ConfigDict(extra='forbid')
    dim_numbers: RandomConfig = RandomConfig(
        values=DEFAULT_DIM_NUMBERS, weights=DEFAULT_DIM_WEIGHTS
    )
    dim_values: Optional[Union[RandomConfig, List[RandomConfig]]] = RandomConfig(
        values=DEFAULT_DIM_VALUES, weights=DEFAULT_DIM_VALUES_WEIGHTS
    )
    max_length: int = MAX_NUMBER_OF_ELEMENTS

    @model_validator(mode="after")
    def check_params(self):
        if isinstance(self.dim_values, list):
            dim_values_len = len(self.dim_values)
            max_dim_numbers = max(self.dim_numbers.values)
            if dim_values_len != max_dim_numbers:
                raise ValueError(f"dim_values_length {dim_values_len} is not equal to "
                                 f"max(dim_numbers) {max_dim_numbers}, please check")
        return self


class RangeConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')
    valid: RandomConfig = RandomConfig(values=DEFAULT_RANGES_VALUES)
    invalid: RandomConfig = RandomConfig()
    valid_weights: float = VALID_WEIGHTS

    def update(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


class BoundaryConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')
    has_empty: Optional[bool] = True
    has_infnan: Optional[bool] = True
    has_scalar: Optional[bool] = True
    has_upper_border: Optional[bool] = True
    has_lower_border: Optional[bool] = True

    # 用于记录哪些字段是用户显式设置的
    _user_set_fields: set = PrivateAttr(default_factory=set)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._user_set_fields = set(kwargs.keys())

    @property
    def user_set_fields(self):
        return self._user_set_fields


class InputDesignConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')
    name: Optional[str] = None
    aclnn_name: Optional[str] = None
    type: Optional[str]
    required: bool = True
    backward: Optional[bool] = True
    align_32B: Optional[bool] = None
    outlier_values: Optional[List[float]] = None
    dtypes: Optional[RandomConfig] = RandomConfig(values=DEFAULT_DTYPES)
    ranges: Optional[RangeConfig] = RangeConfig()
    shapes: Optional[InputsShape] = InputsShape()
    tuple_numbers: Optional[RandomConfig] = RandomConfig(values=[1])
    boundary: Optional[BoundaryConfig] = BoundaryConfig()
    expected_error_msg: Optional[str] = None

    @model_validator(mode="after")
    def set_default_ranges_by_type(self):
        type_ = self.type
        if isinstance(self.ranges.invalid.values, list):
            if len(self.ranges.invalid.values) == 0:
                self.ranges.invalid.values = self.ranges.valid.values
            return self
        else:
            if 'tensor' in type_ and not self.ranges.invalid.values:
                self.ranges.update(**dict(invalid=RandomConfig(values=DEFAULT_TENSOR_RANGES_INVALID_VALUES)))
            if 'scalar' in type_ and not self.ranges.invalid.values:
                self.ranges.update(**dict(invalid=RandomConfig(values=DEFAULT_SCALAR_RANGES_INVALID_VALUES)))
            if 'attr' in type_ and not self.ranges.invalid.values:
                self.ranges.update(**dict(invalid=RandomConfig(values=DEFAULT_ATTR_RANGES_INVALID_VALUES)))
            return self

    @model_validator(mode="after")
    def check_extra_params(self):
        """非tensor类的inputs的边界参数不允许设置, 且默认值为None"""
        if self.type not in ["tensor", "tensors", "tensor_tuple"] and self.boundary:
            # 检查 boundary 中是否有用户显式设置的值
            for field_name in self.boundary.user_set_fields:
                raise ValueError(f"type: '{self.type}' does not support parameters: {field_name}")
        return self

    @model_validator(mode="after")
    def check_type_params(self):
        """检查 type 参数是否合法"""
        if self.type not in DEFAULT_TYPES:
            raise ValueError(f"type: '{self.type}' is not a valid type")
        return self

    @model_validator(mode="after")
    def check_tuple_numbers_param(self):
        """检查 tuple_numbers 参数是否合法"""
        if self.tuple_numbers.values != [1]:
            if self.type not in ["tensors", "tensor_tuple", "scalars", "scalar_tuple", "attrs", "attr_tuple"]:
                raise ValueError(f"type: '{self.type}' does not support tuple_numbers, please check.")
        return self

    @model_validator(mode="after")
    def check_outlier_values_param(self):
        """检查 outlier_values 参数是否合法"""
        if self.outlier_values and len(self.outlier_values) != 2:
            raise ValueError(f"outlier_values: '{self.typoutlier_valuese}' should have two values, please check.")
        return self


class StandardConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')
    acc: Optional[Union[str, dict]] = "default"
    perf: Optional[Union[str, List[float]]] = "not_key"

    @model_validator(mode="after")
    def check_single_bm_type(self):
        if not isinstance(self.acc, dict):
            return self
        name = list(self.acc.keys())[0]
        if name != "single_bm":
            return self
        if "type" not in self.acc[name].keys():
            self.acc[name]["type"] = SingBenchmarkType.HIGH_PRECISION.value
        acc_type = self.acc[name].get("type")
        if acc_type not in SingBenchmarkType.__members__.values():
            raise ValueError(f"type: {acc_type} is not a valid single benchmark type")
        return self

    def is_acc_benchmark(self):
        if isinstance(self.acc, str):
            name = self.acc
        else:
            name = list(self.acc.keys())[0]

        return "benchmark" in name

    def is_bm_benchmark(self):
        if isinstance(self.acc, str):
            name = self.acc
        else:
            name = list(self.acc.keys())[0]

        return "bm" in name


class DesignConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')

    def __init__(self, path):
        with open(path, "r") as fout:
            config = yaml.safe_load(fout)
        super().__init__(**config)
        self.adapt_shape_distribution()

    name: Optional[str] = None
    aclnn_name: Optional[str] = None
    triton_name: Optional[str] = None
    version: Optional[str] = None
    expected_error_msg: Optional[str] = None
    api: str = "pytorch"
    api_type: str = "function"
    aclnn_api_type: str = "aclnn_function"
    triton_api_type: str = "triton_function"
    fusion_api_type: str = "fusion_function"
    dist_api_type: str = "dist_function"
    generate: str = "default"
    standard: Optional[StandardConfig] = StandardConfig()
    backward: bool = False
    outputs: Optional[Union[str, int]] = None
    inputs: List[InputDesignConfig]
    tensor_input: InputDesignConfig = None
    method_inputs: List[InputDesignConfig] = None
    shape_distributions: Optional[List[List[Union[int, float]]]] = [
        [10000000, 0.3],
        [100000000, 0.1],
        [1000000000, 0.01],
    ]
    dtype_numbers: Optional[int] = DEFAULT_DTYPE_NUMBERS
    extra_numbers: Optional[Union[str, int]] = 0
    _is_gen_extra: Optional[bool] = False

    @property
    def is_gen_extra(self):
        return self._is_gen_extra

    @model_validator(mode="after")
    def check_dtype_numbers_and_extra_numbers_is_valid(self):
        if self.dtype_numbers < 0:
            raise ValueError("dtype_numbers should be non-negative but is set to negative!")
        if isinstance(self.extra_numbers, int) and self.extra_numbers < 0:
            raise ValueError("extra_numbers should be a non-negative int or string 'all' but is set to negative!")
        if isinstance(self.extra_numbers, str) and self.extra_numbers != 'all':
            raise ValueError("extra_numbers should be a non-negative int or string 'all' but is set to other string!")
        return self

    @is_gen_extra.setter
    def is_gen_extra(self, bool_value):
        self._is_gen_extra = bool_value

    def adapt_shape_distribution(self):
        all_shapes = []
        if self.inputs:
            all_shapes.extend(
                [
                    cur_input.shapes
                    for cur_input in self.inputs
                    if isinstance(cur_input, InputDesignConfig)
                ]
            )
        if isinstance(self.tensor_input, InputDesignConfig):
            all_shapes.append(self.tensor_input.shapes)
        if self.method_inputs:
            all_shapes.extend(
                [
                    cur_input.shapes
                    for cur_input in self.inputs
                    if isinstance(cur_input, InputDesignConfig)
                ]
            )

        total_length = sum(cur_shape.max_length for cur_shape in all_shapes)
        ratio = total_length / (MAX_NUMBER_OF_ELEMENTS * len(all_shapes))
        for shape_distribution in self.shape_distributions:
            shape_distribution[0] = round(shape_distribution[0] * ratio)
