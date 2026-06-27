# Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
import logging
from enum import Enum
import torch
import numpy as np
from packaging import version

EXCEPT_TYPES = {
    "int": "int32",
    "float": "fp32",
    "float16": "fp16",
    "float32": "fp32",
    "float64": "fp64",
    "bfloat16": "bf16",
    "double": "fp64",
}


class NpDtype(Enum):
    FP64 = np.float64
    FP32 = np.float32
    FP16 = np.float16
    BF16 = np.float16
    HF32 = np.float32
    TF32 = np.float32
    INT64 = np.int64
    UINT64 = np.uint64
    INT32 = np.int32
    UINT32 = np.uint32
    INT16 = np.int16
    UINT16 = np.uint16
    INT8 = np.int8
    UINT8 = np.uint8
    BOOL = np.bool_

    @classmethod
    def get(cls, key):
        try:
            return cls[key.upper()].value
        except KeyError as e:
            if key in EXCEPT_TYPES:
                except_data = EXCEPT_TYPES[key]
                logging.error(
                    f"Datatype: '{key}' is not excepted, replace with '{except_data}'"
                )
                return cls[except_data.upper()].value
            else:
                raise e


class TorchDtype(Enum):
    FP64 = torch.float64
    FP32 = torch.float32
    FP16 = torch.float16
    BF16 = torch.bfloat16
    HF32 = torch.float32
    TF32 = torch.float32
    INT64 = torch.int64
    INT32 = torch.int32
    INT16 = torch.int16
    INT8 = torch.int8
    UINT8 = torch.uint8
    BOOL = torch.bool
    COMPLEX64 = torch.complex64
    COMPLEX128 = torch.complex128
    FP8E4M3 = torch.float8_e4m3fn
    FP8E5M2 = torch.float8_e5m2

    if version.parse(torch.__version__) >= version.parse("2.3"):
        UINT64 = torch.uint64
        UINT32 = torch.uint32
        UINT16 = torch.uint16

    @classmethod
    def get(cls, key):
        try:
            return cls[key.upper()].value
        except KeyError as e:
            if key in EXCEPT_TYPES:
                except_data = EXCEPT_TYPES[key]
                logging.error(
                    f"Datatype: '{key}' is not excepted, replace with '{except_data}'"
                )
                return cls[except_data.upper()].value
            else:
                raise e


class StandardDtype(Enum):
    INT = int
    INT8_T = np.int8
    INT32_T = np.int32
    INT64_T = int
    UINT8_T = np.uint8
    UINT32_T = np.uint32
    UINT64_T = np.uint64
    DOUBLE = float
    FLOAT = np.float32
    ATTR_BOOL = bool
    STRING = str
    NON_PARAM = str


class SingBenchmarkType(str, Enum):
    HIGH_PRECISION = "high_precision"
    HIGH_PERFORMANCE = "high_performance"
    VECTOR_FUSED = "vector_fused"


class IntervalType(Enum):
    Open = "open"
    Close = "close"


class ExtraTensorType(Enum):
    SCALAR = "scalar"
    LOWER_BORDER = "lower_border"
    UPPER_BORDER = "upper_border"
    EMPTY = "empty"
    INFNAN = "infnan"
    NORMAL = "normal"
