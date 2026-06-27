# -*- coding: UTF-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""
版权信息：华为技术有限公司，版本所有(C) 2026-2026
修改记录：2026/3/19 16:14
功能：定义数据生成脚本共用的数据模型
"""
from enum import Enum


class DispatcherTargetType(Enum):
    METHOD = "method"
    CLASS = "class"
