# -*- coding: UTF-8 -*-
"""
版权信息：华为技术有限公司，版本所有(C) 2025-2025
修改记录：2025/12/23 11:45
功能：定义不同角色的参数的取值数据模型实现
"""
import random
from typing import List

import numpy
import torch
from scipy.stats import norm

from agent.generators.common_utils.common_dispatcher import CommonDispatcher
from agent.generators.common_utils.logger_util import get_logger
from agent.generators.data_definition.constants import DataMatchMap
from agent.generators.data_definition.param_models_def import StaticModel, NormalModel, UniformModel, IntUniformModel, LogUniformModel, \
    ChoiceModel, ParamRangeRoleRules, DispatcherTargetType


class ParamRangeValueModel(CommonDispatcher):
    def __init__(self, operator_name, param_name):
        self.logger = get_logger()
        self.operator_name = operator_name
        self.param_name = param_name

    @staticmethod
    def data_clip(tensor_data, post_process, clip_min, clip_max):
        """
        对生成的tensor进行clip操作
        :param tensor_data: 需要进行clip操作的tensor
        :param post_process: clip操作类型
        :param clip_min: 数值下限
        :param clip_max: 数值上限
        :return: torch.tensor
        """
        if post_process == "abs_clip_min":
            result_tensor = torch.clip(torch.abs(tensor_data), min=clip_min)
        elif post_process == "clip_max":
            result_tensor = torch.clip(tensor_data, max=clip_max)
        else:
            result_tensor = tensor_data
        return result_tensor

    def get_data_type_in_torch(self, acl_data_type):
        """
        通过acl_type获取数据在torch中对应的数据类型
        :param acl_data_type: acl中的数据类型字段，如BFLOAT16,FLOAT32
        :return: torch.float16
        """
        data_dtype = DataMatchMap.TENSOR_DTYPE_TRANSFER_TORCH_MAP.get(acl_data_type)
        if data_dtype is None:
            self.logger.error(
                "Param range model, static model, can't analysis dtype, operator_name : %s, param name : %s",
                self.operator_name, self.param_name)
            data_dtype = torch.float16
        return data_dtype

    # @CommonDispatcher.register(ParamRangeRoleRules.STATIC.value, target_type=DispatcherTargetType.METHOD.value)
    def static_model_generate(self, size: List, acl_data_type: str, model_def: StaticModel):
        """
        生成static模型的数据
        :param size: 数据大小
        :param acl_data_type: 数据类型
        :param model_def: static模型定义
        :return: torch.tensor
        """
        self.logger.debug(
            "Start generate param range by static, operator name: %s, param name: %s, size: %s, data type: %s",
            self.operator_name, self.param_name, size, acl_data_type)
        model_value = model_def.value
        data_dtype = self.get_data_type_in_torch(acl_data_type)
        if not isinstance(model_value, str):
            model_tensor = torch.full(size, fill_value=model_value, dtype=data_dtype)
            if model_def.post_process:
                model_tensor = ParamRangeValueModel.data_clip(model_tensor, model_def.post_process, model_def.clip_min,
                                                              model_def.clip_max)
            return model_tensor
        base_value = None
        if model_value.lower() == "nan":
            base_value = torch.nan
        elif model_value.lower() == "infinity":
            base_value = torch.inf
        elif model_value.lower() == "-infinity":
            base_value = -torch.inf
        else:
            self.logger.error("Can't analysis value in 'Static Model', value : %s", model_value)
        if base_value is None:
            return None
        model_tensor = torch.full(size, fill_value=base_value, dtype=data_dtype)
        self.logger.debug(
            "End generate param range by static, operator name: %s, param name: %s, size: %s, data type: %s",
            self.operator_name, self.param_name, size, acl_data_type)
        return model_tensor.tolist()

    # @CommonDispatcher.register(ParamRangeRoleRules.NORMAL.value, target_type=DispatcherTargetType.METHOD.value)
    def normal_model_generate(self, size: List, acl_data_type: str, model_def: NormalModel):
        """
        生成Normal类模型的数据
        :param size: 数据大小
        :param acl_data_type: 数据类型
        :param model_def: normal类模型定义
        :return: torch.tensor
        """
        self.logger.debug(
            "Start generate param range by normal, operator name: %s, param name: %s, size: %s, data type: %s",
            self.operator_name, self.param_name, size, acl_data_type)
        data_dtype = self.get_data_type_in_torch(acl_data_type)
        model_tensor = torch.normal(mean=model_def.mean, std=model_def.std, size=size)
        model_tensor = torch.tensor(model_tensor, dtype=data_dtype)
        if model_def.post_process:
            model_tensor = ParamRangeValueModel.data_clip(model_tensor, model_def.post_process, model_def.clip_min,
                                                          model_def.clip_max)
        self.logger.debug(
            "End generate param range by normal, operator name: %s, param name: %s, size: %s, data type: %s",
            self.operator_name, self.param_name, size, acl_data_type)
        return model_tensor.tolist()

    # @CommonDispatcher.register(ParamRangeRoleRules.UNIFORM.value, target_type=DispatcherTargetType.METHOD.value)
    def uniform_model_generate(self, size: List, acl_data_type: str, model_def: UniformModel):
        """
        生成Uniform类模型的数据
        :param size: 数据大小
        :param acl_data_type: 数据类型
        :param model_def: uniform类模型定义
        :return: torch.tensor
        """
        self.logger.debug(
            "Start generate param range by uniform, operator name: %s, param name: %s, size: %s, data type: %s",
            self.operator_name, self.param_name, size, acl_data_type)
        data_dtype = self.get_data_type_in_torch(acl_data_type)
        model_tensor = torch.randn(size=size, dtype=data_dtype)
        model_tensor = model_tensor.uniform_(model_def.min, model_def.max)
        if model_def.post_process:
            model_tensor = ParamRangeValueModel.data_clip(model_tensor, model_def.post_process, model_def.clip_min,
                                                          model_def.clip_max)
        self.logger.debug(
            "End generate param range by uniform, operator name: %s, param name: %s, size: %s, data type: %s",
            self.operator_name, self.param_name, size, acl_data_type)
        return model_tensor.tolist()

    # @CommonDispatcher.register(ParamRangeRoleRules.INTUNIFORM.value, target_type=DispatcherTargetType.METHOD.value)
    def intuniform_model_generate(self, size: List, acl_data_type: str, model_def: IntUniformModel):
        """
        生成IntUniform类模型的数据
        :param size: 数据大小
        :param acl_data_type: 数据类型
        :param model_def: intuniform类模型定义
        :return: torch.tensor
        """
        self.logger.debug(
            "Start generate param range by intuniform, operator name: %s, param name: %s, size: %s, data type: %s",
            self.operator_name, self.param_name, size, acl_data_type)
        data_dtype = self.get_data_type_in_torch(acl_data_type)
        model_tensor = torch.randint(low=int(model_def.min), high=int(model_def.max), size=size, dtype=data_dtype)
        if model_def.post_process:
            model_tensor = ParamRangeValueModel.data_clip(model_tensor, model_def.post_process, model_def.clip_min,
                                                          model_def.clip_max)
        self.logger.debug(
            "End generate param range by intuniform, operator name: %s, param name: %s, size: %s, data type: %s",
            self.operator_name, self.param_name, size, acl_data_type)
        return model_tensor.tolist()

    # @CommonDispatcher.register(ParamRangeRoleRules.LOGUNIFORM.value, target_type=DispatcherTargetType.METHOD.value)
    def loguniform_model_generate(self, size: List, acl_data_type: str, model_def: LogUniformModel):
        """
        生成LogUniform类模型的数据
        :param size: 数据大小
        :param acl_data_type: 数据类型
        :param model_def: loguniform类模型定义
        :return: torch.tensor
        """
        self.logger.debug(
            "Start generate param range by loguniform, operator name: %s, param name: %s, size: %s, data type: %s",
            self.operator_name, self.param_name, size, acl_data_type)
        data_dtype = DataMatchMap.TENSOR_DTYPE_TRANSFER_TORCH_MAP.get(acl_data_type, torch.float16)
        low = numpy.log(max(model_def.min, 1e-9))
        high = numpy.log(max(model_def.max, low + 1e-9))
        init_tensor = torch.randn(size=size, dtype=data_dtype)
        uniform_tensor = init_tensor.uniform_(low, high)
        model_tensor = torch.exp(uniform_tensor)
        if model_def.post_process:
            model_tensor = ParamRangeValueModel.data_clip(model_tensor, model_def.post_process, model_def.clip_min,
                                                          model_def.clip_max)
        self.logger.debug(
            "End generate param range by loguniform, operator name: %s, param name: %s, size: %s, data type: %s",
            self.operator_name, self.param_name, size, acl_data_type)
        return model_tensor.tolist()

    # @CommonDispatcher.register(ParamRangeRoleRules.CHOICE.value, target_type=DispatcherTargetType.METHOD.value)
    def choice_model_generate(self, size: List, acl_data_type, model_def: ChoiceModel):
        """
        生成choice类模型的数据
        :param size: 数据大小
        :param acl_data_type: 数据类型
        :param model_def: choice类模型定义
        :return: torch.tensor
        """
        self.logger.debug(
            "Start generate param range by choice, operator name: %s, param name: %s, size: %s, data type: %s",
            self.operator_name, self.param_name, size, acl_data_type)
        model_array = numpy.random.choice(model_def.options, size=size)
        model_tensor = torch.tensor(model_array)
        if model_def.post_process:
            model_tensor = ParamRangeValueModel.data_clip(model_tensor, model_def.post_process, model_def.clip_min,
                                                          model_def.clip_max)
        self.logger.debug(
            "End generate param range by choice, operator name: %s, param name: %s, size: %s, data type: %s",
            self.operator_name, self.param_name, size, acl_data_type)
        return model_tensor.tolist()


class ParamRangeValueModelStatic(CommonDispatcher):
    def __init__(self, operator_name, param_name):
        self.logger = get_logger()
        self.operator_name = operator_name
        self.param_name = param_name

    @CommonDispatcher.register(ParamRangeRoleRules.STATIC.value, target_type=DispatcherTargetType.METHOD.value)
    def static_model_generate(self, size: List, acl_data_type: str, model_def: StaticModel):
        """
        生成static模型的数据
        :param size: 数据大小
        :param acl_data_type: 数据类型
        :param model_def: static模型定义
        :return: torch.tensor
        """
        self.logger.debug(
            "Start generate param range by static, operator name: %s, param name: %s, size: %s, data type: %s",
            self.operator_name, self.param_name, size, acl_data_type)
        static_model_data = model_def.value
        self.logger.debug(
            "End generate param range by static, operator name: %s, param name: %s, size: %s, data type: %s",
            self.operator_name, self.param_name, size, acl_data_type)
        return static_model_data

    @CommonDispatcher.register(ParamRangeRoleRules.NORMAL.value, target_type=DispatcherTargetType.METHOD.value)
    def normal_model_generate(self, size: List, acl_data_type: str, model_def: NormalModel):
        """
        生成Normal类模型的数据
        :param size: 数据大小
        :param acl_data_type: 数据类型
        :param model_def: normal类模型定义
        :return: torch.tensor
        """
        self.logger.debug(
            "Start generate param range by normal, operator name: %s, param name: %s, size: %s, data type: %s",
            self.operator_name, self.param_name, size, acl_data_type)
        quantile_value_up = norm.ppf(0.975, model_def.mean, model_def.std).item()
        quantile_value_low = norm.ppf(0.025, model_def.mean, model_def.std).item()
        static_model_data = []
        if model_def.clip_min is not None:
            static_model_data.append(max(quantile_value_low, model_def.clip_min))
        else:
            static_model_data.append(quantile_value_low)
        if model_def.clip_max is not None:
            static_model_data.append(min(quantile_value_up, model_def.clip_max))
        else:
            static_model_data.append(quantile_value_up)
        self.logger.debug(
            f"End generate param range by normal, operator name: {self.operator_name}, param name: {self.param_name}, "
            f"size: {size}, data type: {acl_data_type}, data range : {static_model_data}")
        return static_model_data

    @CommonDispatcher.register(ParamRangeRoleRules.UNIFORM.value, target_type=DispatcherTargetType.METHOD.value)
    def uniform_model_generate(self, size: List, acl_data_type: str, model_def: UniformModel):
        """
        生成Uniform类模型的数据
        :param size: 数据大小
        :param acl_data_type: 数据类型
        :param model_def: uniform类模型定义
        :return: torch.tensor
        """
        self.logger.debug(
            "Start generate param range by uniform, operator name: %s, param name: %s, size: %s, data type: %s",
            self.operator_name, self.param_name, size, acl_data_type)

        static_model_data = [model_def.min, model_def.max]

        self.logger.debug(
            "End generate param range by uniform, operator name: %s, param name: %s, size: %s, data type: %s",
            self.operator_name, self.param_name, size, acl_data_type)
        return static_model_data

    @CommonDispatcher.register(ParamRangeRoleRules.INTUNIFORM.value, target_type=DispatcherTargetType.METHOD.value)
    def intuniform_model_generate(self, size: List, acl_data_type: str, model_def: IntUniformModel):
        """
        生成IntUniform类模型的数据
        :param size: 数据大小
        :param acl_data_type: 数据类型
        :param model_def: intuniform类模型定义
        :return: torch.tensor
        """
        self.logger.debug(
            "Start generate param range by intuniform, operator name: %s, param name: %s, size: %s, data type: %s",
            self.operator_name, self.param_name, size, acl_data_type)
        static_model_data = [model_def.min, model_def.max]
        self.logger.debug(
            "End generate param range by intuniform, operator name: %s, param name: %s, size: %s, data type: %s",
            self.operator_name, self.param_name, size, acl_data_type)
        return static_model_data

    @CommonDispatcher.register(ParamRangeRoleRules.INTUNIFORMODD.value, target_type=DispatcherTargetType.METHOD.value)
    def intuniformodd_model_generate(self, size: List, acl_data_type: str, model_def: IntUniformModel):
        """
        生成IntUniformOdd类模型的数据
        :param size: 数据大小
        :param acl_data_type: 数据类型
        :param model_def: intuniformodd类模型定义
        :return: torch.tensor
        """
        self.logger.debug(
            "Start generate param range by intuniformodd, operator name: %s, param name: %s, size: %s, data type: %s",
            self.operator_name, self.param_name, size, acl_data_type)
        static_model_data = [model_def.min, model_def.max]
        self.logger.debug(
            "End generate param range by intuniformodd, operator name: %s, param name: %s, size: %s, data type: %s",
            self.operator_name, self.param_name, size, acl_data_type)
        return static_model_data

    @CommonDispatcher.register(ParamRangeRoleRules.LOGUNIFORM.value, target_type=DispatcherTargetType.METHOD.value)
    def loguniform_model_generate(self, size: List, acl_data_type: str, model_def: LogUniformModel):
        """
        生成LogUniform类模型的数据
        :param size: 数据大小
        :param acl_data_type: 数据类型
        :param model_def: loguniform类模型定义
        :return: torch.tensor
        """
        self.logger.debug(
            "Start generate param range by loguniform, operator name: %s, param name: %s, size: %s, data type: %s",
            self.operator_name, self.param_name, size, acl_data_type)
        static_model_data = [model_def.min, model_def.max]
        self.logger.debug(
            "End generate param range by loguniform, operator name: %s, param name: %s, size: %s, data type: %s",
            self.operator_name, self.param_name, size, acl_data_type)
        return static_model_data

    @CommonDispatcher.register(ParamRangeRoleRules.CHOICE.value, target_type=DispatcherTargetType.METHOD.value)
    def choice_model_generate(self, size: List, acl_data_type, model_def: ChoiceModel):
        """
        生成choice类模型的数据
        :param size: 数据大小
        :param acl_data_type: 数据类型
        :param model_def: choice类模型定义
        :return: torch.tensor
        """
        self.logger.debug(
            "Start generate param range by choice, operator name: %s, param name: %s, size: %s, data type: %s",
            self.operator_name, self.param_name, size, acl_data_type)
        static_model_data = random.choice(model_def.options)
        self.logger.debug(
            "End generate param range by choice, operator name: %s, param name: %s, size: %s, data type: %s",
            self.operator_name, self.param_name, size, acl_data_type)
        return static_model_data
