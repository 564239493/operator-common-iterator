# -*- coding: UTF-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
"""
版权信息：华为技术有限公司，版本所有(C) 2025-2025
修改记录：2025/12/23 20:34
功能：定义tensor参数的shape构建
"""
import json
import math
import os.path
import random
from collections import defaultdict
from typing import List, Tuple, Dict

from pydantic import ValidationError

from agent.generators.common_utils.common_dispatcher import CommonDispatcher
from agent.generators.common_utils.logger_util import LazyLogger
from agent.generators.data_definition.common_models import DispatcherTargetType
from agent.generators.data_definition.constants import ParamModelConfig
from agent.generators.data_definition.param_models_def import SingleShapeStrategy, ParamShapeRoleRules

logger = LazyLogger()


class ParamShapeModel(CommonDispatcher):
    def __init__(self, operator_name, shape_definitions_file_path=None, shape_pools: Dict[str, List] = None,
                 shape_strategy: Dict[str, SingleShapeStrategy] = None):
        # shape_pools: Dict[str, List], shape_strategies: Dict[str, SingleShapeStrategy]
        self.shape_definitions_file_path = shape_definitions_file_path
        self.operator_name = operator_name
        if shape_pools is not None and shape_strategy is not None:
            self.shape_pools = shape_pools
            self.shape_strategies = shape_strategy
        else:
            self.shape_pools, self.shape_strategies = ParamShapeModel.init_shape_model_definition(
                self.shape_definitions_file_path)
        self.param_counter = defaultdict(int)

    @staticmethod
    def init_shape_model_definition(shape_definitions_file_path=None) -> Tuple[Dict[str, List],
    Dict[str, SingleShapeStrategy]]:
        if shape_definitions_file_path is None:
            logger.info("User's shape definitions is None, use default shape definitions file: %s",
                        ParamModelConfig.SHAPE_DEFINITIONS_FILE_PATH)
            shape_definitions_file_path = ParamModelConfig.SHAPE_DEFINITIONS_FILE_PATH
        if not os.path.exists(shape_definitions_file_path):
            raise FileNotFoundError("Can't find shape definitions file : ", shape_definitions_file_path)
        with open(shape_definitions_file_path, "r", encoding="utf-8") as f:
            shape_definitions = json.load(f)
        shape_pools = shape_definitions.get("pools")
        if shape_pools is None:
            raise KeyError("Can't analysis shape pools in shape definitions, key : 'pools' is not exist")
        shape_strategies_map = shape_definitions.get("strategies")
        shape_strategies = defaultdict(SingleShapeStrategy)
        for strategy_name, strategy in shape_strategies_map.items():
            try:
                strategy_instance = SingleShapeStrategy(**strategy, strategy_name=strategy_name)
                shape_strategies[strategy_name] = strategy_instance
            except ValidationError as e:
                logger.error("Can't analysis shape model: %s, err mse: %s", strategy_name, str(e))
        if shape_strategies is None:
            raise KeyError("Can't analysis shape shape_strategies in shape definitions, key : 'strategies' not exist")
        return shape_pools, shape_strategies

    @staticmethod
    def is_feature_dim(dims, dim_index):
        """
        判断当前维度的位置，选取shape取值时，对后两维的shape使用其他策略处理，此处判断是不是后两维
        :param dims: shape总维度
        :param dim_index: 当前index
        :return: 如果是后两维，则返回false，否则返回true
        """
        return dim_index >= dims - 2

    @staticmethod
    def tensor_elements_limit_check(current_elements, current_shape_value,
                                    not_gratify_value=ParamModelConfig.TENSOR_SHAPE_MIN_VALUE,
                                    max_elements=ParamModelConfig.TENSOR_TENSOR_ELEMENT_LIMIT):
        """
        判断当前tensor的元素个数是否超出上限，如果超出，则将当前dim的shape_value设置为1，避免元素个数超出上限，导致内存溢出，
        tensor元素最大上限可配置
        :param current_elements: 当前以后的元素个数
        :param current_shape_value: 当前dim的shape取值
        :param not_gratify_value: 不满足limit_check条件时的shape取值
        :param max_elements: 允许的tensor元素个数上限
        :return: 校验后的shape取值
        """
        try:
            if current_elements * current_shape_value > max_elements:
                return not_gratify_value
            return current_shape_value
        except TypeError as e:
            logger.error("tensor elements limit check failed, err msg : %s", str(e))
            return not_gratify_value

    def infer_pool_by_param(self, param_name: str, strategy_config: SingleShapeStrategy):
        """
        根据参数名称获取对应shape的pool
        :param param_name: 参数名称
        :param strategy_config: shape建模策略的配置
        :return: pool取值列表
        """
        if param_name is None:
            default_pool_name = strategy_config.default_pool
            return self.shape_pools.get(default_pool_name, self.shape_strategies.get(ParamShapeRoleRules.DEFAULT.value))
        param_name_lower = param_name.lower()
        # level1 Override
        overrides = strategy_config.param_overrides
        if overrides is not None:
            override_pool = overrides.get(param_name) or overrides.get(param_name_lower)
            if override_pool is not None and override_pool in self.shape_pools:
                return self.shape_pools.get(override_pool)
        # level2 rules
        role_rules = strategy_config.role_rules
        for rule in role_rules:
            for key_word in rule.keywords:
                if key_word.lower() in param_name_lower and rule.pool in self.shape_pools:
                    return self.shape_pools.get(rule.pool)

        return self.shape_pools.get(ParamShapeRoleRules.DEFAULT.value)

    def get_strategy_pool(self, strategy_name, param_name=None):
        """
        根据策略名称获取策略取值候选池，如果策略名称不在已定义的策略全集中，则选择默认候选池，策略全集中必包含默认候选池 “default”
        :param strategy_name: 策略名称
        :param param_name: 参数名称，若参数名称为空，返回默认的
        :return: List， pool
        """
        strategy_config = self.shape_strategies.get(strategy_name)
        # 当strategy_config为None时，使用shape_pool的default值
        # 当param_name为None时，如果strategy_config的base_pool存在，则使用base_pool，否则使用default_pool，如果两个都不存在，则使用
        # self.shape_pool中的默认值
        if strategy_config is None:
            strategy_pool = self.shape_pools.get(ParamShapeRoleRules.DEFAULT.value)
            logger.warning(
                "Can't find match shape strategy by operator name, operator_name : %s, param name: %s, "
                "current strategy: %s, use default strategy pool: %s",
                self.operator_name, param_name, strategy_name, strategy_pool)
        elif param_name is None:
            if strategy_config.base_pool is None:
                strategy_pool_name = strategy_config.default_pool
            else:
                strategy_pool_name = strategy_config.base_pool
            strategy_pool = self.shape_pools.get(strategy_pool_name)
        else:
            strategy_pool = self.infer_pool_by_param(param_name, strategy_config)
        return strategy_config, strategy_pool

    def get_candidate_pool(self, dims, random_probability=0.7):
        """
        当dims维度较高时，避免选择数值比较大的shape值，设置较小的shape取值候选池
        :param dims: shape维度值
        :param random_probability: 随机概率阈值，通过随机数值域概率值的对比，选择不同的候选池
        :return: shape取值候选池
        """
        if dims > ParamModelConfig.TENSOR_HIGH_DIMENSION_BOUNDARY:
            candidate_pool = [
                ParamModelConfig.TENSOR_SHAPE_MIN_VALUE] if random.random() < random_probability \
                else self.shape_pools.get("high_dimensional_shape")
        else:
            candidate_pool = self.shape_pools.get("small_integers")
        return candidate_pool

    def hybrid_sampling_shape_value(self, param_name, strategy_pool):
        """
        利用混合采样算法选择shape填充值，混合采样算法为：一种“确定性遍历 + 统计分布采样”的混合生成算法。
        设某维度的候选池为P=p_1,p_2,…,p_n，p_i为候选池中的shape取值，该参数已生成的次数为k。第k次生成的维度值d_k服从以下分段函数：
        if k < n: d_k = p_k; else: d_k = Round(N(mu, sigma^2))
        mu = (min(P) + max(P)) / 2; sigma = (max(P) - min(P)) / 6
        阶段一：确定性枚举。对于每个参数，前 N 次生成严格按照采样池中的预设值（如 768, 1024, 2048）输出。
        确保所有典型的、高频使用的、已知的边界值（如 Llama 的标准参数）在测试初期被 100% 覆盖。
        阶段二：高斯模糊。当取遍典型值时，算法自动切换为基于池子分布特征的 高斯分布采样。以池子的最小/最大值为区间，
        生成正态分布的随机整数，探索非标准维度。分布参数遵循"3σ"  准则覆盖池定义的范围
        同时对选择的d_k的上下限做处理：choose_shape_value = Clip(d_k, 1. 65535)
        :param param_name: 参数名称，用于统计当前该参数shape取值已生成次数，即k
        :param strategy_pool: 策略对应的shape取值候选池
        :return: 指定dim_index的shape取值
        """
        if not strategy_pool:
            return ParamModelConfig.TENSOR_SHAPE_MIN_VALUE
        param_choose_count = self.param_counter.get(param_name, 0)
        # 阶段一
        if param_choose_count < len(strategy_pool):
            shape_value = max(ParamModelConfig.TENSOR_SHAPE_MIN_VALUE,
                              min(strategy_pool[param_choose_count], ParamModelConfig.TENSOR_SHAPE_MAX_VALUE))
            return shape_value
        # 阶段二
        min_pool_value = min(strategy_pool)
        max_pook_value = max(strategy_pool)
        if min_pool_value == max_pook_value:
            return min_pool_value
        normal_distribution_mu = (min_pool_value + max_pook_value) / 2.0
        normal_distribution_sigma = (max_pook_value - min_pool_value) / 6.0
        normal_distribution_random_value = random.gauss(normal_distribution_mu, normal_distribution_sigma)
        random_shape_value = int(round(normal_distribution_random_value))
        shape_value = max(ParamModelConfig.TENSOR_SHAPE_MIN_VALUE,
                          min(random_shape_value, ParamModelConfig.TENSOR_SHAPE_MAX_VALUE))
        return shape_value

    def fill_shape_safely(self, shape_list, indices_to_index, pool, param_name=None):
        """
        shape中设置特殊值后，如large, odd等，填充剩下位置的值
        :param shape_list: 设置特殊值后的shape
        :param indices_to_index: 需要填充shape取值的索引
        :param pool: shape可取值的候选池
        :param param_name: 参数名称
        :return: 填充后的shape
        """
        current_elements = math.prod(shape_list)
        for index in indices_to_index:
            choose_shape_value = ParamModelConfig.TENSOR_SHAPE_MIN_VALUE
            for _ in range(ParamModelConfig.RANDOM_GENERATE_SHAPE_TIME_LIMIT):
                if param_name:
                    choose_value = self.hybrid_sampling_shape_value(param_name, pool)
                else:
                    choose_value = random.choice(pool)
                choose_shape_value = ParamShapeModel.tensor_elements_limit_check(current_elements, choose_value)
            max_elements_limit = int(ParamModelConfig.TENSOR_TENSOR_ELEMENT_LIMIT * 0.8)
            choose_shape_value = ParamShapeModel.tensor_elements_limit_check(current_elements=current_elements,
                                                                             current_shape_value=choose_shape_value,
                                                                             max_elements=max_elements_limit)
            shape_list[index] = choose_shape_value
            current_elements *= choose_shape_value
        return shape_list

    @CommonDispatcher.register(ParamShapeRoleRules.TYPICAL.value, target_type=DispatcherTargetType.METHOD.value)
    def typical_strategy(self, dims: int, param_name: str) -> List:
        """
        基于shape的典型取值，通过混合采样算法选择shape的取值，匹配硬件的最佳性能路径
        :param dims: shape的维度
        :param param_name: 参数的名称
        :return: shape list，如：[2,4,8]
        """
        logger.debug("Start generate param shape by model: 'Typical', operator name: %s, param name: %s, dims: %s",
                     self.operator_name, param_name, dims)
        if dims <= 0:
            logger.error("Tensor shape dim is invalid: %s, strategy : %s, operator_name : %s, param name： %s", dims,
                         ParamShapeRoleRules.TYPICAL.value, self.operator_name, param_name)
            return []
        _, strategy_pool = self.get_strategy_pool(ParamShapeRoleRules.TYPICAL.value, param_name)
        strategy_shape = [ParamModelConfig.TENSOR_SHAPE_MIN_VALUE] * dims
        current_elements = 1
        for dim_index in range(dims - 1, -1, -1):
            if ParamShapeModel.is_feature_dim(dims, dim_index):
                shape_value = self.hybrid_sampling_shape_value(param_name, strategy_pool)
                choose_shape_value = ParamShapeModel.tensor_elements_limit_check(current_elements, shape_value)
                self.param_counter[param_name] += 1
            else:
                candidate_pool = self.get_candidate_pool(dims)
                choose_shape_value = 1
                for _ in range(ParamModelConfig.RANDOM_GENERATE_SHAPE_TIME_LIMIT):
                    choose_value = random.choice(candidate_pool)
                    choose_shape_value = ParamShapeModel.tensor_elements_limit_check(current_elements, choose_value)
            strategy_shape[dim_index] = choose_shape_value
            current_elements *= choose_shape_value
        if current_elements > ParamModelConfig.TENSOR_TENSOR_ELEMENT_LIMIT:
            strategy_shape = [ParamModelConfig.TENSOR_SHAPE_MIN_VALUE] * (dims - 1) + [strategy_shape[-1]]
        logger.debug("End generate param shape by typical, operator name: %s, param name: %s, dims: %s",
                     self.operator_name, param_name, dims)
        return strategy_shape

    @CommonDispatcher.register(ParamShapeRoleRules.HAS_LARGE_SIZE.value, target_type=DispatcherTargetType.METHOD.value)
    def large_size_strategy(self, dims: int, param_name: str) -> List:
        """
        包含接近极限显存的大维度取值，用来测验证大块内存的申请、搬运以及索引计算是否有溢出风险
        :param dims: shape的维度
        :param param_name: 参数名称
        :return: shape list, [4096, 2048]
        """
        logger.debug(
            "Start generate param shape by model: 'Has_Large_Size', operator name: %s, param name: %s, dims: %s",
            self.operator_name, param_name, dims)
        strategy_config, strategy_pool = self.get_strategy_pool(ParamShapeRoleRules.HAS_LARGE_SIZE.value)
        large_value = ParamModelConfig.DEFAULT_LARGE_FIX_DIM if strategy_config is None else strategy_config.fixed_large_dim

        strategy_shape = [ParamModelConfig.TENSOR_SHAPE_MIN_VALUE] * dims
        if dims <= 0:
            logger.error("Tensor shape dim is invalid: %s, strategy : %s, operator_name : %s, param name： %s", dims,
                         ParamShapeRoleRules.HAS_LARGE_SIZE.value, self.operator_name, param_name)
            return []
        large_value_index = random.randint(0, dims - 1)
        large_value = min(large_value, ParamModelConfig.TENSOR_TENSOR_ELEMENT_LIMIT)
        strategy_shape[large_value_index] = large_value
        small_pool = self.shape_pools.get("small_shape_value")
        left_value_indices = [index for index in range(dims) if index != large_value_index]
        random.shuffle(left_value_indices)
        strategy_shape_with_fill = self.fill_shape_safely(strategy_shape, left_value_indices, small_pool)
        logger.debug("End generate param shape by has_large_size, operator name: %s, param name: %s, dims: %s",
                     self.operator_name, param_name, dims)
        return strategy_shape_with_fill

    @CommonDispatcher.register(ParamShapeRoleRules.HAS_ODD_SIZE.value, target_type=DispatcherTargetType.METHOD.value)
    def odd_strategy(self, dims: int, param_name: str) -> List:
        """
        包含奇数或质数的维度（如 3, 7, 13）。用于深度测试底层的补齐逻辑及非对齐内存访问的正确性
        :param dims: shape的维度
        :param param_name: 参数名称
        :return: shape list, [3,7,13]
        """
        logger.debug(
            "Start generate param shape by model: 'Has_Odd_Size', operator name: %s, param name: %s, dims: %s",
            self.operator_name, param_name, dims)
        if dims <= 0:
            logger.error("Tensor shape dim is invalid: %s, strategy : %s, operator_name : %s, param name： %s", dims,
                         ParamShapeRoleRules.HAS_ODD_SIZE.value, self.operator_name, param_name)
            return []
        _, strategy_pool = self.get_strategy_pool(ParamShapeRoleRules.HAS_ODD_SIZE.value)
        strategy_shape = [ParamModelConfig.TENSOR_SHAPE_MIN_VALUE] * dims
        indices = list(range(dims))
        random.shuffle(indices)
        strategy_shape = self.fill_shape_safely(strategy_shape, indices, strategy_pool)
        logger.debug("End generate param shape by has_odd_size, operator name: %s, param name: %s, dims: %s",
                     self.operator_name, param_name, dims)
        return strategy_shape

    @CommonDispatcher.register(ParamShapeRoleRules.HAS_SIZE_1.value, target_type=DispatcherTargetType.METHOD.value)
    def one_size_strategy(self, dims: int, param_name: str) -> List:
        """
        包含大小为 1 的维度。专门用于触发和验证广播机制的正确性
        :param dims: shape的维度
        :param param_name: 参数名称
        :return: shape list, [1,4,8]
        """
        logger.debug(
            "Start generate param shape by model: 'Has_One_Size', operator name: %s, param name: %s, dims: %s",
            self.operator_name, param_name, dims)
        _, strategy_pool = self.get_strategy_pool(ParamShapeRoleRules.HAS_SIZE_1.value)
        strategy_shape = [ParamModelConfig.TENSOR_SHAPE_MIN_VALUE] * dims
        if dims <= 0:
            logger.error("Tensor shape dim is invalid: %s, strategy : %s, operator_name : %s, param name： %s", dims,
                         ParamShapeRoleRules.HAS_SIZE_1.value, self.operator_name, param_name)
            return []
        one_value_index = random.randint(0, dims - 1)
        strategy_shape[one_value_index] = 1
        indices_fill = [index for index in range(dims) if index != one_value_index]
        random.shuffle(indices_fill)
        strategy_shape = self.fill_shape_safely(strategy_shape, indices_fill, strategy_pool)
        logger.debug("End generate param shape by has_one_size, operator name: %s, param name: %s, dims: %s",
                     self.operator_name, param_name, dims)
        return strategy_shape
