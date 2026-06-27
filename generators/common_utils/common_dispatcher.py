# -*- coding: UTF-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
"""
版权信息：华为技术有限公司，版本所有(C) 2025-2025
修改记录：2025/12/25 15:26
功能：通用调度器类，可以用于方法注册，也可以用于类的注册，需注意注册使用标识符的唯一性
1. 定义一个方法/类注册表（通常是属性字典，键为字符串标识，值为方法对象/类）；
2. 提供「注册接口」：将方法或类与字符串标识绑定，存入注册表；
3. 提供「调用接口」：接收字符串标识，从注册表匹配并执行对应方法或类。
使用示例
******** 注册方法(实例 / 类 / 静态方法，自动识别)
# 1. 定义业务类，继承 CommonDispatcher 以使用方法调度能力
class BusinessCalculator(CommonDispatcher):
    # 注册实例方法（自动识别为 method）
    @CommonDispatcher.register("calculate_sum")
    def calc_sum(self, a: int, b: int):
        # 实例方法
        return f"[实例方法]  {a} + {b} = {a + b}"

    # 注册类方法（自动识别为 method，注意：类方法装饰器需放在外层）
    @classmethod
    @CommonDispatcher.register("print_class_info")
    def show_class_detail(cls):
        # 类方法
        return f"[类方法]  类名：{cls.__name__} | 方法数：{len(cls.method_registry)} | 类数：{len(cls.class_registry)}"

    # 注册静态方法（自动识别为 method，注意：静态方法装饰器需放在外层）
    @staticmethod
    @CommonDispatcher.register("check_positive")
    def is_positive(num: int):
        # 静态方法
        return f"[静态方法]  {num} 是正数：{num > 0}"

    # 注册实例方法（手动指定 target_type="method"，避免歧义）
    @CommonDispatcher.register("calculate_product", target_type="method")
    def calc_product(self, a: int, b: int):
        # 手动指定target_type,实例方法
        return f"[实例方法]  {a} × {b} = {a * b}"

******** 注册类（自动识别，用于后续实例化）
# 2. 注册普通类（自动识别为 class）
@CommonDispatcher.register("student")
class Student:
    def __init__(self, name: str, age: int):
        self.name = name
        self.age = age

    def __str__(self):
        return f"[学生实例]  姓名：{self.name} | 年龄：{self.age} 岁"

# 注册普通类（手动指定 target_type="class"，避免歧义）
@CommonDispatcher.register("teacher", target_type="class")
class Teacher:
    def __init__(self, name: str, subject: str):
        self.name = name
        self.subject = subject

    def __str__(self):
        return f"[老师实例]  姓名：{self.name} | 教授科目：{self.subject}"

# 注册复杂类（带方法的类，不影响注册）
@CommonDispatcher.register("manager")
class Manager:
    def __init__(self, name: str, department: str):
        self.name = name
        self.department = department

    def get_info(self):
        return f"[管理员实例]  姓名：{self.name} | 负责部门：{self.department}"

    def __str__(self):
        return self.get_info()
******** 调度调用()

if __name__ == "__main__":
    # 1. 查看注册信息
    CommonDispatcher.show_registries()

    # 2. 实例化调度器
    dispatcher = BusinessCalculator()

    # -------------------------- 方式1：不传入 target_type（自动识别，简洁） --------------------------
    print("\n" + "-"*40 + " 方式1：不传入 target_type（自动识别） " + "-"*40)
    # 自动识别为方法
    sum_result = dispatcher.dispatch("calculate_sum", 10, 20)
    class_info = dispatcher.dispatch("print_class_info")
    positive_result = dispatcher.dispatch("check_positive", 99)

    # 自动识别为类
    student = dispatcher.dispatch("student", "Alice", 20)
    teacher = dispatcher.dispatch("teacher", name="Bob", subject="数学")

    # 打印结果
    print(sum_result)
    print(class_info)
    print(positive_result)
    print(student)
    print(teacher)

    # -------------------------- 方式2：传入 target_type（明确无歧义，优先推荐复杂场景） --------------------------
    print("\n" + "-"*40 + " 方式2：传入 target_type（手动指定） " + "-"*40)
    # 手动指定 target_type="method"
    product_result = dispatcher.dispatch("calculate_sum", 30, 40, target_type="method")
    # 手动指定 target_type="class"
    teacher2 = dispatcher.dispatch("teacher", name="Charlie", subject="英语", target_type="class")

    # 打印结果
    print(product_result)
    print(teacher2)
    
    # -------------------------- 方式3：类参数通过dict传入 --------------------------------------------------
    # 简化方式：字典解包 **dict，直接传入 kwargs
    student_dict = {"name": "Alice", "age": 20}
    # 自动识别 + 字典解包
    student = dispatcher.dispatch("student", **student_dict)
    # 手动指定 target_type + 字典解包
    teacher_dict = {"name": "Bob", "subject": "数学"}
    teacher = dispatcher.dispatch("teacher", target_type="class", **teacher_dict)

    print(student)
    print(teacher)

    # -------------------------- 异常场景测试 --------------------------
    print("\n" + "-"*40 + " 异常场景测试 " + "-"*40)
    # 异常1：传入无效的 target_type
    try:
        dispatcher.dispatch("calculate_sum", 10, 20, target_type="func")
    except ValueError as e:
        print(f"异常（无效 target_type）：{e}")

    # 异常2：标识未注册
    try:
        dispatcher.dispatch("invalid_key")
    except ValueError as e:
        print(f"异常（标识未注册）：{e}")

    # 异常3：手动指定类型，但标识不存在
    try:
        dispatcher.dispatch("invalid_method", target_type="method")
    except ValueError as e:
        print(f"异常（方法标识不存在）：{e}")
"""
from agent.generators.common_utils.logger_util import LazyLogger
from agent.generators.data_definition.common_models import DispatcherTargetType

logger = LazyLogger()

class CommonDispatcher:
    # 类级别的双注册表：隔离方法与类，避免标识冲突
    method_registry = {}  # {字符串标识: 方法对象}
    class_registry = {}  # {字符串标识: 类对象}

    # -------------------------- 通用注册装饰器（自动识别方法/类） --------------------------
    @classmethod
    def register(cls, key: str, target_type: str = None):
        """
        通用注册装饰器：支持自动/手动区分方法（实例/类/静态）和类
        :param key: 唯一字符串标识（用于后续调度）
        :param target_type: None（自动识别）/"method"/"class"，手动指定可避免歧义
        :return: 装饰器函数
        """

        def decorator(obj):
            # 步骤1：确定目标类型（自动识别 or 手动指定）
            if target_type is None:
                # 自动识别逻辑：
                # 1. 静态方法/类方法 直接判定为 method
                # 2. 可调用对象且不是 type（类的类型是 type），判定为 method（实例方法）
                # 3. 是 type 类型，判定为 class
                if isinstance(obj, (staticmethod, classmethod)):
                    actual_type = DispatcherTargetType.METHOD.value
                elif callable(obj) and not isinstance(obj, type):
                    actual_type = DispatcherTargetType.METHOD.value
                elif isinstance(obj, type):
                    actual_type = DispatcherTargetType.CLASS.value
                else:
                    raise TypeError(
                        "Unsupported register type：%s, only support function(instance/class_function/staticmethod)"
                        " or class",
                        type(obj))
            else:
                # 手动指定类型，做合法性校验
                all_target_types = [target.value for target in DispatcherTargetType]
                if target_type not in all_target_types:
                    raise ValueError("iInvalid target_type: %s, only supported target type: %s", target_type,
                                     all_target_types)
                actual_type = target_type

            # 步骤2：根据类型注册到对应注册表，避免重复注册
            if actual_type == DispatcherTargetType.METHOD.value:
                if key in cls.method_registry:
                    raise ValueError("Method identifier: %s' already existed, can't be registered again", key)
                cls.method_registry[key] = obj
                # logger.info("Method: %s register success", key)
            elif actual_type == DispatcherTargetType.CLASS.value:
                if key in cls.class_registry:
                    raise ValueError(f"Class identifier: %s already existed, can't be registered again", key)
                cls.class_registry[key] = obj
                #logger.info(f"Class: %s register success", key

            # 步骤3：返回原对象，不影响其正常使用（装饰器无侵入性）
            return obj

        return decorator

    # -------------------------- 方法调度（内部使用，兼容三种方法类型） --------------------------
    def _dispatch_method(self, method_key: str, *args, **kwargs):
        """
        内部方法：调度方法（兼容实例/类/静态方法）
        :param method_key: 方法字符串标识
        :param args: 方法位置参数
        :param kwargs: 方法关键字参数
        :return: 方法执行结果
        """
        if method_key not in self.method_registry:
            raise ValueError("Unregistered method identifier: %s", method_key)
        target_method = self.method_registry[method_key]

        # 区分方法类型，正确绑定调用对象
        if isinstance(target_method, staticmethod):
            return target_method(*args, **kwargs)
        elif isinstance(target_method, classmethod):
            return target_method(self.__class__, *args, **kwargs)
        else:  # 实例方法
            return target_method(self, *args, **kwargs)

    # -------------------------- 类调度（内部使用，实例化类） --------------------------
    @classmethod
    def _dispatch_class(cls, class_key: str, init_dict: dict = None, *args, **kwargs):
        """
        内部方法：调度类（支持传入字典初始化，兼容原有参数格式）
        :param class_key: 类字符串标识
        :param init_dict: 类初始化参数字典（key 对应类的必需属性/初始化参数）
        :param args: 原有位置参数（可选，优先级低于 init_dict）
        :param kwargs: 原有关键字参数（可选，优先级：init_dict > kwargs）
        :return: 类的实例对象
        """
        if class_key not in cls.class_registry:
            raise KeyError(f"Unregistered method identifier: {class_key}")
            # 核心：整合参数字典与原有 kwargs，init_dict 优先级更高（覆盖重复 key）
        target_class = cls.class_registry.get(class_key)
        init_kwargs = {}
        # 1. 先加入原有 kwargs
        if kwargs:
            init_kwargs.update(kwargs)
        # 2. 再加入 init_dict（若存在，覆盖重复 key）
        if init_dict is not None and isinstance(init_dict, dict):
            init_kwargs.update(init_dict)

        # 3. 实例化：支持 args（位置参数） + init_kwargs（整合后的关键字参数）
        # 优先使用 init_kwargs（字典），也可保留 args 兼容原有逻辑
        try:
            return target_class(*args, **init_kwargs)
        except TypeError as e:
            raise TypeError("Class %s: instantiation failure, parameter mismatch, err msg: %s", target_class.__name__,
                            str(e))

    # -------------------------- 兼容版：统一调度入口（支持传/不传 target_type） --------------------------
    def dispatch(self, key: str, *args, target_type: str = None, init_dict: dict = None, **kwargs):
        """
        统一调度入口：
        1.  兼容「手动指定 target_type」和「自动识别」
        2.  类调度支持传入 init_dict（参数字典）完成实例化
        3.  兼容原有位置参数和关键字参数
        :param key: 方法/类的字符串标识
        :param target_type: 可选参数，None（自动识别）/"method"/"class"
        :param init_dict: 类初始化专属参数字典（仅对类调度生效）
        :param args: 方法执行/类实例化的位置参数
        :param kwargs: 方法执行/类实例化的关键字参数
        :return: 方法执行结果 或 类实例对象
        """
        # 场景1：手动传入 target_type（优先执行，明确无歧义）
        if target_type is not None:
            # 先校验 target_type 合法性
            all_target_types = [target.value for target in DispatcherTargetType]
            if target_type not in all_target_types:
                raise ValueError("Invalid dispatch type: %s', only supported target type: %s", target_type,
                                 all_target_types)
            # 按指定类型调度
            if target_type == DispatcherTargetType.METHOD.value:
                return self._dispatch_method(key, *args, **kwargs)
            elif target_type == DispatcherTargetType.CLASS.value:
                return self.__class__._dispatch_class(key, init_dict, *args, **kwargs)

        # 场景2：未传入 target_type（自动识别标识类型）
        else:
            # 检查标识在两个注册表中的存在情况
            is_in_method = key in self.method_registry
            is_in_class = key in self.class_registry

            # 子场景2.1：标识既在方法表，又在类表（理论上注册阶段已避免，此处做兜底）
            if is_in_method and is_in_class:
                raise ValueError("identifier: %s existed in both method registry and class registry, conflict exists. "
                                 "Please modify the identifier.", key)
            # 子场景2.2：仅在方法注册表（调度方法）
            elif is_in_method:
                return self._dispatch_method(key, *args, **kwargs)
            # 子场景2.3：仅在类注册表（调度类）
            elif is_in_class:
                return self.__class__._dispatch_class(key, init_dict, *args, **kwargs)
            # 子场景2.4：两边都不存在（未注册）
            else:
                raise ValueError("identifier: %s not registered, neither a method nor a class.", key)

    # -------------------------- 辅助方法：查看所有注册信息（便于调试） --------------------------
    @classmethod
    def show_registries(cls):
        """打印所有已注册的方法和类标识"""
        logger.info("\n" + "=" * 30)
        logger.info("List of registered method identifiers: ")
        if not cls.method_registry:
            logger.info("No registered methods")
        else:
            for idx, method_key in enumerate(cls.method_registry.keys(), 1):
                logger.info("  | %s | %s |", idx, method_key)

        logger.info("\nList of registered class identifiers：")
        if not cls.class_registry:
            logger.info("No registered classes")
        else:
            for idx, class_key in enumerate(cls.class_registry.keys(), 1):
                logger.info("  | %s | %s |", idx, class_key)
        logger.info("=" * 30 + "\n")
