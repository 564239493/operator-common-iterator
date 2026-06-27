import re

import os
import torch


class GlobalConfig:
    """全局配置"""
    # 默认配置文件路径
    CONFIG_FILES_BASE_PATH = ""
    # pict执行文件路径
    PICT_EXE_PATH = r"operator_param_combine/pict.exe"
    # 默认用例json数据保存路径
    CASE_RESULT_SAVE_PATH = r"../data/output"
    # 默认rule.json数据保存路径
    RULE_DATA_SAVE_PATH = r"../data/input/rule"
    # 参数语义角色识别保存路径
    PARAM_ROLE_RESULT_SAVE_PATH = r"../data/input/param_semantic_role"
    # 处理函数关键字标记
    PROCESS_FUNCTION_MARKER = "GetWorkspaceSize"
    # 默认生成用例数量
    DEFAULT_CASE_COUNT = 10
    # 单个用例最大生成尝试次数
    MAX_GENERATION_ATTEMPTS = 100
    # 解析参数语义角色定义文件中的模型类型标签，即global_role_definition.json
    ROLE_RULE_TYPE_KEY = "type"
    # 对于可选参数，以OPTIONAL_PARAM_PROBABILITY的概率选择是否为其赋值
    OPTIONAL_PARAM_PROBABILITY = 0.5
    # 每个用例使用Z3求解次数的上限，如果每个算子使用Z3求解次数的次数超出该上限，则终止求解过程，输出已有用例，并通知工作人员查看是否约束冲突导致无解
    Z3_SOLVE_TIME_LIMIT = 10
    # Z3表达式处理幂运算时，对于x ** 2类的场景，即底数为变量，指数为常量的情况，只能展开为for循环来处理，需要限制指数的最大值，避免消耗大量时间
    # 构建表达式，导致求解器卡死
    Z3_EXPONENT_MAX_VALUE = 3
    # 不区分平台时，代码通用平台的关键字
    COMMON_PLATFORM = "common"


class ParamModelConfig:
    """参数模型配置"""
    # shape模型配置文件路径 —— 相对于 ``agent.generators`` 包目录的绝对路径，
    # 避免上层进程的工作目录不同导致找不到 config。
    _GENERATORS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    SHAPE_DEFINITIONS_FILE_PATH = os.path.join(
        _GENERATORS_DIR, "configs", "shape_definitions.json"
    )
    # tensor填充值模型配置文件路径
    GLOBAL_ROLE_DEFINITIONS_PATH = os.path.join(
        _GENERATORS_DIR, "configs", "global_role_definitions.json"
    )
    # tensor shape中的每一维可以取得最大值
    TENSOR_SHAPE_MAX_VALUE = 65535
    # tensor shape中的每一维可以取到的最小值
    TENSOR_SHAPE_MIN_VALUE = 1
    # tensor中所有元素个数上限，即shape中所有元素的乘积
    TENSOR_TENSOR_ELEMENT_LIMIT = 100_000_000
    # tensor shape数据模型中用于区分高维和低维的边界，小于此值为低维矩阵，对应shape建模配置文件中的small_integers，
    # 否则为高维矩阵，对应配置文件中的high_dimensional_shape
    TENSOR_HIGH_DIMENSION_BOUNDARY = 4
    # 这里为啥取5
    RANDOM_GENERATE_SHAPE_TIME_LIMIT = 5
    # 当shape_definitions.json中没有对Has_Large_Size的策略定义时，此策略下shape可以取的最大值，仅在Has_Large_Size策略中使用
    DEFAULT_LARGE_FIX_DIM = 4096
    # 数据语义角色全集
    PARAM_ROLE_ONTOLOGY = [
        "role_data_generic",  # 1. 通用数据 (x1, x2, ...) -> 边界
        "role_data_positive_only",  # 2. 仅正数数据 (sqrt, log 的输入) -> > 0
        "role_scale_param",  # 3. 尺度参数 (gamma) -> 测 0, 1, Inf,随机扰动
        "role_bias_param",  # 4. 偏置参数 (beta) -> 测 0, Inf,-Inf,随机扰动
        "role_index_tensor",  # 5. 索引张量 (MoE routing, gather) -> 整数, [0, N)
        "role_quant_scale",  # 6. 量化尺度 (quant_scale) -> 正浮点数
        "role_quant_offset",  # 7. 量化偏移 (quant_offset/zp) -> 整数
        "role_epsilon_float",  # 8. 极小值 (epsilon) -> 1e-5, 0
        "role_shape_attribute",  # 9. 形状属性 (shape, stride, dim) -> 整数, 测 1, 奇数
        "role_unclassified",  # 非典型角色
    ]
    # 默认的数据语义角色，仅在参数数据语义角色缺失时使用
    DEFAULT_PARAM_ROLE = "role_data_generic"
    # 默认的tensor填充值模型，仅在输入的填充值模型缺失时使用
    DEFAULT_PARAM_RANGE_RULE = [{"type": "Static", "value": 0.0, "weight": 1.0}]
    # 默认的参数类型，只有在规则数据中未解析到参数type相关规则时才会使用
    DEFAULT_ATK_TYPE = "attr"
    # 默认的参数数据类型，只有在规则数据中未解析到dtype相关规则时才会使用
    DEFAULT_PARAM_DTYPE = "fp16"
    # 默认的参数数据类型，原始文档中的名称
    DEFAULT_PARAM_DTYPE_DTYPE_IN_ORIGINAL_DOC = "FLOAT16"
    # 默认的tensor维度值，只有在规则数据中未解析到dim相关的规则时才会使用
    DEFAULT_TENSOR_SHAPE_DIM = 1
    # 默认的tensor维度值最小值，只有在规则数据中未解析到dim相关的规则时才会使用
    DEFAULT_TENSOR_SHAPE_DIM_MIN = 1
    # 默认的tensor维度值最大值，只有在规则数据中未解析到dim相关的规则时才会使用
    DEFAULT_TENSOR_SHAPE_DIM_MAX = 8
    # 默认数组类参数的长度
    DEFAULT_LIST_LENGTH = 1
    # tensor shape每一维取值模型全集
    DIM_VALUE_PROFILE_LIST = ["Has_Large_Size", "Has_Size_1", "Has_Odd_Size", "Typical"]
    # 定义属于float, int, bool等可枚举的数据类型
    FLOAT_DTYPE = ["fp16", "fp32", "fp64", "bfp16", "bf16", "fp", "double"]
    INT_DTYPE = ["int", "int16", "int8", "int32", "int64", "uint8", "uint16", "uint32", "uint64"]
    BOOL_DTYPE = ["bool"]

    # 数值类参数以及bool类参数填充值模型全集
    FLOAT_TENSOR_DATA_PROFILE = ["Typical", "PosNormal", "NegNormal", "Zero", "One", "NaN", "PosInf", "NegInf",
                                 "SubNormal"]
    INT_TENSOR_DATA_PROFILE = ["Pos", "Neg", "Zero", "One", "Max", "Min"]
    BOOL_DATA_PROFILE = [True, False]
    # tensor类型参数数据type
    TENSOR_ATK_TYPE = ["tensor", "tensors"]
    # 数组类参数的数据类型
    LIST_ATK_TYPE = ["tensors", "scalars", "attrs"]
    # 用于识别参数组合后转换为参数名+属性名称的标题使用的pattern
    OPERATOR_NAME_RE_PATTERN = re.compile(
        r'^([a-zA-Z0-9]+)_(dim_count|dim_property|dtype|data_profile|memory|value|mode|param_type)$')
    # 严格应用人工规则修正的参数间约束类型
    # STRICT_CONSTRAINT_TYPE = ["shape_equality", "type_equality"]
    STRICT_CONSTRAINT_TYPE = []
    # 参数format可取值
    FORMAT_VALUE_LIST = ["nchw", "nhwc", "nc", "cn", "fractal_nz", "nchw16", "nchw8", "chwn8", "nhwc8", "nhwc16"]
    # 所有的tensor参数的shape中每一维的值都必须大于等于0，为shape属性的隐藏条件
    SHAPE_VALUE_MIN_EXPR = "all(d >= 0 for d in {}.shape)"
    # 参数模型定义中不涉及的参数value值
    NOT_RELEVANT_PARAM_VALUE = "N/A"
    # 暂不支持的数据类型
    UNSUPPORT_DTYPE = ["COMPLEX64", "COMPLEX128"]


class DataMatchMap:
    """配置各类数据映射字典"""
    # 在生成实际数据的时候使用，用于设置tensor中数值的数据类型
    TENSOR_DTYPE_TRANSFER_TORCH_MAP = {"ACL_FLOAT": torch.float, "ACL_FLOAT16": torch.float16, "ACL_INT8": torch.int8,
                                       "ACL_INT32": torch.int32, "ACL_UINT8": torch.uint8, "ACL_INT16": torch.int16,
                                       "ACL_UINT16": torch.uint16, "ACL_UINT32": torch.uint32, "ACL_INT64": torch.int64,
                                       "ACL_UINT64": torch.uint64, "ACL_DOUBLE": torch.float64, "ACL_BOOL": torch.bool,
                                       "ACL_STRING": str, "ACL_COMPLEX64": torch.complex64,
                                       "ACL_COMPLEX128": torch.complex128,
                                       "ACL_BF16": torch.bfloat16, "ACL_INT4": torch.int, "ACL_UINT1": torch.uint8,
                                       "ACL_COMPLEX32": torch.complex32, "FLOAT": torch.float32,
                                       "FLOAT16": torch.float16,
                                       "FLOAT32": torch.float32, "FLOAT64": torch.float64, "BFLOAT16": torch.bfloat16,
                                       "float32": torch.float32, "float16": torch.float16, "float64": torch.float64,
                                       "bfloat16": torch.bfloat16, "INT4": torch.int, "INT8": torch.int8,
                                       "INT16": torch.int16, "INT32": torch.int32, "UINT8": torch.uint8,
                                       "UINT16": torch.uint16, "UINT32": torch.uint32, "UINT64": torch.uint64,
                                       "INT64": torch.int64, "COMPLEX64": torch.complex64,
                                       "COMPLEX128": torch.complex128,
                                       "DOUBLE": torch.float64}

    # 在case_config中只生成数据生成方法字段，不生成实际数据时使用，用于适配ATK框架
    ACL_DTYPE_TRANSFER_TENSOR_MAP = {"INT4": "int", "INT8": "int8", "INT16": "int16", "INT32": "int32",
                                     "UINT8": "uint8", "INT": "int",
                                     "UINT16": "uint16", "UINT32": "uint32", "UINT64": "uint64", "INT64": "int64",
                                     "BFLOAT16": "bf16", "FLOAT16": "fp16", "FLOAT32": "fp32", "FLOAT64": "fp64",
                                     "float32": "fp32", "float16": "fp16", "float64": "fp64", "COMPLEX64": "complex64",
                                     "COMPLEX128": "complex128", "FLOAT": "fp32", "DOUBLE": "double", "char": "string",
                                     "ACL_FLOAT16": "fp16", "float": "fp32",
                                     "ACL_FLOAT32": "fp32", "ACL_FLOAT64": "fp64", "ACL_FLOAT": "fp32",
                                     "ACL_BF16": "bf16", "BOOL": "bool", "STRING": "string", "CHAR": "string",
                                     "string": "string", "bool": "bool", "double": "double", "int64_t": "int",
                                     "int64": "int"}

    # 如果type字段在ACL_TYPE_TRANSFER_ATK_MAP中，则转换为MAP中的值，否则默认为attr
    ACL_TYPE_TRANSFER_ATK_MAP = {"aclTensor": "tensor", "aclScalar": "scalar", "aclIntArray": "attrs",
                                 "aclFloatArray": ":attrs", "aclBoolArray": "attrs", "aclTensorList": "tensors",
                                 "aclScalarList": "scalars"}

    PLATFORM_MAP = {"Atlas A3 训练系列产品/Atlas A3 推理系列产品": "A3_train_series_or_A3_infer_series",
                    "Atlas A2 训练系列产品/Atlas A2 推理系列产品": "A2_train_series_or_A2_infer_series",
                    "Atlas 200I/500 A2 推理产品": "Atlas_200_or_A2_infer_product",
                    "Atlas 推理系列产品": "Atlas_infer_series", "Atlas 训练系列产品": "Atlas_train_series"}
    # 构建参数具体值到模型名称的映射，如果不在此映射表中，则根据role识别名称确定
    PARAM_VALUE_TO_ROLE_MODEL = {0: "Zero", 1: "One"}
    # 表达式中的关键词替换
    EXPR_KEYWORD_REPLACE = {"'nullptr'": None, }
    # 约束类型映射关系表
    CONSTRAINT_TYPE_MAP = {"shape": ["shape_equality", "shape_broadcast", "shape_choice", "shape_dependency"],
                           "dtype": ["type_equality", "type_dependency"],
                           "range_value": ["presence_dependency", "value_dependency"]}
    # 构建Z3变量的时候，初始化时，需要根据不同的变量类型以及数据类型，将ACL_TYPE_TRANSFER_ATK_MAP中的value值映射到Z3中定义的var_type,
    # 当前Z3的变量类型只支持tensor、scalar、list，对于scalar, list,.数据类型只支持：int、float、bool
    Z3_VAR_TYPE_MAP = {"tensor": "tensor", "tensors": "tensor", "scalar": "scalar", "scalars": "list", "attr": "scalar",
                       "attrs": "list"}
    # 根据参数数据类型设置可取值范围，防止未提供取值范围约束时，求解结果中的取值范围超出可表示上限
    DTYPE_SPECS = {
        "int": (-128, 127, True), "int8": (-128, 127, True), "uint8": (0, 255, True),
        "int16": (-32768, 32767, True), "int32": (-2 ** 31, 2 ** 31 - 1, True),
        "uint32": (0, 2 ** 32 - 1, True), "uint64": (0, 2 ** 64 - 1, True),
        "uint16": (0, 65535, True), "int64": (-2 ** 63, 2 ** 63 - 1, True), "bfp16": (-3.389e38, 3.389e38, False),
        "bf16": (-3.389e38, 3.389e38, False),
        "fp16": (-65504, 65504, False), "float": (-3.4e38, 3.4e38, False),
        "fp32": (-3.4e38, 3.4e38, False), "fp64": (-1.797e308, 1.797e308, False),
        "double": (-1.797e308, 1.797e308, False),
        "bool": (None, None, None), "string": (None, None, None)
    }
    # 表达式异常字符替换
    EXPR_ABNORMAL_KEYWORD_REPLACE = {
        'nullptr': None,
        'TRUE': True,
        'FALSE': False,
    }
