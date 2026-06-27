# -*- coding: UTF-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
"""
版权信息：华为技术有限公司，版本所有(C) 2025-2025
修改记录：2025/12/30 17:29
功能：算子参数主函数入口

该模块从 operator_case_generator 迁移而来，并按 operator-agent 的需要进行了改造：

- ``single_operator_handle`` 同时支持 **约束 JSON (dict)** 和 **约束文件路径** 两种入参形式，
  这样上层（节点 / 路由 / MCP 工具）可以直接传入 ``document_versions.json_constraints`` 解析后的
  Python 字典对象，无需在调用方做额外的文件读写。
- ``batch_operator_handel`` 保持原有的文件目录批处理语义不变。
- ``main`` 函数保留 CLI 入口能力，可独立运行。
"""
import argparse
import os.path
import time
from typing import List, Mapping, Union

from agent.generators.common_utils.data_handle_utils import DataHandleUtil
from agent.generators.common_utils.logger_util import LazyLogger, init_logger, DocumentLogContext
from agent.generators.common_model_definition import OperatorRule
from agent.generators.data_definition.constants import GlobalConfig
from agent.generators.data_definition.param_models_def import RunPlatform
from agent.generators.operator_param_combine.param_combination_generate import ParamCombinationGenerator
from agent.generators.operator_param_models.batch_case_generate import OperatorCaseGenerator
from pydantic import ValidationError

logger = LazyLogger()


def _build_constraint_data(operator_constraint: Union[str, os.PathLike, Mapping]) -> Union[OperatorRule, None]:
    """根据入参类型构造 ``OperatorRule`` 约束对象。

    Args:
        operator_constraint: 支持以下两种类型：
            * **str / os.PathLike** — 算子约束 JSON 文件路径
            * **Mapping (dict)** — 已解析的算子约束 JSON 字典

    Returns:
        构造好的 ``OperatorRule`` 对象；若读取或校验失败则返回 ``None``。
    """
    if isinstance(operator_constraint, (str, os.PathLike)):
        return DataHandleUtil.handle_operator_rule_data(operator_constraint)
    if isinstance(operator_constraint, Mapping):
        try:
            return OperatorRule(**dict(operator_constraint))
        except ValidationError as e:
            logger.error(
                f"Operator constraint data validation failed, err msg : {str(e)}")
            return None
    logger.error(
        f"Unsupported operator_constraint type : {type(operator_constraint).__name__}, "
        f"expected str / os.PathLike / Mapping")
    return None


def _resolve_operator_name(operator_constraint: Union[str, os.PathLike, Mapping],
                            operator_rule: OperatorRule) -> str:
    """从入参与已构造的 ``OperatorRule`` 中提取算子名。

    优先使用规则对象上的 ``operator_name`` 字段；若不存在则从文件路径 / 字典名中尝试推断。
    """
    if operator_rule is not None and getattr(operator_rule, "operator_name", ""):
        return operator_rule.operator_name
    if isinstance(operator_constraint, (str, os.PathLike)):
        return os.path.splitext(os.path.basename(os.fspath(operator_constraint)))[0]
    return ""


def single_operator_handle(operator_constraint, platform=RunPlatform.ATLAS_A3_TRAIN_AND_INFER_SERIES.value,
                           case_num=1) -> List:
    """
    算子用例生成的主入口。

    支持两种入参形式：

    1. **约束文件路径（兼容旧用法）**

       .. code-block:: python

           single_operator_handle("/path/to/aclnnFoo_constraints.json",
                                  platform="Atlas A3 训练系列产品/Atlas A3 推理系列产品",
                                  case_num=10)

    2. **约束 JSON 字典（推荐用法）**

       上层（节点 / 路由 / MCP）通常已经从 ``document_versions.json_constraints`` 字段读到
       Python 字典对象，此时可以直接传入而无需在调用方做额外的 IO：

       .. code-block:: python

           cases = single_operator_handle(constraints_dict, case_num=10)

    :param operator_constraint: 算子约束文件路径 *或* 约束 JSON 字典
    :param platform: 执行机对应的平台
        - "Atlas 推理系列产品" -> Platform_G2
        - "Atlas A3 训练系列产品" -> Platform_G1
    :param case_num: 生成用例个数
    :return: ``List[CaseConfig]``，已通过 inter-parameter 约束求解与修正
    """
    # 正式生成代码依赖 ``init_logger`` 初始化文件 logger，这里做一次惰性兜底。
    try:
        from agent.generators.common_utils.logger_util import init_logger as _init_logger, get_logger as _get_logger
        try:
            _get_logger()
        except RuntimeError:
            _init_logger(log_name="operator_generator", log_dir="./logs/generator")
    except Exception:  # pragma: no cover - 防呆
        pass

    operator_case_generate = OperatorCaseGenerator()
    operator_constraint_data = _build_constraint_data(operator_constraint)
    if operator_constraint_data is None:
        logger.error("Failed to build operator constraint data, abort generation")
        return []
    operator_name = _resolve_operator_name(operator_constraint, operator_constraint_data)
    logger.info(f"Start handle operator, operator name : {operator_name}")
    effective_operator_constraint_data = DataHandleUtil.select_effective_parameters(operator_constraint_data,
                                                                                     target_platform=platform)
    if effective_operator_constraint_data is None:
        logger.error(f"Effective operator rule data is None, operator name : {operator_name}")
        return []
    param_combination_generator = ParamCombinationGenerator(operator_rule_data=effective_operator_constraint_data,
                                                            case_num=case_num)
    param_combination_list = param_combination_generator.get_param_combination_input()
    case_list = operator_case_generate.handle_single_operator(
        operator_constraint_data=effective_operator_constraint_data,
        param_combination_list=param_combination_list, target_platform=platform,
        case_num=case_num)
    return case_list


def batch_operator_handel(operator_constraint_directory, operators: List = None,
                          platform=RunPlatform.ATLAS_A3_TRAIN_AND_INFER_SERIES.value,
                          case_save_path=None, case_num=1):
    """
    批量处理算子
    :param operator_constraint_directory: 算子约束文件目录
    :param operators: 需要生成用例的算子的名称
    :param platform: 执行机对应平台
    :param case_save_path: 用例保存路径
    :param case_num: 生成用例的个数
    :return: None
    """
    init_logger(log_name="main")
    data_handle_utils = DataHandleUtil()
    if not os.path.exists(operator_constraint_directory):
        raise FileNotFoundError("Operator constraint directory not existed")
    file_list = os.listdir(operator_constraint_directory)
    if operators is not None and len(operators) == 0:
        logger.error("Operators is empty, no operator need to be solve")
        return
    # 默认算子识别结果的文件为json文件
    operator_constraint_file_list = [file_name for file_name in file_list if file_name.endswith(".json")]
    operator_constraint_num = len(operator_constraint_file_list)
    logger.info(f"Start handle operator constraint, operator file num : {operator_constraint_num}")
    if case_save_path is None:
        case_save_path = GlobalConfig.CASE_RESULT_SAVE_PATH
    if not os.path.exists(case_save_path):
        os.makedirs(case_save_path)
    for index, file in enumerate(operator_constraint_file_list):
        operator_constraint_path = os.path.join(operator_constraint_directory, file)
        logger.info(
            f"Start handle operator, file index : {index + 1}/{operator_constraint_num}, "
            f"operator constraint path : {operator_constraint_path}")
        operator_name, _ = os.path.splitext(file)
        if operators is not None and operator_name not in operators:
            logger.error(f"Operators is not None and {operator_name} not in operators")
            continue
        time_str = time.strftime("%Y%m%d%H%M%S", time.localtime())
        with DocumentLogContext(f"{operator_name}_{time_str}"):
            case_list = single_operator_handle(operator_constraint_path, platform=platform,
                                               case_num=case_num)
            data_handle_utils.save_cases_to_json(api_name=operator_name, generate_case_list=case_list,
                                                 json_save_path=case_save_path)

        logger.info(
            f"End handle operator, file index : {index + 1}/{operator_constraint_num}, "
            f"operator constraint path : {operator_constraint_path}")
    logger.info(f"End handle operator constraint, operator file num : {operator_constraint_num}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--operator_constraint_directory", type=str, required=False, default=None,
                        help="Operator constraint json file directory")
    parser.add_argument("--operator_constraint_path", type=str, required=False, default=None,
                        help="Operator constraint json path")
    parser.add_argument("--platform", type=str, default=RunPlatform.ATLAS_A3_TRAIN_AND_INFER_SERIES.value,
                        required=False,
                        help="Test case executor environment platform")
    parser.add_argument("--case_save_path", type=str, default="output", required=False,
                        help="Case result json save path")
    parser.add_argument("--operators", type=str, default=None, required=False,
                        help="Name of the operator to be generated")
    parser.add_argument("--case_num", type=int, default=1, required=False,
                        help="Name of the operator to be generated")
    args = parser.parse_args()
    if args.operator_constraint_directory is None and args.operator_constraint_path is None:
        raise ValueError("operator_constraint_directory and operator_constraint_path cannot be empty at the same time")
    elif args.operator_constraint_directory is not None:
        operators = args.operators.split(",") if args.operators is not None else None
        batch_operator_handel(operator_constraint_directory=args.operator_constraint_directory, operators=operators,
                              platform=args.platform, case_save_path=args.case_save_path, case_num=args.case_num)
    else:
        operator_name, _ = os.path.splitext(os.path.basename(args.operator_constraint_path))
        time_str = time.strftime("%Y%m%d%H%M%S", time.localtime())
        init_logger(log_name=operator_name + "_" + time_str)
        case_list = single_operator_handle(operator_constraint_path=args.operator_constraint_path,
                                           platform=args.platform, case_num=args.case_num)
        DataHandleUtil.save_cases_to_json(api_name=operator_name, generate_case_list=case_list,
                                          json_save_path=args.case_save_path)


if __name__ == '__main__':
    main()
