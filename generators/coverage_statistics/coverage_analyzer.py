"""
覆盖率统计入口 — coverage_analyzer.py

功能：
  基于算子的约束 JSON（定义参数属性可能取值）和生成的最终用例 JSON，
  计算并报告 1-pair 和 2-pair 覆盖率，执行过程使用本项目日志系统记录。

使用方式：
  单算子模式：
    python coverage_analyzer.py \
        --case_file ../data/output/aclnnFoo.json \
        --constraint_file ../data/input/rule/Foo.json

  批量模式（自动匹配目录下同名的 case JSON 和 constraint JSON）：
    python coverage_analyzer.py \
        --case_dir ../data/output \
        --constraint_dir ../data/input/rule \
        --format markdown \
        --output_dir ../data/coverage_reports

参数说明：
  --case_file      单个生成的 case JSON 文件路径
  --case_dir       case JSON 文件所在目录（批量模式）
  --constraint_file 单个算子约束 JSON 文件路径
  --constraint_dir  算子约束 JSON 文件所在目录（批量模式）
  --format         输出格式：console（默认）| markdown
  --output_dir     报告保存目录（可选）
"""

import argparse
import json
import os
import sys


# ============================================================
# 路径设置
# 将项目脚本目录加入 sys.path，使项目模块可导入
# ============================================================
PROJECT_SCRIPTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, PROJECT_SCRIPTS_DIR)

from agent.generators.common_utils.logger_util import LazyLogger, init_logger
from agent.generators.modeling_coverage.attribute_defs import build_domain_from_constraint
from agent.generators.modeling_coverage.case_parser import parse_case_file
from agent.generators.modeling_coverage.coverage_calculator import compute_coverage
from agent.generators.modeling_coverage.reporter import format_report, format_markdown_report

logger = LazyLogger()


# ============================================================
# 单算子覆盖率分析
# ============================================================

def run_single(case_path: str, constraint_path: str, output_format: str = "console") -> str:
    """
    对单个算子执行覆盖率分析。

    分析流程：
      1. 从算子约束 JSON 构建属性域定义（可能取值全集）
      2. 从生成的 case JSON 解析每个用例的参数属性值
      3. 计算 1-pair 和 2-pair 覆盖率
      4. 格式化为报告字符串

    Args:
        case_path: 最终生成的 case JSON 路径
        constraint_path: 算子约束 JSON 路径
        output_format: "console" 或 "markdown"

    Returns:
        str: 格式化后的覆盖率报告
    """
    # --- 文件存在性校验 ---
    if not os.path.exists(case_path):
        logger.error(f"Case file not found: {case_path}")
        return f"[ERROR] Case file not found: {case_path}"
    if not os.path.exists(constraint_path):
        logger.error(f"Constraint file not found: {constraint_path}")
        return f"[ERROR] Constraint file not found: {constraint_path}"

    logger.info(f"Starting coverage analysis: case={case_path}, constraint={constraint_path}")

    # --- 步骤 1: 构建属性域定义（分母）---
    logger.info("Step 1/4: Building attribute domain from constraint...")
    domain = build_domain_from_constraint(constraint_path)

    # --- 步骤 2: 解析用例属性值（分子）---
    logger.info("Step 2/4: Parsing generated cases...")
    case_records_list = parse_case_file(case_path)
    if not case_records_list:
        logger.warning(f"No valid cases found in {case_path}")
        return f"[WARN] No valid cases found in {case_path}"
    logger.info(f"  Total cases parsed: {len(case_records_list)}")

    # --- 步骤 3: 计算覆盖率 ---
    logger.info("Step 3/4: Computing coverage...")
    result = compute_coverage(case_records_list, domain)

    # --- 步骤 4: 格式化输出 ---
    logger.info("Step 4/4: Formatting report...")
    operator_name = domain.operator_name
    case_count = len(case_records_list)
    title = f"Coverage Report - {operator_name}"

    logger.info(f"Coverage analysis complete: operator={operator_name}, cases={case_count}")

    if output_format == "markdown":
        return format_markdown_report(result, domain, case_count, title)
    return format_report(result, domain, case_count, title)


# ============================================================
# 批量覆盖率分析
# ============================================================

def run_batch(case_dir: str, constraint_dir: str, output_format: str = "console", output_dir: str = None):
    """
    批量处理目录下的所有算子。

    自动匹配逻辑：
      - case_dir 下的每个 *.json 文件
      - 在 constraint_dir 中查找同名的约束 JSON 文件
      - 匹配成功则执行覆盖率分析
    """
    # --- 目录校验 ---
    if not os.path.exists(case_dir):
        logger.error(f"Case directory not found: {case_dir}")
        print(f"[ERROR] Case directory not found: {case_dir}")
        return
    if not os.path.exists(constraint_dir):
        logger.error(f"Constraint directory not found: {constraint_dir}")
        print(f"[ERROR] Constraint directory not found: {constraint_dir}")
        return

    # --- 获取 case 文件列表 ---
    case_files = [f for f in os.listdir(case_dir) if f.endswith(".json")]
    if not case_files:
        logger.warning(f"No JSON files found in {case_dir}")
        print(f"[WARN] No JSON files found in {case_dir}")
        return

    logger.info(f"Batch mode: {len(case_files)} case files found in {case_dir}")

    # --- 逐个处理 ---
    for case_file in sorted(case_files):
        case_path = os.path.join(case_dir, case_file)
        constraint_path = os.path.join(constraint_dir, case_file)

        # 约束文件不存在则跳过
        if not os.path.exists(constraint_path):
            logger.warning(f"No matching constraint file for {case_file}, skipped")
            print(f"[SKIP] No matching constraint file for {case_file}")
            continue

        logger.info(f"Processing: {case_file}")
        print(f"\n{'=' * 72}")
        print(f"  Processing: {case_file}")
        print(f"{'=' * 72}")

        report = run_single(case_path, constraint_path, output_format)

        # 保存到文件
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            ext = ".md" if output_format == "markdown" else ".txt"
            out_path = os.path.join(output_dir, case_file.replace(".json", f"_coverage{ext}"))
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(report)
            logger.info(f"Report saved: {out_path}")
            print(f"  Report saved: {out_path}")

        print(report)

    logger.info("Batch processing completed")


# ============================================================
# CLI 入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Coverage statistics for operator test cases",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python coverage_analyzer.py --case_file out.json --constraint_file rule.json
  python coverage_analyzer.py --case_dir ../data/output --constraint_dir ../data/input/rule --format markdown --output_dir ./reports
        """,
    )
    parser.add_argument("--case_file", type=str, default=None, help="Path to generated case JSON")
    parser.add_argument("--case_dir", type=str, default=None, help="Directory of case JSON files")
    parser.add_argument("--constraint_file", type=str, default=None, help="Path to operator constraint JSON")
    parser.add_argument("--constraint_dir", type=str, default=None, help="Directory of constraint JSON files")
    parser.add_argument("--format", type=str, default="console", choices=["console", "markdown"], help="Output format")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save reports")

    args = parser.parse_args()

    # 初始化日志系统（使用项目已有的日志机制）
    init_logger(log_name="coverage_analyzer", log_dir="./logs")
    logger.info("Coverage Analyzer started")

    # 模式选择：批量 or 单算子
    if args.case_dir and args.constraint_dir:
        logger.info(f"Mode: batch | case_dir={args.case_dir}, constraint_dir={args.constraint_dir}")
        run_batch(args.case_dir, args.constraint_dir, args.format, args.output_dir)
    elif args.case_file and args.constraint_file:
        logger.info(f"Mode: single | case_file={args.case_file}, constraint_file={args.constraint_file}")
        report = run_single(args.case_file, args.constraint_file, args.format)
        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
            base = os.path.splitext(os.path.basename(args.case_file))[0]
            ext = ".md" if args.format == "markdown" else ".txt"
            out_path = os.path.join(args.output_dir, f"{base}_coverage{ext}")
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(report)
            logger.info(f"Report saved: {out_path}")
            print(f"Report saved: {out_path}")
        print(report)
    else:
        parser.print_help()

    logger.info("Coverage Analyzer finished")


if __name__ == "__main__":
    main()
