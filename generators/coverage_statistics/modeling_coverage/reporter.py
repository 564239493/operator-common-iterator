"""
覆盖率报告格式化输出模块

功能：将 CoverageResult 格式化为可读的文本报告。
      支持两种输出格式：
        1. console 格式：终端直接打印，含进度条形图
        2. markdown 格式：适合写入文件或 CI 展示

报告中包含：
  - 基本统计：算子名称、用例数
  - 1-pair 覆盖：每个参数每个属性的逐项覆盖率
  - 2-pair 覆盖：跨参数同属性配对的逐项覆盖率
  - 平均覆盖率汇总
"""

from typing import Dict, List

from .coverage_calculator import CoverageResult
from .attribute_defs import OperatorAttributeDomain


# ============================================================
# 辅助函数
# ============================================================

def _rate_bar(rate: float, width: int = 20) -> str:
    """
    生成进度条形图。

    用 20 个 Unicode 方块字符直观展示覆盖率：
      - █ (filled block): 已覆盖部分
      - ░ (shade block):  未覆盖部分

    例如 75% 覆盖率：
      ███████████████░░░░░
    """
    filled = int(rate * width)
    bar = "█" * filled + "░" * (width - filled)
    return bar


def _pct(rate: float) -> str:
    """将小数格式的覆盖率转为百分比字符串，保留两位小数。"""
    return f"{rate * 100:6.2f}%"


# ============================================================
# Console 格式报告
# ============================================================

def format_report(
    result: CoverageResult,
    domain: OperatorAttributeDomain,
    case_count: int,
    title: str = "Coverage Report",
) -> str:
    """
    生成 console 格式的覆盖率报告。

    布局：
      1. 标题区（算子名称、用例数）
      2. 1-pair 覆盖表（参数/属性/已覆盖/可能/覆盖率/进度条）
      3. 平均 1-pair 覆盖率
      4. 2-pair 覆盖表（属性对/已覆盖/可能/覆盖率/进度条）
      5. 平均 2-pair 覆盖率
    """
    lines = []
    sep = "=" * 72
    lines.append(sep)
    lines.append(f"  {title}")
    lines.append(f"  算子: {domain.operator_name}  |  用例数: {case_count}")
    lines.append(sep)
    lines.append("")

    # ----------------------------------------------------------
    # 1-pair 覆盖部分
    # ----------------------------------------------------------
    lines.append("─" * 72)
    lines.append("  一、1-pair 覆盖（每参数每属性取值覆盖）")
    lines.append("─" * 72)
    lines.append("")

    header = f"{'参数':<24} {'属性':<22} {'已覆盖/可能':<14} {'覆盖率':<10} 分布"
    lines.append(header)
    lines.append("─" * len(header))

    all_one_rates = []
    for param_name in sorted(result.one_pair.keys()):
        param_result = result.one_pair[param_name]
        first_attr = True

        for attr_name in sorted(param_result.keys()):
            info = param_result[attr_name]
            label = param_name if first_attr else ""
            display = (
                f"{label:<24} {attr_name:<22} "
                f"{info['count']:>3}/{info['total']:<8} "
                f"{_pct(info['rate']):<10} {_rate_bar(info['rate'])}"
            )
            lines.append(display)
            first_attr = False
            all_one_rates.append(info["rate"])

        # 参数无任何可枚举属性时的占位
        if first_attr:
            lines.append(f"{param_name:<24} {'(无可枚举属性)':<22}")

    # 平均 1-pair 覆盖率
    if all_one_rates:
        avg = sum(all_one_rates) / len(all_one_rates)
        lines.append("")
        lines.append(f"{'平均 1-pair 覆盖率':>50}  {_pct(avg)}  {_rate_bar(avg)}")

    lines.append("")

    # ----------------------------------------------------------
    # 2-pair 覆盖部分
    # ----------------------------------------------------------
    lines.append("─" * 72)
    lines.append("  二、2-pair 覆盖（跨参数同属性类型配对覆盖）")
    lines.append("─" * 72)
    lines.append("")

    header2 = f"{'属性对':<50} {'已覆盖/可能':<14} {'覆盖率':<10} 分布"
    lines.append(header2)
    lines.append("─" * len(header2))

    all_two_rates = []
    for key in sorted(result.two_pair.keys()):
        info = result.two_pair[key]
        display = (
            f"{key:<50} "
            f"{info['count']:>3}/{info['total']:<8} "
            f"{_pct(info['rate']):<10} {_rate_bar(info['rate'])}"
        )
        lines.append(display)
        all_two_rates.append(info["rate"])

    # 平均 2-pair 覆盖率
    if all_two_rates:
        avg = sum(all_two_rates) / len(all_two_rates)
        lines.append("")
        lines.append(f"{'平均 2-pair 覆盖率':>67}  {_pct(avg)}  {_rate_bar(avg)}")

    lines.append("")
    lines.append(sep)

    return "\n".join(lines)


# ============================================================
# Markdown 格式报告
# ============================================================

def format_markdown_report(
    result: CoverageResult,
    domain: OperatorAttributeDomain,
    case_count: int,
    title: str = "Coverage Report",
) -> str:
    """
    生成 Markdown 格式的覆盖率报告。

    适合：
      - 保存为 .md 文件在 Git 仓库中追踪覆盖率变化
      - 在 CI/CD 的 Pipeline 中作为 Artifact 展示
      - 在 Merge Request 中作为评论输出
    """
    lines = []
    lines.append(f"# {title}")
    lines.append("")
    lines.append(f"- **算子**: {domain.operator_name}")
    lines.append(f"- **用例数**: {case_count}")
    lines.append("")

    # ----------------------------------------------------------
    # 1-pair 覆盖表格
    # ----------------------------------------------------------
    lines.append("## 一、1-pair 覆盖（每参数每属性取值覆盖）")
    lines.append("")

    rows = []
    all_one_rates = []
    for param_name in sorted(result.one_pair.keys()):
        param_result = result.one_pair[param_name]
        for attr_name in sorted(param_result.keys()):
            info = param_result[attr_name]
            rows.append((param_name, attr_name, info["count"], info["total"], info["rate"]))
            all_one_rates.append(info["rate"])

    if rows:
        lines.append("| 参数 | 属性 | 已覆盖 | 可能取值 | 覆盖率 |")
        lines.append("|------|------|-------:|--------:|------:|")
        for p, a, c, t, r in rows:
            lines.append(f"| {p} | {a} | {c} | {t} | {_pct(r)} |")
        lines.append("")
        avg = sum(all_one_rates) / len(all_one_rates)
        lines.append(f"**平均 1-pair 覆盖率**: {_pct(avg)}")
    else:
        lines.append("_无可用数据_")

    lines.append("")

    # ----------------------------------------------------------
    # 2-pair 覆盖表格
    # ----------------------------------------------------------
    lines.append("## 二、2-pair 覆盖（跨参数同属性类型配对覆盖）")
    lines.append("")

    rows2 = []
    all_two_rates = []
    for key in sorted(result.two_pair.keys()):
        info = result.two_pair[key]
        rows2.append((key, info["count"], info["total"], info["rate"]))
        all_two_rates.append(info["rate"])

    if rows2:
        lines.append("| 属性对 | 已覆盖 | 可能组合 | 覆盖率 |")
        lines.append("|--------|-------:|--------:|------:|")
        for k, c, t, r in rows2:
            lines.append(f"| {k} | {c} | {t} | {_pct(r)} |")
        lines.append("")
        avg = sum(all_two_rates) / len(all_two_rates)
        lines.append(f"**平均 2-pair 覆盖率**: {_pct(avg)}")
    else:
        lines.append("_无可用数据_")

    lines.append("")
    return "\n".join(lines)
