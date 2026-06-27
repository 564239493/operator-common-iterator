# -*- coding: UTF-8 -*-
"""Session B — 结果分析器 & 提示词优化器。

读取 Session A 的执行结果（约束、用例、执行日志），独立 LLM session 分析根因。
若根因是约束提取问题，产出优化后的提示词；若是生成器/执行器自身 bug，标记终止。

**与 Session A 完全隔离** — 不共享任何对话上下文，仅通过文件传递数据。

Usage:
    python result_analyzer.py \
        --prompt prompts/operator_constraints_extract_v1.md \
        --doc docs/aclnnAlltoAllMatmul.md \
        --constraints iterations/iter_001/constraints.json \
        --cases iterations/iter_001/cases.json \
        --exec-result iterations/iter_001/execution_result.json \
        --output iterations/iter_001/analysis.json
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from backends import create_backend, LLMBackend

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# Session B 的 System Prompt
# ═══════════════════════════════════════════════════════════════════════════════

ANALYZER_SYSTEM_PROMPT = """# 算子约束提取提示词优化专家

你是一名**昇腾 CANN 算子约束提取提示词的优化专家**。你的任务是：
1. 分析算子约束提取 → 用例生成 → 用例执行的完整流水线结果
2. **精准诊断**失败根因
3. 若根因是提示词问题，**针对性修改提示词**

---

## 输入说明

你将收到以下信息：

### 1. 当前使用的约束提取提示词
一份长达数百行的 Markdown 提示词模板（目前是 v1 版本），用于引导 LLM 从算子文档中提取结构化约束。

### 2. 算子原始文档
CANN 算子的官方说明文档（Markdown 格式）。

### 3. 提取出的约束 JSON
使用上述提示词从算子文档中提取出的结构化约束数据。

### 4. 生成的测试用例
基于约束 JSON 生成的算子 ATK 测试用例列表。

### 5. 用例执行结果日志
用例在远程环境上的实际执行结果，包括：
- 通过/失败统计
- 失败用例的详情（失败原因、参数配置等）
- 运行日志

---

## 根因分类（三选一）

你必须将失败归因于以下**三类之一**：

### 类型 A：约束提取问题（constraint_extraction）
**特征**：提示词存在缺漏、歧义、错误，导致提取出的约束与算子文档实际要求**不一致**。

典型表现：
- 提示词未覆盖文档中**明确存在**的某类约束（如缺少对"空 tensor 约束"的提取规则）
- 提示词中的 schema 字段无法表达文档中的约束语义（如缺少 expr_type 枚举值）
- 提示词描述模糊导致 LLM 理解偏差（如对"可选参数"的定义与文档不一致）
- dtype / format / 平台名等受控字典不完整
- 跨参数约束未被正确提取（如文档写"x1 和 x2 的 dtype 必须一致"但约束中缺失）
- shape 约束提取遗漏（如文档写"shape 为 2 维"但约束里 dimensions 为空）

**行动**：**必须输出改进后的完整提示词**，修改策略：
- 保持提示词整体结构不变（10 章结构）
- **精准修改**存在缺漏/错误的章节
- 可新增 schema 字段 / expr_type 枚举 / 示例
- 修改点要能**追溯到具体失败用例**
- 不要大改——只改有问题的部分

### 类型 B：用例生成逻辑问题（generator_bug）
**特征**：约束提取正确，但生成的用例不符合约束，是生成器代码自身的问题。

典型表现：
- 约束 JSON 本身正确完整，但生成的用例参数值**超出**约束范围
- 生成器对 `allowed_range_value` / `dimensions` 的解释有 bug
- Z3 约束求解逻辑错误导致不可满足的用例
- 参数组合生成逻辑（pairwise / random）与约束冲突

**行动**：**不要修改提示词**。输出具体 bug 描述。

### 类型 C：执行逻辑/环境问题（executor_bug）
**特征**：约束正确、用例正确，但执行过程出现非预期的错误。

典型表现：
- SSH 连接问题、环境配置问题
- ATK 框架自身 bug
- 硬件资源不足 / 超时
- 用例依赖的 HCCL 集合通信库版本不兼容

**行动**：**不要修改提示词**。输出具体问题描述。

---

## 输出格式

你必须**严格**输出以下 JSON 结构（不要包含任何额外文字）：

```json
{
  "root_cause": "constraint_extraction | generator_bug | executor_bug",
  "analysis": "根本原因分析（300 字以内）",
  "specific_issues": [
    "具体问题描述 1（关联到失败用例 ID）",
    "具体问题描述 2"
  ],
  "modified_sections": ["被修改的章节名列表（仅 root_cause=constraint_extraction 时填写）"],
  "improved_prompt": "改进后的完整提示词（仅 root_cause=constraint_extraction 时需要，保持原结构，精准修改）",
  "generator_issue": "生成器 bug 描述（仅 root_cause=generator_bug 时填写）",
  "executor_issue": "执行器问题描述（仅 root_cause=executor_bug 时填写）"
}
```

---

## 分析原则

1. **从失败用例出发**：先看哪些用例失败了，失败原因是什么，反推是约束没提取对还是生成/执行有问题
2. **对照文档核对**：把失败用例的参数配置和算子文档的要求逐条对照
3. **保守修改提示词**：只改确实有问题的地方，不要"顺便优化"没问题的部分
4. **可追溯**：每个修改要能说清楚"因为哪个失败用例/哪条文档约束而改"
5. **保持通用性**：修改要考虑对其他算子的兼容性——不能为了当前算子把通用规则改坏"""


# ═══════════════════════════════════════════════════════════════════════════════
# 数据结构
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class AnalysisResult:
    """Session B 的分析结果。"""
    root_cause: str  # "constraint_extraction" | "generator_bug" | "executor_bug"
    analysis: str
    specific_issues: list[str] = field(default_factory=list)
    modified_sections: list[str] = field(default_factory=list)
    improved_prompt: str = ""
    generator_issue: str = ""
    executor_issue: str = ""

    @property
    def should_continue(self) -> bool:
        """是否应该继续迭代（约束提取问题）而不是终止。"""
        return self.root_cause == "constraint_extraction"

    @property
    def should_stop(self) -> bool:
        """是否应该终止迭代。"""
        return self.root_cause in ("generator_bug", "executor_bug")


def _extract_json_from_text(text: str) -> str:
    """从 LLM 输出中提取纯 JSON 字符串。"""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines)
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start:end + 1]
    return text


def _build_user_prompt(
    current_prompt: str,
    operator_doc: str,
    constraints_json: str,
    cases_json: str,
    exec_result_json: str,
) -> str:
    """构建 Session B 的分析 user prompt."""
    return f"""## 当前使用的约束提取提示词

```markdown
{current_prompt}
```

---

## 算子原始文档

```markdown
{operator_doc}
```

---

## 提取出的约束 JSON

```json
{constraints_json}
```

---

## 生成的测试用例

```json
{cases_json}
```

---

## 用例执行结果日志

```json
{exec_result_json}
```

---

请按照你的 System Prompt 中的规则进行分析，输出严格 JSON 结构。"""


def analyze_and_optimize(
    prompt_path: str | Path,
    doc_path: str | Path,
    constraints_path: str | Path,
    cases_path: str | Path,
    exec_result_path: str | Path,
    *,
    backend: LLMBackend | None = None,
    temperature: float = 0.3,
) -> AnalysisResult:
    """分析执行结果，判断根因并（如需要）优化提示词。

    这是一个**独立的 LLM session**——完全看不到 Session A 的对话内容，
    仅通过文件内容获取信息，实现"记忆隔离"。

    Args:
        prompt_path: 当前使用的提示词文件路径。
        doc_path: 算子文档文件路径。
        constraints_path: Session A 产出的约束 JSON 路径。
        cases_path: 生成的用例 JSON 路径。
        exec_result_path: 执行结果 JSON 路径。
        backend: LLM 后端实例（None 则默认用 APIBackend）。
        temperature: LLM temperature。

    Returns:
        AnalysisResult 对象。
    """
    if backend is None:
        backend = create_backend("api")

    prompt_path = Path(prompt_path)
    doc_path = Path(doc_path)
    constraints_path = Path(constraints_path)
    cases_path = Path(cases_path)
    exec_result_path = Path(exec_result_path)

    # 读取所有输入
    current_prompt = prompt_path.read_text(encoding="utf-8")
    operator_doc = doc_path.read_text(encoding="utf-8")

    constraints_json = constraints_path.read_text(encoding="utf-8") if constraints_path.exists() else "{}"
    cases_json = cases_path.read_text(encoding="utf-8") if cases_path.exists() else "[]"

    if exec_result_path.exists():
        exec_result_json = exec_result_path.read_text(encoding="utf-8")
    else:
        # 如果执行尚未完成，传空结果
        exec_result_json = json.dumps({
            "status": "not_executed",
            "error_message": "用例执行尚未完成或无结果",
            "passed": 0, "failed": 0, "records": [],
        }, ensure_ascii=False)

    user_prompt = _build_user_prompt(
        current_prompt=current_prompt,
        operator_doc=operator_doc,
        constraints_json=constraints_json,
        cases_json=cases_json,
        exec_result_json=exec_result_json,
    )

    logger.info("=== Session B: 分析 & 优化 LLM 调用开始 [backend=%s] ===", backend.name)
    logger.info("Prompt chars: %d | Doc chars: %d", len(current_prompt), len(operator_doc))

    try:
        response = backend.invoke(
            system_prompt=ANALYZER_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=temperature,
        )
        # backend.invoke returns raw text
        raw_output = response.strip()
    except Exception as e:
        logger.error("Session B LLM 调用失败: %s", e)
        return AnalysisResult(
            root_cause="executor_bug",
            analysis=f"分析 LLM 调用失败: {e}",
            executor_issue=str(e),
        )

    logger.info("=== Session B: LLM 返回完成 (len=%d) ===", len(raw_output))

    # ── 解析 JSON ──
    json_text = _extract_json_from_text(raw_output)

    try:
        result_dict = json.loads(json_text)
    except json.JSONDecodeError as e:
        logger.error("Session B JSON 解析失败: %s", e)
        logger.debug("Raw output (first 2000 chars):\n%s", raw_output[:2000])
        return AnalysisResult(
            root_cause="executor_bug",
            analysis=f"分析结果 JSON 解析失败: {e}",
            executor_issue=raw_output[:1000],
        )

    return AnalysisResult(
        root_cause=result_dict.get("root_cause", "executor_bug"),
        analysis=result_dict.get("analysis", ""),
        specific_issues=result_dict.get("specific_issues", []),
        modified_sections=result_dict.get("modified_sections", []),
        improved_prompt=result_dict.get("improved_prompt", ""),
        generator_issue=result_dict.get("generator_issue", ""),
        executor_issue=result_dict.get("executor_issue", ""),
    )


def main():
    parser = argparse.ArgumentParser(description="Session B: 结果分析 & 提示词优化")
    parser.add_argument("--prompt", required=True, help="当前提示词 MD 文件路径")
    parser.add_argument("--doc", required=True, help="算子文档 MD 文件路径")
    parser.add_argument("--constraints", required=True, help="Session A 产出的约束 JSON")
    parser.add_argument("--cases", required=True, help="生成的用例 JSON")
    parser.add_argument("--exec-result", required=True, help="执行结果 JSON")
    parser.add_argument("--output", required=True, help="分析结果输出 JSON 路径")
    parser.add_argument("--improved-prompt-output", default=None,
                        help="如产出改进提示词，保存到此路径")
    parser.add_argument("--model", default=None, help="覆盖默认模型")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    result = analyze_and_optimize(
        prompt_path=args.prompt,
        doc_path=args.doc,
        constraints_path=args.constraints,
        cases_path=args.cases,
        exec_result_path=args.exec_result,
        model=args.model,
    )

    # 保存分析结果
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({
        "root_cause": result.root_cause,
        "analysis": result.analysis,
        "specific_issues": result.specific_issues,
        "modified_sections": result.modified_sections,
        "generator_issue": result.generator_issue,
        "executor_issue": result.executor_issue,
        "should_continue": result.should_continue,
        "should_stop": result.should_stop,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("分析结果已保存到: %s", output_path)

    # 如果产出改进提示词，保存
    if result.improved_prompt and args.improved_prompt_output:
        improved_path = Path(args.improved_prompt_output)
        improved_path.parent.mkdir(parents=True, exist_ok=True)
        improved_path.write_text(result.improved_prompt, encoding="utf-8")
        logger.info("改进提示词已保存到: %s", improved_path)

    # 打印摘要
    print(f"\n{'='*60}")
    print(f"Session B 分析结果")
    print(f"{'='*60}")
    print(f"根因: {result.root_cause}")
    print(f"是否继续迭代: {result.should_continue}")
    print(f"是否终止: {result.should_stop}")
    print(f"\n分析摘要: {result.analysis}")
    if result.specific_issues:
        print(f"\n具体问题:")
        for issue in result.specific_issues:
            print(f"  - {issue}")
    if result.modified_sections:
        print(f"\n修改的章节: {', '.join(result.modified_sections)}")
    print(f"{'='*60}")

    if result.should_stop:
        sys.exit(2)  # 非 0 表示需要终止
    elif result.should_continue:
        sys.exit(0)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
