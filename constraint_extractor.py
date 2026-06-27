# -*- coding: UTF-8 -*-
"""Session A — 约束提取器。

使用通用提示词 + 算子文档 → 调用 LLM 提取结构化约束 JSON。
每次调用是独立的 LLM session，不与分析 session 共享上下文。

支持自动重试：Pydantic 校验失败时，将错误反馈给 LLM 重试（最多 3 次）。

Usage:
    python constraint_extractor.py \
        --prompt prompts/operator_constraints_extract_v1.md \
        --doc docs/aclnnAlltoAllMatmul.md \
        --output iterations/iter_001/constraints.json \
        --operator-name aclnnAlltoAllMatmul
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Optional

from pydantic import ValidationError

from backends import create_backend, LLMBackend

logger = logging.getLogger(__name__)

# 最大重试次数（含首次，即最多 3 次 LLM 调用）
MAX_RETRIES = 3


def _strip_markdown_code_block(text: str) -> str:
    """移除 LLM 输出中可能包裹的 ```json ... ``` 标记。"""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines)
    return text.strip()


def _extract_json_from_text(text: str) -> str:
    """从 LLM 输出中提取纯 JSON 字符串。"""
    text = _strip_markdown_code_block(text)
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start:end + 1]
    return text


def _validate_against_schema(constraints: dict) -> str:
    """用 generators/common_model_definition.py 的 OperatorRule 做 Pydantic 校验。

    Returns:
        空字符串表示校验通过；否则返回格式化的错误信息。
    """
    try:
        from generators.common_model_definition import OperatorRule
        OperatorRule(**dict(constraints))
        return ""
    except ImportError:
        logger.warning("无法导入 OperatorRule，跳过校验")
        return ""
    except ValidationError as e:
        return str(e)


def _build_user_prompt(operator_name: str, operator_doc: str) -> str:
    """构建初始 user prompt。"""
    return f"""请从下列算子说明文档中提取约束。

## 算子名称
{operator_name}

## 算子说明文档（已转换为 Markdown）
```markdown
{operator_doc}
```

## 你的任务
1. 完整阅读算子说明文档；
2. 按《算子约束提取通用提示词》第 3 章 schema 输出 JSON；
3. 内部执行自检清单（第 9 章 7 项）；
4. **仅返回 JSON 字符串**，不要包含任何解释、代码块标记或额外文字。"""


def _build_retry_prompt(operator_name: str, operator_doc: str, previous_json: str,
                         validation_error: str, attempt: int) -> str:
    """构建重试 user prompt，包含上次的 JSON 和校验错误。"""
    return f"""## 上次提取结果存在格式错误，请修正后重新输出

### 上次输出的 JSON（有错误）
```json
{previous_json}
```

### Pydantic 校验错误
```
{validation_error}
```

### 修正要求
1. **只修正上述错误**——不要改变已经正确的部分
2. 严格遵循提示词第 3 章的 JSON Schema
3. 确保字段名、类型、层级结构与 schema **完全一致**
4. 字段中不允许出现 schema 未定义的 key（extra="forbid"）
5. **仅返回修正后的 JSON 字符串**，无任何额外内容

---

## 算子名称
{operator_name}

## 算子说明文档（已转换为 Markdown）
```markdown
{operator_doc}
```"""


def extract_constraints(
    prompt_path: str | Path,
    doc_path: str | Path,
    operator_name: str = "",
    *,
    backend: LLMBackend | None = None,
    temperature: float = 0.1,
    max_retries: int = MAX_RETRIES,
) -> tuple[dict | None, str, int]:
    """从算子文档中提取结构化约束，自动重试直到 Pydantic 校验通过。

    Args:
        prompt_path: 通用提示词 Markdown 文件路径。
        doc_path: 算子文档 Markdown 文件路径。
        operator_name: 算子名称（为空时从 doc 文件名推断）。
        backend: LLM 后端实例（None 则默认用 APIBackend）。
        temperature: LLM temperature。
        max_retries: 最多 LLM 调用次数（默认 3）。

    Returns:
        (constraints_dict, raw_output, attempts) —
        constraints_dict 为 None 表示全部重试失败；
        raw_output 是最后一次 LLM 的原始输出；
        attempts 是实际 LLM 调用次数。
    """
    if backend is None:
        backend = create_backend("api")

    prompt_path = Path(prompt_path)
    doc_path = Path(doc_path)

    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
    if not doc_path.exists():
        raise FileNotFoundError(f"Operator doc not found: {doc_path}")

    prompt_template = prompt_path.read_text(encoding="utf-8")
    operator_doc = doc_path.read_text(encoding="utf-8")

    if not operator_name:
        operator_name = doc_path.stem

    raw_output = ""
    constraints: dict | None = None
    last_error = ""

    for attempt in range(1, max_retries + 1):
        logger.info("=== Session A: 约束提取 第 %d/%d 次 LLM 调用 [backend=%s] ===",
                    attempt, max_retries, backend.name)
        logger.info("Operator: %s | Prompt: %s", operator_name, prompt_path.name)

        # 构建 prompt（首次和重试不同）
        if attempt == 1:
            user_prompt = _build_user_prompt(operator_name, operator_doc)
        else:
            user_prompt = _build_retry_prompt(
                operator_name, operator_doc,
                json.dumps(constraints, ensure_ascii=False, indent=2)[:8000],
                last_error, attempt,
            )

        # ── 调用 LLM（通过 backend，每次是独立 session）──
        logger.info(">>> 开始调用 backend.invoke() [attempt=%d]", attempt)
        try:
            raw_output = backend.invoke(
                system_prompt=prompt_template,
                user_prompt=user_prompt,
                temperature=temperature,
            )
        except Exception as e:
            logger.error("LLM 调用失败 (attempt=%d): %s", attempt, e)
            return None, str(e), attempt

        logger.info("第 %d 次 LLM 返回完成 (len=%d)", attempt, len(raw_output))

        # ── 解析 JSON ──
        json_text = _extract_json_from_text(raw_output)
        try:
            constraints = json.loads(json_text)
        except json.JSONDecodeError as e:
            logger.error("第 %d 次 JSON 解析失败: %s", attempt, e)
            last_error = f"JSON 解析错误: {e}"
            if attempt < max_retries:
                logger.info("将在第 %d 次重试中反馈此错误", attempt + 1)
            continue

        # ── Pydantic 校验（阻塞）──
        validation_error = _validate_against_schema(constraints)
        if not validation_error:
            logger.info("Pydantic 校验通过 (第 %d 次尝试)", attempt)
            return constraints, raw_output, attempt
        else:
            logger.warning("第 %d 次 Pydantic 校验失败: %s", attempt, validation_error[:500])
            last_error = validation_error
            if attempt < max_retries:
                logger.info("将在第 %d 次重试中反馈校验错误", attempt + 1)

    # 全部重试失败
    logger.error("全部 %d 次尝试均未通过 Pydantic 校验", max_retries)
    return None, raw_output, max_retries


def main():
    parser = argparse.ArgumentParser(description="Session A: 算子约束提取")
    parser.add_argument("--prompt", required=True, help="通用提示词 MD 文件路径")
    parser.add_argument("--doc", required=True, help="算子文档 MD 文件路径")
    parser.add_argument("--output", required=True, help="约束 JSON 输出路径")
    parser.add_argument("--operator-name", default="", help="算子名称（可选）")
    parser.add_argument("--model", default=None, help="覆盖默认模型")
    parser.add_argument("--temperature", type=float, default=0.1, help="LLM temperature")
    parser.add_argument("--max-retries", type=int, default=MAX_RETRIES,
                        help=f"最大 LLM 调用次数（默认 {MAX_RETRIES}）")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    constraints, raw_output, attempts = extract_constraints(
        prompt_path=args.prompt,
        doc_path=args.doc,
        operator_name=args.operator_name,
        model=args.model,
        temperature=args.temperature,
        max_retries=args.max_retries,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if constraints is not None:
        output_path.write_text(json.dumps(constraints, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("约束 JSON 已保存到: %s (用时 %d 次 LLM 调用)", output_path, attempts)
    else:
        debug_path = output_path.with_suffix(".raw.txt")
        debug_path.write_text(raw_output, encoding="utf-8")
        logger.error("全部 %d 次重试失败，原始输出已保存到: %s", attempts, debug_path)
        sys.exit(1)


if __name__ == "__main__":
    main()
