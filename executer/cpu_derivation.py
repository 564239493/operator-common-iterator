"""Step 2 of ExecuterAgent: enhance ATK executor with CPU golden reference.

Uses LLM with ``aclnn-cpu-golden-derivation.md`` as skill context to update
the generated ``{operator_name}_atk_executor.py`` with CPU reference implementation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from langchain_openai import ChatOpenAI

from agent.core.config import settings
from agent.nodes.state import PipelineState

logger = logging.getLogger(__name__)

_RESOURCES_DIR = Path(__file__).resolve().parent / "resources"
_CPU_DERIVATION_SKILL = _RESOURCES_DIR / "aclnn-cpu-golden-derivation.md"

_SYSTEM_PROMPT = """\
你是一个昇腾算子测试专家。根据提供的算子文档、CPU Golden Reference 推导指南，\
修改更新 ATK 算子测试执行 py 文件，补充 cpu_golden_reference 函数的实现。

要求：
1. 仔细阅读算子文档，理解算子的功能、参数含义、数据类型要求和计算逻辑
2. 根据算子名称匹配推导指南中对应的 CPU 实现
3. 结合算子文档中的具体参数描述和计算规则，实现准确的 CPU 参考计算
4. 替换 cpu_golden_reference 中的 NotImplementedError 为实际实现
5. 保持文件其他部分不变
6. 只输出修改后的完整 Python 文件代码，不要包含 markdown 代码块标记
"""


async def exec_cpu_derivation_node(state: PipelineState) -> dict[str, Any]:
    """Enhance ATK executor with CPU golden reference via LLM."""
    if state.get("error"):
        return {"error": state.get("error")}

    operator_name = state.get("operator_name", "")
    executor_path = state.get("atk_executor_path")
    executor_code = state.get("atk_executor_code", "")
    operator_doc = state.get("content", "")

    if not executor_path or not executor_code:
        return {"error": "atk_executor_path or atk_executor_code missing"}

    logger.info("exec_cpu_derivation: enhancing CPU reference for %s", operator_name)

    try:
        skill_content = _CPU_DERIVATION_SKILL.read_text(encoding="utf-8")

        llm = ChatOpenAI(
            api_key=settings.active_api_key.get_secret_value(),
            base_url=settings.active_base_url,
            model=settings.active_model,
            temperature=0.3,
        )

        doc_section = ""
        if operator_doc:
            doc_section = f"## 算子文档\n{operator_doc}\n\n"

        user_prompt = (
            f"请根据以下算子文档和 aclnn-cpu-golden-derivation.md 推导指南，"
            f"帮我修改更新 ATK 的 {operator_name}_atk_executor.py 算子测试执行 py 文件 CPU后端执行算子部分的代码。\n\n"
            f"{doc_section}"
            f"## 推导指南\n{skill_content}\n\n"
            f"## 当前 ATK Executor 代码\n```python\n{executor_code}\n```\n\n"
            f"请输出修改后的完整 Python 代码。"
        )

        response = await llm.ainvoke([
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ])

        logger.debug("exec_cpu_derivation: LLM raw response:\n%s", response.content)

        updated_code = response.content.strip()
        if updated_code.startswith("```"):
            lines = updated_code.splitlines()
            lines = [l for l in lines if not l.startswith("```")]
            updated_code = "\n".join(lines)

        Path(executor_path).write_text(updated_code, encoding="utf-8")

        logger.info("exec_cpu_derivation: updated %s", executor_path)
        return {
            "atk_executor_code": updated_code,
            "error": None,
        }
    except Exception as e:
        logger.exception("exec_cpu_derivation failed for %s", operator_name)
        return {"error": str(e)}
