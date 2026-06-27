# -*- coding: UTF-8 -*-
"""LLM Backend 抽象层。

支持两种模式，每次调用都是**独立的无状态 session**：
- APIBackend: 通过 langchain_openai 调用 DeepSeek / Z.AI / OpenAI 等
- ClaudeBackend: 通过 anthropic SDK 直接调用 Claude API

Usage:
    from backends import create_backend

    backend = create_backend("api")      # DeepSeek API
    backend = create_backend("claude")   # Anthropic Claude API

    result = backend.invoke(system_prompt, user_prompt)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from config import get_settings, LLMProvider

logger = logging.getLogger(__name__)


class LLMBackend(ABC):
    """LLM 调用后端抽象。每次 invoke() 是独立的无状态 session。"""

    @abstractmethod
    def invoke(self, system_prompt: str, user_prompt: str,
               temperature: float = 0.1) -> str:
        """发送 system + user prompt，返回 LLM 原始文本输出。"""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """后端名称（用于日志）。"""
        ...


# ═══════════════════════════════════════════════════════════════════════════════
# APIBackend — 通过 langchain_openai 调用（DeepSeek / Z.AI / OpenAI 兼容）
# ═══════════════════════════════════════════════════════════════════════════════

class APIBackend(LLMBackend):
    """基于 langchain_openai.ChatOpenAI 的 API 后端。

    支持所有 OpenAI 兼容 API：DeepSeek、Z.AI、OpenAI、MiniMax 等。
    配置从 .env 读取，与 operator-agent 解耦。
    """

    def __init__(self, model: str | None = None):
        settings = get_settings()
        self._settings = settings
        self._model = model or settings.active_model
        self._base_url = settings.active_base_url
        self._api_key = settings.active_api_key.get_secret_value()
        self._provider = settings.llm_provider.value

    @property
    def name(self) -> str:
        return f"api({self._provider}:{self._model})"

    def invoke(self, system_prompt: str, user_prompt: str,
               temperature: float = 0.1) -> str:
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(
            api_key=self._api_key,
            base_url=self._base_url,
            model=self._model,
            temperature=temperature,
        )
        response = llm.invoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ])
        return response.content.strip()


# ═══════════════════════════════════════════════════════════════════════════════
# CliAgentBackend — 通过子进程调用 CLI 命令行工具（claude / opencode 等）
# ═══════════════════════════════════════════════════════════════════════════════

class CliAgentBackend(LLMBackend):
    """通过 subprocess 调用 CLI Agent 工具（claude / opencode CLI）。

    每次 invoke() 启动一个新的子进程 —— 天然 session 隔离。
    不需要任何 API key，依赖本地安装的 CLI 工具。

    配置项（.env）:
        CLI_AGENT_BIN: CLI 可执行文件路径或命令名（默认 "claude"）
        CLI_AGENT_ARGS: 额外参数模板，{prompt} 会被替换为实际 prompt 拼接
                        （默认 "-p {prompt} --print --output-format text"）
    """

    def __init__(self, model: str | None = None):
        settings = get_settings()
        self._bin = settings.cli_agent_bin
        self._args_template = settings.cli_agent_args
        # model 在当前 CLI 模式下通常不需要（由 CLI 自身决定），保留接口一致性
        self._model = model or ""

    @property
    def name(self) -> str:
        return f"cli_agent({self._bin})"

    def invoke(self, system_prompt: str, user_prompt: str,
               temperature: float = 0.1) -> str:
        import subprocess
        import tempfile

        # 将 system + user prompt 合并为一个完整 prompt
        full_prompt = f"{system_prompt}\n\n---\n\n{user_prompt}"

        # 限制长度避免命令行过大（超过 128KB 则写临时文件）
        if len(full_prompt) > 120_000:
            return self._invoke_via_file(full_prompt, temperature)

        # 构造命令行
        cmd = self._build_command(full_prompt)

        logger.info("CliAgent: spawning %s", self._bin)
        logger.debug("Cmd: %s", " ".join(cmd[:4]) + " ...")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=600,  # 10 分钟超时
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"CLI agent 超时 (600s): {self._bin}")
        except FileNotFoundError:
            raise RuntimeError(
                f"CLI agent 未找到: {self._bin}。"
                f"请确认已安装并在 PATH 中，或设置 CLI_AGENT_BIN 为完整路径。"
            )

        if result.returncode != 0:
            logger.warning("CLI agent 返回非零退出码 %d", result.returncode)
            if result.stderr:
                logger.warning("Stderr: %s", result.stderr[:500])

        output = result.stdout.strip()
        if not output and result.stderr:
            # 有些 CLI 工具把结果输出到 stderr
            output = result.stderr.strip()

        return output

    def _invoke_via_file(self, prompt: str, temperature: float) -> str:
        """超长 prompt 走临时文件传递。"""
        import subprocess
        import tempfile
        from pathlib import Path

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", encoding="utf-8", delete=False,
        ) as f:
            f.write(prompt)
            tmp_path = f.name

        try:
            cmd = self._build_command(f"@file:{tmp_path}")
            result = subprocess.run(
                cmd, capture_output=True, text=True,
                encoding="utf-8", errors="replace", timeout=600,
            )
            return result.stdout.strip() or result.stderr.strip()
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def _build_command(self, prompt: str) -> list[str]:
        """构建 CLI 命令行。"""
        # 用双引号包裹 prompt
        escaped = prompt.replace('"', '\\"')
        args = self._args_template.replace("{prompt}", f'"{escaped}"')
        return [self._bin] + args.split()


# ═══════════════════════════════════════════════════════════════════════════════
# 工厂函数
# ═══════════════════════════════════════════════════════════════════════════════

def create_backend(backend_type: str = "api", model: str | None = None) -> LLMBackend:
    """创建 LLM 后端实例。

    Args:
        backend_type:
            - "api":  通过 langchain_openai 调用 LLM API（DeepSeek / Z.AI）
            - "agent": 通过 subprocess 调用 CLI 工具（claude / opencode CLI）
        model: 覆盖 .env 中的模型名（仅 api 模式有效）。

    Raises:
        ValueError: 不支持的 backend_type。
    """
    backend_type = backend_type.lower().strip()

    if backend_type == "api":
        return APIBackend(model=model)

    if backend_type == "agent":
        return CliAgentBackend(model=model)

    raise ValueError(
        f"Unsupported backend type: '{backend_type}'. "
        f"Choose 'api' or 'agent'."
    )
