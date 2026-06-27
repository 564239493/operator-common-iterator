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
import time
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

    def __init__(
        self,
        model: str | None = None,
        chunk_idle_timeout: float | None = None,
        total_timeout: float | None = None,
    ):
        settings = get_settings()
        self._settings = settings
        self._model = model or settings.active_model
        self._base_url = settings.active_base_url
        self._api_key = settings.active_api_key.get_secret_value()
        self._provider = settings.llm_provider.value
        # 流式调用超时控制：
        # - chunk_idle_timeout: SSE 任意两个 chunk 之间允许的最大空闲间隔
        # - total_timeout: httpx 层的总读超时兜底
        self._chunk_idle_timeout = (
            chunk_idle_timeout if chunk_idle_timeout is not None
            else settings.llm_chunk_idle_timeout
        )
        self._total_timeout = (
            total_timeout if total_timeout is not None
            else settings.llm_total_timeout
        )
        # 进度日志间隔（秒）— 长响应避免每 20 chunk 打一行日志刷屏
        self._progress_log_interval = 60.0

    @property
    def name(self) -> str:
        return f"api({self._provider}:{self._model})"

    def _next_with_deadline(self, queue, deadline: float):
        """从 queue.Queue 取下一个 chunk；超过 deadline 仍没拿到则抛 RuntimeError。

        必须用后台线程把阻塞迭代器 (openai stream) 喂入 queue，主线程才能
        在 deadline 时主动 timeout — 同步 next() 无法被打断。
        """
        try:
            return queue.get(timeout=max(0.1, deadline - time.monotonic()))
        except Exception as e:
            # queue.Empty / queue.Full 都视为超时
            raise RuntimeError(
                f"LLM 流空闲超时 (chunk_idle_timeout={self._chunk_idle_timeout}s)"
            ) from e

    def invoke(self, system_prompt: str, user_prompt: str,
               temperature: float = 0.1) -> str:
        """直接调用 openai SDK 流式 API — 绕过 langchain 缓冲。

        关键实现：openai SDK 的 stream 在底层同步阻塞读 HTTP body，无法
        主动打断。因此我们把迭代逻辑放到后台线程，主线程通过 queue.Queue
        在 deadline 时主动超时。这是 SSE 卡死场景下唯一可靠的方案。

        Per-chunk idle 超时：chunk 之间超过 chunk_idle_timeout 没新数据就
        abort（默认 60s）；total_timeout 是 httpx 总读超时兜底（默认 600s）。
        """
        import queue
        import threading
        from openai import OpenAI

        client = OpenAI(
            api_key=self._api_key,
            base_url=self._base_url,
            timeout=self._total_timeout,
            max_retries=0,
        )

        logger.info(
            "APIBackend: 开始调用 LLM (model=%s, chunk_idle=%.0fs, total=%.0fs)",
            self._model, self._chunk_idle_timeout, self._total_timeout,
        )
        t_start = time.monotonic()

        chunk_count = 0
        chunk_queue: queue.Queue = queue.Queue(maxsize=1000)
        stream_error: list[Exception] = []  # 后台线程报错时塞这里

        def _producer():
            """后台线程：从 openai stream 读 chunk 喂入队列。"""
            try:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
                stream = client.chat.completions.create(
                    model=self._model,
                    messages=messages,
                    temperature=temperature,
                    stream=True,
                )
                for chunk in stream:
                    chunk_queue.put(chunk)
                chunk_queue.put(None)  # 哨兵 — 流结束
            except Exception as e:
                stream_error.append(e)
                chunk_queue.put(None)

        producer = threading.Thread(target=_producer, daemon=True)
        producer.start()

        try:
            content_parts: list[str] = []
            t_first_chunk: float | None = None
            t_last_chunk = t_start
            t_last_progress_log = t_start
            max_idle_gap = 0.0
            _next = self._next_with_deadline

            while True:
                # 首 chunk deadline 用 t_start；之后用 t_last_chunk
                deadline = (
                    t_start + self._chunk_idle_timeout
                    if chunk_count == 0
                    else t_last_chunk + self._chunk_idle_timeout
                )
                chunk = _next(chunk_queue, deadline)

                # 哨兵 = 流结束（None）或后台异常
                if chunk is None:
                    if stream_error:
                        raise stream_error[0]
                    break

                now = time.monotonic()
                gap = now - t_last_chunk
                if gap > max_idle_gap:
                    max_idle_gap = gap
                chunk_count += 1
                t_last_chunk = now

                if t_first_chunk is None:
                    t_first_chunk = now
                    logger.info(
                        "APIBackend: 收到第一个 chunk (耗时 %.2fs)",
                        t_first_chunk - t_start,
                    )
                elif now - t_last_progress_log >= self._progress_log_interval:
                    # 每隔 _progress_log_interval 秒打一次进度（默认 60s），
                    # 避免长响应产生过多 INFO 日志
                    logger.info(
                        "APIBackend: 已接收 %d chunks (累计 %.2fs, 距上一个 %.2fs)",
                        chunk_count, now - t_start, gap,
                    )
                    t_last_progress_log = now

                # 从 chunk 提取增量文本
                if chunk.choices:
                    delta = chunk.choices[0].delta
                    if delta and delta.content:
                        content_parts.append(delta.content)

            # ── 循环结束 ──
            if chunk_count == 0:
                elapsed = time.monotonic() - t_start
                logger.error(
                    "APIBackend: LLM 流返回 0 chunks (耗时 %.2fs)", elapsed,
                )
                raise RuntimeError(
                    f"LLM stream returned 0 chunks (model={self._model}, elapsed={elapsed:.1f}s)"
                )

            content = "".join(content_parts).strip()
            elapsed = time.monotonic() - t_start
            first_latency = (t_first_chunk - t_start) if t_first_chunk else 0.0
            logger.info(
                "APIBackend: LLM 流式返回完成 "
                "(total=%.2fs, first=%.2fs, chunks=%d, max_idle=%.2fs, len=%d)",
                elapsed, first_latency, chunk_count, max_idle_gap, len(content),
            )

            if not content:
                logger.warning("APIBackend: LLM 流式响应内容为空")

            return content
        except RuntimeError as e:
            elapsed = time.monotonic() - t_start
            if "LLM 流空闲超时" in str(e) or "returned 0 chunks" in str(e):
                logger.error(
                    "APIBackend: %s (chunk_idle=%.0fs, total=%.0fs, 已收 %d chunks, 耗时 %.1fs)",
                    str(e), self._chunk_idle_timeout, self._total_timeout,
                    chunk_count, elapsed,
                )
            raise
        except Exception as e:
            elapsed = time.monotonic() - t_start
            logger.error("APIBackend: LLM 调用失败 (耗时 %.1fs): %s", elapsed, e)
            raise


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
