"""Pydantic models for ATK remote execution results.

The :class:`ExecutionResult` is the canonical return shape of the
``exec_run_atk`` node — it captures the SSH command outcome plus the
structured ATK report data and log content, so the rest of the system
can serialize it into ``exec_results`` rows and SSE events uniformly.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ReportRecord(BaseModel):
    """A single row from the ATK xlsx report.

    The ATK report schema is loosely typed, so we keep the well-known
    fields explicit (id / run_result / failure_reason / case_json) and
    stash every other column under ``extra`` for forward compatibility.
    """

    id: str | None = Field(default=None, description="用例 ID (xlsx 中标识列).")
    run_result: str | None = Field(
        default=None, description="运行结果列 (pass / fail / skip / error 等).",
    )
    failure_reason: str | None = Field(default=None, description="失败原因列.")
    case_json: dict[str, Any] | None = Field(
        default=None, description="用例 JSON 信息列 (反序列化后的对象).",
    )
    extra: dict[str, Any] = Field(
        default_factory=dict, description="xlsx 中其它列的兜底存储.",
    )


class TaskReportData(BaseModel):
    """Structured data extracted from the ATK ``report/`` xlsx file."""

    report_path: str | None = Field(
        default=None, description="Remote xlsx 报告的本地缓存路径 (供前端下载).",
    )
    sheet_name: str | None = Field(default=None, description="读取的工作表名.")
    record_count: int = Field(default=0, description="解析出的报告记录数.")
    passed: int = Field(default=0, description="通过的用例数.")
    failed: int = Field(default=0, description="失败的用例数.")
    report_records: list[ReportRecord] = Field(
        default_factory=list, description="逐条解析出的报告记录.",
    )
    parse_error: str | None = Field(
        default=None,
        description="报告解析阶段的错误信息 (不影响主流程, 仅记录).",
    )


class ExecutionResult(BaseModel):
    """Canonical result object emitted by ``exec_run_atk``.

    ``status`` distinguishes three execution outcomes:

    * ``success`` — atk command exited 0, results extracted (or attempted)
    * ``failed`` — atk command exited non-zero (test failure, not infra error)
    * ``timeout`` — atk command exceeded the configured timeout
    * ``error`` — engine-level failure (SSH / SFTP / file IO); the node
      itself returns ``error=...`` in that case, but downstream
      consumers may also surface ``status="error"`` when they wrap an
      unhandled exception.
    """

    status: Literal["success", "failed", "timeout", "error"] = Field(
        default="success", description="ATK 命令的执行结果状态.",
    )
    exit_code: int | None = Field(
        default=None, description="远端 ``atk task`` 命令的退出码.",
    )
    stdout: str = Field(default="", description="远端命令的 stdout 截取.")
    stderr: str = Field(default="", description="远端命令的 stderr 截取.")
    duration: float = Field(
        default=0.0,
        description="整个执行阶段耗时 (秒, 含 SSH / 上传 / atk 命令 / 拉取).",
    )
    task_report_data: TaskReportData = Field(
        default_factory=TaskReportData, description="从 ATK report/ 目录提取的结构化结果.",
    )
    log_content: str = Field(default="", description="远端 ``log/atk.log`` 内容.")
    error_message: str | None = Field(
        default=None, description="错误描述 (仅在 status=error 或 failed 时填充).",
    )
    remote_output_dir: str | None = Field(
        default=None,
        description="远端 ATK 实际使用的输出目录 (含 operator_name 前缀).",
    )


__all__ = ["ExecutionResult", "TaskReportData", "ReportRecord"]