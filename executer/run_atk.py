"""Step 3 of ExecuterAgent: run the generated ATK executor on a remote host.

The flow:

1. Open an SSH connection to the operator-selected server
   (``state["server_info"]``).  Engine-level failures (TCP, auth, SFTP)
   short-circuit the pipeline with ``state["error"]`` set.

2. SFTP-upload the cases JSON and the ATK executor script to the
   canonical remote locations:
   - ``/home/operator_atk/cases/{operator_name}_cases.json``
   - ``/home/operator_atk/atk_executor/{operator_name}_executor.py``

3. Run the ``atk node --backend cpu task`` command via
   ``source <env_init> && ...`` so the CANN environment is loaded before
   atk starts.  The command's exit code decides ``status``
   (``success`` / ``failed`` / ``timeout``).

4. Find the latest ``<operator_name>_*`` output directory under
   ``/home/operator_atk/atk_output``, download the ``report/`` xlsx and
   ``log/atk.log``, parse them into an :class:`ExecutionResult`.

5. Return the result as a dict (so it slots into the existing
   ``state["exec_result"]`` field via the LangGraph reducer).

Result extraction never aborts the main flow — parsing failures are
captured into ``ExecutionResult.task_report_data.parse_error`` /
``ExecutionResult.error_message``.  Only engine-level SSH / SFTP / IO
failures surface as ``state["error"]``.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from agent.nodes.executer_subgraph.execution_result import ExecutionResult
from agent.nodes.executer_subgraph.report_parser import (
    collect_remote_artifacts,
    make_local_cache_dir,
)
from agent.nodes.executer_subgraph.ssh_executor import (
    CommandResult,
    ServerEndpoint,
    SSHEngineError,
    connect,
    find_latest_output_dir,
    run,
    sftp_upload,
)
from agent.nodes.state import PipelineState

logger = logging.getLogger(__name__)


# ── Remote layout (the ATK project's standard paths) ───────────────────────

_REMOTE_HOME = "/home/operator_atk"
_REMOTE_CASES_DIR = f"{_REMOTE_HOME}/cases"
_REMOTE_EXECUTOR_DIR = f"{_REMOTE_HOME}/atk_executor"
_REMOTE_OUTPUT_ROOT = f"{_REMOTE_HOME}/atk_output"

# Default CANN env init script on Ascend hosts.  Operators may override
# per-server via ``state["server_info"]["env_init_script"]``.
_DEFAULT_ENV_INIT = "/usr/local/Ascend/ascend-toolkit/set_env.sh"

# Cap on the remote ``atk node --backend cpu task`` invocation.  ATK
# runs can be slow on large case sets; 30 minutes is the existing default
# upper bound.
_DEFAULT_ATK_TIMEOUT = 1800.0


def _remote_cases_path(operator_name: str) -> str:
    return f"{_REMOTE_CASES_DIR}/{operator_name}_cases.json"


def _remote_executor_path(operator_name: str) -> str:
    return f"{_REMOTE_EXECUTOR_DIR}/{operator_name}_executor.py"


def _resolve_env_init(server_info: dict[str, Any] | None) -> str:
    if server_info:
        custom = server_info.get("env_init_script")
        if isinstance(custom, str) and custom.strip():
            return custom.strip()
    return _DEFAULT_ENV_INIT


def _build_atk_command(
    operator_name: str,
    task_type: str,
    env_init: str,
) -> str:
    """Compose the ``atk node --backend cpu task`` command line.

    Output goes to ``/home/operator_atk/atk_output`` so we can scan for
    the per-operator result directory after the command completes.
    """
    cases_remote = _remote_cases_path(operator_name)
    executor_remote = _remote_executor_path(operator_name)
    return (
        f"cd {_REMOTE_HOME} && "
        f"atk node --backend cpu task "
        f"-c {cases_remote} "
        f"-p {executor_remote} "
        f"--task {task_type} "
        f"--bind_cpu_type BIND_IN_PHYSICAL"
    )


def _resolve_project_root() -> Path:
    # run_atk.py: executer_subgraph/ → nodes/ → agent/ → src/ → agent_pkg/ → packages/ → operator-agent/
    return Path(__file__).resolve().parents[6]


def _classify_status(
    cmd_result: CommandResult,
    timeout: float,
    *,
    timed_out: bool,
) -> tuple[str, int | None]:
    """Map raw command outcome → (status, exit_code_for_result).

    Note: timeout is signalled at the engine layer — :func:`ssh_executor.run`
    raises :class:`SSHEngineError` instead of returning, so ``timed_out``
    only fires when we *catch* that error and decide to mark a timeout.
    """
    if timed_out:
        return "timeout", cmd_result.exit_code
    if cmd_result.exit_code == 0:
        return "success", cmd_result.exit_code
    return "failed", cmd_result.exit_code


# ── Main node ───────────────────────────────────────────────────────────────

async def exec_run_atk_node(state: PipelineState) -> dict[str, Any]:
    """Run ATK test cases on the selected remote machine."""
    if state.get("error"):
        return {"error": state.get("error")}

    operator_name = state.get("operator_name", "")
    executor_path = state.get("atk_executor_path")
    cases_path = state.get("cases_path")
    server_info = state.get("server_info") or {}

    if not operator_name:
        return {"error": "operator_name is required"}
    if not executor_path:
        return {"error": "atk_executor_path is required"}
    if not cases_path:
        return {"error": "cases_path is required"}
    if not (server_info.get("ip") and server_info.get("username") and server_info.get("password")):
        return {"error": "server_info is incomplete (need ip/username/password)"}

    task_type = state.get("task_type") or "accuracy"
    execution_count = int(state.get("execution_count") or 1)
    env_init = _resolve_env_init(server_info)

    endpoint = ServerEndpoint.from_server_row(server_info)
    run_id = state.get("run_id", "run")
    project_root = _resolve_project_root()
    cache_dir = make_local_cache_dir(project_root, operator_name, run_id)

    overall_start = time.monotonic()
    result = ExecutionResult()

    logger.info(
        "exec_run_atk: operator=%s server=%s task=%s exec_count=%d",
        operator_name, endpoint.host, task_type, execution_count,
    )

    # ── 1. Connect ─────────────────────────────────────────────────────────
    try:
        conn = await connect(endpoint, timeout=30.0)
    except SSHEngineError as e:
        logger.exception("exec_run_atk: SSH connect failed for %s", operator_name)
        return {"error": f"SSH 连接失败: {e}"}

    try:
        # ── 2. SFTP upload cases + executor ────────────────────────────────
        try:
            await sftp_upload(conn, cases_path, _remote_cases_path(operator_name))
            await sftp_upload(conn, executor_path, _remote_executor_path(operator_name))
        except SSHEngineError as e:
            logger.exception("exec_run_atk: SFTP upload failed for %s", operator_name)
            result.status = "error"
            result.error_message = f"SFTP 上传失败: {e}"
            result.duration = time.monotonic() - overall_start
            return {"exec_result": result.model_dump(), "error": str(e)}

        # ── 3. Run atk command ─────────────────────────────────────────────
        cmd = _build_atk_command(operator_name, task_type, env_init)
        logger.info("exec_run_atk: running %s", cmd)

        cmd_result: CommandResult | None = None
        timed_out = False
        try:
            cmd_result = await run(conn, cmd, timeout=_DEFAULT_ATK_TIMEOUT)
        except SSHEngineError as e:
            # ssh_executor.run raises on transport / timeout.  When the
            # failure was timeout, we synthesise a partial CommandResult
            # so downstream classification still has something to chew on.
            if "超时" in str(e):
                timed_out = True
                cmd_result = CommandResult(exit_code=-1, stdout="", stderr=str(e), duration=_DEFAULT_ATK_TIMEOUT)
                logger.warning("exec_run_atk: remote atk command timed out for %s", operator_name)
            else:
                logger.exception("exec_run_atk: remote atk command failed for %s", operator_name)
                result.status = "error"
                result.error_message = str(e)
                result.duration = time.monotonic() - overall_start
                return {"exec_result": result.model_dump(), "error": str(e)}

        assert cmd_result is not None  # for type checkers
        result.exit_code = cmd_result.exit_code
        result.stdout = cmd_result.stdout
        result.stderr = cmd_result.stderr
        result.status, _ = _classify_status(cmd_result, _DEFAULT_ATK_TIMEOUT, timed_out=timed_out)

        # ── 4. Discover + download + parse outputs ─────────────────────────
        # Result extraction failures MUST NOT abort the main flow, so each
        # step is wrapped — errors are recorded in the result and logged.

        try:
            output_dir = await find_latest_output_dir(
                conn, _REMOTE_OUTPUT_ROOT, operator_name,
            )
        except SSHEngineError as e:
            logger.warning("exec_run_atk: listdir failed: %s", e)
            output_dir = None

        result.remote_output_dir = output_dir

        if output_dir:
            try:
                report_data, log_content, _latest_xlsx = await collect_remote_artifacts(
                    conn, output_dir, cache_dir,
                )
                result.task_report_data = report_data
                result.log_content = log_content
                if report_data.parse_error:
                    logger.warning(
                        "exec_run_atk: report parse error for %s: %s",
                        operator_name, report_data.parse_error,
                    )
                logger.info(
                    "exec_run_atk: extracted %d report records (%d passed / %d failed) for %s",
                    report_data.record_count, report_data.passed, report_data.failed, operator_name,
                )
            except Exception as e:  # pragma: no cover — defensive
                logger.exception("exec_run_atk: artifact collection failed for %s", operator_name)
                result.error_message = f"结果提取失败: {e}"
        else:
            logger.warning(
                "exec_run_atk: no output dir found under %s for %s",
                _REMOTE_OUTPUT_ROOT, operator_name,
            )
            result.error_message = (
                f"未找到 {operator_name}_ 前缀的输出目录 ({_REMOTE_OUTPUT_ROOT})"
            )

        # ── 5. Final classification ────────────────────────────────────────
        # If ATK failed and we couldn't even find an output directory, that
        # already counts as a hard failure; keep the atk-exit-code signal.
        if result.status == "failed" and not result.task_report_data.report_records:
            # No structured records → record a useful message so the UI
            # doesn't show an empty failure.
            if not result.error_message:
                result.error_message = (
                    f"atk 命令退出码={result.exit_code}, 且未解析出任何用例记录"
                )

    finally:
        try:
            conn.close()
        except Exception:  # pragma: no cover — cleanup best effort
            pass

    result.duration = time.monotonic() - overall_start

    # ── Persist the structured ExecutionResult alongside the raw artifacts ──
    # Sibling to atk.log + report/*.xlsx so the local cache dir
    # (``execution_results/<operator_name>/<run_id>/``) carries everything
    # an operator needs to inspect / replay without re-running ATK.
    # Best-effort: never abort the main flow on a write failure.
    try:
        result_json_path = cache_dir / "result.json"
        result_json_path.write_text(
            result.model_dump_json(indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info("exec_run_atk: wrote result.json to %s", result_json_path)
    except Exception as e:  # pragma: no cover — defensive
        logger.warning("exec_run_atk: failed to write result.json for %s: %s", operator_name, e)

    # ── Emit a flat exec_result dict that matches the existing contract ───
    # The route's ``save_exec_results`` consumes ``total``/``passed``/
    # ``failed``; we add the new fields alongside.
    flat = result.model_dump()
    flat["total"] = result.task_report_data.record_count
    flat["passed"] = result.task_report_data.passed
    flat["status"] = result.status

    logger.info(
        "exec_run_atk: done for %s status=%s exit=%s records=%d duration=%.2fs",
        operator_name, result.status, result.exit_code,
        result.task_report_data.record_count, result.duration,
    )

    return {"exec_result": flat, "error": None}