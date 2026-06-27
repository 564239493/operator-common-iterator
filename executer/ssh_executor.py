"""Thin asyncssh wrapper for the ATK execution pipeline.

Encapsulates the four primitives the ``exec_run_atk`` node needs:

* ``connect``     — open a single shared :class:`asyncssh.SSHClientConnection`
* ``sftp_upload`` — push a local file to a remote path
* ``run``         — execute a shell command, capture stdout/stderr/exit_code
* ``find_output`` — locate the latest ``<operator_name>_*`` directory under
  ``/home/operator_atk/atk_output``

Every method is a small coroutine that takes a connection (so the caller
controls connection lifecycle / retry policy) and raises concrete
exceptions on engine-level failures (SSH / SFTP / IO).  Result-extraction
errors are the caller's responsibility — see :mod:`run_atk` for the policy
that "result extraction never aborts the main flow".
"""

from __future__ import annotations

import asyncio
import logging
import socket
from dataclasses import dataclass
from typing import Any

import asyncssh

logger = logging.getLogger(__name__)


class SSHEngineError(RuntimeError):
    """Raised when an engine-level operation (connect / SFTP / IO) fails.

    ``exec_run_atk`` catches this and surfaces it via ``state["error"]``,
    which is the signal the rest of the pipeline uses to mark the run as
    a hard failure.
    """


@dataclass(frozen=True)
class ServerEndpoint:
    """Resolved SSH target — taken straight from the ``servers`` table."""

    host: str
    port: int
    username: str
    password: str

    @classmethod
    def from_server_row(cls, server: dict[str, Any]) -> "ServerEndpoint":
        return cls(
            host=server["ip"],
            port=int(server.get("port") or 22),
            username=server["username"],
            password=server["password"],
        )


@dataclass
class CommandResult:
    """Captured output of a remote shell command."""

    exit_code: int
    stdout: str
    stderr: str
    duration: float


# ── Connectivity ────────────────────────────────────────────────────────────

async def tcp_probe(host: str, port: int, timeout: float = 10.0) -> None:
    """Cheap TCP-level reachability check before opening SSH.

    Mirrors the pre-check in ``routes/servers.py`` — distinguishes
    "host unreachable" from "auth failure" so the UI can show a clearer
    error than the generic asyncssh traceback.
    """
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        await asyncio.get_event_loop().run_in_executor(
            None, sock.connect, (host, port),
        )
        sock.close()
    except socket.timeout as e:
        raise SSHEngineError(f"连接超时: {host}:{port} 无响应") from e
    except ConnectionRefusedError as e:
        raise SSHEngineError(f"连接被拒绝: {host}:{port}") from e
    except OSError as e:
        raise SSHEngineError(f"网络错误: {e}") from e


async def connect(
    endpoint: ServerEndpoint,
    *,
    timeout: float = 30.0,
) -> asyncssh.SSHClientConnection:
    """Open an SSH connection; engine-level failures raise :class:`SSHEngineError`."""
    await tcp_probe(endpoint.host, endpoint.port, timeout=timeout)

    try:
        conn = await asyncio.wait_for(
            asyncssh.connect(
                endpoint.host,
                port=endpoint.port,
                username=endpoint.username,
                password=endpoint.password,
                known_hosts=None,
            ),
            timeout=timeout,
        )
    except asyncssh.PermissionDenied as e:
        raise SSHEngineError(f"SSH 认证失败: {endpoint.username}@{endpoint.host}") from e
    except asyncssh.TimeoutError as e:
        raise SSHEngineError("SSH 认证超时") from e
    except Exception as e:
        raise SSHEngineError(f"SSH 连接失败: {e}") from e

    logger.info("ssh_executor: connected to %s@%s:%d", endpoint.username, endpoint.host, endpoint.port)
    return conn


# ── File transfer ───────────────────────────────────────────────────────────

async def sftp_upload(
    conn: asyncssh.SSHClientConnection,
    local_path: str,
    remote_path: str,
) -> None:
    """Upload ``local_path`` to ``remote_path`` via SFTP.

    Creates the parent directory on the remote side if missing (idempotent).
    """
    import os  # local import — keeps top of module stdlib-only
    import pathlib

    local = pathlib.Path(local_path)
    if not local.exists():
        raise SSHEngineError(f"本地文件不存在: {local_path}")

    parent = os.path.dirname(remote_path) or "."
    mkdir_cmd = f"mkdir -p '{parent}'"
    try:
        await conn.run(mkdir_cmd, check=False)
    except Exception as e:  # pragma: no cover — mkdir rarely fails
        logger.warning("ssh_executor: mkdir -p %s failed (continuing): %s", parent, e)

    try:
        async with conn.start_sftp_client() as sftp:
            # asyncssh exposes makedirs on the SFTP client
            try:
                await sftp.makedirs(parent, exist_ok=True)
            except (AttributeError, OSError):
                # Older asyncssh may lack exist_ok; ignore EEXIST
                try:
                    await sftp.makedirs(parent)
                except OSError:
                    pass
            await sftp.put(str(local), remote_path)
    except Exception as e:
        raise SSHEngineError(f"SFTP 上传失败: {local_path} -> {remote_path}: {e}") from e

    logger.info("ssh_executor: uploaded %s -> %s (%d bytes)", local_path, remote_path, local.stat().st_size)


# ── Shell execution ─────────────────────────────────────────────────────────

async def run(
    conn: asyncssh.SSHClientConnection,
    command: str,
    *,
    timeout: float = 1800.0,
) -> CommandResult:
    """Run ``command`` on the remote shell and capture its output.

    ``timeout`` is the upper bound for command execution.  Returning
    ``exit_code`` rather than raising lets the caller decide whether a
    non-zero exit is "test failed" (record, don't abort) or
    "engine error" (abort).  We only raise :class:`SSHEngineError`
    on the transport / timeout layer.
    """
    import time

    loop = asyncio.get_event_loop()
    started = loop.time()
    try:
        completed = await asyncio.wait_for(conn.run(command, check=False), timeout=timeout)
    except asyncio.TimeoutError as e:
        raise SSHEngineError(
            f"远端命令执行超时 ({timeout}s): {command[:200]}"
        ) from e
    except Exception as e:
        raise SSHEngineError(f"远端命令执行失败: {e}") from e

    duration = loop.time() - started
    return CommandResult(
        exit_code=int(completed.exit_status) if completed.exit_status is not None else -1,
        stdout=str(completed.stdout or ""),
        stderr=str(completed.stderr or ""),
        duration=float(duration),
    )


# ── ATK output discovery ────────────────────────────────────────────────────

async def find_latest_output_dir(
    conn: asyncssh.SSHClientConnection,
    output_root: str,
    operator_prefix: str,
) -> str | None:
    """Return the most recent ``<output_root>/<operator_prefix>*`` directory.

    We pick by lexicographic descending order on the directory name — ATK
    typically stamps output directories with ``YYYYMMDD_HHMMSS_<hash>``
    suffixes that sort correctly under that scheme.  Falls back to mtime
    if the names don't sort meaningfully.
    """
    # List directories matching the prefix, sorted by mtime desc, take first.
    # Use ``ls -td`` which orders by mtime and ``head -1`` for the newest.
    cmd = (
        f"if [ -d '{output_root}' ]; then "
        f"ls -1td '{output_root}'/{operator_prefix}* 2>/dev/null | head -1; "
        f"else echo __MISSING__; fi"
    )
    result = await run(conn, cmd, timeout=30.0)
    line = (result.stdout or "").strip().splitlines()
    if not line:
        return None
    candidate = line[0].strip()
    if not candidate or candidate == "__MISSING__":
        return None
    return candidate


__all__ = [
    "CommandResult",
    "ServerEndpoint",
    "SSHEngineError",
    "connect",
    "find_latest_output_dir",
    "run",
    "sftp_upload",
    "tcp_probe",
]