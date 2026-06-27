#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""diag_idle_timeout.py — 验证 SSE 卡顿时 per-chunk idle 超时是否生效。

启一个本地 SSE mock server：
  - 立即写 SSE header + 一个 data chunk
  - 然后永久 sleep 不再发任何数据，模拟服务端卡死

调用 APIBackend.invoke()，期望 60 秒左右抛 RuntimeError("LLM 流空闲超时 ...")。
"""
from __future__ import annotations

import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer


def make_handler(stop_event: threading.Event):
    """生成 SSE handler：发 1 个 data chunk 后阻塞。"""
    class SSEHandler(BaseHTTPRequestHandler):
        def log_message(self, *args, **kwargs):
            pass  # 静默日志

        def do_POST(self):
            # OpenAI 兼容 SSE 头
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.flush()

            # 发 1 个 data chunk（OpenAI ChatCompletionChunk 格式）
            payload = {
                "id": "test-1",
                "object": "chat.completion.chunk",
                "choices": [{
                    "index": 0,
                    "delta": {"role": "assistant", "content": "hello"},
                    "finish_reason": None,
                }],
            }
            line = f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
            self.wfile.write(line.encode("utf-8"))
            self.wfile.flush()

            # 阻塞 — 模拟服务端卡死，客户端应触发 idle 超时
            stop_event.wait(timeout=180)
            try:
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            except Exception:
                pass

    return SSEHandler


def start_server(stop_event: threading.Event):
    """在 127.0.0.1:0 启 server，返回 (server, port)。"""
    server = HTTPServer(("127.0.0.1", 0), make_handler(stop_event))
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, port


def test_v2_idle_timeout(chunk_idle: float, total: float) -> tuple[float, str]:
    """V2: 1 个 chunk 后服务端卡死，期望 60s 左右 abort。

    Returns:
        (elapsed_seconds, result_text)
    """
    stop_event = threading.Event()
    server, port = start_server(stop_event)
    base_url = f"http://127.0.0.1:{port}/v1"

    # 关键：通过环境变量覆盖 base_url，构造指向本地 mock server 的 APIBackend
    import os
    os.environ["DEEPSEEK_BASE_URL"] = base_url
    # 重新加载 settings
    import importlib
    import config as cfg_mod
    importlib.reload(cfg_mod)
    import backends
    importlib.reload(backends)
    from backends import APIBackend

    api_backend = APIBackend(
        model="mock",
        chunk_idle_timeout=chunk_idle,
        total_timeout=total,
    )

    print(f"[diag] mock server @ {base_url}  chunk_idle={chunk_idle}s  total={total}s")
    t0 = time.monotonic()
    err_msg = ""
    try:
        content = api_backend.invoke(
            system_prompt="x", user_prompt="hi", temperature=0.1,
        )
        err_msg = f"❌ 意外成功 (content_len={len(content)})"
    except RuntimeError as e:
        elapsed = time.monotonic() - t0
        err_msg = f"{type(e).__name__}: {e} (elapsed={elapsed:.2f}s)"
    except Exception as e:
        elapsed = time.monotonic() - t0
        err_msg = f"{type(e).__name__}: {e} (elapsed={elapsed:.2f}s)"
    finally:
        stop_event.set()
        server.shutdown()
        server.server_close()

    return time.monotonic() - t0, err_msg


def main():
    print("=" * 60)
    print("V2 测试: SSE 卡顿时 per-chunk idle 超时应在 60 秒左右 abort")
    print("=" * 60)
    print("(注意: 这次故意把超时设短一些 = 10s，以加快验证)\n")

    # 把超时设短到 10s 加速验证
    elapsed, result = test_v2_idle_timeout(chunk_idle=10.0, total=600.0)
    print(f"\n[diag] 耗时: {elapsed:.2f}s")
    print(f"[diag] 结果: {result}")

    if "空闲超时" in result and 9.0 <= elapsed <= 14.0:
        print(f"\n✅ V2 通过 — 在 {elapsed:.1f}s 处触发 idle 超时 (期望 10±4s)")
        sys.exit(0)
    elif "空闲超时" in result:
        print(f"\n⚠️  V2 部分通过 — 触发了 idle 超时但耗时 {elapsed:.1f}s 不在 9-14s 区间")
        sys.exit(1)
    else:
        print(f"\n❌ V2 失败 — 期望 idle 超时，实际: {result}")
        sys.exit(2)


if __name__ == "__main__":
    main()