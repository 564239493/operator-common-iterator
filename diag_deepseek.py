#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""直接调用 DeepSeek API 诊断脚本 — 绕过 langchain_openai。

用法:
    python diag_deepseek.py [--prompt-file PATH] [--model MODEL]

无任何参数时使用最小 payload 测试延迟；
指定 --prompt-file 可加载真实提示词评估完整响应时间。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import httpx


def read_env_var(name: str) -> str | None:
    """直接从 .env 中读一个键（避免引入 config/settings 依赖）"""
    env_path = Path.cwd() / ".env"
    if not env_path.exists():
        env_path = Path(__file__).resolve().parent / ".env"
    if not env_path.exists():
        return None
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        if k.strip() == name:
            return v.strip().strip('"').strip("'")
    return None


def call_direct(api_key: str, base_url: str, model: str,
                messages: list[dict], timeout: float = 60.0) -> dict:
    """直接 POST 到 /chat/completions，返回完整结果与耗时。"""
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.1,
        "stream": False,
    }

    print(f"[diag] POST {url}")
    print(f"[diag] model={model}  payload_keys={list(payload.keys())}")

    t0 = time.monotonic()
    try:
        with httpx.Client(timeout=timeout) as client:
            # 先用 stream 流式观察何时开始收到字节
            t_request_start = time.monotonic()
            with client.stream("POST", url, headers=headers, json=payload) as resp:
                status_line_at = time.monotonic() - t_request_start
                print(f"[diag] 收到 HTTP 状态行 (耗时 {status_line_at:.2f}s): {resp.status_code} {resp.reason_phrase}")
                resp.read()
                body_at = time.monotonic() - t_request_start
                print(f"[diag] 收到完整响应体 (耗时 {body_at:.2f}s, body 大小 {len(resp.content)} bytes)")
                return {
                    "ok": resp.status_code == 200,
                    "status": resp.status_code,
                    "elapsed_total": body_at,
                    "elapsed_status": status_line_at,
                    "body_bytes": len(resp.content),
                    "body_text": resp.text[:500] if resp.status_code == 200 else resp.text[:500],
                }
    except httpx.TimeoutException as e:
        return {"ok": False, "error": "timeout", "detail": str(e),
                "elapsed_total": time.monotonic() - t0}
    except httpx.HTTPError as e:
        return {"ok": False, "error": "http", "detail": str(e),
                "elapsed_total": time.monotonic() - t0}


def call_streaming(api_key: str, base_url: str, model: str,
                   messages: list[dict], timeout: float = 60.0) -> dict:
    """流式调用，看每个 chunk 何时到达。"""
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.1,
        "stream": True,
    }

    print(f"\n[diag-stream] POST {url} (stream=True)")
    t0 = time.monotonic()
    chunks = 0
    first_chunk_at = None
    last_chunk_at = None
    full_content = ""
    try:
        with httpx.Client(timeout=timeout) as client:
            with client.stream("POST", url, headers=headers, json=payload) as resp:
                status_at = time.monotonic() - t0
                print(f"[diag-stream] 状态行 (耗时 {status_at:.2f}s): {resp.status_code} {resp.reason_phrase}")
                if resp.status_code != 200:
                    resp.read()
                    return {"ok": False, "status": resp.status_code, "body": resp.text[:500]}
                buf = []
                for line in resp.iter_lines():
                    if not line:
                        continue
                    now = time.monotonic() - t0
                    chunks += 1
                    if first_chunk_at is None:
                        first_chunk_at = now
                        print(f"[diag-stream] 第一个数据 chunk (耗时 {now:.2f}s)")
                    last_chunk_at = now
                    if line.startswith("data:"):
                        data = line[5:].strip()
                        if data == "[DONE]":
                            print(f"[diag-stream] 收到 [DONE] (耗时 {now:.2f}s, 共 {chunks} chunks)")
                            break
                        try:
                            obj = json.loads(data)
                            delta = obj.get("choices", [{}])[0].get("delta", {}).get("content", "")
                            full_content += delta
                        except Exception:
                            pass
                    if chunks % 20 == 0:
                        print(f"[diag-stream] 收到 {chunks} chunks (耗时 {now:.2f}s)")

                total = time.monotonic() - t0
                print(f"[diag-stream] 流结束 (总耗时 {total:.2f}s, chunks={chunks})")
                return {
                    "ok": True,
                    "status": resp.status_code,
                    "elapsed_total": total,
                    "elapsed_first_chunk": first_chunk_at,
                    "elapsed_status": status_at,
                    "chunk_count": chunks,
                    "content_len": len(full_content),
                    "content_preview": full_content[:200],
                }
    except httpx.TimeoutException as e:
        return {"ok": False, "error": "timeout", "detail": str(e),
                "elapsed_total": time.monotonic() - t0,
                "chunks_so_far": chunks}


def main():
    parser = argparse.ArgumentParser(description="直接调用 DeepSeek API 诊断")
    parser.add_argument("--prompt-file", default=None, help="可选: 加载真实提示词")
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--timeout", type=float, default=60.0, help="HTTP 客户端超时")
    parser.add_argument("--skip-stream", action="store_true", help="跳过流式测试")
    args = parser.parse_args()

    api_key = read_env_var("DEEPSEEK_API_KEY") or read_env_var("ZAI_API_KEY")
    if not api_key:
        print("错误: 未找到 DEEPSEEK_API_KEY/ZAI_API_KEY，请检查 .env 文件", file=sys.stderr)
        sys.exit(2)

    base_url = read_env_var("DEEPSEEK_BASE_URL") or "https://api.deepseek.com"
    model = read_env_var("DEEPSEEK_MODEL") or args.model

    print(f"[diag] api_key 长度={len(api_key)}  base_url={base_url}  model={model}")

    # 构造 messages
    if args.prompt_file:
        text = Path(args.prompt_file).read_text(encoding="utf-8")
        messages = [
            {"role": "system", "content": "你是一个测试助手"},
            {"role": "user", "content": text},
        ]
        print(f"[diag] 使用真实提示词 ({args.prompt_file}, {len(text)} chars)")
    else:
        messages = [
            {"role": "system", "content": "你是一个测试助手"},
            {"role": "user", "content": "请用 1 句话回答: 1+1=?"},
        ]
        print(f"[diag] 使用最小 payload 测试")

    # ── 测试 1: 非流式 ──
    print("\n" + "=" * 60)
    print("测试 1: 非流式 (stream=False)")
    print("=" * 60)
    r1 = call_direct(api_key, base_url, model, messages, timeout=args.timeout)
    print(f"\n[diag] 非流式结果: {json.dumps(r1, ensure_ascii=False, indent=2)}")

    if args.skip_stream:
        return

    # ── 测试 2: 流式 ──
    print("\n" + "=" * 60)
    print("测试 2: 流式 (stream=True)")
    print("=" * 60)
    r2 = call_streaming(api_key, base_url, model, messages, timeout=args.timeout)
    print(f"\n[diag] 流式结果: {json.dumps(r2, ensure_ascii=False, indent=2)}")

    # ── 诊断结论 ──
    print("\n" + "=" * 60)
    print("诊断结论")
    print("=" * 60)
    if r1.get("ok") and r2.get("ok"):
        t1 = r1["elapsed_total"]
        t2 = r2["elapsed_total"]
        first_chunk = r2.get("elapsed_first_chunk", 0)
        print(f"非流式响应耗时: {t1:.2f}s")
        print(f"流式首字节耗时: {first_chunk:.2f}s")
        print(f"流式总耗时:     {t2:.2f}s")
        if first_chunk > 30:
            print("⚠️  推论: 模型生成第一个 token 都很慢 → DeepSeek 服务端慢")
        elif t2 - first_chunk > 30:
            print("⚠️  推论: 模型已开始输出，但 token-by-token 慢 → 推理慢或网络吞吐低")
        else:
            print("✅  API 正常，速度也在合理范围内")
    elif not r1.get("ok") and r1.get("error") == "timeout":
        print("❌  非流式直接超时 → 问题在 httpx/网络或 langchain 客户端")
    elif r1.get("status", 200) >= 400:
        print(f"❌  DeepSeek 返回 {r1.get('status')} — API 调用错误 (见 body)")


if __name__ == "__main__":
    main()