#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""诊断 langchain_openai ChatOpenAI 行为 — 启用 httpx DEBUG 日志看到底卡在哪。

启用 HTTPX_LOG_LEVEL=DEBUG 后，会看到：
- 每个 socket 建立
- TLS 握手
- 每次发送/接收字节
如果 hang 在 socket 接收阶段，会看到 send 但看不到任何 recv。
"""
from __future__ import annotations

import logging
import os
import sys
import time

# 强制在导入 langchain 之前打开 httpx DEBUG
os.environ.setdefault("HTTPX_LOG_LEVEL", "DEBUG")

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s.%(msecs)03d [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
# 关掉一些噪音
for noisy in ["httpcore", "asyncio", "urllib3"]:
    logging.getLogger(noisy).setLevel(logging.INFO)

from langchain_openai import ChatOpenAI
from config import get_settings


def main():
    s = get_settings()
    api_key = s.deepseek_api_key.get_secret_value() or s.zai_api_key.get_secret_value()
    base_url = s.active_base_url
    model = s.active_model
    print(f"[diag-langchain] base_url={base_url}  model={model}  key_len={len(api_key)}",
          flush=True)

    # 关键：尝试各种正确的超时参数名
    print("\n=== 尝试 1: ChatOpenAI(max_retries=0, timeout=60) ===", flush=True)
    t0 = time.monotonic()
    try:
        llm = ChatOpenAI(
            api_key=api_key,
            base_url=base_url,
            model=model,
            temperature=0.1,
            max_retries=0,        # 关闭 SDK 内置重试，便于观察单次请求
            timeout=60,           # langchain-openai 0.3+ 用 timeout
        )
        resp = llm.invoke([
            {"role": "system", "content": "你是一个测试助手"},
            {"role": "user", "content": "用 1 句话回答: 1+1=?"},
        ])
        print(f"\n[diag-langchain] ✅ 成功 (耗时 {time.monotonic()-t0:.2f}s)")
        print(f"[diag-langchain] response.content={resp.content!r}")
    except Exception as e:
        print(f"\n[diag-langchain] ❌ 失败: {type(e).__name__}: {e}",
              flush=True)
        print(f"[diag-langchain] 耗时 {time.monotonic()-t0:.2f}s", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()