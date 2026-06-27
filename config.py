# -*- coding: UTF-8 -*-
"""operator-common-iterate 独立 LLM 配置。

不依赖 operator-project/operator-agent，完全自包含。
读取本目录下的 .env 文件，也支持环境变量覆盖。
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMProvider(str, Enum):
    ZAI = "zai"
    DEEPSEEK = "deepseek"


_PROVIDER_DEFAULTS: dict[LLMProvider, dict[str, str]] = {
    LLMProvider.ZAI: {
        "base_url": "https://api.z.ai/api/paas/v4/",
        "model": "glm-5.1",
    },
    LLMProvider.DEEPSEEK: {
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-v4-flash",
    },
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---- LLM 厂商选择 ----
    llm_provider: LLMProvider = LLMProvider.DEEPSEEK

    # ---- DeepSeek ----
    deepseek_api_key: SecretStr = SecretStr("")
    deepseek_base_url: str = _PROVIDER_DEFAULTS[LLMProvider.DEEPSEEK]["base_url"]
    deepseek_model: str = _PROVIDER_DEFAULTS[LLMProvider.DEEPSEEK]["model"]

    # ---- Z.AI ----
    zai_api_key: SecretStr = SecretStr("")
    zai_base_url: str = _PROVIDER_DEFAULTS[LLMProvider.ZAI]["base_url"]
    zai_model: str = _PROVIDER_DEFAULTS[LLMProvider.ZAI]["model"]

    # ---- CLI Agent (agent backend) ----
    cli_agent_bin: str = "claude"
    cli_agent_args: str = "-p {prompt} --print --output-format text"

    # ---- 共用参数 ----
    llm_temperature: float = Field(default=0.1, ge=0.0, le=2.0)

    @property
    def active_api_key(self) -> SecretStr:
        match self.llm_provider:
            case LLMProvider.ZAI:
                return self.zai_api_key
            case LLMProvider.DEEPSEEK:
                return self.deepseek_api_key

    @property
    def active_base_url(self) -> str:
        match self.llm_provider:
            case LLMProvider.ZAI:
                return self.zai_base_url
            case LLMProvider.DEEPSEEK:
                return self.deepseek_base_url

    @property
    def active_model(self) -> str:
        match self.llm_provider:
            case LLMProvider.ZAI:
                return self.zai_model
            case LLMProvider.DEEPSEEK:
                return self.deepseek_model


# 查找 .env 文件：优先当前目录，其次脚本所在目录
def _find_env() -> Path:
    cwd_env = Path.cwd() / ".env"
    if cwd_env.exists():
        return cwd_env
    script_env = Path(__file__).resolve().parent / ".env"
    if script_env.exists():
        return script_env
    return cwd_env  # fallback


# 单例
_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _env = _find_env()
        _settings = Settings(_env_file=str(_env))
    return _settings
