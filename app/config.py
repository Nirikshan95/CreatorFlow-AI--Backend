from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, AliasChoices
from pydantic import field_validator
from typing import Optional
from dotenv import load_dotenv
from pathlib import Path

# Backend microservice loads only its own env file.
load_dotenv()



class Settings(BaseSettings):
    openai_api_key: Optional[str] = Field(None, validation_alias=AliasChoices("openai_api_key", "OPENAI_API_KEY"))
    huggingface_api_token: Optional[str] = Field(
        None,
        validation_alias=AliasChoices(
            "huggingface_api_token",
            "HUGGINGFACEHUB_API_TOKEN",
            "HF_TOKEN",
            "HUGGINGFACEHUB_ACCESS_KEY",
            "HUGGINGFACEHUB_ACCESS_TOKEN",
        ),
    )
    openrouter_api_key: Optional[str] = Field(None, validation_alias=AliasChoices("openrouter_api_key", "OPENROUTER_API_KEY"))
    
    # HuggingFace Model Configuration (set in .env)
    hf_heavy_model: str = Field(
        "Qwen/Qwen2.5-72B-Instruct",
        validation_alias=AliasChoices("hf_heavy_model", "HF_HEAVY_MODEL")
    )
    hf_flash_model: str = Field(
        "Qwen/Qwen2.5-32B-Instruct",
        validation_alias=AliasChoices("hf_flash_model", "HF_FLASH_MODEL")
    )
    
    # OpenRouter Model Configuration (set in .env)
    or_heavy_model: str = Field(
        "google/gemini-2.0-pro-exp-02-05:free",
        validation_alias=AliasChoices("or_heavy_model", "OR_HEAVY_MODEL")
    )
    or_flash_model: str = Field(
        "google/gemini-2.0-flash-lite-preview-02-05:free",
        validation_alias=AliasChoices("or_flash_model", "OR_FLASH_MODEL")
    )
    or_fallback_model: str = Field(
        "meta-llama/llama-3.3-70b-instruct:free",
        validation_alias=AliasChoices("or_fallback_model", "OR_FALLBACK_MODEL")
    )

    
    database_path: str = "data/content_history.db"

    api_host: str = "0.0.0.0"
    api_port: int = 8000
    debug: bool = True
    langchain_tracing_v2: bool = False
    langchain_api_key: Optional[str] = None
    langchain_project: str = "YTA-System"
    chroma_db_path: str = "backend/data/chroma_db"
    enable_vector_db: bool = False
    channel_profile_path: str = "data/channel_profile.json"

    @field_validator("debug", mode="before")
    @classmethod
    def parse_debug_bool(cls, value):
        if isinstance(value, bool):
            return value
        if value is None:
            return True
        text = str(value).strip().lower()
        if text in {"1", "true", "yes", "on", "debug", "dev", "development"}:
            return True
        if text in {"0", "false", "no", "off", "release", "prod", "production"}:
            return False
        return True


settings = Settings()
