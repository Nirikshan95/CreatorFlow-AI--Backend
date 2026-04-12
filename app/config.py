from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, AliasChoices
from typing import Optional



class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(".env", "backend/.env", "../.env", "backend/../.env"),
        case_sensitive=False,
        extra="ignore",
    )

    openai_api_key: Optional[str] = Field(None, validation_alias=AliasChoices("openai_api_key", "OPENAI_API_KEY"))
    huggingface_api_token: Optional[str] = Field(None, validation_alias=AliasChoices("huggingface_api_token", "HUGGINGFACEHUB_API_TOKEN", "HF_TOKEN", "HUGGINGFACEHUB_ACCESS_KEY"))
    openrouter_api_key: Optional[str] = Field(None, validation_alias=AliasChoices("openrouter_api_key", "OPENROUTER_API_KEY"))
    
    # Model Configuration
    primary_model: str = Field("Qwen/Qwen2.5-72B-Instruct", validation_alias=AliasChoices("primary_model", "PRIMARY_MODEL"))
    fallback_model: str = Field("meta-llama/llama-3.1-8b-instruct:free", validation_alias=AliasChoices("fallback_model", "FALLBACK_MODEL"))

    
    database_path: str = "data/content_history.db"

    api_host: str = "0.0.0.0"
    api_port: int = 8000
    debug: bool = True
    langchain_tracing_v2: bool = False
    langchain_api_key: Optional[str] = None
    langchain_project: str = "YTA-System"
    chroma_db_path: str = "backend/data/chroma_db"
    enable_vector_db: bool = False


settings = Settings()
