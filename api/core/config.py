import os
from typing import List, Set
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    API_TITLE: str = "PaddleOCR API Service"
    API_VERSION: str = "1.0.0"
    API_DESCRIPTION: str = "Production-ready FastAPI API for PaddleOCR"
    CORS_ORIGINS: List[str] = ["*"]
    MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10MB
    ALLOWED_EXTENSIONS: Set[str] = {"png", "jpg", "jpeg", "webp", "bmp", "tiff", "gif"}
    DEFAULT_LANG: str = "vi"
    TIMEOUT_SECONDS: int = 30
    USE_GPU: bool = False
    RATE_LIMIT_LIMIT: int = 60
    RATE_LIMIT_PERIOD: int = 60  # in seconds
    LOG_LEVEL: str = "INFO"

    # Support loading from .env file
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()
