import os
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings configuration"""

    # Gemini LLM Configuration
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-2.5-flash"

    # Elasticsearch Configuration
    ELASTICSEARCH_HOST: str = "localhost"
    ELASTICSEARCH_PORT: int = 9200
    ELASTICSEARCH_USERNAME: Optional[str] = None
    ELASTICSEARCH_PASSWORD: Optional[str] = None
    ELASTICSEARCH_INDEX_PREFIX: str = "smart_home"

    # Embedding Model Configuration
    EMBEDDING_MODEL: str = "mixedbread-ai/mxbai-embed-large-v1"
    EMBEDDING_DIMENSION: int = 1024

    # Redis Configuration (for message broker)
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0

    # Agent Configuration
    AGENT_UPDATE_INTERVAL: int = 30  # seconds
    MAX_AGENT_RETRIES: int = 3
    AGENT_TIMEOUT: int = 60  # seconds

    # Energy System Configuration
    ENERGY_PRICE_UPDATE_INTERVAL: int = 300  # 5 minutes
    DEVICE_POLLING_INTERVAL: int = 10  # seconds
    COMFORT_OPTIMIZATION_INTERVAL: int = 120  # 2 minutes

    # Logging Configuration
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "smart_home_energy.log"

    # API Configuration
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000

    # Development Settings
    DEBUG: bool = False
    TESTING: bool = False

    class Config:
        env_file = ".env"
        case_sensitive = True


# Global settings instance
settings = Settings()
