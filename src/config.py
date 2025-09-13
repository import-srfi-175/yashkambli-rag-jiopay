"""
Configuration management for the JioPay RAG Chatbot.
"""
import os
from typing import List, Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # API Keys
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    google_api_key: str = Field(..., env="GOOGLE_API_KEY")
    
    # Vector Database Configuration
    chroma_persist_directory: str = Field("./data/chroma_db", env="CHROMA_PERSIST_DIRECTORY")
    faiss_index_path: str = Field("./data/faiss_index", env="FAISS_INDEX_PATH")
    
    # Scraping Configuration
    scraping_delay: float = Field(1.0, env="SCRAPING_DELAY")
    max_concurrent_requests: int = Field(5, env="MAX_CONCURRENT_REQUESTS")
    user_agent: str = Field(
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
        env="USER_AGENT"
    )
    
    # Application Configuration
    app_host: str = Field("0.0.0.0", env="APP_HOST")
    app_port: int = Field(8000, env="APP_PORT")
    debug: bool = Field(True, env="DEBUG")
    
    # Data Paths
    data_dir: str = Field("./data", env="DATA_DIR")
    scraped_data_dir: str = Field("./data/scraped", env="SCRAPED_DATA_DIR")
    processed_data_dir: str = Field("./data/processed", env="PROCESSED_DATA_DIR")
    embeddings_dir: str = Field("./data/embeddings", env="EMBEDDINGS_DIR")
    vector_store_dir: str = Field("./data/vector_store", env="VECTOR_STORE_DIR")
    evaluation_data_dir: str = Field("./data/evaluation", env="EVALUATION_DATA_DIR")
    
    # Model Configuration
    default_embedding_model: str = Field("text-embedding-3-small", env="DEFAULT_EMBEDDING_MODEL")
    default_chunk_size: int = Field(512, env="DEFAULT_CHUNK_SIZE")
    default_chunk_overlap: int = Field(64, env="DEFAULT_CHUNK_OVERLAP")
    default_top_k: int = Field(5, env="DEFAULT_TOP_K")
    
    # Evaluation Configuration
    test_set_size: int = Field(10, env="TEST_SET_SIZE")
    evaluation_metrics: List[str] = Field(
        ["precision", "recall", "f1", "mrr"], 
        env="EVALUATION_METRICS"
    )
    
    # JioPay URLs
    jiopay_business_url: str = "https://jiopay.com/business"
    jiopay_help_url: str = "https://jiopay.com/help"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get application settings."""
    return settings
