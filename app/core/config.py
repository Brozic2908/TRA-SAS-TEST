import os
import urllib.parse
import socket
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    # Database Settings
    POSTGRES_USER: str = Field(default="postgres")
    POSTGRES_PASSWORD: str = Field(default="MyP@ss:word123")
    POSTGRES_DB: str = Field(default="customs_rag")
    POSTGRES_PORT: int = Field(default=5432)
    POSTGRES_HOST: str = Field(default="localhost")
    
    DATABASE_URL: str = Field(default="")

    # AI API Keys
    GROQ_API_KEY: str = Field(default="")
    GOOGLE_API_KEY: str = Field(default="")
    TAVILY_API_KEY: str = Field(default="")
    OPENROUTER_API_KEY: str = Field(default="")

    # Chế độ Offline Fallback
    OFFLINE_MODE: bool = Field(default=False)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    def get_database_url(self) -> str:
        password_quoted = urllib.parse.quote_plus(self.POSTGRES_PASSWORD)
        
        host = self.POSTGRES_HOST
        if host == "db":
            try:
                socket.getaddrinfo("db", self.POSTGRES_PORT)
            except socket.gaierror:
                host = "localhost"
        
        if self.DATABASE_URL and self.POSTGRES_PASSWORD not in self.DATABASE_URL:
            url = self.DATABASE_URL
            if url.startswith("postgresql://"):
                url = url.replace("postgresql://", "postgresql+asyncpg://", 1)
            return url
            
        return f"postgresql+asyncpg://{self.POSTGRES_USER}:{password_quoted}@{host}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"

settings = Settings()
