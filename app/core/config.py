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
    POSTGRES_HOST: str = Field(default="localhost")  # Fallback to localhost for local testing
    
    # Computed Database URL (prefer environment DATABASE_URL if provided)
    DATABASE_URL: str = Field(default="")

    # API Keys
    GOOGLE_API_KEY: str = Field(default="")
    TAVILY_API_KEY: str = Field(default="")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    def get_database_url(self) -> str:
        # Mã hóa mật khẩu tránh ký tự đặc biệt như ':' và '@'
        password_quoted = urllib.parse.quote_plus(self.POSTGRES_PASSWORD)
        
        # Tự động phát hiện nếu host 'db' không phân giải được (chạy bên ngoài Docker)
        # thì sẽ tự động fallback sang 'localhost'
        host = self.POSTGRES_HOST
        if host == "db":
            try:
                socket.getaddrinfo("db", self.POSTGRES_PORT)
            except socket.gaierror:
                host = "localhost"
        
        # Nếu DATABASE_URL chứa mật khẩu thô chưa mã hóa từ docker-compose, ta bỏ qua và tự dựng lại
        if self.DATABASE_URL and self.POSTGRES_PASSWORD not in self.DATABASE_URL:
            url = self.DATABASE_URL
            if url.startswith("postgresql://"):
                url = url.replace("postgresql://", "postgresql+asyncpg://", 1)
            return url
            
        return f"postgresql+asyncpg://{self.POSTGRES_USER}:{password_quoted}@{host}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"



settings = Settings()
