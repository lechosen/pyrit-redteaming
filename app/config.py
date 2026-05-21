
# app/config.py
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # Azure Entra ID (company-approved App Registration)
    azure_tenant_id: str
    azure_client_id: str
    azure_client_secret: str

    # Optional defaults for convenience (not required)
    azure_inference_default_endpoint: str | None = None
    azure_inference_default_model: str | None = None

    # 👇 NEW: Accept unknown .env keys (e.g., GLOBAL_MEMORY_LABELS used by PyRIT)
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore"   # <- prevent extra_forbidden errors
    )

settings = Settings()
