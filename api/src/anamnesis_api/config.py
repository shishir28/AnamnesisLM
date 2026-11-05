from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ANAMNESIS_", env_file=".env", extra="ignore")

    checkpoint_path: Path = (
        Path(__file__).resolve().parents[3] / "llm/checkpoints/extraction/v9-second-seed.pt"
    )
    device: Literal["auto", "cpu", "cuda"] = "auto"
    max_new_tokens: int = Field(default=400, ge=1, le=1024)
