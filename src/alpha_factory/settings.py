"""Settings: environment variables for secrets and paths, YAML in configs/ for behaviour."""

import subprocess
from pathlib import Path
from typing import Any, Literal, TypeVar

import yaml
from pydantic import BaseModel, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from alpha_factory.hashing import sha256_file

T = TypeVar("T", bound=BaseModel)

CONFIG_NAMES = ("sandbox", "data", "llm", "loop")

# Settings field -> environment variable name, for error messages.
SECRET_ENV_NAMES = {
    "massive_api_key": "MASSIVE_API_KEY",
    "massive_s3_key_id": "MASSIVE_S3_ACCESS_KEY_ID",
    "massive_s3_secret": "MASSIVE_S3_SECRET_ACCESS_KEY",
    "massive_s3_endpoint": "MASSIVE_S3_ENDPOINT",
    "massive_s3_bucket": "MASSIVE_S3_BUCKET",
    "llm_base_url": "LLM_BASE_URL",
    "llm_api_key": "LLM_API_KEY",
    "discord_webhook_url": "DISCORD_WEBHOOK_URL",
}


class MissingSettingError(RuntimeError):
    """A command needs a setting that is not set."""


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="AF_", env_file=".env", extra="ignore", env_ignore_empty=True
    )

    env: Literal["dev", "prod"] = "dev"
    data_dir: Path = Path("data")
    store_path: Path = Path("data/store.sqlite")
    config_dir: Path = Path("configs")
    log_level: str = "INFO"
    log_json: bool = False

    massive_api_key: SecretStr | None = Field(default=None, validation_alias="MASSIVE_API_KEY")
    massive_s3_key_id: SecretStr | None = Field(
        default=None, validation_alias="MASSIVE_S3_ACCESS_KEY_ID"
    )
    massive_s3_secret: SecretStr | None = Field(
        default=None, validation_alias="MASSIVE_S3_SECRET_ACCESS_KEY"
    )
    massive_s3_endpoint: str | None = Field(default=None, validation_alias="MASSIVE_S3_ENDPOINT")
    massive_s3_bucket: str | None = Field(default=None, validation_alias="MASSIVE_S3_BUCKET")
    llm_base_url: str | None = Field(default=None, validation_alias="LLM_BASE_URL")
    llm_api_key: SecretStr | None = Field(default=None, validation_alias="LLM_API_KEY")
    discord_webhook_url: SecretStr | None = Field(
        default=None, validation_alias="DISCORD_WEBHOOK_URL"
    )

    def load(self, name: str, model: type[T]) -> T:
        """Load configs/<name>.yaml into a model. Unknown keys fail (models forbid extras)."""
        raw = yaml.safe_load((self.config_dir / f"{name}.yaml").read_text())
        return model.model_validate(raw)

    def config_hashes(self) -> dict[str, str]:
        """sha256 of each committed config file that exists. Recorded with every run."""
        return {
            n: sha256_file(p)
            for n in CONFIG_NAMES
            if (p := self.config_dir / f"{n}.yaml").is_file()
        }

    def require(self, *fields: str) -> None:
        """Fail at startup, naming the environment variables that are missing."""
        missing = [SECRET_ENV_NAMES[f] for f in fields if getattr(self, f) is None]
        if missing:
            raise MissingSettingError(
                f"missing required environment variables: {', '.join(missing)}"
            )

    def redacted(self) -> dict[str, Any]:
        """Settings as plain data with every secret replaced by a marker."""
        out: dict[str, Any] = {}
        for key, value in self.model_dump().items():
            if isinstance(value, SecretStr):
                out[key] = "**********" if value.get_secret_value() else ""
            elif isinstance(value, Path):
                out[key] = str(value)
            else:
                out[key] = value
        return out

    def check_prod_tree_clean(self) -> None:
        """AF_ENV=prod refuses to run with a dirty git tree."""
        if self.env != "prod":
            return
        status = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, check=False
        )
        if status.returncode != 0 or status.stdout.strip():
            raise RuntimeError("AF_ENV=prod requires a clean git working tree")
