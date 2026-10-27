"""Base models and shared types."""

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, StringConstraints, model_validator


class LLMModel(BaseModel):
    """Base for everything an LLM writes. Small, flat, strict."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


class Record(BaseModel):
    """Base for everything code writes and stores."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: int = 1

    @model_validator(mode="before")
    @classmethod
    def _drop_computed_fields(cls, data: Any) -> Any:
        """Let a stored dump (which includes computed fields) round-trip through validation."""
        if isinstance(data, dict):
            computed = cls.model_computed_fields
            return {k: v for k, v in data.items() if k not in computed}  # pyright: ignore[reportUnknownVariableType]
        return data


Slug = Annotated[str, StringConstraints(pattern=r"^[a-z][a-z0-9_]{2,39}$")]
Horizon = Literal["h1", "h2", "h4"]
Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
