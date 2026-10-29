"""Prompt templates: Jinja files in prompts/, versioned by file name."""

from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from alpha_factory.hashing import sha256_canonical

_env = Environment(
    loader=FileSystemLoader(Path(__file__).parent / "prompts"),
    undefined=StrictUndefined,
    keep_trailing_newline=True,
    trim_blocks=False,
    autoescape=False,
)
_env.policies["json.dumps_kwargs"] = {"sort_keys": True, "ensure_ascii": False}

PROPOSER = "proposer_v1"
REFLECTOR = "reflector_v1"
REPAIR = "repair_v1"
MANAGER = "manager_v1"


def render(name: str, **context: object) -> str:
    """Render prompts/<name>.md.j2 (for example "proposer_v1.system")."""
    return _env.get_template(f"{name}.md.j2").render(**context).rstrip() + "\n"


def prompt_hash(system: str, user: str) -> str:
    return sha256_canonical({"system": system, "user": user})
