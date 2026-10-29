"""Assembling the loop from settings: store, data, evaluation backend, proposers, notifier."""

import os
from dataclasses import dataclass

from alpha_factory.agents.llm_client import LLMClient, OpenAICompatClient
from alpha_factory.agents.llm_proposer import LLMProposer
from alpha_factory.agents.reflect import LLMReflector
from alpha_factory.agents.template import TemplateProposer
from alpha_factory.compute.worker import EvalBackend, Evaluator, LocalBackend
from alpha_factory.contracts import SandboxSpec
from alpha_factory.data import MarketData
from alpha_factory.eval.calibration import load_calibration
from alpha_factory.orchestrator import Orchestrator
from alpha_factory.research.config import LLMConfig, LoopConfig
from alpha_factory.research.notify import Notifier
from alpha_factory.research.store import ResearchStore
from alpha_factory.sandbox import load_sandbox
from alpha_factory.settings import Settings


def load_loop_config(settings: Settings) -> LoopConfig:
    return settings.load("loop", LoopConfig)


def load_llm_config(settings: Settings) -> LLMConfig:
    return settings.load("llm", LLMConfig)


def make_llm_client(settings: Settings, cfg: LLMConfig) -> LLMClient | None:
    """The OpenAI-compatible client, or None when the base URL or key is not set."""
    base = (
        settings.llm_base_url
        if cfg.base_url_env == "LLM_BASE_URL"
        else os.environ.get(cfg.base_url_env)
    )
    key = settings.llm_api_key.get_secret_value() if settings.llm_api_key else None
    if not base or not key:
        return None
    return OpenAICompatClient(cfg, base, key)


def dev_world(settings: Settings) -> tuple[MarketData, SandboxSpec]:
    data = MarketData("dev", settings=settings)
    return data, load_sandbox(settings, dataset_hash=data.manifest.dataset_hash)


def make_backend(
    settings: Settings, cfg: LoopConfig, data: MarketData, sandbox: SandboxSpec
) -> EvalBackend:
    if cfg.evaluation.backend == "modal":
        from alpha_factory.compute.modal_dev import ModalBackend

        return ModalBackend(os.environ.get("AF_MODAL_ENVIRONMENT", "dev"))
    cal = load_calibration(settings.data_dir, sandbox.content_hash(), data.manifest.dataset_hash)
    floor = {str(k): v for k, v in cal.ic_floor.items()} if cal else None
    return LocalBackend(Evaluator(data, sandbox, floor))


@dataclass
class Runtime:
    orchestrator: Orchestrator
    store: ResearchStore
    cfg: LoopConfig
    sandbox: SandboxSpec
    data: MarketData
    llm: LLMClient | None


def build_runtime(settings: Settings) -> Runtime:
    cfg = load_loop_config(settings)
    store = ResearchStore(settings.store_path)
    data, sandbox = dev_world(settings)
    llm = make_llm_client(settings, load_llm_config(settings))
    if cfg.proposer == "llm" and llm is None:
        settings.require("llm_base_url", "llm_api_key")  # fails with the variable names
    proposers: dict[str, object] = {"template": TemplateProposer()}
    reflector = None
    if llm is not None:
        proposers["llm"] = LLMProposer(llm, sandbox)
        reflector = LLMReflector(llm, store, sandbox)
    webhook = (
        settings.discord_webhook_url.get_secret_value() if settings.discord_webhook_url else None
    )
    orchestrator = Orchestrator(
        store=store, cfg=cfg, backend=make_backend(settings, cfg, data, sandbox), sandbox=sandbox,
        manifest=data.manifest, proposers=proposers,  # type: ignore[arg-type]
        notify=Notifier(webhook).notify, reflector=reflector,
    )  # fmt: skip
    return Runtime(orchestrator, store, cfg, sandbox, data, llm)
