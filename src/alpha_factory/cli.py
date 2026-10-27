"""The `af` command tree. Only `config` and `synth` work in Plan 1; the rest name their plan."""

import time
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from alpha_factory.logging import configure_logging
from alpha_factory.settings import Settings

app = typer.Typer(name="af", help="Alpha Factory: intraday signal research.", no_args_is_help=True)
console = Console()
err = Console(stderr=True)

STUB_EXIT_CODE = 2


def _stub(plan: str, command: str) -> None:
    """Exit with code 2, naming the plan that implements the command."""
    err.print(f"[yellow]`af {command}` is implemented in {plan}.[/yellow]")
    raise typer.Exit(STUB_EXIT_CODE)


def _group(name: str, help_text: str) -> typer.Typer:
    group = typer.Typer(help=help_text, no_args_is_help=True)
    app.add_typer(group, name=name)
    return group


@app.callback()
def _main() -> None:
    try:
        settings = Settings()
        settings.check_prod_tree_clean()
    except (RuntimeError, ValueError) as e:
        err.print(f"[red]{e}[/red]")
        raise typer.Exit(1) from e
    configure_logging(settings.log_level, settings.log_json)


# --- config (Plan 1) -------------------------------------------------------------------------
config = _group("config", "Inspect configuration.")


@config.command("show")
def config_show() -> None:
    """Print resolved settings (secrets redacted) and config file hashes."""
    try:
        settings = Settings()
    except ValueError as e:
        err.print(f"[red]invalid settings: {e}[/red]")
        raise typer.Exit(1) from e
    table = Table("setting", "value", title="Settings")
    for key, value in settings.redacted().items():
        table.add_row(key, "" if value is None else str(value))
    console.print(table)
    hashes = Table("config", "sha256", title="Config hashes")
    for name, digest in settings.config_hashes().items():
        hashes.add_row(name, digest)
    console.print(hashes)


# --- synth (Plan 1) --------------------------------------------------------------------------
synth = _group("synth", "Synthetic market for tests and demos.")


@synth.command("build")
def synth_build(
    out: Annotated[Path, typer.Option("--out", help="Output directory.")],
    seed: Annotated[int, typer.Option("--seed", help="RNG seed.")] = 7,
    tickers: Annotated[int, typer.Option(help="Number of tickers.")] = 200,
    sessions: Annotated[int, typer.Option(help="Number of sessions.")] = 500,
) -> None:
    """Build the synthetic market: bars, universe, sectors and a manifest."""
    from alpha_factory.synthetic.market import SyntheticMarket

    t0 = time.perf_counter()
    try:
        manifest = SyntheticMarket(seed=seed, n_tickers=tickers, n_sessions=sessions).build(out)
    except ValueError as e:
        err.print(f"[red]{e}[/red]")
        raise typer.Exit(1) from e
    rows = sum(f.rows for f in manifest.files)
    console.print(f"built {rows:,} rows in {time.perf_counter() - t0:.1f}s -> {out}")
    console.print(f"dataset_hash {manifest.dataset_hash}")


# --- data (Plan 2) ---------------------------------------------------------------------------
data = _group("data", "Fetch, build, check and describe market data.")


def _data_stub(name: str) -> None:
    _stub("Plan 2", f"data {name}")


for _name in ("download", "ref", "build", "qa", "manifest", "verify", "publish"):
    data.command(_name, help=f"Plan 2: data {_name}.")(lambda n=_name: _data_stub(n))


# --- eval (Plan 3) ---------------------------------------------------------------------------
evaluate = _group("eval", "Evaluate factors and calibrate the pipeline.")


@evaluate.command("factor", context_settings={"ignore_unknown_options": True})
def eval_factor(expr: Annotated[str, typer.Argument(help="DSL expression.")]) -> None:
    """Plan 3: evaluate one expression."""
    _stub("Plan 3", "eval factor")


@evaluate.command("null")
def eval_null() -> None:
    """Plan 3: run null calibration."""
    _stub("Plan 3", "eval null")


@evaluate.command("canary")
def eval_canary() -> None:
    """Plan 3: run the leakage canary."""
    _stub("Plan 3", "eval canary")


# --- loop, lib, family, goal, report (Plan 4) ------------------------------------------------
loop = _group("loop", "Run and control the research loop.")


def _loop_stub(name: str) -> None:
    _stub("Plan 4", f"loop {name}")


for _name in ("run", "status", "pause", "resume"):
    loop.command(_name, help=f"Plan 4: loop {_name}.")(lambda n=_name: _loop_stub(n))

lib = _group("lib", "Factor libraries.")


@lib.command("freeze")
def lib_freeze() -> None:
    """Plan 4: freeze the pool into a versioned library."""
    _stub("Plan 4", "lib freeze")


family = _group("family", "Mechanism families.")


@family.command("activate")
def family_activate() -> None:
    """Plan 4: reactivate a family."""
    _stub("Plan 4", "family activate")


goal = _group("goal", "Research goals.")


def _goal_stub(name: str) -> None:
    _stub("Plan 4", f"goal {name}")


for _name in ("brief", "propose", "add", "approve", "close", "list"):
    goal.command(_name, help=f"Plan 4: goal {_name}.")(lambda n=_name: _goal_stub(n))

report = _group("report", "Reports from the research store.")


def _report_stub(name: str) -> None:
    _stub("Plan 4", f"report {name}")


for _name in ("run", "trials", "beliefs", "ab"):
    report.command(_name, help=f"Plan 4: report {_name}.")(lambda n=_name: _report_stub(n))


# --- final-eval (Plans 3 and 4) --------------------------------------------------------------
@app.command("final-eval")
def final_eval(lib_version: Annotated[str, typer.Argument(metavar="LIB")]) -> None:
    """Plans 3 and 4: sealed test evaluation; sealed Modal environment only."""
    _stub("Plans 3 and 4", "final-eval")
