"""The `af` command tree. Only `config` and `synth` work in Plan 1; the rest name their plan."""

import datetime
import time
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer
from rich.console import Console
from rich.table import Table

from alpha_factory.logging import configure_logging
from alpha_factory.settings import MissingSettingError, Settings

if TYPE_CHECKING:
    from alpha_factory.data.config import DataConfig
    from alpha_factory.data.layout import DataLayout

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


class KindChoice(StrEnum):
    dev = "dev"
    full = "full"
    both = "both"


def _data_context() -> "tuple[Settings, DataConfig, DataLayout]":
    from alpha_factory.data.config import DataConfig
    from alpha_factory.data.layout import DataLayout

    settings = Settings()
    try:
        cfg = settings.load("data", DataConfig)
    except (OSError, ValueError) as e:
        err.print(f"[red]cannot load configs/data.yaml: {e}[/red]")
        raise typer.Exit(1) from e
    return settings, cfg, DataLayout(settings.data_dir)


def _need(settings: Settings, *fields: str) -> None:
    try:
        settings.require(*fields)
    except MissingSettingError as e:
        err.print(f"[red]{e}[/red]")
        raise typer.Exit(1) from e


def _kinds(choice: KindChoice) -> list[str]:
    return ["full", "dev"] if choice is KindChoice.both else [choice.value]


@data.command("download")
def data_download(
    start: Annotated[
        datetime.datetime | None,
        typer.Option(formats=["%Y-%m-%d"], help="First session (default: config)."),
    ] = None,
    end: Annotated[
        datetime.datetime | None,
        typer.Option(formats=["%Y-%m-%d"], help="Last session (default: config)."),
    ] = None,
) -> None:
    """Fetch minute flat files. Resumable: files already present with matching size are skipped."""
    from concurrent.futures import ThreadPoolExecutor

    from alpha_factory.data.calendar import TradingCalendar
    from alpha_factory.data.clients import S3Store, download_file
    from alpha_factory.data.massive import minute_key

    settings, cfg, layout = _data_context()
    _need(
        settings,
        "massive_s3_key_id",
        "massive_s3_secret",
        "massive_s3_endpoint",
        "massive_s3_bucket",
    )
    assert settings.massive_s3_key_id and settings.massive_s3_secret  # narrowed by _need
    assert settings.massive_s3_endpoint and settings.massive_s3_bucket
    store = S3Store(
        settings.massive_s3_endpoint,
        settings.massive_s3_bucket,
        settings.massive_s3_key_id.get_secret_value(),
        settings.massive_s3_secret.get_secret_value(),
    )
    lo = start.date() if start else cfg.start
    hi = min(end.date() if end else cfg.end, datetime.date.today() - datetime.timedelta(days=1))
    sessions = TradingCalendar(lo, hi).sessions(lo, hi)
    counts: dict[str, int] = {"downloaded": 0, "skipped": 0, "missing": 0}

    def one(d: datetime.date) -> str:
        return download_file(store, minute_key(d), layout.raw_minute(d), cfg.download.retries)

    with ThreadPoolExecutor(cfg.download.concurrency) as ex:
        for result in ex.map(one, sessions):
            counts[result] += 1
    console.print(f"{len(sessions)} sessions: {counts}")
    if counts["missing"]:
        err.print("[yellow]some sessions have no file (latest session may be unpublished)[/yellow]")


@data.command("ref")
def data_ref(
    offline: Annotated[
        bool, typer.Option(help="Rebuild ref/ tables from cached JSON only.")
    ] = False,
) -> None:
    """Fetch and cache tickers, splits, dividends and SIC codes, then build the ref/ tables."""
    from alpha_factory.data.clients import RestClient
    from alpha_factory.data.reference import build_ref_tables, fetch_reference

    settings, cfg, layout = _data_context()
    if not offline:
        _need(settings, "massive_api_key")
        assert settings.massive_api_key
        client = RestClient(settings.massive_api_key.get_secret_value(), cfg.download.retries)
        fetch_reference(
            client, layout, datetime.date.today(), cfg.start, cfg.types, cfg.download.concurrency
        )
    counts = build_ref_tables(layout)
    console.print(f"ref tables: {counts}")


@data.command("build")
def data_build(
    kind: Annotated[KindChoice, typer.Option(help="dev, full or both.")] = KindChoice.both,
) -> None:
    """Build bars, factors, universe and sectors, then write the manifest. `both` also runs QA."""
    from alpha_factory.data.build import build_dataset
    from alpha_factory.hashing import sha256_file

    settings, cfg, layout = _data_context()
    params_hash = sha256_file(settings.config_dir / "data.yaml")
    t0 = time.perf_counter()
    for k in _kinds(kind):
        m = build_dataset(layout, cfg, k, params_hash)  # pyright: ignore[reportArgumentType]
        console.print(f"{k}: {m.dataset_id}  {m.n_sessions} sessions, {m.n_tickers} instruments")
    console.print(f"built in {time.perf_counter() - t0:.0f}s")
    if kind is KindChoice.both:
        _run_qa(cfg, layout, ["dev", "full"])
    else:
        console.print("run `af data build --kind both` or `af data qa` after building both kinds")


def _run_qa(cfg: "DataConfig", layout: "DataLayout", kinds: list[str]) -> None:
    from alpha_factory.data.qa import run_qa

    failed = False
    for k in kinds:
        try:
            report = run_qa(layout, cfg, k)  # pyright: ignore[reportArgumentType]
        except FileNotFoundError as e:
            err.print(f"[red]{e}[/red]")
            raise typer.Exit(1) from e
        table = Table("check", "passed", "value", "threshold", "detail", title=f"QA: {k}")
        for c in report.checks:
            table.add_row(
                c.name,
                "yes" if c.passed else "[red]NO[/red]",
                f"{c.value:.6g}",
                f"{c.threshold:.6g}",
                c.detail[:100],
            )
        console.print(table)
        failed |= not report.passed
    if failed:
        raise typer.Exit(1)


@data.command("qa")
def data_qa(
    kind: Annotated[KindChoice, typer.Option(help="dev, full or both.")] = KindChoice.both,
) -> None:
    """Run QA on the current build(s) and write qa.json. Exits 1 if any check fails."""
    _, cfg, layout = _data_context()
    _run_qa(cfg, layout, ["dev", "full"] if kind is KindChoice.both else [kind.value])


@data.command("manifest")
def data_manifest() -> None:
    """Show the current builds' manifests and hashes."""
    from alpha_factory.data.build import read_manifest

    _, _, layout = _data_context()
    for k in ("dev", "full"):
        b = layout.current_build(k)  # pyright: ignore[reportArgumentType]
        if b is None:
            console.print(f"{k}: no build")
            continue
        m = read_manifest(b)
        qa = b / "qa.json"
        status = "no QA"
        if qa.exists():
            from alpha_factory.contracts.data import QAReport

            status = (
                "QA passed" if QAReport.model_validate_json(qa.read_text()).passed else "QA FAILED"
            )
        console.print(
            f"{k}: {m.dataset_id}  {m.start}..{m.end}  {m.n_sessions} sessions  "
            f"{m.n_tickers} instruments  {len(m.files)} files  {status}"
        )
        console.print(f"    dataset_hash {m.dataset_hash}")
        console.print(f"    params_hash  {m.params_hash}  commit {m.code_commit}")


@data.command("verify")
def data_verify() -> None:
    """Re-hash every file of the current builds against their manifests."""
    from alpha_factory.data.build import verify_build

    _, _, layout = _data_context()
    bad = False
    for k in ("dev", "full"):
        b = layout.current_build(k)  # pyright: ignore[reportArgumentType]
        if b is None:
            console.print(f"{k}: no build")
            continue
        problems = verify_build(b)
        console.print(f"{k}: {'ok' if not problems else f'{len(problems)} problems'}")
        for p in problems[:20]:
            err.print(f"  {p}")
        bad |= bool(problems)
    if bad:
        raise typer.Exit(1)


@data.command("publish")
def data_publish() -> None:
    """Not specified in Plan 2."""
    err.print("[yellow]`af data publish` is not specified in any plan yet.[/yellow]")
    raise typer.Exit(STUB_EXIT_CODE)


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
