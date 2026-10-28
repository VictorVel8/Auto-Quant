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
    from alpha_factory.contracts.sandbox import SandboxSpec
    from alpha_factory.data import MarketData
    from alpha_factory.data.config import DataConfig
    from alpha_factory.data.layout import DataLayout
    from alpha_factory.eval.context import EvalContext

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


class HorizonChoice(StrEnum):
    h1 = "h1"
    h2 = "h2"
    h4 = "h4"


def _eval_context(settings: Settings) -> "tuple[MarketData, SandboxSpec, EvalContext]":
    """(MarketData, SandboxSpec, EvalContext) on the dev dataset."""
    from alpha_factory.data import DatasetNotReadyError, MarketData
    from alpha_factory.eval.context import build_context
    from alpha_factory.sandbox import SandboxConfigError, load_sandbox

    try:
        data = MarketData("dev", settings=settings)
        sandbox = load_sandbox(settings, dataset_hash=data.manifest.dataset_hash)
    except (DatasetNotReadyError, SandboxConfigError, OSError) as e:
        err.print(f"[red]{e}[/red]")
        raise typer.Exit(1) from e
    return data, sandbox, build_context(data, sandbox)


@evaluate.command("factor", context_settings={"ignore_unknown_options": True})
def eval_factor(
    expr: Annotated[str, typer.Argument(help="DSL expression.")],
    horizon: Annotated[HorizonChoice, typer.Option(help="Label horizon.")] = HorizonChoice.h1,
    sign: Annotated[int, typer.Option(help="+1 or -1: the sign of the expected IC.")] = 1,
    n_trials: Annotated[int, typer.Option(help="Trials evaluated so far (sets the t bar).")] = 1,
    condition: Annotated[str | None, typer.Option(help="Optional bool expression.")] = None,
) -> None:
    """Evaluate one expression on the search folds. For exploring by hand: nothing is recorded."""
    from alpha_factory.dsl import DSLError, compile_expr
    from alpha_factory.eval.calibration import load_calibration
    from alpha_factory.eval.evaluate import TrialInfo
    from alpha_factory.eval.evaluate import evaluate as run_eval
    from alpha_factory.eval.pool import Pool

    if sign not in (1, -1):
        err.print("[red]--sign must be 1 or -1[/red]")
        raise typer.Exit(1)
    try:
        factor = compile_expr(expr, condition=condition)
    except DSLError as e:
        err.print(f"[red]{e}[/red]")
        raise typer.Exit(1) from e
    settings = Settings()
    _, sandbox, ctx = _eval_context(settings)
    cal = load_calibration(settings.data_dir, sandbox.content_hash(), ctx.dataset_hash)
    floor = {str(k): v for k, v in cal.ic_floor.items()} if cal else None
    t0 = time.perf_counter()
    r = run_eval(
        factor,
        horizon.value,
        sign=1 if sign > 0 else -1,
        pool=Pool(),
        sandbox=sandbox,
        context=ctx,
        trial=TrialInfo(n_trials=n_trials, null_floor=floor),
    )
    table = Table("field", "value", title=f"{factor.expression}  ({horizon.value}, sign {sign:+d})")
    rows = [
        (
            "verdict",
            f"{r.reason.value}" + (f"  flags: {[f.value for f in r.flags]}" if r.flags else ""),
        ),
        ("ic_mean / t-stat", f"{r.ic_mean:+.4f} / {r.ic_tstat:+.2f}"),
        ("worst fold / icir", f"{r.ic_worst_fold:+.4f} / {r.icir:.3f}"),
        ("folds", ", ".join(f"{f.fold_id}: {f.ic_mean:+.4f}" for f in r.folds)),
        ("by year", ", ".join(f"{y}: {v:+.4f}" for y, v in sorted(r.ic_by_year.items()))),
        ("by horizon", ", ".join(f"{h}: {v:+.4f}" for h, v in r.ic_by_horizon.items())),
        ("incremental IC", f"{r.incremental_ic:+.4f}"),
        ("turnover", f"{r.turnover:.3f}"),
        ("net Sharpe (val)", "-" if r.net_sharpe_val is None else f"{r.net_sharpe_val:+.2f}"),
        ("regimes", ", ".join(f"{k}: {v:+.4f}" for k, v in r.ic_by_regime.items())),
        (
            "null floor",
            "no calibration saved" if cal is None else f"{cal.ic_floor[horizon.value]:.4f}",
        ),
        ("hypothesis refuted", str(r.hypothesis_refuted)),
        ("seconds", f"{time.perf_counter() - t0:.1f}"),
    ]
    for k, v in rows:
        table.add_row(k, v)
    console.print(table)


@evaluate.command("null")
def eval_null(
    n: Annotated[int, typer.Option("--n", help="Random expressions to evaluate.")] = 1000,
    seed: Annotated[int, typer.Option(help="RNG seed.")] = 0,
) -> None:
    """Null calibration: IC floor per horizon from random formulas, plus the label-shuffle check."""
    from alpha_factory.eval.calibration import (
        SHUFFLE_TOLERANCE,
        label_shuffle_check,
        run_null,
        save_calibration,
    )

    settings = Settings()
    _, _, ctx = _eval_context(settings)
    t0 = time.perf_counter()

    def progress(i: int) -> None:
        if i % 50 == 0:
            console.print(f"  {i}/{n}")

    cal = run_null(ctx, n=n, seed=seed, progress=progress)
    shuffle = label_shuffle_check(ctx, "h1", seed=seed)
    cal = cal.model_copy(update={"shuffle_mean_ic": shuffle})
    path = save_calibration(cal, settings.data_dir)
    console.print(f"IC floors (99th percentile of |IC| over {n} random formulas): {cal.ic_floor}")
    ok = abs(shuffle) <= SHUFFLE_TOLERANCE
    console.print(
        f"label shuffle mean IC {shuffle:+.5f} "
        f"({'ok' if ok else 'FAILED'}, limit {SHUFFLE_TOLERANCE})"
    )
    console.print(f"saved {path} in {time.perf_counter() - t0:.0f}s")
    if not ok:
        raise typer.Exit(1)


@evaluate.command("canary")
def eval_canary() -> None:
    """Leakage canary: a deliberately leaky feature must score high and be flagged LOOKAHEAD."""
    from alpha_factory.eval.calibration import CANARY_MULTIPLE, load_calibration, run_canary

    settings = Settings()
    data, sandbox, ctx = _eval_context(settings)
    cal = load_calibration(settings.data_dir, sandbox.content_hash(), ctx.dataset_hash)
    if cal is None:
        err.print("[red]run `af eval null` first: the canary is compared with the null floor[/red]")
        raise typer.Exit(1)
    report = run_canary(ctx, data, cal.ic_floor["h1"])
    console.print(
        f"canary IC {report.ic:+.4f} vs null floor {report.floor:.4f}: {report.ratio:.1f}x"
    )
    console.print(
        f"above {CANARY_MULTIPLE:.0f}x the floor: {report.scores_high}; "
        f"flagged LOOKAHEAD: {report.flagged_lookahead}"
    )
    if not (report.scores_high and report.flagged_lookahead):
        err.print(
            "[red]the canary did not behave like a leak: the pipeline cannot be trusted yet[/red]"
        )
        raise typer.Exit(1)


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
    """Sealed test evaluation. final_eval() exists (Plan 3); loading LIB needs Plan 4."""
    err.print(
        "[yellow]`af final-eval` needs the frozen library and combiner from Plan 4. The sealed "
        "evaluation itself is alpha_factory.sandbox.final_eval.final_eval.[/yellow]"
    )
    raise typer.Exit(STUB_EXIT_CODE)
