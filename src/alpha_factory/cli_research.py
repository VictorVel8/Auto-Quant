"""af loop | goal | lib | family | report | final-eval, and af data publish."""

import datetime
import signal
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.table import Table

from alpha_factory.settings import MissingSettingError, Settings


def register(app: typer.Typer, data_group: typer.Typer, console: Console, err: Console) -> None:
    def group(name: str, help_text: str) -> typer.Typer:
        g = typer.Typer(help=help_text, no_args_is_help=True)
        app.add_typer(g, name=name)
        return g

    def fail(message: str) -> typer.Exit:
        err.print(f"[red]{message}[/red]")
        return typer.Exit(1)

    def store_of(settings: Settings) -> Any:
        from alpha_factory.research.store import ResearchStore

        return ResearchStore(settings.store_path)

    # --- loop ------------------------------------------------------------------------------
    loop = group("loop", "Run and control the research loop.")

    @loop.command("run")
    def loop_run(
        once: Annotated[bool, typer.Option(help="Run one cycle and exit.")] = False,
    ) -> None:
        """Run the orchestrator (a systemd service on the mini PC). One loop per store."""
        from alpha_factory.orchestrator import LoopAlreadyRunningError, LoopLock
        from alpha_factory.research.runtime import build_runtime

        settings = Settings()
        try:
            with LoopLock(settings.store_path):
                rt = build_runtime(settings)
                stop = {"now": False}
                signal.signal(signal.SIGTERM, lambda *_: stop.update(now=True))
                signal.signal(signal.SIGINT, lambda *_: stop.update(now=True))
                if once:
                    console.print("progressed" if rt.orchestrator.run_once() else "nothing to do")
                else:
                    console.print(
                        f"loop running (proposer: {rt.cfg.proposer}, backend: {rt.cfg.evaluation.backend})"
                    )
                    rt.orchestrator.run_forever(should_stop=lambda: stop["now"])
        except LoopAlreadyRunningError as e:
            raise fail(str(e)) from e
        except Exception as e:  # missing secrets, no dev build, sandbox mismatch, ...
            if isinstance(
                e, MissingSettingError | OSError | ValueError | RuntimeError | PermissionError
            ):
                raise fail(str(e)) from e
            raise

    @loop.command("status")
    def loop_status() -> None:
        """The active goal, the batch state, today's spend and trial counts."""
        s = store_of(Settings())
        from alpha_factory.contracts import ResearchGoal

        gid = s.active_goal_id()
        t = Table("field", "value", title="Loop status")
        t.add_row(
            "paused", f"{s.paused()}" + (f" ({s.get_flag('paused_reason')})" if s.paused() else "")
        )
        if gid:
            g = next(x for x in s.goals() if x["goal_id"] == gid)
            goal = ResearchGoal.model_validate_json(g["goal_json"])
            t.add_row(
                "active goal", f"{goal.title} ({g['trials_used']}/{goal.budget_trials} trials)"
            )
        else:
            t.add_row("active goal", "none (approve one with `af goal approve`)")
        unfinished = s.unfinished_batches()
        t.add_row(
            "batch in progress",
            f"{unfinished[0]['batch_id']} at {unfinished[0]['state']}" if unfinished else "none",
        )
        t.add_row("batches today", str(s.batches_today()))
        t.add_row("LLM spend today", f"${s.spend_today():.2f}")
        t.add_row("trials", f"{s.total_trials()} total; by reason {s.reason_counts()}")
        t.add_row("pool", str(len(s.pool())))
        console.print(t)

    @loop.command("pause")
    def loop_pause() -> None:
        """Set the pause flag; the loop checks it between states."""
        store_of(Settings()).set_flag("paused", "1")
        console.print("paused")

    @loop.command("resume")
    def loop_resume() -> None:
        s = store_of(Settings())
        s.set_flag("paused", "0")
        console.print("resumed")

    @loop.command("reeval")
    def loop_reeval(trial_id: Annotated[str, typer.Argument()]) -> None:
        """Re-evaluate a stored trial from its stored inputs and compare the numbers."""
        from alpha_factory.orchestrator.ops import reevaluate
        from alpha_factory.research.runtime import build_runtime

        rt = build_runtime(Settings())
        r = reevaluate(
            rt.store, rt.orchestrator.backend, rt.sandbox, rt.data.manifest.dataset_hash, trial_id
        )
        console.print("identical" if r.identical else f"DIFFERENT: {r.differences}")
        if not r.identical:
            raise typer.Exit(1)

    # --- goals ---------------------------------------------------------------------------------
    goal = group("goal", "Research goals: drafted by a manager model, approved by you.")

    @goal.command("brief")
    def goal_brief(
        out: Annotated[Path, typer.Option(help="Where to write the brief.")] = Path("brief.md"),
    ) -> None:
        """Write brief.md from the store (no sealed numbers), for the manager or a chat model."""
        from alpha_factory.agents.manager import brief

        out.write_text(brief(store_of(Settings())))
        console.print(f"wrote {out}")

    @goal.command("propose")
    def goal_propose() -> None:
        """Send the brief to the manager model and save one to three pending goals."""
        from alpha_factory.agents.manager import propose_goals
        from alpha_factory.research.runtime import load_llm_config, make_llm_client

        settings = Settings()
        client = make_llm_client(settings, load_llm_config(settings))
        if client is None:
            try:
                settings.require("llm_base_url", "llm_api_key")
            except MissingSettingError as e:
                raise fail(str(e)) from e
            raise fail("no LLM client")
        for gid in propose_goals(client, store_of(settings)):
            console.print(f"pending goal {gid}")

    @goal.command("add")
    def goal_add(file: Annotated[Path, typer.Argument()]) -> None:
        """Add goals from a YAML file (for example goals/bootstrap.yaml)."""
        from alpha_factory.agents.manager import goals_from_file

        for gid in goals_from_file(store_of(Settings()), file):
            console.print(f"pending goal {gid}")

    @goal.command("approve")
    def goal_approve(goal_id: str) -> None:
        s = store_of(Settings())
        try:
            s.approve_goal(goal_id)
        except KeyError as e:
            raise fail(f"no pending goal {goal_id}") from e
        console.print(
            "active" if s.active_goal_id() == goal_id else "approved, queued behind the active goal"
        )

    @goal.command("close")
    def goal_close(
        goal_id: str,
        abandon: Annotated[bool, typer.Option(help="Mark abandoned, not done.")] = False,
    ) -> None:
        s = store_of(Settings())
        s.close_goal(goal_id, "abandoned" if abandon else "done", "closed by you")
        s.activate_next_goal()
        console.print("closed")

    @goal.command("list")
    def goal_list() -> None:
        from alpha_factory.contracts import ResearchGoal

        s = store_of(Settings())
        t = Table("id", "status", "title", "trials", "accepted", title="Goals")
        for g in s.goals():
            goal_ = ResearchGoal.model_validate_json(g["goal_json"])
            t.add_row(
                g["goal_id"],
                g["status"],
                goal_.title,
                f"{g['trials_used']}/{goal_.budget_trials}",
                str(s.goal_acceptances(g["goal_id"])),
            )
        console.print(t)

    # --- families and libraries -------------------------------------------------------------------
    family = group("family", "Mechanism families.")

    @family.command("activate")
    def family_activate(name: str) -> None:
        """Reactivate an exhausted or refuted family when a new idea warrants it."""
        s = store_of(Settings())
        s.set_family_status(name, "active")
        console.print(f"{name} is active")

    lib = group("lib", "Factor libraries: frozen snapshots of the pool plus a combiner.")

    @lib.command("freeze")
    def lib_freeze(
        combiner: Annotated[str, typer.Option(help="rank_average or ridge.")] = "rank_average",
        horizon: Annotated[str, typer.Option(help="Target horizon of the combiner.")] = "h2",
    ) -> None:
        """Freeze the active pool, report the combined signal on F1-F4 and the holdout (for you only)."""
        from alpha_factory.contracts import CombinerSpec, FrozenLibrary
        from alpha_factory.eval.context import build_context
        from alpha_factory.eval.library import combined_report
        from alpha_factory.research.runtime import dev_world

        settings = Settings()
        s = store_of(settings)
        pool = s.pool()
        if not pool:
            raise fail("the pool is empty: nothing to freeze")
        data, sandbox = dev_world(settings)
        spec = CombinerSpec.model_validate({"kind": combiner, "target_horizon": horizon})
        frozen = FrozenLibrary(
            lib_version=s.next_lib_version(), sandbox_hash=sandbox.content_hash(),
            factors=tuple(p.lib_factor() for p in pool), combiner=spec,
            frozen_at=datetime.datetime.now(datetime.UTC),
        )  # fmt: skip
        report = combined_report(
            frozen,
            sandbox,
            build_context(data, sandbox),
            build_context(data, sandbox, "holdout"),
            data,
        )
        s.add_library(frozen, report.model_dump_json())
        path = Path("libs") / f"{frozen.lib_version}.json"
        path.parent.mkdir(exist_ok=True)
        path.write_text(frozen.model_dump_json(indent=2))
        console.print(f"froze {frozen.lib_version} ({len(pool)} factors, {combiner}) -> {path}")
        console.print(report.model_dump_json(indent=2))
        console.print(f"spend the sealed test with:  af final-eval {frozen.lib_version}")

    @lib.command("import-final")
    def lib_import_final(file: Path, lib_version: Annotated[str, typer.Option("--lib")]) -> None:
        """Record a final-evaluation report file in the store's final_evals table."""
        from alpha_factory.contracts import FinalEvalReport

        report = FinalEvalReport.model_validate_json(file.read_text())
        s = store_of(Settings())
        s.record_final_eval(lib_version, report)
        console.print(
            f"recorded; the sealed window has now been used {len(s.final_evals())} time(s)"
        )

    # --- reports -------------------------------------------------------------------------------------
    report = group("report", "Reports from the research store.")

    def combined_ic_fn(settings: Settings) -> Any:
        """Combined validation IC of a set of accepted factors, when dev data is available."""
        try:
            from alpha_factory.compute.specs import PoolLooRequest  # noqa: F401
            from alpha_factory.compute.worker import build_pool  # noqa: F401
            from alpha_factory.dsl import compile_expr
            from alpha_factory.eval.combine import rank_average
            from alpha_factory.eval.context import build_context
            from alpha_factory.eval.evaluate import search_daily_ic
            from alpha_factory.eval.pool import Pool, PoolEntry
            from alpha_factory.research.runtime import dev_world

            data, sandbox = dev_world(settings)
            ctx = build_context(data, sandbox)
        except Exception:
            return None

        def fn(factors: list[Any]) -> float | None:
            pool = Pool(
                tuple(
                    PoolEntry(f.factor_id, compile_expr(f.canonical_expr), f.sign) for f in factors
                )
            )
            frames = {e.factor_id: pool.scores(e, ctx.panel.df) for e in pool.entries}
            combined = rank_average(frames, {e.factor_id: e.sign for e in pool.entries})
            daily = search_daily_ic(combined, ctx, "h1", 1)
            return float(daily["ic"].mean()) if daily.height else None  # type: ignore[arg-type]

        return fn

    @report.command("run")
    def report_run(out: Annotated[Path | None, typer.Option(help="Output file.")] = None) -> None:
        """A single static HTML file in reports/."""
        from alpha_factory.research.report import ab_comparison, render_html

        s = store_of(Settings())
        target = out or Path("reports") / f"report_{datetime.date.today()}.html"
        target.parent.mkdir(exist_ok=True)
        ab = (
            ab_comparison(s) if s.proposer_trials("llm") and s.proposer_trials("template") else None
        )
        target.write_text(render_html(s, ab))
        console.print(f"wrote {target}")

    @report.command("trials")
    def report_trials(
        goal_id: Annotated[str | None, typer.Option("--goal")] = None,
        reason: Annotated[str | None, typer.Option()] = None,
        family: Annotated[str | None, typer.Option()] = None,
        limit: int = 40,
    ) -> None:
        s = store_of(Settings())
        t = Table(
            "trial", "reason", "family", "horizon", "sign", "source", title="Trials (newest first)"
        )
        for tr in s.trials_filtered(goal_id, reason, family, limit):
            t.add_row(
                tr.trial_id,
                tr.reason.value,
                tr.family or "",
                tr.horizon or "",
                str(tr.sign_used or ""),
                tr.source,
            )
        console.print(t)

    @report.command("beliefs")
    def report_beliefs() -> None:
        s = store_of(Settings())
        t = Table(
            "id",
            "status",
            "family",
            "horizon",
            "evidence",
            "statement",
            "supersedes",
            title="Beliefs",
        )
        for b in s.beliefs(include_retired=True):
            status = "retired" if b.retired else b.status.value
            t.add_row(
                b.belief_id,
                status,
                b.family,
                b.horizon,
                f"{b.n_support} for, {b.n_contradict} against",
                b.statement,
                b.supersedes or "",
            )
        console.print(t)

    @report.command("ab")
    def report_ab() -> None:
        """Template versus LLM at equal trial budgets: acceptance with Wilson intervals, spend."""
        from alpha_factory.research.report import ab_comparison

        settings = Settings()
        s = store_of(settings)
        if not (s.proposer_trials("llm") and s.proposer_trials("template")):
            raise fail("need evaluated trials from both proposers to compare them")
        t = Table(
            "proposer",
            "trials",
            "accepted",
            "rate (95% CI)",
            "best combined IC",
            "USD / acceptance",
            title="Equal-budget comparison",
        )
        for r in ab_comparison(s, combined_ic_fn(settings)):
            t.add_row(r.proposer, str(r.evaluated), str(r.accepted), f"{r.rate:.1%} ({r.ci[0]:.1%} to {r.ci[1]:.1%})",
                      "n/a" if r.best_combined_ic is None else f"{r.best_combined_ic:.4f}",
                      "n/a" if r.spend_per_acceptance is None else f"{r.spend_per_acceptance:.2f}")  # fmt: skip
        console.print(t)

    # --- the sealed final evaluation -------------------------------------------------------------------
    @app.command("final-eval")
    def final_eval_cmd(
        lib_version: Annotated[str, typer.Argument(metavar="LIB")],
        local: Annotated[
            bool, typer.Option(help="Use the full dataset on this machine, not Modal.")
        ] = False,
        libs_dir: Annotated[Path, typer.Option(help="Where libs/lib_vN.json lives.")] = Path(
            "libs"
        ),
        dev_dataset_hash: Annotated[
            str | None, typer.Option(help="Hash of the dev build the sandbox was frozen on.")
        ] = None,
    ) -> None:
        """Score a frozen library once on the sealed window. Run it from your MacBook."""
        import os

        from alpha_factory.contracts import FinalEvalReport, FinalEvalRequest, FrozenLibrary
        from alpha_factory.data import DatasetNotReadyError, MarketData
        from alpha_factory.eval.library import LibraryScorer, combiner_hash
        from alpha_factory.sandbox import SandboxConfigError, load_sandbox
        from alpha_factory.sandbox.final_eval import FinalEvalError, FinalEvalLedger, final_eval

        settings = Settings()
        path = libs_dir / f"{lib_version}.json"
        if not path.exists():
            raise fail(f"no library file {path}; freeze one with `af lib freeze`")
        lib = FrozenLibrary.model_validate_json(path.read_text())
        dev_hash = dev_dataset_hash
        if dev_hash is None:
            try:
                dev_hash = MarketData("dev", settings=settings).manifest.dataset_hash
            except DatasetNotReadyError as e:
                raise fail(f"{e}; pass --dev-dataset-hash") from e
        try:
            sandbox = load_sandbox(settings, dataset_hash=dev_hash)
        except SandboxConfigError as e:
            raise fail(str(e)) from e
        if sandbox.content_hash() != lib.sandbox_hash:
            raise fail("the library was frozen under a different sandbox")
        request = FinalEvalRequest(
            lib_version=lib.lib_version,
            combiner_hash=combiner_hash(lib),
            sandbox_hash=sandbox.content_hash(),
        )
        try:
            if local:
                full = MarketData("full", settings=settings, sealed=True)
                ledger = FinalEvalLedger(settings.data_dir / "final_eval_ledger.jsonl")
                result = final_eval(
                    request,
                    library=LibraryScorer(lib, sandbox, full),
                    ledger=ledger,
                    sandbox=sandbox,
                    data=full,
                )
                uses = ledger.count()
            else:
                import modal

                from alpha_factory.compute.modal_sealed import FULL_APP

                fn = modal.Function.from_name(
                    FULL_APP,
                    "final_eval_remote",
                    environment_name=os.environ.get("AF_MODAL_ENVIRONMENT", "sealed"),
                )
                result = FinalEvalReport.model_validate_json(
                    fn.remote(request.model_dump_json(), lib.model_dump_json(), dev_hash)
                )
                uses = -1
        except (FinalEvalError, DatasetNotReadyError, PermissionError) as e:
            raise fail(str(e)) from e
        out = Path("reports") / f"final_eval_{lib.lib_version}.json"
        out.parent.mkdir(exist_ok=True)
        out.write_text(result.model_dump_json(indent=2))
        console.print(result.model_dump_json(indent=2))
        console.print(
            f"wrote {out}; record it with `af lib import-final {out} --lib {lib.lib_version}`"
        )
        if uses > 0:
            console.print(
                f"the sealed window has now been used {uses} time(s): each use weakens it"
            )

    # --- data publish ---------------------------------------------------------------------------------
    @data_group.command("publish")
    def data_publish(
        kind: Annotated[
            str, typer.Option(help="dev (full is uploaded only from the MacBook).")
        ] = "dev",
    ) -> None:
        """Upload a build and the null calibration to its Modal volume."""
        import os

        from alpha_factory.compute.modal_dev import DEV_VOLUME, publish_build
        from alpha_factory.data.layout import DataLayout
        from alpha_factory.eval.calibration import calibration_path
        from alpha_factory.research.runtime import dev_world
        from alpha_factory.sandbox import SandboxConfigError

        settings = Settings()
        if kind not in ("dev", "full"):
            raise fail("--kind must be dev or full")
        build = DataLayout(settings.data_dir).current_build(kind)  # type: ignore[arg-type]
        if build is None:
            raise fail(f"no {kind} build; run `af data build`")
        cal = None
        if kind == "dev":
            try:
                _, sandbox = dev_world(settings)
                p = calibration_path(settings.data_dir, sandbox.content_hash())
                cal = p if p.exists() else None
            except (SandboxConfigError, OSError, PermissionError) as e:
                err.print(f"[yellow]no calibration uploaded: {e}[/yellow]")
        env = os.environ.get("AF_MODAL_ENVIRONMENT", kind if kind == "dev" else "sealed")
        volume = DEV_VOLUME
        if kind == "full":
            from alpha_factory.compute.modal_sealed import FULL_VOLUME

            volume = FULL_VOLUME
        published = publish_build(build, kind, env, volume_name=volume, calibration=cal)
        console.print(f"published {published} to {volume} in '{env}'")
