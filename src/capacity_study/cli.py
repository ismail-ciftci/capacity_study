"""Command line: python -m capacity_study <command> ..."""
import argparse
from pathlib import Path

from capacity_study.config import load_config


def _study(p: argparse.ArgumentParser) -> None:
    p.add_argument("--study", type=Path, required=True, help="study directory, e.g. runs/main")


def _code_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--allow-dirty", action="store_true", help="run on an uncommitted tree (diff is recorded)")
    p.add_argument("--allow-code-change", action="store_true",
                   help="resume checkpoints written by another commit (recorded in events)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="capacity_study")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("fetch-data", help="one-time price download (refuses to overwrite)")
    p.add_argument("--config", type=Path, required=True)

    p = sub.add_parser("prepare", help="build the feature bank from the raw snapshot")
    p.add_argument("--config", type=Path, required=True)

    p = sub.add_parser("init-study", help="create a study directory bound to a config")
    p.add_argument("--config", type=Path, required=True)
    _study(p)
    p.add_argument("--allow-dirty", action="store_true")

    p = sub.add_parser("enqueue", help="add a phase's jobs to the ledger (tuning also adds final jobs)")
    _study(p)
    p.add_argument("phase", choices=["calibration", "tuning"])

    p = sub.add_parser("freeze-ranges", help="derive main-run tuning ranges from completed calibration")
    _study(p)

    p = sub.add_parser("launch", help="reconcile the ledger and start workers")
    _study(p)
    p.add_argument("--devices", default="cuda:0", help="comma-separated, e.g. cuda:0,cuda:1")
    p.add_argument("--workers-per-device", type=int, required=True)
    p.add_argument("--mem-budget-mb", type=float, default=None, help="per-device budget for job estimates")
    p.add_argument("--stale-seconds", type=float, default=600.0)
    _code_flags(p)

    p = sub.add_parser("worker", help="(spawned by launch) claim and run jobs until none remain")
    _study(p)
    p.add_argument("--device", required=True)
    p.add_argument("--mem-budget-mb", type=float, required=True)
    p.add_argument("--max-consecutive-failures", type=int, default=3)
    p.add_argument("--idle-poll-seconds", type=float, default=10.0)
    p.add_argument("--heartbeat-seconds", type=float, default=15.0)
    _code_flags(p)

    p = sub.add_parser("pause", help="ask workers to checkpoint and exit at the next epoch end")
    _study(p)

    p = sub.add_parser("resume", help="clear the pause flag and launch")
    _study(p)
    p.add_argument("--devices", default="cuda:0")
    p.add_argument("--workers-per-device", type=int, required=True)
    p.add_argument("--mem-budget-mb", type=float, default=None)
    p.add_argument("--stale-seconds", type=float, default=600.0)
    _code_flags(p)

    p = sub.add_parser("status", help="job counts, running jobs, failures")
    _study(p)

    p = sub.add_parser("retry", help="move a failed job back to pending")
    _study(p)
    p.add_argument("job_id")

    p = sub.add_parser("bench", help="measure throughput and memory for worker-count selection")
    _study(p)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--workers", default="1,2,4,6,8")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from capacity_study import orchestrate
    from capacity_study.store import StudyPaths

    if args.command == "fetch-data":
        from capacity_study.data.fetch import fetch_prices
        print(fetch_prices(load_config(args.config)))
    elif args.command == "prepare":
        from capacity_study.data.bank import prepare_bank
        print(prepare_bank(load_config(args.config)))
    elif args.command == "init-study":
        print(orchestrate.init_study(args.config, args.study, args.allow_dirty))
    elif args.command == "enqueue":
        print(f"added {orchestrate.enqueue(args.study, args.phase)} jobs")
    elif args.command == "freeze-ranges":
        print(orchestrate.freeze(args.study))
    elif args.command in ("launch", "resume"):
        if args.command == "resume":
            StudyPaths(args.study).pause_flag.unlink(missing_ok=True)
        devices = [d.strip() for d in args.devices.split(",") if d.strip()]
        return orchestrate.launch(args.study, devices, args.workers_per_device, args.mem_budget_mb,
                                  args.allow_dirty, args.allow_code_change, args.stale_seconds)
    elif args.command == "worker":
        return orchestrate.worker_main(args.study, args.device, args.mem_budget_mb, args.allow_dirty,
                                       args.allow_code_change, args.max_consecutive_failures,
                                       args.idle_poll_seconds, args.heartbeat_seconds)
    elif args.command == "pause":
        flag = StudyPaths(args.study).pause_flag
        flag.write_text("pause requested\n")
        print(f"wrote {flag}; workers stop at their next epoch end")
    elif args.command == "status":
        print(orchestrate.status(args.study))
    elif args.command == "retry":
        from capacity_study.ledger import Ledger
        ledger = Ledger(StudyPaths(args.study).ledger)
        try:
            ledger.retry(args.job_id)
        finally:
            ledger.close()
    elif args.command == "bench":
        from capacity_study.bench import bench
        print(bench(args.study, args.device, args.epochs, [int(w) for w in args.workers.split(",")]))
    return 0
