"""Run the context benchmark as one run directory per model/strategy condition."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.make_plots import main as plot_main
from benchmarks.run_context_benchmark import DEFAULT_DATASETS, DEFAULT_STRATEGIES, main as run_benchmark
from context_budget_lab.metrics import write_summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:11434/v1")
    parser.add_argument("--models", default="qwen2.5:3b")
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--strategies", default=",".join(DEFAULT_STRATEGIES))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=180)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args(argv)

    run_dirs: list[Path] = []
    for model in _split_csv(args.models):
        for strategy in _split_csv(args.strategies):
            condition_out = args.out / _slug(model if not args.mock else "simulated") / _slug(strategy)
            before = _run_dirs(condition_out)
            run_args = [
                "--base-url",
                args.base_url,
                "--model",
                model,
                "--datasets",
                args.datasets,
                "--strategies",
                strategy,
                "--out",
                str(condition_out),
                "--max-tokens",
                str(args.max_tokens),
                "--temperature",
                str(args.temperature),
                "--repeats",
                str(args.repeats),
                "--seed",
                str(args.seed),
            ]
            if args.limit is not None:
                run_args.extend(["--limit", str(args.limit)])
            if args.mock:
                run_args.append("--mock")

            exit_code = run_benchmark(run_args)
            if exit_code != 0:
                return exit_code

            run_dir = _new_run_dir(condition_out, before)
            write_summary(run_dir)
            plot_main([str(run_dir)])
            run_dirs.append(run_dir)

    for run_dir in run_dirs:
        print(run_dir)
    return 0


def _run_dirs(path: Path) -> set[Path]:
    if not path.exists():
        return set()
    return {child for child in path.iterdir() if child.is_dir()}


def _new_run_dir(path: Path, before: set[Path]) -> Path:
    after = _run_dirs(path)
    new_dirs = sorted(after - before, key=lambda item: item.stat().st_mtime)
    if new_dirs:
        return new_dirs[-1]
    existing = sorted(after, key=lambda item: item.stat().st_mtime)
    if existing:
        return existing[-1]
    raise RuntimeError(f"benchmark did not create a run directory under {path}")


def _slug(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in value).strip("-")


def _split_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


if __name__ == "__main__":
    raise SystemExit(main())
