"""Summarize a context-budget benchmark run directory."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from context_budget_lab.metrics import format_table, write_summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args(argv)
    rows = write_summary(args.run_dir)
    print(format_table(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
