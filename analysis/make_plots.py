"""Create a dependency-free SVG chart from a benchmark summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from context_budget_lab.metrics import write_summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    summary_path = args.run_dir / "summary.json"
    if summary_path.exists():
        rows = json.loads(summary_path.read_text(encoding="utf-8"))
    else:
        rows = write_summary(args.run_dir)
    out = args.out or args.run_dir / "latency_quality.svg"
    out.write_text(_svg(rows), encoding="utf-8")
    print(out)
    return 0


def _svg(rows: list[dict[str, Any]]) -> str:
    width = 920
    height = 360
    margin_left = 180
    bar_height = 28
    gap = 18
    max_latency = max((row.get("avg_latency_s") or 0.001 for row in rows), default=0.001)
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fafafa"/>',
        '<text x="24" y="36" font-family="Arial, sans-serif" font-size="20" font-weight="700">Context Strategy Smoke: Latency and Quality</text>',
        '<text x="24" y="62" font-family="Arial, sans-serif" font-size="13" fill="#555">Bars show average latency; labels show answer/citation/evidence quality.</text>',
    ]
    for idx, row in enumerate(rows):
        y = 96 + idx * (bar_height + gap)
        latency = row.get("avg_latency_s") or 0.0
        answer = row.get("avg_answer_score")
        citation = row.get("avg_citation_score")
        evidence = row.get("avg_evidence_recall")
        bar_width = int((latency / max_latency) * 560) if max_latency else 0
        lines.extend(
            [
                f'<text x="24" y="{y + 20}" font-family="Arial, sans-serif" font-size="13">{_escape(str(row["strategy"]))}</text>',
                f'<rect x="{margin_left}" y="{y}" width="{max(2, bar_width)}" height="{bar_height}" fill="#2563eb"/>',
                f'<text x="{margin_left + bar_width + 10}" y="{y + 19}" font-family="Arial, sans-serif" font-size="12" fill="#222">{latency:.3f}s  A:{answer} C:{citation} E:{evidence}</text>',
            ]
        )
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def _escape(value: str) -> str:
    return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


if __name__ == "__main__":
    raise SystemExit(main())
