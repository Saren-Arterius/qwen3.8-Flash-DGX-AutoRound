# SPDX-License-Identifier: LicenseRef-PolyForm-Noncommercial-1.0.0
"""Render fixed, source-backed CSV summaries as self-contained SVG charts."""
from pathlib import Path
import csv
import html

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "assets"


def rows(name):
    with (ROOT / "results" / name).open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def svg(title, subtitle, labels, values, unit, maximum):
    width, height = 780, 120 + len(labels) * 65
    parts = ['<!-- SPDX-License-Identifier: CC-BY-NC-4.0 -->',
             f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">',
             '<rect width="100%" height="100%" fill="#0d1722"/>',
             f'<text x="30" y="40" fill="#f4f7fb" font-size="24" font-family="system-ui,sans-serif" font-weight="700">{html.escape(title)}</text>',
             f'<text x="30" y="64" fill="#aabac8" font-size="13" font-family="system-ui,sans-serif">{html.escape(subtitle)}</text>']
    for i, (label, value) in enumerate(zip(labels, values)):
        y = 94 + i * 65
        bar = round(560 * value / maximum, 1)
        parts += [f'<text x="30" y="{y+17}" fill="#d6e0ea" font-size="16" font-family="system-ui,sans-serif">{html.escape(label)}</text>',
                  f'<rect x="150" y="{y}" width="560" height="25" rx="5" fill="#1d3342"/>',
                  f'<rect x="150" y="{y}" width="{bar}" height="25" rx="5" fill="#2dc4c5"/>',
                  f'<text x="{160+bar}" y="{y+18}" fill="#ffffff" font-size="14" font-family="system-ui,sans-serif" font-weight="700">{value:g} {unit}</text>']
    parts.append('</svg>')
    return '\n'.join(parts) + '\n'


throughput = rows("throughput.csv")
(OUT / "throughput.svg").write_text(svg(
    "Aggregate throughput by stream count",
    "Measured v16b equal-length decode windows; 600 tokens per stream",
    [r["streams"] + " streams" for r in throughput],
    [float(r["aggregate_tok_s"]) for r in throughput], "tok/s", 170), encoding="utf-8")

ladder = rows("speed-ladder.csv")
(OUT / "speed-ladder.svg").write_text(svg(
    "Decode step time by recipe",
    "Measured candidate and interleaved coding windows",
    [r["recipe"] for r in ladder], [float(r["step_ms"]) for r in ladder],
    "ms", 75), encoding="utf-8")
