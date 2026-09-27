# SPDX-License-Identifier: LicenseRef-PolyForm-Noncommercial-1.0.0
"""Render source-backed measurement tables as transparent GitHub SVGs."""
from pathlib import Path
import csv

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs" / "results"
OUT = ROOT / "docs" / "assets"
ACCENT, INK, MUTED, GRID = "#087f8c", "#182b38", "#435b68", "#82939b"
FONT = 'Inter, Manrope, "Space Grotesk", system-ui, Arial, sans-serif'


def rows(name):
    with (DATA / name).open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def start(title, subtitle):
    return [
        '<!-- SPDX-License-Identifier: CC-BY-NC-4.0 -->',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="900" height="470" viewBox="0 0 900 470" role="img" aria-label="{title}">',
        f'<style>text{{font-family:{FONT}}}.accent{{fill:{ACCENT}}}.ink{{fill:{INK}}}.muted{{fill:{MUTED}}}@media(prefers-color-scheme:dark){{.accent{{fill:#68d9e0}}.ink{{fill:#e8f0f4}}.muted{{fill:#a9bbc5}}}}</style>',
        f'<text x="60" y="49" class="readable ink" font-size="27" font-weight="750">{title}</text>',
        f'<text x="60" y="78" class="readable muted" font-size="15">{subtitle}</text>',
    ]


def finish(parts, filename):
    parts.append('</svg>')
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / filename).write_text('\n'.join(parts) + '\n', encoding='utf-8')


def throughput_chart():
    samples = rows('throughput.csv')
    assert [int(row['streams']) for row in samples] == [1, 2, 3, 4, 5]
    values = [float(row['peak_decode_tok_s']) for row in samples]
    parts = start('Peak decode throughput', 'Measured v16b decode · one GB10 · tok/s')
    left, right, bottom, top = 105, 827, 366, 127
    for tick in (0, 50, 100, 150):
        y = bottom - (tick / 160) * (bottom - top)
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" stroke="{GRID}" opacity=".36"/>')
        parts.append(f'<text x="{left-18}" y="{y+5:.1f}" text-anchor="end" class="readable muted" font-size="14">{tick}</text>')
    points = [(left + i * (right - left) / 4, bottom - value / 160 * (bottom - top)) for i, value in enumerate(values)]
    path = ' '.join(f'{x:.1f},{y:.1f}' for x, y in points)
    parts.append(f'<polyline points="{path}" fill="none" stroke="{ACCENT}" stroke-width="5" stroke-linecap="round" stroke-linejoin="round"/>')
    for count, ((x, y), value) in enumerate(zip(points, values), 1):
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="8" fill="{ACCENT}" stroke="white" stroke-width="2"/>')
        parts.append(f'<text x="{x:.1f}" y="{y-19:.1f}" text-anchor="middle" class="readable accent" font-size="19" font-weight="750">{value:.1f}</text>')
        parts.append(f'<text x="{x:.1f}" y="{bottom+30}" text-anchor="middle" class="readable ink" font-size="16" font-weight="650">{count}</text>')
    parts.append(f'<text x="{(left+right)/2:.1f}" y="443" text-anchor="middle" class="readable muted" font-size="15">Concurrent streams</text>')
    parts.append('<text x="23" y="256" transform="rotate(-90 23 256)" text-anchor="middle" class="readable muted" font-size="15">Peak tok/s</text>')
    finish(parts, 'throughput.svg')


def speed_ladder_chart():
    samples = rows('speed-ladder.csv')
    parts = start('Decode step time', 'Measured coding windows · one GB10 · lower is faster')
    left, right = 180, 775
    for tick in (0, 20, 40, 60):
        x = left + tick / 75 * (right - left)
        parts.append(f'<line x1="{x:.1f}" y1="116" x2="{x:.1f}" y2="380" stroke="{GRID}" opacity=".36"/>')
        parts.append(f'<text x="{x:.1f}" y="407" text-anchor="middle" class="readable muted" font-size="14">{tick}</text>')
    for index, row in enumerate(samples):
        y = 138 + index * 62
        value = float(row['step_ms'])
        width = value / 75 * (right - left)
        label = row['recipe'].replace('original', 'Original')
        parts.append(f'<text x="{left-22}" y="{y+21}" text-anchor="end" class="readable ink" font-size="17" font-weight="650">{label}</text>')
        parts.append(f'<rect x="{left}" y="{y}" width="{width:.1f}" height="28" rx="5" fill="{ACCENT}"/>')
        parts.append(f'<text x="{left+width+14:.1f}" y="{y+21}" class="readable accent" font-size="17" font-weight="750">{value:g}</text>')
    parts.append('<text x="478" y="443" text-anchor="middle" class="readable muted" font-size="15">Milliseconds per decode step</text>')
    finish(parts, 'speed-ladder.svg')


if __name__ == '__main__':
    throughput_chart()
    speed_ladder_chart()
