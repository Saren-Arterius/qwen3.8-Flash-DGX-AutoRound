# SPDX-License-Identifier: LicenseRef-PolyForm-Noncommercial-1.0.0
"""Render source-backed CSV measurements as explicit light and dark SVGs."""
from pathlib import Path
import csv
from decimal import Decimal, ROUND_HALF_UP

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs" / "results"
OUT = ROOT / "docs" / "assets"
FONT = 'Inter, Manrope, "Space Grotesk", system-ui, Arial, sans-serif'
THEMES = {
    "light": {"ink": "#182b38", "muted": "#435b68", "accent": "#087f8c",
              "grid": "#82939b", "point": "#ffffff"},
    "dark": {"ink": "#e8f0f4", "muted": "#a9bbc5", "accent": "#68d9e0",
             "grid": "#607583", "point": "#0d1117"},
}


def rows(name):
    with (DATA / name).open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def start(title, subtitle, colors):
    return [
        '<!-- SPDX-License-Identifier: CC-BY-NC-4.0 -->',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="900" height="470" viewBox="0 0 900 470" role="img" aria-label="{title}">',
        f'<style>text{{font-family:{FONT}}}</style>',
        f'<text x="60" y="49" fill="{colors["ink"]}" font-size="27" font-weight="750">{title}</text>',
        f'<text x="60" y="78" fill="{colors["muted"]}" font-size="15">{subtitle}</text>',
    ]


def finish(parts, filename):
    parts.append('</svg>')
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / filename).write_bytes(('\n'.join(parts) + '\n').encode('utf-8'))


def throughput_chart(theme, colors):
    samples = rows('throughput.csv')
    assert [int(row['streams']) for row in samples] == list(range(1, 9))
    values = [float(row['peak_decode_tok_s']) for row in samples]
    parts = start('Peak decode throughput',
                  'Copy-heavy decode · three rounds per stream count · one GB10', colors)
    left, right, bottom, top = 105, 827, 366, 127
    scale = 230
    for tick in (0, 50, 100, 150, 200):
        y = bottom - (tick / scale) * (bottom - top)
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" stroke="{colors["grid"]}" opacity=".40"/>')
        parts.append(f'<text x="{left-18}" y="{y+5:.1f}" text-anchor="end" fill="{colors["muted"]}" font-size="14">{tick}</text>')
    points = [(left + i * (right - left) / 7, bottom - value / scale * (bottom - top))
              for i, value in enumerate(values)]
    path = ' '.join(f'{x:.1f},{y:.1f}' for x, y in points)
    parts.append(f'<polyline points="{path}" fill="none" stroke="{colors["accent"]}" stroke-width="5" stroke-linecap="round" stroke-linejoin="round"/>')
    for count, ((x, y), value) in enumerate(zip(points, values), 1):
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="8" fill="{colors["accent"]}" stroke="{colors["point"]}" stroke-width="2"/>')
        label_x, anchor = (x + 17, 'start') if count == 1 else (x, 'middle')
        parts.append(f'<text x="{label_x:.1f}" y="{y-19:.1f}" text-anchor="{anchor}" fill="{colors["accent"]}" font-size="19" font-weight="750">{value:.1f}</text>')
        parts.append(f'<text x="{x:.1f}" y="{bottom+30}" text-anchor="middle" fill="{colors["ink"]}" font-size="16" font-weight="650">{count}</text>')
    parts.append(f'<text x="{(left+right)/2:.1f}" y="443" text-anchor="middle" fill="{colors["muted"]}" font-size="15">Concurrent streams</text>')
    parts.append(f'<text x="23" y="256" transform="rotate(-90 23 256)" text-anchor="middle" fill="{colors["muted"]}" font-size="15">Peak tok/s</text>')
    finish(parts, f'throughput-{theme}.svg')


def speed_ladder_chart(theme, colors):
    samples = rows('speed-ladder.csv')
    parts = start('Decode step time',
                  'Measured coding windows · one GB10 · lower is faster', colors)
    left, right = 180, 775
    for tick in (0, 20, 40, 60):
        x = left + tick / 75 * (right - left)
        parts.append(f'<line x1="{x:.1f}" y1="116" x2="{x:.1f}" y2="380" stroke="{colors["grid"]}" opacity=".40"/>')
        parts.append(f'<text x="{x:.1f}" y="407" text-anchor="middle" fill="{colors["muted"]}" font-size="14">{tick}</text>')
    for index, row in enumerate(samples):
        y = 138 + index * 62
        value = float(row['step_ms'])
        width = value / 75 * (right - left)
        label = row['recipe'].replace('original', 'Original')
        parts.append(f'<text x="{left-22}" y="{y+21}" text-anchor="end" fill="{colors["ink"]}" font-size="17" font-weight="650">{label}</text>')
        parts.append(f'<rect x="{left}" y="{y}" width="{width:.1f}" height="28" rx="5" fill="{colors["accent"]}"/>')
        label_value = Decimal(row['step_ms']).quantize(Decimal('0.1'), rounding=ROUND_HALF_UP)
        parts.append(f'<text x="{left+width+14:.1f}" y="{y+21}" fill="{colors["accent"]}" font-size="17" font-weight="750">{label_value}</text>')
    parts.append(f'<text x="478" y="443" text-anchor="middle" fill="{colors["muted"]}" font-size="15">Milliseconds per decode step</text>')
    finish(parts, f'speed-ladder-{theme}.svg')


if __name__ == '__main__':
    for theme, colors in THEMES.items():
        throughput_chart(theme, colors)
        speed_ladder_chart(theme, colors)
