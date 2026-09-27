#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-PolyForm-Noncommercial-1.0.0
"""Resolve every local Markdown and HTML image/link target and fragment."""

from __future__ import annotations

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parent.parent


def anchors(path: Path) -> set[str]:
    found: set[str] = set()
    seen: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        heading = re.match(r"^#{1,6}\s+(.+?)\s*#*\s*$", line)
        if heading is None:
            continue
        title = re.sub(r"<[^>]+>", "", heading.group(1))
        title = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", title)
        key = re.sub(r"\s+", "-", re.sub(r"[^\w\- ]", "", title.lower()).strip())
        count = seen.get(key, 0)
        found.add(f"{key}-{count}" if count else key)
        seen[key] = count + 1
    return found


def main() -> None:
    checked = 0
    for md in ROOT.rglob("*.md"):
        if ".git" in md.parts:
            continue
        content = md.read_text(encoding="utf-8")
        links = re.findall(r"!?(?:\[[^\]]*\])\(([^)]+)\)", content)
        links += re.findall(r"<(?:img|a)\s+[^>]*(?:src|href)=\"([^\"]+)\"", content)
        for link in links:
            destination = link.split('"')[0].strip()
            if destination.startswith(("https://", "http://", "mailto:")):
                continue
            local, _, fragment = destination.partition("#")
            target = (md.parent / local).resolve() if local else md
            assert target.is_relative_to(ROOT) and target.exists(), (md, destination)
            if fragment and target.suffix.lower() == ".md":
                assert fragment in anchors(target), (md, destination)
            checked += 1
    print(f"Local Markdown links: {checked} resolved")


if __name__ == "__main__":
    main()
