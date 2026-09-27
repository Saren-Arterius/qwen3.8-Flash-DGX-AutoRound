#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-PolyForm-Noncommercial-1.0.0
"""Portable public-tree and Git-blob privacy gate, including binary containers."""

from __future__ import annotations

import gzip
from pathlib import Path
import re
import struct
import subprocess
import zlib


ROOT = Path(__file__).resolve().parent.parent
PATTERNS = [
    ("POSIX home", r"/h[o]me/[A-Za-z0-9._-]+"),
    ("Windows home", r"(?:[A-Za-z]:\\|/)Users[/\\][^/\\\s]+"),
    ("private IP", r"(?:192\.168|10\.0)\.\d+\.\d+"),
    ("local hostname", r"\.lo" r"cal\b(?!\()"),
    ("MAC address", r"\b(?:[0-9a-f]{2}:){5}[0-9a-f]{2}\b"),
    ("credential", r"\b(?:hf_[A-Za-z0-9]{20,}|sk[-][A-Za-z0-9]+|gh[p]_[A-Za-z0-9]+|AKIA[0-9A-Z]{16})\b"),
    ("credential label", r"api[_]key|token[=]|Bearer\s+[A-Za-z0-9._-]+"),
    ("private key", r"BEGIN\s+(?:RSA\s+|OPENSSH\s+|EC\s+)?PRIVATE\s+KEY|ssh[-](?:rsa|ed25519)"),
    ("non-public authorship", r"\b(?:clau[d]e|anthrop[i]c|chatg[p]t|openai\s+media\s+service|gpt[-]image|c2p[a]|trainedalgorithmicmedi[a]|ai\s+assistants|co[-]authored[-]by|generated\s+with)\b"),
]
RX = [(label, re.compile(pattern, re.I)) for label, pattern in PATTERNS]
IPV4 = re.compile(r"(?<!\d)(?:\d{1,3}\.){3}\d{1,3}(?!\d)")
ALLOWED_BIND = {"scripts/serve-intel-ar.sh", "config/v16b/serve.sh"}
WILDCARD_BIND = ".".join(("0", "0", "0", "0"))
SHELLCHECK_VERSION = ".".join(("0", "11", "0", "1"))
PNG_CRITICAL = {b"IHDR", b"IDAT", b"IEND"}


def check_png(data: bytes, name: str) -> None:
    assert data.startswith(b"\x89PNG\r\n\x1a\n"), name
    pos = 8
    chunks: list[bytes] = []
    while pos < len(data):
        length = struct.unpack(">I", data[pos : pos + 4])[0]
        end = pos + length + 12
        assert end <= len(data), name
        kind = data[pos + 4 : pos + 8]
        assert zlib.crc32(data[pos + 4 : end - 4]) == struct.unpack(">I", data[end - 4 : end])[0], name
        chunks.append(kind)
        pos = end
        if kind == b"IEND":
            break
    assert pos == len(data) and chunks[0] == b"IHDR" and chunks[-1] == b"IEND", name
    assert set(chunks) <= PNG_CRITICAL, (name, chunks)


def scan(data: bytes, name: str) -> list[str]:
    if name.endswith(".png"):
        check_png(data, name)
        return []
    if name.endswith(".gz"):
        data = gzip.decompress(data)
    findings: list[str] = []
    for line_no, line in enumerate(data.decode("utf-8").splitlines(), 1):
        for label, rx in RX:
            if rx.search(line):
                findings.append(f"{name}:{line_no}: {label}")
        for ip in IPV4.findall(line):
            if ip == SHELLCHECK_VERSION and "shellcheck-py==" in line:
                continue
            if ip != WILDCARD_BIND or name not in ALLOWED_BIND:
                findings.append(f"{name}:{line_no}: IP address")
    return findings


def main() -> None:
    files = [
        p for p in ROOT.rglob("*")
        if p.is_file()
        and not {".git", ".venv", "__pycache__"} & set(p.relative_to(ROOT).parts)
    ]
    findings: list[str] = []
    for path in files:
        findings.extend(scan(path.read_bytes(), path.relative_to(ROOT).as_posix()))
    commits = subprocess.check_output(["git", "-C", str(ROOT), "rev-list", "--all"], text=True).splitlines()
    assert commits, "no reachable Git history"
    blobs = 0
    for commit in commits:
        names = subprocess.check_output(["git", "-C", str(ROOT), "ls-tree", "-r", "-z", "--name-only", commit]).decode().rstrip("\0").split("\0")
        for name in names:
            blob = subprocess.check_output(["git", "-C", str(ROOT), "cat-file", "blob", f"{commit}:{name}"])
            findings.extend(scan(blob, name))
            blobs += 1
        metadata = subprocess.check_output(["git", "-C", str(ROOT), "show", "-s", "--format=%an%n%ae%n%cn%n%ce%n%B", commit], text=True)
        assert not re.search(r"(?im)^(?:Co[-]Authored[-]By|Signed[-]off[-]by|Reviewed[-]by):", metadata)
        findings.extend(scan(metadata.encode(), "commit metadata"))
    for finding in findings:
        print(finding)
    assert not findings, f"privacy gate found {len(findings)} findings"
    print(f"Privacy: {len(files)} working files, {blobs} Git blobs in {len(commits)} reachable commits, PNG chunks and gzip checked; 0 findings")


if __name__ == "__main__":
    main()
