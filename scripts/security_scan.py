#!/usr/bin/env python3
"""Fail when tracked content or Git metadata contains common secrets or PII."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SELF = Path(__file__).resolve()
FORBIDDEN_PATHS = {
    "INTERVIEW.md",
    "PLAN.md",
}
FORBIDDEN_PREFIXES = (
    "notes/private/",
    "results/runs/",
)
PATTERNS = {
    "private key": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "AWS access key": re.compile(r"(?:AKIA|ASIA)[0-9A-Z]{16}"),
    "GitHub token": re.compile(
        r"(?:gh[pousr]_[A-Za-z0-9_]{20,}|github_pat_[A-Za-z0-9_]{20,})"
    ),
    "OpenAI-style secret": re.compile(r"sk-[A-Za-z0-9_-]{20,}"),
    "Windows user path": re.compile(r"[A-Za-z]:\\Users\\[^\\\s]+", re.I),
    "Unix home path": re.compile(r"/home/[^/\s]+"),
    "GPU UUID": re.compile(r"GPU-[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", re.I),
    "non-private email": re.compile(
        r"\b[A-Z0-9._%+-]+@(?!users\.noreply\.github\.com\b)[A-Z0-9.-]+\.[A-Z]{2,}\b",
        re.I,
    ),
}


def git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
    )
    return result.stdout


def tracked_files() -> list[Path]:
    return [ROOT / name for name in git("ls-files").splitlines() if name]


def main() -> int:
    findings: list[str] = []
    for path in tracked_files():
        relative = path.relative_to(ROOT).as_posix()
        if relative in FORBIDDEN_PATHS or relative.startswith(FORBIDDEN_PREFIXES):
            findings.append(f"forbidden tracked path: {relative}")
        if path == SELF:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for label, pattern in PATTERNS.items():
            for match in pattern.finditer(text):
                line = text.count("\n", 0, match.start()) + 1
                findings.append(f"{relative}:{line}: {label}")

    for email in {line.strip() for line in git("log", "--all", "--format=%ae").splitlines()}:
        if email and not email.lower().endswith("@users.noreply.github.com"):
            findings.append("Git history contains a non-private author email")

    if findings:
        print("security scan failed:", file=sys.stderr)
        for finding in findings:
            print(f"- {finding}", file=sys.stderr)
        return 1
    print("security scan passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
