#!/usr/bin/env python3
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Enforce the single TE<->PyTorch binary boundary.

Only ``transformer_engine/pytorch/csrc/torch_backend.{h,cpp}`` may include
libtorch/ATen/c10 headers or name ``at::`` / ``c10::`` / ``c10d::`` /
``torch::`` symbols. Every other translation unit must talk to PyTorch through
the aliases and free functions that ``torch_backend.h`` exposes.

Files that have not been migrated yet are grandfathered in via
``torch_boundary_allowlist.txt`` (paths relative to the csrc dir). The guard
fails if:
  * a non-boundary, non-allowlisted file touches the torch ABI, or
  * an allowlisted file no longer touches it (stale entry -> remove it), or
  * an allowlisted path does not exist.

Comments and string literals are stripped before scanning, so mentioning the
tokens in a comment is fine.

Usage: check_torch_boundary.py [TE_ROOT]   (default: $TE_PATH or cwd)
"""

import os
import re
import sys
from pathlib import Path

CSRC_REL = "transformer_engine/pytorch/csrc"
BOUNDARY = {"torch_backend.h", "torch_backend.cpp"}
SOURCE_SUFFIXES = {".cpp", ".h", ".hpp", ".cuh", ".cu"}

# Include of a libtorch/ATen/c10 header, or a qualified at::/c10::/c10d::/torch:: name.
INCLUDE_RE = re.compile(r'#\s*include\s*[<"](?:torch|ATen|c10)/')
SYMBOL_RE = re.compile(r'\b(?:at|c10|c10d|torch)::')


def strip_comments_and_strings(text: str) -> str:
    """Blank out // and /* */ comments and "..."/'...' literals (newlines kept)."""
    out = []
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        two = text[i:i + 2]
        if two == "//":
            i += 2
            while i < n and text[i] != "\n":
                i += 1
        elif two == "/*":
            i += 2
            while i < n and text[i:i + 2] != "*/":
                out.append("\n" if text[i] == "\n" else " ")
                i += 1
            i += 2
        elif c in "\"'":
            quote = c
            out.append(" ")
            i += 1
            while i < n and text[i] != quote:
                if text[i] == "\\" and i + 1 < n:
                    i += 1
                out.append("\n" if text[i] == "\n" else " ")
                i += 1
            i += 1
            out.append(" ")
        else:
            out.append(c)
            i += 1
    return "".join(out)


def scan(path: Path):
    """Return list of (lineno, text) lines that touch the torch ABI."""
    raw = path.read_text(encoding="utf-8", errors="replace")
    code = strip_comments_and_strings(raw)
    hits = []
    for lineno, line in enumerate(code.splitlines(), start=1):
        if INCLUDE_RE.search(line) or SYMBOL_RE.search(line):
            hits.append(lineno)
    return hits


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else os.environ.get("TE_PATH", ".")).resolve()
    csrc = root / CSRC_REL
    if not csrc.is_dir():
        print(f"error: csrc dir not found: {csrc}", file=sys.stderr)
        return 2

    allowlist_file = root / "qa/L0_pytorch_lint/torch_boundary_allowlist.txt"
    allowlist = set()
    if allowlist_file.is_file():
        for line in allowlist_file.read_text().splitlines():
            line = line.split("#", 1)[0].strip()
            if line:
                allowlist.add(line)

    violations = []          # (rel, lineno) touching torch outside the boundary
    still_allowlisted = set()  # allowlisted files that still touch torch (expected)

    for path in sorted(csrc.rglob("*")):
        if path.suffix not in SOURCE_SUFFIXES or not path.is_file():
            continue
        rel = path.relative_to(csrc).as_posix()
        if rel in BOUNDARY:
            continue
        hits = scan(path)
        if not hits:
            continue
        if rel in allowlist:
            still_allowlisted.add(rel)
        else:
            violations.extend((rel, ln) for ln in hits)

    stale = sorted(allowlist - still_allowlisted)
    missing = sorted(p for p in allowlist if not (csrc / p).is_file())

    ok = True
    if violations:
        ok = False
        print("TE<->torch boundary violations (route these through torch_backend.h):")
        for rel, ln in violations:
            print(f"  {CSRC_REL}/{rel}:{ln}")
    if stale:
        ok = False
        print("\nStale allowlist entries (migrated -- remove from "
              "torch_boundary_allowlist.txt):")
        for rel in stale:
            print(f"  {rel}")
    if missing:
        ok = False
        print("\nAllowlist entries pointing at non-existent files (remove them):")
        for rel in missing:
            print(f"  {rel}")

    if ok:
        print(f"torch boundary OK: only torch_backend.* touches the torch ABI "
              f"({len(still_allowlisted)} file(s) still pending migration).")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
