"""Build the advisor DOCX from the canonical Markdown and verify text parity.

The Markdown file is the single source of truth.  Pandoc embeds the figures in
the Word file, so the two files can look different while their textual content
must remain identical.  This script refuses to replace the canonical DOCX when
the normalized Markdown and DOCX text differ.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import re
import subprocess
import tempfile
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "TEZ_DANISMANI_MEVCUT_DURUM_VE_LITERATUR_OZETI.md"
OUTPUT = ROOT / "TEZ_DANISMANI_MEVCUT_DURUM_VE_LITERATUR_OZETI.docx"
IMAGE_LINE = re.compile(r"(?m)^!\[[^\]]*\]\([^\)]*\)\s*$")


def _run_pandoc(args: list[str], *, stdin: bytes | None = None) -> bytes:
    completed = subprocess.run(
        ["pandoc", *args],
        cwd=ROOT,
        input=stdin,
        capture_output=True,
        check=False,
    )
    if completed.returncode:
        message = completed.stderr.decode("utf-8", errors="replace")
        raise RuntimeError(f"pandoc failed ({completed.returncode}): {message}")
    return completed.stdout


def _normalize(value: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", value)).strip()


def _pandoc_ast(path: Path, *, source_format: str | None = None) -> object:
    args = [str(path), "--to=json"]
    if source_format:
        args.append(f"--from={source_format}")
    return json.loads(_run_pandoc(args).decode("utf-8"))


def _ast_text(ast: object) -> str:
    """Extract visible text in structural order while ignoring style metadata."""

    tokens: list[str] = []

    def walk(node: object) -> None:
        if isinstance(node, list):
            for item in node:
                walk(item)
            return
        if not isinstance(node, dict):
            return

        node_type = node.get("t")
        content = node.get("c")
        if node_type == "Image":
            # Embedded media count and presence are verified separately.  DOCX
            # readers do not round-trip Markdown alt text consistently.
            return
        if node_type == "Str" and isinstance(content, str):
            tokens.append(content)
            return
        if node_type in {"Space", "SoftBreak", "LineBreak"}:
            tokens.append(" ")
            return
        if node_type in {"Code", "CodeBlock"} and isinstance(content, list):
            tokens.append(str(content[-1]))
            return
        if node_type == "Math" and isinstance(content, list):
            tokens.append(str(content[-1]))
            return
        if node_type == "Link" and isinstance(content, list):
            # Traverse only the human-visible label, not URL/title metadata.
            walk(content[1])
            return
        walk(content)

    walk(ast)
    return _normalize(" ".join(tokens))


def _figure_count() -> int:
    return len(IMAGE_LINE.findall(SOURCE.read_text(encoding="utf-8")))


def _docx_figure_count(path: Path) -> int:
    listing = subprocess.run(
        ["tar", "-tf", str(path)],
        cwd=ROOT,
        capture_output=True,
        check=False,
    )
    if listing.returncode:
        message = listing.stderr.decode("utf-8", errors="replace")
        raise RuntimeError(f"could not inspect DOCX media: {message}")
    members = listing.stdout.decode("utf-8").splitlines()
    return sum(name.startswith("word/media/") for name in members)


def _assert_text_parity(docx: Path) -> None:
    expected = _ast_text(_pandoc_ast(SOURCE, source_format="gfm"))
    actual = _ast_text(_pandoc_ast(docx))
    if expected == actual:
        return
    diff = "\n".join(
        difflib.unified_diff(
            expected.split(),
            actual.split(),
            fromfile=SOURCE.name,
            tofile=docx.name,
            n=5,
        )
    )
    raise RuntimeError(f"Markdown/DOCX text mismatch:\n{diff[:4000]}")


def build(*, verify_only: bool = False) -> None:
    if verify_only:
        _assert_text_parity(OUTPUT)
        print(f"verified text parity: {SOURCE.name} == {OUTPUT.name}")
        return

    lock_file = OUTPUT.with_name(f"~${OUTPUT.name[2:]}")
    if lock_file.exists():
        raise RuntimeError(f"close Word before rebuilding: {lock_file.name}")

    with tempfile.NamedTemporaryFile(
        prefix="advisor_report_", suffix=".docx", dir=ROOT, delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        args = [
            str(SOURCE),
            "--from=gfm",
            "--to=docx",
            f"--resource-path={ROOT}",
            "--output",
            str(temporary),
        ]
        _run_pandoc(args)
        _assert_text_parity(temporary)
        expected_figures = _figure_count()
        actual_figures = _docx_figure_count(temporary)
        if actual_figures != expected_figures:
            raise RuntimeError(
                f"figure mismatch: Markdown={expected_figures}, DOCX={actual_figures}"
            )
        temporary.replace(OUTPUT)
    finally:
        temporary.unlink(missing_ok=True)

    digest = hashlib.sha256(OUTPUT.read_bytes()).hexdigest()
    print(f"built {OUTPUT.name}")
    print(f"verified text parity and {expected_figures} embedded figures")
    print(f"sha256={digest}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    build(verify_only=args.verify_only)


if __name__ == "__main__":
    main()
