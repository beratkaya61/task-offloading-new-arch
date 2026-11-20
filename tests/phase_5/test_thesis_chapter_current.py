"""The generated thesis method chapter must match the repository sources."""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "build_thesis_method_chapter.py"


def test_thesis_method_chapter_is_current() -> None:
    spec = importlib.util.spec_from_file_location("build_chapter", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    expected = module.render()
    actual = module.OUTPUT.read_text(encoding="utf-8")
    assert actual == expected, (
        "docs/thesis/03_VERI_VE_YONTEM.md is stale; run "
        "scripts/build_thesis_method_chapter.py"
    )
