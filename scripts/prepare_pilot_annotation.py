"""Create one offline annotation form after the pilot corpus is human-approved."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from task_offloading.semantic import (
    audit_annotation_release,
    audit_corpus,
    load_jsonl,
    packet_issues,
    render_annotation_form,
)

ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / "data" / "semantic_benchmark" / "pilot"


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value: Any = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "İnsan onaylı pilot corpus'tan çevrimdışı kör etiketleme formu "  # noqa: RUF001
            "üret."
        )
    )
    parser.add_argument(
        "--annotator",
        required=True,
        choices=("annotator_a", "annotator_b", "annotator_c"),
        help="Formun ait olduğu gerçek insan takma kimliği.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "artifacts" / "semantic_annotation",
        help="Git dışında tutulacak form dizini.",  # noqa: RUF001
    )
    parser.add_argument(
        "--prefill-json",
        type=Path,
        help="Aynı annotator'ın daha önce verdiği kısmi cevap taslağı.",  # noqa: RUF001
    )
    parser.add_argument(
        "--round-id",
        choices=("round_1", "round_2"),
        default="round_1",
        help="Kör paket ve protokol turu.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    corpus = load_jsonl(PILOT / "tasks.v1.jsonl")
    corpus_schema = _load_json(ROOT / "schemas" / "semantic_task.schema.json")
    corpus_audit = audit_corpus(corpus, corpus_schema, expected_count=20)
    if not corpus_audit.is_valid:
        raise ValueError(
            "pilot corpus audit failed:\n- " + "\n- ".join(corpus_audit.issues)
        )

    release = audit_annotation_release(corpus)
    if not release.is_ready:
        raise ValueError(
            "annotation release blocked:\n- "
            + "\n- ".join(release.issues)
            + "\nA real human must review and approve every task text first."
        )

    if args.round_id == "round_2":
        if args.annotator == "annotator_c":
            raise ValueError("annotator_c is not part of pilot Round 2")
        if args.prefill_json is not None:
            raise ValueError("Round 2 must start fresh without prefilled answers")
        packet_path = (
            PILOT / "round_2" / "packets" / f"{args.annotator}.v1.1.jsonl"
        )
        expected_packet_version = "1.1.0"
        expected_protocol_version = "1.1.0"
    else:
        packet_path = PILOT / "packets" / f"{args.annotator}.v1.jsonl"
        expected_packet_version = "1.0.0"
        expected_protocol_version = "1.0.0"
    packet = load_jsonl(packet_path)
    issues = packet_issues(
        packet,
        corpus,
        annotator_id=args.annotator,
        packet_version=expected_packet_version,
        protocol_version=expected_protocol_version,
    )
    if issues:
        raise ValueError("blind packet audit failed:\n- " + "\n- ".join(issues))

    packet_sha256 = _sha256(packet_path)
    initial_responses: dict[str, Any] | None = None
    if args.prefill_json is not None:
        initial_responses = _load_json(args.prefill_json.resolve())

    html = render_annotation_form(
        packet,
        annotator_id=args.annotator,
        packet_sha256=packet_sha256,
        initial_responses=initial_responses,
    )
    output_dir: Path = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.round_id == "round_2":
        suffix = ".round_2.pilot.v1.1.html"
    else:
        suffix = ".prefilled.pilot.v1.html" if initial_responses else ".pilot.v1.html"
    output_path = output_dir / f"{args.annotator}{suffix}"
    output_path.write_text(html, encoding="utf-8", newline="\n")
    print(f"form={output_path}")
    print(f"packet_sha256={packet_sha256}")
    print(f"form_sha256={_sha256(output_path)}")
    print("network_access=disabled")


if __name__ == "__main__":
    main()
