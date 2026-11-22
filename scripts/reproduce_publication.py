"""Validate or execute the frozen publication experiment suite.

The default mode is read-only verification.  Long-running training starts only
with ``--execute``; every training command in the suite uses checkpoint resume.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, TypedDict, cast

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "publication_suite.v1.json"
DEFAULT_EVIDENCE = ROOT / "configs" / "publication_evidence.v1.json"


class OutputSpec(TypedDict):
    path: str
    kind: str


class StageSpec(TypedDict, total=False):
    id: str
    scope: str
    kind: str
    commands: list[list[str]]
    expected_metadata: dict[str, Any]
    outputs: list[OutputSpec]
    note: str


class SuiteSpec(TypedDict):
    suite_version: str
    frozen_on: str
    python: str
    dataset_env: str
    experiment_contract: dict[str, int]
    runtime_packages: list[str]
    stages: list[StageSpec]


def _load_suite(path: Path) -> SuiteSpec:
    value: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("publication suite root must be an object")
    suite = cast(SuiteSpec, value)
    stage_ids = [stage["id"] for stage in suite["stages"]]
    if len(stage_ids) != len(set(stage_ids)):
        raise ValueError("publication suite stage ids must be unique")
    return suite


def _safe_path(relative: str) -> Path:
    path = (ROOT / relative).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError(f"path escapes repository root: {relative}")
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _non_finite_paths(value: object, prefix: str = "$") -> list[str]:
    if isinstance(value, Mapping):
        problems: list[str] = []
        for key, item in value.items():
            problems.extend(_non_finite_paths(item, f"{prefix}.{key}"))
        return problems
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        problems = []
        for index, item in enumerate(value):
            problems.extend(_non_finite_paths(item, f"{prefix}[{index}]"))
        return problems
    if isinstance(value, float) and not math.isfinite(value):
        return [prefix]
    return []


def _verify_metadata(
    data: Mapping[str, Any], expected: Mapping[str, Any], stage_id: str
) -> list[str]:
    issues = []
    for key, expected_value in expected.items():
        actual = data.get(key)
        if actual != expected_value:
            issues.append(
                f"{stage_id}: metadata {key}={actual!r}, expected {expected_value!r}"
            )
    return issues


def _verify_offloading_result(path: Path, stage: StageSpec) -> list[str]:
    value: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        return [f"{stage['id']}: result root is not an object"]
    data = cast(dict[str, Any], value)
    issues = _verify_metadata(data, stage.get("expected_metadata", {}), stage["id"])
    non_finite = _non_finite_paths(data)
    if non_finite:
        issues.append(
            f"{stage['id']}: {len(non_finite)} non-finite values; first={non_finite[0]}"
        )
    try:
        expected_decisions = int(data["test_episodes"]) * int(data["tasks_per_episode"])
        expected_runs = int(data["seeds"])
        ppo = cast(Mapping[str, Any], cast(Mapping[str, Any], data["results"])["ppo"])
        if not ppo:
            issues.append(f"{stage['id']}: empty PPO result family")
        for condition, raw_runs in ppo.items():
            if not isinstance(raw_runs, list):
                issues.append(f"{stage['id']}/{condition}: runs must be a list")
                continue
            runs = cast(list[Mapping[str, Any]], raw_runs)
            if len(runs) != expected_runs:
                issues.append(
                    f"{stage['id']}/{condition}: {len(runs)} runs, "
                    f"expected {expected_runs}"
                )
            for index, run in enumerate(runs):
                if float(run["decisions"]) != float(expected_decisions):
                    issues.append(
                        f"{stage['id']}/{condition}[{index}]: "
                        f"decisions={run['decisions']}, expected {expected_decisions}"
                    )
    except (KeyError, TypeError, ValueError) as exc:
        issues.append(f"{stage['id']}: invalid result schema: {exc}")
    return issues


def _select_stages(suite: SuiteSpec, requested: Sequence[str]) -> list[StageSpec]:
    by_id = {stage["id"]: stage for stage in suite["stages"]}
    if requested:
        missing = sorted(set(requested) - set(by_id))
        if missing:
            raise ValueError(f"unknown stage ids: {', '.join(missing)}")
        return [by_id[stage_id] for stage_id in requested]
    return [stage for stage in suite["stages"] if stage["scope"] == "current_pc"]


def _validate_commands(stages: Sequence[StageSpec]) -> list[str]:
    issues = []
    for stage in stages:
        if stage["kind"] == "manual_optional":
            continue
        if not stage.get("commands"):
            issues.append(f"{stage['id']}: no commands")
        for command in stage.get("commands", []):
            if not command or command[0] != "{python}":
                issues.append(f"{stage['id']}: command must start with {{python}}")
            for token in command[1:]:
                if token.startswith("/") or ":\\" in token:
                    issues.append(f"{stage['id']}: absolute command token: {token}")
        for output in stage.get("outputs", []):
            _safe_path(output["path"])
    return issues


def verify(stages: Sequence[StageSpec]) -> tuple[list[str], dict[str, str]]:
    issues = _validate_commands(stages)
    hashes: dict[str, str] = {}
    for stage in stages:
        for output in stage.get("outputs", []):
            path = _safe_path(output["path"])
            if not path.exists():
                issues.append(f"{stage['id']}: missing output {output['path']}")
                continue
            hashes[output["path"]] = _sha256(path)
            if output["kind"] == "offloading_result":
                issues.extend(_verify_offloading_result(path, stage))
    return issues, hashes


def _verify_evidence(path: Path) -> tuple[list[str], dict[str, str]]:
    if not path.exists():
        return [f"missing publication evidence manifest: {path}"], {}
    value: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("files"), list):
        return [f"invalid publication evidence manifest: {path}"], {}
    issues = []
    hashes: dict[str, str] = {}
    seen: set[str] = set()
    for raw_entry in cast(list[Any], value["files"]):
        if not isinstance(raw_entry, dict):
            issues.append("publication evidence file entry is not an object")
            continue
        entry = cast(dict[str, Any], raw_entry)
        relative = str(entry.get("path", ""))
        expected = str(entry.get("sha256", ""))
        if not relative or len(expected) != 64:
            issues.append(f"invalid publication evidence entry: {entry!r}")
            continue
        if relative in seen:
            issues.append(f"duplicate publication evidence path: {relative}")
            continue
        seen.add(relative)
        file_path = _safe_path(relative)
        if not file_path.exists():
            issues.append(f"publication evidence missing: {relative}")
            continue
        actual = _sha256(file_path)
        hashes[relative] = actual
        if actual != expected:
            issues.append(
                f"publication evidence drift: {relative}; "
                f"actual={actual}, expected={expected}"
            )
    return issues, hashes


def _resolved_command(command: Sequence[str]) -> list[str]:
    return [sys.executable if token == "{python}" else token for token in command]


def execute(stages: Sequence[StageSpec], *, allow_frozen_overwrite: bool) -> None:
    for stage in stages:
        if stage["kind"] == "manual_optional":
            print(f"skip manual optional stage: {stage['id']}")
            continue
        frozen_outputs = [
            output["path"]
            for output in stage.get("outputs", [])
            if output["kind"] == "offloading_result"
            and _safe_path(output["path"]).exists()
        ]
        if frozen_outputs and not allow_frozen_overwrite:
            joined = ", ".join(frozen_outputs)
            raise RuntimeError(
                f"stage {stage['id']} would overwrite frozen results: {joined}; "
                "use --allow-frozen-overwrite only for an intentional new evidence run"
            )
        print(f"== {stage['id']} ==", flush=True)
        for command in stage.get("commands", []):
            resolved = _resolved_command(command)
            print(" ".join(resolved), flush=True)
            completed = subprocess.run(resolved, cwd=ROOT, check=False)
            if completed.returncode:
                raise RuntimeError(
                    f"stage {stage['id']} failed with exit code {completed.returncode}"
                )


def _runtime_versions(packages: Sequence[str]) -> dict[str, str]:
    versions = {"python": sys.version.split()[0]}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "missing"
    return versions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--stage", action="append", default=[])
    parser.add_argument("--allow-frozen-overwrite", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--list", action="store_true")
    mode.add_argument("--validate-config", action="store_true")
    args = parser.parse_args()

    suite = _load_suite(args.config)
    stages = _select_stages(suite, cast(list[str], args.stage))
    if args.list:
        for stage in suite["stages"]:
            print(f"{stage['id']}\t{stage['scope']}\t{stage['kind']}")
        return

    command_issues = _validate_commands(stages)
    if args.validate_config:
        if command_issues:
            raise RuntimeError("\n".join(command_issues))
        print(f"valid suite config: {suite['suite_version']} ({len(stages)} stages)")
        return

    if args.execute:
        execute(stages, allow_frozen_overwrite=args.allow_frozen_overwrite)

    issues, hashes = verify(stages)
    if args.execute:
        # Host wall-clock decision-overhead fields may change on a fresh run.
        # Structural/statistical contracts still fail closed; new hashes are
        # printed so an intentional evidence-version bump can freeze them.
        evidence_hashes: dict[str, str] = {}
    else:
        evidence_issues, evidence_hashes = _verify_evidence(args.evidence)
        issues.extend(evidence_issues)
    summary = {
        "suite_version": suite["suite_version"],
        "selected_stages": [stage["id"] for stage in stages],
        "runtime_versions": _runtime_versions(suite["runtime_packages"]),
        "output_sha256": dict(sorted(hashes.items())),
        "evidence_sha256": dict(sorted(evidence_hashes.items())),
        "issues": issues,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if issues:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
