"""Aggregate NEP-large distributions used by the hybrid simulator (no raw rows).

* Edge background load: per (site, hour) mean of ``VM_CPU.cpu_rate`` over rows
  with a finite rate in [0, 1]; the distribution of these site-hour means is
  summarised by its 1st..99th percentiles.
* Cloud one-way propagation: ``SITE_RTT.rtt`` in milliseconds over rows with
  0 < rtt <= 1000 and from_site_id != to_site_id; percentiles 1..99 via
  DuckDB ``approx_quantile`` (t-digest) because the table has ~5e8 rows.

Only percentile tables and row counts are written; raw NEP rows never leave
the local extraction directory (research-only licence).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import time
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "configs" / "data_profiles" / "nep_quantiles_v1.json"
PERCENTILES = [p / 100 for p in range(1, 100)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--extracted-root",
        type=Path,
        default=Path("D:/task_offloading_datasets/nep/extracted/Full_trace"),
    )
    args = parser.parse_args()
    started = time.perf_counter()
    con = duckdb.connect()
    con.execute("PRAGMA threads=4")
    con.execute("PRAGMA memory_limit='2GB'")
    con.execute("PRAGMA temp_directory='D:/task_offloading_datasets/nep/duckdb_tmp'")
    con.execute("PRAGMA preserve_insertion_order=false")
    cpu = (args.extracted_root / "VM_CPU.csv").as_posix()
    rtt = (args.extracted_root / "SITE_RTT.csv").as_posix()
    quantiles = ", ".join(str(p) for p in PERCENTILES)

    cpu_stats = con.execute(
        f"""
        WITH src AS (
            SELECT site_id, TRY_CAST(cpu_rate AS DOUBLE) AS r,
                   TRY_CAST(report_ts AS BIGINT) AS ts
            FROM read_csv('{cpu}', header=true, all_varchar=true,
                          ignore_errors=true, strict_mode=false)
        ), site_hour AS (
            SELECT site_id, ts // 3600 AS hour, avg(r) AS load,
                   count(*) AS valid_rows
            FROM src
            WHERE r IS NOT NULL AND isfinite(r) AND r BETWEEN 0 AND 1
              AND ts IS NOT NULL AND site_id IS NOT NULL
            GROUP BY site_id, hour
        )
        SELECT NULL, sum(valid_rows), count(*),
               quantile_cont(load, [{quantiles}]), avg(load)
        FROM site_hour
        """
    ).fetchone()
    print(f"cpu done {time.perf_counter() - started:.0f}s", flush=True)
    rtt_stats = con.execute(
        f"""
        WITH src AS (
            SELECT from_site_id AS a, to_site_id AS b,
                   TRY_CAST(rtt AS DOUBLE) AS r
            FROM read_csv('{rtt}', header=true, all_varchar=true,
                          ignore_errors=true, strict_mode=false)
        )
        SELECT count(*),
               count(*) FILTER (WHERE r IS NOT NULL AND isfinite(r) AND r > 0
                                AND r <= 1000 AND a <> b),
               approx_quantile(r, [{quantiles}]) FILTER (
                   WHERE r IS NOT NULL AND isfinite(r) AND r > 0 AND r <= 1000
                     AND a <> b),
               avg(r) FILTER (WHERE r IS NOT NULL AND isfinite(r) AND r > 0
                              AND r <= 1000 AND a <> b)
        FROM src
        """
    ).fetchone()
    print(f"rtt done {time.perf_counter() - started:.0f}s", flush=True)
    assert cpu_stats is not None and rtt_stats is not None
    profile = {
        "profile_version": "1.0.0",
        "profile_id": "nep_quantiles_v1",
        "derived_on": dt.date.today().isoformat(),
        "source_id": "edge_workloads_traces_nep_large",
        "source_archive_sha256": (
            "ff80a07e25f8055fed1a64439194a47721d9bc2330c79c9a8201649072852816"
        ),
        "aggregate_only": True,
        "raw_rows_included": False,
        "percentiles": PERCENTILES,
        "edge_site_hour_cpu_load": {
            "definition": "mean VM cpu_rate per (site, hour); rows with finite "
            "rate in [0, 1]",
            "source_rows": "see nep_large_full_trace_ff80a07e.json",
            "valid_rows": cpu_stats[1],
            "site_hours": cpu_stats[2],
            "quantiles": cpu_stats[3],
            "mean": cpu_stats[4],
        },
        "inter_site_rtt_ms": {
            "definition": "SITE_RTT.rtt in ms; 0 < rtt <= 1000, from != to; "
            "approximate quantiles (DuckDB approx_quantile, t-digest); "
            "malformed CSV lines skipped",
            "source_rows": rtt_stats[0],
            "valid_rows": rtt_stats[1],
            "quantiles": rtt_stats[2],
            "mean": rtt_stats[3],
        },
        "duckdb_version": duckdb.__version__,
    }
    OUTPUT.write_text(json.dumps(profile, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUTPUT} in {time.perf_counter() - started:.0f}s")


if __name__ == "__main__":
    main()
