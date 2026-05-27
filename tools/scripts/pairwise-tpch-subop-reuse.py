#!/usr/bin/env python3
import argparse
import itertools
import json
import os
import re
import subprocess
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class CmdResult:
    argv: List[str]
    returncode: int
    stdout: str
    stderr: str
    wall_ms: float


def run_cmd(argv: List[str], env: Dict[str, str], timeout_s: int) -> CmdResult:
    t0 = time.time()
    proc = subprocess.run(
        argv,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        timeout=timeout_s,
    )
    wall_ms = (time.time() - t0) * 1000.0
    return CmdResult(
        argv=argv,
        returncode=proc.returncode,
        stdout=proc.stdout.decode("utf-8", errors="replace"),
        stderr=proc.stderr.decode("utf-8", errors="replace"),
        wall_ms=wall_ms,
    )


_RE_RUNSQL_ROW = re.compile(r"^\s*(\d+\.sql)\s+.*\s(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)\s*$")
_RE_SUBOP_REUSE = re.compile(r"^//\s+reuse_targets:\s+query\[0\]=(\d+)\s+query\[1\]=(\d+)\s*$", re.M)
_RE_SUBOP_REUSE_NO_TABLE = re.compile(r"^//\s+reuse_targets_no_table:\s+query\[0\]=(\d+)\s+query\[1\]=(\d+)\s*$", re.M)
_RE_SUBOP_REUSE_MAPPED = re.compile(r"^//\s+reuse_targets_q0_mapped:\s+(\d+)\s*$", re.M)
_RE_SUBOP_REUSE_MAPPED_NO_TABLE = re.compile(r"^//\s+reuse_targets_q0_mapped_no_table:\s+(\d+)\s*$", re.M)
_RE_SUBOP_TIMING_TOTAL = re.compile(r"^//\s+timing:\s+optimization_ms=.*\s+execution_time_ms=([0-9.eE+\-]+)\s*$", re.M)
_RE_SUBOP_TIMING_PER_RUN = re.compile(r"^//\s+timing_execution_time_ms:\s+per_run=\[([^\]]*)\]\s+total=([0-9.eE+\-]+)\s*$", re.M)


def parse_run_sql_execution_time_ms(stdout: str) -> Optional[float]:
    # TimingPrinter prints a header row and a single result row with "... executionTime total".
    # We parse the row for "<N>.sql ... <executionTime> <total>" and take executionTime.
    for line in stdout.splitlines():
        m = _RE_RUNSQL_ROW.match(line)
        if not m:
            continue
        # m.group(2)=executionTime, m.group(3)=total
        return float(m.group(2))
    return None


def parse_subop_metrics(stdout: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    m = _RE_SUBOP_REUSE.search(stdout)
    if m:
        out["reuse_targets"] = {"q0": int(m.group(1)), "q1": int(m.group(2))}
    m = _RE_SUBOP_REUSE_NO_TABLE.search(stdout)
    if m:
        out["reuse_targets_no_table"] = {"q0": int(m.group(1)), "q1": int(m.group(2))}
    m = _RE_SUBOP_REUSE_MAPPED.search(stdout)
    if m:
        out["reuse_targets_q0_mapped"] = int(m.group(1))
    m = _RE_SUBOP_REUSE_MAPPED_NO_TABLE.search(stdout)
    if m:
        out["reuse_targets_q0_mapped_no_table"] = int(m.group(1))

    m = _RE_SUBOP_TIMING_TOTAL.search(stdout)
    if m:
        out["execution_time_ms_total"] = float(m.group(1))

    m = _RE_SUBOP_TIMING_PER_RUN.search(stdout)
    if m:
        per_run_raw = m.group(1).strip()
        per_run = [float(x) for x in per_run_raw.split(",") if x.strip() != ""]
        out["execution_time_ms_per_run"] = per_run
        out["execution_time_ms_per_run_total"] = float(m.group(2))
        # Common case: [synthetic, q0, q1]. If only 2 entries, treat as [q0, q1].
        if len(per_run) >= 3:
            out["execution_time_ms_q0_q1"] = per_run[1] + per_run[2]
        elif len(per_run) == 2:
            out["execution_time_ms_q0_q1"] = per_run[0] + per_run[1]
        elif len(per_run) == 1:
            out["execution_time_ms_q0_q1"] = per_run[0]

    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-dir", default="build/lingodb-release", help="e.g. build/lingodb-release")
    ap.add_argument("--db", default="resources/data/tpch-1", help="db directory")
    ap.add_argument("--sql-dir", default="resources/sql/tpch", help="directory with 1.sql..22.sql")
    ap.add_argument("--out", default="tmp_pairwise_tpch_release.json", help="output JSON path")
    ap.add_argument("--timeout-s", type=int, default=600, help="per command timeout")
    ap.add_argument("--skip-existing", action="store_true", help="if out exists, resume by skipping done pairs")
    args = ap.parse_args()

    run_sql = os.path.join(args.build_dir, "run-sql")
    subop = os.path.join(args.build_dir, "subop-state-reuse")

    if not os.path.exists(run_sql) or not os.path.exists(subop):
        raise SystemExit(f"missing binaries under {args.build_dir} (need run-sql and subop-state-reuse)")

    env = dict(os.environ)
    env.pop("LINGODB_SKIP_EXECUTE", None)
    env.pop("LINGODB_SUBOP_FORCE_SEQUENTIAL", None)

    # Precompute run-sql execution times.
    singles: Dict[int, Dict[str, Any]] = {}
    for q in range(1, 23):
        sql = os.path.join(args.sql_dir, f"{q}.sql")
        r = run_cmd([run_sql, sql, args.db], env=env, timeout_s=args.timeout_s)
        exec_ms = parse_run_sql_execution_time_ms(r.stdout) if r.returncode == 0 else None
        singles[q] = {
            "q": q,
            "sql": sql,
            "returncode": r.returncode,
            "execution_time_ms": exec_ms,
            "wall_ms": r.wall_ms,
        }

    existing_pairs: Dict[str, Any] = {}
    results: List[Dict[str, Any]] = []
    if args.skip_existing and os.path.exists(args.out):
        with open(args.out, "r", encoding="utf-8") as f:
            prev = json.load(f)
        for row in prev.get("pairs", []):
            existing_pairs[row["pair_id"]] = row
        results.extend(prev.get("pairs", []))

    pairs = list(itertools.combinations(range(1, 23), 2))
    total = len(pairs)
    done = 0
    t_all = time.time()

    for a, b in pairs:
        pair_id = f"Q{a}-Q{b}"
        if pair_id in existing_pairs:
            done += 1
            continue

        sql_a = os.path.join(args.sql_dir, f"{a}.sql")
        sql_b = os.path.join(args.sql_dir, f"{b}.sql")

        r = run_cmd([subop, args.db, sql_a, sql_b], env=env, timeout_s=args.timeout_s)
        metrics = parse_subop_metrics(r.stdout)

        row: Dict[str, Any] = {
            "pair_id": pair_id,
            "q0": a,
            "q1": b,
            "subop": {
                "returncode": r.returncode,
                "wall_ms": r.wall_ms,
                **metrics,
            },
            "run_sql": {
                "q0_execution_time_ms": singles[a]["execution_time_ms"],
                "q1_execution_time_ms": singles[b]["execution_time_ms"],
            },
        }
        q0 = singles[a]["execution_time_ms"]
        q1 = singles[b]["execution_time_ms"]
        if q0 is not None and q1 is not None:
            row["run_sql"]["q0_plus_q1_execution_time_ms"] = q0 + q1

        # Ratios (when data is available).
        if "execution_time_ms_total" in metrics and q0 is not None and q1 is not None:
            denom = q0 + q1
            if denom > 0:
                row["ratios"] = {
                    "subop_total_over_run_sql_sum": metrics["execution_time_ms_total"] / denom,
                }
                if "execution_time_ms_q0_q1" in metrics:
                    row["ratios"]["subop_q0_q1_over_run_sql_sum"] = metrics["execution_time_ms_q0_q1"] / denom

        # Store a small stderr tail for debugging failures.
        if r.returncode != 0:
            row["subop"]["stderr_tail"] = "\n".join(r.stderr.splitlines()[-30:])

        results.append(row)
        done += 1
        if done % 10 == 0 or done == total:
            elapsed = time.time() - t_all
            print(f"[{done}/{total}] last={pair_id} elapsed_s={elapsed:.1f}")

            out_obj = {
                "meta": {
                    "build_dir": args.build_dir,
                    "db": args.db,
                    "sql_dir": args.sql_dir,
                    "generated_at_unix": time.time(),
                },
                "singles_run_sql": singles,
                "pairs": results,
            }
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump(out_obj, f, indent=2, sort_keys=True)

    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

