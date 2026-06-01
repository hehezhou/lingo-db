#!/usr/bin/env python3
import argparse
import hashlib
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
_RE_SUBOP_RESULT_BLOCK = re.compile(
    r"^// result_begin: query\[(\d+)\]\s*$\n(.*?)^// result_end: query\[\1\]\s*$",
    re.M | re.S,
)


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


def normalize_result_block(text: Optional[str]) -> Optional[str]:
    if text is None:
        return None
    lines = [line.rstrip() for line in text.splitlines()]
    while lines and lines[0] == "":
        lines.pop(0)
    while lines and lines[-1] == "":
        lines.pop()
    if not lines:
        return None
    return "\n".join(lines)


def hash_text(text: Optional[str]) -> Optional[str]:
    if text is None:
        return None
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def parse_run_sql_result_block(stdout: str) -> Optional[str]:
    lines = stdout.splitlines()
    start = None
    for i, line in enumerate(lines):
        if line.startswith("|"):
            start = i
            break
    if start is None:
        return None

    block: List[str] = []
    for line in lines[start:]:
        stripped = line.rstrip()
        if stripped == "":
            if block:
                break
            continue
        if stripped.startswith("|") or set(stripped) == {"-"}:
            block.append(stripped)
            continue
        if block:
            break
    return normalize_result_block("\n".join(block))


def parse_subop_result_blocks(stdout: str) -> Dict[int, str]:
    out: Dict[int, str] = {}
    for m in _RE_SUBOP_RESULT_BLOCK.finditer(stdout):
        q_idx = int(m.group(1))
        block = normalize_result_block(m.group(2))
        if block is not None:
            out[q_idx] = block
    return out


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


def build_subop_result_payload(
    cmd_result: CmdResult,
    singles: Dict[int, Dict[str, Any]],
    single_result_blocks: Dict[int, Optional[str]],
    q0: int,
    q1: int,
) -> Dict[str, Any]:
    metrics = parse_subop_metrics(cmd_result.stdout)
    subop_results = parse_subop_result_blocks(cmd_result.stdout)

    payload: Dict[str, Any] = {
        "returncode": cmd_result.returncode,
        "wall_ms": cmd_result.wall_ms,
        **metrics,
    }

    expected0 = single_result_blocks.get(q0)
    expected1 = single_result_blocks.get(q1)
    actual0 = subop_results.get(0)
    actual1 = subop_results.get(1)
    result_compare: Dict[str, Any] = {
        "q0_expected_sha256": singles[q0].get("result_sha256"),
        "q1_expected_sha256": singles[q1].get("result_sha256"),
        "q0_actual_sha256": hash_text(actual0),
        "q1_actual_sha256": hash_text(actual1),
        "q0_match": None,
        "q1_match": None,
        "all_match": None,
    }

    q0_expected_hash = singles[q0].get("result_sha256")
    q1_expected_hash = singles[q1].get("result_sha256")
    q0_actual_hash = result_compare["q0_actual_sha256"]
    q1_actual_hash = result_compare["q1_actual_sha256"]
    if q0_expected_hash is not None and q0_actual_hash is not None:
        result_compare["q0_match"] = q0_expected_hash == q0_actual_hash
    if q1_expected_hash is not None and q1_actual_hash is not None:
        result_compare["q1_match"] = q1_expected_hash == q1_actual_hash
    if result_compare["q0_match"] is not None and result_compare["q1_match"] is not None:
        result_compare["all_match"] = bool(result_compare["q0_match"] and result_compare["q1_match"])
    if result_compare["q0_match"] is False:
        result_compare["q0_expected_preview"] = "\n".join(expected0.splitlines()[:12]) if expected0 else None
        result_compare["q0_actual_preview"] = "\n".join(actual0.splitlines()[:12]) if actual0 else None
    if result_compare["q1_match"] is False:
        result_compare["q1_expected_preview"] = "\n".join(expected1.splitlines()[:12]) if expected1 else None
        result_compare["q1_actual_preview"] = "\n".join(actual1.splitlines()[:12]) if actual1 else None
    payload["result_compare"] = result_compare

    if cmd_result.returncode != 0:
        payload["stderr_tail"] = "\n".join(cmd_result.stderr.splitlines()[-30:])

    return payload


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
    single_result_blocks: Dict[int, Optional[str]] = {}
    for q in range(1, 23):
        sql = os.path.join(args.sql_dir, f"{q}.sql")
        r = run_cmd([run_sql, sql, args.db], env=env, timeout_s=args.timeout_s)
        exec_ms = parse_run_sql_execution_time_ms(r.stdout) if r.returncode == 0 else None
        result_block = parse_run_sql_result_block(r.stdout) if r.returncode == 0 else None
        single_result_blocks[q] = result_block
        singles[q] = {
            "q": q,
            "sql": sql,
            "returncode": r.returncode,
            "execution_time_ms": exec_ms,
            "result_sha256": hash_text(result_block),
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
        r_no_reuse = run_cmd([subop, "--no-reuse-rewrite", args.db, sql_a, sql_b], env=env, timeout_s=args.timeout_s)

        row: Dict[str, Any] = {
            "pair_id": pair_id,
            "q0": a,
            "q1": b,
            "subop": build_subop_result_payload(r, singles, single_result_blocks, a, b),
            "subop_no_reuse_rewrite": build_subop_result_payload(
                r_no_reuse, singles, single_result_blocks, a, b
            ),
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
        subop_metrics = row["subop"]
        subop_no_reuse_metrics = row["subop_no_reuse_rewrite"]
        if "execution_time_ms_total" in subop_metrics and q0 is not None and q1 is not None:
            denom = q0 + q1
            if denom > 0:
                row["ratios"] = {
                    "subop_total_over_run_sql_sum": subop_metrics["execution_time_ms_total"] / denom,
                }
                if "execution_time_ms_q0_q1" in subop_metrics:
                    row["ratios"]["subop_q0_q1_over_run_sql_sum"] = subop_metrics["execution_time_ms_q0_q1"] / denom
                if "execution_time_ms_total" in subop_no_reuse_metrics:
                    row["ratios"]["subop_no_reuse_total_over_run_sql_sum"] = (
                        subop_no_reuse_metrics["execution_time_ms_total"] / denom
                    )
                if "execution_time_ms_q0_q1" in subop_no_reuse_metrics:
                    row["ratios"]["subop_no_reuse_q0_q1_over_run_sql_sum"] = (
                        subop_no_reuse_metrics["execution_time_ms_q0_q1"] / denom
                    )
        if (
            "execution_time_ms_total" in subop_metrics
            and "execution_time_ms_total" in subop_no_reuse_metrics
            and subop_no_reuse_metrics["execution_time_ms_total"] > 0
        ):
            row.setdefault("ratios", {})
            row["ratios"]["subop_reuse_over_same_path_no_reuse_total"] = (
                subop_metrics["execution_time_ms_total"] / subop_no_reuse_metrics["execution_time_ms_total"]
            )
        if (
            "execution_time_ms_q0_q1" in subop_metrics
            and "execution_time_ms_q0_q1" in subop_no_reuse_metrics
            and subop_no_reuse_metrics["execution_time_ms_q0_q1"] > 0
        ):
            row.setdefault("ratios", {})
            row["ratios"]["subop_reuse_over_same_path_no_reuse_q0_q1"] = (
                subop_metrics["execution_time_ms_q0_q1"] / subop_no_reuse_metrics["execution_time_ms_q0_q1"]
            )

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
