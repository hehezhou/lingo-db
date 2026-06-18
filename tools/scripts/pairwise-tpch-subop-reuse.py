#!/usr/bin/env python3
import argparse
import hashlib
import itertools
import json
import os
import re
import statistics
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
_RE_SUBOP_REUSE = re.compile(
    r"^//\s+reuse_targets:\s+(?:query\[0\]=(\d+)\s+query\[1\]=(\d+)|per_query=\[(\d+),(\d+)\])\s*$",
    re.M,
)
_RE_SUBOP_REUSE_NO_TABLE = re.compile(
    r"^//\s+reuse_targets_no_table:\s+(?:query\[0\]=(\d+)\s+query\[1\]=(\d+)|per_query=\[(\d+),(\d+)\])\s*$",
    re.M,
)
_RE_SUBOP_REUSE_MAPPED = re.compile(r"^//\s+reuse_targets_(?:q0|synthetic)_mapped:\s+(\d+)\s*$", re.M)
_RE_SUBOP_REUSE_MAPPED_NO_TABLE = re.compile(
    r"^//\s+reuse_targets_(?:q0|synthetic)_mapped_no_table:\s+(\d+)\s*$",
    re.M,
)
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


def parse_reuse_count_pair(match: re.Match[str]) -> Dict[str, int]:
    q0 = match.group(1) or match.group(3)
    q1 = match.group(2) or match.group(4)
    return {"q0": int(q0), "q1": int(q1)}


def parse_subop_metrics(stdout: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    m = _RE_SUBOP_REUSE.search(stdout)
    if m:
        out["reuse_targets"] = parse_reuse_count_pair(m)
    m = _RE_SUBOP_REUSE_NO_TABLE.search(stdout)
    if m:
        out["reuse_targets_no_table"] = parse_reuse_count_pair(m)
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


def summarize_reuse_counts(subop_payload: Dict[str, Any]) -> Dict[str, Any]:
    reuse = subop_payload.get("reuse_targets")
    reuse_no_table = subop_payload.get("reuse_targets_no_table")
    summary: Dict[str, Any] = {}
    if reuse:
        total = int(reuse.get("q0", 0)) + int(reuse.get("q1", 0))
        summary["reuse_targets_total"] = total
        summary["has_reuse"] = total > 0
    if reuse_no_table:
        total_no_table = int(reuse_no_table.get("q0", 0)) + int(reuse_no_table.get("q1", 0))
        summary["reuse_targets_no_table_total"] = total_no_table
        summary["has_reuse_no_table"] = total_no_table > 0
    if "reuse_targets_q0_mapped" in subop_payload:
        summary["reuse_targets_q0_mapped"] = subop_payload["reuse_targets_q0_mapped"]
    if "reuse_targets_q0_mapped_no_table" in subop_payload:
        summary["reuse_targets_q0_mapped_no_table"] = subop_payload["reuse_targets_q0_mapped_no_table"]
    return summary


def parse_pair_spec(spec: str) -> List[Tuple[int, int]]:
    pairs: List[Tuple[int, int]] = []
    if not spec.strip():
        return pairs
    for raw in spec.split(","):
        token = raw.strip().upper()
        if not token:
            continue
        m = re.fullmatch(r"Q?(\d+)\s*[-:]\s*Q?(\d+)", token)
        if not m:
            raise SystemExit(f"invalid pair spec entry: {raw!r}")
        a = int(m.group(1))
        b = int(m.group(2))
        if a == b:
            raise SystemExit(f"invalid self pair: {raw!r}")
        if not (1 <= a <= 22 and 1 <= b <= 22):
            raise SystemExit(f"pair out of TPC-H range 1..22: {raw!r}")
        if a > b:
            a, b = b, a
        pairs.append((a, b))
    seen = set()
    out: List[Tuple[int, int]] = []
    for p in pairs:
        if p in seen:
            continue
        seen.add(p)
        out.append(p)
    return out


def compute_ratios(
    subop_metrics: Dict[str, Any],
    subop_no_reuse_metrics: Dict[str, Any],
    q0_execution_time_ms: Optional[float],
    q1_execution_time_ms: Optional[float],
) -> Dict[str, float]:
    ratios: Dict[str, float] = {}
    if "execution_time_ms_total" in subop_metrics and q0_execution_time_ms is not None and q1_execution_time_ms is not None:
        denom = q0_execution_time_ms + q1_execution_time_ms
        if denom > 0:
            ratios["subop_total_over_run_sql_sum"] = subop_metrics["execution_time_ms_total"] / denom
            if "execution_time_ms_q0_q1" in subop_metrics:
                ratios["subop_q0_q1_over_run_sql_sum"] = subop_metrics["execution_time_ms_q0_q1"] / denom
            if "execution_time_ms_total" in subop_no_reuse_metrics:
                ratios["subop_no_reuse_total_over_run_sql_sum"] = (
                    subop_no_reuse_metrics["execution_time_ms_total"] / denom
                )
            if "execution_time_ms_q0_q1" in subop_no_reuse_metrics:
                ratios["subop_no_reuse_q0_q1_over_run_sql_sum"] = (
                    subop_no_reuse_metrics["execution_time_ms_q0_q1"] / denom
                )
    if (
        "execution_time_ms_total" in subop_metrics
        and "execution_time_ms_total" in subop_no_reuse_metrics
        and subop_no_reuse_metrics["execution_time_ms_total"] > 0
    ):
        ratios["subop_reuse_over_same_path_no_reuse_total"] = (
            subop_metrics["execution_time_ms_total"] / subop_no_reuse_metrics["execution_time_ms_total"]
        )
        ratios["subop_speedup_over_same_path_no_reuse_total"] = (
            subop_no_reuse_metrics["execution_time_ms_total"] / subop_metrics["execution_time_ms_total"]
        )
    if (
        "execution_time_ms_q0_q1" in subop_metrics
        and "execution_time_ms_q0_q1" in subop_no_reuse_metrics
        and subop_no_reuse_metrics["execution_time_ms_q0_q1"] > 0
    ):
        ratios["subop_reuse_over_same_path_no_reuse_q0_q1"] = (
            subop_metrics["execution_time_ms_q0_q1"] / subop_no_reuse_metrics["execution_time_ms_q0_q1"]
        )
        ratios["subop_speedup_over_same_path_no_reuse_q0_q1"] = (
            subop_no_reuse_metrics["execution_time_ms_q0_q1"] / subop_metrics["execution_time_ms_q0_q1"]
        )
    return ratios


def summarize_values(values: List[float]) -> Dict[str, Any]:
    if not values:
        return {}
    return {
        "values": values,
        "min": min(values),
        "max": max(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
    }


def geo_mean(values: List[float]) -> Optional[float]:
    positive = [v for v in values if v > 0]
    if len(positive) != len(values) or not positive:
        return None
    return statistics.geometric_mean(positive)


def aggregate_pair_runs(pair_runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"repetitions": len(pair_runs)}
    for section in ("subop", "subop_no_reuse_rewrite"):
        section_out: Dict[str, Any] = {}
        for metric in ("execution_time_ms_total", "execution_time_ms_q0_q1", "wall_ms"):
            values = [run[section][metric] for run in pair_runs if metric in run.get(section, {})]
            summary = summarize_values(values)
            if summary:
                section_out[metric] = summary
        out[section] = section_out
    ratio_out: Dict[str, Any] = {}
    ratio_keys = sorted({k for run in pair_runs for k in run.get("ratios", {}).keys()})
    for key in ratio_keys:
        values = [run["ratios"][key] for run in pair_runs if key in run.get("ratios", {})]
        summary = summarize_values(values)
        if summary:
            ratio_out[key] = summary
    out["ratios"] = ratio_out
    return out


def compact_pair_row(row: Dict[str, Any]) -> Dict[str, Any]:
    pair_runs = row.get("pair_runs") or [
        {
            "subop": row.get("subop", {}),
            "subop_no_reuse_rewrite": row.get("subop_no_reuse_rewrite", {}),
        }
    ]
    reuse_times = [
        run["subop"]["execution_time_ms_total"]
        for run in pair_runs
        if "execution_time_ms_total" in run.get("subop", {})
    ]
    no_reuse_times = [
        run["subop_no_reuse_rewrite"]["execution_time_ms_total"]
        for run in pair_runs
        if "execution_time_ms_total" in run.get("subop_no_reuse_rewrite", {})
    ]
    reuse_geo = geo_mean(reuse_times)
    no_reuse_geo = geo_mean(no_reuse_times)
    summary = row.get("reuse_summary", {})
    return {
        "query_pair": row["pair_id"],
        "reuse_count": summary.get("reuse_targets_no_table_total"),
        "state_reuse_total_time_geo_mean_ms": reuse_geo,
        "no_reuse_total_time_geo_mean_ms": no_reuse_geo,
        "speedup": (no_reuse_geo / reuse_geo) if reuse_geo and no_reuse_geo else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-dir", default="build/lingodb-release", help="e.g. build/lingodb-release")
    ap.add_argument("--db", default="resources/data/tpch-1", help="db directory")
    ap.add_argument("--sql-dir", default="resources/sql/tpch", help="directory with 1.sql..22.sql")
    ap.add_argument("--out", default="tmp_pairwise_tpch_release.json", help="output JSON path")
    ap.add_argument("--timeout-s", type=int, default=600, help="per command timeout")
    ap.add_argument("--skip-existing", action="store_true", help="if out exists, resume by skipping done pairs")
    ap.add_argument("--pairs", default="", help="comma-separated whitelist, e.g. Q7-Q8,Q8-Q9")
    ap.add_argument("--repetitions", type=int, default=1, help="number of reuse/no-reuse measurements per pair")
    ap.add_argument("--compact-output", action="store_true", help="write only a compact JSON list")
    args = ap.parse_args()
    if args.repetitions < 1:
        raise SystemExit("--repetitions must be >= 1")

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

    pairs = parse_pair_spec(args.pairs) if args.pairs else list(itertools.combinations(range(1, 23), 2))
    total = len(pairs)
    done = 0
    reuse_pair_count = sum(1 for row in results if row.get("reuse_summary", {}).get("has_reuse"))
    reuse_no_table_pair_count = sum(
        1 for row in results if row.get("reuse_summary", {}).get("has_reuse_no_table")
    )
    t_all = time.time()

    for a, b in pairs:
        pair_id = f"Q{a}-Q{b}"
        if pair_id in existing_pairs:
            done += 1
            continue

        sql_a = os.path.join(args.sql_dir, f"{a}.sql")
        sql_b = os.path.join(args.sql_dir, f"{b}.sql")

        pair_runs: List[Dict[str, Any]] = []
        for repetition in range(args.repetitions):
            r = run_cmd([subop, args.db, sql_a, sql_b], env=env, timeout_s=args.timeout_s)
            r_no_reuse = run_cmd([subop, "--no-reuse-rewrite", args.db, sql_a, sql_b], env=env, timeout_s=args.timeout_s)
            subop_payload = build_subop_result_payload(r, singles, single_result_blocks, a, b)
            no_reuse_payload = build_subop_result_payload(r_no_reuse, singles, single_result_blocks, a, b)
            run_row: Dict[str, Any] = {
                "repetition": repetition + 1,
                "subop": subop_payload,
                "subop_no_reuse_rewrite": no_reuse_payload,
                "ratios": compute_ratios(
                    subop_payload,
                    no_reuse_payload,
                    singles[a]["execution_time_ms"],
                    singles[b]["execution_time_ms"],
                ),
            }
            pair_runs.append(run_row)

        row: Dict[str, Any] = {
            "pair_id": pair_id,
            "q0": a,
            "q1": b,
            "subop": pair_runs[0]["subop"],
            "subop_no_reuse_rewrite": pair_runs[0]["subop_no_reuse_rewrite"],
            "pair_runs": pair_runs,
            "pair_run_aggregate": aggregate_pair_runs(pair_runs),
            "run_sql": {
                "q0_execution_time_ms": singles[a]["execution_time_ms"],
                "q1_execution_time_ms": singles[b]["execution_time_ms"],
            },
        }
        row["reuse_summary"] = summarize_reuse_counts(row["subop"])
        if row["reuse_summary"].get("has_reuse"):
            reuse_pair_count += 1
        if row["reuse_summary"].get("has_reuse_no_table"):
            reuse_no_table_pair_count += 1
        q0 = singles[a]["execution_time_ms"]
        q1 = singles[b]["execution_time_ms"]
        if q0 is not None and q1 is not None:
            row["run_sql"]["q0_plus_q1_execution_time_ms"] = q0 + q1

        row["ratios"] = pair_runs[0]["ratios"]

        results.append(row)
        done += 1
        if done % 10 == 0 or done == total:
            elapsed = time.time() - t_all
            print(
                f"[{done}/{total}] last={pair_id} reuse_pairs={reuse_pair_count} "
                f"reuse_no_table_pairs={reuse_no_table_pair_count} elapsed_s={elapsed:.1f}"
            )

            out_obj = {
                "meta": {
                    "build_dir": args.build_dir,
                    "db": args.db,
                    "sql_dir": args.sql_dir,
                    "generated_at_unix": time.time(),
                    "reuse_pair_count": reuse_pair_count,
                    "reuse_no_table_pair_count": reuse_no_table_pair_count,
                    "pairs_whitelist": args.pairs,
                    "repetitions": args.repetitions,
                },
                "singles_run_sql": singles,
                "pairs": results,
            }
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump([compact_pair_row(row) for row in results] if args.compact_output else out_obj,
                          f, indent=2, sort_keys=True)

    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
