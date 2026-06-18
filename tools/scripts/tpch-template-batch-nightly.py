#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import re
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


MODES = ["reuse_off", "reuse_on_bloom_on", "reuse_on_bloom_off"]

TIMING_RE = re.compile(
    r"^//\s+timing:\s+optimization_ms=([0-9.eE+\-]+)\s+execution_time_ms=([0-9.eE+\-]+)\s*$",
    re.M,
)
TIMING_EXEC_RE = re.compile(r"^//\s+timing_execution_time_ms:.*\btotal=([0-9.eE+\-]+)\s*$", re.M)
REUSE_TARGETS_RE = re.compile(r"^//\s+reuse_targets:\s+per_query=\[([^\]]*)\]\s*$", re.M)
REUSE_TARGETS_NO_TABLE_RE = re.compile(r"^//\s+reuse_targets_no_table:\s+per_query=\[([^\]]*)\]\s*$", re.M)
REUSE_MAPPED_RE = re.compile(r"^//\s+reuse_targets_synthetic_mapped:\s+([0-9]+)\s*$", re.M)
REUSE_MAPPED_NO_TABLE_RE = re.compile(r"^//\s+reuse_targets_synthetic_mapped_no_table:\s+([0-9]+)\s*$", re.M)


def parse_int_list(spec: str) -> List[int]:
    return [int(x.strip()) for x in spec.split(",") if x.strip()]


def parse_str_list(spec: str) -> List[str]:
    return [x.strip() for x in spec.split(",") if x.strip()]


def parse_counts(text: str, regex: re.Pattern[str]) -> Optional[List[int]]:
    m = regex.search(text)
    if not m:
        return None
    body = m.group(1).strip()
    if not body:
        return []
    return [int(x.strip()) for x in body.split(",") if x.strip()]


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def extract_result_blocks(stdout: str) -> List[str]:
    blocks: List[str] = []
    cur: List[str] = []
    in_block = False
    for line in stdout.splitlines():
        if line.startswith("// result_begin:"):
            in_block = True
            cur = [line]
            continue
        if in_block:
            cur.append(line)
            if line.startswith("// result_end:"):
                blocks.append("\n".join(cur))
                in_block = False
    return blocks


def rowset_text_for_block(block: str) -> str:
    rows: List[str] = []
    seen_header = False
    for line in block.splitlines():
        if not line.startswith("|"):
            continue
        if not seen_header:
            seen_header = True
            continue
        if "..." in line:
            continue
        rows.append(line)
    rows.sort()
    return "\n".join(rows)


def result_hashes(stdout: str) -> Tuple[List[str], List[str]]:
    exact_hashes: List[str] = []
    rowset_hashes: List[str] = []
    for block in extract_result_blocks(stdout):
        exact_hashes.append(sha256_text(block.strip()))
        rowset_hashes.append(sha256_text(rowset_text_for_block(block)))
    return exact_hashes, rowset_hashes


def mismatch_indices(hashes: List[str], ref_hashes: List[str]) -> List[int]:
    n = min(len(hashes), len(ref_hashes))
    out = [i for i in range(n) if hashes[i] != ref_hashes[i]]
    if len(hashes) != len(ref_hashes):
        out.extend(range(n, max(len(hashes), len(ref_hashes))))
    return out


def geo_mean(values: List[float]) -> Optional[float]:
    values = [v for v in values if v > 0]
    if not values:
        return None
    return statistics.geometric_mean(values)


def load_existing(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"records": []}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def record_key(record: Dict[str, Any]) -> Tuple[str, int, int, str, int]:
    return (
        str(record["db_label"]),
        int(record["template"]),
        int(record["batch_size"]),
        str(record["mode"]),
        int(record["rep"]),
    )


def make_db_label(db_path: str) -> str:
    return Path(db_path).name


def query_paths(query_dir: Path, template: int, batch_size: int) -> List[str]:
    return [str(query_dir / f"q{template}_{i}.sql") for i in range(1, batch_size + 1)]


def build_argv(subop_binary: str, mode: str, db: str, sqls: List[str]) -> List[str]:
    argv = [subop_binary, db, *sqls]
    if mode == "reuse_off":
        argv.insert(1, "--no-reuse-rewrite")
    return argv


def build_env(mode: str) -> Dict[str, str]:
    env = dict(os.environ)
    env.pop("LINGODB_SKIP_EXECUTE", None)
    env["LINGODB_SUBOP_FORCE_SEQUENTIAL"] = "1"
    if mode == "reuse_on_bloom_off":
        env["LINGODB_DISABLE_FILTER_PRED_BLOOM_ADAPTATION"] = "1"
    else:
        env.pop("LINGODB_DISABLE_FILTER_PRED_BLOOM_ADAPTATION", None)
    return env


def run_one(
    subop_binary: str,
    db: str,
    query_dir: Path,
    template: int,
    batch_size: int,
    mode: str,
    rep: int,
    timeout_s: int,
    ref_record: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    sqls = query_paths(query_dir, template, batch_size)
    argv = build_argv(subop_binary, mode, db, sqls)
    env = build_env(mode)
    started = time.time()
    proc = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, timeout=timeout_s)
    wall_ms = (time.time() - started) * 1000.0
    stdout = proc.stdout.decode("utf-8", errors="replace")
    stderr = proc.stderr.decode("utf-8", errors="replace")

    exact_hashes, rowset_hashes = result_hashes(stdout)
    timing = TIMING_RE.search(stdout)
    timing_exec = TIMING_EXEC_RE.search(stdout)
    execution_time_ms_total = None
    optimization_ms = None
    if timing:
        optimization_ms = float(timing.group(1))
        execution_time_ms_total = float(timing.group(2))
    elif timing_exec:
        execution_time_ms_total = float(timing_exec.group(1))

    exact_mismatches: List[int] = []
    rowset_mismatches: List[int] = []
    order_only_mismatches: List[int] = []
    reference_missing = mode != "reuse_off" and ref_record is None
    if ref_record is not None:
        exact_mismatches = mismatch_indices(exact_hashes, ref_record.get("result_exact_block_hashes", []))
        rowset_mismatches = mismatch_indices(rowset_hashes, ref_record.get("result_rowset_block_hashes", []))
        order_only_mismatches = [i for i in exact_mismatches if i not in set(rowset_mismatches)]

    text_for_reuse = stdout + "\n" + stderr
    mapped = REUSE_MAPPED_RE.search(text_for_reuse)
    mapped_no_table = REUSE_MAPPED_NO_TABLE_RE.search(text_for_reuse)
    record: Dict[str, Any] = {
        "db": db,
        "db_label": make_db_label(db),
        "template": template,
        "batch_size": batch_size,
        "mode": mode,
        "rep": rep,
        "argv": argv,
        "returncode": proc.returncode,
        "wall_ms": wall_ms,
        "optimization_ms": optimization_ms,
        "execution_time_ms_total": execution_time_ms_total,
        "reuse_targets": parse_counts(text_for_reuse, REUSE_TARGETS_RE),
        "reuse_targets_no_table": parse_counts(text_for_reuse, REUSE_TARGETS_NO_TABLE_RE),
        "reuse_targets_synthetic_mapped": int(mapped.group(1)) if mapped else None,
        "reuse_targets_synthetic_mapped_no_table": int(mapped_no_table.group(1)) if mapped_no_table else None,
        "result_block_count": len(exact_hashes),
        "result_exact_hash": sha256_text("\n".join(exact_hashes)),
        "result_rowset_hash": sha256_text("\n".join(rowset_hashes)),
        "result_exact_block_hashes": exact_hashes,
        "result_rowset_block_hashes": rowset_hashes,
        "exact_mismatches_vs_reuse_off": exact_mismatches,
        "rowset_mismatches_vs_reuse_off": rowset_mismatches,
        "order_only_mismatches_vs_reuse_off": order_only_mismatches,
        "reference_missing": reference_missing,
        "stderr_tail": "\n".join(stderr.splitlines()[-30:]),
    }
    if proc.returncode != 0:
        record["stdout_tail"] = "\n".join(stdout.splitlines()[-30:])
    return record


def build_summary(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, int, int, str], List[Dict[str, Any]]] = {}
    for r in records:
        grouped.setdefault((r["db_label"], r["template"], r["batch_size"], r["mode"]), []).append(r)
    out: List[Dict[str, Any]] = []
    for (db_label, template, batch_size, mode), rows in sorted(grouped.items()):
        ok_rows = [r for r in rows if r.get("returncode") == 0]
        times = [r["execution_time_ms_total"] for r in ok_rows if r.get("execution_time_ms_total") is not None]
        rowset_mismatch_runs = sum(1 for r in rows if r.get("rowset_mismatches_vs_reuse_off"))
        exact_mismatch_runs = sum(1 for r in rows if r.get("exact_mismatches_vs_reuse_off"))
        reference_missing_runs = sum(1 for r in rows if r.get("reference_missing"))
        reuse_counts = [
            sum(r["reuse_targets_no_table"])
            for r in ok_rows
            if isinstance(r.get("reuse_targets_no_table"), list)
        ]
        out.append(
            {
                "db_label": db_label,
                "template": template,
                "batch_size": batch_size,
                "mode": mode,
                "runs": len(rows),
                "success_runs": len(ok_rows),
                "rowset_mismatch_runs": rowset_mismatch_runs,
                "exact_mismatch_runs": exact_mismatch_runs,
                "reference_missing_runs": reference_missing_runs,
                "total_time_geo_mean_ms": geo_mean(times),
                "total_time_median_ms": statistics.median(times) if times else None,
                "reuse_count_no_table_median_sum": statistics.median(reuse_counts) if reuse_counts else None,
            }
        )
    return out


def write_json(path: Path, meta: Dict[str, Any], records: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "meta": {**meta, "updated_at_unix": time.time()},
        "records": records,
        "summary": build_summary(records),
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
    tmp.replace(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subop-binary", default="build/lingodb-release/subop-state-reuse")
    ap.add_argument("--dbs", default="resources/data/tpch-1,resources/data/tpch-10")
    ap.add_argument("--query-dir", default="queries")
    ap.add_argument("--out", required=True)
    ap.add_argument("--templates", default=",".join(str(i) for i in range(1, 23)))
    ap.add_argument("--batch-sizes", default="2,4,8,16,32,64,128")
    ap.add_argument("--modes", default=",".join(MODES))
    ap.add_argument("--repetitions", type=int, default=5)
    ap.add_argument("--timeout-s", type=int, default=3600)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--progress-every", type=int, default=10)
    args = ap.parse_args()

    dbs = parse_str_list(args.dbs)
    templates = parse_int_list(args.templates)
    batch_sizes = parse_int_list(args.batch_sizes)
    modes = parse_str_list(args.modes)
    for mode in modes:
        if mode not in MODES:
            raise SystemExit(f"unknown mode {mode}; expected one of {MODES}")
    if "reuse_off" not in modes:
        raise SystemExit("modes must include reuse_off because it is the correctness reference")

    out_path = Path(args.out)
    query_dir = Path(args.query_dir)
    data = load_existing(out_path) if args.resume else {"records": []}
    records: List[Dict[str, Any]] = data.get("records", [])
    done = {record_key(r) for r in records}
    refs: Dict[Tuple[str, int, int, int], Dict[str, Any]] = {}
    for r in records:
        if r.get("mode") == "reuse_off" and r.get("returncode") == 0:
            refs[(r["db_label"], r["template"], r["batch_size"], r["rep"])] = r

    max_batch = max(batch_sizes)
    for template in templates:
        missing = [str(query_dir / f"q{template}_{i}.sql") for i in range(1, max_batch + 1)
                   if not (query_dir / f"q{template}_{i}.sql").exists()]
        if missing:
            raise SystemExit(f"missing query files for template {template}: {missing[:5]}")

    meta = {
        "dbs": dbs,
        "query_dir": args.query_dir,
        "subop_binary": args.subop_binary,
        "templates": templates,
        "batch_sizes": batch_sizes,
        "modes": modes,
        "repetitions": args.repetitions,
        "timeout_s": args.timeout_s,
        "created_at_unix": data.get("meta", {}).get("created_at_unix", time.time()),
    }

    total_cases = len(dbs) * len(templates) * len(batch_sizes) * len(modes) * args.repetitions
    completed = len(done)
    started = time.time()
    write_json(out_path, meta, records)

    for db in dbs:
        db_label = make_db_label(db)
        for template in templates:
            for batch_size in batch_sizes:
                for rep in range(args.repetitions):
                    ordered_modes = ["reuse_off"] + [m for m in modes if m != "reuse_off"]
                    for mode in ordered_modes:
                        key = (db_label, template, batch_size, mode, rep)
                        if key in done:
                            continue
                        ref_key = (db_label, template, batch_size, rep)
                        ref_record = refs.get(ref_key)
                        if mode != "reuse_off" and ref_record is None:
                            off_key = (db_label, template, batch_size, "reuse_off", rep)
                            if off_key not in done:
                                off_record = run_one(args.subop_binary, db, query_dir, template, batch_size,
                                                     "reuse_off", rep, args.timeout_s, None)
                                records.append(off_record)
                                done.add(off_key)
                                completed += 1
                                if off_record.get("returncode") == 0:
                                    refs[ref_key] = off_record
                                    ref_record = off_record
                                write_json(out_path, meta, records)
                            else:
                                ref_record = refs.get(ref_key)

                        record = run_one(args.subop_binary, db, query_dir, template, batch_size, mode, rep,
                                         args.timeout_s, ref_record if mode != "reuse_off" else None)
                        records.append(record)
                        done.add(key)
                        completed += 1
                        if mode == "reuse_off" and record.get("returncode") == 0:
                            refs[ref_key] = record
                        write_json(out_path, meta, records)

                        should_print = completed % args.progress_every == 0 or completed == total_cases
                        has_problem = record["returncode"] != 0 or bool(record["rowset_mismatches_vs_reuse_off"])
                        if should_print or has_problem:
                            elapsed = time.time() - started
                            print(
                                f"[{completed}/{total_cases}] db={db_label} q{template} batch={batch_size} "
                                f"mode={mode} rep={rep} rc={record['returncode']} "
                                f"rowset_mismatches={len(record['rowset_mismatches_vs_reuse_off'])} "
                                f"time_ms={record['execution_time_ms_total']} elapsed_s={elapsed:.1f}",
                                flush=True,
                            )

    write_json(out_path, meta, records)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
