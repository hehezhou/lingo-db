#!/usr/bin/env python3
import argparse
import collections
import json
import os
import selectors
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


MODES = ["reuse_off", "reuse_on_bloom_on", "reuse_on_bloom_off", "reuse_on_bloom_on_no_hiv_disjoint"]


def parse_int_list(spec: str) -> List[int]:
    return [int(x.strip()) for x in spec.split(",") if x.strip()]


def parse_str_list(spec: str) -> List[str]:
    return [x.strip() for x in spec.split(",") if x.strip()]


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


def make_db_label(db_path: str) -> str:
    return Path(db_path).name


def build_env() -> Dict[str, str]:
    env = dict(os.environ)
    env.pop("LINGODB_SKIP_EXECUTE", None)
    return env


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
        reuse_union_counts = [
            sum(r["reuse_targets_union_no_table"])
            for r in ok_rows
            if isinstance(r.get("reuse_targets_union_no_table"), list)
        ]
        reuse_build_step_counts = [
            sum(r["reuse_targets_build_step_no_table"])
            for r in ok_rows
            if isinstance(r.get("reuse_targets_build_step_no_table"), list)
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
                "reuse_count_union_no_table_median_sum":
                    statistics.median(reuse_union_counts) if reuse_union_counts else None,
                "reuse_count_build_step_no_table_median_sum":
                    statistics.median(reuse_build_step_counts) if reuse_build_step_counts else None,
            }
        )
    return out


def annotate_mismatches(records: List[Dict[str, Any]]) -> None:
    refs: Dict[Tuple[str, int, int, int], Dict[str, Any]] = {}
    for r in records:
        r.setdefault("exact_mismatches_vs_reuse_off", [])
        r.setdefault("rowset_mismatches_vs_reuse_off", [])
        r.setdefault("order_only_mismatches_vs_reuse_off", [])
        r.setdefault("reference_missing", False)
        if r.get("mode") == "reuse_off" and r.get("returncode") == 0:
            refs[(r["db_label"], r["template"], r["batch_size"], r["rep"])] = r
    for r in records:
        if r.get("mode") == "reuse_off":
            continue
        ref = refs.get((r["db_label"], r["template"], r["batch_size"], r["rep"]))
        if ref is None:
            r["reference_missing"] = True
            continue
        rowset = r.get("result_rowset_block_hashes", [])
        ref_rowset = ref.get("result_rowset_block_hashes", [])
        r["rowset_mismatches_vs_reuse_off"] = mismatch_indices(rowset, ref_rowset)
        # The in-process batch driver records rowset hashes only; exact output order is intentionally ignored.
        r["exact_mismatches_vs_reuse_off"] = []
        r["order_only_mismatches_vs_reuse_off"] = []


def write_json(path: Path, meta: Dict[str, Any], records: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    annotate_mismatches(records)
    data = {
        "meta": {**meta, "updated_at_unix": time.time()},
        "records": records,
        "summary": build_summary(records),
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
    tmp.replace(path)


def run_batch_driver_for_mode(
    subop_binary: str,
    db: str,
    query_dir: Path,
    mode: str,
    templates: List[int],
    batch_sizes: List[int],
    repetitions: int,
    out_path: Path,
    timeout_s: int,
) -> List[Dict[str, Any]]:
    argv = [
        subop_binary,
        "--batch-nightly",
        db,
        "--mode",
        mode,
        "--query-dir",
        str(query_dir),
        "--out",
        str(out_path),
        "--templates",
        ",".join(str(x) for x in templates),
        "--batch-sizes",
        ",".join(str(x) for x in batch_sizes),
        "--repetitions",
        str(repetitions),
    ]
    env = build_env()
    timeout = None if timeout_s <= 0 else timeout_s
    proc = subprocess.Popen(
        argv,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    assert proc.stdout is not None
    sel = selectors.DefaultSelector()
    sel.register(proc.stdout, selectors.EVENT_READ)
    tail: collections.deque[str] = collections.deque(maxlen=80)
    start = time.monotonic()
    while True:
        if timeout is not None and time.monotonic() - start > timeout:
            proc.kill()
            raise SystemExit(f"batch driver timed out for db={db} mode={mode} after {timeout_s}s")
        for key, _ in sel.select(timeout=0.5):
            line = key.fileobj.readline()
            if not line:
                continue
            print(line, end="", flush=True)
            tail.append(line.rstrip("\n"))
        if proc.poll() is not None:
            rest = proc.stdout.read()
            if rest:
                for line in rest.splitlines(True):
                    print(line, end="", flush=True)
                    tail.append(line.rstrip("\n"))
            break
    if proc.returncode != 0:
        if tail:
            print("\n".join(tail), flush=True)
        raise SystemExit(f"batch driver failed for db={db} mode={mode} rc={proc.returncode}")
    with out_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("records", [])


def run_batch_driver(args: argparse.Namespace) -> None:
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
    max_batch = max(batch_sizes)
    for template in templates:
        missing = [str(query_dir / f"q{template}_{i}.sql") for i in range(1, max_batch + 1)
                   if not (query_dir / f"q{template}_{i}.sql").exists()]
        if missing:
            raise SystemExit(f"missing query files for template {template}: {missing[:5]}")

    meta = {
        "driver": "subop-state-reuse --batch-nightly",
        "dbs": dbs,
        "query_dir": args.query_dir,
        "subop_binary": args.subop_binary,
        "templates": templates,
        "batch_sizes": batch_sizes,
        "modes": modes,
        "repetitions": args.repetitions,
        "timeout_s": args.timeout_s,
        "created_at_unix": time.time(),
    }

    records: List[Dict[str, Any]] = []
    partial_dir = out_path.parent / "by-db-mode"
    partial_dir.mkdir(parents=True, exist_ok=True)
    for db in dbs:
        db_label = make_db_label(db)
        ordered_modes = ["reuse_off"] + [m for m in modes if m != "reuse_off"]
        for mode in ordered_modes:
            partial = partial_dir / f"{db_label}-{mode}.json"
            if args.resume and partial.exists():
                with partial.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                mode_records = data.get("records", [])
                print(f"reuse existing {partial} ({len(mode_records)} records)", flush=True)
            else:
                print(f"run batch driver db={db_label} mode={mode}", flush=True)
                mode_records = run_batch_driver_for_mode(
                    args.subop_binary, db, query_dir, mode, templates, batch_sizes,
                    args.repetitions, partial, args.timeout_s)
            records.extend(mode_records)
            write_json(out_path, meta, records)

    write_json(out_path, meta, records)
    print(f"wrote {out_path}", flush=True)


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
    args = ap.parse_args()

    run_batch_driver(args)


if __name__ == "__main__":
    main()
