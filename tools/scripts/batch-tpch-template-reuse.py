#!/usr/bin/env python3
import argparse
import json
import math
import os
import re
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


TIMING_TOTAL_RE = re.compile(r"^//\s+timing:\s+optimization_ms=.*\s+execution_time_ms=([0-9.eE+\-]+)\s*$", re.M)

MODES = ["reuse_on_bloom_on", "reuse_on_bloom_off", "reuse_off"]


def parse_int_list(spec: str) -> List[int]:
    return [int(x.strip()) for x in spec.split(",") if x.strip()]


def run_cmd(argv: List[str], env: Dict[str, str], timeout_s: int) -> Dict[str, Any]:
    t0 = time.time()
    proc = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, timeout=timeout_s)
    wall_ms = (time.time() - t0) * 1000.0
    stdout = proc.stdout.decode("utf-8", errors="replace")
    stderr = proc.stderr.decode("utf-8", errors="replace")
    m = TIMING_TOTAL_RE.search(stdout)
    return {
        "argv": argv,
        "returncode": proc.returncode,
        "wall_ms": wall_ms,
        "execution_time_ms_total": float(m.group(1)) if m else None,
        "stderr_tail": "\n".join(stderr.splitlines()[-30:]) if proc.returncode != 0 else "",
    }


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


def record_key(record: Dict[str, Any]) -> tuple[int, int, str, int]:
    return (int(record["template"]), int(record["batch_size"]), str(record["mode"]), int(record["rep"]))


def build_summary(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[tuple[int, int, str], List[Dict[str, Any]]] = {}
    for r in records:
        grouped.setdefault((r["template"], r["batch_size"], r["mode"]), []).append(r)

    out: List[Dict[str, Any]] = []
    for (template, batch_size, mode), rows in sorted(grouped.items()):
        times = [r["execution_time_ms_total"] for r in rows if r.get("execution_time_ms_total") is not None]
        summary: Dict[str, Any] = {
            "template": template,
            "batch_size": batch_size,
            "mode": mode,
            "runs": len(rows),
            "success_runs": len(times),
            "total_time_geo_mean_ms": geo_mean(times),
        }
        if times:
            summary.update({
                "total_time_min_ms": min(times),
                "total_time_max_ms": max(times),
                "total_time_median_ms": statistics.median(times),
            })
        out.append(summary)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subop-binary", default="build/lingodb-release/subop-state-reuse")
    ap.add_argument("--db", default="resources/data/tpch-1")
    ap.add_argument("--query-dir", default="queries")
    ap.add_argument("--out", default="tmp_batch_tpch_template_reuse_20260617.json")
    ap.add_argument("--templates", default=",".join(str(i) for i in range(1, 23)))
    ap.add_argument("--batch-sizes", default="2,4,8,16,32,64,128")
    ap.add_argument("--repetitions", type=int, default=5)
    ap.add_argument("--timeout-s", type=int, default=2400)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    templates = parse_int_list(args.templates)
    batch_sizes = parse_int_list(args.batch_sizes)
    query_dir = Path(args.query_dir)
    out_path = Path(args.out)

    data = load_existing(out_path) if args.resume else {"records": []}
    records: List[Dict[str, Any]] = data.get("records", [])
    done = {record_key(r) for r in records}

    base_env = dict(os.environ)
    base_env.pop("LINGODB_SKIP_EXECUTE", None)
    base_env.pop("LINGODB_SUBOP_FORCE_SEQUENTIAL", None)

    total_cases = len(templates) * len(batch_sizes) * len(MODES) * args.repetitions
    completed = len(done)
    t0 = time.time()

    for template in templates:
        template_queries = [query_dir / f"q{template}_{i}.sql" for i in range(1, max(batch_sizes) + 1)]
        missing = [str(p) for p in template_queries if not p.exists()]
        if missing:
            raise SystemExit(f"missing query files for template {template}: {missing[:5]}")

        for batch_size in batch_sizes:
            sqls = [str(query_dir / f"q{template}_{i}.sql") for i in range(1, batch_size + 1)]
            for mode in MODES:
                for rep in range(args.repetitions):
                    key = (template, batch_size, mode, rep)
                    if key in done:
                        continue

                    argv = [args.subop_binary, args.db, *sqls]
                    env = dict(base_env)
                    if mode == "reuse_off":
                        argv.insert(1, "--no-reuse-rewrite")
                    elif mode == "reuse_on_bloom_off":
                        env["LINGODB_DISABLE_FILTER_PRED_BLOOM_ADAPTATION"] = "1"
                    elif mode == "reuse_on_bloom_on":
                        env.pop("LINGODB_DISABLE_FILTER_PRED_BLOOM_ADAPTATION", None)
                    else:
                        raise AssertionError(mode)

                    result = run_cmd(argv, env, args.timeout_s)
                    record = {
                        "template": template,
                        "batch_size": batch_size,
                        "mode": mode,
                        "rep": rep,
                        **result,
                    }
                    records.append(record)
                    done.add(key)
                    completed += 1

                    data = {
                        "meta": {
                            "db": args.db,
                            "query_dir": args.query_dir,
                            "subop_binary": args.subop_binary,
                            "templates": templates,
                            "batch_sizes": batch_sizes,
                            "modes": MODES,
                            "repetitions": args.repetitions,
                            "generated_at_unix": time.time(),
                        },
                        "records": records,
                        "summary": build_summary(records),
                    }
                    with out_path.open("w", encoding="utf-8") as f:
                        json.dump(data, f, indent=2, sort_keys=True)

                    if completed % 25 == 0 or completed == total_cases:
                        elapsed = time.time() - t0
                        print(
                            f"[{completed}/{total_cases}] template={template} batch={batch_size} "
                            f"mode={mode} rep={rep} elapsed_s={elapsed:.1f}",
                            flush=True,
                        )

    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
