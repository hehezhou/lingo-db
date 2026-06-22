#!/usr/bin/env python3
import argparse
import json
import os
import re
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List


_RE_SCAN_CPU = re.compile(
    r"^//\s+scan_cpu_profile:\s+scan_cpu_ms=([0-9.eE+\-]+)\s+total_cpu_ms=([0-9.eE+\-]+)\s+percent=([0-9.eE+\-]+)\s+scan_calls=(\d+)\s*$",
    re.M,
)
_RE_EXEC_TIME = re.compile(r"^//\s+timing_execution_time_ms:\s+per_run=\[([^\]]*)\]\s+total=([0-9.eE+\-]+)\s*$", re.M)


def run_one(argv: List[str], env: Dict[str, str], timeout_s: int) -> Dict[str, Any]:
    t0 = time.time()
    proc = subprocess.run(
        argv,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        timeout=timeout_s,
    )
    wall_ms = (time.time() - t0) * 1000.0
    stdout = proc.stdout.decode("utf-8", errors="replace")
    stderr = proc.stderr.decode("utf-8", errors="replace")
    profiles = [
        {
            "scan_cpu_ms": float(m.group(1)),
            "total_cpu_ms": float(m.group(2)),
            "percent": float(m.group(3)),
            "scan_calls": int(m.group(4)),
        }
        for m in _RE_SCAN_CPU.finditer(stdout)
    ]
    exec_match = _RE_EXEC_TIME.search(stdout)
    return {
        "argv": argv,
        "returncode": proc.returncode,
        "wall_ms": wall_ms,
        "stdout": stdout,
        "stderr": stderr,
        "profiles": profiles,
        "execution_time_ms": float(exec_match.group(2)) if exec_match else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure TPCH Q1 scan CPU share in sequential and parallel subop-state-reuse runs.")
    parser.add_argument("--build-dir", default="build/lingodb-release")
    parser.add_argument("--db", default="resources/data/tpch-1")
    parser.add_argument("--sql", default="resources/sql/tpch/1.sql")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--timeout-s", type=int, default=300)
    args = parser.parse_args()

    stamp = time.strftime("%Y%m%d-%H%M%S")
    query_name = Path(args.sql).stem
    out_dir = Path(args.out_dir or f"experiments/state-reuse-results/{stamp}-tpch1-q{query_name}-scan-cpu-profile")
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    exe = str(Path(args.build_dir) / "subop-state-reuse")
    base_argv = [exe, args.db, args.sql]
    modes = [
        ("sequential", {"LINGODB_SUBOP_FORCE_SEQUENTIAL": "1"}),
        ("parallel", {}),
    ]

    results: Dict[str, Any] = {
        "build_dir": args.build_dir,
        "db": args.db,
        "sql": args.sql,
        "reps": args.reps,
        "modes": {},
    }

    for mode, extra_env in modes:
        mode_runs = []
        for rep in range(args.reps):
            env = os.environ.copy()
            env["LINGODB_SCAN_CPU_PROFILE"] = "1"
            env.update(extra_env)
            run = run_one(base_argv, env, args.timeout_s)
            (raw_dir / f"{mode}-rep{rep}.stdout.txt").write_text(run["stdout"])
            (raw_dir / f"{mode}-rep{rep}.stderr.txt").write_text(run["stderr"])
            stored = dict(run)
            stored["stdout"] = str(raw_dir / f"{mode}-rep{rep}.stdout.txt")
            stored["stderr"] = str(raw_dir / f"{mode}-rep{rep}.stderr.txt")
            mode_runs.append(stored)
            if run["returncode"] != 0:
                results["modes"][mode] = mode_runs
                (out_dir / "results.json").write_text(json.dumps(results, indent=2))
                return run["returncode"]
        results["modes"][mode] = mode_runs

    (out_dir / "results.json").write_text(json.dumps(results, indent=2))

    summary_lines = ["mode,scan_cpu_ms_avg,total_cpu_ms_avg,percent_avg,scan_calls_avg,execution_time_ms_avg"]
    for mode, runs in results["modes"].items():
        profiles = [run["profiles"][0] for run in runs if run["profiles"]]
        exec_times = [run["execution_time_ms"] for run in runs if run["execution_time_ms"] is not None]
        summary_lines.append(
            f"{mode},"
            f"{statistics.mean(p['scan_cpu_ms'] for p in profiles) if profiles else ''},"
            f"{statistics.mean(p['total_cpu_ms'] for p in profiles) if profiles else ''},"
            f"{statistics.mean(p['percent'] for p in profiles) if profiles else ''},"
            f"{statistics.mean(p['scan_calls'] for p in profiles) if profiles else ''},"
            f"{statistics.mean(exec_times) if exec_times else ''}"
        )
    (out_dir / "summary.csv").write_text("\n".join(summary_lines) + "\n")

    print(f"wrote {out_dir / 'results.json'}")
    print(f"wrote {out_dir / 'summary.csv'}")
    print("mode,rep,scan_cpu_ms,total_cpu_ms,percent,scan_calls,execution_time_ms")
    for mode, runs in results["modes"].items():
        for rep, run in enumerate(runs):
            profile = run["profiles"][0] if run["profiles"] else {}
            print(
                f"{mode},{rep},{profile.get('scan_cpu_ms','')},{profile.get('total_cpu_ms','')},"
                f"{profile.get('percent','')},{profile.get('scan_calls','')},{run.get('execution_time_ms','')}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
