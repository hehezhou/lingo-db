#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

STAMP="${STAMP:-$(date +%Y%m%d-%H%M%S)}"
OUT_DIR="${OUT_DIR:-experiments/state-reuse-results/${STAMP}-tpch-template-batch-nightly}"
OUT="${OUT:-${OUT_DIR}/results.json}"
LOG="${LOG:-${OUT_DIR}/run.log}"

SUBOP_BINARY="${SUBOP_BINARY:-build/lingodb-release/subop-state-reuse}"
DBS="${DBS:-resources/data/tpch-1,resources/data/tpch-10}"
QUERY_DIR="${QUERY_DIR:-queries}"
TEMPLATES="${TEMPLATES:-1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22}"
BATCH_SIZES="${BATCH_SIZES:-2,4,8,16,32,64,128}"
MODES="${MODES:-reuse_off,reuse_on_bloom_on,reuse_on_bloom_off,reuse_on_bloom_on_no_hiv_disjoint}"
REPETITIONS="${REPETITIONS:-5}"
TIMEOUT_S="${TIMEOUT_S:-0}"
RESUME="${RESUME:-1}"

mkdir -p "$OUT_DIR"

echo "tpch template batch nightly"
echo "  out: $OUT"
echo "  log: $LOG"
echo "  subop: $SUBOP_BINARY"
echo "  dbs: $DBS"
echo "  query_dir: $QUERY_DIR"
echo "  templates: $TEMPLATES"
echo "  batch_sizes: $BATCH_SIZES"
echo "  modes: $MODES"
echo "  repetitions: $REPETITIONS"
echo "  timeout_s: $TIMEOUT_S"

cmd=(
  python3 tools/scripts/tpch-template-batch-nightly.py
  --subop-binary "$SUBOP_BINARY"
  --dbs "$DBS"
  --query-dir "$QUERY_DIR"
  --out "$OUT"
  --templates "$TEMPLATES"
  --batch-sizes "$BATCH_SIZES"
  --modes "$MODES"
  --repetitions "$REPETITIONS"
  --timeout-s "$TIMEOUT_S"
)

if [[ "$RESUME" != "0" ]]; then
  cmd+=(--resume)
fi

"${cmd[@]}" 2>&1 | tee -a "$LOG"
