#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./run_full_experiments.sh [runs] [simd]

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

RUNS="${1:-20}"
SIMD_ARG="${2:-0}"

# Default instances & MPI proc counts
PROCS_LIST=${PROCS_LIST:-"2 4 8 16 32"}
INSTANCES=${INSTANCES:-"berlin52 kroB200 u2319"}
NODES=${NODES:-4}  # MPI 實驗用的節點數

SEQ_BIN="tsp_experiment_seq"
MPI_BIN="tsp_experiment_mpi"

have_mpi() {
  command -v mpic++ >/dev/null 2>&1 && command -v run >/dev/null 2>&1
}

lookup_best() {
  local inst="$1"
  awk -v inst="$inst" '$1==inst {print $3}' data/instances/solutions | head -n1
}

# SIMD 設定
SIMD_FLAGS=""
if [[ "$SIMD_ARG" =~ ^(on|ON|1|true|True|simd|SIMD)$ ]]; then
  SIMD_FLAGS="-DUSE_TSP_SIMD -mavx2 -mfma"
  echo "[config] SIMD enabled (-DUSE_TSP_SIMD -mavx2 -mfma)"
else
  echo "[config] SIMD disabled"
fi

echo "[config] RUNS       = $RUNS"
echo "[config] INSTANCES  = $INSTANCES"
echo "[config] PROCS_LIST = $PROCS_LIST"
echo "[config] NODES      = $NODES"

# ==== 1. 編譯階段 (修正重點：統一使用 mpic++) ====

if have_mpi; then
  echo "[build] Detected MPI environment. Using mpic++ for ALL binaries to ensure consistency."

  # A. 編譯 Sequential Baseline
  # 使用 mpic++ 但不加 -DUSE_MPI。
  # 這確保了 seq 和 mpi 版本使用完全相同的 compiler wrapper 和 optimization flags。
  echo "[build] compiling sequential binary ($SEQ_BIN)..."
  mpic++ -O3 -std=c++17 $SIMD_FLAGS main.cpp src/*.cpp -o "$SEQ_BIN"

  # B. 編譯 MPI Parallel Versions
  # 加上 -DUSE_MPI 和 -DUSE_MPI_V2
  echo "[build] compiling MPI binary ($MPI_BIN)..."
  mpic++ -O3 -std=c++17 -DUSE_MPI -DUSE_MPI_V2 $SIMD_FLAGS main.cpp src/*.cpp -o "$MPI_BIN"

  MPI_AVAILABLE=true
else
  echo "[warn] MPI toolchain not found. Falling back to g++ for sequential only."
  echo "[build] compiling sequential binary ($SEQ_BIN)..."
  g++ -O3 -std=c++17 $SIMD_FLAGS main.cpp src/*.cpp -o "$SEQ_BIN"
  MPI_AVAILABLE=false
fi

OUT_FILE="$(mktemp)"

# ==== 2. 執行實驗階段 ====
for INSTANCE in $INSTANCES; do
  INSTANCE_PATH="data/instances/${INSTANCE}.tsp"

  if [ ! -f "$INSTANCE_PATH" ]; then
    echo "[warn] instance file not found: $INSTANCE_PATH, skipping."
    continue
  fi

  BEST_KNOWN="$(lookup_best "$INSTANCE")"
  if [ -z "$BEST_KNOWN" ]; then
    echo "[warn] best-known for $INSTANCE not found, skipping."
    continue
  fi

  echo
  echo "==================== Instance: $INSTANCE (best-known = $BEST_KNOWN) ===================="

  # ----------------------------------------------------------------
  # 修正重點：Sequential 執行
  # 使用 run --mpi=none -N 1 -n 1
  # 確保它被排程系統丟到一個 Compute Node 上執行，而不是在 Login Node
  # ----------------------------------------------------------------

  # 1) NN baseline (只跑 1 次，不需要多跑)
  echo "[run] NN baseline (sequential) on $INSTANCE..."
  if $MPI_AVAILABLE; then
      run --mpi=none -N 1 -n 1 -c 1 -- "$ROOT_DIR/$SEQ_BIN" NN "$INSTANCE_PATH" "$BEST_KNOWN" 1 | tee -a "$OUT_FILE"
  else
      "$ROOT_DIR/$SEQ_BIN" NN "$INSTANCE_PATH" "$BEST_KNOWN" 1 | tee -a "$OUT_FILE"
  fi

  # 2) dPSO_seq baseline (跑 RUNS 次)
  echo "[run] dPSO_seq baseline (sequential) on $INSTANCE, runs = $RUNS..."
  if $MPI_AVAILABLE; then
      run --mpi=none -N 1 -n 1 -c 1 -- "$ROOT_DIR/$SEQ_BIN" dPSO_seq "$INSTANCE_PATH" "$BEST_KNOWN" "$RUNS" | tee -a "$OUT_FILE"
  else
      "$ROOT_DIR/$SEQ_BIN" dPSO_seq "$INSTANCE_PATH" "$BEST_KNOWN" "$RUNS" | tee -a "$OUT_FILE"
  fi

  # 3) MPI 版本
  if $MPI_AVAILABLE; then
    for p in $PROCS_LIST; do
      # 這裡保持不變，使用 --mpi=pmix 和多節點配置
      echo "[run] dPSO_MPI         on $INSTANCE, NODES=$NODES, p=$p..."
      run --mpi=pmix -N "$NODES" -n "$p" \
        "$ROOT_DIR/$MPI_BIN" dPSO_MPI "$INSTANCE_PATH" "$BEST_KNOWN" "$RUNS" \
        | tee -a "$OUT_FILE"

      echo "[run] dPSO_MPI_v2      on $INSTANCE, NODES=$NODES, p=$p..."
      run --mpi=pmix -N "$NODES" -n "$p" \
        "$ROOT_DIR/$MPI_BIN" dPSO_MPI_v2 "$INSTANCE_PATH" "$BEST_KNOWN" "$RUNS" \
        | tee -a "$OUT_FILE"

      echo "[run] dPSO_MPI_island  on $INSTANCE, NODES=$NODES, p=$p..."
      run --mpi=pmix -N "$NODES" -n "$p" \
        "$ROOT_DIR/$MPI_BIN" dPSO_MPI_island "$INSTANCE_PATH" "$BEST_KNOWN" "$RUNS" \
        | tee -a "$OUT_FILE"
    done
  fi
done

# ==== 3. 輸出報表 ====
echo
printf "%-12s %-4s %-16s %-10s %-12s %-12s %-12s %-15s %-8s %-15s\n" \
  "Instance" "p" "Algorithm" "Best Cost" "Mean Cost" "Best Err(%)" "Mean Err(%)" "Mean Time(ms)" "Comm(%)" "Comm_time(ms)"
printf "%s\n" "----------------------------------------------------------------------------------------------------------------------------------"

while IFS=',' read -r alg inst procs best mean besterr meanerr meantime comm; do
  [ -z "$alg" ] && continue

  # 計算 Comm%
  pct="-"
  if [ -n "${meantime:-}" ] && [ -n "${comm:-}" ]; then
    pct=$(python3 - <<EOF
try:
    t=float("${meantime}")
    c=float("${comm}")
    print(f"{(c/t*100):.2f}" if t>0 else "-")
except Exception:
    print("-")
EOF
)
  fi

  printf "%-12s %-4s %-16s %-10s %-12s %-12s %-12s %-15s %-8s %-15s\n" \
    "$inst" "${procs:-1}" "$alg" "$best" "$mean" "$besterr" "$meanerr" "$meantime" "$pct" "$comm"
done < "$OUT_FILE"

rm -f "$OUT_FILE"