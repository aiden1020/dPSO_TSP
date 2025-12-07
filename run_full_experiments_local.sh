#!/usr/bin/env bash
set -euo pipefail

# Run the full experiment suite on multiple TSPLIB instances
# with a single, consistent configuration.
#
# Usage:
#   ./run_full_experiments.sh [runs] [simd]
#
#   runs : number of runs for stochastic algorithms (default: 20)
#   simd : 0/off or 1/on (default: 0 = SIMD disabled)
#
# Env overrides:
#   PROCS_LIST : "2 4 8 14" by default
#   INSTANCES  : "berlin52 kroB200 u2319" by default
#
# This script will:
#   1. Compile sequential and MPI binaries once (with/without SIMD).
#   2. For each instance:
#        - Run NN baseline (sequential)
#        - Run dPSO_seq baseline (sequential, RUNS 次)
#        - If MPI is available:
#            * For p in PROCS_LIST:
#                - Run dPSO_MPI
#                - Run dPSO_MPI_v2
#                - Run dPSO_MPI_island
#   3. Summarize all CSV lines into a single table:
#        Instance, p, Algorithm, Best/Mean cost, Err%, Time, Comm%, Comm_time

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

RUNS="${1:-20}"
SIMD_ARG="${2:-0}"

# Default instances & MPI proc counts
PROCS_LIST=${PROCS_LIST:-"2 4 8 14"}
INSTANCES=${INSTANCES:-"berlin52 kroB200 dsj1000"}

SEQ_BIN="tsp_experiment_seq"
MPI_BIN="tsp_experiment_mpi"

have_mpi() {
  command -v mpic++ >/dev/null 2>&1 && command -v mpirun >/dev/null 2>&1
}

lookup_best() {
  local inst="$1"
  awk -v inst="$inst" '$1==inst {print $3}' data/instances/solutions | head -n1
}

# SIMD 設定（整個實驗只決定一次）
ENABLE_SIMD=false
SIMD_FLAGS=""
if [[ "$SIMD_ARG" =~ ^(on|ON|1|true|True|simd|SIMD)$ ]]; then
  ENABLE_SIMD=true
  SIMD_FLAGS="-DUSE_TSP_SIMD -mavx2 -mfma"
  echo "[config] SIMD enabled (-DUSE_TSP_SIMD -mavx2 -mfma)"
else
  echo "[config] SIMD disabled"
fi

echo "[config] RUNS = $RUNS"
echo "[config] INSTANCES = $INSTANCES"
echo "[config] PROCS_LIST = $PROCS_LIST"

# 一次編譯，所有實驗共用
echo "[build] compiling sequential binary..."
g++ -O2 -std=c++17 $SIMD_FLAGS main.cpp src/*.cpp -o "$SEQ_BIN"

MPI_AVAILABLE=false
if have_mpi; then
  MPI_AVAILABLE=true
  echo "[build] compiling MPI binary (naive + v2 + island)..."
  mpic++ -O2 -std=c++17 -DUSE_MPI -DUSE_MPI_V2 $SIMD_FLAGS main.cpp src/*.cpp -o "$MPI_BIN"
else
  echo "[warn] MPI toolchain not found; MPI experiments will be skipped."
fi

OUT_FILE="$(mktemp)"

# ==== 逐個 instance 跑完整實驗 ====
for INSTANCE in $INSTANCES; do
  INSTANCE_PATH="data/instances/${INSTANCE}.tsp"

  if [ ! -f "$INSTANCE_PATH" ]; then
    echo "[warn] instance file not found: $INSTANCE_PATH, skipping."
    continue
  fi

  BEST_KNOWN="$(lookup_best "$INSTANCE")"
  if [ -z "$BEST_KNOWN" ]; then
    echo "[warn] best-known for $INSTANCE not found in data/instances/solutions, skipping."
    continue
  fi

  echo
  echo "==================== Instance: $INSTANCE (best-known = $BEST_KNOWN) ===================="

  # 1) NN baseline（通常內部已經從每個城市起點掃過，不需要多次跑）
  echo "[run] NN baseline (sequential) on $INSTANCE..."
  "$ROOT_DIR/$SEQ_BIN" NN "$INSTANCE_PATH" "$BEST_KNOWN" 1 | tee -a "$OUT_FILE"

  # 2) dPSO_seq baseline（RUNS 次）
  echo "[run] dPSO_seq baseline (sequential) on $INSTANCE, runs = $RUNS..."
  "$ROOT_DIR/$SEQ_BIN" dPSO_seq "$INSTANCE_PATH" "$BEST_KNOWN" "$RUNS" | tee -a "$OUT_FILE"

  # 3) MPI 版本（如果有 MPI）
  if $MPI_AVAILABLE; then
    for p in $PROCS_LIST; do
      echo "[run] dPSO_MPI         on $INSTANCE, p=$p, runs = $RUNS..."
      mpirun -bind-to core --map-by core -np "$p" \
        "$ROOT_DIR/$MPI_BIN" dPSO_MPI "$INSTANCE_PATH" "$BEST_KNOWN" "$RUNS" \
        | tee -a "$OUT_FILE"

      echo "[run] dPSO_MPI_v2      on $INSTANCE, p=$p, runs = $RUNS..."
      mpirun -bind-to core --map-by core -np "$p" \
        "$ROOT_DIR/$MPI_BIN" dPSO_MPI_v2 "$INSTANCE_PATH" "$BEST_KNOWN" "$RUNS" \
        | tee -a "$OUT_FILE"

      echo "[run] dPSO_MPI_island on $INSTANCE, p=$p, runs = $RUNS..."
      mpirun -bind-to core --map-by core -np "$p" \
        "$ROOT_DIR/$MPI_BIN" dPSO_MPI_island "$INSTANCE_PATH" "$BEST_KNOWN" "$RUNS" \
        | tee -a "$OUT_FILE"
    done
  fi
done

echo
printf "%-12s %-4s %-16s %-10s %-12s %-12s %-12s %-15s %-8s %-15s\n" \
  "Instance" "p" "Algorithm" "Best Cost" "Mean Cost" "Best Err(%)" "Mean Err(%)" "Mean Time(ms)" "Comm(%)" "Comm_time(ms)"
printf "%s\n" "----------------------------------------------------------------------------------------------------------------------------------"

# 這裡假設所有 C++ binary 會額外輸出 CSV：
# alg,inst,procs,best,mean,besterr,meanerr,meantime,comm
while IFS=',' read -r alg inst procs best mean besterr meanerr meantime comm; do
  # 跳過不是 CSV 的行（例如空行或沒有 alg）
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