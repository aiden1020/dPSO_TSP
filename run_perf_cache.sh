#!/usr/bin/env bash
set -euo pipefail

# 使用 perf 觀察 dPSO_seq / dPSO_MPI / dPSO_MPI_v2 / dPSO_MPI_island 的 cache 狀況。
# 目標：比較不同 MPI 版本在同一組實驗條件下的 cache misses / miss rate，
#       搭配 run_island_hypothesis.sh 的 compute_ms，一起驗證「island 計算端較快」的假說。
#
# 用法：
#   ./run_perf_cache.sh [instance] [runs] [p] [simd_flag]
#     instance  : TSPLIB 名稱（預設 berlin52）
#     runs      : 每個演算法重複次數（預設 100）
#     p         : MPI ranks 數（預設 4）
#     simd_flag : 0/off 或 1/on（預設 0 = 不開 SIMD）
#
# 注意：
#   - 需要系統支援 perf（可能需 sudo 或調整 perf_event_paranoid）。
#   - perf 只做統計與輸出，結果請你自己對照 run_island_hypothesis.sh 的時間表。

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

INSTANCE="${1:-berlin52}"
RUNS="${2:-100}"
P="${3:-4}"
SIMD_ARG="${4:-0}"

INSTANCE_PATH="data/instances/${INSTANCE}.tsp"

SEQ_BIN="tsp_experiment_seq"
MPI_BIN="tsp_experiment_mpi"

have_mpi() {
  command -v mpic++ >/dev/null 2>&1 && command -v mpirun >/dev/null 2>&1
}

lookup_best() {
  awk -v inst="$INSTANCE" '$1==inst {print $3}' data/instances/solutions | head -n1
}

if [ ! -f "$INSTANCE_PATH" ]; then
  echo "[error] instance not found: $INSTANCE_PATH" >&2
  exit 1
fi

BEST_KNOWN="$(lookup_best)"
if [ -z "$BEST_KNOWN" ]; then
  echo "[error] best-known for $INSTANCE not found in data/instances/solutions" >&2
  exit 1
fi

ENABLE_SIMD=false
SIMD_FLAGS=""
if [[ "$SIMD_ARG" =~ ^(on|ON|1|true|True|simd|SIMD)$ ]]; then
  ENABLE_SIMD=true
  SIMD_FLAGS="-DUSE_TSP_SIMD -mavx2 -mfma"
  echo "[config] SIMD enabled (-DUSE_TSP_SIMD -mavx2 -mfma)"
else
  echo "[config] SIMD disabled"
fi

echo "[config] INSTANCE = $INSTANCE (best-known = $BEST_KNOWN)"
echo "[config] RUNS     = $RUNS"
echo "[config] p        = $P"

echo "[build] compiling sequential binary..."
g++ -O2 -std=c++17 $SIMD_FLAGS main.cpp src/*.cpp -o "$SEQ_BIN"

MPI_AVAILABLE=false
if have_mpi; then
  MPI_AVAILABLE=true
  echo "[build] compiling MPI binary (naive + v2 + island)..."
  mpic++ -O2 -std=c++17 -DUSE_MPI -DUSE_MPI_V2 $SIMD_FLAGS main.cpp src/*.cpp -o "$MPI_BIN"
else
  echo "[warn] MPI toolchain not found; MPI 部分將無法執行。"
fi

# 觀察的硬體事件（可依需要調整）
EVENTS="cycles,instructions,cache-references,cache-misses,L1-dcache-loads,L1-dcache-load-misses,LLC-loads,LLC-load-misses"

echo
echo "===== perf stat: dPSO_seq (p=1) ====="
perf stat -e "$EVENTS" \
  "$ROOT_DIR/$SEQ_BIN" dPSO_seq "$INSTANCE_PATH" "$BEST_KNOWN" "$RUNS"

if $MPI_AVAILABLE; then
  echo
  echo "===== perf stat: dPSO_MPI (p=$P) ====="
  perf stat -e "$EVENTS" \
    mpirun -bind-to core --map-by core -np "$P" \
      "$ROOT_DIR/$MPI_BIN" dPSO_MPI "$INSTANCE_PATH" "$BEST_KNOWN" "$RUNS"

  echo
  echo "===== perf stat: dPSO_MPI_v2 (p=$P) ====="
  perf stat -e "$EVENTS" \
    mpirun -bind-to core --map-by core -np "$P" \
      "$ROOT_DIR/$MPI_BIN" dPSO_MPI_v2 "$INSTANCE_PATH" "$BEST_KNOWN" "$RUNS"

  echo
  echo "===== perf stat: dPSO_MPI_island (p=$P) ====="
  perf stat -e "$EVENTS" \
    mpirun -bind-to core --map-by core -np "$P" \
      "$ROOT_DIR/$MPI_BIN" dPSO_MPI_island "$INSTANCE_PATH" "$BEST_KNOWN" "$RUNS"
fi

