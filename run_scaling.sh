#!/usr/bin/env bash
set -euo pipefail

# Runs scaling experiments for kroB200 comparing sequential dPSO vs naive MPI dPSO.
# Produces a small table with T_seq, T_naive, and speedup for p = 1,2,4,8 (configurable via PROCS_LIST).

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PROCS_LIST=${PROCS_LIST:-"1 2 4 8 14"}
INSTANCE="${1:-kroB200}"  # override via first arg, e.g., ./run_scaling.sh berlin52
SEQ_BIN="tsp_experiment"
MPI_BIN="tsp_experiment_mpi"

have_mpi() {
  command -v mpic++ >/dev/null 2>&1 && command -v mpirun >/dev/null 2>&1
}

parse_time() {
  local algo="$1" file="$2"
  awk -v inst="$INSTANCE" -v alg="$algo" '$1==inst && $2==alg {print $7}' "$file"
}

echo "[build] compiling sequential baseline..."
g++ -O2 -std=c++17 main.cpp src/*.cpp -o "$SEQ_BIN"

echo "[run] sequential baseline (p=1)..."
SEQ_OUT="$(mktemp)"
"$ROOT_DIR/$SEQ_BIN" --profile "$INSTANCE" | tee "$SEQ_OUT"
T_SEQ=$(parse_time "dPSO_seq" "$SEQ_OUT")
rm -f "$SEQ_OUT"

printf "\n%-6s %-12s %-12s %-12s\n" "p" "T_seq(ms)" "T_naive(ms)" "Speedup"
printf "%-6s %-12s %-12s %-12s\n" "1" "$T_SEQ" "$T_SEQ" "1.0"

if ! have_mpi; then
  echo "MPI toolchain not found (mpic++/mpirun). Skipping MPI scaling."
  exit 0
fi

echo "[build] compiling naive MPI baseline..."
mpic++ -O2 -std=c++17 -DUSE_MPI main.cpp src/*.cpp -o "$MPI_BIN"

for p in $PROCS_LIST; do
  if [ "$p" -eq 1 ]; then
    continue # already reported p=1 case
  fi
  echo "[run] naive MPI baseline p=$p..."
  MPI_OUT="$(mktemp)"
  mpirun -np "$p" "$ROOT_DIR/$MPI_BIN" --profile "$INSTANCE" | tee "$MPI_OUT"
  T_NAIVE=$(parse_time "dPSO_mpi" "$MPI_OUT")
  rm -f "$MPI_OUT"

  if [ -z "$T_NAIVE" ]; then
    echo "Warning: could not parse time for p=$p; skipping."
    continue
  fi
  SPEEDUP=$(python3 - <<EOF
t_seq=float("$T_SEQ")
t_naive=float("$T_NAIVE")
print(f"{t_seq/t_naive:.2f}")
EOF
)
  printf "%-6s %-12s %-12s %-12s\n" "$p" "$T_SEQ" "$T_NAIVE" "$SPEEDUP"
done
