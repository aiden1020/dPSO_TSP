#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

SEQUENTIAL_BIN="tsp_experiment"
MPI_BIN="tsp_experiment_mpi"
PROCS="${PROCS:-4}"

echo "Compiling sequential dPSO/NN baseline..."
g++ -O2 -std=c++17 main.cpp src/*.cpp -o "$SEQUENTIAL_BIN"

echo "Running sequential baselines..."
"$ROOT_DIR/$SEQUENTIAL_BIN"

if command -v mpic++ >/dev/null 2>&1 && command -v mpirun >/dev/null 2>&1; then
  echo "Compiling naive MPI dPSO baseline (-DUSE_MPI)..."
  mpic++ -O2 -std=c++17 -DUSE_MPI main.cpp src/*.cpp -o "$MPI_BIN"

  echo "Running naive MPI dPSO baseline with ${PROCS} ranks..."
  mpirun -np "$PROCS" "$ROOT_DIR/$MPI_BIN"
else
  echo "Skipping MPI baseline: mpic++/mpirun not found in PATH."
fi
