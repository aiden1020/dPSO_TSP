# dPSO-TSP C++ Experimental Platform

This document describes the goals, structure, build, and execution methods of the dPSO-TSP experimental platform. Content is updated to reflect the latest implementation.

## 1. Project Goals

This project aims to build a high-performance C++ experimental platform for researching, evaluating, and comparing the efficiency of **Discrete Particle Swarm Optimization (dPSO)** in solving the **Symmetric Traveling Salesman Problem (TSP)**, including both sequential and parallel (MPI) versions.

## 2. Project Structure

```
├── main.cpp                     # Main entry point, supports command-line arguments
├── run_full_experiments_local.sh # Automation script for running full experiments locally
├── plot.py                      # Python script for generating result plots
├── data/
│   └── instances/
│       ├── berlin52.tsp         # TSPLIB instance
│       └── solutions            # Known best solutions for some instances
├── results/                     # Stores results such as plots
└── src/
    ├── baseline_nn.h, .cpp      # Baseline: Nearest Neighbor algorithm
    ├── dps_tsp.h, .cpp          # Sequential dPSO algorithm
    ├── dps_tsp_mpi.h, .cpp      # Naive MPI dPSO (Master-Worker)
    ├── dps_tsp_mpi_v2.h, .cpp   # Naive MPI v2 (Lazy-sync)
    ├── dps_tsp_mpi_island.h, .cpp # Island Model MPI dPSO
    ├── dpso_params.h            # dPSO algorithm hyperparameters
    ├── tsp_instance.h, .cpp     # TSPLIB data reading and instance handling
    └── utils.h, .cpp            # Shared utilities
```

## 3. Dependencies

- **C++ Compiler**: Supports C++17 standard (e.g., `g++`).
- **MPI Toolchain**: `mpic++` and `mpirun` (for parallel versions).
- **Python 3**: `run_full_experiments_local.sh` for communication time calculation, `plot.py` for plotting.
- **Shell Tools**: `awk` (used by `run_full_experiments_local.sh` to read best solutions).

## 4. Building the Project

You can compile different versions of the executable as needed.

### A) Compile Sequential Version

This version runs the `NN` and `dPSO_seq` algorithms.

```bash
g++ -O2 -std=c++17 main.cpp src/*.cpp -o tsp_experiment_seq
```

### B) Compile MPI Version

This version runs all parallel dPSO algorithms.

```bash
# -DUSE_MPI -DUSE_MPI_V2 enables all MPI-related code
mpic++ -O2 -std=c++17 -DUSE_MPI -DUSE_MPI_V2 main.cpp src/*.cpp -o tsp_experiment_mpi
```

### C) (Optional) Enable SIMD Optimization

If your CPU supports AVX2, you can add SIMD flags for better performance.

```bash
# Add the following flags to g++ or mpic++
# -DUSE_TSP_SIMD -mavx2 -mfma
```

## 5. Running Experiments

### A) Manual Execution

The program accepts experiment settings via command-line arguments:
`./<binary> <algorithm> <instance_path> <best_known_cost> <num_runs>`

**Example:**
```bash
# Run sequential dPSO algorithm, 20 runs on berlin52
./tsp_experiment_seq dPSO_seq data/instances/berlin52.tsp 7542 20

# Run Island Model MPI algorithm with 4 cores
mpirun -np 4 ./tsp_experiment_mpi dPSO_MPI_island data/instances/berlin52.tsp 7542 20
```

### B) Automated Script (Recommended)

`run_full_experiments_local.sh` is the recommended way to run full benchmark tests. It automatically compiles and runs multiple instances and core combinations, and generates formatted result tables.

```bash
# Run with default settings (20 runs, SIMD off)
./run_full_experiments_local.sh

# Custom: 30 runs, SIMD on
./run_full_experiments_local.sh 30 1
```

## 6. Implemented Algorithms

You can specify the algorithm to run via command-line arguments:
- `NN`: Nearest Neighbor algorithm (greedy)
- `dPSO_seq`: Sequential dPSO
- `dPSO_MPI`: Naive Master-Worker MPI dPSO
- `dPSO_MPI_v2`: Optimized Master-Worker (Lazy-sync) MPI dPSO
- `dPSO_MPI_island`: Island Model MPI dPSO (asynchronous island model)

## 7. Output

### Output Format

The C++ program outputs results in CSV format to `stdout`:
`algorithm,instance,procs,best_cost,mean_cost,best_err,mean_err,mean_time_ms,mean_comm_ms`

## 8. Experimental Results

Below are results for `kroB200` and `u2319` TSP instances under different algorithms.

### Instance: kroB200

| Method               | Mean Err% | Best Err% | Total Time (ms) | Speedup vs seq |
| :------------------- | :-------- | :-------- | :-------------- | :------------- |
| Sequential (baseline)| 21.46     | 15.44     | 518.45          | 1.0×           |
| naive MPI            | 21.69     | 17.32     | 312.60          | 1.7×           |
| lazy-sync MPI        | 23.61     | 18.90     | 90.94           | 5.7×           |
| Island MPI (Ours)    | 9.73      | 7.56      | 63.61           | 8.1×           |

### Instance: u2319

| Method               | Mean Err% | Best Err% | Total Time (ms) | Speedup vs seq |
| :------------------- | :-------- | :-------- | :-------------- | :------------- |
| Serial (baseline)    | 18.53     | 17.30     | 4871.66         | 1.0×           |
| naive MPI            | 18.63     | 18.34     | 1579.15         | 3.1×           |
| lazy-sync MPI        | 18.70     | 18.29     | 690.43          | 7.1×           |
| Island MPI (Ours)    | 17.06     | 15.97     | 492.01          | 9.9×           |

