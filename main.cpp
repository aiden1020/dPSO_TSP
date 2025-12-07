#include <iostream>
#include <vector>
#include <string>
#include <iomanip>
#include <chrono>
#include <numeric>
#include <stdexcept> // For std::runtime_error, std::invalid_argument

#ifdef USE_MPI
#include <mpi.h>
#include "src/dps_tsp_mpi.h"
#include "src/dps_tsp_mpi_v2.h" // Include the v2 implementation
#include "src/dps_tsp_mpi_island.h" // Island model
#endif

#include "src/tsp_instance.h"
#include "src/utils.h"
#include "src/baseline_nn.h"
#include "src/dps_tsp.h"
#include "src/dpso_params.h"

// --- Helper Struct for Results ---
struct RunStats {
    double best_cost = std::numeric_limits<double>::max();
    double mean_cost = 0;
    double mean_time_ms = 0;
    double mean_comm_ms = 0; // <--- 新增此行
    int num_runs = 0;

    void update(double cost, double time_ms, double comm_ms = 0.0) {
        if (cost < best_cost) best_cost = cost;
        mean_cost += cost;
        mean_time_ms += time_ms;
        mean_comm_ms += comm_ms; // <--- 新增此行
        num_runs++;
    }

    void finalize() {
        if (num_runs > 0) {
            mean_cost /= num_runs;
            mean_time_ms /= num_runs;
            mean_comm_ms /= num_runs; // <--- 新增此行
        }
    }
};


// --- Main experiment function ---
void run_experiment(const std::string& algorithm, const std::string& instance_path, int best_known_cost, int num_runs) {
    int world_rank = 0;
    int world_size = 1;
#ifdef USE_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
#endif

    TSPInstance instance(instance_path);
    RunStats stats;

    // Set up PSO parameters
    DpsoTsp::Parameters pso_params;
    pso_params.swarm_size = DPSO_SWARM_SIZE;
    pso_params.max_iter = DPSO_MAX_ITER;
    pso_params.inertia_weight = DPSO_INERTIA_WEIGHT;
    pso_params.cognitive_weight = DPSO_COGNITIVE_WEIGHT;
    pso_params.social_weight = DPSO_SOCIAL_WEIGHT;
    pso_params.max_velocity_len = std::max(DPSO_MIN_VELOCITY_LEN, instance.get_dimension() / DPSO_VELOCITY_SCALE);
    pso_params.mutation_prob = DPSO_MUTATION_PROB;
    pso_params.local_search_attempts = DPSO_LOCAL_SEARCH_ATTEMPTS;

    auto now_ms = []() {
#ifdef USE_MPI
        return MPI_Wtime() * 1000.0;
#else
        using clock = std::chrono::high_resolution_clock;
        return std::chrono::duration<double, std::milli>(clock::now().time_since_epoch()).count();
#endif
    };

    for (int i = 0; i < num_runs; ++i) {
        double start_time, end_time, cost;

        if (algorithm == "NN") {
            // In NN's case, we run from all start cities and average, so we do it once.
            if (i == 0) {
                 double total_cost = 0;
                 double total_time = 0;
                 double best_nn_cost = std::numeric_limits<double>::max();
                 for(int city = 0; city < instance.get_dimension(); ++city) {
                    start_time = now_ms();
                    auto tour = solve_nn(instance, city);
                    end_time = now_ms();
                    double current_cost = instance.calculate_tour_length(tour);
                    if(current_cost < best_nn_cost) best_nn_cost = current_cost;
                    total_cost += current_cost;
                    total_time += (end_time - start_time);
                 }
                 stats.update(best_nn_cost, total_time / instance.get_dimension(), 0.0);
                 stats.mean_cost = total_cost / instance.get_dimension(); // Override mean cost
            }
            continue; // Finish after one aggregate run
        }
        else if (algorithm == "dPSO_seq") {
            start_time = now_ms();
            DpsoTsp solver(instance, pso_params);
            solver.solve();
            end_time = now_ms();
            cost = solver.get_gbest_cost();
            stats.update(cost, (end_time - start_time), 0.0);
        }
#ifdef USE_MPI
        else if (algorithm == "dPSO_MPI") {
            start_time = now_ms();
            DpsoTspNaiveMpi solver(instance, pso_params, MPI_COMM_WORLD);
            solver.solve();
            end_time = now_ms();
            double local_comm = solver.get_comm_time();
            double accum_comm = 0.0;
            MPI_Reduce(&local_comm, &accum_comm, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
            if (world_rank == 0) {
                cost = solver.get_gbest_cost();
                double avg_comm = accum_comm / world_size;
                stats.update(cost, (end_time - start_time), avg_comm);
            }
        }
        else if (algorithm == "dPSO_MPI_v2") {
            start_time = now_ms();
            DpsoTspNaiveMpi_v2 solver(instance, pso_params, MPI_COMM_WORLD);
            solver.solve();
            end_time = now_ms();
            double local_comm = solver.get_comm_time();
            double accum_comm = 0.0;
            MPI_Reduce(&local_comm, &accum_comm, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
            if (world_rank == 0) {
                cost = solver.get_gbest_cost();
                double avg_comm = accum_comm / world_size;
                stats.update(cost, (end_time - start_time), avg_comm);
            }
        }
        else if (algorithm == "dPSO_MPI_island") {
            start_time = now_ms();
            DpsoTspIslandMpi solver(instance, pso_params, MPI_COMM_WORLD);
            solver.solve();
            end_time = now_ms();
            double local_comm = solver.get_comm_time();
            double accum_comm = 0.0;
            MPI_Reduce(&local_comm, &accum_comm, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
            if (world_rank == 0) {
                cost = solver.get_gbest_cost();
                double avg_comm = accum_comm / world_size;
                stats.update(cost, (end_time - start_time), avg_comm);
            }
        }
#endif
        else {
            if (world_rank == 0) {
                throw std::invalid_argument("Unknown algorithm: " + algorithm);
            }
        }
    }

    if (world_rank == 0) {
        stats.finalize();
        double best_err = ((stats.best_cost - best_known_cost) / best_known_cost) * 100.0;
        double mean_err = ((stats.mean_cost - best_known_cost) / best_known_cost) * 100.0;
        
        // CSV-style output for scripts
        std::cout << algorithm << ","
                  << instance.get_name() << ","
                  << world_size << ","
                  << std::fixed << std::setprecision(2) << stats.best_cost << ","
                  << stats.mean_cost << ","
                  << best_err << ","
                  << mean_err << ","
                  << stats.mean_time_ms << ","      // 注意這裡加了逗號
                  << stats.mean_comm_ms << std::endl; // <--- 新增此行
    }
}


int main(int argc, char** argv) {
    int world_rank = 0;
#ifdef USE_MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    unsigned int seed = static_cast<unsigned int>(std::chrono::system_clock::now().time_since_epoch().count());
    seed += static_cast<unsigned int>(world_rank * 7919);
    Random::seed(seed);
#else
    Random::seed(std::chrono::system_clock::now().time_since_epoch().count());
#endif

    if (argc < 5) {
        if (world_rank == 0) {
            std::cerr << "Usage: " << argv[0] << " <algorithm> <instance_path> <best_known_cost> <num_runs>" << std::endl;
            std::cerr << "Supported algorithms: NN, dPSO_seq, dPSO_MPI, dPSO_MPI_v2" << std::endl;
        }
#ifdef USE_MPI
        MPI_Finalize();
#endif
        return 1;
    }

    try {
        std::string algorithm = argv[1];
        std::string instance_path = argv[2];
        int best_known_cost = std::stoi(argv[3]);
        int num_runs = std::stoi(argv[4]);
        
        run_experiment(algorithm, instance_path, best_known_cost, num_runs);

    } catch (const std::exception& e) {
        if (world_rank == 0) {
            std::cerr << "Error: " << e.what() << std::endl;
        }
#ifdef USE_MPI
        MPI_Finalize();
#endif
        return 1;
    }

#ifdef USE_MPI
    MPI_Finalize();
#endif
    return 0;
}
