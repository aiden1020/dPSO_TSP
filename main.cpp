#include <iostream>
#include <vector>
#include <string>
#include <iomanip>
#include <chrono>
#include <numeric>
#include <unordered_map>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cctype>
#ifdef USE_MPI
#include <mpi.h>
#include "src/dps_tsp_mpi.h"
#endif

#include "src/tsp_instance.h"
#include "src/utils.h"
#include "src/baseline_nn.h"
#include "src/dps_tsp.h"

// --- Experiment Configuration (defaults) ---
const std::vector<std::string> DEFAULT_INSTANCE_NAMES = {
    "berlin52",
    "kroB200"
    // Add more instance names here
};

const int NUM_RUNS = 20; // Number of runs for stochastic algorithms

// --- Helpers for parsing best-known solutions ---
std::string trim_copy(std::string s) {
    auto not_space = [](int ch) { return !std::isspace(ch); };
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), not_space));
    s.erase(std::find_if(s.rbegin(), s.rend(), not_space).base(), s.end());
    return s;
}

std::unordered_map<std::string, int> load_best_known(const std::string& filepath) {
    std::unordered_map<std::string, int> best_map;
    std::ifstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open best-known file: " + filepath);
    }

    std::string line;
    while (std::getline(file, line)) {
        // Remove inline comments and trim
        auto comment_pos = line.find('#');
        if (comment_pos != std::string::npos) {
            line = line.substr(0, comment_pos);
        }
        line = trim_copy(line);
        if (line.empty()) continue;

        auto colon_pos = line.find(':');
        if (colon_pos == std::string::npos) continue;

        std::string name = trim_copy(line.substr(0, colon_pos));
        std::string rest = trim_copy(line.substr(colon_pos + 1));

        std::stringstream ss(rest);
        long long value;
        if (!(ss >> value)) continue; // Skip lines that don't start with a number

        best_map[name] = static_cast<int>(value);
    }

    return best_map;
}

// --- Helper Struct for Results ---
struct RunStats {
    double best_cost = std::numeric_limits<double>::max();
    double worst_cost = 0;
    double mean_cost = 0;
    double mean_error = 0;
    double mean_time_ms = 0;
    
    void update(double cost, double time_ms, double best_known) {
        if (cost < best_cost) best_cost = cost;
        if (cost > worst_cost) worst_cost = cost;
        mean_cost += cost;
        mean_error += ((cost - best_known) / best_known) * 100.0;
        mean_time_ms += time_ms;
    }

    void finalize(int num_runs) {
        if (num_runs > 0) {
            mean_cost /= num_runs;
            mean_error /= num_runs;
            mean_time_ms /= num_runs;
        }
    }
};

struct TimingStats {
    double total_ms = 0;
    double init_ms = 0;
    double update_ms = 0;
    double update_move_ms = 0;
    double update_eval_ms = 0;
    double comm_ms = 0;

    void update(const DpsoTsp::Timing& t) {
        total_ms += t.total_ms;
        init_ms += t.init_ms;
        update_ms += t.update_ms;
        update_move_ms += t.update_move_ms;
        update_eval_ms += t.update_eval_ms;
        comm_ms += t.comm_ms;
    }

    void finalize(int num_runs) {
        if (num_runs > 0) {
            total_ms /= num_runs;
            init_ms /= num_runs;
            update_ms /= num_runs;
            update_move_ms /= num_runs;
            update_eval_ms /= num_runs;
            comm_ms /= num_runs;
        }
    }
};


void print_header() {
    std::cout << std::left
              << std::setw(12) << "Instance"
              << std::setw(12) << "Algorithm"
              << std::setw(10) << "Best Cost"
              << std::setw(12) << "Mean Cost"
              << std::setw(12) << "Best Err(%)"
              << std::setw(12) << "Mean Err(%)"
              << std::setw(15) << "Mean Time(ms)"
              << std::endl;
    std::cout << std::string(85, '-') << std::endl;
}

void print_results(const std::string& name, const std::string& algo, const RunStats& stats, double best_known) {
    double best_error = ((stats.best_cost - best_known) / best_known) * 100.0;
    std::cout << std::left << std::fixed << std::setprecision(2)
              << std::setw(12) << name
              << std::setw(12) << algo
              << std::setw(10) << stats.best_cost
              << std::setw(12) << stats.mean_cost
              << std::setw(12) << best_error
              << std::setw(12) << stats.mean_error
              << std::setw(15) << stats.mean_time_ms
              << std::endl;
}

void print_pso_timing(const std::string& name, const TimingStats& tstats) {
    double overhead_ms = std::max(0.0, tstats.total_ms - tstats.init_ms - tstats.update_ms - tstats.comm_ms);
    double total = std::max(tstats.total_ms, 1e-9); // guard divide-by-zero
    auto pct = [total](double x) { return (x / total) * 100.0; };

    std::cout << "  [dPSO timing] " << name
              << " | total " << std::fixed << std::setprecision(2) << tstats.total_ms << " ms"
              << " (init " << tstats.init_ms << " ms, updates " << tstats.update_ms
              << " ms [move " << tstats.update_move_ms << " ms, eval " << tstats.update_eval_ms << " ms]"
              << ", comm " << tstats.comm_ms << " ms, overhead " << overhead_ms << " ms)"
              << " | pct init " << pct(tstats.init_ms) << "%, updates " << pct(tstats.update_ms)
              << "% [move " << pct(tstats.update_move_ms) << "%, eval " << pct(tstats.update_eval_ms) << "%]"
              << ", comm " << pct(tstats.comm_ms) << "%, overhead " << pct(overhead_ms) << "%\n";
}


int main(int argc, char** argv) {
#ifdef USE_MPI
    MPI_Init(&argc, &argv);
    int world_rank = 0;
    int world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    bool use_mpi = world_size > 1;
    unsigned int seed = static_cast<unsigned int>(std::chrono::system_clock::now().time_since_epoch().count());
    seed += static_cast<unsigned int>(world_rank * 7919); // decorrelate between ranks
    Random::seed(seed);
#else
    Random::seed(std::chrono::system_clock::now().time_since_epoch().count());
#endif

    const std::string best_known_path = "data/instances/solutions";
    std::unordered_map<std::string, int> best_known_map;
    try {
        best_known_map = load_best_known(best_known_path);
    } catch (const std::exception& e) {
        std::cerr << "Error loading best-known solutions: " << e.what() << std::endl;
#ifdef USE_MPI
        MPI_Finalize();
#endif
        return 1;
    }

#ifdef USE_MPI
    if (world_rank == 0) {
        print_header();
    }
#else
    print_header();
#endif

    bool enable_profile = false;
    // Allow overriding instance list via command-line args; default if none provided.
    std::vector<std::string> instance_names;
    if (argc > 1) {
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--profile" || arg == "-p") {
                enable_profile = true;
            } else {
                instance_names.emplace_back(std::move(arg));
            }
        }
    } else {
        instance_names = DEFAULT_INSTANCE_NAMES;
    }
    if (instance_names.empty()) {
        instance_names = DEFAULT_INSTANCE_NAMES;
    }

    for (const auto& instance_name : instance_names) {
        std::string instance_file = "data/instances/" + instance_name + ".tsp";
        try {
            TSPInstance instance(instance_file);

            int best_known_len = -1;
            auto it = best_known_map.find(instance.get_name());
            if (it != best_known_map.end()) {
                best_known_len = it->second;
            } else {
                std::cerr << "Warning: Best known solution for " << instance.get_name()
                          << " not specified in " << best_known_path << std::endl;
            }

            if (best_known_len == -1) {
                continue;
            }

#ifdef USE_MPI
            if (!use_mpi || world_rank == 0) {
#endif
            // --- Run Nearest Neighbor (NN) Baseline ---
            RunStats nn_stats;
            for (int i = 0; i < instance.get_dimension(); ++i) { // Run NN from every possible start city
                auto start_time = std::chrono::high_resolution_clock::now();
                std::vector<int> nn_tour = solve_nn(instance, i);
                auto end_time = std::chrono::high_resolution_clock::now();
                double nn_cost = instance.calculate_tour_length(nn_tour);
                long long duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
                nn_stats.update(nn_cost, duration / 1000.0, best_known_len);
            }
            nn_stats.finalize(instance.get_dimension());
            print_results(instance.get_name(), "NN", nn_stats, best_known_len);
#ifdef USE_MPI
            }
#endif


#ifdef USE_MPI
            if (use_mpi) {
                // --- Run Naive Parallel MPI Discrete PSO (dPSO) Baseline ---
                DpsoTsp::Parameters pso_params;
                pso_params.swarm_size = 100;
                pso_params.max_iter = 1500;
                pso_params.inertia_weight = 0.7;
                pso_params.cognitive_weight = 1.4;
                pso_params.social_weight = 1.6;
                pso_params.max_velocity_len = std::max(10, instance.get_dimension() / 5); // scale velocity with problem size
                pso_params.mutation_prob = 0.05;
                pso_params.local_search_attempts = 20;

                RunStats pso_stats;
                TimingStats pso_timing;
                for (int i = 0; i < NUM_RUNS; ++i) {
                    double start_time = MPI_Wtime();
                    DpsoTspNaiveMpi solver(instance, pso_params, MPI_COMM_WORLD);
                    solver.solve();
                    double end_time = MPI_Wtime();

                    double duration_ms = (end_time - start_time) * 1000.0;
                    double gbest_cost = solver.get_gbest_cost();

                    if (enable_profile) {
                        // Aggregate timings across ranks
                        auto local_timing = solver.get_timing();
                        double accum_total = 0, accum_init = 0, accum_update = 0, accum_move = 0, accum_eval = 0, accum_comm = 0;
                        MPI_Reduce(&local_timing.total_ms, &accum_total, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
                        MPI_Reduce(&local_timing.init_ms, &accum_init, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
                        MPI_Reduce(&local_timing.update_ms, &accum_update, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
                        MPI_Reduce(&local_timing.update_move_ms, &accum_move, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
                        MPI_Reduce(&local_timing.update_eval_ms, &accum_eval, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
                        MPI_Reduce(&local_timing.comm_ms, &accum_comm, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

                        if (world_rank == 0) {
                            DpsoTsp::Timing avg_timing{
                                accum_total / world_size,
                                accum_init / world_size,
                                accum_update / world_size,
                                accum_move / world_size,
                                accum_eval / world_size,
                                accum_comm / world_size
                            };
                            pso_timing.update(avg_timing);
                        }
                    }
                    if (world_rank == 0) {
                        pso_stats.update(gbest_cost, duration_ms, best_known_len);
                    }
                }
                if (world_rank == 0) {
                    pso_stats.finalize(NUM_RUNS);
                    print_results(instance.get_name(), "dPSO_mpi", pso_stats, best_known_len);
                    if (enable_profile) {
                        pso_timing.finalize(NUM_RUNS);
                        print_pso_timing(instance.get_name(), pso_timing);
                    }
                }
            } else {
#endif
            // --- Run Discrete PSO (dPSO) Baseline ---
            DpsoTsp::Parameters pso_params;
            pso_params.swarm_size = 100;
            pso_params.max_iter = 1500;
            pso_params.inertia_weight = 0.7;
            pso_params.cognitive_weight = 1.4;
            pso_params.social_weight = 1.6;
            pso_params.max_velocity_len = std::max(10, instance.get_dimension() / 5); // scale velocity with problem size
            pso_params.mutation_prob = 0.05;
            pso_params.local_search_attempts = 20;

            RunStats pso_stats;
            TimingStats pso_timing;
            for (int i = 0; i < NUM_RUNS; ++i) {
                DpsoTsp solver(instance, pso_params);
                auto start_time = std::chrono::high_resolution_clock::now();
                solver.solve();
                auto end_time = std::chrono::high_resolution_clock::now();
                long long duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
                pso_stats.update(solver.get_gbest_cost(), duration / 1000.0, best_known_len);
                if (enable_profile) {
                    pso_timing.update(solver.get_timing());
                }
            }
            pso_stats.finalize(NUM_RUNS);
            print_results(instance.get_name(), "dPSO_seq", pso_stats, best_known_len);
            if (enable_profile) {
                pso_timing.finalize(NUM_RUNS);
                print_pso_timing(instance.get_name(), pso_timing);
            }
#ifdef USE_MPI
            }
#endif

        } catch (const std::exception& e) {
            std::cerr << "Error processing " << instance_file << ": " << e.what() << std::endl;
        }
    }

#ifdef USE_MPI
    MPI_Finalize();
#endif
    return 0;
}
