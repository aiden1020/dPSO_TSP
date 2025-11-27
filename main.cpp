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

#include "src/tsp_instance.h"
#include "src/utils.h"
#include "src/baseline_nn.h"
#include "src/dps_tsp.h"

// --- Experiment Configuration ---
const std::vector<std::string> INSTANCE_NAMES = {
    "berlin52",
    "kroB200"
    // "a280",
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


int main() {
    Random::seed(std::chrono::system_clock::now().time_since_epoch().count());

    const std::string best_known_path = "data/instances/solutions";
    std::unordered_map<std::string, int> best_known_map;
    try {
        best_known_map = load_best_known(best_known_path);
    } catch (const std::exception& e) {
        std::cerr << "Error loading best-known solutions: " << e.what() << std::endl;
        return 1;
    }

    print_header();

    for (const auto& instance_name : INSTANCE_NAMES) {
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
            for (int i = 0; i < NUM_RUNS; ++i) {
                DpsoTsp solver(instance, pso_params);
                auto start_time = std::chrono::high_resolution_clock::now();
                solver.solve();
                auto end_time = std::chrono::high_resolution_clock::now();
                long long duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
                pso_stats.update(solver.get_gbest_cost(), duration / 1000.0, best_known_len);
            }
            pso_stats.finalize(NUM_RUNS);
            print_results(instance.get_name(), "dPSO_seq", pso_stats, best_known_len);

        } catch (const std::exception& e) {
            std::cerr << "Error processing " << instance_file << ": " << e.what() << std::endl;
        }
    }

    return 0;
}
