#ifndef DPSO_TSP_H
#define DPSO_TSP_H

#include "tsp_instance.h"
#include "utils.h"
#include <vector>

// A Swap Operation, which is the basic component of a particle's velocity
struct SwapOp {
    int city_idx1;
    int city_idx2;
};

// Represents a particle in the Discrete PSO algorithm
struct Particle {
    std::vector<int> position;      // Current tour (a permutation)
    std::vector<SwapOp> velocity;   // A sequence of swap operations
    
    std::vector<int> pbest_position; // Personal best tour found so far
    double pbest_cost;               // Cost of the personal best tour
    
    double cost;                     // Cost of the current tour
};

// The main class for the Discrete PSO solver for TSP
class DpsoTsp {
public:
    struct Parameters {
        int swarm_size = 50;
        int max_iter = 1000;
        double inertia_weight = 0.7; // w
        double cognitive_weight = 1.5; // c1
        double social_weight = 1.5; // c2
        int max_velocity_len = 10; // Max number of swaps in a velocity vector
        double mutation_prob = 0.05; // Probability of a particle mutating
        int local_search_attempts = 0; // How many 2-opt attempts after each move
    };

    struct Timing {
        double total_ms = 0.0;
        double init_ms = 0.0;
        double update_ms = 0.0;
        double update_move_ms = 0.0; // velocity/position update + mutation
        double update_eval_ms = 0.0; // tour length + local search
        double comm_ms = 0.0; // kept for parity with MPI timing; always 0 in seq
    };

    DpsoTsp(const TSPInstance& instance, const Parameters& params);

    // Run the optimization
    void solve();

    // Get the best tour found by the swarm
    const std::vector<int>& get_gbest_position() const { return gbest_position; }
    
    // Get the cost of the best tour
    double get_gbest_cost() const { return gbest_cost; }

    // Get the convergence curve (best cost at each iteration)
    const std::vector<double>& get_convergence_curve() const { return convergence_curve; }

    // Timing breakdown from the last run
    Timing get_timing() const { return last_timing; }

private:
    const TSPInstance& instance;
    Parameters params;
    
    std::vector<Particle> swarm;
    std::vector<int> gbest_position;
    double gbest_cost;

    std::vector<double> convergence_curve;
    Timing last_timing;

    void initialize_swarm();
    void update_particle(Particle& p);

    // Core dPSO functions
    std::vector<SwapOp> calculate_diff(const std::vector<int>& from, const std::vector<int>& to);
    void apply_velocity(std::vector<int>& position, const std::vector<SwapOp>& velocity);
};

#endif // DPSO_TSP_H
