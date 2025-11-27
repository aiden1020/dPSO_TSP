#include "dps_tsp.h"
#include "baseline_nn.h"
#include <iostream>
#include <algorithm>
#include <limits>

namespace {
// Lightweight stochastic 2-opt improvement. Limited attempts keep runtime modest.
double two_opt_local_search(std::vector<int>& tour, const TSPInstance& instance, double current_cost, int attempts) {
    if (attempts <= 0) return current_cost;
    double best_cost = current_cost;

    for (int k = 0; k < attempts; ++k) {
        int i = Random::get_int(0, static_cast<int>(tour.size()) - 2);
        int j = Random::get_int(i + 1, static_cast<int>(tour.size()) - 1);

        reverse_segment(tour, i, j);
        double new_cost = instance.calculate_tour_length(tour);

        if (new_cost < best_cost) {
            best_cost = new_cost;
        } else {
            // Revert if no improvement
            reverse_segment(tour, i, j);
        }
    }

    return best_cost;
}
} // namespace

DpsoTsp::DpsoTsp(const TSPInstance& instance, const Parameters& params)
    : instance(instance), params(params), gbest_cost(std::numeric_limits<double>::max()) {}

void DpsoTsp::solve() {
    initialize_swarm();
    convergence_curve.clear();
    convergence_curve.reserve(params.max_iter);

    for (int iter = 0; iter < params.max_iter; ++iter) {
        for (Particle& p : swarm) {
            update_particle(p);
        }
        convergence_curve.push_back(gbest_cost);
        // Optional: Add logging for each iteration
        // std::cout << "Iter " << iter << ": Best Cost = " << gbest_cost << std::endl;
    }
}

void DpsoTsp::initialize_swarm() {
    swarm.clear();
    swarm.resize(params.swarm_size);
    int n = instance.get_dimension();

    // Seed swarm with one NN tour to give the swarm a strong starting point.
    if (!swarm.empty()) {
        Particle& p0 = swarm[0];
        p0.position = solve_nn(instance, 0);
        p0.cost = instance.calculate_tour_length(p0.position);
        p0.pbest_position = p0.position;
        p0.pbest_cost = p0.cost;
        p0.velocity.clear();

        gbest_cost = p0.cost;
        gbest_position = p0.position;
    }

    for (int i = 0; i < params.swarm_size; ++i) {
        Particle& p = swarm[i];
        if (i == 0 && gbest_cost < std::numeric_limits<double>::max()) {
            // Already initialized with NN; optionally apply a small shuffle for diversity
            p.velocity.clear();
        } else {
            p.position = generate_random_tour(n);
            p.cost = instance.calculate_tour_length(p.position);
            
            p.pbest_position = p.position;
            p.pbest_cost = p.cost;
            
            // Initialize velocity as an empty set of swaps
            p.velocity.clear();

            if (p.cost < gbest_cost) {
                gbest_cost = p.cost;
                gbest_position = p.position;
            }
        }
    }
}

void DpsoTsp::update_particle(Particle& p) {
    std::vector<SwapOp> new_velocity;

    // 1. Inertia component
    int inertia_size = static_cast<int>(p.velocity.size() * params.inertia_weight);
    if (inertia_size > 0) {
        new_velocity.insert(new_velocity.end(), p.velocity.begin(), p.velocity.begin() + inertia_size);
    }

    // 2. Cognitive component
    std::vector<SwapOp> diff_pbest = calculate_diff(p.position, p.pbest_position);
    int pbest_swaps_count = static_cast<int>(diff_pbest.size() * params.cognitive_weight * Random::get_double());
    if (pbest_swaps_count > 0 && pbest_swaps_count <= diff_pbest.size()) {
       new_velocity.insert(new_velocity.end(), diff_pbest.begin(), diff_pbest.begin() + pbest_swaps_count);
    }

    // 3. Social component
    std::vector<SwapOp> diff_gbest = calculate_diff(p.position, gbest_position);
    int gbest_swaps_count = static_cast<int>(diff_gbest.size() * params.social_weight * Random::get_double());
     if (gbest_swaps_count > 0 && gbest_swaps_count <= diff_gbest.size()) {
        new_velocity.insert(new_velocity.end(), diff_gbest.begin(), diff_gbest.begin() + gbest_swaps_count);
    }
    
    // 4. Velocity Clamping
    if (new_velocity.size() > params.max_velocity_len) {
        new_velocity.resize(params.max_velocity_len);
    }
    p.velocity = new_velocity;

    // 5. Apply new velocity to update position
    apply_velocity(p.position, p.velocity);

    // 6. Mutation
    if (Random::get_double() < params.mutation_prob) {
        int idx1 = Random::get_int(0, p.position.size() - 1);
        int idx2 = Random::get_int(0, p.position.size() - 1);
        swap_cities(p.position, idx1, idx2);
    }

    // 7. Local search refinement (stochastic 2-opt) and evaluation
    p.cost = instance.calculate_tour_length(p.position);
    p.cost = two_opt_local_search(p.position, instance, p.cost, params.local_search_attempts);

    if (p.cost < p.pbest_cost) {
        p.pbest_cost = p.cost;
        p.pbest_position = p.position;

        if (p.cost < gbest_cost) {
            gbest_cost = p.cost;
            gbest_position = p.position;
        }
    }
}

// Simple implementation to find swaps that transform 'from' tour into 'to' tour
std::vector<SwapOp> DpsoTsp::calculate_diff(const std::vector<int>& from, const std::vector<int>& to) {
    std::vector<SwapOp> diff;
    if (from == to) return diff;

    std::vector<int> temp_pos = from;
    int n = from.size();
    std::vector<int> city_to_idx(n);

    for(int i=0; i<n; ++i) {
        city_to_idx[temp_pos[i]] = i;
    }

    for (int i = 0; i < n; ++i) {
        if (temp_pos[i] != to[i]) {
            int city_to_find = to[i];
            int current_city_at_pos = temp_pos[i];

            int idx_to_swap_with = city_to_idx[city_to_find];

            // Perform the swap
            std::swap(temp_pos[i], temp_pos[idx_to_swap_with]);
            
            // Record the swap
            diff.push_back({i, idx_to_swap_with});

            // Update the mapping
            city_to_idx[current_city_at_pos] = idx_to_swap_with;
            city_to_idx[city_to_find] = i;
        }
    }
    return diff;
}

void DpsoTsp::apply_velocity(std::vector<int>& position, const std::vector<SwapOp>& velocity) {
    for (const auto& op : velocity) {
        // Ensure indices are valid before swapping
        if (op.city_idx1 >= 0 && op.city_idx1 < position.size() &&
            op.city_idx2 >= 0 && op.city_idx2 < position.size()) {
            std::swap(position[op.city_idx1], position[op.city_idx2]);
        }
    }
}
