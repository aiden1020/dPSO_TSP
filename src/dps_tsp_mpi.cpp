#ifdef USE_MPI

#include "dps_tsp_mpi.h"
#include "baseline_nn.h"
#include <algorithm>
#include <limits>
#include <chrono>

DpsoTspNaiveMpi::DpsoTspNaiveMpi(const TSPInstance& instance, const Parameters& params, MPI_Comm comm)
    : instance(instance), params(params), comm(comm), gbest_cost(std::numeric_limits<double>::max()) {
    MPI_Comm_rank(comm, &world_rank);
    MPI_Comm_size(comm, &world_size);

    int base = params.swarm_size / world_size;
    int rem = params.swarm_size % world_size;
    local_particle_count = base + (world_rank < rem ? 1 : 0);
}

std::vector<SwapOp> DpsoTspNaiveMpi::calculate_diff(const std::vector<int>& from, const std::vector<int>& to) {
    std::vector<SwapOp> diff;
    if (from == to) return diff;

    std::vector<int> temp_pos = from;
    int n = static_cast<int>(from.size());
    std::vector<int> city_to_idx(n);

    for (int i = 0; i < n; ++i) {
        city_to_idx[temp_pos[i]] = i;
    }

    for (int i = 0; i < n; ++i) {
        if (temp_pos[i] != to[i]) {
            int city_to_find = to[i];
            int current_city_at_pos = temp_pos[i];

            int idx_to_swap_with = city_to_idx[city_to_find];

            std::swap(temp_pos[i], temp_pos[idx_to_swap_with]);
            diff.push_back({i, idx_to_swap_with});

            city_to_idx[current_city_at_pos] = idx_to_swap_with;
            city_to_idx[city_to_find] = i;
        }
    }
    return diff;
}

void DpsoTspNaiveMpi::apply_velocity(std::vector<int>& position, const std::vector<SwapOp>& velocity) {
    for (const auto& op : velocity) {
        if (op.city_idx1 >= 0 && op.city_idx1 < static_cast<int>(position.size()) &&
            op.city_idx2 >= 0 && op.city_idx2 < static_cast<int>(position.size())) {
            std::swap(position[op.city_idx1], position[op.city_idx2]);
        }
    }
}

double DpsoTspNaiveMpi::two_opt_local_search(std::vector<int>& tour, double current_cost) {
    int attempts = params.local_search_attempts;
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
            reverse_segment(tour, i, j);
        }
    }
    return best_cost;
}

void DpsoTspNaiveMpi::initialize_swarm() {
    int n = instance.get_dimension();
    local_swarm.clear();
    local_swarm.resize(std::max(0, local_particle_count));

    // All ranks build the same NN tour for seeding; cheap and avoids extra broadcast.
    std::vector<int> nn_tour = solve_nn(instance, 0);
    double nn_cost = instance.calculate_tour_length(nn_tour);
    gbest_position = nn_tour;
    gbest_cost = nn_cost;

    for (int i = 0; i < local_particle_count; ++i) {
        Particle& p = local_swarm[i];
        if (world_rank == 0 && i == 0) {
            p.position = nn_tour;
        } else {
            p.position = generate_random_tour(n);
        }
        p.cost = instance.calculate_tour_length(p.position);
        p.pbest_position = p.position;
        p.pbest_cost = p.cost;
        p.velocity.clear();
    }
}

void DpsoTspNaiveMpi::update_particle(Particle& p) {
    std::vector<SwapOp> new_velocity;
    double move_start = MPI_Wtime();

    int inertia_size = static_cast<int>(p.velocity.size() * params.inertia_weight);
    if (inertia_size > 0) {
        new_velocity.insert(new_velocity.end(), p.velocity.begin(), p.velocity.begin() + inertia_size);
    }

    std::vector<SwapOp> diff_pbest = calculate_diff(p.position, p.pbest_position);
    int pbest_swaps_count = static_cast<int>(diff_pbest.size() * params.cognitive_weight * Random::get_double());
    if (pbest_swaps_count > 0 && pbest_swaps_count <= static_cast<int>(diff_pbest.size())) {
        new_velocity.insert(new_velocity.end(), diff_pbest.begin(), diff_pbest.begin() + pbest_swaps_count);
    }

    std::vector<SwapOp> diff_gbest = calculate_diff(p.position, gbest_position);
    int gbest_swaps_count = static_cast<int>(diff_gbest.size() * params.social_weight * Random::get_double());
    if (gbest_swaps_count > 0 && gbest_swaps_count <= static_cast<int>(diff_gbest.size())) {
        new_velocity.insert(new_velocity.end(), diff_gbest.begin(), diff_gbest.begin() + gbest_swaps_count);
    }

    if (static_cast<int>(new_velocity.size()) > params.max_velocity_len) {
        new_velocity.resize(params.max_velocity_len);
    }
    p.velocity = new_velocity;

    apply_velocity(p.position, p.velocity);

    if (Random::get_double() < params.mutation_prob) {
        int idx1 = Random::get_int(0, static_cast<int>(p.position.size()) - 1);
        int idx2 = Random::get_int(0, static_cast<int>(p.position.size()) - 1);
        swap_cities(p.position, idx1, idx2);
    }
    double move_end = MPI_Wtime();

    p.cost = instance.calculate_tour_length(p.position);
    p.cost = two_opt_local_search(p.position, p.cost);
    double eval_end = MPI_Wtime();

    timing.update_move_ms += (move_end - move_start) * 1000.0;
    timing.update_eval_ms += (eval_end - move_end) * 1000.0;

    if (p.cost < p.pbest_cost) {
        p.pbest_cost = p.cost;
        p.pbest_position = p.position;
    }
}

void DpsoTspNaiveMpi::solve() {
    timing = {};
    auto total_start = MPI_Wtime();

    auto init_start = MPI_Wtime();
    initialize_swarm();
    auto init_end = MPI_Wtime();
    timing.init_ms = (init_end - init_start) * 1000.0;

    int n = instance.get_dimension();
    std::vector<int> local_best_tour = gbest_position;
    double local_best_cost = gbest_cost;

    for (int iter = 0; iter < params.max_iter; ++iter) {
        local_best_cost = gbest_cost;
        local_best_tour = gbest_position;

        auto update_start = MPI_Wtime();
        for (Particle& p : local_swarm) {
            update_particle(p);
            if (p.pbest_cost < local_best_cost) {
                local_best_cost = p.pbest_cost;
                local_best_tour = p.pbest_position;
            }
        }
        auto update_end = MPI_Wtime();
        timing.update_ms += (update_end - update_start) * 1000.0;

        auto comm_start = MPI_Wtime();
        struct {
            double cost;
            int rank;
        } local_pair{local_best_cost, world_rank}, global_pair{};

        MPI_Allreduce(&local_pair, &global_pair, 1, MPI_DOUBLE_INT, MPI_MINLOC, comm);

        std::vector<int> gbest_candidate = gbest_position;
        if (world_rank == global_pair.rank) {
            gbest_candidate = local_best_tour;
        }
        MPI_Bcast(gbest_candidate.data(), n, MPI_INT, global_pair.rank, comm);
        gbest_position = gbest_candidate;
        gbest_cost = global_pair.cost;
        auto comm_end = MPI_Wtime();
        timing.comm_ms += (comm_end - comm_start) * 1000.0;
    }

    auto total_end = MPI_Wtime();
    timing.total_ms = (total_end - total_start) * 1000.0;
}

#endif // USE_MPI
