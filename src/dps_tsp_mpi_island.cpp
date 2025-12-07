#ifdef USE_MPI

#include "dps_tsp_mpi_island.h"
#include "baseline_nn.h"
#include <algorithm>
#include <limits>
#include <chrono>

DpsoTspIslandMpi::DpsoTspIslandMpi(const TSPInstance& instance, const Parameters& params, MPI_Comm comm)
    : instance(instance), params(params), comm(comm), gbest_cost(std::numeric_limits<double>::max()) {
    MPI_Comm_rank(comm, &world_rank);
    MPI_Comm_size(comm, &world_size);
    int base = params.swarm_size / world_size;
    int rem = params.swarm_size % world_size;
    local_particle_count = base + (world_rank < rem ? 1 : 0);
}

void DpsoTspIslandMpi::initialize_swarm() {
    int n = instance.get_dimension();
    swarm.clear();
    swarm.resize(std::max(0, local_particle_count));

    // Seed one NN tour for a good starting point.
    std::vector<int> nn_tour = solve_nn(instance, 0);
    double nn_cost = instance.calculate_tour_length(nn_tour);
    gbest_position = nn_tour;
    gbest_cost = nn_cost;

    for (int i = 0; i < local_particle_count; ++i) {
        Particle& p = swarm[i];
        if (i == 0 && world_rank == 0) {
            p.position = nn_tour;
        } else {
            p.position = generate_random_tour(n);
        }
        p.cost = instance.calculate_tour_length(p.position);
        p.pbest_position = p.position;
        p.pbest_cost = p.cost;
        p.velocity.clear();

        if (p.cost < gbest_cost) {
            gbest_cost = p.cost;
            gbest_position = p.position;
        }
    }
}

void DpsoTspIslandMpi::update_particle(Particle& p) {
    std::vector<SwapOp> new_velocity;
    double move_start = MPI_Wtime();

    int inertia_size = static_cast<int>(p.velocity.size() * params.inertia_weight);
    if (inertia_size > 0) {
        new_velocity.insert(new_velocity.end(), p.velocity.begin(), p.velocity.begin() + inertia_size);
    }

    std::vector<SwapOp> diff_pbest = pso_calculate_diff(p.position, p.pbest_position);
    int pbest_swaps_count = static_cast<int>(diff_pbest.size() * params.cognitive_weight * Random::get_double());
    if (pbest_swaps_count > 0 && pbest_swaps_count <= static_cast<int>(diff_pbest.size())) {
        new_velocity.insert(new_velocity.end(), diff_pbest.begin(), diff_pbest.begin() + pbest_swaps_count);
    }

    std::vector<SwapOp> diff_gbest = pso_calculate_diff(p.position, gbest_position);
    int gbest_swaps_count = static_cast<int>(diff_gbest.size() * params.social_weight * Random::get_double());
    if (gbest_swaps_count > 0 && gbest_swaps_count <= static_cast<int>(diff_gbest.size())) {
        new_velocity.insert(new_velocity.end(), diff_gbest.begin(), diff_gbest.begin() + gbest_swaps_count);
    }

    if (static_cast<int>(new_velocity.size()) > params.max_velocity_len) {
        new_velocity.resize(params.max_velocity_len);
    }
    p.velocity = new_velocity;

    pso_apply_velocity(p.position, p.velocity);

    if (Random::get_double() < params.mutation_prob) {
        int idx1 = Random::get_int(0, static_cast<int>(p.position.size()) - 1);
        int idx2 = Random::get_int(0, static_cast<int>(p.position.size()) - 1);
        swap_cities(p.position, idx1, idx2);
    }
    double move_end = MPI_Wtime();

    p.cost = instance.calculate_tour_length(p.position);
    p.cost = pso_two_opt_local_search(p.position, instance, p.cost, params.local_search_attempts);
    double eval_end = MPI_Wtime();

    timing.update_move_ms += (move_end - move_start) * 1000.0;
    timing.update_eval_ms += (eval_end - move_end) * 1000.0;

    if (p.cost < p.pbest_cost) {
        p.pbest_cost = p.cost;
        p.pbest_position = p.position;
    }
}

void DpsoTspIslandMpi::migrate_topk(int migration_size) {
    if (world_size == 1 || migration_size <= 0) return;

    int n = instance.get_dimension();
    migration_size = std::min(migration_size, static_cast<int>(swarm.size()));
    if (migration_size == 0) return;

    // Select top-k pbest tours.
    std::vector<std::pair<double, int>> scores;
    scores.reserve(swarm.size());
    for (int i = 0; i < static_cast<int>(swarm.size()); ++i) {
        scores.push_back({swarm[i].pbest_cost, i});
    }
    std::partial_sort(scores.begin(), scores.begin() + migration_size, scores.end(),
                      [](const auto& a, const auto& b) { return a.first < b.first; });

    std::vector<int> send_buffer(n * migration_size);
    for (int k = 0; k < migration_size; ++k) {
        const auto& tour = swarm[scores[k].second].pbest_position;
        std::copy(tour.begin(), tour.end(), send_buffer.begin() + k * n);
    }

    std::vector<int> recv_buffer(n * migration_size, -1);
    int send_to = (world_rank + 1) % world_size;
    int recv_from = (world_rank - 1 + world_size) % world_size;

    MPI_Sendrecv(send_buffer.data(), send_buffer.size(), MPI_INT, send_to, 0,
                 recv_buffer.data(), recv_buffer.size(), MPI_INT, recv_from, 0,
                 comm, MPI_STATUS_IGNORE);

    // Incorporate received tours if they improve gbest/pbest.
    for (int k = 0; k < migration_size; ++k) {
        std::vector<int> tour(recv_buffer.begin() + k * n, recv_buffer.begin() + (k + 1) * n);
        if (tour.size() != static_cast<size_t>(n)) continue;
        double cost = instance.calculate_tour_length(tour);

        // Replace worst particle if improved.
        auto worst_it = std::max_element(swarm.begin(), swarm.end(),
                                         [](const Particle& a, const Particle& b) { return a.pbest_cost < b.pbest_cost; });
        if (worst_it != swarm.end() && cost < worst_it->pbest_cost) {
            worst_it->position = tour;
            worst_it->velocity.clear();
            worst_it->cost = cost;
            worst_it->pbest_position = tour;
            worst_it->pbest_cost = cost;
        }
        if (cost < gbest_cost) {
            gbest_cost = cost;
            gbest_position = std::move(tour);
        }
    }
}

void DpsoTspIslandMpi::solve() {
    timing = {};
    auto total_start = MPI_Wtime();

    auto init_start = MPI_Wtime();
    initialize_swarm();
    auto init_end = MPI_Wtime();
    timing.init_ms = (init_end - init_start) * 1000.0;

    constexpr int migration_interval = 20;
    int migration_size = 1;

    for (int iter = 0; iter < params.max_iter; ++iter) {
        auto update_start = MPI_Wtime();
        for (Particle& p : swarm) {
            update_particle(p);
            if (p.pbest_cost < gbest_cost) {
                gbest_cost = p.pbest_cost;
                gbest_position = p.pbest_position;
            }
        }
        auto update_end = MPI_Wtime();
        timing.update_ms += (update_end - update_start) * 1000.0;

        if ((iter + 1) % migration_interval == 0 || iter == params.max_iter - 1) {
            auto comm_start = MPI_Wtime();
            migrate_topk(migration_size);
            auto comm_end = MPI_Wtime();
            timing.comm_ms += (comm_end - comm_start) * 1000.0;
        }
    }

    auto total_end = MPI_Wtime();
    timing.total_ms = (total_end - total_start) * 1000.0;
}

#endif // USE_MPI
