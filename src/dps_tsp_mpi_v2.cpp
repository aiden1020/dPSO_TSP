#ifdef USE_MPI

#include "dps_tsp_mpi_v2.h"
#include "baseline_nn.h"
#include <algorithm>
#include <limits>
#include <chrono>

DpsoTspNaiveMpi_v2::DpsoTspNaiveMpi_v2(const TSPInstance& instance, const Parameters& params, MPI_Comm comm)
    : instance(instance), params(params), comm(comm), gbest_cost(std::numeric_limits<double>::max()) {
    MPI_Comm_rank(comm, &world_rank);
    MPI_Comm_size(comm, &world_size);

    int base = params.swarm_size / world_size;
    int rem = params.swarm_size % world_size;
    local_particle_count = base + (world_rank < rem ? 1 : 0);
}

void DpsoTspNaiveMpi_v2::initialize_swarm() {
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

void DpsoTspNaiveMpi_v2::update_particle(Particle& p) {
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

void DpsoTspNaiveMpi_v2::solve() {
    timing = {};
    auto total_start = MPI_Wtime();

    auto init_start = MPI_Wtime();
    initialize_swarm();
    auto init_end = MPI_Wtime();
    timing.init_ms = (init_end - init_start) * 1000.0;

    int n = instance.get_dimension();
    std::vector<int> local_best_tour = gbest_position;
    double local_best_cost = gbest_cost;
    constexpr int SYNC_INTERVAL = 10;

    for (int iter = 0; iter < params.max_iter; ++iter) {
        // 1. 本地計算
        local_best_cost = gbest_cost; // 預設為當前已知的最佳
        local_best_tour = gbest_position; 
        bool local_improved = false; // 標記本地是否有比 gbest 更強的解

        auto update_start = MPI_Wtime();
        for (Particle& p : local_swarm) {
            update_particle(p);
            // 只有當粒子比"歷史全域最佳"還好時，才考慮更新
            if (p.pbest_cost < local_best_cost) {
                local_best_cost = p.pbest_cost;
                local_best_tour = p.pbest_position;
                local_improved = true;
            }
        }
        auto update_end = MPI_Wtime();
        timing.update_ms += (update_end - update_start) * 1000.0;

        // 減少同步頻率：每 SYNC_INTERVAL 代或最後一代才同步一次
        if ((iter + 1) % SYNC_INTERVAL != 0 && iter != params.max_iter - 1) {
            continue;
        }

        auto comm_start = MPI_Wtime();

        // 2. Allreduce: 找出全網最好的 cost
        struct {
            double cost;
            int rank;
        } local_pair, global_pair;

        // 這裡很關鍵：如果我沒有比 gbest 好的解，我傳送 max_double 或 gbest_cost
        // 為了避免浮點數誤差，建議邏輯如下：
        // 如果本地有進步，送出新 cost；否則送出一個標記值 (如 infinity 或原 gbest)
        // 但為了簡單起見，直接送 local_best_cost 即可
        local_pair.cost = local_best_cost;
        local_pair.rank = world_rank;

        MPI_Allreduce(&local_pair, &global_pair, 1, MPI_DOUBLE_INT, MPI_MINLOC, comm);

        // 3. 條件式 Bcast: 只有當找到比當前 gbest 更優的解時才廣播
        // 注意：浮點數比較建議用 epsilon，但 TSP 若是整數距離通常無此問題。
        // 若使用 double 距離，建議: global_pair.cost < gbest_cost - 1e-9
        if (global_pair.cost < gbest_cost) {
            
            int n = instance.get_dimension();
            std::vector<int> new_gbest_tour(n);

            // 只有贏家需要準備資料
            if (world_rank == global_pair.rank) {
                // 注意：這裡必須確保 global_pair.rank 真的是持有該路徑的人
                // 因為我們上面 local_best_tour 預設是 gbest_position
                // 如果沒有人進步，global_pair.cost 會等於 gbest_cost
                // 進到這個 if 代表真的有變小，所以 rank 一定是發現新解的人
                 new_gbest_tour = local_best_tour;
            }

            // 廣播路徑 (最耗時的操作，現在只在進步時發生)
            MPI_Bcast(new_gbest_tour.data(), n, MPI_INT, global_pair.rank, comm);

            // 更新本地存儲
            gbest_position = new_gbest_tour;
            gbest_cost = global_pair.cost;
        }
        // else: 沒人進步，什麼都不用做，省下 Bcast 時間

        auto comm_end = MPI_Wtime();
        timing.comm_ms += (comm_end - comm_start) * 1000.0;
    }

    auto total_end = MPI_Wtime();
    timing.total_ms = (total_end - total_start) * 1000.0;
}

#endif // USE_MPI
