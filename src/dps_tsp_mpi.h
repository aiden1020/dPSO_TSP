#pragma once

#ifdef USE_MPI

#include <vector>
#include <mpi.h>
#include "dps_tsp.h"

// Naive parallel (single-swarm) dPSO using MPI. Particles are split across ranks,
// and each iteration performs one Allreduce (minloc) + one Bcast to share gbest.
class DpsoTspNaiveMpi {
public:
    using Parameters = DpsoTsp::Parameters;

    struct Timing {
        double total_ms = 0.0;
        double init_ms = 0.0;
        double update_ms = 0.0; // local compute (particle updates + evaluation + LS)
        double update_move_ms = 0.0; // velocity/position update + mutation
        double update_eval_ms = 0.0; // tour length + local search
        double comm_ms = 0.0;   // MPI collectives per iteration
    };

    DpsoTspNaiveMpi(const TSPInstance& instance, const Parameters& params, MPI_Comm comm);

    void solve();

    const std::vector<int>& get_gbest_position() const { return gbest_position; }
    double get_gbest_cost() const { return gbest_cost; }
    Timing get_timing() const { return timing; }

private:
    const TSPInstance& instance;
    Parameters params;
    MPI_Comm comm;
    int world_rank = 0;
    int world_size = 1;

    std::vector<Particle> local_swarm;
    std::vector<int> gbest_position;
    double gbest_cost;
    int local_particle_count = 0;

    Timing timing;

    void initialize_swarm();
    void update_particle(Particle& p);

    // Helpers duplicated from sequential dPSO
    std::vector<SwapOp> calculate_diff(const std::vector<int>& from, const std::vector<int>& to);
    void apply_velocity(std::vector<int>& position, const std::vector<SwapOp>& velocity);
    double two_opt_local_search(std::vector<int>& tour, double current_cost);
};

#endif // USE_MPI
