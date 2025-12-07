#pragma once

#ifdef USE_MPI

#include <vector>
#include <mpi.h>
#include "dps_tsp.h"

// Island-style MPI dPSO: each rank keeps its own swarm, and periodically migrates top-k tours in a ring.
class DpsoTspIslandMpi {
public:
    using Parameters = DpsoTsp::Parameters;

    struct Timing {
        double total_ms = 0.0;
        double init_ms = 0.0;
        double update_ms = 0.0;      // local compute
        double update_move_ms = 0.0; // velocity/position update + mutation
        double update_eval_ms = 0.0; // tour length + local search
        double comm_ms = 0.0;        // migration time
    };

    // Additional island controls are passed via Parameters:
    // - migration_interval: sync every k iterations (default 20)
    // - migration_size: how many top tours to migrate (default 1)
    DpsoTspIslandMpi(const TSPInstance& instance, const Parameters& params, MPI_Comm comm);

    void solve();

    const std::vector<int>& get_gbest_position() const { return gbest_position; }
    double get_gbest_cost() const { return gbest_cost; }
    Timing get_timing() const { return timing; }
    double get_comm_time() const { return timing.comm_ms; }

private:
    const TSPInstance& instance;
    Parameters params;
    MPI_Comm comm;
    int world_rank = 0;
    int world_size = 1;

    std::vector<Particle> swarm;
    std::vector<int> gbest_position;
    double gbest_cost;
    Timing timing;
    int local_particle_count = 0;

    void initialize_swarm();
    void update_particle(Particle& p);
    void migrate_topk(int migration_size);
};

#endif // USE_MPI
