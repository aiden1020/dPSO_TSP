#pragma once

#ifdef USE_MPI

#include <vector>
#include <mpi.h>
#include "dps_tsp.h"

// Naive parallel dPSO v2: separate class name to allow side-by-side testing.
class DpsoTspNaiveMpi_v2 {
public:
    using Parameters = DpsoTsp::Parameters;

    struct Timing {
        double total_ms = 0.0;
        double init_ms = 0.0;
        double update_ms = 0.0;      // local compute
        double update_move_ms = 0.0; // velocity/position update + mutation
        double update_eval_ms = 0.0; // tour length + local search
        double comm_ms = 0.0;        // collectives
    };

    DpsoTspNaiveMpi_v2(const TSPInstance& instance, const Parameters& params, MPI_Comm comm);

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

    std::vector<Particle> local_swarm;
    std::vector<int> gbest_position;
    double gbest_cost;
    int local_particle_count = 0;

    Timing timing;

    void initialize_swarm();
    void update_particle(Particle& p);
};

#endif // USE_MPI
