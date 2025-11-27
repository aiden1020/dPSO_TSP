#ifndef BASELINE_NN_H
#define BASELINE_NN_H

#include "tsp_instance.h"
#include <vector>

// Solves the TSP using the Nearest Neighbor heuristic
// Starts from the given start_city index
std::vector<int> solve_nn(const TSPInstance& instance, int start_city = 0);

#endif // BASELINE_NN_H
