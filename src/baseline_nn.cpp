#include "baseline_nn.h"
#include <vector>
#include <limits>
#include <numeric>

std::vector<int> solve_nn(const TSPInstance& instance, int start_city) {
    int n = instance.get_dimension();
    if (n == 0) return {};

    std::vector<int> tour;
    tour.reserve(n);
    std::vector<bool> visited(n, false);

    int current_city = start_city;
    tour.push_back(current_city);
    visited[current_city] = true;

    for (int i = 1; i < n; ++i) {
        int next_city = -1;
        double min_dist = std::numeric_limits<double>::max();

        for (int j = 0; j < n; ++j) {
            if (!visited[j]) {
                double dist = instance.get_distance(current_city, j);
                if (dist < min_dist) {
                    min_dist = dist;
                    next_city = j;
                }
            }
        }
        
        if (next_city != -1) {
            current_city = next_city;
            tour.push_back(current_city);
            visited[current_city] = true;
        } else {
            // Should not happen in a complete graph
            break; 
        }
    }

    return tour;
}
