#include "utils.h"

// Initialize the static random engine
std::mt19937 Random::engine(std::random_device{}());

std::vector<int> generate_random_tour(int n) {
    std::vector<int> tour(n);
    std::iota(tour.begin(), tour.end(), 0); // Fill with 0, 1, ..., n-1
    std::shuffle(tour.begin(), tour.end(), Random::get_engine());
    return tour;
}

void swap_cities(std::vector<int>& tour, int i, int j) {
    if (i >= 0 && i < tour.size() && j >= 0 && j < tour.size()) {
        std::swap(tour[i], tour[j]);
    }
}

void reverse_segment(std::vector<int>& tour, int i, int j) {
    if (i < 0 || j >= tour.size() || i > j) return;
    std::reverse(tour.begin() + i, tour.begin() + j + 1);
}
