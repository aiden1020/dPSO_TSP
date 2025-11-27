#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include <random>
#include <numeric>
#include <algorithm>

// A simple wrapper for the C++ random number generator
class Random {
public:
    static void seed(unsigned int s) {
        engine.seed(s);
    }

    // Get a random integer in [min, max]
    static int get_int(int min, int max) {
        std::uniform_int_distribution<int> dist(min, max);
        return dist(engine);
    }

    // Get a random double in [0.0, 1.0)
    static double get_double() {
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        return dist(engine);
    }

    // Get a reference to the underlying engine for algorithms like std::shuffle
    static std::mt19937& get_engine() {
        return engine;
    }

private:
    // Use a static engine to ensure it's seeded only once if desired
    static std::mt19937 engine;
};

// --- Tour Manipulation Utilities ---

// Generates a random tour (permutation of 0 to n-1)
std::vector<int> generate_random_tour(int n);

// Performs a swap operation on a tour
void swap_cities(std::vector<int>& tour, int i, int j);

// Reverses a segment of the tour between indices i and j (inclusive)
void reverse_segment(std::vector<int>& tour, int i, int j);

#endif // UTILS_H
