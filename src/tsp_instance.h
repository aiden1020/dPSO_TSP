#ifndef TSP_INSTANCE_H
#define TSP_INSTANCE_H

#include <vector>
#include <string>
#include <stdexcept>

// Represents a 2D point for city coordinates
struct Point {
    double x, y;
};

// Represents a single TSP problem instance parsed from a TSPLIB file
class TSPInstance {
public:
    // Constructor that parses a TSPLIB file
    explicit TSPInstance(const std::string& filepath);

    // Calculates the total length of a given tour (a permutation of city indices)
    double calculate_tour_length(const std::vector<int>& tour) const;

    // Returns the number of cities
    int get_dimension() const { return n; }

    // Returns the pre-calculated distance between two cities
    double get_distance(int city1_idx, int city2_idx) const;
    
    // Returns the known optimal tour length, or -1 if not available
    int get_best_known_length() const { return best_known_tour_length; }

    // Returns the name of the instance
    std::string get_name() const { return name; }

private:
    std::string name;
    int n; // Number of cities (dimension)
    std::vector<Point> coords; // Coordinates of cities
    std::vector<std::vector<double>> dist_matrix; // Pre-calculated distance matrix
    int best_known_tour_length;

    void parse_file(const std::string& filepath);
    void compute_dist_matrix();
};

#endif // TSP_INSTANCE_H
