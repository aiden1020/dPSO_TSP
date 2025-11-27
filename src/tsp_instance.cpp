#include "tsp_instance.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <limits>

// TSPLIB specifies rounding to the nearest integer for EUC_2D distances
int nint(double x) {
    return static_cast<int>(std::floor(x + 0.5));
}

TSPInstance::TSPInstance(const std::string& filepath) : n(0), best_known_tour_length(-1) {
    parse_file(filepath);
}

void TSPInstance::parse_file(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filepath);
    }

    std::string line;
    std::string section;
    bool reading_coords = false;

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string key;
        ss >> key;

        if (key == "NAME:") {
            ss >> name;
        } else if (key == "NAME") { // Some files have a slightly different format
             if (line.find(":") != std::string::npos) {
                name = line.substr(line.find(":") + 1);
                name.erase(0, name.find_first_not_of(" \t"));
                name.erase(name.find_last_not_of(" \t") + 1);
            }
        } else if (key == "TYPE:") {
            std::string type;
            ss >> type;
            if (type != "TSP") {
                std::cerr << "Warning: This parser is designed for TSP type, but found " << type << std::endl;
            }
        } else if (key == "DIMENSION:") {
            ss >> n;
        } else if (key == "DIMENSION") {
            if (line.find(":") != std::string::npos) {
                std::string dim_str = line.substr(line.find(":") + 1);
                n = std::stoi(dim_str);
            }
        }
        else if (key == "EDGE_WEIGHT_TYPE:") {
            std::string weight_type;
            ss >> weight_type;
            if (weight_type != "EUC_2D") {
                throw std::runtime_error("Unsupported edge weight type: " + weight_type);
            }
        } else if (key == "NODE_COORD_SECTION") {
            reading_coords = true;
            coords.reserve(n);
        } else if (key == "EOF") {
            break;
        } else if (reading_coords) {
            std::stringstream line_parser(line);
            int id;
            double x, y;
            if (line_parser >> id >> x >> y) {
                coords.push_back({x, y});
            }
        }
    }

    if (coords.size() != n) {
        throw std::runtime_error("Mismatch between DIMENSION and number of coordinates read.");
    }

    compute_dist_matrix();
}

void TSPInstance::compute_dist_matrix() {
    dist_matrix.resize(n, std::vector<double>(n));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i == j) {
                dist_matrix[i][j] = 0;
            } else {
                double dx = coords[i].x - coords[j].x;
                double dy = coords[i].y - coords[j].y;
                dist_matrix[i][j] = nint(std::sqrt(dx * dx + dy * dy));
            }
        }
    }
}

double TSPInstance::calculate_tour_length(const std::vector<int>& tour) const {
    if (tour.size() != n) {
        throw std::invalid_argument("Tour size must match instance dimension.");
    }
    double length = 0.0;
    for (size_t i = 0; i < n; ++i) {
        int city_a = tour[i];
        int city_b = tour[(i + 1) % n]; // Wrap around for the last city to the first
        length += dist_matrix[city_a][city_b];
    }
    return length;
}

double TSPInstance::get_distance(int city1_idx, int city2_idx) const {
    if (city1_idx >= 0 && city1_idx < n && city2_idx >= 0 && city2_idx < n) {
        return dist_matrix[city1_idx][city2_idx];
    }
    throw std::out_of_range("City index out of range.");
}
