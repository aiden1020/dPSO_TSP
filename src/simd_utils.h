#pragma once

// This file is only active if AVX2 and FMA are enabled in the compiler
// AND the build defines USE_TSP_SIMD.
#if defined(USE_TSP_SIMD) && defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#include <vector>
#include <cmath>
#include "tsp_instance.h"

// Calculates the tour length using AVX2 and FMA intrinsics.
// Processes tour segments in batches of 4.
inline double calculate_tour_length_avx2(const std::vector<Point>& cities, const std::vector<int>& tour) {
    size_t n = tour.size();
    if (n < 2) return 0.0;
    
    // Vector to accumulate distances for 4 segments at a time
    __m256d total_dist_vec = _mm256_setzero_pd();
    
    size_t i = 0;
    // We process the first (n-1) segments. The loop runs as long as we can load 4 full segments.
    // Each iteration handles tour[i]->tour[i+1], ..., tour[i+3]->tour[i+4].
    // So we must have i+4 < n. To be safe, we check i + 3 < n -1, which is i+4 < n
    size_t vec_limit = (n > 4) ? (n - 1 - 3) : 0;


    for (; i < vec_limit; i += 4) {
        // 'from' cities: tour[i], tour[i+1], tour[i+2], tour[i+3]
        // _mm256_set_pd loads in reverse order: (d, c, b, a)
        __m256d x1 = _mm256_set_pd(cities[tour[i+3]].x, cities[tour[i+2]].x, cities[tour[i+1]].x, cities[tour[i]].x);
        __m256d y1 = _mm256_set_pd(cities[tour[i+3]].y, cities[tour[i+2]].y, cities[tour[i+1]].y, cities[tour[i]].y);

        // 'to' cities: tour[i+1], tour[i+2], tour[i+3], tour[i+4]
        __m256d x2 = _mm256_set_pd(cities[tour[i+4]].x, cities[tour[i+3]].x, cities[tour[i+2]].x, cities[tour[i+1]].x);
        __m256d y2 = _mm256_set_pd(cities[tour[i+4]].y, cities[tour[i+3]].y, cities[tour[i+2]].y, cities[tour[i+1]].y);

        // Calculate differences
        __m256d dx = _mm256_sub_pd(x1, x2);
        __m256d dy = _mm256_sub_pd(y1, y2);
        
        // Fused-Multiply-Add for squared distances: dx*dx + dy*dy
        __m256d dist_sq = _mm256_fmadd_pd(dx, dx, _mm256_mul_pd(dy, dy));
        
        // Calculate square roots
        __m256d distances = _mm256_sqrt_pd(dist_sq);
        
        // Accumulate distances
        total_dist_vec = _mm256_add_pd(total_dist_vec, distances);
    }
    
    // Horizontal sum of the vector to get a single double value
    __m128d hsum = _mm_add_pd(_mm256_castpd256_pd128(total_dist_vec), _mm256_extractf128_pd(total_dist_vec, 1));
    hsum = _mm_hadd_pd(hsum, hsum);
    double total_distance = _mm_cvtsd_f64(hsum);

    // Scalar part for the remainder of the (n-1) segments
    for (; i < n - 1; ++i) {
        double dx_s = cities[tour[i]].x - cities[tour[i+1]].x;
        double dy_s = cities[tour[i]].y - cities[tour[i+1]].y;
        total_distance += std::sqrt(dx_s*dx_s + dy_s*dy_s);
    }

    // Add the last segment that connects the end of the tour back to the start
    double dx_last = cities[tour[n-1]].x - cities[tour[0]].x;
    double dy_last = cities[tour[n-1]].y - cities[tour[0]].y;
    total_distance += std::sqrt(dx_last*dx_last + dy_last*dy_last);

    return total_distance;
}
#endif // __AVX2__ && __FMA__
