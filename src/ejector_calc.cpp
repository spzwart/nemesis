#include <cmath>
#include <limits>
#include <iostream>
#include <tuple>

extern "C" {
   void find_nearest_neighbour(double* x, double* y, double* z, 
                               int num_subsyst, double distance, 
                               double* ejected_bools){
        double threshold_distance = distance * distance;  // Cluster radius squared

        for (int j = 0; j < num_subsyst; j++){
            double min_dist_squared = std::numeric_limits<double>::max();
            double second_min_dist_squared = std::numeric_limits<double>::max();

            for (int i=0; i<num_subsyst; i++){
                if (i != j){
                    double dx = x[i] - x[j];
                    double dy = y[i] - y[j];
                    double dz = z[i] - z[j];
                    double dr_squared = dx*dx + dy*dy + dz*dz;
                    
                    // Check if currently iterated particle is closer than current minima's
                    if (dr_squared < min_dist_squared){
                        min_dist_squared = dr_squared;
                    }else if (dr_squared < second_min_dist_squared){
                        second_min_dist_squared = dr_squared;
                    }
                }
            }
            ejected_bools[j] = (second_min_dist_squared > threshold_distance) ? 1 : 0;
        }
    }
}