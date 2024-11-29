#include <cmath>
#include <limits>
#include <iostream>
#include <tuple>

extern "C" {
   void find_nearest_neighbour(double* x, double* y, double* z, 
                               double* com, int num_subsyst, 
                               double distance, int* ejected_bools){
        double threshold_distance = distance * distance;  // Cluster radius squared

        for (int i = 0; i < num_subsyst; i++){
            double dx = com[0] - x[i];
            double dy = com[1] - y[i];
            double dz = com[2] - z[i];

            double distance_to_center = dx*dx + dy*dy + dz*dz;

            ejected_bools[i] = (distance_to_center > threshold_distance) ? 1 : 0;
        }
    }
}