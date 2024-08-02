#include <cmath>
#include <limits>
#include <iostream>
#include <tuple>

extern "C" {
   void find_nearest_neighbour(double* x, double* y, double* z, int num_subsyst, double distance, double* ejected_bools){
        for (int j = 0; j < num_subsyst; j++){
            double min_dist = std::numeric_limits<double>::max();
            double second_min_dist = std::numeric_limits<double>::max();
            for (int i=0; i<num_subsyst; i++){
                if (i != j){
                    double dx = x[i] - x[j];
                    double dy = y[i] - y[j];
                    double dz = z[i] - z[j];
                    double dr = dx*dx + dy*dy + dz*dz;
                    double dist = std::sqrt(dr);
                    if (dist < min_dist){
                        min_dist = dist;
                    }else if (dist < second_min_dist)
                    {
                        second_min_dist = dist;
                    }
                }
            }
            ejected_bools[j] = (second_min_dist > distance) ? 1 : 0;
        }
    }
}