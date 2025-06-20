#include <cmath>
#include <vector>
#include <omp.h>

#define _TINY_ pow(2.0, -52.)

// Function to calculate the gravitational force on a particle at a given point
extern "C" {
    void find_gravity_at_point(double* pert_mass, double* pert_x, double* pert_y, double* pert_z, 
                               double* particles_x, double* particles_y, double* particles_z,
                               double* ax, double* ay, double* az, int num_extern, int num_subsyst){

        // Parallelise loop so that each iteration i writes to a unique correction acceleration
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < num_extern; i++) {
            double ax_local = 0.0;
            double ay_local = 0.0;
            double az_local = 0.0;

            #pragma omp simd
            for (int j = 0; j < num_subsyst; j++) {
                // Ignore massless particles
                if (pert_mass[j] > _TINY_){
                    double dx = particles_x[i] - pert_x[j];
                    double dy = particles_y[i] - pert_y[j];
                    double dz = particles_z[i] - pert_z[j];

                    double dr_squared = dx*dx + dy*dy + dz*dz;
                    double dr = std::sqrt(dr_squared);
                    double dr_cubed = dr*dr_squared;

                    ax_local -= pert_mass[j] * dx / dr_cubed;
                    ay_local -= pert_mass[j] * dy / dr_cubed;
                    az_local -= pert_mass[j] * dz / dr_cubed;
                }
            }
                    
            ax[i] += ax_local;
            ay[i] += ay_local;
            az[i] += az_local;

        }
    }
}