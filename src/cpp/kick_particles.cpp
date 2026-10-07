#include <cmath>
#include <omp.h>

#define _TINY_ pow(2.0, -52.)

// Function to calculate the gravitational force on a particle at a given point
extern "C" {
    void find_gravity_at_point(
        double* pert_mass, double* pert_x, double* pert_y, double* pert_z, 
        double* particles_x, double* particles_y, double* particles_z,
        double* ax, double* ay, double* az, int num_extern, int num_subsyst
    ){
        // Parallelise loop so that each iteration i writes to a unique correction acceleration
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < num_extern; i++) {
            double ax_local = 0.0;
            double ay_local = 0.0;
            double az_local = 0.0;

            const double xi = particles_x[i];
            const double yi = particles_y[i];
            const double zi = particles_z[i];

            #pragma omp simd
            for (int j = 0; j < num_subsyst; j++) {
                // Ignore massless particles
                if (pert_mass[j] > _TINY_){
                    double dx = xi - pert_x[j];
                    double dy = yi - pert_y[j];
                    double dz = zi - pert_z[j];

                    double dr2 = dx*dx + dy*dy + dz*dz;
                    double dr = std::sqrt(dr2);
                    double inv_dr3 = 1 / (dr*dr2);

                    ax_local -= pert_mass[j] * dx * inv_dr3;
                    ay_local -= pert_mass[j] * dy * inv_dr3;
                    az_local -= pert_mass[j] * dz * inv_dr3;
                }
            }
            ax[i] += ax_local;
            ay[i] += ay_local;
            az[i] += az_local;
        }
    }
}