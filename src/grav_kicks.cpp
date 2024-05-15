#include <cmath>
#include <vector>
#include <iostream>
#include <ostream>

extern "C" {
    void find_gravity_at_point(double* particles_mass, double* particles_x, 
                               double* particles_y, double* particles_z,
                               int num_extern, double* result_ax, double* result_ay, 
                               double* result_az, double* x, 
                               double* y, double* z, int num_subsyst){
        for (int i = 0; i < num_extern; i++) {
            for (int j = 0; j < num_subsyst; j++) {
                double dx = particles_x[i] - x[j];
                double dy = particles_y[i] - y[j];
                double dz = particles_z[i] - z[j];
                double dr = dx*dx + dy*dy + dz*dz;
                double dr_cubed = dr*std::sqrt(dr);

                result_ax[i] -= particles_mass[j] * dx / dr_cubed;
                result_ay[i] -= particles_mass[j] * dy / dr_cubed;
                result_az[i] -= particles_mass[j] * dz / dr_cubed;
            }
        }
    }
}