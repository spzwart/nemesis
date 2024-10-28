#include <cmath>
#include <vector>

// Function to calculate the gravitational force on a particle at a given point
extern "C" {
    void find_gravity_at_point(long double* particles_mass, long double* particles_x, 
                               long double* particles_y, long double* particles_z,
                               long double* x, long double* y, long double* z, 
                               long double* ax, long double* ay, long double* az, 
                               int num_extern, int num_subsyst){
        for (int i = 0; i < num_extern; i++) {
            for (int j = 0; j < num_subsyst; j++) {
                long double dx = particles_x[i] - x[j];
                long double dy = particles_y[i] - y[j];
                long double dz = particles_z[i] - z[j];

                long double dr_squared = dx*dx + dy*dy + dz*dz;
                long double dr = std::sqrt(dr_squared);
                long double dr_cubed = dr*dr_squared;

                ax[i] -= particles_mass[j] * dx / dr_cubed;
                ay[i] -= particles_mass[j] * dy / dr_cubed;
                az[i] -= particles_mass[j] * dz / dr_cubed;
            }
        }
    }
}