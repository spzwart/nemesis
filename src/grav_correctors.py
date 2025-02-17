from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

from amuse.couple.bridge import CalculateFieldForParticles
from amuse.lab import constants, units, Particles, Particle


############################## TO WORK ON ##############################
# 1. AMUSIFY C++ LIBRARY WITH INTERFACE
# 2. RELEASE GIL IN C++ LIBRARY
# 3. GET_POTENTIAL_AT_POINT FUNCTION NOT USED --> TO VALIDATE
########################################################################

SI_UNITS = (1. | units.kg * units.m**-2.) * constants.G



        
def compute_gravity(grav_lib, pert_m, pert_x, pert_y, pert_z, particles: Particles, npert=None) -> tuple:
    """
    Compute gravitational force felt by perturber particles due to externals
    
    Args:
        grav_lib (library):  Library to compute gravity
        pert_m (units.mass):  Mass of perturber particles
        pert_x (units.length):  x-Cartesian coordinates of perturber particles
        pert_y (units.length):  y-Cartesian coordinates of perturber particles
        pert_z (units.length):  z-Cartesian coordinates of perturber particles
        particles (Particles):  Particles to compute gravity on
        npert (int):  Number of perturber particles
    Returns:
        tuple:  Acceleration array of particles (ax, ay, az)
    """
    def convert_array(array, units):
        return array.value_in(units).astype(np.float64)

    ### Won't work if particles < 2    
    num_particles = len(particles)
    
    # Convert positions to SI units
    if npert is not None:
        num_perturber = 1
        perturber_mass = np.array([pert_m.value_in(units.kg)], dtype=np.float64)
        perturber_x = np.array([pert_x.value_in(units.m)], dtype=np.float64)
        perturber_y = np.array([pert_y.value_in(units.m)], dtype=np.float64)
        perturber_z = np.array([pert_z.value_in(units.m)], dtype=np.float64)
    else:
        num_perturber = len(pert_m)
        perturber_mass = convert_array(pert_m, units.kg)
        perturber_x = convert_array(pert_x, units.m)
        perturber_y = convert_array(pert_y, units.m)
        perturber_z = convert_array(pert_z, units.m)
    
    particles_x = convert_array(particles.x, units.m)
    particles_y = convert_array(particles.y, units.m)
    particles_z = convert_array(particles.z, units.m)
    
    # Initialise acceleration arrays
    result_ax = np.zeros(num_particles, dtype=np.float64)
    result_ay = np.zeros(num_particles, dtype=np.float64)
    result_az = np.zeros(num_particles, dtype=np.float64)
    
    grav_lib.find_gravity_at_point(
        perturber_mass,
        perturber_x,
        perturber_y,
        perturber_z,
        particles_x,
        particles_y,
        particles_z,
        result_ax, 
        result_ay, 
        result_az,
        num_particles,
        num_perturber
    )
    return result_ax, result_ay, result_az

class CorrectionFromCompoundParticle(object):
    def __init__(self, grav_lib, particles: Particles, subsystems: Particles, num_of_workers: int):
        """
        Correct force exerted by some parent system on other particles by that of its system.
        
        Args:
            grav_lib (Library): The gravity library (e.g., a wrapped C++ library).
            particles (Particles):  Parent particles to correct force of
            subsystems (Particles):  Collection of subsystems present
            num_of_workers (int):  Number of cores to use
        """
        self.particles = particles
        self.subsystems = subsystems
        self.lib = grav_lib
        self.max_workers = num_of_workers
        self.acc_units = self.particles.vx.unit**2. / self.particles.x.unit
         
    def correct_parents(self, particles: Particles, parent_mass, parent_x, parent_y, parent_z,
                        system_mass, system_x, system_y, system_z, removed_idx: int) -> tuple:
        """
        Compute the differential acceleration on the parent system.
        
        Instead of copying and then removing one particle from the arrays,
        we form the “external_parents” set by logically ignoring the removed element.
        
        Args:
            particles (Particles):  All parent particles
            parent_mass (units.mass):  Mass of parent particles
            parent_x (units.length):  x-Cartesian coordinates of parent particles
            parent_y (units.length):  y-Cartesian coordinates of parent particles
            parent_z (units.length):  z-Cartesian coordinates of parent particles
            system_mass (units.mass):  Mass of system particles
            system_x (units.length):  x-Cartesian coordinates of system particles
            system_y (units.length):  y-Cartesian coordinates of system particles
            system_z (units.length):  z-Cartesian coordinates of system particles
            removed_idx (int):  Index of parent particle to remove
        Returns:
            tuple:  Acceleration array of parent particles (ax, ay, az)
        """
        external_parents = particles - particles[removed_idx]
        
        ax_chd, ay_chd, az_chd = compute_gravity(
                                    grav_lib=self.lib, 
                                    pert_m=system_mass, 
                                    pert_x=system_x+parent_x,
                                    pert_y=system_y+parent_y, 
                                    pert_z=system_z+parent_z, 
                                    particles=external_parents
                                    )
        
        ax_par, ay_par, az_par = compute_gravity(
                                    grav_lib=self.lib, 
                                    pert_m=parent_mass, 
                                    pert_x=parent_x, 
                                    pert_y=parent_y, 
                                    pert_z=parent_z, 
                                    particles=external_parents, 
                                    npert=1
                                    )

        corr_ax = (ax_chd - ax_par) * SI_UNITS
        corr_ay = (ay_chd - ay_par) * SI_UNITS
        corr_az = (az_chd - az_par) * SI_UNITS

        corr_ax = np.insert(corr_ax.value_in(self.acc_units).astype(np.float64), removed_idx, 0.)
        corr_ay = np.insert(corr_ay.value_in(self.acc_units).astype(np.float64), removed_idx, 0.)
        corr_az = np.insert(corr_az.value_in(self.acc_units).astype(np.float64), removed_idx, 0.)
        
        external_parents = None
        
        return (corr_ax | self.acc_units,
                corr_ay | self.acc_units,
                corr_az | self.acc_units)
    
    def get_gravity_at_point(self, radius, x, y, z) -> tuple:
        """
        Compute difference in gravitational acceleration felt by parents
        due to force exerted by parents which host system, and force
        exerted by their system.
        
        :math:`dF = \sum_{j} \left( \sum_{i} F_{i} - F_{j} \right)`
            
        where j is parent and i is system of parent j
        
        Args:
            radius (units.length):  Radius of parent particles
            x (units.length):  x-Cartesian coordinates of parent particles
            y (units.length):  z-Cartesian coordinates of parent particles
            z (units.length):  y-Cartesian coordinates of parent particles
        Returns:
            tuple:  Acceleration array of parent particles (ax, ay, az)
        """
        Nparticles = len(self.particles)
        particles = Particles(Nparticles)
        particles.position = self.particles.position
        
        ax_corr = np.zeros(Nparticles) | self.acc_units
        ay_corr = np.zeros(Nparticles) | self.acc_units
        az_corr = np.zeros(Nparticles) | self.acc_units

        parent_idx = {parent.key: i for i, parent in enumerate(self.particles)}
        futures = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            for parent, system in list(self.subsystems.items()):
                removed_idx = parent_idx.pop(parent.key)
                
                ### Strip relevant properties. Copy leads to leak -> remove_particles() consumes time.
                parent_mass = parent.mass
                parent_x = parent.x
                parent_y = parent.y
                parent_z = parent.z
                
                system_mass = system.mass
                system_pos = system.position + parent.position
                system_x = system_pos.x
                system_y = system_pos.y
                system_z = system_pos.z

                future = executor.submit(
                    self.correct_parents,
                    particles=particles,
                    parent_mass=parent_mass,
                    parent_x=parent_x,
                    parent_y=parent_y,
                    parent_z=parent_z,
                    system_mass=system_mass,
                    system_x=system_x,
                    system_y=system_y,
                    system_z=system_z,
                    removed_idx=removed_idx
                    )
                futures.append(future)

            # Retrieve results and release memory
            for future in as_completed(futures):
                ax, ay, az = future.result()
                ax_corr += ax
                ay_corr += ay
                az_corr += az
        
        ### Clean-up and release memory   
        if Nparticles > 1:
            particles.remove_particles(particles)
        else:
            particles.remove_particle(particles)
        
        return ax_corr, ay_corr, az_corr

    def get_potential_at_point(self, radius, x, y, z) -> np.ndarray:
        """
        Get the potential at a specific location
        
        Args:
            radius (units.length):  Radius of the particle at that location
            x (units.length):  x-Cartesian coordinates of the location
            y (units.length):  y-Cartesian coordinates of the location
            z (units.length):  z-Cartesian coordinates of the location
        Returns:
            Array:  The potential field at the location
        """
        particles = self.particles.copy()
        particles.phi = 0. | (particles.vx.unit**2)
        for parent, sys in list(self.subsystems.items()): 
            copied_system = sys.copy()
            copied_system.position += parent.position
            copied_system.velocity += parent.velocity
            
            code = CalculateFieldForParticles(gravity_constant=constants.G)
            code.particles.add_particles(copied_system)
            
            parts = particles - parent
            phi = code.get_potential_at_point(0.*parts.radius, 
                                              parts.x, 
                                              parts.y, 
                                              parts.z)
            parts.phi += phi
            code.cleanup_code()
            
            code = CalculateFieldForParticles(gravity_constant=constants.G)
            code.particles.add_particle(parent)
            phi = code.get_potential_at_point(0.*parts.radius, 
                                              parts.x, 
                                              parts.y, 
                                              parts.z)
            parts.phi -= phi
            code.cleanup_code()
            
        return particles.phi
    
  
class CorrectionForCompoundParticle(object):  
    def __init__(self, grav_lib, parent_copy: Particle, system: Particles,
                 perturber_mass, perturber_x, perturber_y, perturber_z):
        """Correct force vector exerted by global particles on systems
        
        Args:
            grav_lib (Library): The gravity library (e.g., a wrapped C++ library).
            parent_copy (Particle): A copy of the parent particle.
            system (Particles): The subsystem particles.
            perturber_mass (units.mass): Mass of the perturber particle.
            perturber_x (units.length): x coordinate of the perturber particle.
            perturber_y (units.length): y coordinate of the perturber particle.
            perturber_z (units.length): z coordinate of the perturber particle.
        """
        self.lib = grav_lib
        self.parent_copy = parent_copy
        self.system = system
        
        self.pert_mass = perturber_mass
        self.pert_x = perturber_x
        self.pert_y = perturber_y
        self.pert_z = perturber_z
        
        self.acc_units = system.vx.unit**2. / system.x.unit

    def get_gravity_at_point(self, radius, x, y, z) -> tuple:
        """
        Compute gravitational acceleration felt by system due to parents present.
        
        Args:
            radius (units.length):  Radius of the system particle
            x (units.length):  x coordinate of the system particle
            y (units.length):  y coordinate of the system particle
            z (units.length):  z coordinate of the system particle
        Returns: 
            tuple:  Acceleration array of system particles (ax, ay, az)
        """
        Nsystem = len(self.system)
        corr_ax = np.zeros(Nsystem) | self.acc_units
        corr_ay = np.zeros(Nsystem) | self.acc_units
        corr_az = np.zeros(Nsystem) | self.acc_units
        
        ax_chd, ay_chd, az_chd = compute_gravity(
                                    grav_lib=self.lib, 
                                    pert_m=self.pert_mass, 
                                    pert_x=self.pert_x,
                                    pert_y=self.pert_y,
                                    pert_z=self.pert_z,
                                    particles=self.system
                                    )
        ax_par, ay_par, az_par = compute_gravity(
                                    grav_lib=self.lib, 
                                    pert_m=self.pert_mass, 
                                    pert_x=self.pert_x,
                                    pert_y=self.pert_y,
                                    pert_z=self.pert_z,
                                    particles=self.parent_copy
                                    )
        
        corr_ax += (ax_chd - ax_par) * SI_UNITS
        corr_ay += (ay_chd - ay_par) * SI_UNITS
        corr_az += (az_chd - az_par) * SI_UNITS
        
        return corr_ax, corr_ay, corr_az
    
    def get_potential_at_point(self, radius, x, y, z) -> np.ndarray:
        """
        Get the potential at a specific location
        
        Args:
            radius (units.length):  Radius of the system particle
            x (units.length):  x Location of the system particle
            y (units.length):  y Location of the system particle
            z (units.length):  z Location of the system particle
        Returns:
            Array:  The potential field at the system particle's location
        """
        instance = CalculateFieldForParticles(gravity_constant=constants.G)
        instance.particles.add_particles(self.system)
        phi = instance.get_potential_at_point(0.*radius,
                                              self.parent.x + x,
                                              self.parent.y + y,
                                              self.parent.z + z)
        _phi = instance.get_potential_at_point([0.*self.parent.radius],
                                               [self.parent.x],
                                               [self.parent.y],
                                               [self.parent.z])
        instance.cleanup_code()
        
        return (phi-_phi[0])