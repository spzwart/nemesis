from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import gc

from amuse.couple.bridge import CalculateFieldForParticles
from amuse.lab import constants, units, Particles, Particle


############################## TO WORK ON ##############################
# 1. AMUSIFY C++ LIBRARY WITH INTERFACE
# 2. RELEASE GIL IN C++ LIBRARY
# 3. GET_POTENTIAL_AT_POINT FUNCTION NOT USED --> TO VALIDATE
########################################################################



def compute_gravity(grav_lib, perturber: Particles, particles: Particles) -> tuple:
    """
    Compute gravitational force felt by perturber particles due to externals
    
    Args:
        grav_lib (library):  Library to compute gravity
        perturber (Particles):  Set of perturbing particles
        particles (Particles):  Set of particles feeling force
    Returns:
        tuple:  Acceleration array of particles (ax, ay, az)
    """
    num_particles = len(particles)
    num_perturber = len(perturber)
    
    # Convert positions to SI units
    perturber_mass = perturber.mass.value_in(units.kg).astype(np.float64)
    perturber_x = perturber.x.value_in(units.m).astype(np.float64)
    perturber_y = perturber.y.value_in(units.m).astype(np.float64)
    perturber_z = perturber.z.value_in(units.m).astype(np.float64)
    particles_x = particles.x.value_in(units.m).astype(np.float64)
    particles_y = particles.y.value_in(units.m).astype(np.float64)
    particles_z = particles.z.value_in(units.m).astype(np.float64)
    
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
        result_ax, result_ay, result_az,
        num_particles,
        num_perturber
    )
    return result_ax, result_ay, result_az
    
class CorrectionFromCompoundParticle(object):
    def __init__(self, particles: Particles, subsystems: Particles, library, max_workers: int):
        """
        Correct force exerted by some parent system on 
        other particles by that of its children.
        
        Args:
            particles (Particles):  Parent particles to correct force of
            subsystems (Particles):  Collection of subsystems present
            library (Library):  Python to C++ communication library
            max_workers (int):  Number of cores to use
        """
        self.particles = particles
        self.subsystems = subsystems
        self.lib = library
        self.max_workers = max_workers
        self.acc_units = self.particles.vx.unit**2. / self.particles.x.unit
        
        # Convert acceleration arrays to SI units
        self.SI_units = (1. | units.kg*units.m**-2.) * constants.G
        
    def correct_parents(self, particles: Particles, parent_copy: Particle, system: Particles, removed_idx: int) -> tuple:
        """
        Compute difference in gravitational acceleration exerted onto parents
        
        Args:
            particles (Particles):  All parent particles
            parent_copy (Particles):  Copy of parent particle
            system (Particles):  Copy of selected parent's children
            removed_idx (int):  Index of parent particle
        Returns:
            tuple:  Acceleration array of parent particles (ax, ay, az)
        """
        external_parents = particles - particles[removed_idx]

        ax_chd, ay_chd, az_chd = compute_gravity(self.lib, system, external_parents)
        ax_par, ay_par, az_par = compute_gravity(self.lib, parent_copy, external_parents)

        corr_ax = (ax_chd - ax_par) * self.SI_units
        corr_ay = (ay_chd - ay_par) * self.SI_units
        corr_az = (az_chd - az_par) * self.SI_units

        corr_ax = np.insert(corr_ax.value_in(self.acc_units).astype(np.float64), removed_idx, 0.)
        corr_ay = np.insert(corr_ay.value_in(self.acc_units).astype(np.float64), removed_idx, 0.)
        corr_az = np.insert(corr_az.value_in(self.acc_units).astype(np.float64), removed_idx, 0.)

        del external_parents
        del ax_chd, ay_chd, az_chd
        del ax_par, ay_par, az_par

        return (corr_ax | self.acc_units,
                corr_ay | self.acc_units,
                corr_az | self.acc_units)

    
    def get_gravity_at_point(self, radius, x, y, z) -> tuple:
        """
        Compute difference in gravitational acceleration felt by parents
        due to force exerted by parents which host children, and force
        exerted by their children.
        
        :math:`dF = \sum_{j} \left( \sum_{i} F_{i} - F_{j} \right)`
            
        where j is parent and i is children of parent j
        
        Args:
            radius (units.length):  Radius of parent particles
            x (units.length):  x-Cartesian coordinates of parent particles
            y (units.length):  z-Cartesian coordinates of parent particles
            z (units.length):  y-Cartesian coordinates of parent particles
        Returns:
            tuple:  Acceleration array of parent particles (ax, ay, az)
        """
        particles = self.particles.copy()
        Nparticles = len(particles)
        
        ax_corr = np.zeros(Nparticles) | self.acc_units
        ay_corr = np.zeros(Nparticles) | self.acc_units
        az_corr = np.zeros(Nparticles) | self.acc_units

        parent_idx = {parent.key: i for i, parent in enumerate(self.particles)}
        futures = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            for parent, children in list(self.subsystems.items()):
                removed_idx = parent_idx[parent.key]
                
                # Don't do copy() otherwise lingering references and memory leak
                children_copy = Particles(len(children))
                children_copy.mass = children.mass
                children_copy.position = children.position + parent.position
                
                parent_copy = Particles(1)
                parent_copy.mass = parent.mass
                parent_copy.position = parent.position

                future = executor.submit(self.correct_parents,
                                         particles,
                                         parent_copy,
                                         children_copy,
                                         removed_idx)
                futures.append(future)

                del parent_copy, children_copy

            # Retrieve and accumulate results.
            for future in as_completed(futures):
                ax, ay, az = future.result()
                ax_corr += ax
                ay_corr += ay
                az_corr += az

        futures.clear()
        del futures, particles
        executor.shutdown(wait=True)
        gc.collect()

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
            copied_children = sys.copy()
            copied_children.position += parent.position
            copied_children.velocity += parent.velocity
            
            code = CalculateFieldForParticles(gravity_constant=constants.G)
            code.particles.add_particles(copied_children)
            
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
    def __init__(self, particles: Particles, parent: Particle, system: Particles, library):
        """Correct force vector exerted by global particles on childrens
        
        Args:
            particles (Particles):  All parent particles
            parent (Particle):  Subsystem's parent particle
            system (Particles):  Subsystem Particles
            library (Library):  Python to C++ communication library
        """
        self.particles = particles
        self.parent = parent
        self.system = system
        self.lib = library
        self.SI_units = (1. | units.kg * units.m**-2.) * constants.G
        self.acc_units = system.vx.unit**2. / system.x.unit

    def get_gravity_at_point(self, radius, x, y, z) -> float:
        """
        Compute gravitational acceleration felt by children due to parents present.

        :math:`dF = \sum_{j} \left( \sum_{i} F_{i} - F_{j} \right)`

        where i is parent and j is children of parent i
        
        Args:
            radius (units.length):  Radius of the children particle
            x (units.length):  x Location of the children particle
            y (units.length):  y Location of the children particle
            z (units.length):  z Location of the children particle
        Returns: 
            tuple:  Acceleration array of children particles (ax, ay, az)
        """
        parts = self.particles - self.parent
        system_copy = self.system.copy()
        system_copy.ax = 0. | self.acc_units
        system_copy.ay = 0. | self.acc_units
        system_copy.az = 0. | self.acc_units
        
        parent_copy = Particles(particles=[self.parent])
        system_copy.position += self.parent.position
        
        ax_chd, ay_chd, az_chd = compute_gravity(self.lib, parts, system_copy)
        ax_par, ay_par, az_par = compute_gravity(self.lib, parts, parent_copy)
        
        system_copy.ax += (ax_chd - ax_par) * self.SI_units
        system_copy.ay += (ay_chd - ay_par) * self.SI_units
        system_copy.az += (az_chd - az_par) * self.SI_units
        
        return system_copy.ax, system_copy.ay, system_copy.az
    
    def get_potential_at_point(self, radius, x, y, z) -> np.ndarray:
        """
        Get the potential at a specific location
        
        Args:
            radius (units.length):  Radius of the children particle
            x (units.length):  x Location of the children particle
            y (units.length):  y Location of the children particle
            z (units.length):  z Location of the children particle
        Returns:
            Array:  The potential field at the children particle's location
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