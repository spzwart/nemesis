from concurrent.futures import ThreadPoolExecutor
import numpy as np

from amuse.couple.bridge import CalculateFieldForParticles
from amuse.lab import constants, units, Particles


def compute_gravity(grav_lib, perturber, particles) -> float:
    """
    Compute gravitational force felt by perturber particles due to externals
    
    Args:
        grav_lib (library):  Library to compute gravity
        perturber (particle set):  Set of perturbing particles
        particles (particle set):  Set of particles feeling force
    Returns:
        Array:  Acceleration array of particles
    """
    num_particles = len(particles)
    num_perturber = len(perturber)

    result_ax = np.zeros(num_particles, dtype=np.float128)
    result_ay = np.zeros(num_particles, dtype=np.float128)
    result_az = np.zeros(num_particles, dtype=np.float128)
    grav_lib.find_gravity_at_point(
        perturber.mass.value_in(units.kg).astype(np.float128),
        particles.x.value_in(units.m).astype(np.float128),
        particles.y.value_in(units.m).astype(np.float128),
        particles.z.value_in(units.m).astype(np.float128),
        perturber.x.value_in(units.m).astype(np.float128),
        perturber.y.value_in(units.m).astype(np.float128),
        perturber.z.value_in(units.m).astype(np.float128),
        result_ax, result_ay, result_az,
        num_particles,
        num_perturber
    )
    return result_ax, result_ay, result_az
    
class CorrectionFromCompoundParticle(object):
    def __init__(self, particles, subsystems, library, max_workers):
        """
        Correct force exerted by some parent system on 
        other particles by that of its children.
        
        Args:
            particles (particle set):  Parent particles to correct force of
            subsystems (particle set):  Collection of subsystems present
            library (Library):  Python to C++ communication library
            max_workers (int):  Number of cores to use
        """
        self.particles = particles
        self.acc_units = (self.particles.vx.unit**2./self.particles.x.unit)
        self.subsystems = subsystems
        self.lib = library
        self.max_workers = max_workers
        self.SI_units = (1. | units.kg*units.m**-2.) * constants.G
        
    def correct_parents(self, particles, parent_copy, system, removed_idx) -> None:
        """
        Compute difference in gravitational acceleration exerted onto parents
        
        Args:
            particles (particle set):  All parent particles
            parent_copy (particle set):  Copy of parent particle
            system (particle set):  Copy of selected parent's children
            removed_idx (int):  Index of parent particle
        """
        ax = ay = az = 0. | self.acc_units
        parts = particles - parent_copy
        
        ax_chd, ay_chd, az_chd = compute_gravity(self.lib, system, parts)
        ax_par, ay_par, az_par = compute_gravity(self.lib, parent_copy, parts)
        
        ax += (ax_chd - ax_par) * self.SI_units
        ay += (ay_chd - ay_par) * self.SI_units
        az += (az_chd - az_par) * self.SI_units
        
        ax = np.insert(ax.value_in(self.acc_units).astype(np.float128), removed_idx, 0.)
        ay = np.insert(ay.value_in(self.acc_units).astype(np.float128), removed_idx, 0.)
        az = np.insert(az.value_in(self.acc_units).astype(np.float128), removed_idx, 0.)
        
        ax = ax | self.acc_units
        ay = ay | self.acc_units
        az = az | self.acc_units
            
        return [ax, ay, az]
    
    def get_gravity_at_point(self, radius, x, y, z) -> float:
        """
        Compute difference in gravitational acceleration felt by parents
        due to force exerted by parents which host children, and force
        exerted by their children.
        .. math:: 
            dF = \sum_j (\sum_i F_{i} - F_{j}) 
        where j is parent and i is children of parent j
        
        Args:
            radius (Float):  Radius of parent particles
            x (Float):  x-Cartesian coordinates of parent particles
            y (Float):  z-Cartesian coordinates of parent particles
            z (Float):  y-Cartesian coordinates of parent particles
        Returns:
            Array:  Acceleration array of parent particles
        """
        particles = self.particles.copy_to_memory()
        particles.ax = 0. | self.acc_units
        particles.ay = 0. | self.acc_units
        particles.az = 0. | self.acc_units
        
        parent_idx = {parent.key: i for i, parent in enumerate(particles)}
        futures = [ ]
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            for parent, sys in list(self.subsystems.items()):
                copied_child = sys.copy()

                removed_idx = parent_idx[parent.key]
                copied_child.position += parent.position

                future = executor.submit(self.correct_parents,
                                         particles,
                                         Particles(particles=[parent]),
                                         copied_child,
                                         removed_idx)
                futures.append(future)

            for future in futures:
                ax, ay, az = future.result()
                particles.ax += ax
                particles.ay += ay
                particles.az += az

        return particles.ax, particles.ay, particles.az

    def get_potential_at_point(self, radius, x, y, z) -> np.ndarray:
        """
        Get the potential at a specific location
        
        Args:
            radius (Float):  Radius of the particle at that location
            x (Float):  x-Cartesian coordinates of the location
            y (Float):  y-Cartesian coordinates of the location
            z (Float):  z-Cartesian coordinates of the location
        Returns:
            Array:  The potential field at the location
        """
        particles = self.particles.copy_to_memory()
        particles.phi = 0. | (particles.vx.unit**2)
        for parent, sys in list(self.subsystems.items()): 
            code = CalculateFieldForParticles(gravity_constant=constants.G)
            code.particles.add_particles(sys.copy())
            code.particles.position += parent.position
            code.particles.velocity += parent.velocity
            
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
    def __init__(self, particles, parent, system, library):
        """Correct force vector exerted by global particles on childrens
        
        Args:
            particles (particle set):  All parent particles
            parent (particle):  Subsystem's parent particle
            system (particle set):  Subsystem particle set
            library (Library):  Python to C++ communication library
        """
        self.particles = particles
        self.parent = parent
        self.system = system
        self.lib = library
        self.SI_units = (1. | units.kg * units.m**-2.) * constants.G
        self.acc_units = (system.vx.unit**2 / system.x.unit)

    def get_gravity_at_point(self, radius, x, y, z) -> float:
        """
        Compute gravitational acceleration felt by children due to parents present.
        .. math::
            dF_j = F_{i} - F_{j} 
        where i is parent and j is children of parent i
        
        Args:
            radius (Float):  Radius of the children particle
            x (Float):  x Location of the children particle
            y (Float):  y Location of the children particle
            z (Float):  z Location of the children particle
        Returns: 
            Array:  Acceleration array of children particles
        """
        parts = self.particles - self.parent
        system = self.system.copy_to_memory()
        
        system.ax = 0. | self.acc_units
        system.ay = 0. | self.acc_units
        system.az = 0. | self.acc_units
        
        parent_copy = Particles(particles=[self.parent])
        system.position += self.parent.position
        
        ax_chd, ay_chd, az_chd = compute_gravity(self.lib, parts, system)
        ax_par, ay_par, az_par = compute_gravity(self.lib, parts, parent_copy)
        
        system.ax += (ax_chd - ax_par) * self.SI_units
        system.ay += (ay_chd - ay_par) * self.SI_units
        system.az += (az_chd - az_par) * self.SI_units
        
        return system.ax, system.ay, system.az
    
    def get_potential_at_point(self, radius, x, y, z) -> np.ndarray:
        """
        Get the potential at a specific location
        
        Args:
            radius (Float):  Radius of the children particle
            x (Float):  x Location of the children particle
            y (Float):  y Location of the children particle
            z (Float):  z Location of the children particle
        Returns:
            Array:  The potential field at the children particle's location
        """
        parent = self.parent
        system = self.system.copy_to_memory()
        system.position += parent.position
        parts = self.system
        
        instance = CalculateFieldForParticles(gravity_constant=constants.G)
        instance.particles.add_particles(parts)
        phi = instance.get_potential_at_point(0.*radius,
                                              parent.x + x,
                                              parent.y + y,
                                              parent.z + z)
        _phi = instance.get_potential_at_point([0.*parent.radius],
                                               [parent.x],
                                               [parent.y],
                                               [parent.z])
        instance.cleanup_code()
        
        return (phi-_phi[0])