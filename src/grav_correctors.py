import ctypes
import numpy as np

from amuse.couple.bridge import CalculateFieldForParticles
from amuse.lab import constants, units, Particles


def load_gravity_library():
    """Setup library to allow Python and C++ communication"""
    lib = ctypes.CDLL('./src/gravity.so')  # Adjust the path as necessary
    lib.find_gravity_at_point.argtypes = [
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),  # perturbing mass
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),  # recipient xpos
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),  # recipient ypos
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),  # recipient zpos
        ctypes.c_int,  # num_particles
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),  # ax array
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),  # ay array
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),  # az array
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),  # perturbing xpos
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),  # perturbing ypos
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),  # perturbing zpos
        ctypes.c_int   # num_points
    ]
    lib.find_gravity_at_point.restype = None
    return lib

def system_to_cluster_frame(system, parent):
    """
    Shift children to parent reference frame
    
    Args:
        system:  Particle set to convert
        parent:  Parent particle to convert to
    Returns:
        Particle set with phase-space coordinates shifted in parent reference frame
    """
    system = system.copy_to_memory()
    system.position += parent.position
    return system

def compute_gravity(grav_lib, perturber, particles):
    """
    Compute gravitational force felt by perturber particles due to externals
    
    Args:
        grav_lib (library):       Library to compute gravity
        perturber (ParticleSet):  Set of perturbing particles
        particles (ParticleSet):  Set of particles feeling force
    """
    num_particles = len(particles)
    num_perturber = len(perturber)

    result_ax = np.zeros(num_particles)
    result_ay = np.zeros(num_particles)
    result_az = np.zeros(num_particles)
    
    grav_lib.find_gravity_at_point(
        perturber.mass.value_in(units.kg),
        particles.x.value_in(units.m),
        particles.y.value_in(units.m),
        particles.z.value_in(units.m),
        num_particles,
        result_ax, result_ay, result_az,
        perturber.x.value_in(units.m),
        perturber.y.value_in(units.m),
        perturber.z.value_in(units.m),
        num_perturber
    )
    
    return result_ax, result_ay, result_az
    
class CorrectionFromCompoundParticle(object):
    def __init__(self, particles, subsystems):
        """Correct force exerted by some parent system on 
        other particles by that of its children.
        
        Args:
            particles (ParticleSet):   Parent particles to correct force of
            subsystems (ParticleSet):  Collection of subsystems present
        """
        self.particles = particles
        self.subsystems = subsystems
    
    def get_gravity_at_point(self, radius, x, y, z):
        """
        Compute difference in gravitational acceleration felt by parents
        due to force exerted by parents which host children, and force
        exerted by their children.
        dF = \sum_j (\sum_i F_{i} - F_{j}) where j is parent and i is children of parent j
        
        Args:
            radius (Float):  Radius of parent particles
            x (Float):       x-Cartesian coordinates of parent particles
            y (Float):       z-Cartesian coordinates of parent particles
            z (Float):       y-Cartesian coordinates of parent particles
        Returns:
            Float: Corrected acceleration for affected particles
        """
        particles = self.particles.copy_to_memory()
        acc_units = (particles.vx.unit**2/particles.x.unit)
        
        particles.ax = 0. | acc_units
        particles.ay = 0. | acc_units
        particles.az = 0. | acc_units
        
        lib = load_gravity_library()
        for parent, sys in list(self.subsystems.items()):
            system = sys.copy_to_memory()
            system = system_to_cluster_frame(system, parent)
            
            temp_par = Particles()
            temp_par.add_particle(parent)
            
            parts = particles - parent
            ax_chd, ay_chd, az_chd = compute_gravity(lib, system, parts)
            ax_par, ay_par, az_par = compute_gravity(lib, temp_par, parts)
            
            parts.ax += (ax_chd - ax_par) * (1. | units.kg*units.m**-2) * constants.G
            parts.ay += (ay_chd - ay_par) * (1. | units.kg*units.m**-2) * constants.G
            parts.az += (az_chd - az_par) * (1. | units.kg*units.m**-2) * constants.G
            
        return particles.ax, particles.ay, particles.az
    
    def get_potential_at_point(self, radius, x, y, z):
        """
        Get the potential at a specific location
        
        Args:
            radius (Float):  Radius of the particle at that location
            x (Float):       x-Cartesian coordinates of the location
            y (Float):       y-Cartesian coordinates of the location
            z (Float):       z-Cartesian coordinates of the location
        Returns:
            Float: The potential field at the location
        """
        particles = self.particles.copy_to_memory()
        particles.phi = 0. | (particles.vx.unit**2)
        for parent,sys in list(self.subsystems.items()): 
            code = CalculateFieldForParticles(gravity_constant=constants.G)
            code.particles.add_particles(sys.copy())
            code.particles.position += parent.position
            code.particles.velocity += parent.velocity
            
            parts = particles - parent
            phi = code.get_potential_at_point(0.*parts.radius, 
                                              parts.x, 
                                              parts.y, 
                                              parts.z
                                              )
            parts.phi += phi
            code.cleanup_code()
            
            code = CalculateFieldForParticles(gravity_constant=constants.G)
            code.particles.add_particle(parent)
            phi = code.get_potential_at_point(0.*parts.radius,
                                             parts.x,
                                             parts.y,
                                             parts.z
                                             )
            parts.phi -= phi
            code.cleanup_code()
            
        return particles.phi
    
  
class CorrectionForCompoundParticle(object):  
    def __init__(self, particles, parent, system):
        """Correct force vector exerted by global particles on childrens
        
        Args:
            particles (ParticleSet):  All parent particles
            parent (Particle):        Subsystem's parent particle
            system (ParticleSet):     Subsystem particle set
        """
        self.particles = particles
        self.parent = parent
        self.system = system
    
    def get_gravity_at_point(self, radius, x, y, z):
        """Compute gravitational acceleration felt by children due to parents present.
        dF_j = F_{i} - F_{j} where i is parent and j is children of parent i
        
        Args:
            radius (Float):  Radius of the children particle
            x (Float):       x Location of the children particle
            y (Float):       y Location of the children particle
            z (Float):       z Location of the children particle
        Returns: 
            Float: Gravitational acceleration felt by children particles
        """
        parent = self.parent
        parts = self.particles - parent
        parts = parts[parts.mass != (0 | units.kg)]  # Test particles exert no force
        
        system = self.system.copy_to_memory()
        system = system_to_cluster_frame(system, parent)
        acc_units = (system.vx.unit**2/system.x.unit)
        
        system.ax = 0. | acc_units
        system.ay = 0. | acc_units
        system.az = 0. | acc_units
        
        lib = load_gravity_library()
        
        temp_par = Particles()
        temp_par.add_particle(parent)
        ax_chd, ay_chd, az_chd = compute_gravity(lib, parts, system)
        ax_par, ay_par, az_par = compute_gravity(lib, parts, temp_par)
        
        system.ax += (ax_chd - ax_par) * (1 | units.kg*units.m**-2) * constants.G
        system.ay += (ay_chd - ay_par) * (1 | units.kg*units.m**-2) * constants.G
        system.az += (az_chd - az_par) * (1 | units.kg*units.m**-2) * constants.G
        
        return system.ax, system.ay, system.az
    
    def get_potential_at_point(self, radius, x, y, z):
        """
        Get the potential at a specific location
        
        Args:
            radius (Float):  Radius of the children particle
            x (Float):       x Location of the children particle
            y (Float):       y Location of the children particle
            z (Float):       z Location of the children particle
        Returns:
            Float: The potential field at the children particle's location
        """
        parent = self.parent
        parts = self.system - parent
        instance = CalculateFieldForParticles(gravity_constant=constants.G)
        instance.particles.add_particles(parts)
        phi = instance.get_potential_at_point(0.*radius,
                                              parent.x + x,
                                              parent.y + y,
                                              parent.z + z
                                              )
        _phi = instance.get_potential_at_point([0.*parent.radius],
                                               [parent.x],
                                               [parent.y],
                                               [parent.z]
                                               )
        instance.cleanup_code()
        return (phi-_phi[0])