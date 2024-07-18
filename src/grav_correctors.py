import ctypes
import numpy as np

from amuse.couple.bridge import CalculateFieldForParticles
from amuse.lab import constants, units


def load_gravity_library():
    """Setup library to allow Python and C++ communication"""
    lib = ctypes.CDLL('./src/gravity.so')  # Adjust the path as necessary
    lib.find_gravity_at_point.argtypes = [
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),  # Target mass
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),  # Target x
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),  # Target y
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),  # Target z
        ctypes.c_int,  # num_particles
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),  # Others ax
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),  # Others ay
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),  # Others az
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),  # Target x
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),  # Target y
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),  # Target z
        ctypes.c_int   # num_points
    ]
    lib.find_gravity_at_point.restype = None
    return lib

def system_to_cluster_frame(system, parent):
    """Shift system particles coordinates to the parent"""
    system = system.copy_to_memory()
    system.position += parent.position
    return system

def compute_gravity(grav_lib, perturbers, particles, mode):
    """Compute gravitational force felt by perturbers particle due to externals
    Inputs:
    grav_lib: Library communicating between Python and C++
    perturbers:  Set of perturbing particles
    particles:  Set of particles feeling force
    mode: Mode for array configuration 
          mode = 0: Multiple perturbers and particles
          mode = 1: One perturbers and multiple particles
          mode = 2: Multiple perturbers and one particle
    """
    if mode == 0:
        result_ax = np.zeros(len(particles))
        result_ay = np.zeros(len(particles))
        result_az = np.zeros(len(particles))
        
        grav_lib.find_gravity_at_point(
            perturbers.mass.value_in(units.kg),
            particles.x.value_in(units.m),
            particles.y.value_in(units.m),
            particles.z.value_in(units.m),
            len(particles),
            result_ax, result_ay, result_az,
            perturbers.x.value_in(units.m),
            perturbers.y.value_in(units.m),
            perturbers.z.value_in(units.m),
            len(perturbers)
        )
        
    elif mode == 1:
        result_ax = np.zeros(len(particles))
        result_ay = np.zeros(len(particles))
        result_az = np.zeros(len(particles))
        
        grav_lib.find_gravity_at_point(
            np.array([perturbers.mass.value_in(units.kg)]),
            particles.x.value_in(units.m),
            particles.y.value_in(units.m),
            particles.z.value_in(units.m),
            len(particles),
            result_ax, result_ay, result_az,
            np.array([perturbers.x.value_in(units.m)]),
            np.array([perturbers.y.value_in(units.m)]),
            np.array([perturbers.z.value_in(units.m)]),
            1
        )
    else:
        result_ax = np.zeros(1)
        result_ay = np.zeros(1)
        result_az = np.zeros(1)
        
        grav_lib.find_gravity_at_point(
            perturbers.mass.value_in(units.kg),
            np.asarray([particles.x.value_in(units.m)]),
            np.asarray([particles.y.value_in(units.m)]),
            np.asarray([particles.z.value_in(units.m)]),
            1,
            result_ax, result_ay, result_az,
            perturbers.x.value_in(units.m),
            perturbers.y.value_in(units.m),
            perturbers.z.value_in(units.m),
            len(perturbers)
        )
    return result_ax, result_ay, result_az
    
class CorrectionFromCompoundParticle(object):
    def __init__(self, particles, subsystems):
        """Correct force vector exerted by parents on all 
        other particles present by that of its children.
        Inputs:
        particles:  Parent particles to correct force of
        subsystems:  Collection of subsystems present
        """
        self.particles = particles
        self.subsystems = subsystems
    
    def get_gravity_at_point(self, radius, x, y, z):
        """Compute difference in gravitational acceleration felt by parents
        due to force exerted by parents hosting children, and their children.
        dF = \sum_j (\sum_i F_{i} - F_{j}) where j is parent and i is children of parent j
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
            
            parts = particles - parent
            ax_chd, ay_chd, az_chd = compute_gravity(lib, system, parts, mode=0)
            ax_par, ay_par, az_par = compute_gravity(lib, parent, parts, mode=1)
            
            parts.ax += (ax_chd - ax_par) * (1 | units.kg*units.m**-2) * constants.G
            parts.ay += (ay_chd - ay_par) * (1 | units.kg*units.m**-2) * constants.G
            parts.az += (az_chd - az_par) * (1 | units.kg*units.m**-2) * constants.G
            
        return particles.ax, particles.ay, particles.az
    
    def get_potential_at_point(self, radius, x, y, z):
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
        Inputs:
        particles:  All parent particles
        parent:  Subsystem's parent particle
        system:  Subsystem particle set
        """
        self.particles = particles
        self.parent = parent
        self.system = system
    
    def get_gravity_at_point(self, radius, x, y, z):
        """Compute gravitational acceleration felt by children via 
        all other parents present in the simulation.
        dF_j = F_{i} - F_{j} where i is parent and j is children of parent i
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
        ax_chd, ay_chd, az_chd = compute_gravity(lib, parts, system, mode=0)
        ax_par, ay_par, az_par = compute_gravity(lib, parts, parent, mode=2)
        
        system.ax += (ax_chd - ax_par) * (1 | units.kg*units.m**-2) * constants.G
        system.ay += (ay_chd - ay_par) * (1 | units.kg*units.m**-2) * constants.G
        system.az += (az_chd - az_par) * (1 | units.kg*units.m**-2) * constants.G
        
        return system.ax, system.ay, system.az
      
    def get_potential_at_point(self,radius,x,y,z):
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