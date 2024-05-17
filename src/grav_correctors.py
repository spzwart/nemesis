import ctypes
import numpy as np

from amuse.lab import constants, units


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
        
        # Load the shared library
        lib = ctypes.CDLL('./src/gravity.so')  # Adjust the path as necessary
        lib.find_gravity_at_point.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),  # comp_mass
            np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),  # comp_x
            np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),  # comp_y
            np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),  # comp_z
            ctypes.c_int,  # num_particles
            np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),  # res_ax
            np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),  # res_ay
            np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),  # res_az
            np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),  # int_x
            np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),  # int_y
            np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),  # int_z
            ctypes.c_int   # num_points
        ]
        
        lib.find_gravity_at_point.restype = None
        for parent, sys in list(self.subsystems.items()):
            system = sys.copy_to_memory()
            system.position += parent.position
            
            parts = particles - parent
            result_ax = np.zeros(len(parts))
            result_ay = np.zeros(len(parts))
            result_az = np.zeros(len(parts))
            
            lib.find_gravity_at_point(
                system.mass.value_in(units.kg),
                parts.x.value_in(units.m),
                parts.y.value_in(units.m),
                parts.z.value_in(units.m),
                len(parts),
                result_ax, result_ay, result_az,
                system.x.value_in(units.m),
                system.y.value_in(units.m),
                system.z.value_in(units.m),
                len(system)
                )
            
            result_axm = np.zeros(len(parts))
            result_aym = np.zeros(len(parts))
            result_azm = np.zeros(len(parts))
            
            lib.find_gravity_at_point(
                np.array([parent.mass.value_in(units.kg)]),
                parts.x.value_in(units.m),
                parts.y.value_in(units.m),
                parts.z.value_in(units.m),
                len(parts),
                result_axm, result_aym, result_azm,
                np.array([parent.x.value_in(units.m)]),
                np.array([parent.y.value_in(units.m)]),
                np.array([parent.z.value_in(units.m)]),
                1
                )
            
            parts.ax += (result_ax - result_axm) * (1 | units.kg*units.m**-2) * constants.G
            parts.ay += (result_ay - result_aym) * (1 | units.kg*units.m**-2) * constants.G
            parts.az += (result_az - result_azm) * (1 | units.kg*units.m**-2) * constants.G
        return particles.ax, particles.ay, particles.az
  
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
    
    def get_gravity_at_point(self,radius,x,y,z):
        """Compute gravitational acceleration felt by children via 
        all other parents present in the simulation.
        dF_j = F_{i} - F_{j} where i is parent and j is children of parent i
        """
        parent = self.parent
        parts = self.particles - parent
        
        subsyst = self.system.copy_to_memory()
        acc_units = (subsyst.vx.unit**2/subsyst.x.unit)
        
        subsyst.position += parent.position
        subsyst.ax = 0. | acc_units
        subsyst.ay = 0. | acc_units
        subsyst.az = 0. | acc_units
        
        # Load the shared library
        lib = ctypes.CDLL('./src/gravity.so')  # Adjust the path as necessary
        lib.find_gravity_at_point.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),  # comp_mass
            np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),  # comp_x
            np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),  # comp_y
            np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),  # comp_z
            ctypes.c_int,  # num_particles
            np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),  # res_ax
            np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),  # res_ay
            np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),  # res_az
            np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),  # int_x
            np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),  # int_y
            np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),  # int_z
            ctypes.c_int   # num_points
        ]
        lib.find_gravity_at_point.restype = None
            
        result_ax = np.zeros(len(subsyst))
        result_ay = np.zeros(len(subsyst))
        result_az = np.zeros(len(subsyst))
        
        lib.find_gravity_at_point(
            parts.mass.value_in(units.kg),
            subsyst.x.value_in(units.m),
            subsyst.y.value_in(units.m),
            subsyst.z.value_in(units.m),
            len(subsyst),
            result_ax, result_ay, result_az,
            parts.x.value_in(units.m),
            parts.y.value_in(units.m),
            parts.z.value_in(units.m),
            len(parts)
            )
        
        result_axm = np.zeros(len(subsyst))
        result_aym = np.zeros(len(subsyst))
        result_azm = np.zeros(len(subsyst))
        
        lib.find_gravity_at_point(
            parts.mass.value_in(units.kg),
            np.asarray([parent.x.value_in(units.m)]),
            np.asarray([parent.y.value_in(units.m)]),
            np.asarray([parent.z.value_in(units.m)]),
            1, result_axm, result_aym, result_azm,
            parts.x.value_in(units.m),
            parts.y.value_in(units.m),
            parts.z.value_in(units.m),
            len(parts)
            )
        
        subsyst.ax += (result_ax - result_axm) * (1 | units.kg*units.m**-2) * constants.G
        subsyst.ay += (result_ay - result_aym) * (1 | units.kg*units.m**-2) * constants.G
        subsyst.az += (result_az - result_azm) * (1 | units.kg*units.m**-2) * constants.G
        
        return subsyst.ax, subsyst.ay, subsyst.az