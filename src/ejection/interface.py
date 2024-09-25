from amuse.community import *
from amuse.community.interface import common
from amuse.units import nbody_system

class EjectionInterface(CodeInterface):
    include_headers = ['interface.h']

    def __init__(self, **options):
        CodeInterface.__init__(self, name_of_the_worker="ejection_worker", **options)

    @legacy_function
    def find_nearest_neighbour():
        function = LegacyFunctionSpecification()
        function.addParameter('xcoord', dtype='float64', direction=function.IN, 
                              description="x coordinates",
                              unit=nbody_system.length)
        function.addParameter('ycoord', dtype='float64', direction=function.IN, 
                              description="y coordinates",
                              unit=nbody_system.length)
        function.addParameter('zcoord', dtype='float64', direction=function.IN, 
                              description="z coordinates",
                              unit=nbody_system.length)
        function.addParameter('num_particles', dtype='int32', direction=function.IN, 
                              description="Number of particles")
        function.addParameter('threshold_distance', dtype='float64', direction=function.IN, 
                              description="Threshold distance",
                              unit=nbody_system.length)
        function.addParameter('ejected', dtype='bool', direction=function.OUT, 
                              description="Ejected particles")
        return function

class Ejection(InCodeComponentImplementation):
    def __init__(self, **options):
        InCodeComponentImplementation.__init__(self, EjectionInterface(**options), **options)

    def find_nearest_neighbour(self, xcoord, ycoord, zcoord, threshold_distance):
        # Convert coordinates to nbody units
        num_particles = len(xcoord)
        ejected = [False] * num_particles

        self.overridden().find_nearest_neighbour(xcoord, ycoord, zcoord, num_particles, threshold_distance, ejected)
        return ejected