from amuse.community.interface import gd

import time as pytime
from amuse.units import nbody_system
from amuse.units import generic_unit_converter
from amuse.community.interface import common
from amuse.rfi.core import *

from amuse import datamodel

from amuse.community.hermite0.interface import Hermite

from amuse.community.mercury.interface import Mercury

import numpy

from collections import namedtuple

from amuse.units import units

from amuse.community.octgrav.interface import Octgrav

from amuse.community.smalln.interface import SmallN

from amuse.community.ph4.interface import ph4

from amuse.community.bhtree.interface import BHTree

from amuse.community.fastkick.interface import FastKick

from amuse.couple.bridge import Bridge


from amuse.units.quantities import is_quantity



from amuse.community.interface import stopping_conditions



from amuse.community.huayno.interface import Huayno

from amuse.community.rebound.interface import Rebound

from amuse.io import write_set_to_file

from mpi4py import MPI


class MultiplexingGravitationalDynamicsInterface(PythonCodeInterface, gd.GravitationalDynamicsInterface, gd.GravityFieldInterface, stopping_conditions.StoppingConditionInterface):
    
    def __init__(self, implementation_factory = None, **options):
        if implementation_factory is None:
            implementation_factory = MultiplexingGravitationalDynamicsImplementation
        PythonCodeInterface.__init__(self, implementation_factory = implementation_factory, **options)
    

    @legacy_function
    def new_multiplexed_particle():
        """
        Define a new particle in the stellar dynamics code. The particle is initialized with the provided
        mass, radius, position and velocity. This function returns an index that can be used to refer
        to this particle.
        """
        function = LegacyFunctionSpecification()
        function.must_handle_array = True
        function.addParameter('index_of_the_particle', dtype='int32', direction=function.OUT, description =
            """
            An index assigned to the newly created particle.
            This index is supposed to be a local index for the code
            (and not valid in other instances of the code or in other codes)
            """
            )

        function.addParameter('mass', dtype='float64', direction=function.IN, description = "The mass of the particle")
        function.addParameter('x', dtype='float64', direction=function.IN, description = "The initial position vector of the particle")
        function.addParameter('y', dtype='float64', direction=function.IN, description = "The initial position vector of the particle")
        function.addParameter('z', dtype='float64', direction=function.IN, description = "The initial position vector of the particle")
        function.addParameter('vx', dtype='float64', direction=function.IN, description = "The initial velocity vector of the particle")
        function.addParameter('vy', dtype='float64', direction=function.IN, description = "The initial velocity vector of the particle")
        function.addParameter('vz', dtype='float64', direction=function.IN, description = "The initial velocity vector of the particle")
        function.addParameter('radius', dtype='float64', direction=function.IN, description = "The radius of the particle", default = 0 | nbody_system.length)
        function.addParameter('index_of_the_set', dtype='int32', direction=function.IN, description = "The index of the multiplexed set", default = 0)
        function.result_type = 'int32'
        function.has_units = True
        function.result_doc = """ 0 - OK
            particle was created and added to the model
        -1 - ERROR
            particle could not be created"""
        return function



    @legacy_function
    def get_mass():
        """
            Retrieve the mass of a particle. Mass is a scalar property of a particle,
            this function has one OUT argument.
            """
        function = LegacyFunctionSpecification()
        function.addParameter('index_of_the_particle', dtype='int32', direction=function.IN,
                              description = "Index of the particle to get the state from. This index must have been returned by an earlier call to :meth:`new_particle`")
        function.addParameter('mass', dtype='float64', direction=function.OUT, description = "The current mass of the particle")
        function.result_type = 'int32'
        function.must_handle_array = True
        function.has_units = True
        function.result_doc = """
            0 - OK
                particle was removed from the model
            -1 - ERROR
                particle could not be found
            """
        return function

    @legacy_function
    def get_radius():
        """
        Retrieve the radius of a particle. Radius is a scalar property of a particle,
        this function has one OUT argument.
        """
        function = LegacyFunctionSpecification()
        function.addParameter('index_of_the_particle', dtype='int32', direction=function.IN,
            description = "Index of the particle to get the radius of. This index must have been returned by an earlier call to :meth:`new_particle`")
        function.addParameter('radius', dtype='float64', direction=function.OUT, description = "The current radius of the particle")
        function.result_type = 'int32'
        function.must_handle_array = True
        function.has_units = True
        function.result_doc = """
        0 - OK
            particle was found in the model and the information was retreived
        -1 - ERROR
            particle could not be found
        """
        return function



    @legacy_function
    def get_position():
        """
        Retrieve the position vector of a particle. Position is a vector property,
        this function has 3 OUT arguments.
        """
        function = LegacyFunctionSpecification()
        function.addParameter('index_of_the_particle', dtype='int32', direction=function.IN,
            description = "Index of the particle to get the state from. This index must have been returned by an earlier call to :meth:`new_particle`")
        function.addParameter('x', dtype='float64', direction=function.OUT, description = "The current position vector of the particle")
        function.addParameter('y', dtype='float64', direction=function.OUT, description = "The current position vector of the particle")
        function.addParameter('z', dtype='float64', direction=function.OUT, description = "The current position vector of the particle")
        function.result_type = 'int32'
        function.must_handle_array = True
        function.has_units = True
        function.result_doc = """
        0 - OK
            current value was retrieved
        -1 - ERROR
            particle could not be found
        -2 - ERROR
            not yet implemented
        """
        return function


    @legacy_function
    def get_velocity():
        """
        Retrieve the velocity vector of a particle. Position is a vector property,
        this function has 3 OUT arguments.
        """
        function = LegacyFunctionSpecification()
        function.addParameter('index_of_the_particle', dtype='int32', direction=function.IN,
            description = "Index of the particle to get the velocity from. This index must have been returned by an earlier call to :meth:`new_particle`")
        function.addParameter('vx', dtype='float64', direction=function.OUT, description = "The current x component of the position vector of the particle")
        function.addParameter('vy', dtype='float64', direction=function.OUT, description = "The current y component of the position vector of the particle")
        function.addParameter('vz', dtype='float64', direction=function.OUT, description = "The current z component of the position vector of the particle")
        function.result_type = 'int32'
        function.must_handle_array = True
        function.has_units = True
        function.result_doc = """
        0 - OK
            current value was retrieved
        -1 - ERROR
            particle could not be found
        -2 - ERROR
            not yet implemented
        """
        return function



    @legacy_function
    def get_state():
        """
        Retrieve the current state of a particle. The *minimal* information of a stellar
        dynamics particle (mass, radius, position and velocity) is returned.
        """
        function = LegacyFunctionSpecification()
        function.addParameter('index_of_the_particle', dtype='int32', direction=function.IN,
            description = "Index of the particle to get the state from. This index must have been returned by an earlier call to :meth:`new_particle`")
        function.addParameter('mass', dtype='float64', direction=function.OUT, description = "The current mass of the particle")
        function.addParameter('x', dtype='float64', direction=function.OUT, description = "The current position vector of the particle")
        function.addParameter('y', dtype='float64', direction=function.OUT, description = "The current position vector of the particle")
        function.addParameter('z', dtype='float64', direction=function.OUT, description = "The current position vector of the particle")
        function.addParameter('vx', dtype='float64', direction=function.OUT, description = "The current velocity vector of the particle")
        function.addParameter('vy', dtype='float64', direction=function.OUT, description = "The current velocity vector of the particle")
        function.addParameter('vz', dtype='float64', direction=function.OUT, description = "The current velocity vector of the particle")
        function.addParameter('radius', dtype='float64', direction=function.OUT, description = "The current radius of the particle")
        function.result_type = 'int32'
        function.must_handle_array = True
        function.has_units = True
        function.result_doc = """
        0 - OK
            particle was removed from the model
        -1 - ERROR
            particle could not be found
        """
        return function


    @legacy_function
    def set_mass():
        """
        Update the mass of a particle. Mass is a scalar property of a particle.
        """
        function = LegacyFunctionSpecification()
        function.addParameter('index_of_the_particle', dtype='int32', direction=function.IN,
            description = "Index of the particle for which the state is to be updated. This index must have been returned by an earlier call to :meth:`new_particle`")
        function.addParameter('mass', dtype='float64', direction=function.IN, description = "The new mass of the particle")
        function.result_type = 'int32'
        function.must_handle_array = True
        function.has_units = True
        function.result_doc = """
        0 - OK
            particle was found in the model and the information was set
        -1 - ERROR
            particle could not be found
        -2 - ERROR
            code does not support updating of a particle
        """
        return function


    @legacy_function
    def set_radius():
        """
        Set the radius of a particle. Radius is a scalar property of a particle.
        """
        function = LegacyFunctionSpecification()
        function.addParameter('index_of_the_particle', dtype='int32', direction=function.IN,
            description = "Index of the particle to get the radius of. This index must have been returned by an earlier call to :meth:`new_particle`")
        function.addParameter('radius', dtype='float64', direction=function.IN, description = "The new radius of the particle")
        function.result_type = 'int32'
        function.must_handle_array = True
        function.has_units = True
        function.result_doc = """
        0 - OK
            particle was found in the model and the information was retreived
        -1 - ERROR
            particle could not be found
        """
        return function


    @legacy_function
    def set_position():
        """
        Update the position of a particle.
        """
        function = LegacyFunctionSpecification()
        function.addParameter('index_of_the_particle', dtype='int32', direction=function.IN,
            description = "Index of the particle for which the state is to be updated. This index must have been returned by an earlier call to :meth:`new_particle`")
        function.addParameter('x', dtype='float64', direction=function.IN, description = "The new position vector of the particle")
        function.addParameter('y', dtype='float64', direction=function.IN, description = "The new position vector of the particle")
        function.addParameter('z', dtype='float64', direction=function.IN, description = "The new position vector of the particle")
        function.result_type = 'int32'
        function.must_handle_array = True
        function.has_units = True
        function.result_doc = """
        0 - OK
            particle was found in the model and the information was set
        -1 - ERROR
            particle could not be found
        -2 - ERROR
            code does not support updating of a particle
        """
        return function


    @legacy_function
    def set_velocity():
        """
        Set the velocity vector of a particle.
        """
        function = LegacyFunctionSpecification()
        function.addParameter('index_of_the_particle', dtype='int32', direction=function.IN,
            description = "Index of the particle to get the state from. This index must have been returned by an earlier call to :meth:`new_particle`")
        function.addParameter('vx', dtype='float64', direction=function.IN, description = "The current x component of the velocity vector of the particle")
        function.addParameter('vy', dtype='float64', direction=function.IN, description = "The current y component of the velocity vector of the particle")
        function.addParameter('vz', dtype='float64', direction=function.IN, description = "The current z component of the velocity vector of the particle")
        function.result_type = 'int32'
        function.must_handle_array = True
        function.has_units = True
        function.result_doc = """
        0 - OK
            current value was retrieved
        -1 - ERROR
            particle could not be found
        -2 - ERROR
            not yet implemented
        """
        return function



    @legacy_function
    def set_state():
        """
        Update the current state of a particle. The *minimal* information of a stellar
        dynamics particle (mass, radius, position and velocity) is updated.
        """
        function = LegacyFunctionSpecification()
        function.addParameter('index_of_the_particle', dtype='int32', direction=function.IN,
            description = "Index of the particle for which the state is to be updated. This index must have been returned by an earlier call to :meth:`new_particle`")
        function.addParameter('mass', dtype='float64', direction=function.IN, description = "The new mass of the particle")
        function.addParameter('x', dtype='float64', direction=function.IN, description = "The new position vector of the particle")
        function.addParameter('y', dtype='float64', direction=function.IN, description = "The new position vector of the particle")
        function.addParameter('z', dtype='float64', direction=function.IN, description = "The new position vector of the particle")
        function.addParameter('vx', dtype='float64', direction=function.IN, description = "The new velocity vector of the particle")
        function.addParameter('vy', dtype='float64', direction=function.IN, description = "The new velocity vector of the particle")
        function.addParameter('vz', dtype='float64', direction=function.IN, description = "The new velocity vector of the particle")
        function.addParameter('radius', dtype='float64', direction=function.IN, description = "The new radius of the particle", default = 0 | nbody_system.length)
        function.result_type = 'int32'
        function.must_handle_array = True
        function.has_units = True
        function.result_doc = """
        0 - OK
            particle was found in the model and the information was set
        -1 - ERROR
            particle could not be found
        -2 - ERROR
            code does not support updating of a particle
        -3 - ERROR
            not yet implemented
        """
        return function


    @legacy_function
    def delete_particle():
        """
        Remove the definition of particle from the code. After calling this function the particle is
        no longer part of the model evolution. It is up to the code if the index will be reused.
        This function is optional.
        """
        function = LegacyFunctionSpecification()
        function.must_handle_array = True
        function.addParameter('index_of_the_particle', dtype='int32', direction=function.IN,
            description = "Index of the particle to be removed. This index must have been returned by an earlier call to :meth:`new_particle`")
        function.result_type = 'int32'
        function.result_doc = """
        0 - OK
            particle was removed from the model
        -1 - ERROR
            particle could not be removed
        -2 - ERROR
            not yet implemented
        """
        return function
        

    @legacy_function
    def set_code():
        """
        Update the name of the integration (multiplexed) code
        """
        function = LegacyFunctionSpecification()
        function.addParameter('code', dtype='string', direction=function.IN)
        function.result_type = 'int32'
        function.result_doc = """
        0 - OK
            code name was set
        -1 - ERROR
            code name is not known
        """
        return function


    @legacy_function
    def evolve_model():
        """
        Evolve the model until the given time, or until a stopping condition is set.
        """
        function = LegacyFunctionSpecification()
        function.addParameter('time', dtype='float64', direction=function.IN,
            description = "Model time to evolve the code to. The model will be "
                "evolved until this time is reached exactly or just after.")
        function.result_type = 'int32'
        function.has_units = True
        return function

    @legacy_function
    def get_time():
        """
        Returns the current model time.
        """
        function = LegacyFunctionSpecification()     
        function.addParameter('index', dtype='int32', direction=function.IN, default = -1)   
        function.addParameter('value', dtype='float64', direction=function.OUT)     
        function.result_type = 'i'
        function.has_units = True
        return function
        
    @legacy_function
    def get_begin_time():
        """
        Retrieve the model time to start the evolution at.
        """
        function = LegacyFunctionSpecification()
        function.addParameter('time', dtype='float64', direction=function.OUT,
            description = "The begin time", unit = nbody_system.time)
        function.result_type = 'int32'
        function.has_units = True
        function.result_doc = """
        0 - OK
            Current value of the time was retrieved
        -2 - ERROR
            The code does not have support for querying the begin time
        """
        return function
    @legacy_function
    def set_begin_time():
        """
        Set the model time to start the evolution at. This is an offset for
        all further calculations in the code.
        """
        function = LegacyFunctionSpecification()
        function.addParameter('time', dtype='float64', direction=function.IN,
            description = "The model time to start at", unit = nbody_system.time)
        function.result_type = 'int32'
        function.has_units = True
        function.result_doc = """
        0 - OK
            Time value was changed
        -2 - ERROR
            The code does not support setting the begin time
        """
        return function
        
    @legacy_function
    def set_eps2():
        """
        Update the value of the squared smoothing parameter.
        """
        function = LegacyFunctionSpecification()
        function.addParameter('epsilon_squared', dtype='float64', direction=function.IN,
            description = "The new value of the smooting parameter, squared.")
        function.result_type = 'int32'
        function.has_units = True
        function.result_doc = """
        0 - OK
            Current value of the smoothing parameter was set
        -1 - ERROR
            The code does not have support for a smoothing parameter
        """
        return function
    @legacy_function
    def get_eps2():
        """
        Retrieve the current value of the squared smoothing parameter.
        """
        function = LegacyFunctionSpecification()
        function.addParameter('epsilon_squared', dtype='float64', direction=function.OUT,
            description = "The current value of the smooting parameter, squared.")
        function.result_type = 'int32'
        function.has_units = True
        function.result_doc = """
        0 - OK
            Current value of the smoothing parameter was set
        -1 - ERROR
            The code does not have support for a smoothing parameter
        """
        return function






    @legacy_function
    def new_subset():
        """
        Create a new particle subset (and corresponding code). This subset will evolve seperately from others.
        """
        function = LegacyFunctionSpecification()
        function.can_handle_array = True
        function.addParameter('index_of_the_subset', dtype='int32', direction=function.OUT, description =
            """
            An index assigned to the newly created subset
            """
            )

        function.addParameter('time_offset', dtype='float64', direction=function.IN, description = "Time of the system (defaults to the current model time)", default = -1)
        
        function.addParameter('name_of_the_code', dtype='string', direction=function.IN, description =
            "name of the code to start, if different from the default", default = ""
            )
        function.result_type = 'int32'
        function.has_units = True
        function.result_doc = """ 0 - OK
            code was created
        -1 - ERROR
            code could not be created"""
        return function
        
        



    @legacy_function
    def get_kinetic_energy():
        """
        Retrieve the current kinetic energy of the model
        """
        function = LegacyFunctionSpecification()
        function.addParameter('code_index', dtype='int32', direction=function.IN, description = "Index of the code in rebound", default = 0)
        function.addParameter('kinetic_energy', dtype='float64', direction=function.OUT,
            description = "The kinetic energy of the model")
        function.result_type = 'int32'
        function.has_units = True
        function.result_doc = """
        0 - OK
            Current value of the kinetic energy was set
        -1 - ERROR
            Kinetic energy could not be provided
        """
        return function




    @legacy_function
    def get_potential_energy():
        """
        Retrieve the current potential energy of the model
        """
        function = LegacyFunctionSpecification()
        function.addParameter('code_index', dtype='int32', direction=function.IN, description = "Index of the code in rebound", default = 0)
        function.addParameter('potential_energy', dtype='float64', direction=function.OUT,
            description = "The potential energy of the model")
        function.result_type = 'int32'
        function.has_units = True
        function.result_doc = """
        0 - OK
            Current value of the potential energy was set
        -1 - ERROR
            Kinetic potential could not be provided
        """
        return function
        



    @legacy_function
    def stop_subset():
        """
        Stop a subset code
        """
        function = LegacyFunctionSpecification()
        function.can_handle_array = True
        function.addParameter('index_of_the_subset', dtype='int32', direction=function.IN, description =
            """
            An index assigned to an existing subset
            """
            )

        function.result_type = 'int32'
        function.result_doc = """ 0 - OK
            subset evolving was stopped
        -1 - ERROR
            subset evolving was already stopped"""
        return function
        
        


    @legacy_function
    def get_index_of_the_set():
        """
        Retrieve the index of the subset the particle is part of
        """
        function = LegacyFunctionSpecification()
        function.addParameter('index_of_the_particle', dtype='int32', direction=function.IN,
                              description = "Index of the particle to get the state from. This index must have been returned by an earlier call to :meth:`new_particle`")
        function.addParameter('index_of_the_set', dtype='int32', direction=function.OUT)
        function.result_type = 'int32'
        function.must_handle_array = True
        function.has_units = False
        function.result_doc = """
            0 - OK
                particle was removed from the model
            -1 - ERROR
                particle could not be found
            """
        return function



class SinglePointGravityFieldInterface(object):
    """
    Codes implementing the gravity field interface provide functions to
    calculate the force and potential energy fields at any point.
    """
    
    @legacy_function    
    def get_gravity_at_point():
        """
        Get the gravitational acceleration at the given points. To calculate the force on
        bodies at those points, multiply with the mass of the bodies
        """
        function = LegacyFunctionSpecification()  
        for x in ['eps','x','y','z']:
            function.addParameter(
              x, 
              dtype='float64', 
              direction=function.IN,
              unit=nbody_system.length
            )
        for x in ['ax','ay','az']:
            function.addParameter(
                x, 
                dtype='float64', 
                direction=function.OUT,     
                unit=nbody_system.acceleration
            )
        function.result_type = 'int32' 
        function.can_handle_array = True
        return function
        
    @legacy_function    
    def get_potential_at_point():
        """
        Determine the gravitational potential on any given point
        """
        function = LegacyFunctionSpecification()  
        for x in ['eps','x','y','z']:
            function.addParameter(
                x, 
                dtype='float64', 
                direction=function.IN,
                unit=nbody_system.length
            )
        for x in ['phi']:
            function.addParameter(
                x, 
                dtype='float64', 
                direction=function.OUT,
                unit=nbody_system.potential
            )
        function.result_type = 'int32'
        function.can_handle_array = True
        return function


class MultiplexingGravitationalDynamicsImplementation(object):
    
    def __init__(self):
        self.keys_generator = datamodel.BasicUniqueKeyGenerator(1)
        self.particles = datamodel.Particles()
        self.name_of_the_code = "Hermite"
        CodeDefinition = namedtuple('CodeDefinition', ['code', 'mass_unit', 'speed_unit', 'length_unit', 'time_unit'])
        self.code_names_to_code_classes = {
            "Hermite": CodeDefinition(Hermite,nbody_system.mass, nbody_system.speed, nbody_system.length, nbody_system.time),
            "BHTree": CodeDefinition(BHTree,nbody_system.mass, nbody_system.speed, nbody_system.length, nbody_system.time),
            "SmallN": CodeDefinition(SmallN,nbody_system.mass, nbody_system.speed, nbody_system.length, nbody_system.time),
            "OctGrav": CodeDefinition(Octgrav,nbody_system.mass, nbody_system.speed, nbody_system.length, nbody_system.time),
            "ph4": CodeDefinition(ph4,nbody_system.mass, nbody_system.speed, nbody_system.length, nbody_system.time),
            "FastKick": CodeDefinition(FastKick,nbody_system.mass, nbody_system.speed, nbody_system.length, nbody_system.time),
            "Mercury": CodeDefinition(Mercury,units.MSun, units.AUd, units.AU, units.day)
        }
        self.code = None
        self.begin_time = 0
        self.time = 0
        self.mass_unit = nbody_system.mass
        self.speed_unit = nbody_system.speed
        self.length_unit = nbody_system.length
        self.time_unit = nbody_system.time
        

    def new_multiplexed_particle(self, index_of_the_particle, mass, x, y, z, vx, vy, vz, radius, index_of_the_set):
        new_particles = datamodel.Particles(len(mass), keys_generator = self.keys_generator)
        new_particles.mass = mass
        new_particles.x = x
        new_particles.y = y
        new_particles.z = z
        new_particles.vx = vx
        new_particles.vy = vy
        new_particles.vz = vz
        new_particles.radius = radius
        new_particles.index_of_the_set = index_of_the_set
        index_of_the_particle.value = new_particles.key
        self.add_particles(new_particles)
        return 0
        



    def get_mass(self, index_of_the_particle, mass):
        subset = self.select_particles(index_of_the_particle)
        if not len(subset) == len(index_of_the_particle):
            mass.value = [0] * len(index_of_the_particle)
            return -1
        else:
            mass.value = subset.mass
            return 0
        



    def get_position(self, index_of_the_particle, x, y, z):
        subset = self.select_particles(index_of_the_particle)
        x.value = subset.x
        y.value = subset.y
        z.value = subset.z
        return 0
        


    def get_velocity(self, index_of_the_particle, vx, vy, vz):
        subset = self.select_particles(index_of_the_particle)
        vx.value = subset.vx
        vy.value = subset.vy
        vz.value = subset.vz
        return 0
        


    def get_state(self, index_of_the_particle, mass, x, y, z, vx, vy, vz, radius):
        subset = self.select_particles(index_of_the_particle)
        if not len(subset) == len(index_of_the_particle):
            return -1
        else:
            mass.value = subset.mass
            x.value = subset.x
            y.value = subset.y
            z.value = subset.z
            vx.value = subset.vx
            vy.value = subset.vy
            vz.value = subset.vz
            radius.value = subset.radius
            return 0
        

    def select_particles(self,keys):
        return self.particles._subset(keys)
        



    def set_mass(self, index_of_the_particle, mass):
        self.select_particles(index_of_the_particle).mass = mass
        return 0
        




    def set_position(self, index_of_the_particle, x, y, z):
        subset = self.select_particles(index_of_the_particle)
        subset.x = x     
        subset.y = y
        subset.z = z
        return 0
        



    def set_velocity(self, index_of_the_particle, vx, vy, vz):
        subset = self.select_particles(index_of_the_particle)
        subset.vx = vx
        subset.vy = vy     
        subset.vz = vz     
        return 0
        



    def set_state(self, index_of_the_particle, mass, x, y, z, vx, vy, vz, radius):
        subset = self.select_particles(index_of_the_particle)
        subset.mass = mass
        subset.x = x         
        subset.y = y
        subset.z = z
        subset.vx = vx
        subset.vy = vy
        subset.vz = vz
        subset.radius = radius
        return 0
        



    def delete_particle(self, index_of_the_particle):
        subset = self.select_particles(index_of_the_particle)
        self.remove_particles(subset)
        return 0
        




    def initialize_code(self):
        self.keys_generator = datamodel.BasicUniqueKeyGenerator(1)
        self.particles = datamodel.Particles()
        return 0
        



    def cleanup_code(self):
        self.keys_generator = datamodel.BasicUniqueKeyGenerator(1)
        self.particles = datamodel.Particles()
        if not self.code is None:
            self.code.stop()
            self.code = None
        return 0
        



    def commit_particles(self):
        return 0
        



    def recommit_particles(self):
        return 0
        



    def commit_parameters(self):
        code_definition = self.code_names_to_code_classes[self.name_of_the_code]
        self.factory = code_definition.code
        
        self.mass_unit = code_definition.mass_unit
        self.speed_unit = code_definition.speed_unit
        self.length_unit = code_definition.length_unit
        self.time_unit = code_definition.time_unit
        self.time = self.begin_time
        return 0
        



    def recommit_parameters(self):
        return 0
        



    def set_begin_time(self, time):
        self.begin_time = time
        return 0
        



    def get_begin_time(self, time):
        time.value = self.begin_time
        return 0
        



    def get_eps2(self, epsilon_squared):
        epsilon_squared.value = self.epsilon_squared
        return 0
        



    def set_eps2(self, epsilon_squared):
        self.epsilon_squared = epsilon_squared
        return 0
        



    def get_code(self, code):
        code.value = self.name_of_the_code
        return 0
        



    def set_code(self, code):
        self.name_of_the_code = code
        return 0
        



    def evolve_model(self, time):
        print("TIME:", time)
        if self.code is None:
            factory = self.code_names_to_code_classes[self.name_of_the_code].code
            self.code = factory(redirection="null")#debugger="ddd")
            self.code.parameters.epsilon_squared = self.epsilon_squared             
            
        set_indices = numpy.unique(self.particles.index_of_the_set)
        for index in set_indices:
            particles = self.particles[self.particles.index_of_the_set == index]
            
            self.code.reset()
            self.code.parameters.begin_time = self.time
            self.code.particles.add_particles(particles)
            
            self.code.evolve_model(time)
            self.time = max(self.code.model_time, self.time)
            particles.velocity = self.code.particles.velocity
            particles.position = self.code.particles.position
        #self.time = time
        return 0
        



    def get_time(self, index, time):
        time.value = self.time
        return 0
        



    def synchronize_model(self):
        return 0
        


    def add_particles(self, particles):
        self.particles.add_particles(particles)
        



    def remove_particles(self, particles):
        self.particles.remove_particles(particles)




    def new_subset(self, index_of_the_subset, time_offset, name_of_the_code):
        index_of_the_subset.value = 0
        return 0


    def get_index_of_the_set(self, index_of_the_particle, index_of_the_set):
        subset = self.select_particles(index_of_the_particle)
        if not len(subset) == len(index_of_the_particle):
            index_of_the_set.value = [0] * len(index_of_the_particle)
            return -1
        else:
            index_of_the_set.value = subset.index_of_the_set
            return 0
            
    
    def set_stopping_condition_timeout_parameter(self, value):
      
      return 0


    def get_stopping_condition_maximum_density_parameter(self, value):
      
      value.value = 0
      return 0


    def set_stopping_condition_out_of_box_use_center_of_mass_parameter(self, value):
      
      return 0


    def get_stopping_condition_out_of_box_use_center_of_mass_parameter(self, value):
      
      value.value = 0
      return 0


    def enable_stopping_condition(self, type):
      
        if type == 0:
            self.is_collision_detection_enabled = True
            for x in list(self.codes.values()):
                x.stopping_conditions.collision_detection.enable()
        return 0



    def get_stopping_condition_maximum_internal_energy_parameter(self, value):
      
      value.value = 0
      return 0


    def get_number_of_stopping_conditions_set(self, result):
        if self.is_collision_detection_set:
            result.value = len(self.stopping_condition_particles[0])
        else:
            results.value = 0
        return 0



    def is_stopping_condition_set(self, type, result):
        if type == 0 and self.is_collision_detection_set:
            result.value = 1
        else:
            result.value = 0
        return 0



    def set_stopping_condition_out_of_box_parameter(self, value):
      
      return 0


    def set_stopping_condition_number_of_steps_parameter(self, value):
      
      return 0


    def get_stopping_condition_timeout_parameter(self, value):
      
      value.value = 0
      return 0


    def get_stopping_condition_minimum_internal_energy_parameter(self, value):
      
      value.value = 0
      return 0


    def is_stopping_condition_enabled(self, type, result):
      
      result.value = 1 if type == 0 else 0
      return 0



    def get_stopping_condition_minimum_density_parameter(self, value):
      
      value.value = 0
      return 0


    def get_stopping_condition_number_of_steps_parameter(self, value):
      
      value.value = 0
      return 0


    def disable_stopping_condition(self, type):
      
        if type == 0:
            self.is_collision_detection_enabled = False
            for x in list(self.codes.values()):
                x.stopping_conditions.collision_detection.disable()
        return 0



    def set_stopping_condition_minimum_internal_energy_parameter(self, value):
      
      return 0


    def set_stopping_condition_minimum_density_parameter(self, value):
      
      return 0


    def has_stopping_condition(self, type, result):
      
      result.value = 0
      return 0


    def set_stopping_condition_maximum_density_parameter(self, value):
      
      return 0


    def get_stopping_condition_out_of_box_parameter(self, value):
      
      value.value = 0
      return 0


    def set_stopping_condition_maximum_internal_energy_parameter(self, value):
      
      return 0


    def get_stopping_condition_info(self, index, type, number_of_particles):
      if self.is_collision_detection_set:
          if index < len(self.stopping_condition_particles[0]):
              type.value = 0
              number_of_particles.value = 2
      else:
          type.value = -1
          number_of_particles.value = 0
          return -1
      return 0



    def get_stopping_condition_particle_index(self, index, index_of_the_column, index_of_particle):
        if self.is_collision_detection_set:
            if index < len(self.stopping_condition_particles[index_of_the_column]):
                index_of_particle.value = self.stopping_condition_particles[index_of_the_column][index].key
            return 0
        else:
            index_of_particle.value = -1
            return -1



class MultiplexingGravitationalDynamicsCode(gd.GravitationalDynamics):
    

    def __init__(self, unit_converter = None, remote_code = None, **options):
        self.stopping_conditions = stopping_conditions.StoppingConditions(self)

        if remote_code is None:
            remote_code = MultiplexingGravitationalDynamicsInterface(**options)
        
        gd.GravitationalDynamics.__init__(self, remote_code, unit_converter, **options)


    def define_properties(self, object):
        #gd.GravitationalDynamics.define_properties(self, object)
        object.add_property('get_time', public_name = "model_time")
        

    def define_methods(self, object):
        common.CommonCode.define_methods(self, object)
        self.stopping_conditions.define_methods(object)

        object.add_method(
            'evolve_model',
            (
                object.UNIT,
            ),
            (
                object.ERROR_CODE,
            )
        )

        object.add_method(
            'new_subset',
            (
                object.UNIT,
                object.INDEX,
            ),
            (
                object.INDEX,
                object.ERROR_CODE,
            )
        )


        object.add_method(
            "delete_particle",
            (
                object.INDEX,
            ),
            (
                object.ERROR_CODE,
            )
        )
        object.add_method(
            "get_state",
            (
                object.INDEX,
            ),
            (
                object.UNIT,
                object.UNIT,
                object.UNIT,
                object.UNIT,
                object.UNIT,
                object.UNIT,
                object.UNIT,
                object.UNIT,
                object.ERROR_CODE
            )
        )
        object.add_method(
            "set_state",
            (
                object.INDEX,
                object.UNIT,
                object.UNIT,
                object.UNIT,
                object.UNIT,
                object.UNIT,
                object.UNIT,
                object.UNIT,
                object.UNIT,
            ),
            (
                object.ERROR_CODE
            )
        )
        object.add_method(
            "set_mass",
            (
                object.INDEX,
                object.UNIT,
            ),
            (
                object.ERROR_CODE
            )
        )
        object.add_method(
            "get_mass",
            (
                object.INDEX,
            ),
            (
                object.UNIT,
                object.ERROR_CODE
            )
        )
        object.add_method(
            "set_radius",
            (
                object.INDEX,
                object.UNIT,
            ),
            (
                object.ERROR_CODE
            )
        )
        object.add_method(
            "get_radius",
            (
                object.INDEX,
            ),
            (
                object.UNIT,
                object.ERROR_CODE
            )
        )
        object.add_method(
            "set_position",
            (
                object.INDEX,
                object.UNIT,
                object.UNIT,
                object.UNIT,
            ),
            (
                object.ERROR_CODE
            )
        )
        object.add_method(
            "get_position",
            (
                object.INDEX,
            ),
            (
                object.UNIT,
                object.UNIT,
                object.UNIT,
                object.ERROR_CODE
            )
        )
        object.add_method(
            "set_velocity",
            (
                object.INDEX,
                object.UNIT,
                object.UNIT,
                object.UNIT,
            ),
            (
                object.ERROR_CODE
            )
        )
        object.add_method(
            "get_velocity",
            (
                object.INDEX,
            ),
            (
                object.UNIT,
                object.UNIT,
                object.UNIT,
                object.ERROR_CODE
            )
        )
        

        object.add_method(
            "get_time_step",
            (),
            (
                object.UNIT,
                object.ERROR_CODE,
            )
        )


        object.add_method(
            "get_kinetic_energy",
            (object.INDEX),
            (object.UNIT, object.ERROR_CODE,)
        )


        object.add_method(
            "get_potential_energy",
            (object.INDEX),
            (object.UNIT, object.ERROR_CODE,)
        )


        object.add_method(
            "get_total_radius",
            (),
            (object.UNIT, object.ERROR_CODE,)
        )


        object.add_method(
            "get_center_of_mass_position",
            (),
            (object.UNIT,object.UNIT,object.UNIT, object.ERROR_CODE,)
        )


        object.add_method(
            "get_center_of_mass_velocity",
            (),
            (object.UNIT,object.UNIT,object.UNIT, object.ERROR_CODE,)
        )


        object.add_method(
            "get_total_mass",
            (),
            (object.UNIT, object.ERROR_CODE,)
        )


        object.add_method(
            'get_time',
            (object.INDEX),
            (object.UNIT, object.ERROR_CODE,)
        )


        object.add_method(
            "set_begin_time",
            (
                object.UNIT,
            ),
            (
                object.ERROR_CODE,
            )
        )


        object.add_method(
            "get_begin_time",
            (
            ),
            (
                object.UNIT,
                object.ERROR_CODE,
            )
        )

        
        object.add_method(
            "new_multiplexed_particle",
            (
                object.UNIT,
                object.UNIT,
                object.UNIT,
                object.UNIT,
                object.UNIT,
                object.UNIT,
                object.UNIT,
                object.UNIT,
                object.UNIT,
            ),
            (
                object.INDEX,
                object.ERROR_CODE,
            )
        )
        
        object.add_method(
            "get_eps2",
            (),
            (object.UNIT, object.ERROR_CODE,)
        )
        
        object.add_method(
            "set_eps2",
            (object.UNIT, ),
            (object.ERROR_CODE,)
        )




    def define_particle_sets(self, object):
        object.define_set('particles', 'index_of_the_particle')
        object.set_new('particles', 'new_multiplexed_particle')
        object.set_delete('particles', 'delete_particle')
        object.add_setter('particles', 'set_state')
        object.add_getter('particles', 'get_state')
        object.add_setter('particles', 'set_mass')
        object.add_getter('particles', 'get_mass', names = ('mass',))
        object.add_setter('particles', 'set_position')
        object.add_getter('particles', 'get_position')
        object.add_setter('particles', 'set_velocity')
        object.add_getter('particles', 'get_velocity')
        object.add_setter('particles', 'set_radius')
        object.add_getter('particles', 'get_radius')
        object.add_getter('particles', 'get_index_of_the_set')
        self.stopping_conditions.define_particle_set(object)
        
        


    def define_state(self, object): 
        gd.GravitationalDynamics.define_state(self, object)                   
        object.add_method('EDIT', 'new_multiplexed_particle')
        object.add_method('UPDATE', 'new_multiplexed_particle')
        object.add_transition('RUN', 'UPDATE', 'new_multiplexed_particle', False)
        self.stopping_conditions.define_state(object)
        



    def define_parameters(self, object):
        self.stopping_conditions.define_parameters(object)

        object.add_method_parameter(
            "get_eps2",
            "set_eps2", 
            "epsilon_squared", 
            "smoothing parameter for gravity calculations", 
            default_value = 0.0 | nbody_system.length * nbody_system.length
        )
        object.add_method_parameter(
            "get_begin_time",
            "set_begin_time",
            "begin_time",
            "model time to start the simulation at",
            default_value = 0.0 | nbody_system.time
        )




class TestParticlesInField(MultiplexingGravitationalDynamicsImplementation):
    
    def __init__(self):
        MultiplexingGravitationalDynamicsImplementation.__init__(self)
        



    def commit_parameters(self):
        MultiplexingGravitationalDynamicsImplementation.commit_parameters(self)
        
        return 0
        



    def recommit_parameters(self):
        return 0
        




    def evolve_model(self, time):
        if self.code is None:
            factory = self.code_names_to_code_classes[self.name_of_the_code].code
            self.code = factory(redirection="null")#debugger="ddd")
            self.code.parameters.epsilon_squared = self.epsilon_squared | self.length_unit**2
            
        set_indices = numpy.unique(self.particles.index_of_the_set)
        for index in set_indices:
            particles = self.particles[self.particles.index_of_the_set == index]
            
            self.code.reset()
            self.code.parameters.begin_time = self.time | self.time_unit
            self.code.particles.add_particles(particles)
            
            self.code.evolve_model(time | self.time_unit)
            self.time = max(self.code.model_time.value_in(self.time_unit), self.time)
            particles.velocity = self.code.particles.velocity
            particles.position = self.code.particles.position
        self.time = time
        return 0
        



    def synchronize_model(self):
        return 0
        





class MultiplexingMercuryCode(MultiplexingGravitationalDynamicsCode):
    
    def __init__(self, **options):
        MultiplexingGravitationalDynamicsCode.__init__(self, unit_converter = None, **options)
        self.set_code("Mercury")






    def define_parameters(self, object):
        object.add_method_parameter(
            "get_eps2",
            "set_eps2", 
            "epsilon_squared", 
            "smoothing parameter for gravity calculations", 
            default_value = 0.0 | units.AU**2
        )
        object.add_method_parameter(
            "get_begin_time",
            "set_begin_time",
            "begin_time",
            "model time to start the simulation at",
            default_value = 0.0 | units.yr
        )

class Code(object):
    
    def __init__(self, instance, dt):
        self.instance = instance
        self.dt = dt
        
    def evolve_model(self, endtime):
        if len(self.instance.particles) == 1:
            return self.evolve_single_particle(endtime + self.dt)
        else:
            return self.instance.evolve_model(endtime + self.dt)
    

    @property
    def model_time(self):
        return self.instance.model_time - self.dt
        
    def step_back(self, dt):
        self.dt += dt
        
    def evolve_single_particle(self, endtime):
        particle = self.instance.particles[0]
        particle.position += particle.velocity * (endtime - self.model_time)
                




class MultiplexingGravitationalDynamicsImplementationWithLocalCode(object):
    
    def new_multiplexed_particle(self, index_of_the_particle, mass, x, y, z, vx, vy, vz, radius, index_of_the_set):
        new_particles = datamodel.Particles(len(mass), keys_generator = self.keys_generator)
        new_particles.mass = mass
        new_particles.x = x
        new_particles.y = y
        new_particles.z = z
        new_particles.vx = vx
        new_particles.vy = vy
        new_particles.vz = vz
        new_particles.radius = radius
        new_particles.index_of_the_set = index_of_the_set
        index_of_the_particle.value = new_particles.key

        new_particles_in_set = self.add_particles(new_particles)

        set_indices = numpy.unique(new_particles_in_set.index_of_the_set)
        for index in set_indices:
            particles = new_particles_in_set[new_particles_in_set.index_of_the_set == index]
            
            if not index in self.codes:
                continue
            
            code = self.codes[index]                              
            particles.synchronize_to(code.instance.particles)
            channel = particles.new_channel_to(code.instance.particles)
            channel.copy()                 

        return 0
        




    def get_mass(self, index_of_the_particle, mass):
        subset = self.select_particles(index_of_the_particle)
        if not len(subset) == len(index_of_the_particle):
            mass.value = [0] * len(index_of_the_particle)
            return -1
        else:
            mass.value = subset.mass
            return 0
        



    def get_position(self, index_of_the_particle, x, y, z):
        subset = self.select_particles(index_of_the_particle)
        x.value = subset.x
        y.value = subset.y
        z.value = subset.z
        return 0
        


    def get_velocity(self, index_of_the_particle, vx, vy, vz):
        subset = self.select_particles(index_of_the_particle)
        vx.value = subset.vx
        vy.value = subset.vy
        vz.value = subset.vz
        return 0
        


    def get_state(self, index_of_the_particle, mass, x, y, z, vx, vy, vz, radius):
        subset = self.select_particles(index_of_the_particle)
        if not len(subset) == len(index_of_the_particle):
            return -1
        else:
            mass.value = subset.mass
            x.value = subset.x
            y.value = subset.y
            z.value = subset.z
            vx.value = subset.vx
            vy.value = subset.vy
            vz.value = subset.vz
            radius.value = subset.radius
            return 0
        


    def select_particles(self,keys):
        return self.particles._subset(keys)
        



    def set_mass(self, index_of_the_particle, mass):
        self.select_particles(index_of_the_particle).mass = mass
        return 0
        




    def set_position(self, index_of_the_particle, x, y, z):
        subset = self.select_particles(index_of_the_particle)
        subset.x = x     
        subset.y = y
        subset.z = z
        return 0
        



    def set_velocity(self, index_of_the_particle, vx, vy, vz):
        subset = self.select_particles(index_of_the_particle)
        subset.vx = vx
        subset.vy = vy     
        subset.vz = vz     
        return 0
        



    def set_state(self, index_of_the_particle, mass, x, y, z, vx, vy, vz, radius):
        subset = self.select_particles(index_of_the_particle)
        subset.mass = mass
        subset.x = x             
        subset.y = y
        subset.z = z
        subset.vx = vx
        subset.vy = vy
        subset.vz = vz
        subset.radius = radius
        return 0
        



    def delete_particle(self, index_of_the_particle):
        subset = self.select_particles(index_of_the_particle)
        self.remove_particles(subset)
        return 0
        




    def initialize_code(self):
        self.keys_generator = datamodel.BasicUniqueKeyGenerator(1)
        self.particles = datamodel.Particles()
        print(MPI.Get_processor_name())
        return 0
        



    def cleanup_code(self):
        self.keys_generator = datamodel.BasicUniqueKeyGenerator(1)
        self.particles = datamodel.Particles()
        for code in list(self.codes.values()):
            code.instance.stop()
            code = None
        self.codes = {}
        return 0
        




    def commit_particles(self):
        return 0
        



    def recommit_particles(self):
        return 0
        



    def commit_parameters(self):
        print(list(self.code_names_to_code_classes.keys()))
        code_definition = self.code_names_to_code_classes[self.name_of_the_code]
        self.factory = code_definition.code
        
        self.mass_unit = code_definition.mass_unit
        self.speed_unit = code_definition.speed_unit
        self.length_unit = code_definition.length_unit
        self.time_unit = code_definition.time_unit
        self.time = self.begin_time
        return 0
        




    def recommit_parameters(self):
        return 0
        



    def set_begin_time(self, time):
        self.begin_time = time
        return 0
        



    def get_begin_time(self, time):
        time.value = self.begin_time
        return 0
        



    def get_eps2(self, epsilon_squared):
        epsilon_squared.value = self.epsilon_squared
        return 0
        



    def set_eps2(self, epsilon_squared):
        self.epsilon_squared = epsilon_squared
        return 0
        



    def get_code(self, code):
        code.value = self.name_of_the_code
        return 0
        



    def set_code(self, code):
        self.name_of_the_code = code
        return 0
        



    def evolve_model(self, time):
        self.is_collision_detection_set = False
        set_indices = numpy.unique(self.particles.index_of_the_set)
        statistics = []
            
        for index in set_indices:
            particles = self.particles[self.particles.index_of_the_set == index]
            
            if not index in self.codes:
                continue
            t0 = pytime.time()
            had_a_collision = False
            had_energy_problem = False
            code = self.codes[index]                                                                                                                                                      
            particles.synchronize_to(code.instance.particles)
            channel = particles.new_channel_to(code.instance.particles)
            channel.copy()                                
            start_ke = code.instance.kinetic_energy                                                                                                                                   
            if code.model_time < time:     
                ke0 = code.instance.kinetic_energy                                                     
                pe0 = code.instance.potential_energy                                                            
                model_time0 = code.model_time                                                                            
                code.evolve_model(time)                                      
                                          
                ke1 = code.instance.kinetic_energy                                              
                pe1 = code.instance.potential_energy                                                                                         
                if len(code.instance.particles) > 1 and self.is_collision_detection_enabled and code.instance.stopping_conditions.collision_detection.is_set():
                    channel = code.instance.particles.new_channel_to(particles)
                    channel.copy_attributes(["x","y","z","vx","vy","vz"])
                    p0 = code.instance.stopping_conditions.collision_detection.particles(0)
                    p1 = code.instance.stopping_conditions.collision_detection.particles(1)
                    p0 = p0.get_intersecting_subset_in(self.particles)
                    p1 = p1.get_intersecting_subset_in(self.particles)
                    self.stopping_condition_particles = [p0,p1]
                    self.is_collision_detection_set = True
                    break
                e0 = ke0 + pe0
                e1 = ke1 + pe1
                if numpy.abs((e1 - e0) / e0) > 0.1:
                    had_energy_problem = True
                    timestep_parameter_name = self.timestep_parameter_name()
                    if not timestep_parameter_name is None:
                        original_timestep = getattr(code.instance.parameters, timestep_parameter_name)
                        for i in range(3):
                            new_value = getattr(code.instance.parameters, timestep_parameter_name) / 2.0
                            setattr(code.instance.parameters, timestep_parameter_name, new_value)
                            channel.copy()                                                                                    
                            print("code:", index, ",try:", i, ",large energy change:",numpy.abs((e1 - e0) / e0),"taking smaller ("+timestep_parameter_name+"):",  new_value, original_timestep,  numpy.abs(code.instance.kinetic_energy - ke0) / ke0)                                                             # reset the state of the particles   
                            code.step_back(code.model_time - model_time0)                                                                                             
                            code.evolve_model(time)
                            if self.is_collision_detection_enabled and code.instance.stopping_conditions.collision_detection.is_set():
                                channel = code.instance.particles.new_channel_to(particles)
                                channel.copy_attributes(["x","y","z","vx","vy","vz"])
                                p0 = code.instance.stopping_conditions.collision_detection.particles(0)
                                p1 = code.instance.stopping_conditions.collision_detection.particles(1)
                                p0 = p0.get_intersecting_subset_in(self.particles)
                                p1 = p1.get_intersecting_subset_in(self.particles)
                                self.stopping_condition_particles = [p0,p1]
                                self.is_collision_detection_set = True
                                break
                            pe1 = code.instance.potential_energy
                            ke1 = code.instance.kinetic_energy
                            e1 = ke1 + pe1
                            if numpy.abs((e1 - e0) / e0) < 0.1:
                                break
                        if numpy.abs((e1 - e0) / e0) > 0.1:    
                            print("unresolved kinetic energy pike in code with index:", index, numpy.abs((e1 - e0) / e0), numpy.abs((ke1 - start_ke) / start_ke))
                        else:    
                            print("resolved kinetic energy pike in code with index:", index, numpy.abs((e1 - e0) / e0), numpy.abs((ke1 - start_ke) / start_ke))
                        setattr(code.instance.parameters, timestep_parameter_name, original_timestep)
                        if self.is_collision_detection_set:
                            break

                self.time = max(code.model_time, self.time)
            channel = code.instance.particles.new_channel_to(particles)
            channel.copy_attributes(["x","y","z","vx","vy","vz"])
            
            t1 = pytime.time()
            statistics.append((len(code.instance.particles), t1-t0, had_energy_problem, particles))
        statistics = sorted(statistics, key = lambda x : x[1])
        if len(statistics) > 0:
            print("statistics:", id(self))
            print(statistics[0][:-1])
            if len(statistics) > 1:
                print(statistics[1][:-1])
                print(statistics[-2][:-1])
            print(statistics[-1][:-1])
            p = statistics[-1][-1]
            setid = abs(hash(tuple(p.key)))
            #write_set_to_file(p, "{0}.h5".format(setid),  "amuse", version="2.0")
        return 0
        













    def get_time(self, index, time):
        if index < 0:
            time.value = self.time
            return 0
        else:
            if not index in self.codes:
                return -2
            code = self.codes[index]
            time.value = code.model_time
            return 0
        



    def synchronize_model(self):
        return 0
        


    def add_particles(self, particles):
        return self.particles.add_particles(particles)
        




    def remove_particles(self, particles):
        self.particles.remove_particles(particles)





    def new_subset(self, index_of_the_subset, time_offset, name_of_the_code):
        subset_index = self.counter
        self.counter += 1
        index_of_the_subset.value = subset_index
        if len(name_of_the_code) == 0:
            name_of_the_code = self.name_of_the_code
        factory = self.code_names_to_code_classes[name_of_the_code].code
        code = factory(channel_type = "local")
        if name_of_the_code == "Huayno":
            code.parameters.inttype_parameter = Huayno.inttypes.SHARED4_COLLISIONS
            code.parameters.timestep_parameter = 0.001
        elif name_of_the_code == "HuaynoKepler":
            code.parameters.inttype_parameter = Huayno.inttypes.KEPLER
            code.parameters.timestep_parameter = 0.001
        elif name_of_the_code == "Huayno8":
            code.parameters.inttype_parameter = Huayno.inttypes.SHARED8_COLLISIONS
            code.parameters.timestep_parameter = 0.001
        elif name_of_the_code == "Rebound":
            code.parameters.integrator =  "ias15"
            code.parameters.timestep = 2.0e-06 | nbody_system.time
        elif name_of_the_code == "ReboundWHFast":
            code.parameters.integrator =  "whfast"
            code.parameters.timestep = 2.0e-03 | nbody_system.time
        elif name_of_the_code == "Hermite":
            code.parameters.dt_param = 0.001
        code.parameters.epsilon_squared = self.epsilon_squared
        if is_quantity(time_offset):
            code.parameters.begin_time = time_offset
        elif is_quantity(self.time):
            code.parameters.begin_time = self.time
        code.commit_parameters()
            
        if self.is_collision_detection_enabled:
            code.stopping_conditions.collision_detection.enable()
        self.codes[subset_index] = Code(code, 0 * code.parameters.begin_time)
        return 0
        












    def get_kinetic_energy(self, code_index, kinetic_energy):
        if not code_index in self.codes:
            return -2
        code = self.codes[code_index]
        kinetic_energy.value = code.instance.kinetic_energy
        return 0



    def get_potential_energy(self, code_index, potential_energy):
        if not code_index in self.codes:
            return -2
        code = self.codes[code_index]
        potential_energy.value = code.instance.potential_energy
        return 0



    def stop_subset(self, code_index):
        if not code_index in self.codes:
            return -2
        code = self.codes[code_index]
        code.instance.stop()
        del self.codes[code_index]
        return 0


    def get_index_of_the_set(self, index_of_the_particle, index_of_the_set):
        subset = self.select_particles(index_of_the_particle)
        if not len(subset) == len(index_of_the_particle):
            index_of_the_set.value = [0] * len(index_of_the_particle)
            return -1
        else:
            index_of_the_set.value = subset.index_of_the_set
            return 0



    def set_stopping_condition_timeout_parameter(self, value):
      
      return 0


    def get_stopping_condition_maximum_density_parameter(self, value):
      
      value.value = 0
      return 0


    def set_stopping_condition_out_of_box_use_center_of_mass_parameter(self, value):
      
      return 0


    def get_stopping_condition_out_of_box_use_center_of_mass_parameter(self, value):
      
      value.value = 0
      return 0


    def enable_stopping_condition(self, type):
      
        if type == 0:
            self.is_collision_detection_enabled = True
            for x in list(self.codes.values()):
                x.stopping_conditions.collision_detection.enable()
        return 0



    def get_stopping_condition_maximum_internal_energy_parameter(self, value):
      
      value.value = 0
      return 0


    def get_number_of_stopping_conditions_set(self, result):
        if self.is_collision_detection_set:
            result.value = len(self.stopping_condition_particles[0])
        else:
            results.value = 0
        return 0



    def is_stopping_condition_set(self, type, result):
        if type == 0 and self.is_collision_detection_set:
            result.value = 1
        else:
            result.value = 0
        return 0



    def set_stopping_condition_out_of_box_parameter(self, value):
      
      return 0


    def set_stopping_condition_number_of_steps_parameter(self, value):
      
      return 0


    def get_stopping_condition_timeout_parameter(self, value):
      
      value.value = 0
      return 0


    def get_stopping_condition_minimum_internal_energy_parameter(self, value):
      
      value.value = 0
      return 0


    def is_stopping_condition_enabled(self, type, result):
      
      result.value = 1 if type == 0 and self.is_collision_detection_enabled else 0
      return 0




    def get_stopping_condition_minimum_density_parameter(self, value):
      
      value.value = 0
      return 0


    def get_stopping_condition_number_of_steps_parameter(self, value):
      
      value.value = 0
      return 0


    def disable_stopping_condition(self, type):
      
        if type == 0:
            self.is_collision_detection_enabled = False
            for x in list(self.codes.values()):
                x.instance.stopping_conditions.collision_detection.disable()
        return 0



    def set_stopping_condition_minimum_internal_energy_parameter(self, value):
      
      return 0


    def set_stopping_condition_minimum_density_parameter(self, value):
      
      return 0


    def has_stopping_condition(self, type, result):
      
      result.value = 1 if type == 0 else 0
      return 0



    def set_stopping_condition_maximum_density_parameter(self, value):
      
      return 0


    def get_stopping_condition_out_of_box_parameter(self, value):
      
      value.value = 0
      return 0


    def set_stopping_condition_maximum_internal_energy_parameter(self, value):
      
      return 0


    def get_stopping_condition_info(self, index, type, number_of_particles):
      if self.is_collision_detection_set:
          if index < len(self.stopping_condition_particles[0]):
              type.value = 0
              number_of_particles.value = 2
      else:
          type.value = -1
          number_of_particles.value = 0
          return -1
      return 0



    def get_stopping_condition_particle_index(self, index, index_of_the_column, index_of_particle):
        if self.is_collision_detection_set:
            if index < len(self.stopping_condition_particles[index_of_the_column]):
                index_of_particle.value = self.stopping_condition_particles[index_of_the_column][index].key
            return 0
        else:
            index_of_particle.value = -1
            return -1

    def __init__(self):
        self.keys_generator = datamodel.BasicUniqueKeyGenerator(1)
        self.particles = datamodel.Particles()
        self.name_of_the_code = "Hermite"
        CodeDefinition = namedtuple('CodeDefinition', ['code', 'mass_unit', 'speed_unit', 'length_unit', 'time_unit'])
        self.code_names_to_code_classes = {
            "Hermite": CodeDefinition(Hermite,nbody_system.mass, nbody_system.speed, nbody_system.length, nbody_system.time),
            "Huayno": CodeDefinition(Huayno,nbody_system.mass, nbody_system.speed, nbody_system.length, nbody_system.time),
            "Rebound": CodeDefinition(Rebound,nbody_system.mass, nbody_system.speed, nbody_system.length, nbody_system.time),
            "BHTree": CodeDefinition(BHTree,nbody_system.mass, nbody_system.speed, nbody_system.length, nbody_system.time),
            "SmallN": CodeDefinition(SmallN,nbody_system.mass, nbody_system.speed, nbody_system.length, nbody_system.time),
            "OctGrav": CodeDefinition(Octgrav,nbody_system.mass, nbody_system.speed, nbody_system.length, nbody_system.time),
            "ph4": CodeDefinition(ph4,nbody_system.mass, nbody_system.speed, nbody_system.length, nbody_system.time),
            "FastKick": CodeDefinition(FastKick,nbody_system.mass, nbody_system.speed, nbody_system.length, nbody_system.time),
            "Mercury": CodeDefinition(Mercury,units.MSun, units.AUd, units.AU, units.day),
            "HuaynoKepler": CodeDefinition(Huayno,nbody_system.mass, nbody_system.speed, nbody_system.length, nbody_system.time),
            "Huayno8": CodeDefinition(Huayno,nbody_system.mass, nbody_system.speed, nbody_system.length, nbody_system.time),
            "ReboundWHFast": CodeDefinition(Rebound,nbody_system.mass, nbody_system.speed, nbody_system.length, nbody_system.time),
        }
        self.codes = {}
        self.mass_unit = nbody_system.mass
        self.speed_unit = nbody_system.speed
        self.length_unit = nbody_system.length
        self.time_unit = nbody_system.time
        self.counter = 0
        self.begin_time = 0 | nbody_system.time
        self.time = 0 | nbody_system.time
        self.epsilon_squared = 0 | (self.length_unit**2)
        self.is_collision_detection_enabled = False
        self.stopping_condition_particles = []
        self.is_collision_detection_set = False




    def get_radius(self, index_of_the_particle, radius):
        subset = self.select_particles(index_of_the_particle)
        if not len(subset) == len(index_of_the_particle):
            radius.value = [0] * len(index_of_the_particle)
            return -1
        else:
            radius.value = subset.radius
            return 0
        




    def timestep_parameter_name(self):
        if self.name_of_the_code == "Huayno":
            return None #"timestep_parameter"
        elif self.name_of_the_code == "Rebound":
            return "timestep"
        elif self.name_of_the_code == "Hermite":
            return "dt_param"
        else:
            return None

