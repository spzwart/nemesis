
import numpy
import time as pytime
from numpy import random
from amuse.units.optparse import OptionParser
from amuse.units import nbody_system
from amuse.units import units
from amuse.units import constants
from amuse.community.seba.interface import SeBa
from amuse.community.kepler.interface import Kepler
from .make_planets import make_Nice_planets
from amuse import datamodel
from amuse.io import read_set_from_file
import make_planets_uniform_distribution
from . import make_planets_oligarch
from .run_mnras_cluster import RunModel
from .run_mnras_cluster import RunModelInSingleCode

from amuse.ext.orbital_elements import new_binary_from_orbital_elements


class CreateModel(object):
    
    
    def __init__(self, m1 = 1|units.MSun, m2 = 1|units.MSun, semimajor_axis = 1000 | units.AU , eccentricity = 0.5, number_of_solar_systems = 2, cluster_radius =  2 | units.parsec, seed = -1, number_of_planets = 10, minimum_radius = 100 | units.AU, maximum_radius = 400 | units.AU):
        self.m1 = m1
        self.m2 = m2
        self.semimajor_axis = semimajor_axis
        self.eccentricity = eccentricity
        self.number_of_solar_systems = number_of_solar_systems
        self.cluster_radius = cluster_radius
        self.minimum_radius = minimum_radius
        self.maximum_radius = maximum_radius
        self.number_of_planets = number_of_planets
        self.seed = seed
        if self.seed < 0:
            self.rng = random.RandomState()
        else:
            self.rng = random.RandomState(self.seed)



    def start(self):
        stellar_model = self.create_stars()
        
        
        dynamical_model = stellar_model.copy()
        dynamical_model.subsystem = None
        dynamical_model.original_radius = dynamical_model.radius
        converter = nbody_system.nbody_to_si(1|units.MSun, 1|units.AU)
        
        kepler = Kepler(converter)
        
        systems = []
        
        stars_with_planets = dynamical_model[self.rng.permutation(len(dynamical_model))[0:self.number_of_solar_systems]]
        for star in stars_with_planets:
            system = datamodel.Particles()
            system.add_particle(star)
            system.add_particles(self.create_planets(star, kepler))
            systems.append(system)
        
        kepler.stop()
        
        for system in systems:
            parent = self.new_center_of_mass_particle(system)
            parent.subsystem = system
            dynamical_model.remove_particle(system[0])                                                                                                 # first particle is the original star
            dynamical_model.add_particle(parent)                                                                                                 # replace it with the center of mass particle
            system.position -= parent.position
            system.velocity -= parent.velocity
            
        self.recenter_systems(dynamical_model)
            

            
        dynamical_model.collection_attributes.seed = self.seed                                                                                                                                      
        dynamical_model.collection_attributes.model_time = 0 | units.yr
        dynamical_model.collection_attributes.mass_scale = dynamical_model.mass.sum()
        dynamical_model.collection_attributes.length_scale = self.cluster_radius
        
        dynamical_model.move_to_center()

        self.result = dynamical_model, stellar_model
             





    def create_planets(self, star, kepler):
        if 0:
            phi = 0
            theta = 0
            return make_Nice_planets(star, phi, theta, self.rng, kepler)
        elif 0:
            m, r, f = make_planets_oligarch.new_planet_distribution(self.minimum_radius, self.maximum_radius, 1000 | collision_code_debug.MEarth, 0.0001, mstar = star.mass )
            planets = make_planets_oligarch.make_planets(star, m, r, phi = 0, theta = None, kepler = kepler, rng = self.rng)
            return planets
        else:
            if self.seed < 0:
                rng = self.rng
            else:
                rng = random.RandomState(self.seed)
            rrange = [0,0] | units.AU
            rrange[0], rrange[1] = (self.minimum_radius, self.maximum_radius)
            planets = make_planets_uniform_distribution.make_planets(star.mass, self.number_of_planets, rrange, [0,0], [0.001,1]|units.MJupiter, 2 | units.g/units.cm**3, rng,  kepler)
            planets.position += star.position
            planets.velocity += star.velocity
            return planets
             





    def create_stars(self):
        
        result = new_binary_from_orbital_elements(self.m1, self.m2, self.semimajor_axis, self.eccentricity,true_anomaly = 180,  G = constants.G)
        result.radius = self.get_radius(result)
        return result
             



    def recenter_systems(self, model):
        for parent in self.compound_particles(model):
            parent.position+=parent.subsystem.center_of_mass()
            parent.velocity+=parent.subsystem.center_of_mass_velocity()
            parent.subsystem.move_to_center()      


    def new_center_of_mass_particle(self, particles):
        result = datamodel.Particle()
        result.mass = particles.mass.sum()
        result.position = particles.center_of_mass()
        result.velocity = particles.center_of_mass_velocity()
        return result

    def compound_particles(self, model):
        return model[~numpy.equal(model.subsystem, None)]      

    def get_radius(self, stars):
        code = SeBa()
        code.particles.add_particles(stars)
        result = code.particles.radius     
        code.stop()
        return result

def new_option_parser():
    result = OptionParser()
    result.add_option("--seed", 
                      dest="seed", type="int", default = -1,
                      help="random number seed [%default]")
    result.add_option("-i", "--input",
                      dest="filename", default = "",
                      help="name of the input filename for the initial modal (default empty, model will be generated)")
    result.add_option("-o", "--output",
                      dest="output_filename", default = "",
                      help="name of the output filename (default cluster-planets-<computer node name>.h5)")
    result.add_option("--index",
                      dest="set_index", type="int", default = -1,
                      help="index in the reversed history of the set to restart the calculation from")
    result.add_option("--save-step",
                      dest="save_step", type="int", default = 10,
                      help="number of large steps to do before saving (default 10)")
    result.add_option("--large-step",
                      dest="large_step", type="int", default = 1000,
                      help="number of small steps in one large step, escaper detection is done per large step, so is time reporting(default 1000)")
    result.add_option("-t", unit=units.yr,
                      dest="endtime", type="float", default = 2000|units.yr,
                      help="time to run to [%default]")
    result.add_option("--dt", unit=units.yr,
                      dest="dt", type="float", default = 1|units.yr,
                      help="delta time between bridging [%default]")
    result.add_option("--m1", unit=units.MSun,
                      dest="m1", type="float", default = 1 | units.MSun,
                      help="mass of the first star [%default]")
    result.add_option("--m2", unit=units.MSun,
                      dest="m2", type="float", default = 1 | units.MSun,
                      help="mass of the second star [%default]")
    result.add_option("-a", unit=units.AU,
                      dest="semimajor_axis", type="float", default = 5000 | units.AU,
                      help="semimajor axis [%default]")
    result.add_option("-e", "--eccentricity",
                      dest="eccentricity", type="float", default = 0.6,
                      help="eccentricity [%default]")
    result.add_option("-M", "--nsolar",
                      dest="number_of_solar_systems", type="int", default = 2,
                      help="Number of stars with solar systems in the cluster [%default]")
    
    result.add_option("-P", "--nplanets",
                      dest="nplanets", type="int", default = 1,
                      help="Number of planets in a solar system [%default]")
    result.add_option("--rmin", unit=units.AU,
                      dest="minimum_radius", type="int", default = 100 | units.AU,
                      help="Minimin radius of the planetary disk [%default]")
    result.add_option("--rmax", unit=units.AU,
                      dest="maximum_radius", type="int", default = 400  | units.AU,
                      help="Maximum radius of the planetary disk [%default]")
   
    result.add_option("--radius", unit=units.parsec,
                      dest="cluster_radius", type="float", default = 2 | units.parsec,
                      help="radius of the cluster [%default]")
    result.add_option("--code", 
                      dest="code", default = "huayno",
                      help="name of the code to evolve the model with (not used at restart)")
    result.add_option("--new-code", 
                      dest="newcode", default = "",
                      help="name of the code to evolve the model further (used at restart)")
    result.add_option("--no-escapers",
                  action="store_false", dest="with_escapers", default=True,
                  help="don't detect escapers")
    result.add_option("--no-kick",
                  action="store_false", dest="must_kick_subsystems", default=True,
                  help="don't kick the subsystems, subsystems will evolve without the field of the system")
    result.add_option("--no-evolve",
                  action="store_false", dest="must_evolve", default=True,
                  help="don't evolve the cluster, just the subsystems in the field of the the cluster")
    result.add_option("--no-run",
                  action="store_false", dest="must_run", default=True,
                  help="don't run the cluster, just create the initial conditions")
    result.add_option("-Z", "--ncodes",
                      dest="number_of_sharing_codes", type="int", default = 10,
                      help="Number of codes to divide the subsystems over")
    result.add_option("--dynamic-dt",
                  action="store_true", dest="has_dynamic_timestep", default=False,
                  help="use dynamic / block timesteps")
    result.add_option("--single",
                  action="store_true", dest="must_run_single_code", default=False,
                  help="use one code to do all the gravitational dynamics")
    result.add_option("--timestep", unit=units.day,
                      dest="timestep", type="float", default = 0 | units.day,
                      help="fixed timestep, used by some codes (rebound) [%default]")
    return result






def print_model(model):
    print('*' * 50)
    print(model.collection_attributes)
    print('*' * 50)
    

def runs():
    """
    lgm05 mpiexec.hydra python run_planets_in_cluster.py -N 1024 -M 10 --dt 5 --seed 512 --large-step 40 --save-step 5 -t 1000000 --output cluster-5yr.h5
    lgm02 mpiexec.hydra python run_planets_in_cluster.py -N 1024 -M 10 --dt 10 --seed 512 --large-step 40 --save-step 1 -t 1000000 --output cluster-10yr.h5
    lgm14 mpiexec.hydra python run_planets_in_cluster.py -N 1024 -M 10 --dt 50 --seed 512 --large-step 40 --save-step 1 -t 1000000 --output cluster-50yr.h5
    lgm07 mpiexec.hydra python run_planets_in_cluster.py -N 1024 -M 10 --dt 100 --seed 512 --large-step 40 --save-step 1 -t 1000000 --output cluster-100yr.h5
    lgm14 mpiexec.hydra python run_planets_in_cluster.py -N 1024 -M 10 --dt 1000 --seed 512 --large-step 40 --save-step 1 -t 1000000 --output cluster-1000yr.h5
    lgm18 mpiexec.hydra python run_planets_in_cluster.py -N 1024 -M 10 --dt 10000 --seed 512 --large-step 40 --save-step 1 -t 1000000 --output cluster-10000yr.h5
    
    lgm18 mpiexec.hydra -f hostfile -n 1 python run_planets_in_cluster.py -N 1024 -M 159 --dt 50 --seed 512 --large-step 40 --save-step 1 -t 1000000 --output cluster-50yr.h5
    lgm18 mpiexec.hydra -f hostfile -n 1 python run_planets_in_cluster.py -N 1024 -M 512 --dt 50 --seed 512 --large-step 40 --save-step 1 -t 1000000 --output production-50yr.h5
    lgm18 mpiexec.hydra -f hostfile -n 1 python run_planets_in_cluster.py -N 1024 -M 1024 --dt 100 --seed 512 --large-step 40 --save-step 1 -t 1000000 --output production2-100yr.h5
    lgm18 mpiexec.hydra -f hostfile -n 1 python run_planets_in_cluster.py -N 1024 -M 1024 --dt 100 --seed 512 --large-step 40 --save-step 1 -t 1000000 --output production3-100yr.h5
    
    lgm18 mpiexec.hydra -f hostfile -n 1 python run_planets_in_cluster.py --input production3-100yr.h5 --dt 100 --large-step 40 --save-step 1 -t 1000000 --output production4-100yr.h5
    
    
    lgm04 mpiexec.hydra python run_planets_in_cluster.py --seed 512 --dt=1000 -t 1000000 --save-step=10 --large-step=1 -N 1024 -M 1 --no-kick --no-evolve --timestep 1 --output rebound-nokick-noevolve-1.h5
    lgm09 mpiexec.hydra python run_planets_in_cluster.py --seed 512 --dt=1000 -t 1000000 --save-step=10 --large-step=1 -N 1024 -M 1 --no-kick --no-evolve --timestep 10 --output rebound-nokick-noevolve-10.h5
    lgm12 mpiexec.hydra python run_planets_in_cluster.py --seed 512 --dt=1000 -t 1000000 --save-step=10 --large-step=1 -N 1024 -M 1 --no-kick --no-evolve --timestep 100 --output rebound-nokick-noevolve-100.h5
    lgm13 mpiexec.hydra python run_planets_in_cluster.py --seed 512 --dt=1000 -t 1000000 --save-step=10 --large-step=1 -N 1024 -M 1 --no-kick --no-evolve --timestep 1000 --output rebound-nokick-noevolve-1000.h5
    lgm14 mpiexec.hydra python run_planets_in_cluster.py --seed 512 --dt=1000 -t 1000000 --save-step=10 --large-step=1 -N 1024 -M 1 --no-kick --no-evolve --timestep 0.1 --output rebound-nokick-noevolve-01.h5
    
    lgm09 get rebound-nokick-noevolve-1.h5 rebound-nokick-noevolve-10.h5 rebound-nokick-noevolve-100.h5 rebound-nokick-noevolve-1000.h5 rebound-nokick-noevolve-01.h5
    
    
    lgm14 mpiexec.hydra python run_planets_in_cluster.py --seed 512 --dt=1 -t 1000000 --save-step=10 --large-step=1000 -N 1024 -M 1 --no-kick --no-evolve --timestep 10 --output rebound-noevolve-1.h5
    lgm09 mpiexec.hydra python run_planets_in_cluster.py --seed 512 --dt=10 -t 1000000 --save-step=10 --large-step=100 -N 1024 -M 1 --no-kick --no-evolve --timestep 10 --output rebound-noevolve-10.h5
    lgm12 mpiexec.hydra python run_planets_in_cluster.py --seed 512 --dt=100 -t 1000000 --save-step=10 --large-step=10 -N 1024 -M 1 --no-kick --no-evolve --timestep 10 --output rebound-noevolve-100.h5
    lgm11 mpiexec.hydra python run_planets_in_cluster.py --seed 512 --dt=1000 -t 1000000 --save-step=10 --large-step=1 -N 1024 -M 1 --no-kick --no-evolve --timestep 10 --output rebound-noevolve-1000.h5
    
    
    lgm03 mpiexec.hydra python run_planets_in_cluster.py --seed 512 --dt=1 -t 1000000 --save-step=10 --large-step=1000 -N 1024 -M 1 --no-evolve --timestep 10 --output rebound-kick-noevolve-1.h5
    lgm04 mpiexec.hydra python run_planets_in_cluster.py --seed 512 --dt=10 -t 1000000 --save-step=10 --large-step=100 -N 1024 -M 1 --no-evolve --timestep 10 --output rebound-kick-noevolve-10.h5
    lgm07 mpiexec.hydra python run_planets_in_cluster.py --seed 512 --dt=100 -t 1000000 --save-step=10 --large-step=10 -N 1024 -M 1 --no-evolve --timestep 10 --output rebound-kick-noevolve-100.h5
    lgm12 mpiexec.hydra python run_planets_in_cluster.py --seed 512 --dt=1000 -t 1000000 --save-step=10 --large-step=1 -N 1024 -M 1 --no-evolve --timestep 10 --output rebound-kick-noevolve-1000.h5
    
    lgm14 kill -9
    lgm09 --nosync kill -9
    lgm12 --nosync kill -9
    lgm11 --nosync kill -9
    lgm03 --nosync kill -9
    lgm04 --nosync kill -9
    lgm07 --nosync kill -9
    lgm12 --nosync kill -9
    
    """
    


def setup_delta(host="localhost", port=1883):
    try:
        print("setting up delta")
        from wrench import delta
        from paho.mqtt import client
        
        queue = client.Client()
        queue.connect(host, port, 60)
        queue.loop_start()
        
        replay = delta.ReplayRecordsFromQueue(queue)
        replay.start()
        return replay
    except Exception as ex:
        print(ex)
        pass

def new_binary(
        mass1, mass2, semi_major_axis,
        eccentricity = 0, keyoffset = 1,
        is_at_periapsis = True,
        G =  nbody_system.G 
    ):
    print(mass1, mass2)
    total_mass = mass1 + mass2
    mass_fraction_particle_1 = mass1 / (total_mass)

    binary = datamodel.Particles(keys=list(range(keyoffset, keyoffset+2)))
    binary[0].mass = mass1
    binary[1].mass = mass2
    binary.child1 = None
    binary.child2 = None
    
    mu = G * total_mass

    if is_at_periapsis:
        velocity = numpy.sqrt( mu / semi_major_axis  * ((1.0 + eccentricity)/(1.0 - eccentricity)))
        radius   = semi_major_axis * (1.0 - eccentricity)
    else:
        velocity = numpy.sqrt( mu / semi_major_axis  * ((1.0 - eccentricity)/(1.0 + eccentricity)))
        radius   = semi_major_axis * (1.0 + eccentricity)
        
    binary[0].position = ((1.0 - mass_fraction_particle_1) * radius * [1.0,0.0,0.0])
    binary[1].position = -(mass_fraction_particle_1 * radius * [1.0,0.0,0.0])

    binary[0].velocity = ((1.0 - mass_fraction_particle_1) * velocity * [0.0,1.0,0.0])
    binary[1].velocity = -(mass_fraction_particle_1 * velocity * [0.0,1.0,0.0])   

    return binary
       

if __name__ == "__main__":
    #import fast_halt2
    #fast_halt2.make_managable(None)
    if 0:
        setup_delta()
    o, arguments  = new_option_parser().parse_args()
    #    random.seed(seed=o.seed)
    
    is_restart = o.filename and os.path.exists(o.filename)
    if is_restart:
        with read_set_from_file(o.filename, "amuse", close_file=False, return_context = True, names = ('dynamical_model', 'stars')) as (particles_set, stars):
            if o.set_index >= 0:
                model = list(reversed(list(particles_set.iter_history())))[o.set_index]
                model = model.copy()
            else:
                model = particles_set.copy()
            stars = stars.copy()
                                
            if o.newcode:
                model.collection_attributes.code_name = o.newcode
        
    else:
        uc =  CreateModel(o.m1, o.m2, o.semimajor_axis, o.eccentricity,o.number_of_solar_systems, seed = o.seed, number_of_planets = o.nplanets, minimum_radius = o.minimum_radius, maximum_radius = o.maximum_radius)
        uc.start()
        model, stars = uc.result
        model.collection_attributes.code_name = o.code
        model.collection_attributes.with_escapers = o.with_escapers
    
    runner = RunModelInSingleCode if o.must_run_single_code else RunModel
    
    uc = runner(
        model,
        stars,
        dt = o.dt,
        output_filename = o.output_filename,
        save_step = o.save_step,
        large_step = o.large_step,
        must_kick_subsystems = o.must_kick_subsystems,
        timestep = o.timestep,
        must_evolve = o.must_evolve,
        has_dynamic_timestep = o.has_dynamic_timestep,
        number_of_sharing_codes = o.number_of_sharing_codes
    )
    
    if not is_restart or not (o.output_filename == o.filename):
        uc.save(False)
        
    print_model(uc.model)
    if o.must_run:
        t0 = pytime.time() | units.s
        uc.run(o.endtime)
        t1 = pytime.time() | units.s
        print_model(uc.model)
        print("total time of this run in seconds: ", t1 - t0)
        print("number of days to reach 10 million years:", \
            (((10.0 | units.Myr)/uc.time)*(t1 - t0)).as_quantity_in(units.day))





