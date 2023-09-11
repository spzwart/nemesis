import platform
import numpy
import time as pytime
import os
from numpy import random
from amuse.community.hermite0.interface import Hermite
from amuse.community.huayno.interface import Huayno
from amuse.community.kepler.interface import Kepler
from amuse.community.mercury.interface import Mercury
from amuse.units.optparse import OptionParser
from amuse.community.rebound.interface import Rebound
from amuse.community.seba.interface import SeBa
from amuse.units import constants
from amuse import datamodel
from amuse.units import nbody_system
from amuse.ic.fractalcluster import new_fractal_cluster_model
from amuse.ic.kroupa import new_kroupa_mass_distribution
from amuse.community.ph4.interface import ph4
from amuse.io import read_set_from_file
from amuse.units import units
from amuse.io import write_set_to_file
from .nemesis import CalculateFieldForParticles
from .nemesis import Nemesis
from .nemesis import MultiGravityCodeSubsystem
from . import collision_code
import make_planets_salpeter

from .nemesis import Subsystem
class CreateModel(object):
    
    
    def __init__(self, number_of_stars = 1024, number_of_solar_systems = 10, cluster_radius =  2 | units.parsec, seed = -1):
        self.number_of_stars = number_of_stars
        self.number_of_solar_systems = number_of_solar_systems
        self.cluster_radius = cluster_radius
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
            planets = self.create_planets(star, kepler)
            if planets is None or len(planets) == 0:
                continue
            system = datamodel.Particles()
            system.add_particle(star)
            system.add_particles(planets)
            systems.append(system)
        
        kepler.stop()
        
        for system in systems:
            parent = self.new_center_of_mass_particle(system)
            parent.subsystem = system
            dynamical_model.remove_particle(system[0])                                                                                                         # first particle is the original star
            dynamical_model.add_particle(parent)                                                                                                         # replace it with the center of mass particle
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
        return make_a_planet(star, phi = 0, theta = 0, rng = self.rng, kepler = kepler)
             






    def create_stars(self):
        mass = new_kroupa_mass_distribution(self.number_of_stars, mass_max = 10 | units.MSun, random = self.rng)
        total_mass = mass.sum()
        converter = nbody_system.nbody_to_si(total_mass, self.cluster_radius)
        randomseed = self.seed if self.seed > 0 else None
        result = new_fractal_cluster_model(convert_nbody = converter, fractal_dimension = 1.6, masses = mass, random_seed = randomseed)
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
    result.add_option("-N", "--ncluster",
                      dest="ncluster", type="int", default = 1000,
                      help="Number of stars in the cluster[%default]")
    result.add_option("-M", "--nsolar",
                      dest="number_of_solar_systems", type="int", default = 10,
                      help="Number of stars with solar systems in the cluster[%default]")
    result.add_option("-Z", "--ncodes",
                      dest="number_of_sharing_codes", type="int", default = 10,
                      help="Number of codes to divide the subsystems over")
    
   
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
    result.add_option("--dynamic-dt",
                  action="store_true", dest="has_dynamic_timestep", default=False,
                  help="use dynamic / block timesteps")
    
    
    result.add_option("--timestep", unit=units.day,
                      dest="timestep", type="float", default = 0 | units.day,
                      help="fixed timestep, used by some codes (rebound) [%default]")
    return result





class SubsetCode(object):
    def __init__(self, real_code, subset, G = constants.G):
        self.real_code = real_code
        self.subset = subset
        if self.subset == 0:
            self.real_code._cached_model_time = self.real_code.model_time
            self.real_code._managing_subset = 0
            self.real_code._valid_subsets = set([])
            self.real_code._cached_particles = datamodel.Particles()
            self.real_code.channel_to_model = self.real_code.particles.new_channel_to(self.real_code._cached_particles)
            self.real_code.channel_to_code = self.real_code._cached_particles.new_channel_to(self.real_code.particles)
            self.t0 = self.real_code.get_time(self.subset)
            self.real_code.t0 = self.t0
        elif self.real_code._managing_subset < 0:
            self.real_code._managing_subset = self.subset

        
        self.t0 = self.real_code.get_time(self.subset)
            
        self.particles = datamodel.UpdatingParticlesSubset(self.real_code._cached_particles, lambda x  : x.subset == self.subset)
        self.real_code._valid_subsets.add(self.subset)
        self.G = G





    def evolve_model(self, end_time, only_subset = False):
        if only_subset:
            collision_detection = self.real_code.stopping_conditions.collision_detection
            self.real_code.channel_to_code.copy()
            if self.real_code.t0 is None:
                self.real_code.t0 = self.t0
            self.real_code.evolve_model(end_time + self.real_code.t0, self.subset)
            self.real_code._cached_model_time = self.real_code.get_time(self.subset)
            self.real_code.particles.synchronize_to(self.real_code._cached_particles)
            self.real_code.channel_to_model.copy()
            if collision_detection.is_set():
                self.handle_collision(datamodel.Particles(particles = [collision_detection.particles(0)[0], collision_detection.particles(1)[0]]))
        elif self.subset == self.real_code._managing_subset:
            collision_detection = self.real_code.stopping_conditions.collision_detection
            self.real_code.channel_to_code.copy()
            if self.real_code.t0 is None:
                self.real_code.t0 = self.t0
            self.real_code.evolve_model(end_time + self.real_code.t0)
            self.real_code._cached_model_time = self.real_code.get_time(self.subset)
            self.real_code.particles.synchronize_to(self.real_code._cached_particles)
            self.real_code.channel_to_model.copy()
            if collision_detection.is_set():
                self.handle_collision(datamodel.Particles(particles = [collision_detection.particles(0)[0], collision_detection.particles(1)[0]]))





    @property
    def model_time(self):
        return self.real_code._cached_model_time
   
    
    def stop(self):
        if self.subset >= 0:
            self.real_code._valid_subsets.remove(self.subset)                                                                               
            if self.subset == self.real_code._managing_subset:
                self.real_code._managing_subset = -2
                for i in self.real_code._valid_subsets:
                    self.real_code._managing_subset = i
                    self.real_code.t0 = None
                    break            
            particles = self.real_code.particles
            particles.remove_particles(particles[particles.subset == self.subset])
            
            particles = self.real_code._cached_particles
            particles.remove_particles(particles[particles.subset == self.subset])
            self.real_code.stop_subset(self.subset)
            self.subset = -1






    @property
    def kinetic_energy(self):
        return self.real_code.get_kinetic_energy(self.subset)  


    @property
    def potential_energy(self):
        return self.real_code.get_potential_energy(self.subset)




    def update(self):
        self.real_code.particles.synchronize_to(self.real_code._cached_particles)
        
    def handle_collision(self, collision):
        
        print(len(self.real_code._cached_particles), len(self.real_code.particles))
        
        #collision = self.collided_particles.add_particles(collision)
        p = datamodel.Particles()
        collision = p.add_particles(collision)
        collision_in_real_code = collision.get_intersecting_subset_in(self.real_code._cached_particles)
        subset = collision_in_real_code.subset[0]
        print("handle collision in subset:", subset)
        print("keys:", collision.key)
        
        k0 = self.real_code.get_kinetic_energy(subset)                              
        p0 = self.real_code.get_potential_energy(subset)                              
        self.real_code.particles.remove_particles(collision)
        self.real_code._cached_particles.remove_particles(collision)
        
        total_mass = collision.mass.sum()
        merger_product = datamodel.Particle()
        merger_product.mass = total_mass
        merger_product.position = (collision.position.sum(0)) / 2.0
        merger_product.velocity = ((collision.velocity * collision.mass.reshape((len(collision),1)) / total_mass)).sum(0)
        merger_product.radius = (collision.radius**3).sum()**(1.0/3.0)                                                                                                                                                          
        merger_product.ancestors = collision                                  
        merger_product.subset =    subset                                                                                                                                         
        self.real_code.particles.add_particle(merger_product)                                      
        
        delta_kinetic_energy = -0.5 * (collision.mass.prod()/total_mass) * (collision.velocity[1] - collision.velocity[0]).length_squared()
        delta_potential_energy =  self.G * collision.mass.prod() /  (collision.position[1] - collision.position[0]).length()
        k1 = self.real_code.get_kinetic_energy(subset)                              
        p1 = self.real_code.get_potential_energy(subset)                              
        print(p1 - p0, delta_potential_energy, ( p1 - p0)/ delta_potential_energy)
        print(k1 - k0, delta_kinetic_energy, ( k1 - k0)/ delta_kinetic_energy)
        #self.total_delta_kinetic_energy += delta_kinetic_energy
        #self.total_delta_potential_energy += p1-p0
        
        merger_product.delta_kinetic_energy = delta_kinetic_energy
        merger_product.delta_potential_energy = delta_potential_energy                             
        
                                                                                  
        self.real_code._cached_particles.add_particle(merger_product)
        print(len(self.real_code._cached_particles), len(self.real_code.particles))
        if False and self.stopping_conditions.collision_detection.is_enabled():
            self.stopping_conditions.collision_detection.set(
                collision[0].as_set(),
                collision[1].as_set(),
                merger_product.as_set(),
            )


class RunModel(object):
    
    def __init__(self, model, stars, dt = None, code_name = None,with_escapers = None, output_filename = "", save_step = 10, large_step = 1000, must_kick_subsystems = True , timestep = 0 | units.s, must_evolve = True, has_dynamic_timestep = True, use_threading = False, number_of_sharing_codes = 10):
        self.model = model
        self.stars = stars
        self.star = self.model[0]
        self.time = self.model.collection_attributes.model_time
        self.time_offset = self.time
        self.channels = []
        self.from_model_to_code_channels = []
        
        self.dt = self._get_value_from_model(dt, self.model, "delta_t", 1 | units.yr)
        self.code_name = self._get_value_from_model(code_name, self.model, "code_name", "hermite")
        self.with_escapers = False                                                 #self._get_value_from_model(with_escapers, self.model, "with_escapers", False)
        self.output_filename = output_filename
        self.save_step = save_step
        self.large_step = large_step
        self.must_kick_subsystems = must_kick_subsystems
        self.subsystem_code_timestep = timestep
        self.must_evolve = must_evolve
        self.has_dynamic_timestep = has_dynamic_timestep
        self.number_of_sharing_codes = number_of_sharing_codes
        self.use_threading = use_threading
        










    def _get_value_from_model(self, proposed_value, model, name, default_value):
        if hasattr(model.collection_attributes, name):
            return getattr(model.collection_attributes, name)
        else:
            if proposed_value is None:
                return default_value
            else:
                return proposed_value
            

    def log(self, *args):
        string = ' '.join([str(x) for x in args])
        string += '\n'
        sys.stderr.write(string)
        
    def remove_escaping_particles(self, star, particles, radius, kind = "pebble(s)"):
        escapers = particles[particles.position.lengths()>radius]
        if len(escapers) == 0:
            return escapers
        if self.check_if_bound_on_escaping:
            escapers = escapers[numpy.asarray([is_bound(x, star) for x in escapers])]
        if len(escapers) == 0:
            return escapers
        self.log("removing", len(escapers), " escaping ", kind)
        result = escapers.copy()
        result.escape_time = self.time
        particles.remove_particles(escapers)
        return result
    
    def remove_escaping_pebbles(self, star, radius):
        if len(star.disk_particles) > 0:
            escapers = self.remove_escaping_particles(star, star.disk_particles, radius, "pebble(s)")
            star.escaped_disk_particles.add_particles(escapers)
        
    def remove_escaping_planets(self, star, radius):
        escapers = self.remove_escaping_particles(star, star.planets, radius, "planet(s)")
        star.escaped_planets.add_particles(escapers)
        
    def length_scale(self):
        return self.model.collection_attributes.length_scale
        


    def mass_scale(self):
        return self.model.collection_attributes.mass_scale
    

    def new_converter(self):
        return nbody_system.nbody_to_si(self.mass_scale(), self.length_scale())
        
    def get_code_factory(self):
        if self.code_name == "hermite":
            class Hermite2(Hermite):
                def __init__(self, *args, **kwargs):
                    Hermite.__init__(self,  *args, **kwargs)
                    self.parameters.dt_param = 0.001
                    self.parameters.end_time_accuracy_factor = 0.0
        
            return Hermite2
        elif self.code_name == "huayno":
            class CollisionHuayno(Huayno):
                def __init__(self, *args, **kwargs):
                    Huayno.__init__(self,  *args, **kwargs)
                    #self.parameters.inttype_parameter = Huayno.inttypes.BRIDGE_KDK
                    self.parameters.inttype_parameter = Huayno.inttypes.SHARED4_COLLISIONS
           
            return CollisionHuayno
        elif self.code_name == "mercury":
            return Mercury
        elif self.code_name == "ph4":
            return ph4
            
    







    def create_code(self):
        self.sharing_codes = self.make_sharing_codes(self.number_of_sharing_codes)

        self.cycle_index = 0

        code = self.make_cluster_code()

        self.dt_param = 0.2

        result = Nemesis(
            code, 
            self.make_shared_code if self.number_of_sharing_codes > 0 else self.make_code_for_subsystem, 
            self.make_gravity_code, 
            self.dt,
            must_kick_subsystems = self.must_kick_subsystems, 
            #calculate_radius = radius,
            use_async = False,
            subsystem_factory = MultiGravityCodeSubsystem if self.number_of_sharing_codes > 0 else Subsystem,
            use_threading = self.use_threading,
            eta = self.dt_param,
            threshold = self.dt
        )
        #result.distfunc = timestep
        #result.threshold = self.dt                                                                                                                                                                                                     # (400 | units.AU) **2
        result.particles.add_particles(self.model)
        result.commit_particles()
        return result
        
















    def copy_to_model(self):
        for channel in self.channels:
            channel.copy()
    
    
    def put_parameters_in_model(self):
        self.model.collection_attributes.delta_t = self.dt
        self.model.collection_attributes.code_name = self.code_name
        self.model.collection_attributes.with_escapers = self.with_escapers
        self.model.collection_attributes.model_time = self.time
        self.model.collection_attributes.must_kick_subsystems = self.must_kick_subsystems
        self.model.collection_attributes.subsystem_code_timestep = self.subsystem_code_timestep
        self.model.collection_attributes.must_evolve = self.must_evolve
        
    





    def get_filename(self):
        if self.output_filename:
            return self.output_filename
        else:
            return "cluster-planets-{0}.h5".format(platform.node())
    


    def save(self, append_to_file = True):
        self.put_parameters_in_model()
        if append_to_file:
            names = ('dynamical_model',)
            sets = (self.model,)
        else:
            names = ('dynamical_model', 'stars')
            sets = (self.model, self.stars)
        write_set_to_file(sets, self.get_filename(), "amuse", close_file = True, append_to_file=append_to_file, version="2.0", names = names)
        



    def run(self, end_time):
        self.converter = self.new_converter()
        self.code = self.create_code()
        #self.code = self.create_code_single()
        self.stellar_evolution_code = self.create_se_code()
        print(self.model.collection_attributes.length_scale)
        istep = 0
        
        t0 = pytime.time() | units.s
        t1 = t2 = t0
        
        e0 = self.code.kinetic_energy + self.code.potential_energy
        p = all_particles(self.code.particles)
        E0 = p.kinetic_energy() + p.potential_energy(G=constants.G)
        print(len(self.code.particles))
        print("starting at time: ", self.time.as_quantity_in(units.yr))
        try:
            while self.time < end_time:
                
                t1 = t2
                istep += 1
                self.time  += self.large_step*self.dt
                print("evolving to time: ", self.time.as_quantity_in(units.yr), (self.time - self.time_offset).as_quantity_in(units.yr))
                self.code.evolve_model(self.time - self.time_offset)
                self.report_timings(self.code)
                self.synchronize_sets(self.code.particles, self.model)
                before = self.model.mass.sum()
               
                if 0:
                    self.stellar_evolution_code.evolve_model(self.time)
                    
                    for channel in self.get_se_to_model_channels(self.stellar_evolution_code, self.model):
                        channel.copy_attribute("mass")
                    for channel in self.get_from_model_to_code_channels(self.code, self.model):
                        channel.copy_attribute("mass")

                    self.code.update_systems()
                
                after = self.model.mass.sum()
                
                e1 = self.code.kinetic_energy + self.code.potential_energy
                print(istep, " evolved to time: ", self.time.in_(units.yr),  (e1 - e0) / e0, (after-before).as_quantity_in(units.MSun))
                
                p = all_particles(self.code.particles)
                E1 = p.kinetic_energy() + p.potential_energy(G=constants.G)
                print(istep, " evolved to time: ", (E1 - E0) / E0)
                
                if istep%self.save_step == 0:
                    
                    for channel in self.get_from_code_to_model_channels(self.code, self.model):
                        channel.copy()
                    self.save()
                    self.time_offset = self.time
                    p2 = self.load()
                    
                    p2.radius = p2.original_radius
                    
                    self.stop_code()
                    self.model = p2
                    self.code = self.create_code()
                    
                if self.with_escapers and istep%20 == 0:
                    self.remove_escaping_pebbles(self.star, self.escape_radius)
                    self.remove_escaping_planets(self.star, self.escape_radius)
                
                t2 = pytime.time() | units.s
                print("delta t:", time_to_string(t2-t1), " ---- time to reach 10 million years:", time_to_string((((10.0 | units.Myr) - self.time)/self.time)*(t2 - t0)))
                self.print_statistics_on_subsystems()
                
        except:
            #self.code.report_timings()
            self.code.particles.synchronize_to(self.model)
            for channel in self.get_from_code_to_model_channels(self.code, self.model):
                channel.copy()
            self.save()
            raise
        
    













    def create_se_code(self):
        factory = SeBa
        code = factory()
        code.particles.add_particles(self.stars)
                
        
        stars_without_planets = self.model[numpy.equal(self.model.subsystem, None)]
        systems_with_planets = self.model[~numpy.equal(self.model.subsystem, None)]
        
        self.se_to_model_channels = []
        self.se_to_model_channels.append(code.particles.new_channel_to(stars_without_planets))
        for x in systems_with_planets:
            self.se_to_model_channels.append(code.particles.new_channel_to(x.subsystem))
        
        code.evolve_model(self.time_offset)
        
        return code





    def get_se_to_model_channels(self, code, model):
        stars_without_planets = model[numpy.equal(model.subsystem, None)]
        systems_with_planets = model[~numpy.equal(model.subsystem, None)]
        
        result = []
        result.append(code.particles.new_channel_to(stars_without_planets))
        for x in systems_with_planets:
            result.append(code.particles.new_channel_to(x.subsystem))
       
        return result




    def get_from_model_to_code_channels(self, code, model):
        return self.new_channels_for(model, code.particles)







    def get_from_code_to_model_channels(self, code, model):
        return self.new_channels_for(code.particles, model)







    def create_code_single(self):
        
        result = self.get_code_factory()(self.new_converter())
        result.particles.add_particles(self.model)
        result.commit_particles()
        self.channels.append(result.particles.new_channel_to(self.model))
        self.from_model_to_code_channels.append(self.model.new_channel_to(result.particles))
        
       
        return result
        











    def load(self):
        with read_set_from_file(self.get_filename(), "amuse", close_file=False, return_context = True, names = ('dynamical_model',)) as (particles_set,):
            return particles_set.copy()


    def report_timings(self, code):
        code.report_timings()
    
    def print_statistics_on_subsystems(self):
        print("statistics: number of subsystems", len(self.code.compound_particles()))
    
    def stop_code(self):
        self.code.stop()
        for x in self.sharing_codes:
            x.stop()
        self.code.cluster_code.stop()
        
    def synchronize_sets(self, from_set, to_set):
        from_set.synchronize_to(to_set)
        
        if not hasattr(from_set, 'subsystem'):
            return
        
        to_particles_with_subsystems = to_set[~numpy.equal(to_set.subsystem, None)]
        from_particles_with_subsystems = from_set[~numpy.equal(from_set.subsystem, None)]
        
        if len(to_particles_with_subsystems) != len(from_particles_with_subsystems):
            raise Exception("the number of particles with subsytems do not match!")

        from_in_order = sorted(from_particles_with_subsystems, key=lambda x : x.key)
        to_in_order = sorted(to_particles_with_subsystems, key=lambda x : x.key)
        for x, y in zip(from_in_order,to_in_order):
            if x.key != y.key:
                raise Exception("the parent keys of 2 subsystems do not match, they should ({0},{1})".format(x.key, y.key))
            else:
                x.subsystem.synchronize_to(y.subsystem)
       







    def new_channels_for(self, from_set, to_set):
        result = []
        result.append(from_set.new_channel_to(to_set))
        
        if not hasattr(from_set, 'subsystem'):
            return
        
        from_particles_with_subsystems = from_set[~numpy.equal(from_set.subsystem, None)]
        to_particles_with_subsystems = to_set[~numpy.equal(to_set.subsystem, None)]
        
        if len(to_particles_with_subsystems) != len(from_particles_with_subsystems):
            raise Exception("the number of particles with subsytems do not match!")

        from_in_order = sorted(from_particles_with_subsystems, key=lambda x : x.key)
        to_in_order = sorted(to_particles_with_subsystems, key=lambda x : x.key)
        for x, y in zip(from_in_order,to_in_order ):
            if x.key != y.key:
                raise Exception("the parent keys of 2 subsystems do not match, they should ({0},{1})".format(x.key, y.key))
            else:
                result.append(x.subsystem.new_channel_to(y.subsystem))
       
        return result







    def make_code_for_subsystem(self, particles):
        #converter = nbody_system.nbody_to_si(particles.mass.sum(), particles.position.lengths().max())
        converter = nbody_system.nbody_to_si(particles.mass.sum(),  particles.position.lengths().min())
        code = Rebound(converter)
        code.initialize_code()
        code.parameters.stopping_conditions_timeout = 300 | units.s
        #code.parameters.integrator =  "whfast"
        if self.subsystem_code_timestep > 0 |units.s:
            code.parameters.timestep = self.subsystem_code_timestep
        
            
        if code.stopping_conditions.timeout_detection.is_supported():
            code.stopping_conditions.timeout_detection.enable()
        
            
        if False and code.stopping_conditions.out_of_box_detection.is_supported(): 
            code.parameters.stopping_conditions_out_of_box_size = 3000 | units.AU
            code.parameters.stopping_conditions_out_of_box_use_center_of_mass = False
            code.stopping_conditions.out_of_box_detection.enable()
        
        result = collision_code.CollisionCode(code, G = constants.G)
        result.stopping_conditions.collision_detection.enable()
        result.particles.add_particles(particles)
        result.commit_particles()
        result.particles.ancestors = None
        return result
    def make_nonworking_code(particles):
        result = NullCode()
        result.particles = particles.copy()
        return result
        
        
    def make_shared_rebound_code():
            
        #converter = nbody_system.nbody_to_si(particles.mass.sum(), particles.position.lengths().max())
        converter = nbody_system.nbody_to_si(1 | units.MSun,  100 | units.AU)
        code = Rebound(converter)
        code.initialize_code()
        code.parameters.stopping_conditions_timeout = 300 | units.s
        #code.parameters.integrator =  "whfast"
        if self.subsystem_code_timestep > 0 |units.s:
            code.parameters.timestep = self.subsystem_code_timestep
        
            
        if code.stopping_conditions.timeout_detection.is_supported():
            code.stopping_conditions.timeout_detection.enable()
        
            
        if False and code.stopping_conditions.out_of_box_detection.is_supported(): 
            code.parameters.stopping_conditions_out_of_box_size = 3000 | units.AU
            code.parameters.stopping_conditions_out_of_box_use_center_of_mass = False
            code.stopping_conditions.out_of_box_detection.enable()
        code.stopping_conditions.collision_detection.enable()
        code.commit_parameters()
        code.time_step = code.parameters.timestep
        code.integrator = code.parameters.integrator
        return code
        #result = collision_code.CollisionCode(code, G = constants.G)
        #result.stopping_conditions.collision_detection.enable()
        #return result
    def make_sharing_codes(self, number_of_codes_to_share):
        
        if number_of_codes_to_share > 0:
            result = list([self.make_shared_rebound_code() for _ in range(number_of_codes_to_share)])
            for x in result:
                x._initial = True
        else:
            result = []
        return result
    def make_shared_code(self, particles):
        real_code = self.sharing_codes[self.cycle_index]
        self.cycle_index += 1
        if self.cycle_index >= self.number_of_sharing_codes:
            self.cycle_index = 0
            
        if real_code._initial:
            subset = 0
            real_code._initial = False
        else:
            subset = real_code.new_subset()
        particles.subset = subset
        code = SubsetCode(real_code, subset)
        
        real_code.set_time_step(real_code.time_step, subset)                                                                                                        
        real_code.set_integrator(real_code.integrator, subset)
        real_code.particles.add_particles(particles)
        code.update()
        return code


    def make_gravity_code(self, particles):
        return CalculateFieldForParticles(particles, G = constants.G)

    def make_cluster_code(self):
        class NoEvolve(self.get_code_factory(),NoEvolveMixin):
            pass
        if self.must_evolve:
            code = self.get_code_factory()(self.new_converter())
        else:
            code = NoEvolve(self.new_converter())
        return code

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

def time_to_string(quantity, quantities = (1 | units.s, 2 | units.minute, 1 | units.hour, 0.5 | units.day, 0.1 | units.yr, 0.1 | units.Myr)):
    def smallest_quantity():
        previous = quantities[0]
        for x in quantities[1:]:
            if quantity < x:
                return previous
            previous = x
        return quantities[-1]
    base_quantity = smallest_quantity()
    return str(quantity.as_quantity_in(base_quantity.unit))


def make_a_planet(central_particle, phi=None, theta=None, rng = None, kepler = None):
    if rng is None:
        rng = random
        
    planet_particles = datamodel.Particles(1)
    planet_particles.semimajor_axis = [100] | units.AU
    planet_particles.eccentricity = 0.0
    planet_particles.mass = 1 | units.MJupiter
    planet_particles.radius = 68700 | units.km
    converter = nbody_system.nbody_to_si(central_particle.mass, 1|units.AU)
    if kepler is None:
        kepler = Kepler(converter)
    
    if phi is None:
        phi = numpy.radians(rng.uniform(0, 90, 1)[0])#rotate under x
    if theta is None:
        theta = numpy.radians(rng.uniform(0, 180, 1)[0]) #rotate under y
    psi = 0
    for x in planet_particles:
        make_planets_salpeter.posvel_from_orbital_elements(central_particle.mass, x, kepler, rng)
        make_planets_salpeter.rotate(x, phi, theta, psi) # theta and phi in radians            

    planet_particles.position += central_particle.position
    planet_particles.velocity += central_particle.velocity

    return planet_particles

def all_particles(particles):
    if not hasattr(particles,"subsystem"):
        return particles

    particles_with_subsystem = compound_particles(particles)
    particles_without_subsystem = highlevel_particles(particles)
    
    result = datamodel.Particles()
    result.add_particles(particles_without_subsystem)
    
    for x in particles_with_subsystem:
        sub = result.add_particles(x.subsystem)
        sub.position += x.position
        sub.velocity += x.velocity

    for attr, value in particles.collection_attributes.items():
        setattr(result.collection_attributes, attr, value)

    return result



class RunModelInSingleCode(RunModel):
    
    
        









   
        
   

    def create_code(self):
        result = self.get_code_factory()(self.new_converter())
        result.particles.add_particles(self.model)
        result.commit_particles()
        return result
        














    def copy_to_model(self):
        for channel in self.channels:
            channel.copy()
    
    
    def put_parameters_in_model(self):
        self.model.collection_attributes.delta_t = self.dt
        self.model.collection_attributes.code_name = self.code_name
        self.model.collection_attributes.with_escapers = self.with_escapers
        self.model.collection_attributes.model_time = self.time
        self.model.collection_attributes.must_kick_subsystems = self.must_kick_subsystems
        self.model.collection_attributes.subsystem_code_timestep = self.subsystem_code_timestep
        self.model.collection_attributes.must_evolve = self.must_evolve
        
    





    def get_filename(self):
        if self.output_filename:
            return self.output_filename
        else:
            return "cluster-planets-{0}.h5".format(platform.node())
    


    def save(self, append_to_file = True):
        self.put_parameters_in_model()
        if append_to_file:
            names = ('dynamical_model',)
            sets = (self.model,)
        else:
            names = ('dynamical_model', 'stars')
            sets = (self.model, self.stars)
        write_set_to_file(sets, self.get_filename(), "amuse", append_to_file=append_to_file, version="2.0", names = names)
        


   

   




    def get_se_to_model_channels(self, code, model):
        result = []
        result.append(code.particles.new_channel_to(model))
        return result




    def get_from_model_to_code_channels(self, code, model):
        result = []
        result.append(model.new_channel_to(code.particles))
        return result


    def get_from_code_to_model_channels(self, code, model):
        result = []
        result.append(code.particles.new_channel_to(model))
        return result

    def report_timings(self, code):
        pass

    def __init__(self, model, stars, dt = None, code_name = None,with_escapers = None, output_filename = "", save_step = 10, large_step = 1000, must_kick_subsystems = True , timestep = 0 | units.s, must_evolve = True, has_dynamic_timestep = True, number_of_sharing_codes = 10):
        RunModel.__init__(
            self, 
            all_particles(model), 
            stars,
            dt = dt, 
            code_name = code_name,
            with_escapers = with_escapers, 
            output_filename = output_filename,
            save_step = save_step, 
            large_step = large_step, 
            must_kick_subsystems = must_kick_subsystems ,  #unused
            timestep = timestep,  #unused
            must_evolve = must_evolve,  #unused
            has_dynamic_timestep = has_dynamic_timestep,   #unused
            number_of_sharing_codes = number_of_sharing_codes #unused
        )
        









        














    def print_statistics_on_subsystems(self):
        pass
    def stop_code(self):
        self.code.stop()
def compound_particles(model):
    return model[~numpy.equal(model.subsystem, None)]
 
def highlevel_particles(model):
    return model[numpy.equal(model.subsystem, None)] 
    
class NullCode(object):
    def __init__(self):
        self.model_time = 0 | units.yr
        self.kinetic_energy = 0 | units.m**2 * units.kg * units.s**-2
        self.potential_energy = 0 | units.m**2 * units.kg * units.s**-2
        
    def evolve_model(self, end_time):
        self.model_time = end_time
    
    def stop(self):
        pass
class NoEvolveMixin(object):
    def evolve_model(self, endtime):
        self._model_time = endtime

    @property
    def model_time(self):
        return self._model_time
if __name__ == "__main__":
    #import fast_halt2
    #fast_halt2.make_managable(None)
    if 1:
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

            model.radius = model.original_radius
            stars = stars.copy()
                                
            if o.newcode:
                model.collection_attributes.code_name = o.newcode
        
    else:
        uc =  CreateModel(o.ncluster,o.number_of_solar_systems, seed = o.seed)
        uc.start()
        model, stars = uc.result
        model.collection_attributes.code_name = o.code
        model.collection_attributes.with_escapers = o.with_escapers
    
    uc = RunModel(
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








