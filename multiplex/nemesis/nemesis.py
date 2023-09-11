import time as pytime
import threading
import numpy
from amuse.rfi.channel import AsyncRequestsPool
#from amuse.rfi.async_request import AsyncRequestsPool
from amuse.couple import bridge
from amuse.units import constants
from amuse import datamodel
from amuse.units import nbody_system
from amuse.units import quantities
from amuse.units import units

class ValueHolder(object):
    
    def __init__(self, value = None):
        self.value = value
        
    def __repr__(self):
        return "V({0!r})".format(self.value)

    def __str__(self):
        return "V({0!s})".format(self.value)

    


class Nemesis(object):
    def __init__(self, cluster_code, subsystem_code_factory, gravity_code_factory, timestep, use_async = False, use_threading=False, G = constants.G, eta = 0.2, must_kick_subsystems = True, calculate_radius = None, threshold = None, dist_func = None, subsystem_factory = None):
        self.cluster_code=cluster_code
        self.subsystem_code_factory=subsystem_code_factory
        self.gravity_code_factory = gravity_code_factory
        self.fixit = quantities.zero
        self.particles=datamodel.Particles()
        self.timestep=timestep
        self.subsystem_codes={}
        self.use_threading=use_threading
        print("use_threading:", use_threading)
        self.use_async = use_async
        self.to_code_channels = []
        self.to_model_channels = []
        self.stopping_condition_was_set = False
        self.must_kick_subsystems = must_kick_subsystems
        self.must_handle_one_encounter_per_stopping_condition = False
        self.distfunc = dist_func
        if dist_func is None:
            self.distfunc = self.calculate_distance
        if threshold is None:
            self.threshold = self.timestep
        else:
            self.threshold = threshold
        if subsystem_factory is None:
            self.subsystem_factory = Subsystem
        else:
            self.subsystem_factory = subsystem_factory
            
        if calculate_radius is not None:
            self.calculate_radius = calculate_radius
        self.system = None
        self.subsystems = []
        
        self.particle_to_system={}
        self.G = G

        self.has_collision_detection =  hasattr(self.cluster_code, 'stopping_conditions') and self.cluster_code.stopping_conditions.collision_detection.is_supported()
        if self.has_collision_detection:
            self.cluster_code.stopping_conditions.collision_detection.enable()
       
        self.timings = {}
        self.eta = eta
        
        self.total_delta_kinetic_energy = quantities.zero
        self.total_delta_potential_energy = quantities.zero



    def commit_particles(self):
        if not hasattr(self.particles, 'subsystem'):
            self.particles.subsystem = None
            
        for x in self.noncompound_particles():
            x.original_radius = x.radius
            x.radius = self.get_radius(x)
            
        self.cluster_code.particles.add_particles(self.particles)
        self.system = System(self.particles, self.cluster_code, self.gravity_code_factory)
        for parent in self.compound_particles():

            if hasattr(parent.subsystem, 'subsystem'):
                delattr(parent.subsystem, 'subsystem')
            subsystem = self.subsystem_factory(
                self.particles, 
                parent, 
                parent.subsystem,
                self.subsystem_code_factory, 
                self.gravity_code_factory, 
                self.model_time, 
                self.G,
                is_kicker = self.must_kick_subsystems,
                is_kicked = self.must_kick_subsystems,
                offset = self.model_time
            )
            self.subsystems.append(subsystem)
            self.particle_to_system[parent.key] = subsystem
            
        for x in self.compound_particles():
            x.radius = self.get_radius(x)
        
        
        with self.timer("energy"):
            self.E0 = self.kinetic_energy + self.potential_energy                                                                                                                                                                                                                                                                                                                                                                                        
            self.E0S = self.system.kinetic_energy() + self.system.potential_energy()

    def recommit_particles(self):
        self.particles.synchronize_to(self.cluster_code.particles)
        
        compound_particles = self.compound_particles()
        for parent in compound_particles:
            if len(parent.subsystem) == 0:
                parent.subsystem = None
        compound_particles = self.compound_particles()                                                                                  
        
        for x in self.subsystems:
            key = x.parent.key
            if not self.particles.has_key_in_store(key):
                x.stop()
                if key in self.particle_to_system:
                    del self.particle_to_system[key]                 
        
        for parent in compound_particles:
            model_time = self.model_time
            if not parent.key in self.particle_to_system:
                if hasattr(parent.subsystem, 'subsystem'):
                    delattr(parent.subsystem, 'subsystem')
                subsystem = self.subsystem_factory(
                    self.particles, 
                    parent, 
                    parent.subsystem,
                    self.subsystem_code_factory, 
                    self.gravity_code_factory, 
                    self.model_time, 
                    self.G, 
                    is_kicker = self.must_kick_subsystems,
                    is_kicked = self.must_kick_subsystems,
                    offset = model_time
                )
                
                self.subsystems.append(subsystem)
                self.particle_to_system[parent.key] = subsystem
                
        list(map(lambda x : x.update(), self.subsystems))
        self.subsystems = [x for x in self.subsystems if x.is_valid()]

    def commit_parameters(self):
        pass
    

    def recenter_systems(self, subsystems = None):
        if subsystems is None:
            subsystems = self.subsystems
        list(map(lambda x : x.recenter(), subsystems))  
            


    def compound_particles(self):
        return self.particles[~numpy.equal(self.particles.subsystem, None)]


    def evolve_model(self, tend, timestep=None):
        
        with self.timer("evolve_model"):
            if timestep is None:
                timestep = self.timestep
            if timestep is None:
                timestep = tend-self.model_time                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    
            self.stopping_condition_was_set = False                                                                                                                                                                                             
            
            with self.timer("energy"):
                E0 = self.kinetic_energy + self.potential_energy                                                                                                                                                                                                                                                                                                                                                                                
                E0S = self.system.kinetic_energy() + self.system.potential_energy()
            print((E0-self.E0)/self.E0)
            self.E0 = E0
            self.E0S = E0S
            with self.timer("transfer #1"):
                self.transfer_state_to_codes(self.system, self.subsystems)
            
            reuse_kick = False
            
            while self.model_time < (tend-timestep/2.): 
                #self.check_mergers()
                
                subsystems = list(self.subsystems)                                                                                                                                                                                                                                                     
                system = self.system
                for x in self.subsystems:
                    x.set_e0(system.system)
          
                current_timestep = timestep
                with self.timer("kick #1"):
                    self.kick_codes(system, subsystems, current_timestep/2.0, transfer_to_model = False, reuse_kick = reuse_kick)

                tnext = self.model_time+current_timestep
                with self.timer("drift"):
                    subsystems = self.subsystems
                    system = self.system
                    subsystems = self.drift_codes(system, subsystems, tnext,self.model_time+current_timestep/2)
                    if self.stopping_condition_was_set:
                        raise Exception("stopping condition encountered")
                
                for x in self.subsystems:
                    x.report_de(system.system)

                subsystems = list(self.subsystems)                     
                with self.timer("kick #2"):
                    self.kick_codes(system, subsystems, current_timestep/2.0,  transfer_to_model = True, reuse_kick = False)
                
                with self.timer("energy"):
                    K1 = self.kinetic_energy                                                                                                                                          
                    P1 = self.potential_energy                                                                                                                                                   
                    E1 = K1 + P1                                                                                                                                                               
                                                                                                                                                                                                                                
                    E1S = self.system.kinetic_energy() + self.system.potential_energy()                                                                                                                                                        
                    
                
                    #print "DE before split step:", (E1 - E0) / E0, "just system:", (E1S - E0S)/ (E0S), "ABS:", (E1 - E0)
                
                with self.timer("split"):
                    reuse_kick = self.split_subsystems(subsystems)
                #reuse_kick = False
                #self.report_timings()
                #print self.model_time.as_quantity_in(units.yr)
                #for x in self.subsystems:
                #    x.report(tnext)
                
                #with self.timer("energy"):
                #    E2 = self.kinetic_energy + self.potential_energy                         
                #    print "DE step:", (E2 - E0) / E0, (E2 - E1) / E1
                for x in self.subsystems:
                    x.report_de(system.system)

                reuse_kick = False
                #with self.timer("radius"):
                #    for x in self.subsystems:
                #        x.reset_radius(self)

            if len(self.subsystems) == 0:
                with self.timer("transfer #2"):
                    self.transfer_state_to_model(self.system, self.subsystems)






    def drift_codes(self, system, subsystems, tend, corr_time):
        code = system.code
        if self.has_collision_detection:
            stopping_condition = code.stopping_conditions.collision_detection

        with self.timer("drift-main"):        
            #E0 = self.kinetic_energy + self.potential_energy                          
            while code.model_time < tend*(1-1.e-12):
                print("drift main to:", tend)
                code.evolve_model(tend)
                print("drifted main to:", code.model_time,  code.model_time < tend*(1-1.e-12))
                
                if self.has_collision_detection and stopping_condition.is_set():
                    with self.timer("collision-main"):      
                        self.transfer_state_to_model(system, [])                                                                                             
                        coll_time=code.model_time
                        encounters = self.get_encounters(
                            *self.filter_collisions(
                                stopping_condition.particles(0).get_intersecting_subset_in(self.particles),
                                stopping_condition.particles(1).get_intersecting_subset_in(self.particles)
                            )
                        )
                        encounters = self.get_encounters(
                            
                                stopping_condition.particles(0).get_intersecting_subset_in(self.particles),
                                stopping_condition.particles(1).get_intersecting_subset_in(self.particles)
                        )
                        
                        if len(encounters) > 0:
                            print("handling encounter(s)", len(encounters))
                            new_subsystems = []
                            for x in encounters:
                                new_subsystems.extend(self.handle_collision_in_cluster_prev(x, coll_time, corr_time))
                    
                            self.subsystems = [x for x in self.subsystems if x.is_valid()]
                            subsystems = [x for x in subsystems if x.is_valid()]
            
                            #self.kick_codes(self.system, new_subsystems, corr_time - coll_time, recenter = True, transfer_to_model = False)
                            #subsystems.extend(new_subsystems)
                        
                        #E1 = self.kinetic_energy + self.potential_energy                                                  
                        #print "EDE:", (E1-E0)/E0  , (E1-self.E0)/(self.E0), (E0-self.E0)/(self.E0)
                              
                                                            
                            
        #E1 = self.kinetic_energy + self.potential_energy                                                  
        #print "PDE1:", (E1-E0)/E0  , (E1-self.E0)/(self.E0), (E0-self.E0)/(self.E0)   
        
        with self.timer("transfer drift #1"):                                                               
            self.transfer_state_to_model(system, [])                                                                                       
        #E1 = self.kinetic_energy + self.potential_energy                                                  
        #print "PDE2:", (E1-E0)/E0  , (E1-self.E0)/(self.E0), (E0-self.E0)/(self.E0)                                                                                                                        
        print("drifting main system done")

        with self.timer("drift subsystem"):           
            
            if self.use_async:
                self.drift_subsystems_async(self.subsystems, tend)
            else:
                self.drift_subsystems_sync_or_threaded(self.subsystems, tend)
                    
            
        with self.timer("transfer drift #2"): 
            self.transfer_state_to_model(self.system, self.subsystems)

            #E1 = self.kinetic_energy + self.potential_energy                                                  
            #print "PDE21:", (E1-E0)/E0  , (E1-self.E0)/(self.E0), (E0-self.E0)/(self.E0)                 
              
            for x in  self.subsystems:
                x.synchronize_to_model()
        
            #E1 = self.kinetic_energy + self.potential_energy                                                  
            #print "PDE3:", (E1-E0)/E0  , (E1-self.E0)/(self.E0), (E0-self.E0)/(self.E0)                              
        return subsystems

    def kick_codes(self, system, subsystems, dt, recenter = True, transfer_to_model = True, reuse_kick = False):
        if len(subsystems) > 0:
            if transfer_to_model:
                self.transfer_state_to_model(system, subsystems)
            self.correction_kicks(system, subsystems, dt, reuse_kick = reuse_kick)
            if recenter:
                self.recenter_systems(subsystems)
            self.transfer_state_to_codes(system, subsystems)





    def correction_kicks(self, system, subsystems, dt, reuse_kick = False):
        kicker_subsystems = [x for x in subsystems if x.is_kicker]
        system.kick(kicker_subsystems, dt, self.timestep, reuse_kick = reuse_kick)
        for x in subsystems:
            x.kick(system.system, dt, self.timestep, reuse_kick = reuse_kick)

    def get_potential_at_point(self,radius,x,y,z):
        return self.cluster_code.get_potential_at_point(radius,x,y,z)


    def get_gravity_at_point(self,radius,x,y,z):
        return self.cluster_code.get_gravity_at_point(radius,x,y,z)


    @property
    def potential_energy(self):
        result = self.system.potential_energy([x for x in self.subsystems if x.is_kicker])
        for x in self.subsystems:
            result += x.potential_energy(self.system.system)
        
        return result - self.total_delta_potential_energy




    @property
    def kinetic_energy(self):  
        result = self.system.kinetic_energy()
        for x in self.subsystems:
            result += x.kinetic_energy()
        return result - self.total_delta_kinetic_energy



    @property
    def model_time(self):  
        return self.cluster_code.model_time

    def transfer_state_to_codes(self, system, subsystems):
        system.transfer_state_to_code()
        list(map(lambda x : x.transfer_state_to_code(), subsystems))

    def transfer_state_to_model(self, system, subsystems):
       
        if False and self.use_threading:
            def doit(x):
                x.transfer_state_to_model()
            threads = []
            threads.append(threading.Thread(target = doit, args = (system,)))
            for x in subsystems:
                threads.append(threading.Thread(target = doit, args = (x,)))
            list(map(lambda x : x.start(), threads))
            list(map(lambda x : x.join(), threads))
        else:
            system.transfer_state_to_model()
            list(map(lambda x : x.transfer_state_to_model(), subsystems))

    def stop(self):
        threads=[]
        threads.append(threading.Thread(target=self.system.code.stop) )
        for x in self.subsystems:
            threads.append(threading.Thread(target=x.code.stop) )
        
        if True or not self.use_threading:
            for x in threads:
                x.run()
                
        else:
            for x in threads:
                x.start()
            
            for x in threads:
                x.join()




    def kick_with_field_code(self, particles, field_code, dt):
        names = ('x','y','z', 'vx', 'vy', 'vz')
        x, y, z, vx, vy, vz = particles.get_values_in_store(None, names)
        ax,ay,az=field_code.get_gravity_at_point(
            quantities.zero,
            x,
            y,
            z
        )
        names = ('vx','vy','vz')
        particles.set_values_in_store(
                None,
                names,
                (vx + dt * ax,
                vy + dt * ay,
                vz + dt * az)
        )


    def update_systems(self):
        for parent in self.compound_particles():
            parent.mass = parent.subsystem.mass.sum()
        self.recenter_systems(self.subsystems)
            

    def handle_collision(self, key, code):
        sc = code.stopping_conditions.collision_detection
        p0 = sc.particles(0)[0]
        p1 = sc.particles(1)[0]
        star = code.particles[0]
        if p0 == star:
            print("collision with the star 0, removing the planet, todo add mass to star?")
            code.particles.remove_particle(p1)
        elif p1 == star:
            print("collision with the star 1, removing the planet, todo add mass to star?")
            code.particles.remove_particle(p0)
        else:
            print("collision between planets, remove both at the moment, should do something else!!")
            code.particles.remove_particle(p0)
            code.particles.remove_particle(p1)

    def get_encounters(self, particles0, particles1):
        if self.must_handle_one_encounter_per_stopping_condition:
            particles0 = particles0[:1]
            particles1 = particles1[:1]
            
        encounters = []
        
        from_key_to_encounter = {}
        for particle0, particle1 in zip(particles0, particles1):
            key0 = particle0.key
            key1 = particle1.key
            if key0 in from_key_to_encounter:
                if key1 in from_key_to_encounter:
                    encounter0 = from_key_to_encounter[key0]
                    encounter1 = from_key_to_encounter[key1]
                    if not encounter0 is encounter1:
                        encounter0.add_particles(encounter1)
                        encounter1.remove_particles(encounter1.copy())
                        for x in encounter0:
                            from_key_to_encounter[x.key] = encounter0
                else:
                    encounter = from_key_to_encounter[key0]
                    encounter.add_particle(particle1)
                    from_key_to_encounter[key1] = encounter
            elif key1 in from_key_to_encounter:
                encounter = from_key_to_encounter[key1]
                encounter.add_particle(particle0)
                from_key_to_encounter[key0] = encounter
            else:
                encounter = datamodel.Particles()
                encounter.add_particle(particle0)
                encounter.add_particle(particle1)
                
                encounters.append(encounter)
                from_key_to_encounter[key0] = encounter
                from_key_to_encounter[key1] = encounter
        
        return [x.get_intersecting_subset_in(self.particles) for x in encounters if len(x) > 0]
        


    def handle_collision_in_cluster_prev(self, particles, coll_time, corr_time):
       
        print("handle_collision_in_cluster", particles.key)
        system, subsystems, no_subsystems = self.get_system_for_particles(particles)
        print("subsystems:", len(subsystems), ",no subsystem", len(no_subsystems))
        if len(subsystems) > 0:
            for x in subsystems:
                x.drift(coll_time, only_subset = True)
                x.synchronize_to_model()
                x.transfer_state_to_model()

            #self.kick_codes(system, subsystems, coll_time-corr_time, recenter = False)
        
        new_subsystem = datamodel.Particles()
        
        for x in subsystems:
            print("subsystem length:", len(x.subsystem), x.subsystem.mass.as_quantity_in(units.MSun))
            subsystem_particles = new_subsystem.add_particles(x.subsystem)
            subsystem_particles.position+=x.parent.position
            subsystem_particles.velocity+=x.parent.velocity
            self.total_delta_potential_energy += x.total_delta_potential_energy
            self.total_delta_kinetic_energy += x.total_delta_kinetic_energy
            x.stop()
            
            del self.particle_to_system[x.parent.key]
            
        for x in no_subsystems:
            y = new_subsystem.add_particle(x)
            y.radius = y.original_radius
            
        self.cluster_code.particles.remove_particles(particles)
        self.particles.remove_particles(particles)
        if hasattr(new_subsystem, 'subsystem'):
            delattr(new_subsystem, 'subsystem')
        components = new_subsystem.connected_components(threshold=self.threshold,distfunc=self.distfunc)
        #components = [new_subsystem]
        print("splitting in components:", tuple([len(x) for x in components]))
        result = []
        for x in components:
            if len(x) == 1:
                newparent=self.particles.add_particle(x[0])
                newparent.original_radius=newparent.radius
                newparent.radius = self.get_radius(newparent)
                newparent.subsystem = None
                self.cluster_code.particles.add_particle(newparent)
            else:
                ##for y in x:
                ##    print y.key, y.velocity
                result.append(self.add_new_subsystem(x, coll_time))
        return result

    def handle_collision_in_cluster(self, particles, coll_time, corr_time):
       
        print("handle_collision_in_cluster", particles.key)
        system, subsystems, no_subsystems = self.get_system_for_particles(particles)
        print("subsystems:", [(len(x.subsystem), x.parent.key) for x in subsystems])
        if len(subsystems) > 0:
            for x in subsystems:
                x.drift(coll_time, only_subset = True)
                x.synchronize_to_model()
                x.transfer_state_to_model()

            #self.kick_codes(system, subsystems, coll_time-corr_time, recenter = False)
        
        new_subsystem = datamodel.Particles()
        
        for x in subsystems:
            print(len(x.subsystem))
            subsystem_particles = new_subsystem.add_particles(x.subsystem)
            subsystem_particles.position+=x.parent.position
            subsystem_particles.velocity+=x.parent.velocity
            
            
        for x in no_subsystems:
            y = new_subsystem.add_particle(x)
            y.radius = y.original_radius
            
        single_stars = {}
        for x in no_subsystems:
            single_stars[x.key] = x
        
        multiples = {}
        for x in subsystems:
            key = tuple(sorted(x.subsystem.key))
            multiples[key] = x
            
        #for x in subsystems:
        ##    x.stop()
        #    del self.particle_to_system[x.parent.key]
            
        #self.cluster_code.particles.remove_particles(particles)
        #self.particles.remove_particles(particles)

        #components = new_subsystem.connected_components(threshold=self.threshold,distfunc=self.distfunc)
        components = [new_subsystem]
        print("spliting in components:", tuple([len(x) for x in components]))
        result = []
        reused = set([])
       
        for x in components:
            if len(x) == 1:
                pass
            else:
                for y in x:
                    if y.key in single_stars:
                        # if it was a single star and is now in a subsystem
                        # we need to remove that particle
                        print("captured particle in subsytem:", y.key, y.mass.as_quantity_in(units.MSun))
                        self.cluster_code.particles.remove_particle(y)
                        self.particles.remove_particle(y)
                key = tuple(sorted(x.key))
                if key in multiples:
                    print("unchanged subsystem with length:", len(x))
                    reused.add(key)
        for x in list(multiples.keys()):
            if not x in reused:
                subset = multiples[x]
                print("multiple removed from system", subset.parent.key)
                subset.stop()
                del self.particle_to_system[subset.parent.key]
                self.cluster_code.particles.remove_particle(subset.parent)
                self.particles.remove_particle(subset.parent)
                
        for x in components:
            if len(x) == 1:
                if not x[0].key in single_stars:
                    # if it was not already a single star then
                    # add it!
                    print("particle escaped from subsystem:", x[0].key)
                    newparent=self.particles.add_particle(x[0])
                    newparent.original_radius=newparent.radius
                    newparent.radius = self.get_radius(newparent)
                    self.cluster_code.particles.add_particle(newparent)
                    self.particles.add_particle(particles)
                else:
                    print("unchanged particle with key:", x[0].key)
            else:
                key = tuple(sorted(x.key))
                if not key in multiples:
                    print("added new subsystem with length:", len(x))
                    result.append(self.add_new_subsystem(x, coll_time))
        return result







    def get_active_system(self, time):
        active_subsystems = self.subsystems      #filter(lambda x : x.is_active(time), self.subsystems)
        return self.system, active_subsystems
        

    def get_system_for_particles(self, particles):
        system = System(particles, self.cluster_code, self.gravity_code_factory)
        substsystems = [self.particle_to_system[x.key] if x.key in self.particle_to_system else None for x in particles]
        substsystems = [x for x in substsystems if not x is None]
        no_subsystem = [None if x.key in self.particle_to_system else x for x in particles]
        no_subsystem = [x for x in no_subsystem if not x is None]
        return system, substsystems, no_subsystem
        

    def get_radius(self, particle):
        if particle.subsystem is None:
            set = particle.as_set()         
        else:
            set = particle.subsystem
        return self.calculate_radius(set)
            
       

    def noncompound_particles(self):
        return self.particles[numpy.equal(self.particles.subsystem, None)]


    def split_subsystems(self, subsystems):
        subsystems=[x for x in subsystems if x.is_kicker or x.is_kicked]
        if len(subsystems) == 0:
            return True
        
        if self.distfunc is None:
            return True
        
        to_remove=datamodel.Particles()
        sys_to_add=[]
       
        #E0 = self.kinetic_energy + self.potential_energy
        for x in subsystems:
            parent = x.parent
            #print parent.key, id(x)
            subsys = x.subsystem
            radius = parent.radius
            if len(subsys) == 1:
                components = [subsys]                                                                     
            else:
                components = subsys.connected_components(threshold=self.threshold,distfunc=self.distfunc)
            if len(components)>1 or len(subsys) == 1 :
                print("splitting:", len(components), [len(c) for c in components], [c.key for c in components])
                parent_position=parent.position
                parent_velocity=parent.velocity
                to_remove.add_particle(parent)
                for c in components:
                    sys=c.copy()
                    sys.position+=parent_position
                    sys.velocity+=parent_velocity
                    sys_to_add.append(sys)
                #print self.particle_to_system.keys()
                del self.particle_to_system[parent.key]
                self.total_delta_potential_energy += x.total_delta_potential_energy
                self.total_delta_kinetic_energy += x.total_delta_kinetic_energy
                x.stop()
        if len(to_remove) == 0:
            return True
        
        self.subsystems = [x for x in self.subsystems if x.is_valid()]
        self.particles.remove_particles(to_remove)
        self.cluster_code.particles.remove_particles(to_remove)
        model_time = self.model_time
        for sys in sys_to_add:
            if len(sys)>1:
                if hasattr(sys, 'subsystem'):
                    delattr(sys, 'subsystem')
                ns = self.add_new_subsystem(sys, model_time)
                newparent = ns.parent
            else:
                newparent=self.particles.add_particle(sys[0])
                newparent.original_radius=newparent.radius
                newparent.radius = self.get_radius(newparent)
                newparent.subsystem = None
                self.cluster_code.particles.add_particle(newparent)
        return False

    def add_new_subsystem(self, new_subsystem, time = None):
       

        parent = datamodel.Particle()
        parent.mass = new_subsystem.mass.sum()
        parent.position=new_subsystem.center_of_mass()
        parent.velocity=new_subsystem.center_of_mass_velocity()
        parent.dt = self.timestep
        new_subsystem.move_to_center()
        #print "new_subsystem keys:", self.model_time.as_quantity_in(units.yr), new_subsystem.key
        parent.subsystem = new_subsystem
        parent.radius = self.get_radius(parent)
        print("new subsystem with key:", parent.key, "and radius:", parent.radius.as_quantity_in(units.parsec), " :radius parsec: (free-fall, system)", [x.value_in(units.parsec) for x in self.calculate_radius_debug(parent.subsystem)])
        print("new subsystem mass:", parent.mass.as_quantity_in(units.MSun), ", particles:", len(new_subsystem), ", max mass:", new_subsystem.mass.max().as_quantity_in(units.MSun))
        parent = self.particles.add_particle(parent)
        subsystem = self.subsystem_factory(
            self.particles, 
            parent, 
            parent.subsystem,
            self.subsystem_code_factory, 
            self.gravity_code_factory, 
            self.model_time, 
            self.G, 
            is_kicked = self.must_kick_subsystems,
            is_kicker = self.must_kick_subsystems,
            offset = time
        )
        
        self.subsystems.append(subsystem)
        self.particle_to_system[parent.key] = subsystem
        self.cluster_code.particles.add_particle(parent)
        
        return subsystem

    def drift_code(self, subsystem, time, has_exception = None, result_holder = None):
        try:
            result = subsystem.drift(time, False)                         
            if not result_holder is None:
                result_holder.value = result                                  
        except:
            print("exception in drift_code...")
            if not has_exception is None:
                has_exception.value = True
            raise



    def filter_collisions(self, particles0, particles1):
        result0 = []
        result1 = []
        for particle0, particle1 in zip(particles0, particles1):
            radii = particle0.radius + particle1.radius
            
            r = (particle1.position-particle0.position).length()
            v = (particle1.velocity-particle0.velocity).length()
            cos_angle = numpy.inner((particle1.velocity-particle0.velocity)/v,
                                    (particle1.position-particle0.position)/r)
            angle = numpy.arccos(cos_angle)
            if r < radii  and angle > (numpy.pi * 0.44):
                result0.append(particle0)
                result1.append(particle1)
                           
        return result0, result1
    def timer(self, name):
        
        class Timer(object):
            def __init__(self, owner, name):
                self.owner = owner
                self.name = name
            
            def __enter__(self):
                self.t0 = pytime.time()
                return self
            
            def __exit__(self, type, value, traceback):
                self.t1 = pytime.time()
                self.owner.add_timing(name, self.t1 - self.t0)
        
        return Timer(self, name)
                

    def add_timing(self, name, timing):
        if not name in self.timings:
            self.timings[name] = 0
        self.timings[name] += timing
                
    def report_timings(self):
        names = sorted(self.timings.keys())
        for x in names:
            print("TIMING-", x, ':' , time_to_string(self.timings[x] | units.s))

    def clear_timings(self):
        names = list(self.timings.keys())
        self.timings = {}
        for x in names:
            self.timings[x] = 0.0

    def check_mergers(self):
        collisions = self.system.collisions()
        newsubsystems = []
        for particles in collisions:
            print(particles.key)
            system, subsystems, no_subsystems = self.get_system_for_particles(particles)
        

            new_subsystem = datamodel.Particles()

            for x in subsystems:
                subsystem_particles = new_subsystem.add_particles(x.subsystem)
                subsystem_particles.position+=x.parent.position
                subsystem_particles.velocity+=x.parent.velocity
                x.stop()
                del self.particle_to_system[x.parent.key]
            
            for x in no_subsystems:
                y = new_subsystem.add_particle(x)
                y.radius = y.original_radius
            
            self.cluster_code.particles.remove_particles(particles)
            self.particles.remove_particles(particles)
        
            newsubsystems.append(self.add_new_subsystem(new_subsystem, self.model_time))

        self.subsystems = [x for x in self.subsystems if x.is_valid()]

    def all_particles(self):
        result = Particles()
        for x in self.particles:
            if x.subsystem is None:
                result.add_particle(x)
            else:
                sub = result.add_particles(x.subsystem)
                sub.position += x.position
                sub.velocity += x.velocity
        return result


    def calculate_radius(self, sys):
        #if len(sys) == 0:
        #    return 150 | units.AU

        r2max=(sys.position - sys.center_of_mass()).lengths_squared().max()
        radius =((self.G*sys.total_mass()*self.timestep**2/self.eta**2)**(1./3.))
        radius = radius*((len(sys)+1)/2.)**0.75
        print("radius=", radius.in_(units.parsec))
        return max(radius, 1.05 * r2max**0.5)


    def calculate_distance(self, ipart, jpart):
        dx=ipart.x-jpart.x                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
        dy=ipart.y-jpart.y
        dz=ipart.z-jpart.z
        dr2=dx**2+dy**2+dz**2
        dr=dr2**0.5
        dr3=dr*dr2
        mu=self.G*(ipart.mass+jpart.mass)
        tau=(self.eta/2.0)/2./2.**0.5*(dr3/mu)**0.5
        return tau



    def calculate_radius_debug(self, sys):
        #if len(sys) == 0:
        #    return 150 | units.AU

        r2max=(sys.position - sys.center_of_mass()).lengths_squared().max()
        radius =((self.G*sys.total_mass()*self.timestep**2/self.eta**2)**(1./3.))
        radius = radius*((len(sys)+1)/2.)**0.75
        return (radius, 1.5 * r2max**0.5)


    def drift_subsystems_async(self, subsystems, tend):    
        active = list(subsystems)
        while len(active) > 0:
            print("active subsystems:", len(active))
            pool = AsyncRequestsPool()
            requests = []
            for x in active:
                request = x.drift_async(tend)
                requests.append(request)
                if request:
                    pool.add_request(request)
            pool.waitall()
            if False:
                for x in active:
                    if hasattr(x.code, 'stopping_conditions') and x.code.stopping_conditions.timeout_detection.is_set():
                        print("timout detected on subsystem code for star:", x.code.particles[0].key)
                        self.stopping_condition_was_set = True
                        break

            next_active = []                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    
            for request,x in zip(requests, active):
                if request:
                    result = request.result()
                    if result == True:
                        print("Stopping condition was set, need to continue evolve in next step")
                        next_active.append(x)
              
            active = next_active





    def drift_subsystems_sync_or_threaded(self, subsystems, tend):    
        active = list(subsystems)
        threads = []
        exceptions = []
        result_holders = []
        for x in active:
            has_exception = ValueHolder()
            has_exception.value = False
            exceptions.append(has_exception)
            result_holder = ValueHolder()
            result_holder.value = None
            result_holders.append(result_holder)
            threads.append(threading.Thread(target = self.drift_code, args = (x, tend, has_exception, result_holder)))
            
        if not self.use_threading:
            for x in threads:
                x.run()
                
        else:
            for x in threads:
                x.start()
            
            for x in threads:
                x.join()
            
            for x, subsystem in zip(exceptions, active):
                if x.value:
                    print(subsystem.subsystem.key)
                    raise Exception("Exception found in drift of subsystem {0}", subsystem.parent.key)
            
            statistics = []
            for x, subsystem in zip(result_holders, active):
                if not x.value is None:
                    print("subsystem running time:", x.value)


class CalculateFieldForParticles(bridge.CalculateFieldForParticles):
    def __init__(self, particles = None, gravity_constant = None,
            softening_mode="shared", G = None):
        bridge.CalculateFieldForParticles.__init__(self, particles, gravity_constant, softening_mode, G)
        if False:
            self.particles = self.particles.copy()
        

    """
    faster algorithm to calculcate the gravity field of a particle set
    assumes:
    1. no epsilon
    2. low number of points, loop in python is removed by doing matrix calculations (needing N x M memory)
    """
    def get_gravity_at_point(self,radius,x,y,z):
        names = ("x","y","z", "mass")
        px, py, pz, mass = self.particles.get_values_in_store(None, names)
        gravity_constant = -self.gravity_constant
        n = len(x)
        newshape =(n, 1)
        x = x.reshape(newshape)             
        y = y.reshape(newshape)             
        z = z.reshape(newshape)             
        dx = x - px
        dy = y - py
        dz = z - pz
        dr_squared = ((dx ** 2) + (dy  ** 2) + (dz ** 2))
        dr_twothird = dr_squared**1.5
        m_div_dr = mass / dr_twothird
        ax = gravity_constant * (m_div_dr*dx).sum(1)
        ay = gravity_constant * (m_div_dr*dy).sum(1)
        az = gravity_constant * (m_div_dr*dz).sum(1)

        
        #ax -=  ax[0]
        #ay -=  ay[0]
        #az -=  az[0]
        #or i,(x, y) in enumerate(zip(ax, ay)):
        #    print i, x.as_quantity_in(units.AU/units.s**2), y.as_quantity_in(units.AU/units.s**2)
        #ddd
        return ax, ay, az

    def get_potential_at_point(self,radius,x,y,z):
        names = ("x","y","z", "mass")
        px, py, pz, mass = self.particles.get_values_in_store(None, names)
        n = len(x)
        newshape =(n, 1)
        x = x.reshape(newshape)                         
        y = y.reshape(newshape)                         
        z = z.reshape(newshape)                         
        dx = x - px
        dy = y - py
        dz = z - pz
        dr = ((dx ** 2) + (dy  ** 2) + (dz ** 2)).sqrt()
        
        m_div_dr_squared = mass / dr
        
        gravity_constant = -self.gravity_constant
        result = gravity_constant * m_div_dr_squared.sum(1)
        #old_result = bridge.CalculateFieldForParticles.get_potential_at_point(self,radius,x,y,z)
        #print result - old_result
        return gravity_constant * m_div_dr_squared.sum(1)
       


class CalculcateCorrectionForCompoundParticles(object):  
    def __init__(self,system, gravity_code_factory, parent = None ):
        self.system=system
        self.parent=parent
        self.gravity_code_factory=gravity_code_factory


    def get_gravity_at_point(self,radius,x,y,z):
        parent=self.parent
        px,py,pz = parent.position
        all_except_center_of_mass=self.system - parent
        instance=self.gravity_code_factory(all_except_center_of_mass)
        
        if 1:
            x1 = numpy.zeros(len(x)+1, dtype = x.dtype) | x.unit
            y1 = numpy.zeros(len(x)+1, dtype = x.dtype) | x.unit
            z1 = numpy.zeros(len(x)+1, dtype = x.dtype) | x.unit
            x1[0:-1] = x + px
            y1[0:-1] = y + py
            z1[0:-1] = z + pz
            x1[-1] = px
            y1[-1] = py
            z1[-1] = pz
            ax,ay,az=instance.get_gravity_at_point(None,x1, y1, z1)
            return (ax-ax[-1])[0:-1],(ay-ay[-1])[0:-1],(az-az[-1])[0:-1]
        else:
            ax,ay,az=instance.get_gravity_at_point(None,px+x, py+y, pz+z)
            _ax,_ay,_az=instance.get_gravity_at_point(None,px.as_vector_with_length(1),py.as_vector_with_length(1),pz.as_vector_with_length(1))
            return (ax-_ax[0]),(ay-_ay[0]),(az-_az[0])



    def get_potential_at_point(self,radius,x,y,z):
        parent=self.parent
        px,py,pz = parent.position
        all_except_center_of_mass=self.system - parent
        instance=self.gravity_code_factory(all_except_center_of_mass)
        phi=instance.get_potential_at_point(None,px+x,py+y,pz+z)
        _phi=instance.get_potential_at_point(None,px.as_vector_with_length(1),py.as_vector_with_length(1),pz.as_vector_with_length(1))
        instance.cleanup_code()
        return (phi-_phi[0])



class Subsystem(object):  
    def __init__(self,  system, parent, subsystem, subsystem_code_factory, gravity_code_factory, time, G = constants.G, is_kicked = True, is_kicker = False, offset = None):
        self.system=system
        self.subsystem = subsystem
        self.parent=parent
        self.gravity_code_factory=gravity_code_factory
        self.code =  subsystem_code_factory(subsystem, time)
        self.to_code_channel = subsystem.new_channel_to(self.code.particles)
        self.to_model_channel = self.code.particles.new_channel_to(subsystem)
        self.time = time                 
        self.G = G
        self.is_kicked = is_kicked
        self.is_kicker = is_kicker
        if offset is None:
            self.offset = self.code.model_time
        else:
            self.offset = offset
        self.is_at_start = True

        self.has_async = hasattr(self.code.evolve_model, 'async')
        self.has_kick_calculated = False
        self.set_e0(system, True)







    def get_gravity_at_point(self,system,x,y,z):
        parent=self.parent
        px,py,pz = parent.position
        all_except_center_of_mass= system - parent
        instance=self.gravity_code_factory(all_except_center_of_mass)
        
        if 0:
            x1 = numpy.zeros(len(x)+1, dtype = x.dtype) | x.unit
            y1 = numpy.zeros(len(x)+1, dtype = x.dtype) | x.unit
            z1 = numpy.zeros(len(x)+1, dtype = x.dtype) | x.unit
            x1[0:-1] = x + px
            y1[0:-1] = y + py
            z1[0:-1] = z + pz
            x1[-1] = px
            y1[-1] = py
            z1[-1] = pz
            ax,ay,az=instance.get_gravity_at_point(None,x1, y1, z1)
            return (ax-ax[-1])[0:-1],(ay-ay[-1])[0:-1],(az-az[-1])[0:-1]
        else:
            ax,ay,az=instance.get_gravity_at_point(None,px+x, py+y, pz+z)
            _ax,_ay,_az=instance.get_gravity_at_point(None,px.as_vector_with_length(1),py.as_vector_with_length(1),pz.as_vector_with_length(1))
            return (ax-_ax[0]),(ay-_ay[0]),(az-_az[0])




    def get_potential_at_point(self,system,x,y,z):
        parent=self.parent
        px,py,pz = parent.position
        all_except_center_of_mass=system - parent
        instance=self.gravity_code_factory(all_except_center_of_mass)
        phi=instance.get_potential_at_point(None,px+x,py+y,pz+z)
        _phi=instance.get_potential_at_point(None,px.as_vector_with_length(1),py.as_vector_with_length(1),pz.as_vector_with_length(1))
        instance.cleanup_code()
        return (phi-_phi[0])



    def kick(self, system, dt, min_dt, reuse_kick = False):
        self.is_at_start = not self.is_at_start
             
        if len(system) == 0 or self.is_kicked is False:
            return
        
        particles = self.subsystem
        if reuse_kick:
            names = ('ax','ay','az', 'vx', 'vy', 'vz')
            ax, ay, az, vx, vy, vz = particles.get_values_in_store(None, names)
        else:
            names = ('x','y','z', 'vx', 'vy', 'vz')
            x, y, z, vx, vy, vz = particles.get_values_in_store(None, names)
            ax,ay,az=self.get_gravity_at_point(
                system,
                x,
                y,
                z
            )

        self.time = self.code.model_time
        names = ('vx','vy','vz', 'ax', 'ay', 'az')
        if False:
            print("V(XYZ):",vx.as_quantity_in(units.kms),vy.as_quantity_in(units.kms),vz.as_quantity_in(units.kms))
            print("DV(XYZ):",(dt*ax).as_quantity_in(units.kms), (dt*ay).as_quantity_in(units.kms),(dt*az).as_quantity_in(units.kms))
            print("AX(XYZ):",ax, ay, az)
            print("DT", dt.as_quantity_in(units.yr))
        
        particles.set_values_in_store(
            None,
            names,
            (vx + dt * ax,
            vy + dt * ay,
            vz + dt * az, 
            ax, ay, az)
        )

    def recenter(self):
        subsystem = self.subsystem
        parent = self.parent
        if 0:
            mass = subsystem.mass
            position = subsystem.position
            velocity = subsystem.velocity
            total_mass = mass.sum()
            mass = mass.reshape((len(mass),1))
            center_of_mass_pos = (position * mass).sum(0) / total_mass
            center_of_mass_vel = (velocity * mass).sum(0) / total_mass
            parent.position += center_of_mass_pos
            parent.velocity += center_of_mass_vel
            subsystem.position -= center_of_mass_pos
            subsystem.velocity -= center_of_mass_vel                                                                          
        else:
            if 0:
                mass, x,y,z, vx,vy,vz = subsystem.get_values_in_store(None, ("mass", "x","y","z", "vx","vy","vz"))
                total_mass = mass.sum()
                center_of_mass_x = (mass * x).sum()/total_mass
                center_of_mass_y = (mass * y).sum()/total_mass
                center_of_mass_z = (mass * z).sum()/total_mass
                center_of_mass_vx = (mass * vx).sum()/total_mass
                center_of_mass_vy = (mass * vy).sum()/total_mass
                center_of_mass_vz = (mass * vz).sum()/total_mass
                #parent.position += center_of_mass
                #parent.velocity += center_of_mass_velocity
                subsystem.set_values_in_store(None, ("x","y","z", "vx","vy","vz"),(
                    x - center_of_mass_x,
                    y - center_of_mass_y,
                    z - center_of_mass_z,
                    vx - center_of_mass_vx,
                    vy - center_of_mass_vy,
                    vz - center_of_mass_vz,

                ))
                parent.update( ("x","y","z", "vx","vy","vz"), lambda x,y,z,vx,vy,vz: (
                    x + center_of_mass_x,
                    y + center_of_mass_y,
                    z + center_of_mass_z,
                    vx + center_of_mass_vx,
                    vy + center_of_mass_vy,
                    vz + center_of_mass_vz
                ))
            else:
                center_of_mass = subsystem.center_of_mass()
                center_of_mass_velocity =  subsystem.center_of_mass_velocity()
                
                parent.position += center_of_mass
                parent.velocity += center_of_mass_velocity
                subsystem.position -= center_of_mass
                subsystem.velocity -= center_of_mass_velocity          
            





    def transfer_state_to_code(self):
        self.to_code_channel.copy()

    def transfer_state_to_model(self):
        self.to_model_channel.copy()

    def drift(self, end_time, only_subset = True):
        return self.code.evolve_model(end_time- self.offset)





    def drift_async(self, end_time):
        if self.has_async:                     
            return self.code.evolve_model.async(end_time-self.offset)
        else:
            self.drift(end_time)
            return None


    def update(self):
        if not self.is_valid():
            if not self.code is None:
                self.code.stop()
            self.code = None
            self.to_code_channel = None
            self.from_code_channel = None
            


    def is_valid(self):
        return not self.code is None and len(self.subsystem) > 0

    def get_orbital_elements(self, mass, radius, velocity, G = constants.G):
        try:
            n = len(mass)
            reshape = lambda x : x.reshape([n,1])
            sum = lambda x : x.sum(1)
        except:
            n = 1
            reshape = lambda x : x
            sum = lambda x : x.sum()
        
        momentum = radius.cross(velocity,1,1)
        mu = (G * mass)
        evec =  (velocity.cross(momentum,1,1))/reshape(mu) - (radius/reshape(radius.lengths()))
       
        e = numpy.sqrt(sum(evec**2))
        a = sum(momentum**2) / (mu * (1-e**2))
        p = 2.0 * numpy.pi *  (a**3 / mu).sqrt()
        return e, a, p

    @property
    def model_time(self):
        return self.code.model_time + self.offset

    def kinetic_energy(self):
        return self.code.kinetic_energy

    def potential_energy(self, system):
        result = self.code.potential_energy
        if self.is_kicked:
            names = ('mass', 'x','y','z')
            particles = self.subsystem
            mass, x, y, z = particles.get_values_in_store(None, names)
            potential = self.get_potential_at_point(system, x,y,z)
            result += (potential * mass).sum()/2.0
        return result

    def stop(self):
        if not self.code is None:
            self.code.stop()
            self.code = None
            self.to_code_channel = None
            self.from_code_channel = None
            

    def calculate_dt(self, system, min_dt):
        if len(system) == 0:
            return min_dt
        
        names = ('x','y','z', 'vx', 'vy', 'vz')
        particles = self.subsystem
        x, y, z, vx, vy, vz = particles.get_values_in_store(None, names)
        ax,ay,az=self.get_gravity_at_point(
            system,
            x,
            y,
            z
        )
        return self.calculate_dt_from_cluster_acceleration(ax, ay, az, min_dt)

    def calculate_dt_from_cluster_acceleration(self, ax, ay, az, min_dt):
        max_acceleration_from_field = ((ax**2)+(ay**2)+(az**2)).max().sqrt()
        mass = self.subsystem.mass
        position = self.subsystem.position
        velocity = self.subsystem.velocity
        total_mass = mass.sum()
        mass = mass.reshape((len(mass),1))
        center_of_mass_pos = (position * mass).sum(0) / total_mass
        center_of_mass_vel = (velocity * mass).sum(0) / total_mass
        min_acceleration_from_system = self.G * total_mass/(position.lengths_squared().max())

        self.max_acceleration_from_field = max_acceleration_from_field
        self.min_acceleration_from_system = min_acceleration_from_system
        orbital_periods = self.get_orbital_elements(self.subsystem.mass+total_mass,  position - center_of_mass_pos, velocity - center_of_mass_vel, G = self.G)[2]

        min_period = orbital_periods[1:].min()     #first element is the star
        min_dt_in_system = min_period / 1000                                                                                                                                                                                                                                         #aproximate a dt for the system used by the code
        min_kick_in_system = min_acceleration_from_system * min_dt_in_system

        suggested_dt = ((min_kick_in_system / self.max_acceleration_from_field) / 100)
        current = min_dt
        while suggested_dt > current:
            current *= 2
        return suggested_dt, current / 2

    def calculate_acceleration(self, system):
        if len(system.system) == 0 or self.is_kicked is False:
            return
        names = ('x','y','z')
        particles = self.subsystem
        x, y, z = particles.get_values_in_store(None, names)
        ax,ay,az=self.get_gravity_at_point(
            system.system,
            x,
            y,
            z
        )
        names = ('ax','ay','az')
        particles.set_values_in_store(
            None,
            names,
            (ax,
            ay,
            az)
        )

    def do_kick(self):
        names = ('ax','ay','az', 'vx', 'vy', 'vz')
        particles = self.subsystem
        ax, ay, az, vx, vy, vz = particles.get_values_in_store(None, names)
        names = ('vx','vy','vz')
        dt = self.dt / 2.0
        
        particles.set_values_in_store(
                None,
                names,
                (
                    vx + dt * ax,
                    vy + dt * ay,
                    vz + dt * az
                )
        )


    def synchronize_to_model(self):
        self.code.particles.synchronize_to(self.subsystem)



    def report(self, tref):
        if hasattr(self.code, 'report'):
            print(self.parent.key, self.parent.radius.as_quantity_in(units.parsec), end=' ')
            self.code.report(tref)



    def set_e0(self, system, init = False):
        self.init = init
        self.E0 = self.kinetic_energy() + self.potential_energy(system)
        self.E0I = self.kinetic_energy()

    def report_de(self, system):
        self.E1 = self.kinetic_energy() + self.potential_energy(system)
        self.E1I = self.kinetic_energy()         #+# self.code.potential_energy
        #if abs((self.E1-self.E0) / self.E0) > 0.01:
        #    print "subsystem:", self.parent.key, ' -- ', (self.E1-self.E0) / self.E0, (self.E1I-self.E0I)/self.E0I, ("I" if self.init else "")
        


    def reset_radius(self, nemesis):
        new_radius = nemesis.get_radius(self.parent)
        self.parent.radius = new_radius




    @property
    def total_delta_kinetic_energy(self):
        if hasattr(self.code, 'total_delta_kinetic_energy'):
            return self.code.total_delta_kinetic_energy
        else:
            return quantities.zero
    @property
    def total_delta_potential_energy(self):
        if hasattr(self.code, 'total_delta_potential_energy'):
            return self.code.total_delta_potential_energy
        else:
            return quantities.zero
            
class System(object):  
    def __init__(self,  system, code, gravity_code_factory):
        self.system = system
        self.code =  code
        self.gravity_code_factory = gravity_code_factory
        
        self.subsystems_with_feedback = []
        
        #self.code.stopping_conditions.collision_detection.enable()
        self.to_code_channel = system.new_channel_to(self.code.particles)
        self.to_model_channel = self.code.particles.new_channel_to(system)



    def drift(self, end_time, only_subset = None):
        self.code.evolve_model(end_time)

    def kick(self, subsystems, dt, min_dt, reuse_kick = False):
        if len(subsystems) == 0 :
            print("**no kick")
            return
       
        if not reuse_kick:
            acc_unit = (self.system.vx.unit**2/self.system.x.unit)
            self.system.ax=0.0 | acc_unit
            self.system.ay=0.0 | acc_unit
            self.system.az=0.0 | acc_unit
            for x in subsystems:
                all_except_center_of_mass = self.system - x.parent
                code=self.gravity_code_factory(x.subsystem)#.copy())
                code.particles.position+=x.parent.position
                code.particles.velocity+=x.parent.velocity
                ax,ay,az=code.get_gravity_at_point(quantities.zero,all_except_center_of_mass.x,all_except_center_of_mass.y,all_except_center_of_mass.z)
                #print ax, code
                code.particles.position-=x.parent.position
                code.particles.velocity-=x.parent.velocity
                code=self.gravity_code_factory(x.parent.as_set())
                parent_ax,parent_ay,parent_az=code.get_gravity_at_point(quantities.zero,all_except_center_of_mass.x,all_except_center_of_mass.y,all_except_center_of_mass.z)
                all_except_center_of_mass.ax += ax - parent_ax
                all_except_center_of_mass.ay += ay - parent_ay
                all_except_center_of_mass.az += az - parent_az
        
        names = ('ax','ay','az','vx', 'vy', 'vz')
        ax, ay, az, vx, vy, vz = self.system.get_values_in_store(None, names)
        names = ('vx','vy','vz')
        
        self.system.set_values_in_store(
                None,
                names,
                (vx + ax * dt,
                vy + ay * dt,
                vz + az * dt)
        )

    def recenter(self):
        pass

    def transfer_state_to_code(self):
        self.to_code_channel.copy_attributes(["x","y","z", "vx", "vy", "vz", "mass"])



    def transfer_state_to_model(self):
        self.to_model_channel.copy_attributes(["x","y","z", "vx", "vy", "vz", "mass"])



    def drift_async(self, end_time):
        self.code.evolve_model.async(end_time)

    def update(self):
        pass

    def is_valid(self):
        return True

    def kinetic_energy(self):
        return self.code.kinetic_energy

    def potential_energy(self, subsystems = []):
        result = self.code.potential_energy

        if len(subsystems) == 0:
            return result
        
        
        pot_unit =  (self.system.vx.unit**2)
        self.system.phi=0.0 | pot_unit
        for x in subsystems:
            all_except_center_of_mass = self.system - x.parent
            if len(all_except_center_of_mass) == 0:
                continue
                
            code=self.gravity_code_factory(x.subsystem.copy())
            code.particles.position+=x.parent.position
            code.particles.velocity+=x.parent.velocity
            phi=code.get_potential_at_point(quantities.zero,all_except_center_of_mass.x,all_except_center_of_mass.y,all_except_center_of_mass.z)
            code=self.gravity_code_factory(x.parent.as_set())
            parent_phi=code.get_potential_at_point(quantities.zero,all_except_center_of_mass.x,all_except_center_of_mass.y,all_except_center_of_mass.z)
            all_except_center_of_mass.phi +=  phi - parent_phi
            
        return result + ((self.system.phi * self.system.mass).sum()/2.0)



    def collisions(self):
        result = []
        factor = 1.0
        names = ('x','y','z','radius')
        x, y, z, radius = self.system.get_values_in_store(None, names)
        n = len(x)
        newshape =(n, 1)
        px = x.reshape(newshape)                                                                                         
        py = y.reshape(newshape)                                                                                         
        pz = z.reshape(newshape)                                                                                         
        pradius = radius.reshape(newshape)                                                                                         
        dx = x - px
        dy = y - py
        dz = z - pz
        dr_squared = ((dx ** 2) + (dy  ** 2) + (dz ** 2))
        sumradius_squared = (factor * (pradius + radius)) ** 2
        selection = dr_squared < sumradius_squared
        selection = numpy.triu(selection,0) == 1
        for x, y in enumerate(selection):
            subsystem = self.system[y]
            if len(subsystem) > 1:
                result.append(subsystem)
        return result



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


class MultiGravityCodeSubsystem(Subsystem):  
    
    def drift(self, end_time, only_subset = True):
        return self.code.evolve_model(end_time- self.offset, only_subset)



    def drift_async(self, end_time):
        return self.code.evolve_model_async(end_time-self.offset)



class SubsetCode(object):
    def __init__(self, real_code, subset, G = constants.G):
        self.real_code = real_code
        self.subset = subset
        
        if self.real_code._initial:
            self.real_code._cached_model_time = self.real_code.get_time(self.subset)
            self.real_code._managing_subset = self.subset
            self.real_code._valid_subsets = {}
            self.real_code._cached_particles = datamodel.Particles()
            self.real_code.channel_to_model = self.real_code.particles.new_channel_to(self.real_code._cached_particles)
            self.real_code.channel_to_code = self.real_code._cached_particles.new_channel_to(self.real_code.particles)
            self.t0 = self.real_code.get_time(self.subset)
            self.real_code.t0 = self.t0
            self.real_code._initial = False
        elif self.real_code._managing_subset < 0:
            self.real_code._managing_subset = self.subset
        self.t0 = self.real_code.get_time(self.subset)
        self.particles = datamodel.UpdatingParticlesSubset(self.real_code._cached_particles, lambda x  : x.index_of_the_set == self.subset)
        self.real_code._valid_subsets[self.subset] = self
        self.G = G
        self.total_delta_kinetic_energy = quantities.zero
        self.total_delta_potential_energy = quantities.zero



    def evolve_model(self, end_time, only_subset = False):
        if only_subset:
            while True:
                print(end_time.as_quantity_in(units.yr))
                collision_detection = self.real_code.stopping_conditions.collision_detection
                self.real_code.channel_to_code.copy()
                if self.real_code.t0 is None:
                    self.real_code.t0 = self.t0
                    
                self.real_code.evolve_model(end_time + self.real_code.t0, self.subset)
                #self.real_code._cached_model_time = self.real_code.get_time(self.subset)
                #self.real_code.particles.synchronize_to(self.real_code._cached_particles)
                self.real_code.channel_to_model.copy()
                if collision_detection.is_set():
                    self.handle_collision(datamodel.Particles(particles = [collision_detection.particles(0)[0], collision_detection.particles(1)[0]]))
                else:
                    break
        elif self.subset == self.real_code._managing_subset:
            t0 = pytime.time()
            collision_detection = self.real_code.stopping_conditions.collision_detection
            while True:
                
                self.real_code.channel_to_code.copy()
                if self.real_code.t0 is None:
                    self.real_code.t0 = self.t0
                    
                self.real_code.evolve_model(end_time + self.real_code.t0)
                self.real_code._cached_model_time = self.real_code.get_time(self.subset)
                self.real_code.particles.synchronize_to(self.real_code._cached_particles)
                self.real_code.channel_to_model.copy()
                if collision_detection.is_set():
                    self.handle_collision(datamodel.Particles(particles = [collision_detection.particles(0)[0], collision_detection.particles(1)[0]]))
                else:
                    break
            t1 = pytime.time()
            return (t1-t0, len(self.real_code._valid_subsets))









    @property
    def model_time(self):
        return self.real_code.get_time(self.subset) #self.real_code._cached_model_time
        return self.real_code.get_time(self.subset) #self.real_code._cached_model_time
   
    
    def stop(self):
        if self.subset >= 0:
            del self.real_code._valid_subsets[self.subset]                                                                                                  
            if self.subset == self.real_code._managing_subset:
                self.real_code._managing_subset = -2
                self.real_code.t0 = None
                for i in list(self.real_code._valid_subsets.keys()):
                    self.real_code._managing_subset = i
                    break                                
            particles = self.real_code.particles
            particles.remove_particles(particles[particles.index_of_the_set == self.subset])
            
            particles = self.real_code._cached_particles
            particles.remove_particles(particles[particles.index_of_the_set == self.subset])
            self.real_code.stop_subset(self.subset)
            self.subset = -1









    @property
    def kinetic_energy(self):
        return self.real_code.get_kinetic_energy(self.subset) - self.total_delta_kinetic_energy  




    @property
    def potential_energy(self):
        return self.real_code.get_potential_energy(self.subset)  - self.total_delta_potential_energy





    def update(self):
        self.real_code.particles.synchronize_to(self.real_code._cached_particles)
        
    def handle_collision(self, collision):
        
        #print len(self.real_code._cached_particles), len(self.real_code.particles)
        
        #collision = self.collided_particles.add_particles(collision)
        p = datamodel.Particles()
        collision = p.add_particles(collision)
        collision_in_real_code = collision.get_intersecting_subset_in(self.real_code._cached_particles)
        print(len(collision), len(collision_in_real_code))
        print(collision.key)
        print(collision_in_real_code)
        subset = collision_in_real_code.index_of_the_set[0]
        print("handle collision in subset:", subset)
        print("keys:", collision.key)
        
        k0 = self.real_code.get_kinetic_energy(subset)                                              
        p0 = self.real_code.get_potential_energy(subset)                                              
        self.real_code.particles.remove_particles(collision)
        self.real_code._cached_particles.remove_particles(collision)
        
        total_mass = collision.mass.sum()
        heaviest = collision[0] if collision[0].mass > collision[1].mass else collision[1]
        merger_product = heaviest.copy()
        merger_product.mass = total_mass
        merger_product.position = (collision.position.sum(0)) / 2.0
        merger_product.velocity = ((collision.velocity * collision.mass.reshape((len(collision),1)) / total_mass)).sum(0)
        merger_product.radius = (collision.radius**3).sum()**(1.0/3.0)                                                                                                                                                                          
        merger_product.ancestors = collision                                                  
        merger_product.index_of_the_set =    subset                                                                                                                                                         
        self.real_code.particles.add_particle(merger_product)                                                      
        
        delta_kinetic_energy = -0.5 * (collision.mass.prod()/total_mass) * (collision.velocity[1] - collision.velocity[0]).length_squared()
        delta_potential_energy =  self.G * collision.mass.prod() /  (collision.position[1] - collision.position[0]).length()
        k1 = self.real_code.get_kinetic_energy(subset)                                              
        p1 = self.real_code.get_potential_energy(subset)                                              
        print(p1 - p0, delta_potential_energy, ( p1 - p0)/ delta_potential_energy)
        print(k1 - k0, delta_kinetic_energy, ( k1 - k0)/ delta_kinetic_energy)
        print("COLLDE:", p1 - p0 + k1 - k0)
        #self.total_delta_kinetic_energy += delta_kinetic_energy
        #self.total_delta_potential_energy += p1-p0
        
        merger_product.delta_kinetic_energy = delta_kinetic_energy
        merger_product.delta_potential_energy = delta_potential_energy      
        
        subset_code = self.get_subset_code(subset)
        if subset_code is None:
             print("Could not find the subset code with number: ", subset, ", this is a bug!")
        else:
            subset_code.total_delta_kinetic_energy += delta_kinetic_energy
            subset_code.total_delta_potential_energy += p1-p0
                                                                   
        self.real_code._cached_particles.add_particle(merger_product)
        #print len(self.real_code._cached_particles), len(self.real_code.particles)
        if False and self.stopping_conditions.collision_detection.is_enabled():
            self.stopping_conditions.collision_detection.set(
                collision[0].as_set(),
                collision[1].as_set(),
                merger_product.as_set(),
            )





    def report(self, tref):
        print("model time for", id(self), '-', self.subset , "is: " , self.real_code.get_time(self.subset) / tref, ", nr of particles:", len(self.particles))

    def evolve_model_async(self, end_time, only_subset = False):
        if only_subset:
            return self.evolve_model(end_time, only_subset)
        elif self.subset == self.real_code._managing_subset:
            collision_detection = self.real_code.stopping_conditions.collision_detection
            
            self.real_code.channel_to_code.copy()
            if self.real_code.t0 is None:
                self.real_code.t0 = self.t0
            request = self.real_code.evolve_model.async(end_time + self.real_code.t0, -1)
            def update(function):
                ignore = function()
                self.real_code._cached_model_time = self.real_code.get_time(self.subset)
                self.real_code.particles.synchronize_to(self.real_code._cached_particles)
                self.real_code.channel_to_model.copy()
                
                if collision_detection.is_set():
                    self.handle_collision(datamodel.Particles(particles = [collision_detection.particles(0)[0], collision_detection.particles(1)[0]]))
                return collision_detection.is_set()
            
            request.add_result_handler(update)
            
            return request










    def get_subset_code(self, index):
        if index in self.real_code._valid_subsets:
            return self.real_code._valid_subsets[index]
        return None

