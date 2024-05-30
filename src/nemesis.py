import numpy as np
import os
import queue
import threading

from amuse.community.huayno.interface import Huayno
from amuse.community.ph4.interface import ph4
from amuse.community.seba.interface import SeBa
from amuse.couple import bridge
from amuse.couple.bridge import CalculateFieldForParticles
from amuse.datamodel import Particles
from amuse.ext.basicgraph import UnionFind
from amuse.ext.composition_methods import SPLIT_4TH_S_M4
from amuse.ext.galactic_potentials import MWpotentialBovy2015
from amuse.ext.orbital_elements import orbital_elements_from_binary
from amuse.lab import write_set_to_file
from amuse.units import units, constants

from src.environment_functions import ejection_checker
from src.environment_functions import set_parent_radius, planet_radius
from src.environment_functions import natal_kick_pdf, ZAMS_radius
from src.grav_correctors import CorrectionFromCompoundParticle
from src.grav_correctors import CorrectionForCompoundParticle
from src.hierarchical_particles import HierarchicalParticles

import time as cpu_time

class Nemesis(object):
    def __init__(self, par_conv, child_conv, dt, 
                 code_dt=0.03, par_nworker=1, 
                 dE_track=False, star_evol=False, 
                 gal_field=False):
        """Class setting up the simulation.
        Inputs:
        par_conv:  Parent N-body converter
        child_conv:  Children N-body converter
        dt:  Diagnostic time step
        code_dt:  Internal time step
        par_nworker:  Number of workers for global integrator
        dE_track:  Flag turning on/off energy error tracker
        star_evol:  Flag turning on/off stellar evolution
        gal_field:  Flag turning on/off galactic field
        """
        self.code_timestep = code_dt
        self.par_nworker = par_nworker
        self.star_evol = star_evol
        self.test_particles = None
      
        self.parent_code = self.parent_worker(par_conv)
        self.subsys_code = self.sub_worker
        self.sys_kickers = self.py_worker
        if (self.star_evol):
            self.stellar_code = self.stellar_worker()

        self.child_conv = child_conv
        self.dt = dt
        self.dE_track = dE_track
        self.gal_field = gal_field
        self.use_threading = False
      
        self.particles = HierarchicalParticles(self.parent_code.particles)
        self.subcodes = dict()
        self.time_offsets = dict()
        
        self.timestep = None
        self.coll_dir = None
        self.min_mass_evol = None
        self.ejected_path = None
        
        self.no_ejec = 0
        self.dt_step = 0

    def commit_particles(self, child_conv):
        """Commit particle system"""
        self.particles.recenter_subsystems()
        length_unit = self.particles.radius.unit
        if not hasattr(self.particles, "sub_worker_radius"):
            self.particles.sub_worker_radius = 0. | length_unit

        subsystems = self.particles.collection_attributes.subsystems
        for parent, code in list(self.subcodes.items()):
            if ((parent in subsystems) and \
                (subsystems[parent] is self.subcodes[parent].particles)):
                continue
            self.time_offsets.pop(code)
            del code

        for parent, sys in list(subsystems.items()):
            parent.radius = set_parent_radius(np.sum(sys.mass), self.dt)
            if parent not in self.subcodes:
                gravity_code = self.subsys_code(sys, child_conv)
                offset = (self.model_time - gravity_code.model_time)
                self.time_offsets[gravity_code] = offset
                subsystems[parent] = sys
                self.subcodes[parent] = gravity_code

        if (self.star_evol):
            parti = self.particles.all()
            self.stars = parti[parti.mass > self.min_mass_evol]
            self.stars -= self.stars[self.stars.type == "JMO"]
            stellar_code = self.stellar_code
            stellar_code.particles.add_particle(self.stars)

        if (self.gal_field):
            self.setup_bridge()
        else:
            self.evolve_code = self.parent_code

    def setup_bridge(self):
        self.MWG = MWpotentialBovy2015()
        gravity = bridge.Bridge(use_threading=False,
                                method=SPLIT_4TH_S_M4
                                )
        gravity.add_system(self.parent_code, (self.MWG, ))
        gravity.timestep = self.timestep
        self.grav_bridge = gravity
        self.evolve_code = self.grav_bridge
    
    def stellar_worker(self):
        """Define stellar evolution integrator"""
        return SeBa()

    def parent_worker(self, par_conv):
        """Define global integrator"""
        code = ph4(par_conv, number_of_workers=self.par_nworker)
        code.parameters.epsilon_squared = (0.|units.au)**2
        code.parameters.timestep_parameter = self.code_timestep
        return code

    def py_worker(self):
        """Define algorithm in computing gravitational fields"""
        return CalculateFieldForParticles(gravity_constant=constants.G)
      
    def sub_worker(self, children, child_conv):
        """Defining the local integrator based on system population"""
        code = Huayno(child_conv)
        code.particles.add_particles(children)
        code.parameters.timestep_parameter = 0.1
        code.set_integrator("SHARED4_COLLISIONS")
        return code

    def star_channel_copier(self):
        """Copy attributes from stellar evolution code to local particle set"""
        stars = self.stellar_code.particles
        stars.new_channel_to(self.parent_code.particles).copy_attributes(["mass"])
        for children in self.subcodes.values():
            channel = stars.new_channel_to(children.particles)
            channel.copy_attributes(["mass", "radius"])
            
    def grav_channel_copier(self, grav_particles, local_particles, attributes):
        """Copy attributes from gravitational code to local particle set"""
        channel = grav_particles.new_channel_to(local_particles)
        channel.copy_attributes(attributes)
    
    def evolve_model(self, tend, timestep=None):
        """Evolve the system"""
        if timestep is None:
            timestep = tend-self.model_time
            
        print("============================================================================")
        print("Evolving till: ", (tend-timestep/2.).in_(units.Myr))
        while self.model_time < (tend - timestep/2.)*(1 - 1.e-12):
            evolve_time = self.model_time
            self.dEa = 0 | units.J
            
            t0 = cpu_time.time()
            if (self.star_evol):
                self.stellar_evolution(self.model_time+timestep/2.)
                self.star_channel_copier()
            t1 = cpu_time.time()
            print("Time taken for stellar: {:}".format(t1-t0))

            t0 = cpu_time.time()
            self.correction_kicks(self.particles, 
                                  self.particles.collection_attributes.subsystems,
                                  timestep/2.
                                  )
            t1 = cpu_time.time()
            print("Time taken for kicks: {:}".format(t1-t0))

            t0 = cpu_time.time()
            self.drift_global(self.model_time+timestep, 
                              self.model_time+timestep/2.
                              )
            t1 = cpu_time.time()
            print("Time taken for drift: {:}".format(t1-t0))

            t0 = cpu_time.time()
            self.drift_child(evolve_time+timestep)
            t1 = cpu_time.time()
            print("Time taken for children: {:}".format(t1-t0))

            t0 = cpu_time.time()
            if (self.star_evol):
                stellar_time = self.stellar_code.model_time
                self.stellar_evolution(stellar_time+timestep/2.)
            t1 = cpu_time.time()
            print("Time taken for stellar: {:}".format(t1-t0))

            t0 = cpu_time.time()
            self.correction_kicks(self.particles, 
                                  self.particles.collection_attributes.subsystems,
                                  timestep/2.
                                  )
            t1 = cpu_time.time()
            print("Time taken for kicks: {:}".format(t1-t0))
            
            t0 = cpu_time.time()
            self.grav_channel_copier(self.parent_code.particles,
                                     self.particles,
                                     ["x","y","z","vx","vy","vz"]
                                     )
            
            self.split_subcodes()
            t1 = cpu_time.time()
            print("Time taken for splitting: {:}".format(t1-t0))

            t0 = cpu_time.time()
            ejected_idx = ejection_checker(self.particles.copy_to_memory())
            self.ejection_remover(ejected_idx)
            t1 = cpu_time.time()
            print("Time taken for ejection: {:}".format(t1-t0))

    def split_subcodes(self):
        """Function tracking the dissolution of a parent system"""
        subsystems = self.particles.collection_attributes.subsystems
        subcodes = self.subcodes
            
        for parent, subsys in list(subsystems.items()):
            radius = parent.radius
            self.grav_channel_copier(subcodes[parent].particles, subsys,
                                     ["x","y","z","vx","vy","vz"]
                                     )
            components = subsys.connected_components(threshold=1.75*radius)
            
            if len(components) > 1:  # Checking for dissolution of system
                print("...Splitting subcode...")
                parent_pos = parent.position
                parent_vel = parent.velocity
                parent_type = parent.type
                
                self.particles.remove_particle(parent)
                code = subcodes.pop(parent)
                offset = self.time_offsets.pop(code)
                
                for c in components:
                    sys = c.copy_to_memory()
                    sys.position += parent_pos
                    sys.velocity += parent_vel
                    if len(sys) > 1:  # If system > 1 make a subsystem
                        newcode = self.subsys_code(sys, self.child_conv)
                        newcode.particles.move_to_center()  # Prevent energy drifting
                        
                        self.time_offsets[newcode] = (self.model_time - newcode.model_time)
                        newparent = self.particles.add_subsystem(sys)  # Make a parent particle and add to global
                        subcodes[newparent] = newcode
                        newparent.radius = set_parent_radius(np.sum(sys.mass), self.dt)
                        newparent.type = parent_type
                        
                    else:
                        sys.syst_id = -1
                        newparent = self.particles.add_subsystem(sys)
                        newparent.radius = set_parent_radius(newparent.mass, self.dt)
                    
                del code
               
    def ejection_remover(self, ejected_idx):
        """Output and remove ejected particles from system"""
        for idx_ in ejected_idx:
            self.no_ejec += 1
            ejected_particle = self.particles[idx_]
            
            path = self.ejected_path+"/ejec#{:}_dt_{:}".format(self.no_ejec, self.dt_step)
            if ejected_particle in self.subcodes:
                code = self.subcodes.pop(ejected_particle)
                sys = self.particles.collection_attributes.subsystems[ejected_particle]
                
                write_set_to_file(sys.savepoint(0|units.Myr), path,
                                  'amuse', close_file=True, overwrite_file=True
                                  )
                del code
            
            self.particles.remove_particle(ejected_particle)
        print("# Removed Particles: ", len(ejected_idx))
    
    def parent_merger(self, coll_time, corr_time, coll_set):
        """Resolve the merging of two parent systems.
        Inputs:
        coll_time:  Time of collision
        corr_time:  Collision correction time
        coll_set:  Colliding particle set
        """
        par = self.particles.copy_to_memory()
        subsystems = par.collection_attributes.subsystems
        collsubset, collsyst = self.evolve_coll_offset(coll_set,
                                                       subsystems, 
                                                       coll_time
                                                       )
        dt = (coll_time - corr_time)
        self.correction_kicks(collsubset, collsyst, dt)
        
        newparts = HierarchicalParticles(Particles())
        subsystems = par.collection_attributes.subsystems
        self.grav_channel_copier(self.parent_code.particles,
                                 self.particles,
                                 ["x","y","z","vx","vy","vz"]
                                 )
        
        E0 = self.energy_track
        max_id = 0
        for parti_ in collsubset:
            parti_ = parti_.as_particle_in_set(self.particles)
            max_id = max(max_id, parti_.syst_id)
            if parti_ in self.subcodes:  # Check if collider is a parent with children
                print("...Merging into some existing system...")
                code = self.subcodes.pop(parti_)
                offset = self.time_offsets.pop(code)
                parts = code.particles.copy_to_memory()
                sys = self.particles.collection_attributes.subsystems[parti_]
                
                channel = parts.new_channel_to(sys)
                channel.copy_attributes(["x","y","z","vx","vy","vz"])
                E0 = self.energy_track
                
                sys.position += parti_.position
                sys.velocity += parti_.velocity 
                newparts.add_particles(sys)
                
                del code
              
            else:  # Loop for two parent particle collisions
                print("...Merging with some isolated object...")
                new_parti = newparts.add_particle(parti_)
                new_parti.radius = parti_.sub_worker_radius
                new_parti.type = parti_.type
                
            self.particles.remove_particle(parti_)
        
        newcode = self.subsys_code(newparts, self.child_conv)
        newcode.particles.move_to_center()  # Prevent energy drift
        self.time_offsets[newcode] = self.model_time - newcode.model_time
        newparent = self.particles.add_subsystem(newparts)
        self.subcodes[newparent] = newcode
      
        most_massive_idx = newparts.mass.argmax()
        newparent.type = newparts[most_massive_idx].type
        newparent.radius = set_parent_radius(np.sum(newparts.mass), self.dt)
        
        if max_id <= 0:
            max_id = max(self.particles.all().syst_id) + 1
        newparent.syst_id = max_id
        newparts.syst_id = max_id
        
        dE = self.energy_track - E0
        self.dEa += newcode.particles.potential_energy() \
                    + newcode.particles.kinetic_energy() \
                    + dE
        if (self.gal_field):
            self.setup_bridge()
        
        return newparent
        
    def evolve_coll_offset(self, coll_set, subsystems, coll_time):
        """Function to evolve and/or resync the final moments of collision.
        Inputs:
        coll_set:  Attributes of colliding particle
        collsubsys:  Particle set of colliding particles with key words
        coll_time:  Time of simulation where collision occurs
        """
        collsubset = Particles()
        collsyst = dict()
        subsystems = self.particles.collection_attributes.subsystems
        for parti_ in coll_set:
            collsubset.add_particle(parti_)
            if parti_ in self.subcodes:
                code = self.subcodes[parti_]
                offset = self.time_offsets[code]
                stopping_condition = code.stopping_conditions.collision_detection
                stopping_condition.enable()
                while code.model_time < (coll_time - offset)*(1 - 1.e-12):
                    code.evolve_model(coll_time-offset)
                    if stopping_condition.is_set():
                        coll_time = code.model_time
                        coll_sets = self.find_coll_sets(stopping_condition.particles(0),
                                                        stopping_condition.particles(1)
                                                        )
                        for cs in coll_sets:
                            colliding_particles = Particles()
                            for p in cs:
                                colliding_particles.add_particle(p)
                            newparent = self.handle_collision(subsystems[parti_], 
                                                            parti_, coll_sets, 
                                                            coll_time, code
                                                            )
                        collsubset.remove_particle(parti_)
                        collsubset.add_particle(newparent)
                        subsystems = self.particles.collection_attributes.subsystems
        for parti_ in collsubset:              
            if parti_ in subsystems:
                collsyst[parti_] = subsystems[parti_]
        
        return collsubset, collsyst
    
    def handle_collision(self, children, parent, enc_parti, tcoll, code):
        """Merge two particles if the collision stopping condition is met
        Inputs:
        children:  The children particle set
        enc_parti:  The particles in the collision
        tcoll:  The time-stamp for which the particles collide at
        code:  The integrator used
        """
        subsystems = self.particles.collection_attributes.subsystems
        self.grav_channel_copier(code.particles, children,
                                 ["x","y","z","vx","vy","vz"]
                                 )
        
        ### Save properties
        allparts = self.particles.all()
        nmerge = np.sum(allparts.coll_events) + 1
        print("...Collision #{:} Detected...".format(nmerge))
        
        write_set_to_file(allparts.savepoint(0|units.Myr),
                os.path.join(self.coll_dir, "merger"+str(nmerge)),
                'amuse', close_file=True, overwrite_file=True
                )
        colliding_a = allparts[allparts.key == enc_parti[0].key]
        colliding_b = allparts[allparts.key == enc_parti[1].key]
        
        bin_sys = Particles()
        bin_sys.add_particle(enc_parti[0])
        bin_sys.add_particle(enc_parti[1])
        kepler_elements = orbital_elements_from_binary(bin_sys, G=constants.G)
        sem = kepler_elements[2]
        ecc = kepler_elements[3]
        inc = kepler_elements[4]
        arg_peri = kepler_elements[5]
        asc_node = kepler_elements[6]
        true_anm = kepler_elements[7]
        lines = ["Tcoll: {}".format(tcoll.in_(units.yr)),
                 "Key1: {}".format(enc_parti[0].key),
                 "Key2: {}".format(enc_parti[1].key),
                 "M1: {}".format(enc_parti[0].mass.in_(units.MSun)),
                 "M2: {}".format(enc_parti[1].mass.in_(units.MSun)),
                 "Semi-major axis: {}".format(abs(sem).in_(units.au)),
                 "Eccentricity: {}".format(ecc),
                 "Inclination: {} deg".format(inc),
                 "Argument of Periapsis: {} deg".format(arg_peri),
                 "Longitude of Asc. Node: {} deg".format(asc_node),
                 "True Anomaly: {} deg\n".format(true_anm)
                ]
        with open(os.path.join(self.coll_dir, "merger"+str(nmerge)+'.txt'), 'a') as f:
           for line_ in lines:
                f.write(line_)
                f.write('\n')

        ### Create merger remnant
        new_particle  = Particles(1)

        temp_pset = Particles()
        temp_pset.add_particle(enc_parti[0])
        temp_pset.add_particle(enc_parti[1])
        
        new_particle.mass = temp_pset.total_mass()
        new_particle.position = temp_pset.center_of_mass()
        new_particle.velocity = temp_pset.center_of_mass_velocity()
        new_particle.coll_events = colliding_a.coll_events \
                                  + colliding_b.coll_events + 1
        new_particle.syst_id = max(colliding_a.syst_id, colliding_b.syst_id)
        if "STAR" in colliding_a.type or "STAR" in colliding_b.type:
            new_particle.type = "STAR"
            new_particle.radius = ZAMS_radius(new_particle.mass)
        elif "HOST" in colliding_a.type or "HOST" in colliding_b.type:
            new_particle.type = "HOST"
            new_particle.radius = ZAMS_radius(new_particle.mass)
        else:
            new_particle.type = "PLANET"
            new_particle.radius = planet_radius(new_particle.mass)
        if new_particle.mass > self.min_mass_evol: #Lower limit for star evolution
            self.stellar_code.particles.add_particle(new_particle)

        new_particle.sub_worker_radius = new_particle.radius
        children.add_particles(new_particle)
        children.remove_particles(colliding_a)
        children.remove_particles(colliding_b)
        
        if colliding_a.key == parent.key \
            or colliding_b.key == parent.key:
                if len(children) > 1:
                    newcode = self.subsys_code(children, self.child_conv)
                    newcode.particles.move_to_center()
                    self.time_offsets[newcode] = (self.model_time - code.model_time)
                    newparent = self.particles.add_subsystem(children)
                    self.subcodes[newparent] = newcode
                    newcode.model_time += code.model_time
                else:
                    newparent = self.particles.add_subsystem(children)
                    
                #Re-mapping dictionary to new parent
                newparent.radius = set_parent_radius(np.sum(children.mass), self.dt)
                subsystems = self.particles.collection_attributes.subsystems
                subsystems[newparent] = subsystems.pop(parent)
                old_code = self.subcodes.pop(parent)
                self.time_offsets[newparent] = self.model_time - old_code.model_time
                self.subcodes[newparent] = old_code
            
                self.particles.remove_particle(parent)
                children.synchronize_to(self.subcodes[newparent].particles)
            
        else:
            newparent = parent
            children.synchronize_to(code.particles)
        
        if colliding_a.mass > self.min_mass_evol:
          self.stellar_code.particles.remove_particle(colliding_a)
        if colliding_b.mass > self.min_mass_evol:
          self.stellar_code.particles.remove_particle(colliding_b)

        return newparent
    
    def handle_supernova(self, SN_detect, bodies):
        """Function handling SN explosions
        Inputs:
        SN_detect: Detected particle set undergoing SN
        bodies:    All bodies undergoing stellar evolution
        """
        SN_particle = SN_detect.particles(0)
        for ci in range(len(SN_particle)):
            SN_parti = Particles(particles=SN_particle)
            natal_kick = natal_kick_pdf()
            natal_kick_x = natal_kick[0]
            natal_kick_y = natal_kick[1]
            natal_kick_z = natal_kick[2]
            
            SN_parti = SN_parti.get_intersecting_subset_in(bodies)
            SN_parti.vx += natal_kick_x
            SN_parti.vy += natal_kick_y
            SN_parti.vz += natal_kick_z

    def find_coll_sets(self,p1,p2):
        coll_sets = UnionFind()
        for p,q in zip(p1,p2):
            coll_sets.union(p,q)
        return coll_sets.sets()

    def stellar_evolution(self, dt):
        """Evolve stellar evolution"""
        code = self.stellar_code
        SN_detection = code.stopping_conditions.supernova_detection
        SN_detection.enable()
        while code.model_time < dt*(1 - 1.e-12):
            code.evolve_model(dt)
            if SN_detection.is_set():
                print("...Detection: SN Explosion...")
                self.handle_supernova(SN_detection, self.stars)
    
    def drift_global(self, dt, corr_time):
        """Evolve parent system for dt"""
        stopping_condition = self.parent_code.stopping_conditions.collision_detection
        stopping_condition.enable()
        while self.parent_code.model_time < dt*(1 - 1e-12):
            print("Evolve code: ", self.evolve_code, self.evolve_code.model_time.in_(units.Myr), len(self.evolve_code.particles))
            print("Parent code: ", self.parent_code, self.parent_code.model_time.in_(units.Myr), len(self.parent_code.particles))
            self.evolve_code.evolve_model(dt)
            if stopping_condition.is_set():
                print("!!! Parent Merger !!!")
                print("Bridge time: ", self.evolve_code.model_time.in_(units.Myr))
                print("Parent time: ", self.parent_code.model_time.in_(units.Myr))
                coll_time = self.parent_code.model_time
                coll_sets = self.find_coll_sets(stopping_condition.particles(0), 
                                                stopping_condition.particles(1)
                                                )
                for cs in coll_sets:
                    self.parent_merger(coll_time, corr_time, cs)
                    
    def drift_child(self, dt):
        """Evolve children system for dt."""
        def evolve_code(lock):
            """Algorithm to evolve individual children codes"""
            try:
                parent = parent_queue.get(timeout=1)  # Timeout to prevent blocking indefinitely
            except queue.Empty:
                print("!!! Error: No children system !!!")

            code = self.subcodes[parent]
            evol_time = dt - self.time_offsets[code]
            stopping_condition = code.stopping_conditions.collision_detection
            stopping_condition.enable()
            while code.model_time < evol_time*(1 - 1e-12):
                code.evolve_model(evol_time)
                if stopping_condition.is_set():
                    print("!!! COLLIDING CHILDREN !!!")
                    with lock:  # All threads stop until resolve collision
                        subsystems = self.particles.collection_attributes.subsystems
                        coll_time = code.model_time
                        coll_sets = self.find_coll_sets(stopping_condition.particles(0),
                                                        stopping_condition.particles(1)
                                                        )
                        for cs in coll_sets:
                            colliding_particles = Particles()
                            for p in cs:
                                colliding_particles.add_particle(p)
                            parent = self.handle_collision(subsystems[parent], parent, 
                                                           colliding_particles, coll_time, 
                                                           code
                                                          )
                            if len(code.particles) == 1:
                                single_parents.append(parent)
            parent_queue.task_done()
        
        single_parents = [ ]
        parent_queue = queue.Queue()
        for parent in self.subcodes.keys():
            parent_queue.put(parent)
            self.key = parent.key
            
        lock = threading.Lock()
        threads = []
        for worker in range(len(self.subcodes.values())):
            th = threading.Thread(target=evolve_code, args=(lock,))
            th.daemon = True
            th.start()
            threads.append(th)
            
        for th in threads:
            th.join()  # Wait for all threads to finish
            
        subsystems = self.particles.collection_attributes.subsystems
        for parent in single_parents:  # Remove single children systems:
            old_subcode = self.subcodes.pop(parent)
            old_offset = self.time_offsets.pop(old_subcode)
            old_subsyst = subsystems.pop(parent)
            
            del old_subcode
            del old_offset      
            del old_subsyst
                
    def kick_particles(self, particles, corr_code, dt):
        """Apply correction kicks on particles
        Inputs:
        particles:  Particles to correct accelerations of
        corr_code:  Code containing information on difference in gravitational field
        dt:  Time-step of correction kick
        """
        parts = particles.copy_to_memory()
        ax,ay,az = corr_code.get_gravity_at_point(parts.radius,
                                                  parts.x, 
                                                  parts.y, 
                                                  parts.z
                                                  )
        parts.vx += dt*ax
        parts.vy += dt*ay
        parts.vz += dt*az
        
        channel = parts.new_channel_to(particles)
        channel.copy_attributes(["vx","vy","vz"])

    def correction_kicks(self, particles, subsystems, dt):
        """Apply correcting kicks onto children and parent particles"""
        if subsystems is not None and len(particles) > 1:
            attributes = ["x","y","z","vx","vy","vz"]
            
            # Kick parent particles
            self.grav_channel_copier(self.parent_code.particles, 
                                     self.particles, attributes
                                     )
            corr_chd = CorrectionFromCompoundParticle(self.particles, 
                                                      subsystems
                                                      )
            self.kick_particles(self.particles, corr_chd, dt)
            self.grav_channel_copier(self.particles, 
                                     self.parent_code.particles,
                                     ["vx","vy","vz"]
                                     )
            
            # Kick children
            corr_par = CorrectionForCompoundParticle(self.particles, 
                                                     None, None
                                                     )
            for parent, subsyst in subsystems.items():
                self.grav_channel_copier(self.subcodes[parent].particles, 
                                         subsyst, attributes
                                         )
                corr_par.parent = parent
                corr_par.system = subsyst
                self.kick_particles(subsyst, corr_par, dt)
                self.grav_channel_copier(subsyst, 
                                         self.subcodes[parent].particles, 
                                         ["vx","vy","vz"]
                                         )
                
    def child_energy_calc(self):
        """Calculate total energy of children systems"""
        E = 0 | units.J
        for child_ in self.subcodes.values():
            E += child_.kinetic_energy + child_.potential_energy
        return E
  
    def get_potential_at_point(self, radius, x, y, z):
        """Get the potential field at some position (x,y,z)"""
        phi = self.parent_code.get_potential_at_point(radius, x, y, z)
        return phi

    def get_gravity_at_point(self, radius, x, y, z):
        """Get gravitational force felt at some position (x,y,z)"""
        ax,ay,az = self.parent_code.get_gravity_at_point(radius, x, y, z)
        return ax, ay, az

    @property
    def potential_energy(self):
        """Track the potential energy of the system"""
        Ep = self.parent_code.potential_energy
        corrector = CorrectionForCompoundParticle(self.particles, 
                                                  None, self.sys_kickers,
                                                  self.particles.all()
                                                  )
        for parent, code in self.subcodes.items():
            Ep += code.potential_energy
            if len(self.particles)>1:
                corrector.parent = parent
                parts = code.particles.copy()
                potential = corrector.get_potential_at_point(parts.radius,
                                                             parts.x,
                                                             parts.y,
                                                             parts.z
                                                             )
                Ep += potential
        return Ep

    @property
    def kinetic_energy(self): 
        """Track the kinetic energy of the system""" 
        Ek = self.parent_code.kinetic_energy
        for code in self.subcodes.values():
            Ek += code.kinetic_energy
        return Ek

    @property
    def energy_track(self):
        """Extract energy of all particles"""
        p = self.particles.all()
        Eall = p.kinetic_energy() + p.potential_energy()
        return Eall
    
    @property
    def model_time(self):  
        """Extract the global integrator model time"""
        return self.parent_code.model_time