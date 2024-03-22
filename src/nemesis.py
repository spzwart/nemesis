import multiprocessing
import numpy as np
import os
import queue
import threading
import time as cpu_time

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

from src.environment_functions import parent_radius, planet_radius, ZAMS_radius
from src.grav_correctors import CorrectionFromCompoundParticle, CorrectionForCompoundParticle
from src.hierarchical_particles import HierarchicalParticles


def potential_energy(system, get_potential):
    parts = system.particles.copy()
    pot = get_potential(parts.radius, parts.x, parts.y, parts.z)
    return (pot*parts.mass).sum()/2 


class Nemesis(object):
    def __init__(self, par_conv, chd_conv, dt, 
                 code_dt=0.01, par_nworker=1, 
                 dE_track=False, star_evol=False, 
                 gal_field=False):
        """Class setting up the simulation.
           Inputs:
           par_conv:      Parent N-body converter
           chd_conv:      Children N-body converter
           dt:            Diagnostic time step
           code_dt:       Internal time step
           par_nworker:   Number of workers for global integrator
           dE_track:      Track energy changes
           star_evol:     Flag turning on/off stellar evolution
           gal_field:     Flag turning on/off galactic field
        """

        self.code_timestep = code_dt
        self.par_nworker = par_nworker
        self.star_evol = star_evol
      
        self.parent_code = self.parent_worker(par_conv)
        self.subsys_code = self.sub_worker
        self.sys_kickers = self.py_worker
        if (self.star_evol):
            self.stellar_code = self.stellar_worker()

        self.chd_conv = chd_conv
        self.dt = dt
        self.dE_track = dE_track
        self.gal_field = gal_field
        self.use_threading = False

        self.event_key = [ ]
        self.event_time = [ ]
        self.event_type = [ ]
      
        self.particles = HierarchicalParticles(self.parent_code.particles)
        self.subcodes = dict()
        self.time_offsets = dict()
        self.timestep = None
        self.coll_dir = None
        self.min_mass_evol = None

    def commit_particles(self, chd_conv):
        """Commit particle system"""

        self.particles.recenter_subsystems()
        length_unit = self.particles.radius.unit
        if not hasattr(self.particles, "sub_worker_radius"):
            self.particles.sub_worker_radius = 0. | length_unit

        subsystems = self.particles.collection_attributes.subsystems
        subcodes = self.subcodes
        for parent, code in list(subcodes.items()):
            if ((parent in subsystems) and \
                (subsystems[parent] is subcodes[parent].particles)):
                continue
            self.time_offsets.pop(code)
            del code

        for parent, sys in subsystems.items():
            parent.radius = parent_radius(np.sum(sys.mass), self.dt)
            if parent not in subcodes:
                gravity_code = self.subsys_code(sys, chd_conv)
                offset = (self.model_time-gravity_code.model_time)
                self.time_offsets[gravity_code] = offset
                subsystems[parent] = sys
                subcodes[parent] = gravity_code

        if (self.star_evol):
            parti = self.particles.all()
            self.stars = parti[parti.type == "STAR"]
            stellar_code = self.stellar_code
            stellar_code.particles.add_particle(self.stars)

        if (self.gal_field):
            self.MWG = MWpotentialBovy2015()
            gravity = bridge.Bridge(use_threading=False,
                                    method=SPLIT_4TH_S_M4
                                    )
            gravity.add_system(self.parent_code, (self.MWG, ))
            gravity.timestep = self.timestep
            self.grav_bridge = gravity
 
    def recommit_particles(self):
        self.commit_particles()

    def channel_makers(self):
        """Copy global code data to local particle set"""
        parents = self.parent_code.particles
        self.par_code_to_local = parents.new_channel_to(self.particles)

    def stellar_worker(self):
        return SeBa()

    def parent_worker(self, par_conv):
        """Defining the global integrator"""

        code = ph4(par_conv, number_of_workers=self.par_nworker)
        code.parameters.epsilon_squared = (0.|units.au)**2
        code.parameters.timestep_parameter = self.code_timestep
        return code

    def py_worker(self):
        """Defining the bridging mechanism"""
        return CalculateFieldForParticles(gravity_constant=constants.G)
      
    def sub_worker(self, cset, chd_conv):
        """Defining the local integrator based on system population"""

        code = Huayno(chd_conv)
        code.particles.add_particles(cset)
        code.set_integrator("SHARED4_COLLISIONS")
        return code

    def grav_channel_copier(self):
        self.par_code_to_local.copy_attributes(["mass","vx","vy","vz","x","y","z"])
        subsystems = self.particles.collection_attributes.subsystems
        for parent, child_code in self.subcodes.items():
            children = subsystems[parent]
            channel = child_code.particles.new_channel_to(children)
            channel.copy()

    def star_channel_copier(self):
        stars = self.stellar_code.particles
        stars.new_channel_to(self.parent_code.particles).copy_attributes(["mass"])
        for children in self.subcodes.values():
            channel = stars.new_channel_to(children.particles, 
                                           attributes=["mass", "radius"],
                                           target_names=["mass", "radius"]
                                           )
            channel.copy()

    def evolve_model(self, tend, timestep=None):
        """Evolve the different integrators"""

        if timestep is None:
            timestep = tend-self.model_time
        evol_time = self.model_time
        
        while evol_time < (tend-timestep/2.):
            evol_time = self.model_time
            self.dEa = 0 | units.J
            self.save_snap = False
            t2 = cpu_time.time()
            if (self.star_evol):
                self.stellar_evolution(evol_time+timestep/2.)
                self.star_channel_copier()
                t1 = cpu_time.time()
                print("Time taken for Star Evol. : ", t1-t2)
            self.corr_kick_children(timestep/2.)
            t2 = cpu_time.time()
            print("Time taken for Kicking: ", t2-t1)
            self.drift_global(evol_time+timestep, 
                              evol_time+timestep/2.)
            t1 = cpu_time.time()
            print("Time taken for Global", t1-t2)
            self.drift_child(evol_time+timestep)
            t2 = cpu_time.time()
            print("Time taken for Local", t2-t1)
            self.corr_kick_children(timestep/2.)
            t1 = cpu_time.time()
            print("Time taken for Kicking: ", t1-t2)
            self.split_subcodes()
            t1 = cpu_time.time()
            print("Time taken for Splitting: ", t2-t1)
            t2 = cpu_time.time()
            if (self.star_evol):
                self.stellar_evolution(evol_time+timestep/2.)
                self.star_channel_copier()
                t1 = cpu_time.time()
                print("Time taken for Star Evol. : ", t1-t2)
        self.grav_channel_copier()

    def energy_track(self):
        """Extract energy of all particles"""

        p = self.particles.all()
        Eall = p.kinetic_energy()+p.potential_energy()
        return Eall

    def split_subcodes(self):
        """Function tracking the dissolution of a parent system"""

        subsystems = self.particles.collection_attributes.subsystems
        subcodes = self.subcodes
        for parent, subsys in list(subsystems.items()):
            radius = parent.radius
            components = subsys.connected_components(threshold=1.75*radius)
            if len(components) > 1:  # Checking for dissolution of system
                print("...Splitting subcode...")
                self.save_snap = True
                parent_pos = parent.position
                parent_vel = parent.velocity
              
                self.particles.remove_particle(parent)
                code = subcodes.pop(parent)  # Extract and remove dissolved system integrator
                offset = self.time_offsets.pop(code)
              
                keys = [ ]
                for c in components:
                    sys = c.copy_to_memory()
                    sys.position += parent_pos 
                    sys.velocity += parent_vel
                    if len(sys)>1:  # If system > 1 make a subsystem
                        newcode = self.subsys_code(sys, self.chd_conv)
                        self.time_offsets[newcode] = (self.model_time - newcode.model_time)
                        newcode.particles.add_particles(sys)
                        newparent = self.particles.add_subsystem(sys)  # Make a parent particle and add to global
                        subcodes[newparent] = newcode

                        newparent.radius = parent_radius(np.sum(sys.mass), self.dt)
                        keys = np.concatenate((keys, sys.key), axis=None)
                    else:                              
                        newparent = self.particles.add_subsystem(sys)
                        newparent.radius = parent_radius(newparent.mass, self.dt)
                        keys = np.concatenate((keys, sys.key), axis=None)
                        
                # Track dissolution event
                self.event_key = np.concatenate((self.event_key, keys), axis=None)
                self.event_time = np.concatenate((self.event_time, self.model_time), axis=None)
                self.event_type = np.concatenate((self.event_type, "Parent Dissolve"), axis=None)

                del code
                self.particles.remove_particle(parent)  # New parent systems

    def parent_merger(self, coll_time, corr_time, coll_set):
        """Inputs:
        coll_time:   Time of collision
        corr_time:   Time to correct integration after collision occurs
        coll_set:    Colliding particle set
        """
      
        self.save_snap = True
        par = self.particles.copy_to_memory()
        subsystems = par.collection_attributes.subsystems
        collsubset, collsyst = self.evolve_coll_offset(coll_set,subsystems, 
                                                       coll_time
                                                       )
      
        self.correction_kicks(collsubset, collsyst, coll_time-corr_time)
  
        keys = [ ]
        newparts = HierarchicalParticles(Particles())
        subsystems = par.collection_attributes.subsystems
        for parti_ in collsubset:
            parti_ = parti_.as_particle_in_set(self.particles)
            if parti_ in self.subcodes:  # Check if collider is a parent with children
                code = self.subcodes.pop(parti_)
                offset = self.time_offsets.pop(code)
                parts = code.particles.copy_to_memory()
                chd = self.particles.collection_attributes.subsystems[parti_]
              
                parts.position += parti_.position
                parts.velocity += parti_.velocity
                parts.syst_id = parti_.syst_id
                parts.type = chd.type
                newparts.add_particles(parts)
                
                del code
                keys = np.concatenate((keys, parts.key), axis=None)
              
            else:  # Loop for two parent particle collisions
                new_parti = newparts.add_particle(parti_)
                new_parti.radius = parti_.sub_worker_radius
                new_parti.syst_id = parti_.syst_id
                keys = np.concatenate((keys, parti_.key), axis=None)
              
            self.particles.remove_particle(parti_)
            self.particles.synchronize_to(self.parent_code.particles)
        newcode = self.subsys_code(newparts, self.chd_conv)
        self.time_offsets[newcode] = (self.model_time-newcode.model_time)
        newparent = self.particles.add_subsystem(newparts)
        self.subcodes[newparent] = newcode
      
        most_massive_idx = newparts.mass.argmax()
        newparent.type = newparts[most_massive_idx].type
        newparent.radius = parent_radius(np.sum(newparts.mass), self.dt)
        if len(newparts[newparts.syst_id <= 0]) == len(newparts):
            newparent.syst_id = -1
        else:
            newparent.syst_id = max(newparts.syst_id)
      
        self.event_key = np.concatenate((self.event_key, keys), axis=None)
        self.event_time = np.concatenate((self.event_time, self.model_time), axis=None)
        self.event_type = np.concatenate((self.event_type, "Parent Merger"), axis=None)
  
    def evolve_coll_offset(self, coll_set, subsystems, coll_time):
        """Function to evolve and/or resync the final moments of collision.
          Inputs:
          coll_set:    Attributes of colliding particle
          collsubsys:  Particle set of colliding particles with key words
          coll_time:   Time of simulation where collision occurs
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
                while code.model_time < (coll_time - offset):
                    code.evolve_model(coll_time-offset)
                    if stopping_condition.is_set():
                        coll_time = code.model_time
                        coll_sets = Particles(particles=[stopping_condition.particles(0), 
                                                         stopping_condition.particles(1)
                                                        ]
                                              )
                        collsubset.remove_particle(parti_)
                        subsystems = self.particles.collection_attributes.subsystems
                        newparent = self.handle_collision(subsystems[parti_], 
                                                          parti_, coll_sets, 
                                                          coll_time, code
                                                          )
                        collsubset.add_particle(newparent)
                        subsystems = self.particles.collection_attributes.subsystems
        
        for parti_ in collsubset:              
            if parti_ in subsystems:
                collsyst[parti_] = subsystems[parti_]
        self.grav_channel_copier()
      
        return collsubset, collsyst
    
    def handle_collision(self, children, parent, enc_parti, tcoll, code):
      """Merge two particles if the collision stopping condition is met
         Inputs:
         children:   The children particle set
         enc_parti:  The particles in the collision
         tcoll:      The time-stamp for which the particles collide at
         code:       The integrator used
      """
    
      ### Save properties
      #self.grav_channel_copier()
      allparts = self.particles.all()
      colliding_a = allparts[allparts.key == enc_parti[0].key]
      colliding_b = allparts[allparts.key == enc_parti[1].key]
    
      nmerge = np.sum(allparts.coll_events)+1
      write_set_to_file(allparts.savepoint(0|units.Myr),
          os.path.join(self.coll_dir, "merger"+str(nmerge)),
          'amuse', close_file=True, overwrite_file=True
          )
      print("...Collision #{:} Detected...".format(nmerge))

      bin_sys = Particles()
      bin_sys.add_particle(colliding_a)
      bin_sys.add_particle(colliding_b)
      kepler_elements = orbital_elements_from_binary(bin_sys, G=constants.G)
      sem = kepler_elements[2]
      ecc = kepler_elements[3]
      inc = kepler_elements[4]
      arg_peri = kepler_elements[5]
      asc_node = kepler_elements[6]
      true_anm = kepler_elements[7]
      lines = ["Tcoll: {}".format(tcoll.in_(units.yr)),
               "Key1: {}".format(colliding_a.key),
               "Key2: {}".format(colliding_b.key),
               "M1: {}".format(colliding_a.mass.in_(units.MSun)),
               "M2: {}".format(colliding_b.mass.in_(units.MSun)),
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
      temp_pset.add_particle(colliding_a)
      temp_pset.add_particle(colliding_b)
    
      new_particle.mass = temp_pset.total_mass()
      new_particle.position = temp_pset.center_of_mass()
      new_particle.velocity = temp_pset.center_of_mass_velocity()
      new_particle.coll_events = np.sum(temp_pset.coll_events)+1
      new_particle.syst_id = max(temp_pset.syst_id)
      if "STAR" in colliding_a.type or "STAR" in colliding_b.type:
          new_particle.type = "STAR"
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
              newcode = self.subcodes(children, self.chd_conv)
              self.time_offsets[newcode] = (self.model_time - code.model_time)
              newcode.particles.add_particles(children)
              newparent = self.particles.add_subsytem(newcode.particles)
              self.subcodes[newparent] = newcode
          else:
              newparent = self.particles.add_subsystem(children)
        
          #Changing dictionary mappings
          subsystems = self.particles.collection_attributes.subsystems
          subsystems[newparent] = subsystems.pop(parent)
          old_code = self.subcodes.pop(parent)
          self.time_offsets[newparent] = self.time_offsets[old_code]
          self.subcodes[newparent] = old_code
        
          self.particles.remove_particle(parent)
          children.synchronize_to(self.subcodes[newparent].particles)
          newparent.radius = parent_radius(np.sum(children.mass), self.dt)
    
      if colliding_a.type == "STAR":
        self.stellar_code.particles.remove_particle(colliding_a)
      if colliding_b.type == "STAR":
        self.stellar_code.particles.remove_particle(colliding_b)

      self.event_key = np.concatenate((self.event_key, 
                                      [colliding_a.key, colliding_b.key]), 
                                      axis=None
                                      )
      self.event_time = np.concatenate((self.event_time, tcoll), axis=None)
      self.event_type = np.concatenate((self.event_type, "Merger"), axis=None)

      return newparent
    
    def handle_supernova(self, SN_detect, bodies, time):
        """Function handling SN explosions
          Inputs:
          SN_detect: Detected particle set undergoing SN
          bodies:    All bodies undergoing stellar evolution
          time:      Time of SN explosion
        """
      
        for ci in range(len(SN_detect.particles(0))):
            SN_parti = Particles(particles=SN_detect.particles(0))
            natal_kick_x = SN_parti.natal_kick_x
            natal_kick_y = SN_parti.natal_kick_y
            natal_kick_z = SN_parti.natal_kick_z
            SN_parti = SN_parti.get_intersecting_subset_in(bodies)
            SN_parti.vx += natal_kick_x
            SN_parti.vy += natal_kick_y
            SN_parti.vz += natal_kick_z

            self.event_key = np.concatenate((self.event_key, SN_parti.key), 
                                            axis=None
                                            )
            self.event_time = np.concatenate((self.event_time, time), 
                                             axis=None
                                             )
            self.event_type = np.concatenate((self.event_type, "SN Event"), 
                                             axis=None
                                             )

    def find_coll_sets(self,p1,p2):
        coll_sets = UnionFind()
        for p,q in zip(p1,p2):
            coll_sets.union(p,q)
        return coll_sets.sets()

    def stellar_evolution(self, dt):
      code = self.stellar_code
      SN_detection = code.stopping_conditions.supernova_detection
      SN_detection.enable()
      while code.model_time < dt*(1-1.e-12):
          code.evolve_model(dt)
          if SN_detection.is_set():
              print("...Detection: SN Explosion...")
              self.handle_supernova(SN_detection, self.stars, code.model_time)
    
    def drift_global(self, dt, corr_time):
        """Evolve parent system for dt"""

        if (self.gal_field):
          codes = [self.grav_bridge, self.parent_code]
        else:
          codes = [self.parent_code]
    
        stopping_condition = codes[-1].stopping_conditions.collision_detection
        stopping_condition.enable()
        while codes[0].model_time < dt*(1-1e-12):
            codes[0].evolve_model(dt)
            if stopping_condition.is_set():
                coll_time = codes[0].model_time
                coll_sets = self.find_coll_sets(stopping_condition.particles(0), 
                                                stopping_condition.particles(1)
                                                )
                if (self.dE_track):
                    E0 = self.energy_track()
                t0 = cpu_time.time()
                for cs in coll_sets:
                    self.parent_merger(coll_time, corr_time, cs)
                t1 = cpu_time.time()
                print("Handle collision: ", t1-t0)
                if (self.dE_track):
                    E1 = self.energy_track()
                    self.dEa += (E1-E0)

    def drift_child(self, dt):
        """Evolve children system for dt."""
      
        def evolve_code():
            try:
                parent = parent_queue.get(timeout=1)  # Timeout to prevent blocking indefinitely
            except queue.Empty:
                print("!!! Error: No children system !!!")
          
            code = self.subcodes[parent]
            evol_time = dt - self.time_offsets[code]
            stopping_condition = code.stopping_conditions.collision_detection
            stopping_condition.enable()
              
            while code.model_time < evol_time*(1-1e-12):
                code.evolve_model(evol_time)
                if stopping_condition.is_set():
                    print("!!! COLLIDING CHILDREN !!!")   
                    subsystems = self.particles.collection_attributes.subsystems
                    coll_time = code.model_time
                    coll_sets = Particles(particles=[stopping_condition.particles(0), 
                                                     stopping_condition.particles(1)
                                                    ]
                                         )
                    self.handle_collision(subsystems[parent], parent, 
                                          coll_sets, coll_time, code
                                          )
            parent_queue.task_done()
        parent_queue = queue.Queue()
        for parent, sys_code in self.subcodes.items():
            parent_queue.put(parent)
      
        threads = []
        ntasks = min(multiprocessing.cpu_count(), len(self.subcodes.values()))
        for worker in range(ntasks):
            th = threading.Thread(target=evolve_code)
            th.daemon = True
            th.start()
            threads.append(th)
            #cpu_time.sleep(1)
      
        for th in threads:
            th.join()  # Wait for all threads to finish
      
        subsystems = self.particles.collection_attributes.subsystems
        for p, k in list(subsystems.items()): #Remove single children systems
            if len(k) <= 1:
                old_subcode = self.subcodes.pop(p)
                old_offset = self.time_offsets.pop(old_subcode)
                old_subsyst = subsystems.pop(p)
              
                del old_subcode
                del old_offset      
                del old_subsyst

    def kick_particles(self, particles, corr_code, dt):
        parts = particles.copy_to_memory()
        ax,ay,az = corr_code.get_gravity_at_point(parts.radius,
                                                  parts.x, parts.y, parts.z
                                                  )
                     
        parts.vx = parts.vx+dt*ax
        parts.vy = parts.vy+dt*ay
        parts.vz = parts.vz+dt*az
        channel = parts.new_channel_to(particles)
        channel.copy_attributes(["vx","vy","vz"])

    def correction_kicks(self, particles, subsystems, dt):
        if subsystems is not None and len(particles) > 1:
            corr_chd = CorrectionFromCompoundParticle(particles, 
                                                      subsystems, 
                                                      self.sys_kickers
                                                      )
            self.kick_particles(particles, corr_chd, dt)

            corr_par = CorrectionForCompoundParticle(particles, None, 
                                                    self.sys_kickers
                                                    )
            for parent, subsyst in subsystems.items():
                corr_par.parent = parent
                self.kick_particles(subsyst, corr_par, dt)

    def corr_kick_children(self,dt):
        if (self.dE_track):
            E0 = self.energy_track()
        subsystems = self.particles.collection_attributes.subsystems
        self.correction_kicks(self.particles, subsystems, dt)
        self.particles.recenter_subsystems()
        if (self.dE_track):
            E1 = self.energy_track()
            self.dEa += (E1-E0)

    def child_energy_calc(self):
        E = 0 | units.J
        for child_ in self.subcodes.values():
            E += child_.kinetic_energy + child_.potential_energy
        return E
  
    def get_potential_at_point(self, radius, x, y, z):
        phi = self.parent_code.get_potential_at_point(radius, x, y, z)
        return phi

    def get_gravity_at_point(self, radius, x, y, z):
        ax,ay,az = self.parent_code.get_gravity_at_point(radius, x, y, z)
        return ax, ay, az

    @property
    def potential_energy(self):
        Ep = self.parent_code.potential_energy
        corrector = CorrectionForCompoundParticle(self.particles, 
                                                  None, self.sys_kickers
                                                  )
        for parent, code in self.subcodes.items():
            Ep += code.potential_energy
            if len(self.particles)>1:
                corrector.parent=parent
                Ep += potential_energy(code, corrector.get_potential_at_point)
        return Ep

    @property
    def kinetic_energy(self):  
        Ek = self.parent_code.kinetic_energy
        for code in self.subcodes.values():
            Ek += code.kinetic_energy
        return Ek

    @property
    def model_time(self):  
        return self.parent_code.model_time
