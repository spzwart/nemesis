import numpy as np
import threading
import time as cpu_time

from amuse.community.ph4.interface import ph4
from amuse.community.huayno.interface import Huayno
from amuse.community.mercury.interface import Mercury
from amuse.community.seba.interface import SeBa
from amuse.community.smalln.interface import SmallN
from amuse.couple import bridge
from amuse.couple.bridge import CalculateFieldForParticles
from amuse.datamodel import Particles
from amuse.ext.basicgraph import UnionFind
from amuse.ext.composition_methods import SPLIT_4TH_S_M4
from amuse.ext.galactic_potentials import MWpotentialBovy2015
from amuse.units import units, constants

from src.environment_functions import EnvironmentFunctions
from src.grav_correctors import CorrForCompoundParti
from src.hierarchical_particles import HierarchicalParticles


def potential_energy(system, get_potential):
  parts = system.particles.copy()
  pot = get_potential(parts.radius, parts.x, parts.y, parts.z)
  return (pot*parts.mass).sum()/2 


class Nemesis(object):
  def __init__(self, par_conv, child_conv, dt, 
               code_dt, par_nworker, dE_track, 
               star_evol, gal_field):
    """Class setting up the simulation, checking 
       for dissolution of systems and evolving system.
       
       Inputs:
       par_conv:     Parent system N-body converter
       child_conv:   Children N-body converter
       dt:           Simulation time step
       code_dt:      Time-step parameter for codes
       par_nworker:  Number of workers for global integrator
       dE_track:     Track energy changes
       star_evol:    Flag turning on/off stellar evolution
       gal_field:    Flag turning on/off galactic field
    """

    self.code_timestep = code_dt
    self.par_nworker = par_nworker
    self.star_evol = star_evol
    
    if (self.star_evol):
      self.stellar_code = self.stellar_worker()
    self.parent_code = self.parent_worker(par_conv)
    self.subsys_code = self.sub_worker
    self.sys_kickers = self.py_worker
    self.dE_track = dE_track
    self.grav_bridge = None

    self.child_conv = child_conv
    self.par_conv = par_conv
    self.dt = dt
    
    self.event_key = [ ]
    self.event_mass = [ ]
    self.event_time = [ ]
    self.event_type = [ ]
    
    self.par_particles = None
    self.particles = HierarchicalParticles(self.parent_code.particles)
    self.timestep = None
    self.subcodes = dict()
    self.time_offsets = dict()
    self.use_threading = True
    self.gal_field = gal_field

    self.env_setup = EnvironmentFunctions()

  def commit_particles(self, child_conv):
    """Commit particle system"""

    subsystems = self.particles.collection_attributes.subsystems
    subcodes = self.subcodes
    self.particles.recenter_subsystems()

    length_unit = self.particles.radius.unit
    if not hasattr(self.particles, "sub_worker_radius"):
      self.particles.sub_worker_radius = 0. | length_unit
    self.init_parent_radius()

    for parent, code in list(subcodes.items()):
      if ((parent in subsystems) and \
          (subsystems[parent] is subcodes[parent].particles)):
        continue
      self.time_offsets.pop(code)
      del code

    for parent, sys in subsystems.items():
      if parent not in subcodes:
        gravity_code = self.subsys_code(sys, child_conv)
        self.time_offsets[gravity_code] = (self.model_time-gravity_code.model_time)
        gravity_code.particles.add_particles(sys)
        subsystems[parent] = sys
        subcodes[parent] = gravity_code


    if (self.star_evol):
      parti = self.particles.all()
      self.stars = parti[parti.type=="STAR"]
      self.stars.sub_worker_radius = self.stars.radius
      stellar_code = self.stellar_code
      stellar_code.particles.add_particle(self.stars)

    if (self.gal_field):
      self.MWG = MWpotentialBovy2015()
      gravity = bridge.Bridge(use_threading=False,
                              method=SPLIT_4TH_S_M4)
      gravity.add_system(self.parent_code, (self.MWG, ))
      gravity.timestep = self.timestep
      self.grav_bridge = gravity
 
  def recommit_particles(self):
    self.commit_particles()

  def init_parent_radius(self):
    """Initialise parent system radius"""
    
    parents, systems = zip(*self.particles.collection_attributes.subsystems.items())
    parents = list(parents)
    chd_mass = np.array([np.sum(chd.mass) for chd in systems])
    for parent, mass in zip(parents, chd_mass):
      parent.radius = self.env_setup.parent_radius(mass, self.dt)
    for p, s in zip(parents, systems):
      if len(s)>3:
        p.radius *= 100000
    isolated = self.particles[self.particles.radius==(0|units.m)]
    isolated.radius = self.env_setup.parent_radius(isolated.mass, self.dt)

  def channel_makers(self):
    """Function to make various channels:
       Global integrator ---> Local parent particle set
       Stellar code --------> Stellar particle set
       Stellar code --------> Global integrator
       Stellar code --------> Parent particle set
       Stellar code --------> Children integrator
       Children integrator -> Complete particle set
    """
    
    par_code = self.parent_code
    self.chnl_from_gc_to_gp = par_code.particles.new_channel_to(self.particles)
    if (self.star_evol):
      str_code = self.stellar_code
      self.chnl_from_sc_to_sp = str_code.particles.new_channel_to(self.stars,
                                    attributes = ["mass", "radius"],
                                    target_names = ["mass", "sub_worker_radius"])
      self.chnl_sc_to_gcp = str_code.particles.new_channel_to(par_code.particles,
                                    attributes = ["mass",], target_names = ["mass"])
    
    if (self.star_evol):
      self.chnl_from_sc_to_ccp = [ ]
    self.chnl_from_cc_to_all = [ ]
    self.subsys_keys = [ ]
    for sys_ in self.subcodes.values():
      sys_parti = sys_.particles
      most_massive = (sys_parti.mass).argmax()
      self.subsys_keys.append(sys_parti[most_massive].key)

      if (self.star_evol):
        chnl_temp = str_code.particles.new_channel_to(self.particles.all(),
                        attributes = ["radius", "mass"],
                        target_names = ["sub_worker_radius", "mass"])
        self.chnl_from_sc_to_ccp.append(chnl_temp)

      chnl_temp = self.particles.all().new_channel_to(sys_.particles,
                        attributes = ["sub_worker_radius", "mass"],
                        target_names = ["radius", "mass"])
      self.chnl_from_cc_to_all.append(chnl_temp)

  def grav_channel_copier(self):
    self.chnl_from_gc_to_gp.copy()
    for chnl_ in self.chnl_from_cc_to_all:
        chnl_.copy()
  
  def star_channel_copier(self):
    self.chnl_sc_to_gcp.copy()
    for chnl_ in self.chnl_from_sc_to_ccp:
        chnl_.copy()

  def stellar_worker(self):
    return SeBa()

  def parent_worker(self, par_conv):
      """Defining the global integrator"""
      code = ph4(par_conv, number_of_workers=self.par_nworker)
      code.parameters.epsilon_squared = 0. | units.au**2
      code.parameters.timestep_parameter = self.code_timestep
      return code

  def py_worker(self):
      """Defining the bridging mechanism"""
      return CalculateFieldForParticles(gravity_constant=constants.G)
      
  def sub_worker(self, cset, child_conv):
      """Defining the local integrator based on system population"""
      if len(cset)==2:
        return SmallN(child_conv)

      no_stars = len(cset[cset.type=="STAR"])+len(cset[cset.type=="BINARY STAR"])
      if no_stars==1:
        return Mercury(child_conv)

      code = Huayno(child_conv)
      code.parameters.inttype_parameter=code.inttypes.SHARED4
      return code 

  def evolve_model(self, tend, timestep=None):
    """Evolve the different integrators"""

    if timestep is None and (self.stellar_code):
      timestep = tend-min(self.model_time, self.stellar_code.model_time)
    else:
      timestep = tend-self.model_time

    while self.model_time < (tend-timestep/2.):
      self.dEa = 0 | units.J
      self.save_snap = False
      
      t2 = cpu_time.time()
      if (self.star_evol):
        self.stellar_evolution(self.model_time+timestep/2.)
      self.star_channel_copier()
      t1 = cpu_time.time()
      print("Star: ", t1-t2)
      self.kick_codes(timestep/2.)
      t2 = cpu_time.time()
      print("Kicking: ", t2-t1)
      self.drift_global(self.model_time+timestep, 
                        self.model_time+timestep/2)
      t1 = cpu_time.time()
      print("Global", t1-t2)
      self.drift_child(self.model_time+timestep)
      t2 = cpu_time.time()
      print("Local", t2-t1)
      self.kick_codes(timestep/2.)
      t1 = cpu_time.time()
      print("Kicking: ", t1-t2)
      self.split_subcodes()
      t1 = cpu_time.time()
      print("Splitting: ", t2-t1)
      if (self.star_evol):
        self.stellar_evolution(self.model_time+timestep/2.)
    self.grav_channel_copier()

  def energy_track(self):
    """Extract energy of all particles"""
    p = self.particles.all()
    Eall = p.kinetic_energy()+p.potential_energy()
    return Eall

  def split_subcodes(self):
    """Function tracking the dissolution of a parent system"""
    subsystems=self.particles.collection_attributes.subsystems
    subcodes=self.subcodes
    for parent,subsys in list(subsystems.items()):
      radius=parent.radius
      components=subsys.connected_components(threshold=1.5*radius)
      if len(components) > 1:                  #Checking for dissolution of system
        print("...Splitting subcode...")
        self.save_snap = True
        self.ftype = "parent_dissolve"
        code = subcodes.pop(parent)            #Extract and remove dissolved system integrator
        offset = self.time_offsets.pop(code)

        key_arr = [ ]
        mass_arr = [ ]
        event_time = self.model_time
        event_type = "Parent Dissolve"
        for c in components:
          sys = c.copy_to_memory()
          sys.position+=parent.position 
          sys.velocity+=parent.velocity
          if len(sys)>1:                       #If system > 1 make a subsystem
            newcode = self.subsys_code(sys, self.child_conv)
            self.time_offsets[newcode] = (self.model_time - newcode.model_time)
            newcode.particles.add_particles(sys)
            newparent = self.particles.add_subsystem(sys) #Make a parent particle and add to global
            subcodes[newparent] = newcode

            if (self.star_evol):
              chnl_temp = self.stellar_code.particles.new_channel_to(newcode.particles,
                                                    attributes = ["radius", "mass"],
                                                    target_names = ["radius", "mass"])
              self.chnl_from_sc_to_ccp.append(chnl_temp)
            chnl_temp = newcode.particles.new_channel_to(self.particles.all(),
                            attributes = ["radius", "mass"],
                            target_names = ["sub_worker_radius", "mass"])
            self.chnl_from_cc_to_all.append(chnl_temp)

            max_mass_old_idx = subsys.mass.argmax()
            idx = self.subsys_keys.index(subsys[max_mass_old_idx].key)
            del self.chnl_from_cc_to_all[idx]
            del self.subsys_keys[idx]
            if (self.star_evol):
              del self.chnl_from_sc_to_ccp[idx]

            max_mass_new_idx = sys.mass.argmax()
            self.subsys_keys.append(sys[max_mass_new_idx].key)
            newparent.radius = self.env_setup.parent_radius(np.sum(sys.mass), self.dt)

          else:                              
            newparent = self.particles.add_subsystem(sys)
            newparent.radius = self.env_setup.parent_radius(newparent.mass, self.dt)
        self.particles.remove_particle(parent)   #New parent systems

        key_arr = np.concatenate((key_arr, sys.key), axis=None)
        mass_arr = np.concatenate((mass_arr, np.sum(sys.mass)), axis=None)

        self.event_key = np.concatenate((self.event_key, key_arr), axis=None)
        self.event_mass = np.concatenate((self.event_mass, mass_arr), axis=None)
        self.event_time = np.concatenate((self.event_time, event_time), axis=None)
        self.event_type = np.concatenate((self.event_type, event_type), axis=None)

        del code
        
  def handle_collision(self, coll_time, corr_time, coll_set):
    """
    Inputs:
    coll_time:   Time of collision
    corr_time:   Time to correct integration after collision occurs
    coll_set:    Colliding particle set
    """
    
    self.save_snap = True
    self.ftype = "parent_merger"
    key_arr = [ ]
    mass_arr = [ ]
    event_time = self.model_time
    event_type = "Parent Merger"

    par = self.particles.copy_to_memory()
    subsystems = par.collection_attributes.subsystems
    collsubset, collsubsystems = self.evolve_coll_offset(coll_set, subsystems, coll_time)
    self.correction_kicks(collsubset, collsubsystems, coll_time-corr_time)

    newparts = HierarchicalParticles(Particles())
    for parti_ in coll_set:
      parti_ = parti_.as_particle_in_set(self.particles)
      if parti_ in self.subcodes:  #If collider part of subsystem, add to subsystem
        code = self.subcodes.pop(parti_)
        offset = self.time_offsets.pop(code)
        parts = code.particles.copy_to_memory()

        parts.position+=parti_.position
        parts.velocity+=parti_.velocity
        parts.disk_key = parti_.disk_key
        newparts.add_particles(parts)
        del code

        max_mass = parts.mass.argmax()
        idx = self.subsys_keys.index(parts[max_mass].key)
        del self.chnl_from_sc_to_ccp[idx]
        del self.chnl_from_cc_to_all[idx]
        del self.subsys_keys[idx]

        key_arr = np.concatenate((key_arr, parts.key), axis=None)
        mass_arr = np.concatenate((mass_arr, np.sum(parts.mass)), axis=None)

        print(newparts, parts)
        STOP

      else:  #Loop for two parent particle collisions
        new_parti = newparts.add_particle(parti_)
        new_parti.radius = parti_.sub_worker_radius
        new_parti.disk_key = parti_.disk_key

        key_arr = np.concatenate((key_arr, parti_.key), axis=None)
        mass_arr = np.concatenate((mass_arr, np.sum(parti_.mass)), axis=None)

      self.particles.remove_particle(parti_)
      self.particles.synchronize_to(self.parent_code.particles)
    t1 = cpu_time.time()
    newcode = self.subsys_code(newparts, self.child_conv)
    t2 = cpu_time.time()
    print("Reinit code: ", t2-t1)

    self.time_offsets[newcode] = (self.model_time-newcode.model_time)
    newcode.particles.add_particles(newparts)
    newparent = self.particles.add_subsystem(newparts)
    newparent.radius = self.env_setup.parent_radius(np.sum(newparts.mass), self.dt)
    if len(newparts[newparts.disk_key==-1])==len(newparts):
      newparent.disk_key =- 1
    else:
      newparent.disk_key = max(par.disk_key)+1

    if "STAR" in newparts.type:
      newparent.type = "STAR"
    else:
      newparent.type = newparts[newparts.mass==newparts.mass.max()].type
    self.subcodes[newparent]=newcode

    chnl_temp = newcode.particles.new_channel_to(self.particles.all(),
                        attributes = ["radius", "mass"],
                        target_names = ["sub_worker_radius", "mass"])
    self.chnl_from_cc_to_all.append(chnl_temp)
    if (self.star_evol):
      chnl_temp = self.stellar_code.particles.new_channel_to(newcode.particles,
                                            attributes = ["radius", "mass"],
                                            target_names = ["radius", "mass"])
      self.chnl_from_sc_to_ccp.append(chnl_temp)

    max_mass_idx = newcode.particles.mass.argmax()
    self.subsys_keys.append(newcode.particles.key[max_mass_idx])

    key_arr = np.concatenate((key_arr, newparts.key), axis=None)
    mass_arr = np.concatenate((mass_arr, np.sum(newparts.mass)), axis=None)

    self.event_key = np.concatenate((self.event_key, key_arr), axis=None)
    self.event_mass = np.concatenate((self.event_mass, mass_arr), axis=None)
    self.event_time = np.concatenate((self.event_time, event_time), axis=None)
    self.event_type = np.concatenate((self.event_type, event_type), axis=None)

  def evolve_coll_offset(self, coll_set, subsystems, coll_time):
    """Function to evolve and/or resync the final moments of collision.
    
       Inputs:
       coll_set:    Attributes of colliding particle
       collsubsys:  Particle set of colliding particles with key words
       coll_time:   Time of simulation where collision occurs
       """
    
    collsubset = Particles(2)
    collsubsystems = dict()
    for parti_ in coll_set:
      collsubset.add_particle(parti_)
      if parti_ in self.subcodes:
        code = self.subcodes[parti_]
        offset = self.time_offsets[code]
        code.evolve_model(coll_time-offset)
      if parti_ in subsystems:
        collsubsystems[parti_] = subsystems[parti_]
    
    return collsubset, collsubsystems
    
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
      SN_parti.vx+=natal_kick_x
      SN_parti.vy+=natal_kick_y
      SN_parti.vz+=natal_kick_z

      self.event_key = np.concatenate((self.event_key, SN_parti.key), axis=None)
      self.event_mass = np.concatenate((self.event_mass, SN_parti.mass), axis=None)
      self.event_time = np.concatenate((self.event_time, time), axis=None)
      self.event_type = np.concatenate((self.event_type, "SN Event"), axis=None)

  def find_coll_sets(self,p1,p2):
    coll_sets=UnionFind()
    for p,q in zip(p1,p2):
      print("Radii", (p.radius+q.radius).in_(units.au))
      print("Distance: ", (p.position-q.position).length().in_(units.au))
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
                                        stopping_condition.particles(1))
        if (self.dE_track):
          E0 = self.energy_track()
        t0 = cpu_time.time()
        for cs in coll_sets:
          self.handle_collision(coll_time, corr_time, cs)
        t1 = cpu_time.time()
        print("Handle collision: ", t1-t0)
        if (self.dE_track):
          E1 = self.energy_track()
          self.dEa+=(E1-E0)

  def drift_child(self, dt):
    """Evolve children system for dt."""

    threads = []
    for sys_ in self.subcodes.values():
      offset=self.time_offsets[sys_]
      if offset>dt:
        print("curious?")
      threads.append(threading.Thread(target=sys_.evolve_model, 
                                      args=(dt-offset,)))

    if self.use_threading:
      for x in threads: x.start()
      for x in threads: x.join()  #Run local integration scheme
    else:
      for x in threads: x.run()

  def correction_kicks(self, particles, subsystems, dt):
    if subsystems and len(particles)>1:
      parent, system = zip(*subsystems.items())
      corr_par = np.asarray([CorrForCompoundParti(particles, par, self.sys_kickers) for par in parent])
      for subsyst, corr in zip(system, corr_par):
        parts = subsyst.copy_to_memory()
        gravity = corr.get_gravity_at_point(parts.radius, 
                            parts.x, parts.y, parts.z
                            )
        parts.vx+=dt*gravity[0]
        parts.vy+=dt*gravity[1]
        parts.vz+=dt*gravity[2]
        channel = parts.new_channel_to(subsyst)
        channel.copy_attributes(["vx","vy","vz"])

  def kick_codes(self,dt):
    if (self.dE_track):
      E0 = self.energy_track()
    subsystems = self.particles.collection_attributes.subsystems
    self.correction_kicks(self.particles, subsystems, dt)

    self.particles.recenter_subsystems()
    if (self.dE_track):
      E1 = self.energy_track()
      self.dEa+=(E1-E0)

  def child_energy_calc(self):
    E = 0 | units.J
    for child_ in self.subcodes.values():
      E+=child_.kinetic_energy + child_.potential_energy
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
    corrector = CorrForCompoundParti(self.particles, None, self.sys_kickers)
    for parent, code in self.subcodes.items():
      Ep+=code.potential_energy
      if len(self.particles)>1:
        corrector.parent=parent
        Ep+=potential_energy(code, corrector.get_potential_at_point)
    return Ep

  @property
  def kinetic_energy(self):  
    Ek = self.parent_code.kinetic_energy
    for code in self.subcodes.values():
      Ek+=code.kinetic_energy
    return Ek

  @property
  def model_time(self):  
    return self.parent_code.model_time
