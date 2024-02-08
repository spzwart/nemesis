import numpy as np
import threading

from amuse.community.hermite0.interface import Hermite
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
from amuse.units import units, constants, nbody_system

from src.grav_correctors import CorrForCompoundParti, CorrFromCompoundParti
from src.hierarchical_particles import HierarchicalParticles


def potential_energy(system, get_potential):
  parts = system.particles.copy()
  pot = get_potential(parts.radius, parts.x, parts.y, parts.z)
  return (pot*parts.mass).sum()/2 


class Nemesis(object):
  def __init__(self, par_conv, child_conv, 
               dt, code_dt, gal_field):
    """Class setting up the simulation, checking 
       for dissolution of systems and evolving system.
       
       Inputs:
       par_conv:   Parent system N-body converter
       child_conv: Children N-body converter
       dt:         Simulation time step
       code_dt:    Time-step parameter for codes
       gal_field:  Flag turning on/off galactic field
    """

    self.code_timestep = code_dt
    self.stellar_code = self.stellar_worker()
    self.parent_code = self.parent_worker(par_conv)
    self.subsys_code = self.sub_worker
    self.sys_kickers = self.py_worker
    self.grav_bridge = None

    self.child_conv = child_conv
    self.par_conv = par_conv
    self.dt = dt
    
    self.event_keys = [ ]
    self.ejec_keys = [ ]
    
    self.par_particles = None
    self.particles = HierarchicalParticles(self.parent_code.particles)
    self.timestep = None
    self.subcodes = dict()
    self.time_offsets = dict()
    self.use_threading = True
    self.gal_field = gal_field

    self.psetup = ParticleInit()

  def commit_particles(self, child_conv):
    """Commit particle system"""

    subsystems = self.particles.collection_attributes.subsystems
    subcodes = self.subcodes

    self.particles.recenter_subsystems()
    if not hasattr(self.particles, "sub_worker_radius"):
      self.particles.sub_worker_radius = 0. | self.particles.radius.unit
    self.init_parent_radius()
    
    for parent in subcodes.keys():
      if ((parent in subsystems) and \
          (subsystems[parent] is subcodes[parent].particles)):
        continue
      code = subcodes.pop(parent)
      self.time_offsets.pop(code)
      del code

    for parent, sys in subsystems.items():
      if parent not in subcodes:
        gravity_code = self.subsys_code(sys, child_conv)
        self.time_offsets[gravity_code] = (self.model_time-gravity_code.model_time)
        gravity_code.particles.add_particles(sys)
        subsystems[parent] = sys
        subcodes[parent] = gravity_code
    
    stellar_code = self.stellar_code
    parti = self.particles.all().copy_to_memory()
    self.stars = parti[parti.type=="STAR"]
    self.stars.sub_worker_radius = self.stars.radius
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
      parent.radius = self.psetup.parent_radius(mass, self.dt)
    isolated = self.particles[self.particles.radius==(0|units.m)]
    isolated.radius = self.psetup.parent_radius(isolated.mass, self.dt)

  def channel_makers(self):
    """Function to make various channels:
       Global integrator ---> Local parent particle set
       Stellar code --------> Stellar particle set
       Stellar code --------> Global integrator
       Stellar code --------> Local parent particle set
       Stellar code --------> Children integrator
       Children integrator -> Local complete particle set
    """
    
    par_code = self.parent_code
    str_code = self.stellar_code

    self.chnl_from_gc_to_gp = par_code.particles.new_channel_to(self.particles)
    self.chnl_from_sc_to_sp = str_code.particles.new_channel_to(self.stars,
                                  attributes = ["mass", "radius"],
                                  target_names = ["mass", "sub_worker_radius"])
    self.chnl_sc_to_gcp = str_code.particles.new_channel_to(par_code.particles,
                                  attributes = ["mass",], target_names = ["mass"])
    
    self.chnl_from_sc_to_ccp = [ ]
    self.chnl_from_cc_to_all = [ ]
    self.subsys_keys = [ ]
    for sys_ in self.subcodes.values():
      sys_parti = sys_.particles
      most_massive = (sys_parti.mass).argmax()
      self.subsys_keys.append(sys_parti[most_massive].key)
      
      chnl_temp = str_code.particles.new_channel_to(sys_.particles,
                      attributes = ["radius", "mass"],
                      target_names = ["radius", "mass"])
      self.chnl_from_sc_to_ccp.append(chnl_temp)

      chnl_temp = sys_.particles.new_channel_to(self.particles.all(),
                        attributes = ["radius", "mass"],
                        target_names = ["sub_worker_radius", "mass"])
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

      code = Hermite(par_conv)
      code.parameters.epsilon_squared = 0. | units.au**2
      code.parameters.end_time_accuracy_factor = 0.
      code.parameters.dt_param = self.code_timestep
      return code

  def py_worker(self):
      """Defining the bridging mechanism"""
      return CalculateFieldForParticles(gravity_constant = constants.G)
      
  def sub_worker(self, cset, child_conv):
      """Defining the local integrator based on system population"""
      
      if len(cset)==2:
        child_conv = nbody_system.nbody_to_si(np.sum(cset.mass), 
                                              cset.virial_radius())
        code = SmallN(child_conv)
        code.parameters.timestep_parameter = self.code_timestep
        return code

      smass = sorted(cset.mass)
      if smass[-1]/smass[-2] > 100:
        return Mercury(child_conv)
        
      code = Huayno(child_conv)
      code.parameters.inttype_parameter=code.inttypes.SHARED4
      code.parameters.timestep_parameter = self.code_timestep
      return code

  def evolve_model(self, tend, timestep=None):
    """Bulk of the simulation - Evolve the model until tend"""

    if timestep is None:
      timestep = tend-min(self.model_time, self.stellar_code.model_time)

    while self.model_time < (tend-timestep/2.): 
      self.dEa = 0 | units.J
      self.save_snap = False

      self.stellar_evolution(self.model_time+timestep/2.)
      self.star_channel_copier()
      print(np.sort(self.particles.all().mass.in_(units.MSun))[-2:])
      self.kick_codes(timestep/2.)
      self.drift_global(self.model_time+timestep, 
                        self.model_time+timestep/2)
      self.drift_child(self.model_time+timestep)
      self.kick_codes(timestep/2.)
      self.split_subcodes()
      self.stellar_evolution(self.model_time+timestep/2.)
      self.star_channel_copier()
      self.chnl_from_sc_to_sp.copy()

  def energy_corr(self):
    """Function tracking energy budget during events"""
    
    p = self.particles.all()
    Eall = p.kinetic_energy()+p.potential_energy()
    return Eall

  def split_subcodes(self):
    """Function tracking the dissolution of a parent system"""

    subsystems=self.particles.collection_attributes.subsystems
    subcodes=self.subcodes

    for parent,subsys in list(subsystems.items()):
      radius=parent.radius
      components=subsys.connected_components(threshold=1.75*radius)
      if len(components) > 1:                  #Checking for dissolution of system
        print("...Splitting subcode...")
        self.save_snap = True
        self.ftype = "parent_dissolve"
        code = subcodes.pop(parent)            #Extract and remove integration scheme used till now for dissolving system
        offset = self.time_offsets.pop(code)
        
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

            chnl_temp = self.stellar_code.particles.new_channel_to(newcode.particles,
                                                   attributes = ["radius", "mass"],
                                                   target_names = ["radius", "mass"])
            self.chnl_from_sc_to_ccp.append(chnl_temp)
            chnl_temp = newcode.particles.new_channel_to(self.particles.all(),
                            attributes = ["radius", "mass"],
                            target_names = ["sub_worker_radius", "mass"])
            self.chnl_from_cc_to_all.append(chnl_temp)

            max_mass_old = subsys[subsys.mass==subsys.mass.max()]
            idx = self.subsys_keys.index(max_mass_old.key)
            del self.chnl_from_sc_to_ccp[idx]
            del self.chnl_from_cc_to_all[idx]
            del self.subsys_keys[idx]

            max_mass_new = sys[sys.mass==sys.mass.max()]
            self.subsys_keys.append(max_mass_new.key)
            newparent.radius = self.psetup.parent_radius(np.sum(sys.mass), self.dt)

          else:                              
            newparent = self.particles.add_subsystem(sys)
            newparent.radius = self.psetup.parent_radius(newparent.mass, self.dt)

        self.particles.remove_particle(parent)   #New parent systems
        self.particles.recenter_subsystems()
        
  def handle_collision(self, coll_time, corr_time, coll_set):
    """
    Inputs:
    coll_time:   Time of collision
    corr_time:   Time to correct integration after collision occurs
    coll_set:    Colliding particle set
    """
    
    self.save_snap = True
    self.ftype = "parent_merger"

    par = self.particles.copy_to_memory()
    subsystems = par.collection_attributes.subsystems
    collsubset, collsubsystems = self.evolve_coll_offset(coll_set, subsystems, coll_time)
    self.correction_kicks(collsubset, collsubsystems, coll_time-corr_time)
    E0a = self.energy_corr()
  
    newparts = HierarchicalParticles(Particles())
    for parti_ in coll_set:
      parti_ = parti_.as_particle_in_set(self.particles)
      if parti_ in self.subcodes:  #If collider part of subsystem, add to subsystem
        code = self.subcodes.pop(parti_)
        offset = self.time_offsets.pop(code)
        parts = code.particles.copy_to_memory()
        parts.position+=parti_.position
        parts.velocity+=parti_.velocity
        parts.type = parti_.type
        parts.disk_key = parti_.disk_key
        newparts.add_particles(parts)
        del code

        max_mass = parts[parts.mass==parts.mass.max()]
        idx = self.subsys_keys.index(max_mass.key)
        del self.chnl_from_sc_to_ccp[idx]
        del self.chnl_from_cc_to_all[idx]
        del self.subsys_keys[idx]

      else:  #Loop for two parent particle collisions
        new_parti = newparts.add_particle(parti_)
        new_parti.radius = parti_.sub_worker_radius
        new_parti.disk_key = parti_.disk_key
      
      self.particles.remove_particle(parti_)
      self.particles.synchronize_to(self.parent_code.particles)

    newcode = self.subsys_code(newparts, self.child_conv)
    self.time_offsets[newcode] = (self.model_time-newcode.model_time)
    newcode.particles.add_particles(newparts)
    
    newparent = self.particles.add_subsystem(newparts)
    newparent.position += newparts.center_of_mass()
    newparent.velocity += newparts.center_of_mass_velocity()
    newparts.move_to_center()

    newparent.radius = self.psetup.parent_radius(np.sum(newparts.mass), self.dt)
    if len(newparts[newparts.disk_key==-1])==len(newparts):
      newparent.disk_key =- 1
    else:
      newparent.disk_key = max(par.disk_key)+1

    if "STAR" in newparts.type:
      newparent.type = "STAR"
    else:
      newparent.type = newparts[newparts.mass==newparts.mass.max()].type
    self.subcodes[newparent]=newcode
    
    E1a = self.energy_corr()
    self.dEa+=(E1a-E0a)

    chnl_temp = self.stellar_code.particles.new_channel_to(newcode.particles,
                                          attributes = ["radius", "mass"],
                                          target_names = ["radius", "mass"])
    self.chnl_from_sc_to_ccp.append(chnl_temp)
    chnl_temp = newcode.particles.new_channel_to(self.particles.all(),
                        attributes = ["radius", "mass"],
                        target_names = ["sub_worker_radius", "mass"])
    self.chnl_from_cc_to_all.append(chnl_temp)
    max_mass = newcode.particles[newcode.particles.mass==newcode.particles.mass.max()]
    self.subsys_keys.append(max_mass.key[0])

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

      self.event_type.append("Supernova")
      self.event_time.append(time.value_in(units.yr))
      self.event_keys.append(SN_detect.particles(0).key)

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
        for cs in coll_sets:
          self.handle_collision(coll_time, corr_time, cs)

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
      corrector = CorrFromCompoundParti(particles, subsystems,
                                        self.sys_kickers)
      self.kick_particles(particles,corrector.get_gravity_at_point, dt)

      corrector = CorrForCompoundParti(particles, None, self.sys_kickers)
      for parent, subsys in subsystems.items():
        corrector.parent=parent
        self.kick_particles(subsys, corrector.get_gravity_at_point, dt)

  def kick_codes(self,dt):
    import time as cpu_time

    time1 = cpu_time.time()
    #E0a = self.energy_corr()
    self.correction_kicks(self.particles, 
                          self.particles.collection_attributes.subsystems, 
                          dt)
    self.particles.recenter_subsystems()
    #E1a = self.energy_corr()
    #self.dEa+=(E1a-E0a)
    time2 = cpu_time.time()
    print(time2-time1)
    #STOP

  def kick_particles(self, particles, get_gravity, dt):
    """Function to kick the particles"""
    parts = particles.copy_to_memory()
    ax,ay,az = get_gravity(parts.radius, parts.x, parts.y, parts.z)
    parts.vx = parts.vx+dt*ax
    parts.vy = parts.vy+dt*ay   
    parts.vz = parts.vz+dt*az
    channel = parts.new_channel_to(particles)
    channel.copy_attributes(["vx","vy","vz"])

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
    if len(self.particles)>1:
      corrector = CorrFromCompoundParti(self.particles,
                      self.particles.collection_attributes.subsystems,
                      self.sys_kickers)
      Ep+=potential_energy(self.parent_code, corrector.get_potential_at_point) 
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