import numpy as np
import os
import threading
import time as cpu_time

#from amuse.community.hermite_grx.interface import HermiteGRX
from amuse.community.huayno.interface import Huayno
from amuse.community.mercury.interface import Mercury
from amuse.community.ph4.interface import ph4
from amuse.community.seba.interface import SeBa
from amuse.community.smalln.interface import SmallN
from amuse.couple import bridge
from amuse.couple.bridge import CalculateFieldForParticles
from amuse.datamodel import Particles
from amuse.ext.basicgraph import UnionFind
from amuse.ext.composition_methods import SPLIT_4TH_S_M4
from amuse.ext.galactic_potentials import MWpotentialBovy2015
from amuse.lab import write_set_to_file
from amuse.units import units, constants

from src.environment_functions import EnvironmentFunctions
from src.grav_correctors import CorrForCompoundParti
from src.hierarchical_particles import HierarchicalParticles


def potential_energy(system, get_potential):
  parts = system.particles.copy()
  pot = get_potential(parts.radius, parts.x, parts.y, parts.z)
  return (pot*parts.mass).sum()/2 


class Nemesis(object):
  def __init__(self, par_conv, chd_conv, dt, 
               code_dt, par_nworker=1, chd_nworker=1,
               dE_track=False, star_evol=False, 
               gal_field=False):
    """Class setting up the simulation, checking 
       for dissolution of systems and evolving system.
       
       Inputs:
       par_conv:     Parent system N-body converter
       chd_conv:     Children N-body converter
       dt:           Diagnostic time step
       code_dt:      Internal time step
       par_nworker:  Number of workers for global integrator
       chd_nworker:  Number of workers for children integrator
       dE_track:     Track energy changes
       star_evol:    Flag turning on/off stellar evolution
       gal_field:    Flag turning on/off galactic field
    """

    self.code_timestep = code_dt
    self.par_nworker = par_nworker
    self.chd_nworker = chd_nworker
    self.star_evol = star_evol
    
    self.parent_code = self.parent_worker(par_conv)
    self.subsys_code = self.sub_worker
    self.sys_kickers = self.py_worker
    if (self.star_evol):
      self.stellar_code = self.stellar_worker()

    self.dE_track = dE_track
    self.grav_bridge = None

    self.chd_conv = chd_conv
    self.par_conv = par_conv
    self.dt = dt
    
    self.event_key = [ ]
    self.event_time = [ ]
    self.event_type = [ ]
    
    self.rmax = None
    self.par_particles = None
    self.particles = HierarchicalParticles(self.parent_code.particles)
    self.timestep = None
    self.subcodes = dict()
    self.time_offsets = dict()
    self.use_threading = True
    self.gal_field = gal_field

    self.env_setup = EnvironmentFunctions()

  def commit_particles(self, chd_conv):
    """Commit particle system"""

    subsystems = self.particles.collection_attributes.subsystems
    subcodes = self.subcodes
    self.particles.recenter_subsystems()

    length_unit = self.particles.radius.unit
    if not hasattr(self.particles, "sub_worker_radius"):
      self.particles.sub_worker_radius = 0. | length_unit

    for parent, code in list(subcodes.items()):
      if ((parent in subsystems) and \
          (subsystems[parent] is subcodes[parent].particles)):
        continue
      self.time_offsets.pop(code)
      del code

    for parent, sys in subsystems.items():
      parent.radius = self.env_setup.parent_radius(np.sum(sys.mass), self.dt)
      if parent not in subcodes:
        gravity_code = self.subsys_code(sys, chd_conv)
        self.time_offsets[gravity_code] = (self.model_time-gravity_code.model_time)
        subsystems[parent] = sys
        subcodes[parent] = gravity_code

    if (self.star_evol):
      parti = self.particles.all()
      self.stars = parti[parti.type=="STAR"]
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

  def channel_makers(self):
    """Global integrator to local parent particle set channel"""
    self.global_code_to_parents = self.parent_code.particles.new_channel_to(self.particles) 

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
      
  def sub_worker(self, cset, chd_conv):
      """Defining the local integrator based on system population"""
      if len(cset)==2:
         code = SmallN(chd_conv)
         code.particles.add_particles(cset)
         return code

      no_stars = len(cset[cset.type=="STAR"])+len(cset[cset.type=="BINARY STAR"])
      if no_stars==1:
         code = Mercury(chd_conv)
         code.particles.add_particles(cset)
         return code

      code = ph4(chd_conv)
      #code.parameters.inttype_parameter=code.inttypes.SHARED4
      code.particles.add_particles(cset)
      return code 

  def grav_channel_copier(self):
    self.global_code_to_parents.copy_attributes(["mass", "vx", "vy", "vz", "x", "y", "z"])
    subsystems = self.particles.collection_attributes.subsystems
    for parent, code in self.subcodes.items():
       children = subsystems[parent]
       channel = code.particles.new_channel_to(children)
       channel.copy_attributes(["mass", "vx", "vy", "vz", "x", "y", "z"])

  def star_channel_copier(self):
    stars = self.stellar_code.particles
    stars.new_channel_to(self.parent_code.particles).copy_attributes(["mass"])
    subsystems = self.particles.collection_attributes.subsystems
    for parent, code in self.subcodes.items():
       children = subsystems[parent]
       channel = stars.new_channel_to(children)
       channel.copy_attributes(["mass"])

  def evolve_model(self, tend, timestep=None):
    """Evolve the different integrators"""

    if timestep is None:
      timestep = tend-self.model_time
    
    while self.model_time < (tend-timestep/2.):
      self.dEa = 0 | units.J
      self.save_snap = False
      t2 = cpu_time.time()
      if (self.star_evol):
        self.stellar_evolution(self.model_time+timestep/2.)
        self.star_channel_copier()
        self.particles.all()
        t1 = cpu_time.time()
        print("Time taken for Star Evol. : ", t1-t2)
      self.kick_codes(timestep/2.)
      t2 = cpu_time.time()
      print("Time taken for Kicking: ", t2-t1)
      self.drift_global(self.model_time+timestep, 
                        self.model_time+timestep/2.)
      t1 = cpu_time.time()
      print("Time taken for Global", t1-t2)
      self.drift_child(self.model_time+timestep)
      t2 = cpu_time.time()
      print("Time taken for Local", t2-t1)
      self.kick_codes(timestep/2.)
      t1 = cpu_time.time()
      print("Time taken for Kicking: ", t1-t2)
      self.split_subcodes()
      t1 = cpu_time.time()
      print("Time taken for Splitting: ", t2-t1)
      if (self.star_evol):
        time = self.stellar_code.model_time
        self.stellar_evolution(self.model_time+timestep/2.)
        self.star_channel_copier()
      self.grav_channel_copier()

  def energy_track(self):
    """Extract energy of all particles"""
    p = self.particles.all()
    Eall = p.kinetic_energy()+p.potential_energy()
    return Eall

  def ejection_checker(self):
    allparts = self.particles.all()
    ejec_prosp = allparts[allparts.Nej==0]
    SMBH = ejec_prosp[ejec_prosp.type=="smbh"]
    ejec_prosp -= SMBH

    dv_vect = (ejec_prosp.velocity-SMBH.velocity)
    dv = dv_vect.lengths()
    dr_vect = (ejec_prosp.position-SMBH.position)
    dr = dr_vect.lengths()
    
    trajectory = (dv_vect*dr_vect).lengths()/(dv*dr)
    pKE = 0.5*ejec_prosp.mass*dv**2
    pPE = constants.G*SMBH.mass*ejec_prosp.mass/dr

    ejection = ejec_prosp[(pKE>abs(pPE)) & (trajectory>0) & (dr>self.cluster_dist)]
    ejection.Nej = 1
    if len(ejection)>0:
      self.save_snap = True
      self.Nej = True
      self.event_key = np.concatenate((self.event_key, ejection.key), axis=None)
      self.event_time = np.concatenate((self.event_time, self.model_time), axis=None)
      self.event_type = np.concatenate((self.event_type, "Ejection"), axis=None)

    ejec_prosp -= ejection
    drifters = ejec_prosp[(dr>3*self.cluster_dist)]
    drifters.Nej = 1
    if len(drifters)>0:
      self.Nej = True
      self.event_key = np.concatenate((self.event_key, drifters.key), axis=None)
      self.event_time = np.concatenate((self.event_time, self.model_time), axis=None)
      self.event_type = np.concatenate((self.event_type, "Drifter"), axis=None)

    if len(drifters)>0 or len(ejection)>0:
      Nejec = np.sum(self.particles.all().Nej)
      write_set_to_file(self.particles.all().savepoint(0|units.Myr), 
                        os.path.join(self.ejec_dir, "ejec"+str()), 
                        'amuse', close_file=True, overwrite_file=False)

  def split_subcodes(self):
    """Function tracking the dissolution of a parent system"""
    subsystems=self.particles.collection_attributes.subsystems
    subcodes=self.subcodes
    for parent,subsys in list(subsystems.items()):
       radius=parent.radius
       components=subsys.connected_components(threshold=1.5*radius)
       if len(components)>1:                  #Checking for dissolution of system
          print("...Splitting subcode...")
          self.save_snap = True
          code = subcodes.pop(parent)            #Extract and remove dissolved system integrator
          offset = self.time_offsets.pop(code)
          
          keys = [ ]
          for c in components:
             sys = c.copy_to_memory()
             sys.position+=parent.position 
             sys.velocity+=parent.velocity
             if len(sys)>1:                       #If system > 1 make a subsystem
                newcode = self.subsys_code(sys, self.chd_conv)
                self.time_offsets[newcode] = (self.model_time - newcode.model_time)
                newcode.particles.add_particles(sys)
                newparent = self.particles.add_subsystem(sys) #Make a parent particle and add to global
                subcodes[newparent] = newcode

                newparent.radius = self.env_setup.parent_radius(np.sum(sys.mass), self.dt)
                keys = np.concatenate((keys, sys.key), axis=None)
             else:                              
                newparent = self.particles.add_subsystem(sys)
                newparent.radius = self.env_setup.parent_radius(newparent.mass, self.dt)
                keys = np.concatenate((keys, sys.key), axis=None)

          self.event_key = np.concatenate((self.event_key, keys), axis=None)
          self.event_time = np.concatenate((self.event_time, self.model_time), axis=None)
          self.event_type = np.concatenate((self.event_type, "Parent Dissolve"), axis=None)

          del code
          self.particles.remove_particle(parent)   #New parent systems

  def parent_merger(self, coll_time, corr_time, coll_set):
    """
    Inputs:
    coll_time:   Time of collision
    corr_time:   Time to correct integration after collision occurs
    coll_set:    Colliding particle set
    """
    
    self.save_snap = True

    par = self.particles.copy_to_memory()
    subsystems = par.collection_attributes.subsystems
    collsubset, collsubsystems = self.evolve_coll_offset(coll_set, subsystems, coll_time)
    self.correction_kicks(collsubset, collsubsystems, coll_time-corr_time)

    keys = [ ]
    newparts = HierarchicalParticles(Particles())
    for parti_ in coll_set:
      parti_ = parti_.as_particle_in_set(self.particles)
      if parti_ in self.subcodes:  #If collider part of subsystem, add to subsystem
        code = self.subcodes.pop(parti_)
        offset = self.time_offsets.pop(code)
        parts = code.particles.copy_to_memory()
        chd = self.particles.collection_attributes.subsystems[parti_]
        
        parts.position+=parti_.position
        parts.velocity+=parti_.velocity
        parts.syst_id = parti_.syst_id
        parts.type = chd.type
        newparts.add_particles(parts)

        keys = np.concatenate((keys, parts.key), axis=None)
      else:  #Loop for two parent particle collisions
        new_parti = newparts.add_particle(parti_)
        new_parti.radius = parti_.sub_worker_radius
        new_parti.syst_id = parti_.syst_id

        keys = np.concatenate((keys, parti_.key), axis=None)

      self.particles.remove_particle(parti_)
      self.particles.synchronize_to(self.parent_code.particles)
    
    newcode = self.subsys_code(newparts, self.chd_conv)
    self.time_offsets[newcode] = (self.model_time-newcode.model_time)
    newparent = self.particles.add_subsystem(newparts)
    self.subcodes[newparent]=newcode
    
    most_massive_idx = newparts.mass.argmax()
    newparent.type = newparts[most_massive_idx].type

    newparent.radius = self.env_setup.parent_radius(np.sum(newparts.mass), self.dt)
    if len(newparts[newparts.syst_id<=0])==len(newparts):
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
          self.parent_merger(coll_time, corr_time, cs)
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
