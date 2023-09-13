import threading

from amuse.community.hermite0.interface import Hermite
from amuse.community.huayno.interface import Huayno
from amuse.community.mercury.interface import Mercury
from amuse.community.twobody.twobody import TwoBody
from amuse.couple.bridge import CalculateFieldForParticles
from amuse.ext.basicgraph import UnionFind
from amuse.units import units, constants

from grav_correctors import *
from hierarchical_particles import *
from particle_initialiser import *

def potential_energy(system, get_potential):
  parts=system.particles.copy()
  pot=get_potential(parts.radius,parts.x,parts.y,parts.z)
  return (pot*parts.mass).sum()/2 

class Nemesis(object):
  def __init__(self, par_conv, child_conv, dt, eta):
    """Class setting up the simulation, checking for dissolution of systems and evolving system.
    Inputs:
    par_conv:   Parent system N-body converter
    child_conv: Children N-body converter
    dt:         Simulation time step
    eta:        Parameter influencing parent radius"""

    self.parent_code = self.parent_worker(par_conv)
    self.subsys_code = self.sub_worker
    self.bridge_code = self.py_worker

    self.child_conv = child_conv
    self.par_conv = par_conv
    self.dt = dt
    self.eta = eta
    self.prad_model = None
    
    self.particles = HierarchicalParticles(self.parent_code.particles)
    self.timestep = None
    self.subcodes = dict()
    self.time_offsets = dict()
    self.split_treshold = None
    self.use_threading = True

  def parent_worker(self, par_conv):
      """Defining the global integrator"""
      code = Hermite(par_conv)
      code.parameters.epsilon_squared = 0.| units.AU**2
      code.parameters.end_time_accuracy_factor=0.
      code.parameters.dt_param=0.001
      return code

  def py_worker(self):
      """Defining the bridging mechanism"""
      code = CalculateFieldForParticles(gravity_constant = constants.G)
      return code
      
  def sub_worker(self, cset, child_conv):
      """Defining the local integrator"""
      mode=self.system_type(cset)
      if mode=="twobody":
          code=TwoBody(child_conv)
      elif mode=="solarsystem":
          code=Mercury(child_conv)
      elif mode=="nbody":
          code=Huayno(child_conv)
          code.parameters.inttype_parameter=code.inttypes.SHARED4
      return code
  
  def system_type(self, cset):
    """Choosing relevant integrator for children systems"""
    
    if len(cset)==2:
      return "twobody"
    smass=sorted(cset.mass)
    if smass[-1]/smass[-2] > 100.:
      return "solarsystem"
    return "nbody" 

  def set_parent_particle_radius(self, pset):
    """Set parent system radius"""
    subsystems=self.particles.collection_attributes.subsystems
    parti_func = particle_init()
    if pset in subsystems:
      sys=subsystems[pset]
    else:
      sys=pset.as_set()
    if pset.sub_worker_radius == 0. | pset.radius.unit:
      pset.sub_worker_radius = pset.radius
      if self.radius == 0 | units.m:
        pset.radius = sys.virial_radius()
      else:
        if (1<2):
          pset.radius = parti_func.parent_radius(sys, self.dt, self.eta)
        else:
          pset.sub_worker_radius=pset.radius
          pset.radius = parti_func.parent_radius(sys, self.dt, self.eta)

  def commit_particles(self, child_conv):
    """Commit particle system"""
    self.particles.recenter_subsystems()
    
    if not hasattr(self.particles,"sub_worker_radius"):
      self.particles.sub_worker_radius=0. | self.particles.radius.unit
    for p in self.particles: #Give each parent system a radius
      self.set_parent_particle_radius(p)
      
    subsystems=self.particles.collection_attributes.subsystems
    subcodes=self.subcodes
    for parent in list(subcodes.keys()):
      if parent in subsystems:
         if subsystems[parent] is subcodes[parent].particles:
           continue
      code = subcodes.pop(parent)
      self.time_offsets.pop(code)
      del code
    for parent,sys in list(subsystems.items()):
      if parent not in subcodes:
        code = self.subsys_code(sys, child_conv)
        self.time_offsets[code]=(self.model_time-code.model_time)
        code.particles.add_particles(sys)
        subsystems[parent]=code.particles
        subcodes[parent]=code
        
  def recommit_particles(self):
    self.commit_particles()

  def evolve_model(self, tend, timestep=None):
    """Bulk of the simulation - Evolve the model until tend"""
    if timestep is None:
      timestep = self.timestep
    if timestep is None:
      timestep = tend-self.model_time  
    while self.model_time < (tend-timestep/2.):    
      self.kick_codes(timestep/2.)
      self.drift_codes(self.model_time+timestep, self.model_time+timestep/2)
      self.kick_codes(timestep/2.)
      self.split_subcodes()

  def split_subcodes(self):
    subsystems=self.particles.collection_attributes.subsystems
    subcodes=self.subcodes
    for parent, subsys in list(subsystems.items()): #Iterate through all subsystems
      radius=parent.radius
      components=subsys.connected_components(threshold=1.75*radius)
      if len(components)>1:                    #Checking for dissolution of system
        parentposition=parent.position
        parentvelocity=parent.velocity
        self.particles.remove_particle(parent) #Need new def. of parent, so remove it
        code=subcodes.pop(parent)              #Extract and remove integration scheme used till now for dissolving system
        offset=self.time_offsets.pop(code)
        for c in components:
          sys=c.copy_to_memory()
          sys.position+=parentposition         #Shift children to ghost of parent
          sys.velocity+=parentvelocity
          if len(sys)>1:                       #If system > 1 make a subsystem
            newcode=self.subsys_code(sys)  #Initialise new code for subsystem
            self.time_offsets[newcode]=(self.model_time-newcode.model_time)
            newcode.particles.add_particles(sys)
            newparent=self.particles.add_subsystem(newcode.particles) #Make a parent particle and add to global
            subcodes[newparent]=newcode
          else:                                #If system = 1 make it a parent
            newparent=self.particles.add_subsystem(sys)
          self.set_parent_particle_radius(newparent)
        del code  
      
  def handle_collision(self, coll_time, corr_time, coll_set):
    """
    Inputs:
    coll_time:   Time of collision
    corr_time:   Time to correct integration after collision occurs
    coll_set:    Colliding particle set
    """
    
    subsystems=self.particles.collection_attributes.subsystems
    collsubset = Particles(2)
    collsubsystems = dict()
    for parti_ in coll_set:
      collsubset.add_particle(parti_)
      if parti_ in self.subcodes:
        code=self.subcodes[parti_]
        offset=self.time_offsets[code]
        code.evolve_model(coll_time-offset)
      if parti_ in subsystems:
        collsubsystems[parti_]=subsystems[parti_]

    self.correction_kicks(collsubset,collsubsystems,coll_time-corr_time)
    newparts=HierarchicalParticles(Particles())
    for parti_ in coll_set:
      parti_=parti_.as_particle_in_set(self.particles)
      if parti_ in self.subcodes: #If collider part of subsystem, add to subsystem and
        code = self.subcodes.pop(parti_)
        offset = self.time_offsets.pop(code)
        parts = code.particles.copy_to_memory()
        parts.position += parti_.position
        parts.velocity += parti_.velocity
        newparts.add_particles(parts)
        del code
      else:
        np = newparts.add_particle(parti_)
        np.radius = parti_.sub_worker_radius  
      self.particles.remove_particle(parti_)
    newcode = self.subsys_code(newparts, self.child_conv)
    self.time_offsets[newcode] = (self.model_time-newcode.model_time)
    newcode.particles.add_particles(newparts)
    newparent = self.particles.add_subsystem(newcode.particles)
    self.set_parent_particle_radius(newparent)
    self.subcodes[newparent]=newcode
    
  def find_coll_sets(self,p1,p2):
    coll_sets=UnionFind()
    for p,q in zip(p1,p2):
      print("find_coll_sets:", p, q)
      coll_sets.union(p,q)
    return coll_sets.sets()
    
  def drift_codes(self, dt, corr_time):
    """Evolve parent system and then children system for dt."""
    code=self.parent_code #Calling Hermite
    stopping_condition = code.stopping_conditions.collision_detection
    stopping_condition.enable()
    while code.model_time < dt*(1-1.e-12): #Global integrator
      code.evolve_model(dt)
      if stopping_condition.is_set():
        print("...Collision Detected...")
        coll_time=code.model_time
        print("coll_time:", coll_time.in_(units.yr), dt)
        coll_sets=self.find_coll_sets(stopping_condition.particles(0), stopping_condition.particles(1))
        print("collsets:",len(coll_sets))
        for cs in coll_sets:
          self.handle_collision(coll_time, corr_time, cs)

    threads=[]
    for x in list(self.subcodes.values()):
      offset=self.time_offsets[x]
      if offset>dt:
        print("curious?")
      threads.append(threading.Thread(target=x.evolve_model, args=(dt-offset,)) )
    if self.use_threading:
      for x in threads: x.start()
      for x in threads: x.join()  #Run local integration scheme
    else:
      for x in threads: x.run()

  def correction_kicks(self, particles, subsystems, dt):
    if subsystems and len(particles)>1:
      corrector=correction_from_compound_particles(particles,subsystems,self.bridge_code)
      self.kick_particles(particles,corrector.get_gravity_at_point, dt)

      corrector=correction_for_compound_particles(particles, None, self.bridge_code)
      for parent, subsys in list(subsystems.items()):
        corrector.parent=parent
        self.kick_particles(subsys, corrector.get_gravity_at_point, dt)

  def kick_codes(self,dt):
    self.correction_kicks(self.particles,self.particles.collection_attributes.subsystems,dt)
    self.particles.recenter_subsystems()

  def kick_particles(self, particles, get_gravity, dt):
    """Function to kick the particles"""
    parts=particles.copy_to_memory()
    ax,ay,az=get_gravity(parts.radius,parts.x,parts.y,parts.z)
    parts.vx=parts.vx+dt*ax
    parts.vy=parts.vy+dt*ay   
    parts.vz=parts.vz+dt*az
    channel = parts.new_channel_to(particles)
    channel.copy_attributes(["vx","vy","vz"])

  def get_potential_at_point(self,radius,x,y,z):
    phi=self.parent_code.get_potential_at_point(radius,x,y,z)
    return phi

  def get_gravity_at_point(self,radius,x,y,z):
    ax,ay,az=self.parent_code.get_gravity_at_point(radius,x,y,z)
    return ax,ay,az
  
  @property
  def potential_energy(self):
    Ep=self.parent_code.potential_energy
    if len(self.particles)>1:
      corrector=correction_from_compound_particles(self.particles,
                self.particles.collection_attributes.subsystems,self.bridge_code)
      Ep+=potential_energy(self.parent_code, corrector.get_potential_at_point) 
    corrector=correction_for_compound_particles(self.particles, None, self.bridge_code)
    for parent,code in list(self.subcodes.items()):
      Ep+=code.potential_energy
      if len(self.particles)>1:
        corrector.parent=parent
        Ep+=potential_energy(code,corrector.get_potential_at_point)
    return Ep

  @property
  def kinetic_energy(self):  
    Ek=self.parent_code.kinetic_energy
    for code in list(self.subcodes.values()):
      Ek+=code.kinetic_energy
    return Ek

  @property
  def model_time(self):  
    return self.parent_code.model_time