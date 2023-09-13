import numpy
import threading

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot

from amuse.ext.solarsystem import new_solar_system
from amuse.datamodel import Particle,Particles,ParticlesOverlay
from amuse.units import units,constants,nbody_system
from amuse.couple.bridge import CalculateFieldForParticles

from amuse.datamodel import ParticlesSuperset

from amuse.ext.basicgraph import UnionFind

from amuse.community.twobody.twobody import TwoBody
from amuse.community.hermite0.interface import Hermite
from amuse.community.mercury.interface import Mercury
from amuse.community.ph4.interface import ph4
#from amuse.community.phiGRAPE.interface import PhiGRAPE
from amuse.community.huayno.interface import Huayno
from amuse.community.mi6.interface import MI6


import logging
#logging.basicConfig(level=logging.DEBUG)

def system_type(parts):
  if len(parts)==2:
    return "twobody"
  smass=sorted(parts.mass)
  if smass[-1]/smass[-2] > 100.:
    return "solarsystem"
  return "nbody"    

class correction_from_compound_particles(object):
  def __init__(self, system, subsystems,worker_code_factory):
    self.system=system
    self.subsystems=subsystems
    self.worker_code_factory=worker_code_factory
    
  def get_gravity_at_point(self,radius,x,y,z):
    particles=self.system.copy_to_memory()
    particles.ax=0. | (particles.vx.unit**2/particles.x.unit)
    particles.ay=0. | (particles.vx.unit**2/particles.x.unit)
    particles.az=0. | (particles.vx.unit**2/particles.x.unit)
    for parent,sys in list(self.subsystems.items()): 
      code=self.worker_code_factory()
      code.particles.add_particles(sys.copy_to_memory())
      code.particles.position+=parent.position
      code.particles.velocity+=parent.velocity
      parts=particles-parent
      ax,ay,az=code.get_gravity_at_point(0.*parts.radius,parts.x,parts.y,parts.z)
      parts.ax+=ax
      parts.ay+=ay
      parts.az+=az
      code=self.worker_code_factory()
      code.particles.add_particle(parent)
      ax,ay,az=code.get_gravity_at_point(0.*parts.radius,parts.x,parts.y,parts.z)
      parts.ax-=ax
      parts.ay-=ay
      parts.az-=az
    return particles.ax,particles.ay,particles.az

  def get_potential_at_point(self,radius,x,y,z):
    particles=self.system.copy_to_memory()
    particles.phi=0. | (particles.vx.unit**2)
    for parent,sys in list(self.subsystems.items()): 
      code=self.worker_code_factory()
      code.particles.add_particles(sys.copy())
      code.particles.position+=parent.position
      code.particles.velocity+=parent.velocity
      parts=particles-parent
      phi=code.get_potential_at_point(0.*parts.radius,parts.x,parts.y,parts.z)
      parts.phi+=phi
      code=self.worker_code_factory()
      code.particles.add_particle(parent)
      phi=code.get_potential_at_point(0.*parts.radius,parts.x,parts.y,parts.z)
      parts.phi-=phi
    return particles.phi
  
  
class correction_for_compound_particles(object):  
  def __init__(self,system, parent, worker_code_factory):
    self.system=system
    self.parent=parent
    self.worker_code_factory=worker_code_factory
  
  def get_gravity_at_point(self,radius,x,y,z):
    parent=self.parent
    parts=self.system - parent
    instance=self.worker_code_factory()
    instance.particles.add_particles(parts)
    ax,ay,az=instance.get_gravity_at_point(0.*radius,parent.x+x,parent.y+y,parent.z+z)
    _ax,_ay,_az=instance.get_gravity_at_point([0.*parent.radius],[parent.x],[parent.y],[parent.z])
    instance.cleanup_code()
    return (ax-_ax[0]),(ay-_ay[0]),(az-_az[0])

  def get_potential_at_point(self,radius,x,y,z):
    parent=self.parent
    parts=self.system - parent
    instance=self.worker_code_factory()
    instance.particles.add_particles(parts)
    phi=instance.get_potential_at_point(0.*radius,parent.x+x,parent.y+y,parent.z+z)
    _phi=instance.get_potential_at_point([0.*parent.radius],[parent.x],[parent.y],[parent.z])
    instance.cleanup_code()
    return (phi-_phi[0])

class HierarchicalParticles(ParticlesOverlay):
  def __init__(self, *args,**kwargs):
    ParticlesOverlay.__init__(self,*args,**kwargs)
    self.collection_attributes.subsystems=dict()
  def add_particles(self,parts):  
    _parts=ParticlesOverlay.add_particles(self,parts)
    if hasattr(parts.collection_attributes,"subsystems"):
      for parent,sys in list(parts.collection_attributes.subsystems.items()):
        self.collection_attributes.subsystems[parent.as_particle_in_set(self)]=sys
    return _parts
  def remove_particles(self,parts):
    for p in parts:
      self.collection_attributes.subsystems.pop(p, None)
    ParticlesOverlay.remove_particles(self,parts)
  def add_subsystem(self, sys, recenter=True):
    if len(sys)==1:
      return self.add_particles(sys)[0]
    p=Particle()
    self.assign_parent_attributes(sys, p, relative=False, recenter=recenter)
    parent=self.add_particle(p)
    self.collection_attributes.subsystems[parent]=sys
    return parent
  def assign_subsystem(self, sys, parent, relative=True, recenter=True):
    self.assign_parent_attributes(sys,parent,relative,recenter)
    self.collection_attributes.subsystems[parent]=sys
  def assign_parent_attributes(self,sys,parent, relative=True, recenter=True):
    parent.mass=sys.total_mass()
    if relative:
      pass
    else:
      parent.position=0.*sys[0].position
      parent.velocity=0.*sys[0].velocity
    if recenter:
      parent.position+=sys.center_of_mass()
      parent.velocity+=sys.center_of_mass_velocity()
      sys.move_to_center()
  def recenter_subsystems(self):
    for parent,sys in list(self.collection_attributes.subsystems.items()):
      parent.position+=sys.center_of_mass()
      parent.velocity+=sys.center_of_mass_velocity()
      sys.move_to_center()
  def all(self):
    parts=self.copy_to_memory()
    for parent,sys in list(self.collection_attributes.subsystems.items()):
      parts.remove_particle(parent)
      subsys=parts.add_particles(sys)
      subsys.position+=parent.position
      subsys.velocity+=parent.velocity
    return parts


            
def kick_system(system, get_gravity, dt):
  parts=system.particles.copy_to_memory()
  ax,ay,az=get_gravity(parts.radius,parts.x,parts.y,parts.z)
  parts.vx=parts.vx+dt*ax
  parts.vy=parts.vy+dt*ay
  parts.vz=parts.vz+dt*az
  channel = parts.new_channel_to(system.particles)
  channel.copy_attributes(["vx","vy","vz"])

def kick_particles(particles, get_gravity, dt):
  parts=particles.copy_to_memory()
  ax,ay,az=get_gravity(parts.radius,parts.x,parts.y,parts.z)
  parts.vx=parts.vx+dt*ax
  parts.vy=parts.vy+dt*ay
  parts.vz=parts.vz+dt*az
  channel = parts.new_channel_to(particles)
  channel.copy_attributes(["vx","vy","vz"])

def potential_energy(system, get_potential):
  parts=system.particles.copy()
  pot=get_potential(parts.radius,parts.x,parts.y,parts.z)
  return (pot*parts.mass).sum()/2 

def potential_energy_particles(particles, get_potential):
  parts=particles.copy()
  pot=get_potential(parts.radius,parts.x,parts.y,parts.z)
  return (pot*parts.mass).sum()/2 

class Nemesis(object):
  def __init__(self,parent_code_factory,subcode_factory, worker_code_factory):
    self.parent_code=parent_code_factory()
    self.subcode_factory=subcode_factory
    self.worker_code_factory=worker_code_factory
    self.particles=HierarchicalParticles(self.parent_code.particles)
    self.timestep=None
    self.subcodes=dict()
    self.time_offsets=dict()
    self.split_treshold=None
    self.use_threading=True
    self.radius=None

  def set_parent_particle_radius(self,p):
    subsystems=self.particles.collection_attributes.subsystems
    if p in subsystems:
      sys=subsystems[p]
    else:
      sys=p.as_set()
    if p.sub_worker_radius == 0. | p.radius.unit:
      p.sub_worker_radius=p.radius
      if self.radius is None:
        p.radius=sys.virial_radius()
      else:
        if callable(self.radius): 
          p.radius=self.radius(sys)
        else:
          p.radius=self.radius

  def commit_particles(self):
    self.particles.recenter_subsystems()
    
    if not hasattr(self.particles,"sub_worker_radius"):
      self.particles.sub_worker_radius=0. | self.particles.radius.unit
    for p in self.particles:
      self.set_parent_particle_radius(p)
      
    subsystems=self.particles.collection_attributes.subsystems
    subcodes=self.subcodes
    for parent in list(subcodes.keys()):
      if parent in subsystems:
         if subsystems[parent] is subcodes[parent].particles:
           continue
      code=subcodes.pop(parent)
      self.time_offsets.pop(code)
      del code
    for parent,sys in list(subsystems.items()):
      if parent not in subcodes:
        code=self.subcode_factory(sys)
        self.time_offsets[code]=(self.model_time-code.model_time)
        code.particles.add_particles(sys)
        subsystems[parent]=code.particles
        subcodes[parent]=code
        
  def recommit_particles(self):
    self.commit_particles()

  def commit_parameters(self):
    pass

  def evolve_model(self, tend, timestep=None):
    if timestep is None:
      timestep = self.timestep
    if timestep is None:
      timestep = tend-self.model_time  
    while self.model_time < (tend-timestep/2.):    
      self.kick_codes(timestep/2.)
      self.drift_codes(self.model_time+timestep,self.model_time+timestep/2)
      self.kick_codes(timestep/2.)
      self.split_subcodes()

  def split_subcodes(self):
    subsystems=self.particles.collection_attributes.subsystems
    subcodes=self.subcodes
    for parent,subsys in list(subsystems.items()):
      radius=parent.radius
      components=subsys.connected_components(threshold=1.75*radius)
      if len(components)>1:
        print("splitting:", len(components))
        parentposition=parent.position
        parentvelocity=parent.velocity
        self.particles.remove_particle(parent)
        code=subcodes.pop(parent)
        offset=self.time_offsets.pop(code)
        for c in components:
          sys=c.copy_to_memory()
          sys.position+=parentposition
          sys.velocity+=parentvelocity
          if len(sys)>1:
            newcode=self.subcode_factory(sys)
            self.time_offsets[newcode]=(self.model_time-newcode.model_time)
            newcode.particles.add_particles(sys)
            newparent=self.particles.add_subsystem(newcode.particles)
            subcodes[newparent]=newcode
          else:
            newparent=self.particles.add_subsystem(sys)
          self.set_parent_particle_radius(newparent)
          print("radius:",newparent.radius.in_(units.AU),newparent.sub_worker_radius.in_(units.AU))
        del code  
      
  def handle_collision(self, coll_time,corr_time,coll_set):
    print("collision:", len(coll_set))
    print(coll_set)
    for ci in coll_set:
      print("coll:", ci)

    subsystems=self.particles.collection_attributes.subsystems
    #collsubset=self.particles[0:0]
    collsubset=Particles(2)
    print("collsubset:", collsubset)
    collsubsystems=dict()
    for p in coll_set:
      #collsubset+=p
      collsubset.add_particle(p)
      if p in self.subcodes:
        code=self.subcodes[p]
        offset=self.time_offsets[code]
        code.evolve_model(coll_time-offset)
      if p in subsystems:
        collsubsystems[p]=subsystems[p]

    self.correction_kicks(collsubset,collsubsystems,coll_time-corr_time)
    
    newparts=HierarchicalParticles(Particles())
    for p in coll_set:
      p=p.as_particle_in_set(self.particles)
      if p in self.subcodes:
        code=self.subcodes.pop(p)
        offset=self.time_offsets.pop(code)
        parts=code.particles.copy_to_memory()
        parts.position+=p.position
        parts.velocity+=p.velocity
        newparts.add_particles(parts)
        del code
      else:
        np=newparts.add_particle(p)
        np.radius=p.sub_worker_radius        
      self.particles.remove_particle(p)
    newcode=self.subcode_factory(newparts)
    self.time_offsets[newcode]=(self.model_time-newcode.model_time)
    newcode.particles.add_particles(newparts)
    newparent=self.particles.add_subsystem(newcode.particles)
    self.set_parent_particle_radius(newparent)
    print("radius:",newparent.radius.in_(units.AU),newparent.sub_worker_radius.in_(units.AU))
    self.subcodes[newparent]=newcode
    
  def find_coll_sets(self,p1,p2):
    coll_sets=UnionFind()
    for p,q in zip(p1,p2):
      print("find_coll_sets:", p, q)
      coll_sets.union(p,q)
    return coll_sets.sets()
    
  def drift_codes(self,tend,corr_time):
    code=self.parent_code
    stopping_condition = code.stopping_conditions.collision_detection
    stopping_condition.enable()
    while code.model_time < tend*(1-1.e-12):
      code.evolve_model(tend)
      if stopping_condition.is_set():
        coll_time=code.model_time
        print("coll_time:", coll_time,tend)
        coll_sets=self.find_coll_sets(stopping_condition.particles(0), stopping_condition.particles(1))
        print("collsets:",len(coll_sets))
        for cs in coll_sets:
          print("cs=", cs)
          self.handle_collision(coll_time,corr_time, cs)
          
    threads=[]
    for x in list(self.subcodes.values()):
      offset=self.time_offsets[x]
      if offset>tend:
        print("curious?")
      threads.append(threading.Thread(target=x.evolve_model, args=(tend-offset,)) )
    if self.use_threading:
      for x in threads: x.start()
      for x in threads: x.join()
    else:
      for x in threads: x.run()

  def kick_codes(self,dt):
    self.correction_kicks(self.particles,self.particles.collection_attributes.subsystems,dt)
    self.particles.recenter_subsystems()

  def correction_kicks(self,particles,subsystems,dt):
    if subsystems and len(particles)>1:
      corrector=correction_from_compound_particles(particles,subsystems,self.worker_code_factory)
      kick_particles(particles,corrector.get_gravity_at_point, dt)

      corrector=correction_for_compound_particles(particles, None, self.worker_code_factory)
      for parent,subsys in list(subsystems.items()):
        corrector.parent=parent
        kick_particles(subsys, corrector.get_gravity_at_point, dt)

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
        self.particles.collection_attributes.subsystems,self.worker_code_factory)
      Ep+=potential_energy(self.parent_code,corrector.get_potential_at_point)    
    corrector=correction_for_compound_particles(self.particles, None, self.worker_code_factory)
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


def binary(m1=1.|units.MSun,m2=.0001| units.MSun,r1=None,r2=None,ecc=0,rmin=100 | units.AU):

  mu=constants.G*(m1+m2)
  a=rmin/(1-ecc)
  P=(2*numpy.pi)*(a**3/mu)**0.5

  f1=m2/(m1+m2)
  f2=m1/(m1+m2)  

  rmax=a*(1+ecc)
  r0=rmax

  print('semimajor axis:', a.in_(units.AU))
  print('initial separation:',r0.in_(units.AU))
  print('rmax:',rmax.in_(units.AU))
  print('period:', P.in_(units.yr))
  
  h=(a*mu*(1-ecc**2))**0.5
  v0=h/r0

  bin=Particles(2)

  bin[0].mass=m1
  bin[0].x=r0*f1
  bin[0].vy=v0*f1
  bin[1].mass=m2
  bin[1].x=-r0*f2
  bin[1].vy=-v0*f2

  bin.y=0*r0
  bin.z=0.*r0
  bin.vx=0*v0
  bin.vz=0.*v0
  if r1 is None:
    bin[0].radius=(1.|units.RSun)*(m1/(1.|units.MSun))**(1./3.)
  else:
    bin[0].radius=r1
  if r2 is None:
    bin[1].radius=(1.|units.RSun)*(m2/(1.|units.MSun))**(1./3.)
  else:
    bin[1].radius=r2

  return bin

def smaller_nbody_power_of_two(dt, conv):
  nbdt=conv.to_nbody(dt).value_in(nbody_system.time)
  idt=numpy.floor(numpy.log2(nbdt))
  return conv.to_si( 2**idt | nbody_system.time)

markerstyles=["+","p","o","x"]
linestyles=["-",":","-.","--"]
colors=["r","g","b","y","c"]

def randomparticles_w_ss(N=10,L=500| units.AU,dv=2.5 | units.kms):
  from amuse.ic.salpeter import new_salpeter_mass_distribution

  conv=nbody_system.nbody_to_si(N*1.| units.MSun,1000.| units.AU)
  conv_sub=nbody_system.nbody_to_si(1.| units.MSun,50.| units.AU)

  dt=smaller_nbody_power_of_two(2000. | units.day,conv)
  print(dt.in_(units.day))

  def radius(sys,eta=0.002):
    radius=((constants.G*sys.total_mass()*dt**2/eta)**(1./3.))
    if radius < 10 | units.AU:
      radius=20. | units.AU
    return radius  
#  def radius(sys,eta=0.002):
#    if len(sys)>1:
#      xcm,ycm,zcm=sys.center_of_mass()
#      r2=((sys.x-xcm)**2+(sys.y-ycm)**2+(sys.z-zcm)**2).max()
#      return r2**0.5
#    else:
#      return ((constants.G*sys.total_mass()*dt**2/eta)**(1./3.))

  numpy.random.seed(7654304)

  masses=new_salpeter_mass_distribution(N,mass_min=0.3 | units.MSun,mass_max=10. |units.MSun)
  
  stars=Particles(N,mass=masses)
  
  stars.x=L*numpy.random.uniform(-1.,1.,N)
  stars.y=L*numpy.random.uniform(-1.,1.,N)
  stars.z=L*0.
  stars.vx=dv*numpy.random.uniform(-1.,1.,N)
  stars.vy=dv*numpy.random.uniform(-1.,1.,N)
  stars.vz=dv*0.

  stars.radius=(1.|units.RSun)*(stars.mass/(1.|units.MSun))**(1./3.)

  stars.move_to_center()

  parts=HierarchicalParticles(stars)

  ss=new_solar_system()[[0,5,6,7,8]]
  parts.assign_subsystem(ss,parts[0])

  def parent_worker():
    code=Hermite(conv)
    code.parameters.epsilon_squared=0.| units.AU**2
    code.parameters.end_time_accuracy_factor=0.
    code.parameters.dt_param=0.001
    print(code.parameters.dt_dia.in_(units.yr))
    return code
  
  def sub_worker(parts):
    mode=system_type(parts)
    if mode=="twobody":
      code=TwoBody(conv_sub)
    elif mode=="solarsystem":
      code=Mercury(conv_sub)
#      code=Huayno(conv_sub)
#      code.parameters.inttype_parameter=code.inttypes.SHARED4
    elif mode=="nbody":
      code=Huayno(conv_sub)
      code.parameters.inttype_parameter=code.inttypes.SHARED4
    return code
    
  def py_worker():
    code=CalculateFieldForParticles(gravity_constant = constants.G)
    return code
      
  nemesis=Nemesis( parent_worker, sub_worker, py_worker)
  nemesis.timestep=dt
  nemesis.radius=radius
  nemesis.commit_parameters()
  nemesis.particles.add_particles(parts)
  nemesis.commit_particles()

  tend=2000. | units.yr
  t=0|units.yr
  dtdiag=dt
  
  time=[0.]

  allparts=nemesis.particles.all()
  E=allparts.potential_energy()+allparts.kinetic_energy()
  E1=nemesis.potential_energy+nemesis.kinetic_energy

  com=allparts.center_of_mass()
  mom=allparts.total_momentum()
  ang=allparts.total_angular_momentum()
    
  E0=E
  A0=(ang[0]**2+ang[1]**2+ang[2]**2)**0.5  
  totalE=[ 0.]
  totalA=[ 0.]
  
  ss=nemesis.particles.all()
  x=(ss.x).value_in(units.AU)
  xx=[x]
  y=(ss.y).value_in(units.AU)
  yy=[y]
  
  nstep=0
  while t< tend-dtdiag/2:
    t+=dtdiag
    nemesis.evolve_model(t)  
    print(t.in_(units.yr), end=' ')
    print(len(nemesis.particles), end=' ')

    time.append( t.value_in(units.yr) )

    allparts=nemesis.particles.all()
    E=allparts.potential_energy()+allparts.kinetic_energy()
    E1=nemesis.potential_energy+nemesis.kinetic_energy
    
    ang=allparts.total_angular_momentum()
    A=(ang[0]**2+ang[1]**2+ang[2]**2)**0.5
    totalE.append(abs((E0-E)/E0))
    totalA.append(abs((A0-A)/A0))
    print(totalE[-1],(E-E1)/E)
#    print allparts.potential_energy(),nemesis.potential_energy
  
    ss=nemesis.particles.all()
    x=(ss.x).value_in(units.AU)
    y=(ss.y).value_in(units.AU)
    
    
    xcm=nemesis.particles.x.value_in(units.AU)
    ycm=nemesis.particles.y.value_in(units.AU)
    r=(nemesis.particles.radius).value_in(units.AU)
    
    xx.append(x)
    yy.append(y)
    key=ss.key

    f=pyplot.figure( figsize=(8,8))  
    ax = f.gca()
    circles=[]
    for i in range(len(xx[0])):
      pyplot.plot(xx[-1][i],yy[-1][i],colors[key[i]%numpy.uint64(len(colors))]+
                                markerstyles[key[i]%numpy.uint64(len(markerstyles))],markersize=8,mew=2)
    for p in nemesis.particles:
#      if nemesis.particles.collection_attributes.subsystems.has_key(p):
        c='k'
        code_colors=dict(TwoBody='b',Mercury='r',Huayno='g')
        if p in nemesis.subcodes:
          c=code_colors[nemesis.subcodes[p].__class__.__name__] 
        x=p.x.value_in(units.AU)
        y=p.y.value_in(units.AU)
        r=p.radius.value_in(units.AU)
        circles.append( pyplot.Circle((x,y),r,color=c,lw=0.75,fill=False) )
    for c in circles:
      ax.add_artist(c)  
                                
#    pyplot.plot(xcm,ycm,'k+', markersize=4,mew=1)
    pyplot.xlim(-600,600)
    pyplot.ylim(-600,600)
    pyplot.text(-400,-400,len(nemesis.particles),fontsize=18)
    pyplot.savefig('xy%6.6i.png'%nstep)
    f.clear()
    pyplot.close(f)

    nstep+=1

    
  time=numpy.array(time)
  totalE=numpy.array(totalE)
  totalA=numpy.array(totalA)
  xx=numpy.array(xx)
  yy=numpy.array(yy)

  f=pyplot.figure( figsize=(8,8))  
  pyplot.semilogy(time,totalE,'r')
  pyplot.semilogy(time,totalA,'g')
  pyplot.savefig('dEdA.png')  
    
  f=pyplot.figure( figsize=(8,8))  
  for i in range(len(xx[0])):
    pyplot.plot(xx[:,i],yy[:,i],colors[i%len(colors)]+linestyles[i%len(linestyles)])
  pyplot.xlim(-500,500)
  pyplot.ylim(-500,500)
  pyplot.savefig('all-xy.png')  


if __name__=="__main__":
  from nemesis import randomparticles_w_ss
  randomparticles_w_ss()

#    mencoder "mf://xy*.png" -mf fps=20 -ovc x264 -o movie.avi
  
