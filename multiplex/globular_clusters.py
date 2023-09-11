import numpy

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot
from amuse.lab import *

from amuse.ext.solarsystem import new_solar_system
from amuse.datamodel import Particle,Particles,ParticlesOverlay
from amuse.units import units,constants,nbody_system
from amuse.couple.bridge import CalculateFieldForParticles
from amuse.units.quantities import zero


from amuse.ext.basicgraph import UnionFind

from amuse.community.twobody.twobody import TwoBody
from amuse.community.hermite0.interface import Hermite
from amuse.community.mercury.interface import Mercury
from amuse.community.ph4.interface import ph4
from amuse.community.phiGRAPE.interface import PhiGRAPE
from amuse.community.huayno.interface import Huayno
from amuse.community.mi6.interface import MI6

from nemesis import Nemesis,HierarchicalParticles,system_type

import logging
#logging.basicConfig(level=logging.DEBUG)

def smaller_nbody_power_of_two(dt, conv):
  nbdt=conv.to_nbody(dt).value_in(nbody_system.time)
  idt=numpy.floor(numpy.log2(nbdt))
  return conv.to_si( 2**idt | nbody_system.time)

markerstyles=["+","p","o","x"]
linestyles=["-",":","-.","--"]
colors=["r","g","b","y","c"]

def make_galaxy_model(N, M_galaxy, R_galaxy):
    converter = nbody_system.nbody_to_si(M_galaxy, R_galaxy)
    model = new_plummer_model(N, converter)
    model.radius = 3 | units.parsec
    model.King_W0 = 3
    return model

def initialize_globular_clusters(cluster_population, nstars):

    stars = Particles(0)
    for ci in cluster_population:
        converter = nbody_system.nbody_to_si(ci.mass, ci.radius)
        bodies = new_king_model(nstars, ci.King_W0, converter)
        bodies.parent = ci
        bodies.position += ci.position
        bodies.velocity += ci.velocity
        bodies.name = "star"
        ci.mass = bodies.mass.sum()
        bodies.scale_to_standard(converter)
        stars.add_particles(bodies)
    return stars
  
def globular_clusters(N=10, L=10.| units.kpc, dv=1.0 | units.kms):

  M_galaxy = 1.0e+11 | units.MSun
  R_galaxy = 4.5 | units.kpc
  cluster_population = make_galaxy_model(N, M_galaxy, R_galaxy)
  stars = initialize_globular_clusters(cluster_population, N)
  print(stars.mass.in_(units.MSun))
  xxx

  conv=nbody_system.nbody_to_si(M_galaxy, R_galaxy)
  conv_sub=nbody_system.nbody_to_si(1000.| units.MSun, 10.| units.parsec)

  dt=smaller_nbody_power_of_two(0.01 | units.Myr, conv)
  print(dt.in_(units.day))

  #dt_param=0.02
  dt_param=0.1
  LL=L.value_in(units.kpc)

  def radius(sys,eta=dt_param,_G=constants.G):
    radius=((_G*sys.total_mass()*dt**2/eta**2)**(1./3.))
    return radius*((len(sys)+1)/2.)**0.75

  def timestep(ipart,jpart, eta=dt_param/2,_G=constants.G):
    dx=ipart.x-jpart.x  
    dy=ipart.y-jpart.y
    dz=ipart.z-jpart.z
    dr2=dx**2+dy**2+dz**2
    dr=dr2**0.5
    dr3=dr*dr2
    mu=_G*(ipart.mass+jpart.mass)
    tau=eta/2./2.**0.5*(dr3/mu)**0.5
    return tau
    
  numpy.random.seed(7654304)

  parts=HierarchicalParticles(stars)

#  ss=new_solar_system()[[0,5,6,7,8]]
#  parts.assign_subsystem(ss,parts[0])

  def parent_worker():
    code=Hermite(conv)
    code.parameters.epsilon_squared=0.| units.kpc**2
    code.parameters.end_time_accuracy_factor=0.
    #code.parameters.dt_param=0.001
    code.parameters.dt_param=0.1
    print(code.parameters.dt_dia.in_(units.yr))
    return code
  
  
  def sub_worker(parts):
    mode=system_type(parts)
    if mode=="twobody":
      code=TwoBody(conv_sub)
    elif mode=="solarsystem":
      #code=Mercury(conv_sub)
      code=Huayno(conv_sub)
    elif mode=="nbody":
      code=Huayno(conv_sub)
      code.parameters.inttype_parameter=code.inttypes.SHARED4
    return code
      
  def py_worker():
    code=CalculateFieldForParticles(gravity_constant = constants.G)
    return code
      
  nemesis=Nemesis( parent_worker, sub_worker, py_worker)
  nemesis.timestep=dt
  nemesis.distfunc=timestep
  nemesis.threshold=dt
  nemesis.radius=radius
  nemesis.commit_parameters()
  nemesis.particles.add_particles(parts)
  nemesis.commit_particles()

  tend=1.0 | units.Myr
  t=0|units.yr
  dtdiag=dt*2
  
  time=[0.]

  allparts=nemesis.particles.all()
  E=allparts.potential_energy()+allparts.kinetic_energy()
  E1=nemesis.potential_energy+nemesis.kinetic_energy

  com=allparts.center_of_mass()
  mom=allparts.total_momentum()
  ang=allparts.total_angular_momentum()
    
  E0=E
  A0=(ang[0]**2+ang[1]**2+ang[2]**2)**0.5 
  P0=mom[0].value_in(units.MSun*units.kms)
  totalE=[ 0.]
  totalA=[ 0.]
  totalP=[ 0.]
  
  ss=nemesis.particles.all()
  x=(ss.x).value_in(units.kpc)
  xx=[x]
  y=(ss.y).value_in(units.kpc)
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
    mom=allparts.total_momentum()
    A=(ang[0]**2+ang[1]**2+ang[2]**2)**0.5
    P=mom[0].value_in(units.MSun*units.kms)
    totalE.append(abs((E0-E)/E0))
    totalA.append(abs((A0-A)/A0))
    totalP.append(abs(P0-P))
    print(totalE[-1],(E-E1)/E)
#    print allparts.potential_energy(),nemesis.potential_energy
  
    ss=nemesis.particles.all()
    x=(ss.x).value_in(units.kpc)
    y=(ss.y).value_in(units.kpc)
    lowm=numpy.where( ss.mass.value_in(units.MSun) < 0.1)[0]
    highm=numpy.where( ss.mass.value_in(units.MSun) >= 0.1)[0]
    
    xcm=nemesis.particles.x.value_in(units.kpc)
    ycm=nemesis.particles.y.value_in(units.kpc)
    r=(nemesis.particles.radius).value_in(units.kpc)
    
    xx.append(x)
    yy.append(y)
    key=ss.key

    f=pyplot.figure( figsize=(8,8))  
    ax = f.gca()
    circles=[]
#    for i in range(len(xx[0])):
#      pyplot.plot(xx[-1][i],yy[-1][i],colors[key[i]%numpy.uint64(len(colors))]+
#                                markerstyles[key[i]%numpy.uint64(len(markerstyles))],markersize=8,mew=2)
    pyplot.plot(xx[-1][highm],yy[-1][highm],'go',markersize=8,mew=2)
    pyplot.plot(xx[-1][lowm],yy[-1][lowm],'b+',markersize=6,mew=1.5)

    for p in nemesis.particles:
#      if nemesis.particles.collection_attributes.subsystems.has_key(p):
        c='k'
        ls='solid'
        code_colors=dict(TwoBody='b',Mercury='r',Huayno='g',Hermite='y')
        code_ls=dict(TwoBody='dotted',Mercury='dashed',Huayno='dashdot',Hermite='solid')
        if p in nemesis.subcodes:
          c=code_colors[nemesis.subcodes[p].__class__.__name__] 
          ls=code_ls[nemesis.subcodes[p].__class__.__name__] 
        x=p.x.value_in(units.kpc)
        y=p.y.value_in(units.kpc)
        r=p.radius.value_in(units.kpc)
        circles.append( pyplot.Circle((x,y),r,color=c,lw=0.8,ls=ls,fill=False) )
    for c in circles:
      ax.add_artist(c)  
                                
#    pyplot.plot(xcm,ycm,'k+', markersize=4,mew=1)
    pyplot.xlim(-1.2*LL,1.2*LL)
    pyplot.ylim(-1.2*LL,1.2*LL)
    pyplot.xlabel("kpc")
    #pyplot.text(-580,-580,'%8.2f'%t.value_in(units.yr),fontsize=18)
#    pyplot.text(-400,-400,len(nemesis.particles),fontsize=18)
    pyplot.savefig('xy%6.6i.png'%nstep,bbox_inches='tight')
    f.clear()
    pyplot.close(f)

    nstep+=1

    
  time=numpy.array(time)
  totalE=numpy.array(totalE)
  totalA=numpy.array(totalA)
  totalP=numpy.array(totalP)
  xx=numpy.array(xx)
  yy=numpy.array(yy)

  f=pyplot.figure( figsize=(8,8))  
  pyplot.semilogy(time,totalE,'r')
  pyplot.semilogy(time,totalA,'g')
  pyplot.semilogy(time,totalP,'b')
  pyplot.savefig('dEdA.png')  
    
  f=pyplot.figure( figsize=(8,8))  
  for i in range(len(xx[0])):
    pyplot.plot(xx[:,i],yy[:,i],colors[i%len(colors)]+linestyles[i%len(linestyles)])
  pyplot.xlim(-LL,LL)
  pyplot.ylim(-LL,LL)
  pyplot.savefig('all-xy.png')  


if __name__=="__main__":
  globular_clusters()

  
