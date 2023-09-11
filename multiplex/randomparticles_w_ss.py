import numpy

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot

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

def randomparticles_w_ss(N=20,L=1000.| units.AU,dv=2.5 | units.kms):
  from amuse.ic.salpeter import new_salpeter_mass_distribution

  conv=nbody_system.nbody_to_si(N*1.| units.MSun,1000.| units.AU)
  conv_sub=nbody_system.nbody_to_si(1.| units.MSun,50.| units.AU)

  dt=smaller_nbody_power_of_two(1000. | units.day,conv)
  print(dt.in_(units.day))

  dt_param=0.02
  LL=L.value_in(units.AU)

  def radius(sys,eta=dt_param,_G=constants.G):
    radius=((_G*sys.total_mass()*dt**2/eta**2)**(1./3.))
    
#    xcm,ycm,zcm=sys.center_of_mass()
#    r2max=((sys.x-xcm)**2+(sys.y-ycm)**2+(sys.z-zcm)**2).max()
    
#    if radius < 10 | units.AU:
#      radius=20. | units.AU
#    return max(radius,r2max**0.5)
    return radius*((len(sys)+1)/2.)**0.75

  def timestep(ipart,jpart, eta=dt_param/2,_G=constants.G):
    dx=ipart.x-jpart.x  
    dy=ipart.y-jpart.y
    dz=ipart.z-jpart.z
    dr2=dx**2+dy**2+dz**2
#    if dr2>0:
    dr=dr2**0.5
    dr3=dr*dr2
    mu=_G*(ipart.mass+jpart.mass)
    tau=eta/2./2.**0.5*(dr3/mu)**0.5
    return tau
    
  numpy.random.seed(7654304)

  masses=new_salpeter_mass_distribution(N,mass_min=0.3 | units.MSun,mass_max=10. |units.MSun)
  
  #masses=([1.]*10) | units.MSun
  
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

#  ss=new_solar_system()[[0,5,6,7,8]]
#  parts.assign_subsystem(ss,parts[0])

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

  tend=3200. | units.yr
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
    mom=allparts.total_momentum()
    A=(ang[0]**2+ang[1]**2+ang[2]**2)**0.5
    P=mom[0].value_in(units.MSun*units.kms)
    totalE.append(abs((E0-E)/E0))
    totalA.append(abs((A0-A)/A0))
    totalP.append(abs(P0-P))
    print(totalE[-1],(E-E1)/E)
#    print allparts.potential_energy(),nemesis.potential_energy
  
    ss=nemesis.particles.all()
    x=(ss.x).value_in(units.AU)
    y=(ss.y).value_in(units.AU)
    lowm=numpy.where( ss.mass.value_in(units.MSun) < 0.1)[0]
    highm=numpy.where( ss.mass.value_in(units.MSun) >= 0.1)[0]
    
    xcm=nemesis.particles.x.value_in(units.AU)
    ycm=nemesis.particles.y.value_in(units.AU)
    r=(nemesis.particles.radius).value_in(units.AU)
    
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
        x=p.x.value_in(units.AU)
        y=p.y.value_in(units.AU)
        r=p.radius.value_in(units.AU)
        circles.append( pyplot.Circle((x,y),r,color=c,lw=0.8,ls=ls,fill=False) )
    for c in circles:
      ax.add_artist(c)  
                                
#    pyplot.plot(xcm,ycm,'k+', markersize=4,mew=1)
    pyplot.xlim(-1.2*LL,1.2*LL)
    pyplot.ylim(-1.2*LL,1.2*LL)
    pyplot.xlabel("AU")
    pyplot.text(-580,-580,'%8.2f'%t.value_in(units.yr),fontsize=18)
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
  randomparticles_w_ss()

#    mencoder "mf://xy*.png" -mf fps=20 -ovc x264 -o movie.avi
  
