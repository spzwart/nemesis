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
#from amuse.community.phiGRAPE.interface import PhiGRAPE
from amuse.community.huayno.interface import Huayno
from amuse.community.bhtree.interface import BHTree
from amuse.community.mi6.interface import MI6

#from nemesis import Nemesis, HierarchicalParticles, system_type
from nemesis import nemesis
from nemesis import collision_code
from nemesis import multiplexing_gd

from amuse.couple import bridge
from galactic_model import IntegrateOrbit

import logging
#logging.basicConfig(level=logging.DEBUG)

USE_THREADING = True
GRAVITATIONAL_SOFTENING = (1.0|units.AU)**2
###NEMESIS_UPDATE_TIMESTEP = 0.1|units.Myr
NEMESIS_UPDATE_TIMESTEP = 1.0|units.Myr

def all_particles(nemesis_particles):
    parts=Particles()
    for parent in nemesis_particles:
      if parent.subsystem is None:   
        parts.add_particle(parent)
      else:
        subsys=parts.add_particles(parent.subsystem)
        subsys.position+=parent.position
        subsys.velocity+=parent.velocity
    return parts

def add_new_subsystems(from_set, to_set):
    from_with_subsystem = from_set[~numpy.equal(from_set.subsystem, None)]
    to_without_subsystem = to_set[numpy.equal(to_set.subsystem, None)]
    subsystems_not_copied_yet = to_without_subsystem.get_intersecting_subset_in(from_with_subsystem)
    subsystems_not_copied_yet.new_channel_to(to_without_subsystem).copy_attribute("subsystem")
    
def remove_old_subsystems(from_set, to_set):
    from_without_subsystem = from_set[numpy.equal(from_set.subsystem, None)]
    to_with_subsystem = to_set[~numpy.equal(to_set.subsystem, None)]
    to_with_subsystems_lingering = from_without_subsystem.get_intersecting_subset_in(to_with_subsystem)
    to_with_subsystems_lingering.subsystem = None
    
def extract_subsystems(particles):
    stars = Particles()
    planets = Particles()
    for si in particles:
        if si.subsystem:
            com = si.position
            for pi in si.subsystem:
                if pi.type==1:
                    pi.position += com
                    planets.add_particle(pi)
                else:
                    pi.position += com
                    stars.add_particle(pi)
        else:
            if si.type==1:
                planets.add_particle(si)
            else:
                stars.add_particle(si)
    return planets, stars

def get_stars_to_nemesis_particles_channels(stars, nemesis_particles):
    stars_without_planets = nemesis_particles[numpy.equal(nemesis_particles.subsystem, None)]
    systems_with_planets = nemesis_particles[~numpy.equal(nemesis_particles.subsystem, None)]
        
    result = []
    result.append(stars.new_channel_to(stars_without_planets))
    for x in systems_with_planets:
        result.append(stars.new_channel_to(x.subsystem))
       
    return result

def get_nemesis_particles_to_stars_channels(stars, nemesis_particles):
    stars_without_planets = nemesis_particles[numpy.equal(nemesis_particles.subsystem, None)]
    systems_with_planets = nemesis_particles[~numpy.equal(nemesis_particles.subsystem, None)]
        
    result = []
    result.append(stars_without_planets.new_channel_to(stars))
    for x in systems_with_planets:
        result.append(x.subsystem.new_channel_to(stars))
       
    return result

def get_from_code_to_model_channels(code, model):
        
    if not hasattr(code.particles, 'subsystem'):
        return []
        
    model_particles_without_subsystems = model[numpy.equal(model.subsystem, None)]
    model_particles_with_subsystems = model[~numpy.equal(model.subsystem, None)]
        
    code_particles_without_subsystems = code.particles[numpy.equal(code.particles.subsystem, None)]
    code_particles_with_subsystems = code.particles[~numpy.equal(code.particles.subsystem, None)]
        
    result = []
    result.append(code.particles.new_channel_to(model))
    for x, y in zip(sorted(code_particles_with_subsystems, key=lambda x : x.key), sorted(model_particles_with_subsystems, key=lambda x : x.key)):
        if x.key != y.key:
            print("KEYS DO NOT MATCH < BUT THEY SHOULD!!!", x.key, y.key)
        else:
            result.append(x.subsystem.new_channel_to(y.subsystem))
       
    return result

def synchronize_sets(from_set, to_set):
    from_set.synchronize_to(to_set)
        
    if not hasattr(from_set, 'subsystem'):
        return
        
    to_particles_with_subsystems = to_set[~numpy.equal(to_set.subsystem, None)]
    from_particles_with_subsystems = from_set[~numpy.equal(from_set.subsystem, None)]
        
    if len(to_particles_with_subsystems) != len(from_particles_with_subsystems):
        print("to subsystem:", len(to_particles_with_subsystems))
        print("from subsystem:", len(from_particles_with_subsystems))
        print(to_particles_with_subsystems - from_particles_with_subsystems)
        raise Exception("the number of particles with subsytems do not match!")

    from_in_order = sorted(from_particles_with_subsystems, key=lambda x : x.key)
    to_in_order = sorted(to_particles_with_subsystems, key=lambda x : x.key)
    for x, y in zip(from_in_order,to_in_order):
        if x.key != y.key:
            raise Exception("the parent keys of 2 subsystems do not match, they should ({0},{1})".format(x.key, y.key))
        else:
            x.subsystem.synchronize_to(y.subsystem)

def get_from_model_to_code_channels(code, model):
    if not hasattr(code.particles, 'subsystem'):
        return []
        
    model_particles_without_subsystems = model[numpy.equal(model.subsystem, None)]
    model_particles_with_subsystems = model[~numpy.equal(model.subsystem, None)]

    code_particles_without_subsystems = code.particles[numpy.equal(code.particles.subsystem, None)]
    code_particles_with_subsystems = code.particles[~numpy.equal(code.particles.subsystem, None)]
        
    result = []
    result.append(model_particles_without_subsystems.new_channel_to(code_particles_without_subsystems))
    for x, y in zip(sorted(model_particles_with_subsystems, key=lambda x : x.key), sorted(code_particles_with_subsystems, key=lambda x : x.key)):
        result.append(x.subsystem.new_channel_to(y.subsystem))
       
    return result

#from nemesis import collision_code
#from nemesis import multiplexing_gd

        
def make_shared_code(subsystem_code_name = "Huayno"):
    converter = nbody_system.nbody_to_si(1000 | units.MSun,  100 | units.parsec)
    remote_code = multiplexing_gd.MultiplexingGravitationalDynamicsInterface(
        implementation_factory=multiplexing_gd.MultiplexingGravitationalDynamicsImplementationWithLocalCode
    )
    code = multiplexing_gd.MultiplexingGravitationalDynamicsCode(converter, remote_code = remote_code)
    code.initialize_code()
    code.set_code(subsystem_code_name)
    code.stopping_conditions.collision_detection.enable()
    print(code)
    code.commit_parameters()
    return code
    

def make_sharing_codes(number_of_codes_to_share = 10,  subsystem_code_name = "Huayno"):
    if number_of_codes_to_share > 0:
        result = list([make_shared_code(subsystem_code_name) for _ in range(number_of_codes_to_share)])
        for x in result:
            x._initial = True
    else:
        result = []
    return result

SHARING_CODES = None
CYCLE_INDEX = 0

def nemesis_function(converter, particles):
        
    global SHARING_CODES
    if SHARING_CODES is None:
        SHARING_CODES = make_sharing_codes(10, "Hermite")
#        SHARING_CODES = make_sharing_codes(10, "Huayno8")
#        SHARING_CODES = make_sharing_codes(10, "Huayno")

    def make_code(particles, time):
        #Nbody code for sub-system (planetesimals)
        #result = Hermite(converter) # N-body code for planets
        #result = ph4(converter) # N-body code for planets
        result = Hermite(converter) # N-body code for planets
        #result = Huayno(converter) # N-body code for planets                    
        #result = Huayno(converter, mode="openmp") # N-body code for planets                    
        result.parameters.timestep_parameter = 0.1

        #result.parameters.inttype_parameter = Huayno.inttypes.SHARED4_COLLISIONS
        result.parameters.epsilon_squared = GRAVITATIONAL_SOFTENING

        #result.parameters.stopping_conditions_timeout = 300 | units.s # not used without stopppingc.
        result.parameters.stopping_conditions_timeout = 1 | units.yr # not used without stopppingc.
        #result.stopping_conditions.timeout_detection.enable()        

        # set code begin time (CHECK IF OK)
        #result.parameters.begin_time=bookkeeping.model_parameters.current_time
        # enable collisions in nemesis (between planets and stars)
        result = collision_code.CollisionCode(result, G=constants.G)
        result.stopping_conditions.collision_detection.enable() # print a collision
        result.particles.add_particles(particles)
        #result.commit_particles()
        #result.particles.ancestors = None
        return result
    
    
    def make_shared_code(particles, time):
        global CYCLE_INDEX
        real_code = SHARING_CODES[CYCLE_INDEX]
        CYCLE_INDEX += 1
        if CYCLE_INDEX >= len(SHARING_CODES):
            CYCLE_INDEX = 0
        subset = real_code.new_subset(time)
        particles.index_of_the_set = subset
        code = nemesis.SubsetCode(real_code, subset)
        real_code.particles.add_particles(particles)
        code.update()
        return code
        
    def make_gravity_code(particles):
        return nemesis.CalculateFieldForParticles(particles, G = constants.G)
        
    #Nbody code for global-system (stars)
    #Nbody_code = ph4(converter) # N-body code for stars
    Nbody_code = BHTree(converter, number_of_workers=4) # N-body code for stars
    Nbody_code.parameters.epsilon_squared = GRAVITATIONAL_SOFTENING
    # print result.parameters.timestep_parameter
    # set code begin time (CHECK IF OK - or set Nemesis begin time?)
    
    Nbody_code.commit_parameters() 
    radius_function = lambda x : 1000 | units.parsec

    # bridge timestep (between cluster and planetasimals)   
    #timestep = 100 | units.yr
    timestep = NEMESIS_UPDATE_TIMESTEP
    code = nemesis.Nemesis(Nbody_code, make_shared_code, make_gravity_code, timestep, subsystem_factory = nemesis.MultiGravityCodeSubsystem, G = constants.G, use_threading=USE_THREADING)
    code.particles.add_particles(particles)
    code.commit_particles()
    print("Nemesis:", code.model_time.in_(units.Myr))

    return code

def smaller_nbody_power_of_two(dt, conv):
  nbdt=conv.to_nbody(dt).value_in(nbody_system.time)
  idt=numpy.floor(numpy.log2(nbdt))
  return conv.to_si( 2**idt | nbody_system.time)

markerstyles=["+","p","o","x"]
linestyles=["-",":","-.","--"]
colors=["r","g","b","y","c"]

def plot_figure(stars, nemesis, nstep, LL, filename):
    import matplotlib.cm as cm
    #stars, clusters = extract_subsystems(nemesis_particles)

    ss=nemesis.particles
    #stars = nemesis.particles.copy()
    f=pyplot.figure( figsize=(8,8))  
    ax = f.gca()
    circles=[]
    s = 6*stars.mass/stars.mass.max()
    m = stars.mass/stars.mass.max()
    #pyplot.plot(xx[-1],yy[-1],'b.',markersize=2, mew=1.5)
    if 'xy' in filename:
      for si in range(len(stars)):
        if "A" in stars[si].name:
          c = 'r.'
        else:
          c = 'b.'
        pyplot.plot(stars[si].x.value_in(units.kpc),stars[si].y.value_in(units.kpc),c,markersize=s[si], mew=s[si], alpha=0.5)
    else:
      for si in range(len(ss)):
        if "A" in stars[si].name:
          c = 'r.'
        else:
          c = 'b.'
        pyplot.plot(stars[si].x.value_in(units.kpc),stars[si].z.value_in(units.kpc),c,markersize=s[si], mew=s[si], alpha=0.5)

    """
    for p in nemesis.particles:
        c='k'
        ls='solid'
        code_colors=dict(TwoBody='b',Mercury='r',Huayno='g',Hermite='y')
        code_ls=dict(TwoBody='dotted',Mercury='dashed',Huayno='dashdot',Hermite='solid')
        if nemesis.subcodes.has_key(p):
          c=code_colors[nemesis.subcodes[p].__class__.__name__] 
          ls=code_ls[nemesis.subcodes[p].__class__.__name__]
        if 'xy' in filename:
          x=p.x.value_in(units.kpc)
          y=p.y.value_in(units.kpc)
        else:
          x=p.x.value_in(units.kpc)
          y=p.z.value_in(units.kpc)
        r=p.radius.value_in(units.kpc)
        circles.append( pyplot.Circle((x,y),r,color=c,lw=0.8,ls=ls,fill=False) )
    for c in circles:
      ax.add_artist(c)  
    """

    pyplot.xlim(-1.2*LL,1.2*LL)
    pyplot.ylim(-1.2*LL,1.2*LL)
    pyplot.xlabel("kpc")
    pyplot.savefig(filename+'%6.6i.png'%nstep,bbox_inches='tight')
    #pyplot.show()
    f.clear()
    pyplot.close(f)
  
def globular_clusters(stars, M_galaxy, R_galaxy):

  print("Stars:", len(stars), stars.mass.sum().in_(units.MSun), stars.mass.max().in_(units.MSun), stars.mass.mean().in_(units.MSun))

  stellar = SeBa()
  stellar.particles.add_particles(stars)
  channel_from_stellar = stellar.particles.new_channel_to(stars)
  
  conv=nbody_system.nbody_to_si(M_galaxy, R_galaxy)
  conv_sub=nbody_system.nbody_to_si(1000.| units.MSun, 10.| units.parsec)

  dt=smaller_nbody_power_of_two(1.0 | units.Myr, conv)
  print(dt.in_(units.Myr))
  dt_bridge = dt

  L = 10 | units.kpc
  LL=L.value_in(units.kpc)

  #converter = nbody_system.nbody_to_si(M_galaxy, R_Galaxy)

  
  OS= 20 |(units.kms/units.kpc)
  OB= 40 |(units.kms/units.kpc)
  A= 1300 |(units.kms**2/units.kpc)
  M= 1.4e10 |units.MSun
  m=2    

  phi_bar, phi_sp= -0.34906, -0.34906
  galaxy= IntegrateOrbit(t_end= 0.1|units.Myr,
                         dt_bridge= dt_bridge, 
                         phase_bar= phi_bar, phase_spiral= phi_sp, 
                         omega_spiral= OS, omega_bar= OB, 
                         amplitude= A, m=m, mass_bar= M )
  MWG = galaxy.galaxy()

  nemesis_particles = stars.copy()
  nemesis = nemesis_function(conv, nemesis_particles)
                                           
  channels = get_from_code_to_model_channels(nemesis, nemesis_particles)
  for x in channels:  
      x.copy()
  channels = get_nemesis_particles_to_stars_channels(stars, nemesis_particles)
  for x in channels:  
      x.copy_attributes(["x", "y", "z", "vz", "vy", "vz"])
                        
  gravity = bridge.Bridge(use_threading=False)
  gravity.add_system(nemesis, (MWG,) )
  gravity.timestep = dt_bridge
  
  tend=1.0 | units.Gyr
  t=0|units.yr
  dtdiag=dt*2
  
  time=[0.]

  allparts=nemesis.particles
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
  totaldM=[ 0. ] | units.MSun
  
  ss=nemesis.particles
  x=(ss.x).value_in(units.kpc)
  xx=[x]
  y=(ss.y).value_in(units.kpc)
  yy=[y]
  z=(ss.z).value_in(units.kpc)
  zz=[z]

  nstep=0
  #plot_figure(stars, nemesis, nstep, LL, 'xy_i')
  #plot_figure(stars, nemesis, nstep, LL, 'xz_i')

  """
  #stars = all_particles(nemesis.particles)
  filename = 'nemesis_i%6.6i.amuse'%nstep
  write_set_to_file(stars, filename, "amuse", append_to_file=False, version="2.0")
  """

  """
  filename = 'nemesis_i%6.6i.amuse'%nstep
  write_set_to_file(nemesis.particles.all(), filename, "amuse", append_to_file=False, version="2.0", names = names)
  """

  filename = 'globulars_i%6.6i.amuse'%nstep
  names = ('nemesis_particles', 'stars')
  sets = (nemesis_particles, stars)
  write_set_to_file(sets, filename, "amuse", timestamp = (0|units.Myr), append_to_file=False, version="2.0", names = names)

  """
  filename = 'stars_i%6.6i.amuse'%nstep
  write_set_to_file(stars, filename, "hdf5", timestamp=t, version="2.0", append_to_file=False)
  filename = 'nemesis_i%6.6i.amuse'%nstep
  write_set_to_file(nemesis_particles, filename, "hdf5", timestamp=t, version="2.0", append_to_file=False)
  """
  
  while t< tend-dtdiag/2:
    t+=dtdiag
    nstep+=1

    #s = nemesis.particles.copy()
    gravity.evolve_model(t)
    #p = s.position-nemesis.particles.position
    #print p
    #nemesis.evolve_model(t)  
    print(t.in_(units.Myr), end=' ')
    print(len(nemesis.particles), end=' ')

    #get_from_code_to_model_channels(ode, model)
    nemesis.particles.synchronize_to(nemesis_particles)
    remove_old_subsystems(nemesis.particles, nemesis_particles)
    synchronize_sets(nemesis.particles, nemesis_particles)
    channels = get_from_code_to_model_channels(nemesis, nemesis_particles)
    for x in channels:  
        x.copy()
        channels = get_nemesis_particles_to_stars_channels(stars, nemesis_particles)
    for x in channels:  
        x.copy_attributes(["x", "y", "z", "vz", "vy", "vz"])
    
    m = stars.mass.sum()
    stellar.evolve_model(t)
    channel_from_stellar.copy()
    channels = get_stars_to_nemesis_particles_channels(stars, nemesis_particles)
    for x in channels:  
        x.copy() #_attributes(["mass"])
        #x.copy_attributes(["temperature", "luminosity", "stellar_type"])
    nemesis_particles.synchronize_to(nemesis.particles)

    add_new_subsystems(nemesis_particles, nemesis.particles)
    synchronize_sets(nemesis.particles, nemesis_particles)
    nemesis.recommit_particles()
    channels = get_from_model_to_code_channels(nemesis, nemesis_particles)
    for x in channels:  
        x.copy()
        nemesis.update_systems()
                    
    dm = m-stars.mass.sum()
    totaldM.append(totaldM[-1]+dm)
    print("dM=", totaldM[-1].in_(units.MSun), dm.in_(units.MSun), end=' ')
    #channel_to_nemesis.copy()
    
    time.append( t.value_in(units.yr) )

    """
    #stars = all_particles(nemesis.particles)
    filename = 'nemesis_i%6.6i.amuse'%nstep
    write_set_to_file(stars, filename, "amuse", append_to_file=False, version="2.0")
    """
    
    """
    filename = 'nemesis_i%6.6i.amuse'%nstep
    write_set_to_file(nemesis.particles.all(), filename, "amuse", append_to_file=False, version="2.0", names = names)
    """

    filename = 'globulars_i%6.6i.amuse'%nstep
    names = ('nemesis_particles', 'stars')
    sets = (nemesis_particles, stars)
    write_set_to_file(sets, filename, "amuse", timestamp=t, append_to_file=False, version="2.0", names = names)

    """
    filename = 'stars_i%6.6i.amuse'%nstep
    write_set_to_file(stars, filename, "hdf5", timestamp=t, version="2.0", append_to_file=False)
    filename = 'nemesis_i%6.6i.amuse'%nstep
    write_set_to_file(nemesis_particles, filename, "hdf5", timestamp=t, version="2.0", append_to_file=False)
    """
    
    allparts=nemesis.particles
    E=allparts.potential_energy()+allparts.kinetic_energy()
    E1=nemesis.potential_energy+nemesis.kinetic_energy
    
    ang=allparts.total_angular_momentum()
    mom=allparts.total_momentum()
    A=(ang[0]**2+ang[1]**2+ang[2]**2)**0.5
    P=mom[0].value_in(units.MSun*units.kms)
    totalE.append(abs((E0-E)/E0))
    totalA.append(abs((A0-A)/A0))
    totalP.append(abs(P0-P))
    print("dE=", totalE[-1],(E-E1)/E)
#    print allparts.potential_energy(),nemesis.potential_energy
  
    ss=nemesis.particles
    x=(ss.x).value_in(units.kpc)
    y=(ss.y).value_in(units.kpc)
    y=(ss.z).value_in(units.kpc)
    lowm=numpy.where( ss.mass.value_in(units.MSun) < 1)[0]
    highm=numpy.where( ss.mass.value_in(units.MSun) >= 1)[0]
    print("N=", len(stars), len(ss), len(highm), len(lowm))
    print(stars)
    
    xcm=nemesis.particles.x.value_in(units.kpc)
    ycm=nemesis.particles.y.value_in(units.kpc)
    r=(nemesis.particles.radius).value_in(units.kpc)
    
    xx.append(x)
    yy.append(y)
    zz.append(z)
    key=ss.key

    #plot_figure(stars, nemesis, nstep, LL, 'xy_i')
    #plot_figure(stars, nemesis, nstep, LL, 'xz_i')

  """
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
  """


if __name__=="__main__":
  numpy.random.seed(7654304)
  stars = read_set_from_file("globular_starclusters_in_galactic_potential.amuse", "hdf5")
  M_galaxy = 4.4e+10 | units.MSun
  R_galaxy = 4.5 | units.kpc
  globular_clusters(stars, M_galaxy, R_galaxy)

  
