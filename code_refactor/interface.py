import time as cpu_time

from amuse.units import units, nbody_system

from evol import *
from grav_correctors import *
from hierarchical_particles import *
from nemesis import *
from particle_initialiser import *

def run_code(N, cfrac, mass_min, mass_max, power_law, rcluster, vdisp, tend, dt, eta):
  """Function to run simulation.
     Inputs:
     N:         Number of parent systems
     cfrac:     Fraction of parents with children
     rcluster:  Scale length of the system
     vdisp:     Velocity dispersion
     mass_min:  Minimum stellar mass
     mass_max:  Maximum stellar mass
     power_law: Stellar mass power law function
     tend:      Simulation end time
     dt:        Parent system simulation step-time
     eta:       Parameter influencing parent system radius"""

  def smaller_nbody_power_of_two(dt, conv):
    """Function to scale the dt parameter relative to N-body units"""

    nbdt=conv.to_nbody(dt).value_in(nbody_system.time)
    idt=np.floor(np.log2(nbdt))
    return conv.to_si( 2**idt | nbody_system.time)

  conv_par = nbody_system.nbody_to_si(N*1.| units.MSun, 2*rcluster)    #Converter for parent
  conv_child = nbody_system.nbody_to_si(1.| units.MSun, 50.| units.AU) #Converter for children
  dt = smaller_nbody_power_of_two(dt, conv_par)

  #Initialising parent
  pset_init = particle_init()
  stars = pset_init.cluster_distr(N, rcluster, "Dummy", vdisp, conv_par)
  stars.mass = pset_init.parent_mass_distr(stars, mass_min, mass_max, power_law, "Salpeter")
  stars.radius = pset_init.child_radius(stars.mass)
  stars.move_to_center()
  stars.scale_to_standard(convert_nbody = conv_par)
  parents = HierarchicalParticles(stars)

  #Initialising children
  no_child = np.arange(0,9)
  for par_ in range(int(cfrac*N)):
    print("...Assigning children to system ", par_, "...")
    planet_sys = pset_init.child_particles(no_child, stars, None, None, None, None, None, conv_child)
    parents.assign_subsystem(planet_sys, parents[par_], dt, eta)

  nemesis = Nemesis(conv_par, conv_child, dt, eta)
  nemesis.timestep = dt
  nemesis.radius = pset_init.parent_radius(parents, dt, eta)
  nemesis.particles.add_particles(parents)
  nemesis.commit_particles(conv_child)

  tend = 2000. | units.yr
  t = 0 | units.yr
  dtdiag = dt
  time = [0.]

  allparts = nemesis.particles.all() #Complete particle set of simulation (removes c.o.m & star redundancy)

  E0a = allparts.kinetic_energy()+allparts.potential_energy()
  E0n = nemesis.potential_energy+nemesis.kinetic_energy
  E0c = E0a-E0n
  ang = allparts.total_angular_momentum()
  A0 = (ang[0]**2+ang[1]**2+ang[2]**2)**0.5  
  totalE = [ 0.]
  totalA = [ 0.]
  
  x = (allparts.x).value_in(units.AU)
  y = (allparts.y).value_in(units.AU)
  
  while t < tend-dtdiag/2:
    t += dtdiag
    time.append(t.value_in(units.yr))

    nemesis.evolve_model(t)  
    allparts=nemesis.particles.all()
    E = allparts.potential_energy()+allparts.kinetic_energy()
    E1 = nemesis.potential_energy+nemesis.kinetic_energy
    E2 = E-E1

    if (t.value_in(units.yr) % 50 < dtdiag.value_in(units.yr)):
      print("Time: ", t.in_(units.yr), end=' ')
      print("No. Particles: ", len(nemesis.particles))
      print("Energy error (all):     ", abs(E0a-E)/E0a)
      print("Energy error (parents): ", abs(E0n-E1)/E0n)
      print("Energy error (children):", abs(E0c-E2)/E0c)
    
    ang = allparts.total_angular_momentum()
    A = (ang[0]**2+ang[1]**2+ang[2]**2)**0.5
    totalE.append(abs((E0a-E)/E0a))
    totalA.append(abs((A0-A)/A0))

if __name__=="__main__":
  start_time = cpu_time.time()
  run_code(N = 10, cfrac = 1/10, mass_min = 0.3 | units.MSun, mass_max = 10 | units.MSun, 
           power_law = -2.35, rcluster = 30 | units.pc, vdisp = 2.5 | units.kms,
           tend = 2e3 | units.yr, dt = 2e3 | units.day, eta = 2e-3)
  end_time = cpu_time.time()
  tot_time = end_time - start_time
  print("Total time: ", tot_time, " secs.")
  
