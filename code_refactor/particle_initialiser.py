import numpy as np

from amuse.ic.fractalcluster import new_fractal_cluster_model
from amuse.ic.kroupa import new_kroupa_mass_distribution
from amuse.ic.plummer import new_plummer_model
from amuse.ic.salpeter import new_salpeter_mass_distribution

from amuse.datamodel import Particles
from amuse.units import units, nbody_system, constants
from amuse.ext.solarsystem import new_solar_system

class particle_init(object):
  """Class to initialise particle system"""
  def child_particles(self, Np, pset, mmin, mmax, crad, pos_dis, mass_dis, conv):
      """Initialise planets
         Inputs:
         Np:         Array with possible number of children in subsystem
         pset:       Parent particle set
         mmin:   Minimum child mass (if no distribution, Mmin equal-mass system)
         mmax:   Maximum child mass
         crad:       Child system radius
         pos_dis:    Children position distribution
         mass_dis:   Children mass distribution
         conv:       N-body converter"""

      #num_child = np.random.randint(len(Np))
      #child_sys = Particles(Np[num_child])
      #Define some mass distribution and position distribution
      
      child_sys = new_solar_system()[[0, 5, 6, 7, 8]]

      return child_sys

  def cluster_distr(self, N, rcluster, pos_dis, vdisp, conv):
    """Initialise parent system positional distribution
       Inputs:
       N:        Number of parent systems
       rcluster: Cluster radius
       pos_dis:  Parent position distribution
       vdisp:    Dispersion velocity
       conv:     N-body converter"""
    
    if pos_dis == "Plummer" or pos_dis == "plummer":
      pset = new_plummer_model(N, convert_nbody = conv)
    elif pos_dis == "Fractal" or pos_dis == "fractal":
      pset = new_fractal_cluster_model(N, fractal_dimension = fdim, convert_nbody = conv)
    else:
      pset=Particles(N)
      pset.x = rcluster*np.random.uniform(-1.,1.,N)
      pset.y = rcluster*np.random.uniform(-1.,1.,N)
      pset.z = rcluster*0*np.random.uniform(-1.,1.,N)
      pset.vx = vdisp*np.random.uniform(-1.,1.,N)
      pset.vy = vdisp*np.random.uniform(-1.,1.,N)
      pset.vz = vdisp*0.

    return pset

  def parent_mass_distr(self, pset, mmin, mmax, power_law,  mdis):
    """Initialise parent system mass distribution
       Inputs:
       pset:      Parent particle set
       mmin:      Minimum stellar mass (if no distribution, Mmin equal-mass system)
       mmax:      Maximum stellar mass
       power_law: Power-law relation of mass distribution
       mdis:      String defining mass distribution model"""
    
    if mdis == "Kroupa" or mdis == "kroupa":
      masses = new_kroupa_mass_distribution(len(pset), mmin, mmax, power_law)
    elif mdis == "Salpeter" or mdis == "salpeter":
      masses = new_salpeter_mass_distribution(len(pset), mmin, mmax, power_law)
    else:
      masses = mmin
      
    return masses
  
  def child_radius(self, pmass):
    """Initialise particle radius"""
    return (1 | units.RSun) * (pmass / (1 | units.MSun))**(1/3)
  
  def parent_radius(self, pset, dt, eta):
      """Define merging/dissolution radius of parent systems
      Inputs:
      pset:   Parent particle set 
      dt:     Simulation time-step
      eta:    Simulation time-step/tend"""

      radius=((constants.G*pset.total_mass()*dt**2/eta)**(1./3.))
      if radius < 10 | units.AU:
          radius = 20. | units.AU

      return radius
      