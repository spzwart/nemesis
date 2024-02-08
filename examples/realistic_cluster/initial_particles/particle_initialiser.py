import glob
import numpy as np
import os
from natsort import natsorted
from numpy import random

from amuse.datamodel import Particles, Particle
from amuse.ext.orbital_elements import new_binary_from_orbital_elements
from amuse.lab import read_set_from_file, write_set_to_file
from amuse.units import units, constants


def ZAMS_radius(mass):
  """Define stellar radius at ZAMS"""

  mass = mass.value_in(units.MSun)
  mass_sq = (mass)**2
  r_zams = mass**1.25*(0.1148+0.8604*mass_sq)/(0.04651+mass_sq)
  return r_zams | units.RSun

def new_rotation_matrix_from_euler_angles(phi, theta, chi):
  cosp=np.cos(phi)
  sinp=np.sin(phi)
  cost=np.cos(theta)
  sint=np.sin(theta)
  cosc=np.cos(chi)
  sinc=np.sin(chi)
  return np.array(
      [[cost*cosc, -cosp*sinc + sinp*sint*cosc, sinp*sinc + cosp*sint*cosc], 
        [cost*sinc, cosp*cosc + sinp*sint*sinc, -sinp*cosc + cosp*sint*sinc],
        [-sint,  sinp*cost,  cosp*cost]])

def rotate(position, velocity, phi, theta, psi): # theta and phi in radians
    Runit = position.unit
    Vunit = velocity.unit
    matrix = new_rotation_matrix_from_euler_angles(phi, theta, psi)
    return (np.dot(matrix, position.value_in(Runit)) | Runit,
            np.dot(matrix, velocity.value_in(Vunit)) | Vunit)

def load_particles():
  """Load particles from previous data sets"""

  data_dir = os.path.join("examples", "realistic_cluster", 
                          "initial_particles", "data_files")

  env_files = natsorted(glob.glob(os.path.join(data_dir, "cluster_data/*")))[-1]
  env_data = read_set_from_file(env_files, "hdf5")[:20]

  stars = Particles(len(env_data))
  stars.mass = env_data.mass
  stars.radius = env_data.radius
  stars.velocity = env_data.velocity
  stars.position = env_data.position
  stars.type = "STAR"
  stars.syst_id = -1
  
  planet_data = natsorted(glob.glob(os.path.join(data_dir, "acc3_vis4_cluster/*_planet.npz")))[:2]
  planet_no = np.asarray([int(f_.split("_planet")[0][74:]) for f_ in planet_data])
  
  pset = Particles()
  nsyst = 0
  stars_copy = stars.copy_to_memory()
  for i in range(len(planet_data)):
    nsyst+=1

    host = stars_copy[int(planet_no[i])]
    host.type = "STAR"
    host.radius = ZAMS_radius(host.mass)

    planet = Particle()
    load_data_planet = np.load(planet_data[i], allow_pickle=True)
    load_data_planet.keys()
    planet.mass = load_data_planet['Mc'][-1]+load_data_planet['Me'][-1] | units.MJupiter
    planet.type = "PLANET"
    planet.radius = (1|units.RJupiter)*(planet.mass.value_in(units.MJupiter))**(-0.02) #From Baron et al. 2023
    
    semimajor = load_data_planet['a'][-1] | units.au
    eccentricity = np.random.uniform(0, 0.05)
    inclination = np.arccos(1-2*np.random.uniform(0,1, 1)) | units.rad
    long_asc_node = np.random.uniform(0, 2*np.pi, 1) | units.rad
    true_anomaly = np.random.uniform(0, 2*np.pi, 1)
    arg_periapsis = np.random.uniform(0, 2*np.pi, 1) | units.rad

    phi = np.radians(random.uniform(0.0, 90.0))        #rotate under x
    theta0 = np.radians((random.normal(-90.0,90.0)))   #rotate under y
    theta_inclination = np.radians(random.normal(0, 1.0))
    theta = theta0 + theta_inclination
    psi = np.radians(random.uniform(0.0, 180.0))

    binary_set = new_binary_from_orbital_elements(
                    mass1=planet.mass, mass2=host.mass,
                    semimajor_axis=semimajor,
                    eccentricity=eccentricity,
                    inclination=inclination,
                    longitude_of_the_ascending_node=long_asc_node,
                    true_anomaly=true_anomaly,
                    argument_of_periapsis=arg_periapsis,
                    G=constants.G)
    planet.position = binary_set[0].position
    planet.velocity = binary_set[0].velocity

    planet.position, planet.velocity = rotate(planet.position, 
                                              planet.velocity, 
                                              phi, theta, psi)
    planet.position+=host.position
    planet.velocity+=host.velocity

    host.syst_id = nsyst
    planet.syst_id = nsyst
    pset.add_particle(host)
    pset.add_particle(planet)

    stars-=host

  isol = stars[stars.syst_id==-1]
  isol.radius = ZAMS_radius(isol.mass)
  pset.add_particles(isol)
  output_dir = "examples/realistic_cluster/initial_particles/init_particle_set"
  write_set_to_file(pset, output_dir, "amuse", close_file=True, overwrite_file=True)

load_particles()