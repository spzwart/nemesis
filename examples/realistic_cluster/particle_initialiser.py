import glob
import numpy as np
import os
from natsort import natsorted
from numpy import random

from amuse.datamodel import Particles, Particle
from amuse.ext.orbital_elements import orbital_elements_from_binary
from amuse.ext.protodisk import ProtoPlanetaryDisk
from amuse.ext import solarsystem
from amuse.ext.orbital_elements import new_binary_from_orbital_elements
from amuse.lab import nbody_system, read_set_from_file, write_set_to_file
from amuse.units import units, constants

def ZAMS_radius(mass):
    """Define stellar radius at ZAMS"""
    mass = mass.value_in(units.MSun)
    mass_sq = (mass)**2
    r_zams = mass**1.25*(0.1148+0.8604*mass_sq)/(0.04651+mass_sq)
    return r_zams | units.RSun

def new_rotation_matrix_from_euler_angles(phi, theta, chi):
    """Rotation matrix for planetary system orientation"""
    cosp = np.cos(phi)
    sinp = np.sin(phi)
    cost = np.cos(theta)
    sint = np.sin(theta)
    cosc = np.cos(chi)
    sinc = np.sin(chi)
    return np.array([[cost*cosc, -cosp*sinc+sinp*sint*cosc, sinp*sinc+cosp*sint*cosc], 
                    [cost*sinc, cosp*cosc+sinp*sint*sinc, -sinp*cosc+cosp*sint*sinc],
                    [-sint,  sinp*cost,  cosp*cost]])

def rotate(position, velocity, phi, theta, psi):
    """Rotate planetary system"""
    Runit = position.unit
    Vunit = velocity.unit
    matrix = new_rotation_matrix_from_euler_angles(phi, theta, psi)
    return (np.dot(matrix, position.value_in(Runit)) | Runit,
            np.dot(matrix, velocity.value_in(Vunit)) | Vunit)

def load_particles(ss_equiv):
    """Load particles. Creates a single-planet planetary system."""
    if (ss_equiv):
      NUM_ASTEROID = 1000
      ss = solarsystem.new_solar_system()[:-1]

      for p in ss[1:]:
        bin_sys = Particles()
        bin_sys.add_particle(p)
        bin_sys.add_particle(ss[0])
        kepler_elements = orbital_elements_from_binary(bin_sys, 
                                                      G=constants.G
                                                      )
        period = np.sqrt((4*np.pi**2)/(constants.G*ss[0].mass)*abs(kepler_elements[2])**3)
        p.period = period
    
    # Load cluster data (Wilhem & Portegies Zwart (in works))
    data_dir = os.path.join("examples", "realistic_cluster", 
                            "initial_particles", "data_files",
                            "config_2_vm")
    cluster_data_files = os.path.join(data_dir, "cluster_data/*")
    env_files = natsorted(glob.glob(cluster_data_files))[-1]
    env_data = read_set_from_file(env_files, "hdf5")
    
    stars = Particles(len(env_data))
    stars.mass = env_data.mass
    stars.radius = env_data.radius
    stars.velocity = env_data.velocity
    stars.position = env_data.position
    stars.type = "STAR"
    stars.syst_id = -1

    # Load planet data (Huang & Portegies Zwart (in works))
    planet_data_files = os.path.join(data_dir, "acc3_vis4_cluster/*_planet.npz")
    planet_data = natsorted(glob.glob(planet_data_files))
    planet_no = np.asarray([int(f_.split("_planet")[0][-6:]) for f_ in planet_data])
    
    stars_copy = stars.copy_to_memory()
    particle_set = Particles()
    nsyst = 0
    
    for i in range(len(planet_data)):
      iplanet_df = np.load(planet_data[i], allow_pickle=True)
      semimajor = iplanet_df['a'][-1] | units.au
      mass = iplanet_df['Mc'][-1] + iplanet_df['Me'][-1] | units.MEarth
      
      host = stars_copy[int(planet_no[i])]
      host.mass = iplanet_df['star_mass'] | units.MSun
      
      period = np.sqrt((4*np.pi**2)/(constants.G*host.mass)*abs(semimajor)**3)[0]
      if (ss_equiv):
        nn_per_idx = abs((period - ss.period)).argmin()
        nn_mass_idx = abs((mass - ss.mass)).argmin()
        
        if abs(host.mass - ss[0].mass) <= (0.2 | units.MSun) \
          and nn_per_idx == nn_mass_idx:  # Extract Solar System Like
          nsyst += 1
          print(ss.name[nn_per_idx])
          new_ss = solarsystem.new_solar_system()[:-1]
          solar_system = new_ss[0]
          solar_system += new_ss[3:]
          solar_system[0].type ="HOST"
          solar_system[1:].type ="PLANET"
          
          # Rotate system
          phi = np.radians(random.uniform(0.0, 90.0))
          theta0 = np.radians((random.normal(-90.0,90.0)))
          theta_inclination = np.radians(random.normal(0, 1.0))
          theta = theta0 + theta_inclination
          psi = np.radians(random.uniform(0.0, 180.0))
          
          local_converter = nbody_system.nbody_to_si(host.mass, 1|units.au)
          asteroids = ProtoPlanetaryDisk(NUM_ASTEROID, densitypower=1.5, 
                                        Rmin=1, Rmax=100, q_out=1, 
                                        discfraction=0.01,
                                        convert_nbody=local_converter
                                        ).result
          asteroids.type = "ASTEROID"
          asteroids.mass = 0 | units.MSun
          asteroids.radius = 10 | units.km
          solar_system.add_particles(asteroids)
          
          for p in solar_system:
            p.position, p.velocity = rotate(p.position, 
                                            p.velocity, 
                                            phi, theta, psi
                                            )
          solar_system.position += host.position
          solar_system.velocity += host.velocity
          solar_system.syst_id = nsyst
          particle_set.add_particle(solar_system)
          
          stars -= host
          
      else:
        if period >= 1 | units.yr:  # Neglect tight orbits
          host.radius = ZAMS_radius(host.mass)
          host.type = "STAR"
          nsyst += 1

          planet = Particle()
          iplanet_df.keys()
          planet.mass = mass
          planet.type = "PLANET"

          # Set planet radius using arXiv:2311.12593
          mass_in_earth = planet.mass.value_in(units.MEarth)
          if planet.mass < (7.8|units.MEarth):
            planet.radius = (1|units.REarth)*(mass_in_earth)**0.41
          elif planet.mass < (125|units.MEarth):
            planet.radius = (0.55|units.REarth)*(mass_in_earth)**0.65
          else:
            planet.radius = (14.3|units.REarth)*(mass_in_earth)**(-0.02) 

          # Initialise planetary system orbital orientation
          eccentricity = 0
          inclination = np.arccos(1-2*np.random.uniform(0,1, 1)) | units.rad
          long_asc_node = np.random.uniform(0, 2*np.pi, 1) | units.rad
          true_anomaly = np.random.uniform(0, 2*np.pi, 1)
          arg_periapsis = np.random.uniform(0, 2*np.pi, 1) | units.rad

          # Rotate system
          phi = np.radians(random.uniform(0.0, 90.0))
          theta0 = np.radians((random.normal(-90.0,90.0)))
          theta_inclination = np.radians(random.normal(0, 1.0))
          theta = theta0 + theta_inclination
          psi = np.radians(random.uniform(0.0, 180.0))

          # Form planetary system
          binary_set = new_binary_from_orbital_elements(
                          mass1=planet.mass, 
                          mass2=host.mass,
                          semimajor_axis=semimajor,
                          eccentricity=eccentricity,
                          inclination=inclination,
                          longitude_of_the_ascending_node=long_asc_node,
                          true_anomaly=true_anomaly,
                          argument_of_periapsis=arg_periapsis,
                          G=constants.G
                          )
          planet.position = binary_set[0].position
          planet.velocity = binary_set[0].velocity
          planet.position, planet.velocity = rotate(planet.position, 
                                                    planet.velocity, 
                                                    phi, theta, psi
                                                    )
          planet.position += host.position
          planet.velocity += host.velocity

          host.syst_id = nsyst
          planet.syst_id = nsyst
          particle_set.add_particle(host)
          particle_set.add_particle(planet)

          stars -= host

    isol = stars[stars.syst_id == -1]
    isol.radius = ZAMS_radius(isol.mass)
    particle_set.add_particles(isol)
    
    
    """import matplotlib.pyplot as plt
    ax = plt.figure().add_subplot(projection='3d')
    ax.scatter(particle_set.x.value_in(units.au),
               particle_set.y.value_in(units.au),
               particle_set.z.value_in(units.au)
               )
    
    
    ax.scatter(particle_set[particle_set.syst_id > 0].x.value_in(units.au),
                particle_set[particle_set.syst_id > 0].y.value_in(units.au),
                particle_set[particle_set.syst_id > 0].z.value_in(units.au),
                s=100, color="red"
                )
    ax.scatter(particle_set[(particle_set.syst_id > 0) & (particle_set.mass > 0 | units.kg)].x.value_in(units.au),
                particle_set[(particle_set.syst_id > 0) & (particle_set.mass > 0 | units.kg)].y.value_in(units.au),
                particle_set[(particle_set.syst_id > 0) & (particle_set.mass > 0 | units.kg)].z.value_in(units.au),
                s=200, color="cyan"
                )
    plt.show()"""
    
    output_dir = os.path.join("examples", "realistic_cluster", 
                              "initial_particles", "init_particle_set"
                              )
    write_set_to_file(particle_set, output_dir, "amuse", 
                      close_file=True, overwrite_file=True
                      )
    print("Total #Planetary Systems: ", nsyst, end="")
    print("#Isolated: ", len(particle_set)-2*nsyst)

load_particles(ss_equiv=True)
  