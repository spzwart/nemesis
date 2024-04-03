import glob
import numpy as np
import os
from numpy import random

from amuse.community.fractalcluster.interface import new_fractal_cluster_model
from amuse.datamodel import Particles
from amuse.ext import solarsystem
from amuse.lab import new_kroupa_mass_distribution, new_plummer_model
from amuse.lab import nbody_system, write_set_to_file 
from amuse.units import units

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

def setup_cluster(stellar_pop, cluster_type, virial_radius, virial_ratio):
    """Create cluster particle set"""
    MIN_MASS = 0.08 | units.MSun
    MAX_MASS = 30 | units.MSun
    SUN_MASS = 1 | units.MSun
    MASS_TOLERANCE = 0.2 | units.MSun
    SIM_DIR = "examples/ejecting_suns"
    MODEL_NAME = "N"+str(stellar_pop) + "_model"+str(cluster_type) \
                 + "_rvir" + str(virial_radius.value_in(units.pc)) \
                 + "pc_vir_ratio_"+str(virial_ratio)
                 
    # Creating output directories
    sim_data = os.path.join(SIM_DIR, "sim_data")
    configuration = os.path.join(sim_data, MODEL_NAME)
    initial_set_dir = os.path.join(configuration, "initial_set")
    if not os.path.exists(sim_data):
        os.mkdir(sim_data)
    if not os.path.exists(configuration):
        os.mkdir(configuration)
    if not os.path.exists(initial_set_dir):
        os.mkdir(initial_set_dir)
    N_CONFIG = len(glob.glob(initial_set_dir+"/*"))
    
    converter = nbody_system.nbody_to_si(stellar_pop*(1|units.MSun), virial_radius)
    masses = new_kroupa_mass_distribution(stellar_pop, MIN_MASS, MAX_MASS)
    if cluster_type.lower() == "plummer":
        bodies = new_plummer_model(stellar_pop, convert_nbody=converter)
    elif cluster_type.lower() == "fractal":
        bodies = new_fractal_cluster_model(stellar_pop, 
                                           fractal_dimension=1.6, 
                                           convert_nbody=converter
                                           )
    bodies.mass = masses
    bodies.syst_id = -1
    bodies.type = "STAR"
    
    particle_set = Particles()
    nsyst = 0
    for host in bodies[abs(bodies.mass-SUN_MASS) < MASS_TOLERANCE]:
        nsyst += 1
        host.mass = 1 | units.MSun
        host.radius = ZAMS_radius(host.mass)
        host.syst_id = nsyst
        host.name = "HOST"
        host.type = "HOST"
        
    bodies.scale_to_standard(convert_nbody=converter, virial_ratio=virial_ratio)
    for host in bodies[bodies.syst_id >= 0]:
        planets = solarsystem.new_solar_system()
        orb_planets = planets[3:-1]
        orb_planets.type = "PLANET"

        # Rotate system
        phi = np.radians(random.uniform(0.0, 90.0))
        theta0 = np.radians((random.normal(-90.0,90.0)))
        theta_inclination = np.radians(random.normal(0, 1.0))
        theta = theta0 + theta_inclination
        psi = np.radians(random.uniform(0.0, 180.0))

        # Form planetary system
        for planet in orb_planets:
          planet.position, planet.velocity = rotate(planet.position, 
                                                    planet.velocity, 
                                                    phi, theta, psi
                                                    )
          planet.position += host.position
          planet.velocity += host.velocity

        orb_planets.syst_id = host.syst_id
        particle_set.add_particle(host)
        particle_set.add_particle(orb_planets)

        bodies -= host

    isol = bodies[bodies.syst_id == -1]
    isol.radius = ZAMS_radius(isol.mass)
    particle_set.add_particles(isol)
    
    if nsyst == 10:
      output_dir = os.path.join(initial_set_dir, "run_"+str(N_CONFIG))
      print(output_dir)
      write_set_to_file(particle_set, output_dir, "amuse", 
                        close_file=True, overwrite_file=True
                        )
      nsyst_run = 1
      return nsyst_run
    
    nsyst_run = 0
    return nsyst_run
    
cluster_type = ["Plummer", "Fractal"]
Rvir = [0.5 | units.pc, 1 | units.pc]
Ratio_vir = [0.25, 0.5, 1]
Nstars = [100, 1000]
succesful_conds = 0
while succesful_conds < 10:
    print("...Attempt...")
    nsyst = setup_cluster(stellar_pop=Nstars[0],
                          cluster_type=cluster_type[1], 
                          virial_radius=Rvir[0],
                          virial_ratio=Ratio_vir[0]
                          )
    succesful_conds += nsyst