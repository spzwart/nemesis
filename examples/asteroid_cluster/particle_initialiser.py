import glob
import numpy as np
import os
from numpy import random

from amuse.community.fractalcluster.interface import new_fractal_cluster_model
from amuse.datamodel import Particles
from amuse.ext.orbital_elements import orbital_elements
from amuse.ext.protodisk import ProtoPlanetaryDisk
from amuse.ic import make_planets_oligarch
from amuse.lab import new_kroupa_mass_distribution, new_plummer_model
from amuse.lab import nbody_system, write_set_to_file, constants 
from amuse.units import units


def planet_radius(planet_mass) -> float:
    """
    Define planet radius (arXiv:2311.12593)
    
    Args:
        plant_mass (Float):  Mass of the planet
    Returns:
        Float:  Planet radius
    """
    mass_in_earth = planet_mass.value_in(units.MEarth)
    
    if planet_mass < (7.8 | units.MEarth):
        radius = (1. | units.REarth)*(mass_in_earth)**0.41
        return radius
    
    elif planet_mass < (125. | units.MEarth):
        radius = (0.55 | units.REarth)*(mass_in_earth)**0.65
        return radius
    
    radius = (14.3 | units.REarth)*(mass_in_earth)**(-0.02) 
    return radius

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
    return np.asarray([[cost*cosc, -cosp*sinc+sinp*sint*cosc, sinp*sinc+cosp*sint*cosc], 
                       [cost*sinc, cosp*cosc+sinp*sint*sinc, -sinp*cosc+cosp*sint*sinc],
                       [-sint,  sinp*cost,  cosp*cost]])

def rotate(position, velocity, phi, theta, psi):
    """Rotate planetary system"""
    Runit = position.unit
    Vunit = velocity.unit
    matrix = new_rotation_matrix_from_euler_angles(phi, theta, psi)
    matrix = matrix
    return (np.dot(matrix, position.value_in(Runit)) | Runit,
            np.dot(matrix, velocity.value_in(Vunit)) | Vunit)

def setup_cluster(Nstars, cluster_type, Rvir, Qvir):
    """
    Create cluster particle set
    
    Args:
        Nstars (int):  Number of stars
        cluster_type (str):  Type of cluster
        Rvir (float):  Virial radius
        Qvir (float):  Virial ratio
    """
    NUM_ASTEROID = 1000
    MODEL_NAME = f"OrionLike_{cluster_type}_Qvir_{Qvir}"
    MAX_HOST_MASS = 3 | units.MSun 
    MIN_HOST_MASS = 0.2 | units.MSun
                 
    # Creating output directories
    configuration = os.path.join("asteroid_cluster", MODEL_NAME)
    initial_set_dir = os.path.join(configuration, "initial_particles")
    if not os.path.exists("asteroid_cluster"):
        os.mkdir("asteroid_cluster")
    if not os.path.exists(configuration):
        os.mkdir(configuration)
    if not os.path.exists(initial_set_dir):
        os.mkdir(initial_set_dir)
    N_CONFIG = len(glob.glob(initial_set_dir+"/*"))
    
    masses = new_kroupa_mass_distribution(Nstars, 
                                          mass_min=0.08 | units.MSun, 
                                          mass_max=30. | units.MSun)
    converter = nbody_system.nbody_to_si(masses.sum(), Rvir)
    if cluster_type.lower() == "plummer":
        bodies = new_plummer_model(Nstars, convert_nbody=converter)
    elif cluster_type.lower() == "fractal":
        bodies = new_fractal_cluster_model(Nstars, 
                                           fractal_dimension=1.6, 
                                           convert_nbody=converter)
    bodies.mass = masses
    bodies.syst_id = -1
    bodies.type = "STAR"
    bodies.radius = ZAMS_radius(bodies.mass)
    
    host_masses = (bodies.mass < MAX_HOST_MASS) & (bodies.mass > MIN_HOST_MASS)
    host_stars = bodies[host_masses]
    
    particle_set = Particles()
    nsyst = 0
    for host in host_stars:
        nsyst += 1
        host.radius = ZAMS_radius(host.mass)
        host.syst_id = nsyst
        host.type = "HOST"
        
    converter = nbody_system.nbody_to_si(np.sum(bodies.mass), Rvir)
    bodies.scale_to_standard(convert_nbody=converter, virial_ratio=Qvir)
    for host in bodies[bodies.syst_id >= 1300]:
        disk_mass = np.random.uniform(0.005, 0.03) | units.MSun
        nplanets = np.random.randint(2, 6)
        if host.mass < 0.7 | units.MSun:
            nplanets = np.random.randint(1, 4)
        
        print(f"System {host.syst_id}: Mhost= {host.mass.in_(units.MSun)}, Np={nplanets}")
        host_star = make_planets_oligarch.new_system(star_mass=host.mass,
                                                     star_radius=host.radius,
                                                     disk_minumum_radius=1. | units.au,
                                                     disk_maximum_radius=100. | units.au,
                                                     disk_mass=disk_mass)
        planets = host_star.planets[0]
        planets = planets.random_sample(nplanets)
        planets.type = "PLANET"
        planets.syst_id = host.syst_id
        particle_set.add_particle(host)
        particle_set.add_particle(planets)
        
        local_converter = nbody_system.nbody_to_si(host.mass, 1|units.au)
        asteroids = ProtoPlanetaryDisk(NUM_ASTEROID, densitypower=1.5, 
                                       Rmin=1, Rmax=100, q_out=1, 
                                       discfraction=0.01,
                                       convert_nbody=local_converter
                                       ).result
        asteroids.type = "ASTEROID"
        asteroids.syst_id = host.syst_id
        asteroids.mass = 0 | units.MSun
        asteroids.radius = 10 | units.km
        particle_set.add_particle(asteroids)

        # Rotate system
        phi = np.radians(random.uniform(0.0, 90.0))
        theta0 = np.radians((random.normal(-90.0,90.0)))
        theta_inclination = np.radians(random.normal(0, 1.0))
        theta = theta0 + theta_inclination
        psi = np.radians(random.uniform(0.0, 180.0))

        # Rotate planets and asteroids
        current_system = particle_set[particle_set.syst_id == host.syst_id]
        current_system[1:].position += current_system[0].position
        current_system[1:].velocity += current_system[0].velocity
        for body in current_system:
            body.position, body.velocity = rotate(body.position, 
                                                  body.velocity, 
                                                  phi, theta, psi)
        
        bodies -= host
        
    particle_set.remove_attribute_from_store("eccentricity")
    particle_set.remove_attribute_from_store("semimajor_axis")
    particle_set.remove_attribute_from_store("u")
    
    isol = bodies[bodies.syst_id == -1]
    isol.radius = ZAMS_radius(isol.mass)
    particle_set.add_particles(isol)
    
    print("!!! SAVING !!!")
    print(f"Total number of particles: {len(particle_set)}, Nsyst: {nsyst}")
    output_dir = os.path.join(initial_set_dir, "run_"+str(N_CONFIG))
    write_set_to_file(
        particle_set, output_dir, "amuse", 
        close_file=True, overwrite_file=True
    )
    nsyst_run = 1
    return nsyst_run

# Orion-like cluster
Rvir = 0.5 | units.pc
Qvir = [0.25, 0.5, 0.75]
Nstars = 2500
cluster_type = ["Plummer", "Fractal"]

for cluster in cluster_type:
    for q in Qvir:
        succesful_conds = 0
        while succesful_conds < 5:
            print("...Attempt...")
            nsyst = setup_cluster(Nstars, cluster, Rvir, q)
            succesful_conds += nsyst