"""
This script initialises a star cluster with planetary systems
and minor bodies to test the performance and accuracy of Nemesis.
"""


import numpy as np
import os
from numpy import random

from amuse.datamodel import Particles
from amuse.ext.protodisk import ProtoPlanetaryDisk
from amuse.ic import make_planets_oligarch
from amuse.lab import new_kroupa_mass_distribution, new_plummer_model
from amuse.lab import nbody_system, write_set_to_file, constants
from amuse.units import units


def get_disk_radius(host_mass) -> tuple:
    """
    Calculate disk radius based on host star mass.
    Inner radius defined as when Porb = 1 year.
    Outer radius defined as 117 * M^0.45 in astronomical units.
    Args:
        host_mass (float): Host star mass in solar masses.
    Returns:
        float: Disk radius in astronomical units.
    """
    torb = 1 | units.yr
    Rinner = ((constants.G * host_mass)/(4*np.pi**2) * torb**2)**(1/3)
    Router = 117 * host_mass.value_in(units.MSun)**0.45 | units.au
    return Rinner, Router


def new_rotation_matrix_from_euler_angles(phi, theta, chi) -> np.ndarray:
    """Rotation matrix for planetary system orientation"""
    cosp = np.cos(phi)
    sinp = np.sin(phi)
    cost = np.cos(theta)
    sint = np.sin(theta)
    cosc = np.cos(chi)
    sinc = np.sin(chi)
    return np.asarray([
        [cost*cosc, -cosp*sinc+sinp*sint*cosc, sinp*sinc+cosp*sint*cosc],
        [cost*sinc, cosp*cosc+sinp*sint*sinc, -sinp*cosc+cosp*sint*sinc],
        [-sint,  sinp*cost,  cosp*cost]
        ])


def rotate(position, velocity, phi, theta, psi) -> tuple:
    """Rotate planetary system"""
    Runit = position.unit
    Vunit = velocity.unit
    matrix = new_rotation_matrix_from_euler_angles(phi, theta, psi)
    matrix = matrix
    return (
        np.dot(matrix, position.value_in(Runit)) | Runit,
        np.dot(matrix, velocity.value_in(Vunit)) | Vunit
        )


def setup_cluster(Nstars, Rvir, Qvir, nchild) -> None:
    """
    Create cluster particle set
    Args:
        Nstars (int):  Number of stars
        Rvir (float):  Virial radius
        Qvir (float):  Virial ratio
        nchild (int):  Number of children systems
    """
    # Creating output directories
    configuration = os.path.join("tests", "cluster_test", "data")
    initial_set_dir = os.path.join(configuration, "ICs")
    if not os.path.exists(configuration):
        os.mkdir(configuration)
    if not os.path.exists(initial_set_dir):
        os.mkdir(initial_set_dir)

    masses = new_kroupa_mass_distribution(
        Nstars,
        mass_min=0.08 | units.MSun,
        mass_max=30. | units.MSun
        )
    converter = nbody_system.nbody_to_si(masses.sum(), Rvir)

    bodies = new_plummer_model(Nstars, convert_nbody=converter)
    bodies.mass = masses
    bodies.syst_id = -1
    bodies.type = "STAR"
    mask = (masses > 0.5 | units.MSun) & (masses < 2 | units.MSun)
    host_stars = bodies[mask].random_sample(int(nchild))

    particle_set = Particles()
    for syst_id, host in enumerate(host_stars):
        host.syst_id = syst_id + 1
        host.type = "HOST"

    converter = nbody_system.nbody_to_si(np.sum(bodies.mass), Rvir)
    bodies.scale_to_standard(convert_nbody=converter, virial_ratio=Qvir)
    for host in bodies[bodies.syst_id > 0]:
        min_disk_size, max_disk_size = get_disk_radius(host.mass)

        host_star = make_planets_oligarch.new_system(
            star_mass=host.mass,
            star_radius=host.radius,
            disk_minumum_radius=min_disk_size,
            disk_maximum_radius=max_disk_size,
            disk_mass=0.01*host.mass
        )
        planets = host_star.planets[0]
        Nplanets = min(len(planets), np.random.randint(1, 6))
        planets = planets.random_sample(int(Nplanets))
        planets.type = "PLANET"
        planets.syst_id = host.syst_id

        current_system = Particles()
        current_system.add_particle(host)
        current_system.add_particles(planets)

        print(f"System {syst_id + 1}", end=": ")
        print(f"Mhost= {host.mass.in_(units.MSun)}", end=", ")
        print(f"Np= {Nplanets}, Rdisk= {max_disk_size.in_(units.au)}")

        local_converter = nbody_system.nbody_to_si(host.mass, 1 | units.au)
        asteroids = ProtoPlanetaryDisk(
            NUM_ASTEROID,
            densitypower=1.5,
            radius_min=10.*min_disk_size.value_in(units.au),
            radius_max=max_disk_size.value_in(units.au),
            q_out=1, discfraction=0.01,
            convert_nbody=local_converter
            ).result
        asteroids.type = "ASTEROID"
        asteroids.syst_id = host.syst_id
        asteroids.mass = 0 | units.MSun
        asteroids.radius = 10 | units.km
        current_system.add_particle(asteroids)

        # Rotate the system by random angles to get different orientations
        phi = np.radians(random.uniform(0.0, 90.0, 1)[0])
        theta0 = np.radians((random.normal(-90.0, 90.0, 1)[0]))
        theta_inclination = np.radians(
            random.normal(0, 1.0, (1+Nplanets+NUM_ASTEROID))
            )
        theta_inclination[0] = 0
        theta = theta0 + theta_inclination
        psi = np.radians(random.uniform(0.0, 180.0, 1))[0]
        for i, minor in enumerate(current_system[1:]):
            pos = minor.position
            vel = minor.velocity

            pos, vel = rotate(pos, vel, 0, 0, psi)
            pos, vel = rotate(pos, vel, 0, theta[i], 0)
            pos, vel = rotate(pos, vel, phi, 0, 0)

            minor.position = pos
            minor.velocity = vel

        current_system[1:].position += host.position
        current_system[1:].velocity += host.velocity
        particle_set.add_particles(current_system)

    particle_set.remove_attribute_from_store("eccentricity")
    particle_set.remove_attribute_from_store("semimajor_axis")
    particle_set.remove_attribute_from_store("u")

    isol = bodies[bodies.syst_id == -1]
    particle_set.add_particles(isol)

    Nparticles = len(particle_set)
    Nstars = len(particle_set[particle_set.type == "STAR"])
    Nchild = particle_set.syst_id.max()

    print("!!! SAVING !!!")
    print(f"Total number of particles: {Nparticles}")
    print(f"Total number of stars: {Nstars}")
    print(f"Total number of planetary systems: {Nchild}")
    print(f"Total mass: {particle_set.mass.sum().in_(units.MSun)}")
    print(f"Virial radius: {Rvir.in_(units.pc)}")

    output_dir = os.path.join(initial_set_dir, "run_0")
    write_set_to_file(
        particle_set,
        output_dir,
        "amuse",
        close_file=True,
        overwrite_file=True
    )


NUM_ASTEROID = 200

# Orion-like cluster
Rvir = 0.5 | units.pc
Nstars = 2048
setup_cluster(
    Nstars=Nstars,
    Rvir=Rvir,
    Qvir=0.5,
    nchild=30
)
