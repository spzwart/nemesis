from amuse.lab import new_plummer_model, new_kroupa_mass_distribution
from amuse.lab import write_set_to_file
from amuse.lab import units, nbody_system, constants
from amuse.ic import make_planets_oligarch
from amuse.ext.orbital_elements import orbital_elements

import numpy as np
import os


def _get_plummer_density(particles, scale_radius, fraction=1):
    """
    Calculate Plummer density for each particle. Assume
    you want the density at the scale radius.
    Args:
        particles (Particles): Particle set
        scale_radius (float):  Plummer scale radius
        fraction (float):      Fraction of scale radius to calculate density
    """
    cluster_mass = particles.mass.sum()
    volume_at_scale = (4/3) * np.pi * scale_radius**3
    plummer_coeff = (1 + (fraction)**2)**(5/2)
    density = cluster_mass / (volume_at_scale * plummer_coeff)
    print("Plummer density:", density.in_(units.MSun / units.pc**3))


def ZAMS_radius(mass):
    """
    Define particle radius
    Args:
      mass (float): Stellar mass
    Returns:
      radius (float): Stellar radius
    """
    mass_sq = (mass.value_in(units.MSun))**2
    r_zams = pow(mass.value_in(units.MSun), 1.25) \
        * (0.1148 + 0.8604*mass_sq) / (0.04651 + mass_sq)
    return r_zams | units.RSun


def new_rotation_matrix_from_euler_angles(phi, theta, chi):
    """
    Rotation matrix for planetary system orientation
    Args:
        phi (float):   Rotation angle
        theta (float): Rotation angle
        chi (float):   Rotation angle
    Returns:
        matrix (array): Rotation matrix
    """
    cosp = np.cos(phi)
    sinp = np.sin(phi)
    cost = np.cos(theta)
    sint = np.sin(theta)
    cosc = np.cos(chi)
    sinc = np.sin(chi)
    return np.array([
        [cost*cosc, -cosp*sinc+sinp*sint*cosc, sinp*sinc+cosp*sint*cosc],
        [cost*sinc, cosp*cosc+sinp*sint*sinc, -sinp*cosc+cosp*sint*sinc],
        [-sint,  sinp*cost,  cosp*cost]
        ])


def rotate(position, velocity, phi, theta, psi):
    """
    Rotate planetary system
    Args:
        position (array): Position vector
        velocity (array): Velocity vector
        phi (float):      Rotation angle
        theta (float):    Rotation angle
        psi (float):      Rotation angle
    Returns:
        matrix (array): Rotated position and velocity vector
    """
    Runit = position.unit
    Vunit = velocity.unit
    matrix = new_rotation_matrix_from_euler_angles(phi, theta, psi)
    return (
        np.dot(matrix, position.value_in(Runit)) | Runit,
        np.dot(matrix, velocity.value_in(Vunit)) | Vunit
        )


# Create star cluster with planetary systems
Nparents = 200
Nchildren = 10
rvir = 0.5 | units.pc

masses = new_kroupa_mass_distribution(
  Nparents,
  mass_min=0.5 | units.MSun,
  mass_max=30 | units.MSun
  )
converter = nbody_system.nbody_to_si(masses.sum(), rvir)

bodies = new_plummer_model(
    number_of_particles=Nparents,
    convert_nbody=converter
    )
bodies.mass = masses
bodies.scale_to_standard(convert_nbody=converter, virial_ratio=0.5)
bodies.radius = ZAMS_radius(bodies.mass)
bodies.syst_id = -1
_get_plummer_density(bodies, rvir)

solar_systems = bodies.random_sample(Nchildren)
for i, host in enumerate(solar_systems):
    print(f"\rGenerating planets for host star {i+1}", end="  ", flush=True)
    host_star = make_planets_oligarch.new_system(
      star_mass=host.mass,
      star_radius=host.radius,
      disk_minumum_radius=1. | units.au,
      disk_maximum_radius=50. | units.au,
      disk_mass=0.005 | units.MSun,
    )
    planets = host_star.planets[0][:5]
    planets.remove_attribute_from_store("eccentricity")
    planets.remove_attribute_from_store("semimajor_axis")
    planets.position -= host_star.position
    planets.velocity -= host_star.velocity

    # Randomly rotate planetary system
    phi = np.radians(np.random.uniform(0.0, 90.0, 1)[0])
    theta0 = np.radians((np.random.normal(-90.0, 90.0, 1)[0]))
    theta_inclination = np.radians(np.random.normal(0, 1.0, (1+len(planets))))
    theta_inclination[0] = 0
    theta = theta0 + theta_inclination
    psi = np.radians(np.random.uniform(0.0, 180.0, 1))[0]
    for j, p in enumerate(planets):
        pos = p.position
        vel = p.velocity

        pos, vel = rotate(pos, vel, 0, 0, psi)
        pos, vel = rotate(pos, vel, 0, theta[j], 0)
        pos, vel = rotate(pos, vel, phi, 0, 0)

        p.position = pos
        p.velocity = vel

    planets.position += host.position
    planets.velocity += host.velocity

    host.syst_id = i+1
    planets.syst_id = i+1

    host.type = "HOST"
    planets.type = "PLANET"
    bodies.add_particles(planets)


for id in np.unique(bodies.syst_id):
    if id > 0:
        system = bodies[bodies.syst_id == id]
        major_body = system[system.mass == system.mass.max()]
        minor_bodies = system - major_body

        for p in minor_bodies:
            bin_system = major_body + p
            ke = orbital_elements(bin_system, G=constants.G)

            # Check that planets are bound
            assert ke[3] < 0.1
            assert ke[2] > 0.0 | units.au

output_dir = "examples/basic_cluster/ICs"
if not os.path.exists(output_dir):
    os.makedirs("examples/basic_cluster", exist_ok=True)
    os.mkdir(output_dir)

Run_ID = 0
write_set_to_file(
    bodies,
    f"{output_dir}/nemesis_example_{Run_ID}.amuse",
    "amuse",
    close_file=True,
    overwrite_file=True
)
