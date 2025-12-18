from amuse.lab import new_plummer_model, new_kroupa_mass_distribution
from amuse.lab import write_set_to_file
from amuse.lab import units, nbody_system, constants
from amuse.ic import make_planets_oligarch
from amuse.ext.orbital_elements import orbital_elements

import numpy as np
import os


def new_rotation_matrix_from_euler_angles(phi, theta, chi):
    """
    Rotation matrix for planetary system orientation
    Args:
        phi (float):    Rotation around x-axis
        theta (float):  Rotation around y-axis
        chi (float):    Rotation around z-axis
    Returns:
        np.ndarray:  Rotation matrix
    """
    cosp = np.cos(phi)
    sinp = np.sin(phi)
    cost = np.cos(theta)
    sint = np.sin(theta)
    cosc = np.cos(chi)
    sinc = np.sin(chi)
    return np.asarray([
        [cost*cosc, -cosp*sinc+sinp*sint*cosc,  sinp*sinc+cosp*sint*cosc], 
        [cost*sinc,  cosp*cosc+sinp*sint*sinc, -sinp*cosc+cosp*sint*sinc],
        [-sint,                     sinp*cost,                 cosp*cost]
        ])

def rotate(position, velocity, phi, theta, psi):
    """
    Rotate planetary system
    Args:
        position (Vector):  Position vector of the particle
        velocity (Vector):  Velocity vector of the particle
        phi (float):    Rotation around x-axis
        theta (float):  Rotation around y-axis
        psi (float):    Rotation around z-axis
    Returns:
        tuple:  Rotated position and velocity vectors
    """
    Runit = position.unit
    Vunit = velocity.unit
    matrix = new_rotation_matrix_from_euler_angles(phi, theta, psi)
    matrix = matrix
    return (
        np.dot(matrix, position.value_in(Runit)) | Runit,
        np.dot(matrix, velocity.value_in(Vunit)) | Vunit
        )
    
def orient_orbit(p, psi, theta, phi):
    """
    Orient planet orbit around the host star
    Args:
        p (Particle):  Particle
        psi (float):   Rotation around x-axis
        theta (float): Rotation around y-axis
        phi (float):   Rotation around z-axis
    Returns:
        tuple:  Rotated position and velocity vectors of the planet
    """
    pos = p.position
    vel = p.velocity
    
    pos, vel = rotate(pos, vel, 0, 0, psi)
    pos, vel = rotate(pos, vel, 0, theta, 0)
    pos, vel = rotate(pos, vel, phi, 0, 0)
    
    return pos, vel

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



Nbodies = 150
Nchildren = 5

masses = new_kroupa_mass_distribution(
  Nbodies, 
  mass_min=0.08 | units.MSun, 
  mass_max=30 | units.MSun
  )
converter = nbody_system.nbody_to_si(masses.sum(), 0.2 | units.pc)
bodies = new_plummer_model(
  number_of_particles=Nbodies, 
  convert_nbody=converter
  )

bodies.mass = masses
bodies.scale_to_standard(
  convert_nbody=converter, 
  virial_ratio=0.5
  )
bodies.radius = ZAMS_radius(bodies.mass)
bodies.syst_id = -1


### Initialise child systems
child_systems = bodies.random_sample(Nchildren)
for i, host in enumerate(child_systems):
    host_star = make_planets_oligarch.new_system(
      star_mass=host.mass,
      star_radius=host.radius,
      disk_minumum_radius=1. | units.au,
      disk_maximum_radius=100. | units.au,
      disk_mass=0.01 | units.MSun
      )
    planets = host_star.planets[0][:5]
    planets.remove_attribute_from_store("eccentricity")
    planets.remove_attribute_from_store("semimajor_axis")
    planets.position -= host_star.position
    planets.velocity -= host_star.velocity

    Ntotal = len(planets) + 1  # including host star
    
    # Random system rotation
    phi = np.radians(np.random.uniform(0.0, 90.0, 1)[0])  # x-plane rotation
    theta0 = np.radians((np.random.normal(-90.0, 90.0, 1)[0]))  # y-plane rotation
    theta_inclination = np.radians(np.random.normal(0, 1.0, Ntotal))
    theta_inclination[0] = 0
    theta = theta0 + theta_inclination
    psi = np.radians(np.random.uniform(0.0, 180.0, 1))[0]

    true_anomaly = np.random.uniform(0, 2*np.pi, Ntotal) | units.rad
    for ip, p in enumerate(planets):
      pos, vel = orient_orbit(p, psi, theta[ip], phi)
      p.position = pos
      p.velocity = vel
      p.position += host.position
      p.velocity += host.velocity
    
    host.syst_id = i+1
    planets.syst_id = i+1
    
    host.type = "HOST"
    planets.type = "PLANET"
    bodies.add_particles(planets)

### Sanity check: ensure all planets have low eccentricity
for id in np.unique(bodies.syst_id):
    if id > 0:
      system = bodies[bodies.syst_id == id]
      major_body = system[system.mass == system.mass.max()]
      minor_bodies = system - major_body
      
      for p in minor_bodies:
        bin_system = major_body + p
        ke = orbital_elements(bin_system, G=constants.G)
        assert ke[3] < 1e-5, "Eccentricity is too high!"

output_dir = "examples/basic_cluster/ICs"
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

Run_ID = 0
write_set_to_file(
    bodies, 
    f"{output_dir}/example{Run_ID}.amuse", 
    "amuse",
    close_file=True, 
    overwrite_file=True
)