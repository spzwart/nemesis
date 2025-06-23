from amuse.lab import new_plummer_model, new_kroupa_mass_distribution
from amuse.lab import write_set_to_file
from amuse.lab import units, nbody_system, constants
from amuse.ic import make_planets_oligarch
from amuse.ext.orbital_elements import orbital_elements

import numpy as np
import os


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

Nbodies = 500
Nchildren = 10

masses = new_kroupa_mass_distribution(Nbodies, mass_min=0.5 | units.MSun, mass_max=30 | units.MSun)
converter = nbody_system.nbody_to_si(masses.sum(), 1 | units.pc)
bodies = new_plummer_model(number_of_particles=Nbodies, convert_nbody=converter)
bodies.mass = masses
bodies.scale_to_standard(convert_nbody=converter, virial_ratio=0.5)
bodies.radius = ZAMS_radius(bodies.mass)
bodies.syst_id = -1

solar_systems = bodies[bodies.mass > 0.7 | units.MSun].random_sample(Nchildren)
for i, host in enumerate(solar_systems):
    host_star = make_planets_oligarch.new_system(star_mass=host.mass,
                                                 star_radius=host.radius,
                                                 disk_minumum_radius=1. | units.au,
                                                 disk_maximum_radius=50. | units.au,
                                                 disk_mass=0.005 | units.MSun,)
    planets = host_star.planets[0][:5]
    planets.remove_attribute_from_store("eccentricity")
    planets.remove_attribute_from_store("semimajor_axis")
    
    planets.position -= host_star.position
    planets.velocity -= host_star.velocity
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
        
        print(f"SMA: {ke[2].in_(units.au)}, Ecc: {ke[3]}")


output_dir = "examples/basic_cluster/initial_particles"
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

Run_ID = 0
write_set_to_file(
    bodies, 
    f"{output_dir}/nemesis_example_{Run_ID}.amuse", 
    "amuse",
    close_file=True, 
    overwrite_file=True
)