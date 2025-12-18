from amuse.ext.orbital_elements import generate_binaries, orbital_elements
from amuse.lab import units, Particles, constants, write_set_to_file

import os

ZKL_particles = Particles()

#### Conditions taken from arXiv:1107.2414
# Setup inner binary
inclination = 0
particle_set = generate_binaries(
    1.0 | units.MSun, 
    1.0 | units.MJupiter,
    6.0 | units.au, 
    eccentricity=0.001, 
    inclination=inclination,
    G=constants.G
    )
pos = particle_set[0].position
vel = particle_set[0].velocity

for p in particle_set:
    ZKL_particles.add_particle(p)

# Setup outer binary
minner = ZKL_particles.mass.sum()
outer_binary = generate_binaries(
    1.0 | units.MSun,
    40. | units.MJupiter,
    100. | units.au,
    eccentricity=.6,
    inclination=65 | units.deg,
    true_anomaly=180 | units.deg,
    G=constants.G
    )
outer_binary[1].position -= outer_binary[0].position + pos
outer_binary[1].velocity -= outer_binary[0].velocity + vel


ZKL_particles.add_particle(outer_binary[1])
for p in ZKL_particles[1:]:
    bin_sys = Particles()
    bin_sys.add_particle(ZKL_particles[0])
    bin_sys.add_particle(p)
    kepler = orbital_elements(bin_sys, G=constants.G)
    print(f"Mass 1: {kepler[0].in_(units.MSun)}, Mass 2: {kepler[1].in_(units.MJupiter)}", end=", ")
    print(f"SMA: {kepler[2].in_(units.au)}, Ecc.: {kepler[3]}, Inc.: {kepler[5].in_(units.deg)}")

ZKL_particles.move_to_center()
ZKL_particles[ZKL_particles.mass.argmax()].name = "Sun"
ZKL_particles[ZKL_particles.mass < ZKL_particles.mass].name = "PLANET"
ZKL_particles.syst_id = 1
ZKL_particles.radius = 0 | units.RSun


### Create output directories
fname = f"LZK_JMO.amuse"
output_dir = os.path.join("tests", "ZKL_test", "data", "ICs", fname)
os.makedirs(os.path.dirname(output_dir), exist_ok=True)
write_set_to_file(
    ZKL_particles, 
    output_dir, 
    "amuse", 
    overwrite_file=True,
)