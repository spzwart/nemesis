import numpy as np

from amuse.ext.orbital_elements import generate_binaries
from amuse.ic.kingmodel import new_king_model
from amuse.lab import new_salpeter_mass_distribution
from amuse.lab import Particles, constants, units, nbody_system


class MW_SMBH(object):
    """Class which defines the central SMBH"""
    def __init__(self, mass,
                 position=[0,0,0] | units.pc,
                 velocity=[0,0,0] | units.kms):
        self.mass = mass
        self.position = position
        self.velocity = velocity
        self.bh_rad = (2*constants.G*mass)/(constants.c**2)

class ClusterInitialise(object):
    """Class to initialise the SBH particles"""
    def ZAMS_radius(self, mass):
        """Define particle radius"""
        mass_sq = (mass.value_in(units.MSun))**2
        r_zams = pow(mass.value_in(units.MSun), 1.25) \
                * (0.1148 + 0.8604*mass_sq) / (0.04651 + mass_sq)
        return r_zams | units.RSun

    def isco_radius(self, mass):
        """Set the SBH radius based on the Schwarzschild radius"""
        return 3*(2*constants.G*mass)/(constants.c**2)

    def coll_radius(self, radius):
        """Set the collision radius (10x the rISCO)."""
        factor = 10
        return factor*radius

    def star_mass(self, nStar):
        """Set stellar particle masses"""
        alpha = -1.35   # arXiv:0305423, 0810.2723, 2403.040321
        mass_min = 0.5 | units.MSun  # arXiv:0305423, 
        mass_max = 100 | units.MSun  # arXiv:1505.05473
        return new_salpeter_mass_distribution(nStar, mass_min, 
                                              mass_max, alpha
                                              ) 

    def init_cluster(self, mass, rvir):
        """Initialise the first SBH population. \n
        Inputs:
        nSBH:  The number of SBH particles
        nIMBH:  The number of IMBH particles
        fStar:  Fraction of cluster being stars
        mass:  Mass of SMBH
        star_mdist:  Mass choice for stars (Equal || IMF)
        rvir:  Cluster initial virial radius
        """
        SMBH_parti = MW_SMBH(mass)
        nStar = 15000
        TOTAL_MASS = 30000 | units.MSun
        code_conv = nbody_system.nbody_to_si(TOTAL_MASS, rvir)

        particles = Particles(1)
        particles[0].type = "smbh"
        particles[0].position = [0, 0, 0] | units.pc
        particles[0].velocity = [0, 0, 0] | units.kms
        particles[0].mass = SMBH_parti.mass
        particles[0].radius = self.isco_radius(particles[0].mass)
        particles[0].collision_radius = self.coll_radius(particles[0].radius)
        
        stars = new_king_model(nStar, W0=10, convert_nbody=code_conv)
        stars.type = "star"
        stars.mass = self.star_mass(nStar)
        stars.radius = self.ZAMS_radius(stars.mass)
        stars.collision_radius = stars.radius
        particles.add_particles(stars)
        
        code_conv = nbody_system.nbody_to_si(stars.mass.sum(), rvir)
        particles[1:].scale_to_standard(convert_nbody=code_conv, 
                                        virial_ratio=0.5
                                        )
        particles.Nej = 0
        particles.coll_events = 0
        particles.key_tracker = particles.key
        particles.move_to_center()
        
        return particles