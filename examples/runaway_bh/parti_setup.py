import numpy as np

from amuse.ext.orbital_elements import generate_binaries
from amuse.ic.kingmodel import new_king_model
from amuse.lab import units, nbody_system
from amuse.lab import new_salpeter_mass_distribution
from amuse.lab import Particles
from amuse.lab import constants
from amuse.lab import new_kroupa_mass_distribution


class MW_SMBH(object):
    """Class which defines the central SMBH"""
    def __init__(self, mass,
                 position=[0,0,0] | units.parsec,
                 velocity=[0,0,0] | units.kms):
                 
        self.mass = mass
        self.position = position
        self.velocity = velocity
        self.bh_rad = (2*constants.G*mass)/(constants.c**2)

class IMBH_init(object):
    """Class to initialise the IMBH particles"""
    def __init__(self):
        self.imbh_mdist = 1000 | units.MSun

    def ZAMS_radius(self, mass):
        """Define particle radius"""

        mass_sq = (mass.value_in(units.MSun))**2
        r_zams = pow(mass.value_in(units.MSun), 1.25) \
                *(0.1148 + 0.8604*mass_sq) / (0.04651 + mass_sq)
        return r_zams | units.RSun

    def IMBH_radius(self, mass):
        """Function which sets the IMBH radius based on the Schwarzschild radius"""
        return (2*constants.G*mass)/(constants.c**2)

    def coll_radius(self, radius):
        """Function which sets the collision radius (the rISCO)."""
        return 3*radius

    def star_mass(self, nStar, star_mdist):
        """Set stellar particle masses using Kroupa function"""
        if star_mdist.lower()=="equal":
            pset = Particles(nStar)
            pset.mass = 1 | units.MSun
            return pset.mass
        else:
            return new_kroupa_mass_distribution(nStar, 
                                    0.08 | units.MSun, 
                                    100 | units.MSun)

    def IMBH_first(self, init_parti, fStar, mass,
                   imbh_mdist, star_mdist, bin_star, 
                   rvir):
        """
        Function to initialise the first IMBH population.
        The first particle forms the center of the cluster

        Inputs:
        init_parti:   The number of IMBH particles you wish to simulate
        fStar:        Fraction of cluster being stars
        mass:    Mass of SMBH
        imbh_mdist:   Mass choice for imbhs (Equal || Power-law)
        star_mdist:   Mass choice for stars (Equal || Kroupa)
        bin_star:     Flag to know if initialising binary star-imbh systems
        rvir:         Cluster initial virial radius
        """

        SMBH_parti = MW_SMBH(mass)
        if not (bin_star):
          nStar = int(np.floor(init_parti*fStar))
          init_parti += int(np.floor(init_parti*fStar))
        else:
          nStar = int(init_parti)

        if imbh_mdist.lower()=="equal":
            ibhmass = 300 | units.MSun
        else:
            ibhmass = new_salpeter_mass_distribution(init_parti, 
                                                     100 | units.MSun, 
                                                     1000 | units.MSun, 
                                                     alpha=-2.0)
        mtot = nStar*np.mean(self.star_mass(1, star_mdist))
        mtot += (init_parti-nStar-1)*ibhmass 
        self.code_conv = nbody_system.nbody_to_si(mtot, rvir)

        particles = Particles(1)
        particles[0].type = "smbh"
        IMBH = new_king_model(init_parti, W0=10, 
                              convert_nbody=self.code_conv)

        IMBH.mass = ibhmass
        IMBH.type = "imbh"
        
        particles.add_particles(IMBH)
        particles[0].position = IMBH.center_of_mass()
        particles[0].velocity = IMBH.center_of_mass_velocity()
        particles[0].mass = SMBH_parti.mass
        particles[0].radius = self.IMBH_radius(particles[0].mass)
        particles[0].collision_radius = self.coll_radius(particles[0].radius)
        code_conv = nbody_system.nbody_to_si(particles.mass.sum(), rvir)
        if not (bin_star):
            stars = particles[1:].random_sample(nStar)
            stars.type = "star"
            stars.mass = self.star_mass(nStar, star_mdist)
            code_conv = nbody_system.nbody_to_si(particles[1:].mass.sum(), rvir)
            IMBH = particles[1:]-stars
            #IMBH.position *= 0.2
            particles[1:].scale_to_standard(convert_nbody=code_conv, virial_ratio=0.5)
        
        else:
            particles.scale_to_standard(convert_nbody=code_conv)
            IMBH = particles[particles.type=="imbh"].random_sample(nStar)
            mprim = IMBH.mass
            msec = self.star_mass(nStar, star_mdist)
            sma = np.random.uniform(5, 1000, nStar) | units.au
            ecc = np.sqrt(np.random.uniform(0, np.sqrt(0.9), nStar))
            inc = np.arccos(1-2*np.random.uniform(0, 1, nStar)) | units.rad
            loan = np.random.uniform(0, 2*np.pi, nStar) | units.rad
            aop = np.random.uniform(0, 2*np.pi, nStar) | units.rad
            true_anomaly = np.random.uniform(0, 2*np.pi, nStar)
            primaries, secondaries = generate_binaries(primary_mass=mprim,
                                                        secondary_mass=msec,
                                                        semi_major_axis=sma,
                                                        eccentricity=ecc,
                                                        true_anomaly=true_anomaly, 
                                                        inclination=inc,
                                                        longitude_of_the_ascending_node=loan,
                                                        argument_of_periapsis=aop,
                                                        G=constants.G)
                                                        
            primaries.position += IMBH.position
            primaries.velocity += IMBH.velocity
            primaries.type = "imbh"

            secondaries.position += IMBH.position
            secondaries.velocity += IMBH.velocity
            secondaries.type = "star"
            secondaries.radius = self.ZAMS_radius(secondaries.mass)
            bin_particles = Particles()
            bin_particles.add_particle(primaries)
            bin_particles.add_particles(secondaries)

        particles[particles.type!="star"].radius = self.IMBH_radius(particles[particles.type!="star"].mass)
        particles[particles.type!="star"].collision_radius = self.coll_radius(particles[particles.type!="star"].radius)
        particles[particles.type=="star"].radius = self.ZAMS_radius(particles[particles.type=="star"].mass)
        particles[particles.type=="star"].collision_radius = particles[particles.type=="star"].radius

        particles.Nej = 0
        particles.coll_events = 0
        particles.key_tracker = particles.key
        particles.move_to_center()
        
        return particles