import numpy as np

from amuse.datamodel import Particles
from amuse.ext.galactic_potentials import MWpotentialBovy2015
from amuse.units import units


############################## TO WORK ON ########################################
# 1. CONSIDER ADDING ALGORITHM TO CHECK EJECTIONS AND REMOVE RELEVANT PARTICLES
##################################################################################


MWG = MWpotentialBovy2015()



def galactic_frame(parent_set: Particles, dx, dy, dz, dvx, dvy, dvz) -> Particles:
    """
    Shift particle set to galactic frame.

    Args:
        parent_set (Particles):  The particle set
        dx (units.length): x-coordinate shift in the galactocentric frame
        dy (units.length): y-coordinate shift in the galactocentric frame
        dz (units.length): z-coordinate shift in the galactocentric frame
        dvx (units.length): x-velocity shift in the galactocentric frame
        dvy (units.length): y-velocity shift in the galactocentric frame
        dvz (units.length): z-velocity shift in the galactocentric frame
    Returns:
        Particles: Particle set with galactocentric coordinates
    """
    parent_set.x += dx
    parent_set.y += dy
    parent_set.z += dz
    distance = np.sqrt(dx**2 + dy**2 + dz**2)

    parent_set.vx += dvx
    parent_set.vy += dvy + MWG.circular_velocity(distance)
    parent_set.vz += dvz

    return parent_set

def set_parent_radius(system_mass) -> units.au:
    """
    Merging radius of parent systems. Based on system crossing time.
    - Too large → Poor angular momentum conservation, inaccurate center of mass.
    - Too small → Excessive computation due to frequent small timesteps.

    Args:
       system_mass (units.mass):  Total mass of the system
    Returns:
       units.length: Merging radius of the parent system
    """
    radius = 500. * (system_mass.value_in(units.MSun))**(1./3.) | units.AU
    return radius

def planet_radius(planet_mass) -> units.REarth:
    """
    Compute planet radius (arXiv:2311.12593).

    Args:
        planet_mass (units.mass):  Mass of the planet
    Returns:
        units.mass:  Planet radius
    """
    mass_in_earth = planet_mass.value_in(units.MEarth)

    if mass_in_earth < 7.8:
        return (1. | units.REarth)*(mass_in_earth)**0.41
    elif mass_in_earth < 125:
        return (0.55 | units.REarth)*(mass_in_earth)**0.65
    return (14.3 | units.REarth)*(mass_in_earth)**(-0.02) 

def ZAMS_radius(star_mass) -> units.RSun:
    """
    Define stellar radius at ZAMS.

    Args:
        star_mass (units.mass):  Mass of the star
    Returns:
        units.length:  The ZAMS radius of the star
    """
    mass_in_sun = star_mass.value_in(units.MSun)
    mass_sq = (mass_in_sun)**2.

    numerator = mass_in_sun**1.25 * (0.1148 + 0.8604 * mass_sq)
    denominator = (0.04651 + mass_sq)
    r_zams = numerator / denominator

    return r_zams | units.RSun