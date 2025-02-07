import numpy as np

from amuse.datamodel import Particles
from amuse.ext.galactic_potentials import MWpotentialBovy2015
from amuse.units import units


MWG = MWpotentialBovy2015()

def galactic_frame(parent_set: Particles, dx: float, dy: float, dz: float, 
                   dvx: float, dvy: float, dvz: float) -> Particles:
    """
    Shift particle set to galactic frame

    Args:
        parent_set (Particles):  The particle set
        dx (float):  x-coordinate in galactocentric frame
        dy (float):  y-coordinate in galactocentric frame
        dz (float):  z-coordinate in galactocentric frame
        dvx (float):  x-Velocity in galactocentric frame
        dvy (float):  y-Velocity in galactocentric frame
        dvz (float):  z-Velocity in galactocentric frame
    Returns:
        Particles: Particle set with galactocentric coordinates
    """
    parent_set.x += dx
    parent_set.y += dy
    parent_set.z += dz
    distance = np.sqrt(dx**2 + dy**2 + dz**2)
    
    dvy += MWG.circular_velocity(distance)
    parent_set.vx += dvx
    parent_set.vy += dvy
    parent_set.vz += dvz
    
    return parent_set

def set_parent_radius(tot_mass: float) -> float:
    """
    Merging radius of parent systems. Based on system crossing time.
    
    Too large a radius:
        - Poor conservation of angular momentum
        - Center of mass approximation is poor due to low resolution
    Too small a radius:
        - Slow down the simulation due to shared timesteps. 
        - System crossing time << internal time-step and violent interactions are missed.

    Args:
       tot_mass (float):  Total mass of the system
    Returns:
       float: Merging radius of the parent system
    """
    radius = 100. * (tot_mass.value_in(units.MSun))**(1./3.) | units.AU
    return radius

def planet_radius(planet_mass: float) -> float:
    """
    Define planet radius (arXiv:2311.12593)
    
    Args:
        plant_mass (float):  Mass of the planet
    Returns:
        float:  Planet radius
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
    
def ZAMS_radius(star_mass: float) -> float:
    """
    Define stellar radius at ZAMS
    
    Args:
        star_mass (float):  Mass of the star
    Returns:
        float:  The ZAMS radius of the star
    """
    mass_in_sun = star_mass.value_in(units.MSun)
    mass_sq = (mass_in_sun)**2.
    
    numerator = mass_in_sun**1.25 * (0.1148 + 0.8604 * mass_sq)
    denominator = (0.04651 + mass_sq)
    r_zams = numerator/denominator
    
    return r_zams | units.RSun