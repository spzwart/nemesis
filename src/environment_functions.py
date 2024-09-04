import ctypes
import numpy as np

from amuse.datamodel import Particles
from amuse.ext.galactic_potentials import MWpotentialBovy2015
from amuse.ext.orbital_elements import orbital_elements_from_binary
from amuse.units import units, constants


MWG = MWpotentialBovy2015()

def ejection_checker(particle_set, tidal_field):
    """
    Find ejected systems (particles whose second nearest neighbour is separated with > DIST_THRESHOLD)

    Args:
        particle_set (ParticleSet): The particle set
        tidal_field (Boolean):      1 = Physical tidal radius 0 = hard-coded tidal radius
    Returns:
        Array: Array of booleans flagging ejected particles
    """
    lib = ctypes.CDLL('./src/ejection.so')  # Adjust the path as necessary
    lib.find_nearest_neighbour.argtypes = [
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),  # xcoord
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),  # ycoord
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),  # zcoord
        ctypes.c_int,     # num_particles
        ctypes.c_double,  # NN threshold distance
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),  # ejected bool
    ]
    
    if (tidal_field):
        threshold = tidal_radius(particle_set).value_in(units.m)
    else:
        threshold = (3. | units.pc).value_in(units.m)
    num_parts = len(particle_set)
    
    ejected_bools = np.zeros(len(particle_set))
    lib.find_nearest_neighbour(
        particle_set.x.value_in(units.m),
        particle_set.y.value_in(units.m),
        particle_set.z.value_in(units.m),
        num_parts, threshold,
        ejected_bools
    )
    ejected_idx = np.where(ejected_bools == 1)[0]
    
    return ejected_idx

def galactic_frame(parent_set, dx, dy, dz, dvx, dvy, dvz):
    """
    Shift particle set to galactic frame

    Args:
        parent_set (ParticleSet): The particle set
        dx (Float):               x Distance of cluster to center of galaxy
        dy (Float):               y Distance of cluster to center of galaxy
        dz (Float):               z Distance of cluster to center of galaxy
        dvx (Float):              x-Velocity of cluster in galactocentric frame
        dvy (Float):              y-Velocity of cluster in galactocentric frame
        dvz (Float):              z-Velocity of cluster in galactocentric frame
    Returns:
        ParticleSet: Particle set with galactocentric coordinates
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

def set_parent_radius(tot_mass, diag_dt, pop):
    """
    Merging radius of parent systems

    Args:
       tot_mass (Float):  Total mass of the system
       diag_dt (Float):   Diagnostic timestep
       pop (Float/Int):   Population of the system
    Returns:
       Float: Merging radius of the parent system
    """
    return pop*(constants.G*tot_mass*diag_dt**2.)**(1./3.)

def planet_radius(planet_mass):
    """
    Define planet radius (arXiv:2311.12593)
    
    Args:
        plant_mass (Float):  Mass of the planet
    Returns:
        Float: Planet radius
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

def tidal_radius(parent_set):
    """
    Tidal radius (Spitzer 1987 eqn 5.10)
    
    Args:
        parent_set (ParticleSet):  The parent particle set
    Returns:
        Float: The tidal radius of the cluster
    """
    cluster_galaxy_system = Particles(2)
    
    cluster_mass = parent_set.mass.sum()
    cluster_pos = parent_set.center_of_mass()
    enclosed_mass = MWG.enclosed_mass(cluster_pos.length())
    
    cluster_galaxy_system[0].position = cluster_pos
    cluster_galaxy_system[0].velocity = parent_set.center_of_mass_velocity()
    cluster_galaxy_system[0].mass = cluster_mass
    
    cluster_galaxy_system[1].position = [0., 0., 0.] | units.kpc
    cluster_galaxy_system[1].velocity = [0., 0., 0.] | units.kms
    cluster_galaxy_system[1].mass = enclosed_mass

    kepler_elements = orbital_elements_from_binary(cluster_galaxy_system, G=constants.G)
    ecc = kepler_elements[3]
    return ((cluster_mass/enclosed_mass)/(3.+ecc))**(1./3.) * cluster_pos.length()
    
def ZAMS_radius(star_mass):
    """
    Define stellar radius at ZAMS
    
    Args:
        star_mass (Float):  Mass of the star
    Returns:
        Float: The ZAMS radius of the star
    """
    mass_in_sun = star_mass.value_in(units.MSun)
    mass_sq = (mass_in_sun)**2
    r_zams = mass_in_sun**1.25 * (0.1148 + 0.8604 * mass_sq)/(0.04651 + mass_sq)
    return r_zams | units.RSun