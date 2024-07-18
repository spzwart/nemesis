import ctypes
import numpy as np
from random import choices

from amuse.datamodel import Particles
from amuse.ext.galactic_potentials import MWpotentialBovy2015
from amuse.ext.orbital_elements import orbital_elements_from_binary
from amuse.units import units, constants

DIST_THRESHOLD = 3 | units.pc
SN_MEAN_VEL = 250 
SN_STD_VEL = 190
MWG = MWpotentialBovy2015()
VELOCITY_RANGE = np.linspace(0, 10000, 1000)

def ejection_checker(particle_set):
    """Find ejected systems"""
    lib = ctypes.CDLL('./src/ejection.so')  # Adjust the path as necessary
    lib.find_nearest_neighbour.argtypes = [
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),  # xcoord
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),  # ycoord
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),  # zcoord
        ctypes.c_int,  # num_particles
        ctypes.c_double,  # NN threshold distance
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),  # ejected bool
    ]
    
    threshold = DIST_THRESHOLD.value_in(units.m)
    parts = particle_set.copy()
    num_parts = len(parts)
    
    ejected_bools = np.zeros(len(parts))
    lib.find_nearest_neighbour(
        parts.x.value_in(units.m),
        parts.y.value_in(units.m),
        parts.z.value_in(units.m),
        num_parts, threshold,
        ejected_bools
    )
    ejected_idx = np.where(ejected_bools == 1)[0]
    
    return ejected_idx

def galactic_frame(parent_set, dx, dy, dz, dvx, dvy, dvz):
    """Shift particle set to galactic frame
    Inputs:
    parent_set:  The particle set
    d(x/y/z):  Distance of cluster to center of MW-like galaxy
    d(vx/vy/vz):  Velocity of cluster around galactic potential
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

def set_parent_radius(system_mass, dt):
    """Merging radius of parent systems"""
    radius = 10*(constants.G*system_mass*dt**2)**(1./3.)
    return min(2000 | units.AU, max(50|units.AU, radius))

def planet_radius(planet_mass):
        """Define planet radius based on its mass"""
        mass_in_earth = planet_mass.value_in(units.MEarth)
        if planet_mass < (7.8|units.MEarth):
            radius = (1|units.REarth)*(mass_in_earth)**0.41
            return radius
        if planet_mass < (125|units.MEarth):
            radius = (0.55|units.REarth)*(mass_in_earth)**0.65
            return radius
        radius = (14.3|units.REarth)*(mass_in_earth)**(-0.02) 
        return radius

def natal_kick_pdf():
    """Extract natal kick of SN event"""
    # PDF of SN kicks: arXiv:708071
    weight = np.sqrt(2/np.pi) * (SN_MEAN_VEL**2/SN_STD_VEL**3) \
             * np.exp(-VELOCITY_RANGE**2/(2*SN_STD_VEL**2))
    
    r = [-1,1]
    scalex = np.random.choice(r) | units.kms
    scaley = np.random.choice(r) | units.kms
    scalez = np.random.choice(r) | units.kms
    
    kick_vx = np.array(choices(VELOCITY_RANGE, weight, k=1)) * scalex
    kick_vy = np.array(choices(VELOCITY_RANGE, weight, k=1)) * scaley
    kick_vz = np.array(choices(VELOCITY_RANGE, weight, k=1)) * scalez
    return kick_vx[0], kick_vy[0], kick_vz[0]

def tidal_radius(parent_set):
    """Tidal radius (Spitzer 1987 eqn 5.10). Assume system is point mass"""
    cg_sys = Particles(2)
    cg_sys[0].position = parent_set.center_of_mass()
    cg_sys[0].velocity = parent_set.center_of_mass_velocity()
    cg_sys[0].mass = parent_set.mass.sum()

    cg_sys[1].position = [0, 0, 0] | units.kpc
    cg_sys[1].velocity = [0, 0, 0] | units.kms
    cg_sys[1].mass = MWG.enclosed_mass(cg_sys[0].position.length())

    kepler_elements = orbital_elements_from_binary(cg_sys, G=constants.G)
    ecc = kepler_elements[3]
    coeff = ((cg_sys[0].mass/cg_sys[1].mass)/(3+ecc))**(1/3)
    rtide = coeff*cg_sys[0].position.length()
    return rtide
    
def ZAMS_radius(star_mass):
    """Define stellar radius at ZAMS"""
    mass_in_sun = star_mass.value_in(units.MSun)
    mass_sq = (mass_in_sun)**2
    r_zams = mass_in_sun**1.25*(0.1148+0.8604*mass_sq)/(0.04651+mass_sq)
    return r_zams | units.RSun