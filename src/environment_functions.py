from amuse.datamodel import Particles
from amuse.ext.galactic_potentials import MWpotentialBovy2015
from amuse.ext.orbital_elements import orbital_elements_from_binary
from amuse.units import units, constants


def galactic_frame(parent_set, dx, dy, dz):
    """Shift particle set to the particle set
       Inputs:
       parent_set: The particle set
       d(x/y/z):   Distance of cluster to center of MW-like galaxy
    """

    MWG = MWpotentialBovy2015()
    parent_set.position+=[dx.value_in(units.pc), 
                    dy.value_in(units.pc), 
                    dz.value_in(units.pc)] | units.pc
    parent_set.velocity+=[0,1,0]*MWG.circular_velocity(parent_set.position.lengths())

    return parent_set

def parent_radius(mass, dt):
    """Merging/dissolution radius of parent systems
       Inputs:
       mass:   Parent particle set 
       dt:     Simulation time-step
    """
    return (constants.G*mass*(dt)**2)**(1./3.)

def tidal_radius(parent_set):
    """Tidal radius (Spitzer 1987 eqn 5.10)"""

    MWG = MWpotentialBovy2015()

    cg_sys = Particles(2)
    cg_sys[0].position = parent_set.center_of_mass()
    cg_sys[0].velocity = parent_set.center_of_mass_velocity()
    cg_sys[0].mass = parent_set.mass.sum()

    cg_sys[1].position = [0, 0, 0] | units.kpc
    cg_sys[1].velocity = [0, 0, 0] | units.kms
    cg_sys[1].mass = MWG.enclosed_mass(cg_sys[0].position.length())

    kepler_elements = orbital_elements_from_binary(cg_sys, G=constants.G)
    ecc = kepler_elements[3]
    coeff = ((3+ecc)**(-1)*(cg_sys[0].mass/cg_sys[1].mass))**(1/3)
    rtide = coeff*cg_sys[0].position.length()
    return rtide
