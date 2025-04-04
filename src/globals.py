from amuse.ext.galactic_potentials import MWpotentialBovy2015
from amuse.lab import constants, units

ASTEROID_RADIUS = 100. | units.km
CONNECTED_COEFF = 2.
EPS = 1.e-8
GRAV_THRESHOLD = 5.e-3
MIN_EVOL_MASS = 0.08 | units.MSun
PARENT_RADIUS_COEFF = 1000. | units.au
PARENT_RADIUS_MAX = 5000 | units.au
SI_UNITS = (1. | units.kg * units.m**-2.) * constants.G


MWG = MWpotentialBovy2015()