import time
from amuse.ext.galactic_potentials import MWpotentialBovy2015
from amuse.lab import constants, units

ASTEROID_RADIUS = 10. | units.km
CONNECTED_COEFF = 2.
EPS = 1.e-8
GRAV_THRESHOLD = 5.e-3
MIN_EVOL_MASS = 0.08 | units.MSun
PARENT_RADIUS_COEFF = 500. | units.au
PARENT_RADIUS_MAX = 2500 | units.au
PARENT_N_WORKERS = 8

SI_UNITS = (1. | units.kg * units.m**-2.) * constants.G
START_TIME = time.time()


MWG = MWpotentialBovy2015()