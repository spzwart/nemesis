import numpy
from matplotlib import pyplot 

from amuse.lab import *
from amuse.units import units, quantities
from amuse import plot as aplot
from amuse.ext.molecular_cloud import molecular_cloud
from amuse.ext.evrard_test import body_centered_grid_unit_cube
from amuse.ext.stellar_wind import new_stellar_wind
from amuse.ext.stellar_wind import StarsWithMassLoss
from amuse.couple import bridge
from amuse.community.fractalcluster.interface import new_fractal_cluster_model

def plot_data(filename):
    stars = read_set_from_file(filename, "hdf5", close_file=True)
    print("N=", len(stars))

    f=pyplot.figure( figsize=(8,8))  
    pyplot.scatter(stars.x.value_in(units.kpc),stars.y.value_in(units.kpc))

    pyplot.xlabel("kpc")
    pyplot.xlabel("ypc")

def new_option_parser():
    from amuse.units.optparse import OptionParser
    result = OptionParser()
    result.add_option("-f",
                      dest="filename", default = "",
                      help="input data filename [%default]")
    return result

if __name__ in ("__main__"):
    o, arguments  = new_option_parser().parse_args()
    plot_data(o.filename)
    pyplot.show()
