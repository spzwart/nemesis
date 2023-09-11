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

def load(filename):
    with read_set_from_file(filename, "amuse", close_file=False, return_context = True, names = ('nemesis_particles',)) as (particles_set,):
        return particles_set.copy()

def plot_xy(stars, index, LL):
    f=pyplot.figure( figsize=(8,8))  
    ax = f.gca()
    circles=[]
    s = 6*stars.mass/stars.mass.max()
    m = stars.mass/stars.mass.max()
    for si in range(len(stars)):
        if "A" in stars[si].name:
          c = 'r.'
        else:
          c = 'b.'
          pyplot.plot(stars[si].x.value_in(units.kpc),stars[si].y.value_in(units.kpc),c,markersize=s[si], mew=s[si], alpha=0.5)

    pyplot.xlim(-1.2*LL,1.2*LL)
    pyplot.ylim(-1.2*LL,1.2*LL)
    pyplot.xlabel("x kpc")
    pyplot.ylabel("y kpc")
    filename = "projectedview_xy_i{0:06}.png".format(index)
    pyplot.savefig(filename,bbox_inches='tight')
    f.clear()
    pyplot.close(f)

def plot_xz(stars, index, LL):
    f=pyplot.figure( figsize=(8,8))  
    ax = f.gca()
    circles=[]
    s = 6*stars.mass/stars.mass.max()
    m = stars.mass/stars.mass.max()
    for si in range(len(stars)):
        if "A" in stars[si].name:
          c = 'r.'
        else:
          c = 'b.'
          pyplot.plot(stars[si].x.value_in(units.kpc),stars[si].z.value_in(units.kpc),c,markersize=s[si], mew=s[si], alpha=0.5)

    pyplot.xlim(-1.2*LL,1.2*LL)
    pyplot.ylim(-1.2*LL,1.2*LL)
    pyplot.xlabel("x kpc")
    pyplot.ylabel("x kpc")
    filename = "projectedview_xz_i{0:06}.png".format(index)
    pyplot.savefig(filename,bbox_inches='tight')
    f.clear()
    pyplot.close(f)
    
def plot_data(index, LL):
    filename = "globulars_i{0:06}.amuse".format(index)
    stars = load(filename)
    print(stars)

    plot_xy(stars, index, LL)
    plot_xz(stars, index, LL)

def new_option_parser():
    from amuse.units.optparse import OptionParser
    result = OptionParser()
    result.add_option("-f",
                      dest="filename", default = "",
                      help="input data filename [%default]")
    result.add_option("-l", 
                      dest="lim", type="float", default = 2,
                      help="image size limit [%default]")
    result.add_option("-p", 
                      dest="planet_id", default = "planet",
                      help="planet id [%default]")
    result.add_option("--time_offset", unit=units.day,
                      dest="time_offset", type="float", default = 0.0|units.day,
                      help="offset time [%default]")
    return result

if __name__ in ("__main__"):
    o, arguments  = new_option_parser().parse_args()
    if len(o.filename):
        index = int(o.filename.split("_i")[1].split(".h5")[0])

        plot_data(o.filename, index, o.lim, o.time_offset, o.planet_id)
        pyplot.show()
    else:
        import glob
        list = glob.glob("./"+"globulars_i*.amuse")
        print(list)
        for li in list:
            filename = li.split(".amuse")[0]
            index = int(filename.split("globulars_i")[1])

            figfilename = "projected_view_i{0:04}.png".format(index)
            import os.path
            if os.path.isfile(figfilename):
                print("Skip, file already exists: ", figfilename)
            else:
                print("Process filename:", filename, index)
                plot_data(index, o.lim)

    

    
