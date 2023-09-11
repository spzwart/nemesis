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

def plot_data(index, LL):
    starfile = "stars_i{0:06}.amuse".format(index)
    nemesisfile = "stars_i{0:06}.amuse".format(index)
    stars = read_set_from_file(starfile, "hdf5", close_file=True)
    nemesis = read_set_from_file(nemesisfile, "hdf5", close_file=True)
    pyplot.plot(nemesis.x.value_in(units.kpc),nemesis.y.value_in(units.kpc),'k.')

    f=pyplot.figure( figsize=(8,8))  
    ax = f.gca()
    circles=[]
    s = 6*stars.mass/stars.mass.max()
    m = stars.mass/stars.mass.max()
    if 'xy' in filename:
      for si in range(len(stars)):
        if "A" in stars[si].name:
          c = 'r.'
        else:
          c = 'b.'
        pyplot.plot(stars[si].x.value_in(units.kpc),stars[si].y.value_in(units.kpc),c,markersize=s[si], mew=s[si], alpha=0.5)
    else:
      for si in range(len(stars)):
        if "A" in stars[si].name:
          c = 'r.'
        else:
          c = 'b.'
        pyplot.plot(stars[si].x.value_in(units.kpc),stars[si].z.value_in(units.kpc),c,markersize=s[si], mew=s[si], alpha=0.5)

    pyplot.xlim(-1.2*LL,1.2*LL)
    pyplot.ylim(-1.2*LL,1.2*LL)
    pyplot.xlabel("kpc")
    #pyplot.savefig(filename+'%6.6i.png'%index,bbox_inches='tight')
    pyplot.savefig(filename+'.png',bbox_inches='tight')
    f.clear()
    pyplot.close(f)

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
        index = int(o.filename.split("_i")[1].split(".amuse")[0])

        plot_data(index, o.lim)
        pyplot.show()

    
