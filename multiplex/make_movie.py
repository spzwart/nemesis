import glob
import os.path
from amuse.lab import * 
from matplotlib import pyplot
from amuse.plot import scatter
from read_nemesis import plot_nemesis_particles

def mkmovie(filename):
    r = [] 
    print(list)
    for filename in list:
        print(filename)
        dex = filename.split("_i")[1].split(".")[0]
        index = int(dex)
        bodies = read_set_from_file(filename, "hdf5", close_file=True)
        plot(bodies, index)

def new_option_parser():
    from amuse.units.optparse import OptionParser
    result = OptionParser()
    result.add_option("-f", 
                      dest="filename", default = "globulars_i",
                      help="input filename [%default]")
    result.add_option("--xy", 
                      dest="xy", type="int", default = 0,
                      help="Plotted coordinates xy ot xz [%default]")
    return result
        
if __name__ in ('__main__','__plot__'):
    o, arguments  = new_option_parser().parse_args()
    f = glob.glob(o.filename+"*.amuse")
    index = []
    for fi in f:
        i = int(fi.split("globulars_i")[1].split(".amuse")[0])
        index.append(i)
    indices, filenames = list(zip(*sorted(zip(index, f))))
    for i, fi in zip(indices, filenames):
        figfile = "movie_i%6.6i.png"%i
        if os.path.isfile(figfile):
            print("Skip, file already exists: ", fi)
        else:
            print("Processing file:", fi)
            plot_nemesis_particles(fi, False, i, o.xy)
    
