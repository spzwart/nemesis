from amuse.lab import *

def new_option_parser():
    from amuse.units.optparse import OptionParser
    result = OptionParser()
    result.add_option("-f", 
                      dest="filename", default = "globular_starclusters_in_galactic_potential.amuse",
                      help="input filename [%default]")
    return result

if __name__ in ('__main__', '__plot__'):
    o, arguments  = new_option_parser().parse_args()
    bodies = read_set_from_file(o.filename, "hdf5", close_file=True)
    print(bodies)
    print("N=", len(bodies))

