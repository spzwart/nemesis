import numpy
from amuse.lab import *

def load(filename):
    with read_set_from_file(filename, "amuse", close_file=False, return_context = True, names = ('nemesis_particles',)) as (particles_set,):
        return particles_set.copy()

def load_nemesis_particles(filename):
    p =load(filename)
    print(p)
    stars = Particles(0)
    com = Particles(0)
    for ni in p:
        if ni.subsystem:
            print(ni.position.in_(units.kpc))
            s = ni.subsystem.copy()
            com.add_particle(ni)
            s.position += ni.position
            s.velocity += ni.velocity
            stars.add_particles(s)
        else:
            stars.add_particle(ni)
    return stars, com

def new_option_parser():
    from amuse.units.optparse import OptionParser
    result = OptionParser()
    result.add_option("-f", 
                      dest="filename", default = "globulars_i000010.amuse",
                      help="input filename [%default]")
    return result

if __name__ in ('__main__', '__plot__'):
    o, arguments  = new_option_parser().parse_args()

    n, c = load_nemesis_particles(o.filename)
    print("n=", len(n))
    from matplotlib import pyplot
    print(len(n), len(n.x), len(c))
    nc = 0
    for ci in c:
        si = len(ci.subsystem)
        nc += si
        pyplot.scatter(ci.x.value_in(units.kpc), ci.y.value_in(units.kpc),
                       marker='o', s=100*si, alpha=0.5, lw=2,
                       edgecolors='red',facecolors='white')
    pyplot.scatter(n.x.value_in(units.kpc), n.y.value_in(units.kpc))
    print("N=", len(n), len(c), nc)
    
    pyplot.show()

