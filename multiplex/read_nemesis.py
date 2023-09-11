import numpy
from amuse.lab import *
from matplotlib import pyplot

def load(filename):
    with read_set_from_file(filename, "amuse", close_file=False, return_context = True, names = ('nemesis_particles',)) as (particles_set,):
        return particles_set.copy()

def plot_xy(n, c):
    pyplot.figure(figsize = (12, 12))
    pyplot.xlim(-10, 10)
    pyplot.ylim(-10, 10)
    pyplot.xlabel("x [kpc]")
    pyplot.ylabel("y [kpc]")
    print(len(n), len(n.x), len(c))
    for ci in c:
        si = 100*len(ci.subsystem)
        pyplot.scatter(ci.x.value_in(units.kpc), ci.y.value_in(units.kpc),
                       marker='o', s=si, alpha=0.5, lw=2,
                       edgecolors='red',facecolors='white')
    m = 100*n.mass/n.mass.max()
    c = numpy.log10(n.temperature.value_in(units.K))
    pyplot.scatter(n.x.value_in(units.kpc), n.y.value_in(units.kpc), s=m, c=c, cmap="hot")
    pyplot.text(7, 10.5, str(round(n[0].age.value_in(units.Myr)))+"Myr", fontsize=24)

def plot_xz(n, c):
    #pyplot.figure(figsize = (12, 12))
    pyplot.figure(figsize = (12, 2.4))
    pyplot.xlim(-10, 10)
    pyplot.ylim(-1.0, 1.0)
    pyplot.xlabel("x [kpc]")
    pyplot.ylabel("z [kpc]")
    print(len(n), len(n.x), len(c))
    for ci in c:
        si = 100*len(ci.subsystem)
        pyplot.scatter(ci.x.value_in(units.kpc), ci.z.value_in(units.kpc),
                       marker='o', s=si, alpha=0.5, lw=2,
                       edgecolors='red',facecolors='white')
    m = 100*n.mass/n.mass.max()
    c = numpy.log10(n.temperature.value_in(units.K))
    pyplot.scatter(n.x.value_in(units.kpc), n.z.value_in(units.kpc), s=m, c=c, cmap="hot")
    pyplot.text(7, 1.2, str(round(n[0].age.value_in(units.Myr)))+"Myr", fontsize=24)
    
def plot_nemesis_particles(filename, show=True, index=0, xy=0):
    n, c = load_nemesis_particles(filename)
    print("n=", len(n))
    print(n.mass.max().in_(units.MSun))
    if xy==0:
        plot_xy(n, c)
    else:
        plot_xz(n, c)
    if show:
        pyplot.show()
    else:
        filename = 'movie_i%6.6i.png'%index
        pyplot.savefig(filename)

"""
def plot_nemesis_particles(filename, show=True, index=0, xy=0):
    n, c = load_nemesis_particles(filename)
    print "n=", len(n)
    print len(n), len(n.x), len(c)
    xn = []
    yn = []
    s = []
    for ci in c:
        s.append(100*len(ci.subsystem))
    for ni in n:
        xn.append(ni.x.value_in(units.kpc))
    print c
    x = c.x.value_in(units.kpc)
    if xy==0:
        for ni in n:
            yn.append(ni.value_in(units.kpc))
        y = c.y.value_in(units.kpc)
        pyplot.figure(figsize = (12, 12))
        pyplot.ylim(-10, 10)
    else:
        for ni in n:
            zn.append(ni.z.value_in(units.kpc))
        y = c.z.value_in(units.kpc)
        pyplot.figure(figsize = (12, 1.2))
        pyplot.ylim(-1, 1)
        
    pyplot.xlim(-10, 10)
    pyplot.xlabel("x [kpc]")
    pyplot.ylabel("y [kpc]")

    for i in range(len(xn)):
        pyplot.scatter(xn[i], yn[i], marker='o', s=s[i], alpha=0.5, lw=2,
                   edgecolors='red',facecolors='white')
    m = 50*n.mass/n.mass.max()
    c = n.mass.value_in(units.MSun)
    pyplot.scatter(x, y, s=m, c=c, cmap="hot")
    if show:
        pyplot.show()
    else:
        filename = 'movie_i%6.6i.png'%index
        pyplot.savefig(filename)
"""
    
def load_nemesis_particles(filename):
    p =load(filename)
    stars = Particles(0)
    com = Particles(0)
    for ni in p:
        if ni.subsystem:
            s = ni.subsystem.copy()
            print("N=", len(s))
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
    plot_nemesis_particles(o.filename)

