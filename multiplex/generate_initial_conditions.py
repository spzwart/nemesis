import numpy
from amuse.lab import *
#from amuse.ext.galactics_model import new_galactics_model

def cluster_name(index):
    import string
    ii = int(index/26)
    ri = index%26
    return string.ascii_uppercase[ii]+string.ascii_uppercase[ri]

def GALACTCS_make_galaxy_model(N, M_galaxy, R_galaxy, Mmin, Mmax):
    converter = nbody_system.nbody_to_si(M_galaxy, R_galaxy)
    n_halo = 0
    n_bulge = 0
    n_disk = N
    galaxy = new_galactics_model(n_halo,
                                 converter,
                                 do_scale=True,
                                 bulge_number_of_particles=n_bulge,
                                 disk_number_of_particles=n_disk)
    from amuse.lab import new_powerlaw_mass_distribution
    masses = new_powerlaw_mass_distribution(len(galaxy), Mmin, Mmax, -2.0)
    galaxy.mass = masses
    galaxy.radius = 3.0 | units.parsec
    galaxy.King_W0 = 3
    for i in range(len(galaxy)):
        galaxy[i].name = cluster_name(i)
    return galaxy

def make_galaxy_model(N, M_galaxy, R_galaxy, Mmin, Mmax):
    converter = nbody_system.nbody_to_si(M_galaxy, R_galaxy)
    galaxy = new_plummer_model(N, converter)
    from amuse.lab import new_powerlaw_mass_distribution
    masses = new_powerlaw_mass_distribution(len(galaxy), Mmin, Mmax, -2.0)
    galaxy.mass = masses
    galaxy.radius = 3.0 | units.parsec
    galaxy.King_W0 = 3
    for i in range(len(galaxy)):
        galaxy[i].name = cluster_name(i)
    return galaxy

def make_plummer_galaxy_model(N, M_galaxy, R_galaxy):
    converter = nbody_system.nbody_to_si(M_galaxy, R_galaxy)
    model = new_plummer_model(N, converter)
    model.radius = 3.0 | units.parsec
    model.King_W0 = 3
    model.mass = 20 | units.MSun
    return model

def make_globular_clusters(cluster_population):

    mmean = new_kroupa_mass_distribution(1000).sum()/1000.
    stars = Particles(0)
    for ci in cluster_population:
        nstars = max(1, int(ci.mass/mmean))
        #nstars = 1
        masses = new_kroupa_mass_distribution(nstars)        
        converter = nbody_system.nbody_to_si(masses.sum(), ci.radius)
        bodies = new_king_model(nstars, ci.King_W0, converter)
        bodies.mass = masses
        if len(bodies)>1:
            bodies.scale_to_standard(converter)
        bodies.parent = ci
        bodies.position += ci.position
        bodies.velocity += ci.velocity
        bodies.name = ci.name + "_star"
        ci.mass = bodies.mass.sum()
        bodies.subsystem = None
        stars.add_particles(bodies)
        
    return stars

def initialize_globular_clusters(N, M_glaxy, R_galaxy, Mmin, Mmax):

    if N>=26**2:
        print("Too many clusters to name individually.")
        print("STOP.")
        exit(-1)
    
    cluster_population = make_galaxy_model(N, M_galaxy, R_galaxy, Mmin, Mmax)
    print("cluster:", cluster_population)
    stars = make_globular_clusters(cluster_population)
    #nemesis_particles = stars.copy()
    print("Stars:", len(stars), stars.mass.sum().in_(units.MSun), stars.mass.max().in_(units.MSun), stars.mass.mean().in_(units.MSun))

    return stars

if __name__=="__main__":
    numpy.random.seed(7654304)
    N=5
    M_galaxy = 1.0e+11 | units.MSun
    R_galaxy = 4.5 | units.kpc
    Mmin = 20 | units.MSun
    Mmax = 100 | units.MSun
    stars = initialize_globular_clusters(N, M_galaxy, R_galaxy, Mmin, Mmax)
    filename = "globular_starclusters_in_galactic_potential.amuse"
    write_set_to_file(stars, filename, "hdf5", version="2.0", append_to_file=False)
