import sys
import numpy
from numpy import random
from amuse.lab import *
from amuse.units.quantities import as_vector_quantity

def rotate(pebble, phi, theta, psi): # theta and phi in radians
    Runit = units.AU
    Vunit = units.kms
    B = EulerAngles(phi, theta, psi)
#    print "Euler rotation matrix: ", B
    pebble.position = Runit(numpy.dot(B, pebble.position.value_in(Runit)))

    # rotate the velocity verctor with the transpose of the Euler matrix
    pebble.velocity = Vunit(numpy.dot(B, pebble.velocity.value_in(Vunit)))

# select Eurler angles randomly. 
def random_Euler_angles():
    phi   = 2*numpy.pi*random()
    theta = numpy.acos(1-2*random())
    chi   = 2*numpy.pi*random()
    return phi, theta, chi

def EulerAngles(phi, theta, chi):
    cosp=numpy.cos(phi)
    sinp=numpy.sin(phi)
    cost=numpy.cos(theta)
    sint=numpy.sin(theta)
    cosc=numpy.cos(chi)
    sinc=numpy.sin(chi)
    #see wikipedia: http://en.wikipedia.org/wiki/Rotation_matrix
    return numpy.array(
        [[cost*cosc, -cosp*sinc + sinp*sint*cosc, sinp*sinc + cosp*sint*cosc], 
         [cost*sinc, cosp*cosc + sinp*sint*sinc, -sinp*cosc + cosp*sint*sinc],
         [-sint,  sinp*cost,  cosp*cost]])

def posvel_from_orbital_elements(Mstar, pebble, kepler):

    mean_anomaly = random.uniform(0, 2*numpy.pi, 1)
    kepler.initialize_from_elements(
        Mstar, 
        pebble.semimajor_axis, 
        pebble.eccentricity, 
        mean_anomaly=mean_anomaly)
    pebble.position = as_vector_quantity(kepler.get_separation_vector())
    pebble.velocity = as_vector_quantity(kepler.get_velocity_vector())

    phi = 0
    theta = 0
    psi = random.uniform(0, 360, 1)[0]
    psi = numpy.radians(psi) #rotate under z
    rotate(pebble, phi, theta, psi) # theta and phi in radians            

def make_planets(central_particle, Nplanets, arange, erange, mrange, phi=None, theta=None):

    planet_particles = Particles(Nplanets)
    planet_particles.semimajor_axis = random.uniform(
        arange[0].value_in(units.AU), 
        arange[1].value_in(units.AU), 
        size=Nplanets) | units.AU
    planet_particles.eccentricity = random.uniform(*erange, size=Nplanets)
    planet_particles.mass = new_salpeter_mass_distribution(
        Nplanets, 
        mass_min=mrange[0], 
        mass_max = mrange[1], 
        alpha=-2.35)

    converter = nbody_system.nbody_to_si(central_particle.mass, arange[-1])
    kepler = Kepler(converter)
    kepler.initialize_code()

    if phi is None:
        phi = numpy.radians(random.uniform(0, 90, 1)[0])#rotate under x
    if theta is None:
        theta = numpy.radians(random.uniform(0, 180, 1)[0]) #rotate under y
    psi = 0
    for dpi in planet_particles:
        posvel_from_orbital_elements(central_particle.mass, dpi, kepler)
        rotate(dpi, phi, theta, psi) # theta and phi in radians            

    planet_particles.position += central_particle.position
    planet_particles.velocity += central_particle.velocity

    return planet_particles

def plot(planet_particles):
    from matplotlib import pyplot
    fig = pyplot.figure(figsize=(10,4))
    ax_left = fig.add_subplot(1,2,1)
    ax_right = fig.add_subplot(1,2,2)
    mscale = planet_particles.mass.min()
    ax_left.scatter(planet_particles.x.value_in(units.AU), planet_particles.y.value_in(units.AU), s=planet_particles.mass/mscale)
    ax_right.scatter(planet_particles.x.value_in(units.AU), planet_particles.z.value_in(units.AU), s=planet_particles.mass/mscale)
    pyplot.show()

def new_option_parser():
    from amuse.units.optparse import OptionParser
    result = OptionParser()
    result.add_option("--seed", 
                      dest="seed", type="int", default = 666,
                      help="random number seed [%default]")
    return result

if __name__ in ('__main__', '__plot__'):
    o, arguments  = new_option_parser().parse_args()
    random.seed(seed=o.seed)

    arange = [10, 100] | units.AU
    erange = [0.0, 0.1] 
    mrange = [0.01, 10] | units.MJupiter
    Nplanets = 10
    Mstar = 1|units.MSun
    central_particle = Particle(mass=Mstar)
    central_particle.position = (0,0,0) | units.AU
    central_particle.velocity = (0,0,0) | units.kms
    planet_particles = make_planets(central_particle, Nplanets, arange, erange, mrange)

    plot(planet_particles)
