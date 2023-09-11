import numpy

def lognormal_probability_density_function(x, mean=3, sigma=3):
    p = (1./ (x*numpy.sqrt(2*numpy.pi*sigma**2))) * \
        numpy.exp( -(numpy.log(x)-mean)**2/(2*sigma**2)) 
    return p


rmin = 0.01
rmax = 10.
R = 10**numpy.arange(numpy.log10(rmin), numpy.log10(rmax), 0.01)
rmean = numpy.log(5)
rdisp = numpy.log(3)
#P = lognormal_probability_density_function(R, mean=numpy.log10(rmean), sigma=numpy.log10(rdisp))
P = 300*lognormal_probability_density_function(R, mean=rmean, sigma=rdisp)

m = abs(numpy.log(numpy.random.lognormal(mean=5, sigma=5, size=1000)))

print(R)
print(P)
from matplotlib import pyplot
pyplot.hist(m, 50)
pyplot.plot(R, P)
pyplot.show()
