from amuse.datamodel import Particles
from amuse.ext.orbital_elements import orbital_elements_from_binary
from amuse.units import units, constants


class EnvironmentFunctions(object):
    def tidal_radius(self, pset):
        """Tidal radius (Spitzer 1987 eqn 5.10)"""

        cg_sys = Particles(2)
        cg_sys[0].position = pset.center_of_mass()
        cg_sys[0].velocity = pset.center_of_mass_velocity()
        cg_sys[0].mass = pset.mass.sum()

        cg_sys[1].position = [0, 0, 0] | units.kpc
        cg_sys[1].velocity = [0, 0, 0] | units.kms
        cg_sys[1].mass = self.MWG.enclosed_mass(cg_sys[0].position.length())

        kepler_elements = orbital_elements_from_binary(cg_sys, G=constants.G)
        ecc = kepler_elements[3]
        coeff = ((3+ecc)**(-1)*(cg_sys[0].mass/cg_sys[1].mass))**(1/3)
        rtide = coeff*cg_sys[0].position.length()
        return rtide

    def parent_radius(self, mass, dt):
        """Merging/dissolution radius of parent systems
           Inputs:
           mass:   Parent particle set 
           dt:     Simulation time-step
        """
        return (constants.G*mass*(dt)**2)**(1./3.)
        
    def galactic_frame(self, pset, dx, dy, dz):
        """Shift particle set to the particle set
           Inputs:
           pset:     The particle set
           d(x/y/z): The distance of cluster to center of MW like galaxy
        """

        from nemesis import MWpotentialBovy2015
        MWG = MWpotentialBovy2015()
        pset.position+=[dx.value_in(units.pc), 
                        dy.value_in(units.pc), 
                        dz.value_in(units.pc)] | units.pc
        pset.velocity+=[0,1,0]*MWG.circular_velocity(pset.position.lengths())

        return pset
