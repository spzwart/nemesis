import numpy
from amuse.lab import *
from amuse.io import store
from amuse.community.seba.interface import SeBa

import time
from amuse.lab import *
from amuse.couple import bridge
from amuse.ext.solarsystem import new_solar_system

from amuse.ext.rotating_bridge import Rotating_Bridge
from amuse.community.galaxia.interface import BarAndSpirals3D
from amuse.ext.composition_methods import *
from amuse.units import quantities

I = 0
class IntegrateOrbit(object):
    """
    This class makes the integration of the Sun in the Milky Way
    by using BarAndSpirals3D. 
    galaxy(): Function that sets the desired Galactic model. Any question on the parameters, contact me
    creation_particles_noinertial(): creates a parti le set in a rotating frame
    noinertial_to_inertial(): converts data from rotating to inertial frame
    get_pos_vel_and_orbit(): Makes the evolution of the particle set
    """
    
    def __init__(self, t_end= 10 |units.Myr, dt_bridge=0.5 |units.Myr, method= SPLIT_6TH_SS_M13, phase_bar= 0, phase_spiral= 0, omega_spiral= -20 |(units.kms/units.kpc), amplitude= 650|(units.kms**2/units.kpc), m=4, omega_bar= -50 |(units.kms/units.kpc), mass_bar= 1.1e10 |units.MSun ):
        # Simulation parameters
        self.t_end= t_end
        self.dt_bridge= dt_bridge
        self.method= method
        self.time= 0 |units.Myr
        #galaxy parameters
        self.omega= 0 | (units.kms/units.kpc)
        self.initial_phase= 0
        self.bar_phase= phase_bar
        self.spiral_phase= phase_spiral
        self.omega_spiral= omega_spiral
        self.amplitude= amplitude
        self.rsp= 3.12 |units.kpc
        self.m= m
        self.tan_pitch_angle= 0.227194425
        self.omega_bar= omega_bar
        self.mass_bar= mass_bar
        self.aaxis_bar= 3.12 |units.kpc
        self.axis_ratio_bar= 0.37
        return
    
    def galaxy(self):
        global I
        galaxy= BarAndSpirals3D(redirection='file', redirect_stdout_file="GAL{0}.log".format(I))
        I = I + 1
        galaxy.kinetic_energy=quantities.zero
        galaxy.potential_energy=quantities.zero
        galaxy.parameters.bar_contribution= True
        galaxy.parameters.bar_phase= self.bar_phase
        galaxy.parameters.omega_bar= self.omega_bar
        galaxy.parameters.mass_bar= self.mass_bar
        galaxy.parameters.aaxis_bar= self.aaxis_bar
        galaxy.parameters.axis_ratio_bar= self.axis_ratio_bar 
        galaxy.parameters.spiral_contribution= False
        galaxy.parameters.spiral_phase= self.spiral_phase
        galaxy.parameters.omega_spiral= self.omega_spiral
        galaxy.parameters.amplitude= self.amplitude
        galaxy.parameters.rsp= self.rsp
        galaxy.parameters.m= self.m
        galaxy.parameters.tan_pitch_angle= self.tan_pitch_angle
        galaxy.commit_parameters()
        self.omega= galaxy.parameters.omega_system
        self.initial_phase= galaxy.parameters.initial_phase
        print("INITIAL_PHASE:", self.initial_phase)
        galaxy.kinetic_energy=quantities.zero
        galaxy.potential_energy=quantities.zero
        return galaxy 
        
    def creation_particles_noinertial(self, particles):
        """
        makes trans in a counterclockwise frame.
        If the Galaxy only has bar or only spiral arms, the frame corotates with
        the bar or with the spiral arms. If the Galaxy has bar and spiral arms, the frame corotates with the bar
        """
        no_inertial_system= particles.copy()
        angle= self.initial_phase + self.omega*self.time
        C1= particles.vx + self.omega*particles.y
        C2= particles.vy - self.omega*particles.x
        no_inertial_system.x = particles.x*numpy.cos(angle) + particles.y*numpy.sin(angle)
        no_inertial_system.y = -particles.x*numpy.sin(angle) + particles.y*numpy.cos(angle) 
        no_inertial_system.z = particles.z
        no_inertial_system.vx = C1*numpy.cos(angle) + C2*numpy.sin(angle) 
        no_inertial_system.vy = C2*numpy.cos(angle) - C1*numpy.sin(angle)
        no_inertial_system.vz = particles.vz
        return no_inertial_system    

    def noinertial_to_inertial(self, part_noin, part_in):
        #makes trans in a counterclockwise frame
        angle= self.initial_phase + self.omega*self.time
        C1= part_noin.vx - part_noin.y*self.omega
        C2= part_noin.vy + part_noin.x*self.omega
        part_in.x= part_noin.x*numpy.cos(angle)-part_noin.y*numpy.sin(angle)
        part_in.y= part_noin.x*numpy.sin(angle)+part_noin.y*numpy.cos(angle)
        part_in.z= part_noin.z
        part_in.vx= C1*numpy.cos(angle) - C2*numpy.sin(angle)
        part_in.vy= C1*numpy.sin(angle) + C2*numpy.cos(angle)
        part_in.vz= part_noin.vz
        return

    
    def testing_potential_and_force(self, galaxy, x, y, z):
        dx, dy, dz = 0.001 |units.kpc, 0.001 |units.kpc, 0.001 |units.kpc
        phi1x= galaxy.get_potential_at_point(0 |units.kpc, (x+dx), y, z)
        phi2x= galaxy.get_potential_at_point(0 |units.kpc, (x-dx), y, z)
        f1x= -(phi1x-phi2x)/(2*dx)
        phi1y= galaxy.get_potential_at_point(0 |units.kpc, x, (y+dy), z)
        phi2y= galaxy.get_potential_at_point(0 |units.kpc, x, (y-dy), z)
        f1y= -(phi1y-phi2y)/(2*dy)
        phi1z= galaxy.get_potential_at_point(0 |units.kpc, x, y, (z+dz))
        phi2z= galaxy.get_potential_at_point(0 |units.kpc, x, y, (z-dz))
        f1z= -(phi1z-phi2z)/(2*dz)
        fx,fy,fz= galaxy.get_gravity_at_point(0 |units.kpc, x, y, z)
        print("analytic", "numerical") 
        print(fx.value_in(100*units.kms**2/units.kpc) , f1x.value_in(100*units.kms**2/units.kpc))
        print(fy.value_in(100*units.kms**2/units.kpc) , f1y.value_in(100*units.kms**2/units.kpc))
        print(fz.value_in(100*units.kms**2/units.kpc) , f1z.value_in(100*units.kms**2/units.kpc))
        return

    def get_pos_vel_and_orbit(self, particle_set, pos):
        particle_set.velocity= (-1)*particle_set.velocity
        MW= self.galaxy()
        print("OMEGA:", self.omega.as_quantity_in(1/units.Gyr)) 
        particle_rot= self.creation_particles_noinertial(particle_set)
        gravless= drift_without_gravity(particle_rot)
        
        system= Rotating_Bridge(self.omega, timestep= self.dt_bridge, verbose= False, method= self.method)
        system.add_system(gravless, (MW,), False)
        system.add_system(MW, (), False) # This is to update time inside the interface

        Ei= system.potential_energy+ system.kinetic_energy+ system.jacobi_potential_energy
        energy=[]

        #self.testing_potential_and_force(MW, -1.5|units.kpc, 3|units.kpc, -0.8 |units.kpc)
        

#        pos = particle_set[0].position
        dmin = pos.length()
        tmin = 0|units.Myr
        d = [] | units.kpc
        t = [] | units.Myr
        d.append((pos-particle_set.position).length())
        t.append(self.time)
        while (self.time < self.t_end-self.dt_bridge/2):
            self.time += self.dt_bridge
            system.evolve_model(self.time)
            self.noinertial_to_inertial(particle_rot, particle_set)

            Ef= system.potential_energy+ system.kinetic_energy+ system.jacobi_potential_energy
            dje= (Ef-Ei)/Ei
            energy.append(dje)

            d.append((pos-particle_set.position).length())
            t.append(self.time)
            if d[-1]<dmin:
                dmin = d[-1]
                tmin = self.time
            
            x = particle_set.x
            y = particle_set.y
            
        print("minimum", tmin.in_(units.Myr), dmin.in_(units.parsec))
        bar_angle= self.bar_phase + (self.omega_bar*self.time)
        spiral_angle= self.spiral_phase +  (self.omega_spiral*self.time)
        
           
        return self.time, particle_set[0].x.value_in(units.kpc),  particle_set[0].y.value_in(units.kpc),\
           particle_set[0].z.value_in(units.kpc), particle_set[0].vx.value_in(units.kms), \
           particle_set[0].vy.value_in(units.kms), particle_set[0].vz.value_in(units.kms), \
           bar_angle , spiral_angle, t, d

if __name__ in ('__main__', '__plot__'):

    N = 100
    Rvir = 1| units.parsec
    star_cluster = initialize_star_cluster(N, Rvir)

    filename = "solar_system_with_moons.h5"
    solar_system = read_set_from_file(filename, "hdf5", close_file=True)
    rstar = star_cluster.random_sample(1)
    solar_system.position += rstar.position
    solar_system.velocity += rstar.velocity
    star_cluster.add_particles(solar_system)

    simulation_time= 4600. |units.Myr
    dt_bridge= 5 | units.Myr
    OS= 20 |(units.kms/units.kpc)
    OB= 40 |(units.kms/units.kpc)
    A= 1300 |(units.kms**2/units.kpc)
    M= 1.4e10 |units.MSun
    m=2    

    phi_bar, phi_sp= -0.34906, -0.34906
    inte= IntegrateOrbit(
            t_end= simulation_time, 
            dt_bridge= dt_bridge, 
            phase_bar= phi_bar, phase_spiral= phi_sp, 
            omega_spiral= OS, omega_bar= OB, 
            amplitude= A, m=m, mass_bar= M )

    
    sun_pos = [-8400.0, 0.0, 17.0] | units.parsec
    MWG = inte.galaxy()
    vc = MWG.get_velcirc(sun_pos[0], sun_pos[1], sun_pos[2]) 
    sun_vel = [11.352, (12.24+vc.value_in(units.kms)), 7.41] | units.kms
    
    star_cluster.position += sun_pos
    star_cluster.velocity += sun_vel
    
    cluster_gravity, channels = setup_multi_bridge_system(star_cluster)

    gravity = bridge.Bridge(use_threading=False)
    gravity.add_system(cluster_gravity, (MWG,) )
    t_orb = 2*numpy.pi*sun_pos.length()/sun_vel.length()
    t_end = 100|units.day
    gravity.timestep = min(0.1*t_end, 0.01*t_orb)

    """
    sun_pos = [-8400.0, 0.0, 17.0] | units.parsec
    MWG = MilkyWay_galaxy()
    vc = MWG.vel_circ(sun_pos.length())
    sun_vel = [11.352, (12.24+vc.value_in(units.kms)), 7.41] | units.kms
    
    star_cluster.position += sun_pos
    star_cluster.velocity += sun_vel
    
    cluster_gravity, channels = setup_multi_bridge_system(star_cluster)

    gravity = bridge.Bridge(use_threading=False)
    gravity.add_system(cluster_gravity, (MWG,) )
    t_orb = 2*numpy.pi*sun_pos.length()/sun_vel.length()
    t_end = 100|units.day
    gravity.timestep = min(0.1*t_end, 0.01*t_orb)
    """

    model_time, cpu_time, DDE = integrate_solar_system(gravity, t_end=t_end)
    for ci in channels:
        ci.copy()
    gravity.stop()
    
    plot_xy(star_cluster)
    plot(model_time, DDE, "$t$ [day]", "solar_system_nbb_tday")
    plot(cpu_time, DDE, "$t_{cpu}$ [s]", "solar_system_nbb_tcpu")
