import numpy as np
import os

from amuse.ext.orbital_elements import new_binary_from_orbital_elements
from amuse.lab import constants, Particles, units
from amuse.lab import new_salpeter_mass_distribution, write_set_to_file

def extract_data(array, line):
    """Function to read S-Stars_Params file"""
    index = line.find('±')
    array = np.concatenate((array, float(line[:index])), axis=None)
    return array

def make_systems(no_planets):
    """Initialise S-Stars and Planets"""
    # Kinematic data taken from arXiv:1611.09144
    # Mass taken from arXiv:1708.06353 & arXiv:2202.10827
    file_name = "S-Stars_Params.txt"
    skip_first_line = False

    name = [ ]
    mass = [ ]
    semi_major = [ ]
    eccentricity = [ ]
    inclination = [ ]
    long_asc_node = [ ]
    arg_periapsis = [ ]
    with open(file_name, "r") as data_file:
        for line in data_file:
            if not skip_first_line:
                skip_first_line = True
                continue
            df = line.strip().split(",")
            name = np.concatenate((name, df[0][:5]), axis=None)
            mass = np.concatenate((mass, float(df[1][:5])), axis=None)
            semi_major = extract_data(semi_major, df[2])  # arcsecs
            eccentricity = extract_data(eccentricity, df[3])
            inclination = extract_data(inclination, df[4])
            long_asc_node = extract_data(long_asc_node, df[5])
            arg_periapsis = extract_data(arg_periapsis, df[6])

    particle_set = Particles()
    SMBH = Particles(1)
    SMBH.mass = 4.3e6 | units.MSun  # arXiv:2112.07478
    SMBH.type = "SMBH"
    SMBH.radius = 10*(2*constants.G*SMBH.mass)/(constants.c**2)
    SMBH.position = [0,0,0] | units.au
    SMBH.velocity = [0,0,0] | units.kms
    particle_set.add_particle(SMBH)

    SMBH_dist = 8.249 | units.kpc  # arXiv:2004.07187
    asec_to_rad = (1|units.arcsec)/(1|units.rad)
    for star_ in range(len(name)):
        star_smbh_bin = Particles(no_planets+1)
        star_smbh_bin[0].name = name[star_]
        star_smbh_bin[0].mass = mass[star_] | units.MSun
        
        semi_major_axis = SMBH_dist*np.arctan(semi_major[star_]*asec_to_rad)
        semi_major[star_] = semi_major_axis.value_in(units.au)
        binary_set = new_binary_from_orbital_elements(
                        mass1=SMBH.mass, mass2=star_smbh_bin[0].mass,
                        semimajor_axis=semi_major_axis,
                        eccentricity=eccentricity[star_],
                        inclination=inclination[star_],
                        longitude_of_the_ascending_node=long_asc_node[star_],
                        argument_of_periapsis=arg_periapsis[star_],
                        G=constants.G)
        smbh_temp = binary_set[binary_set.mass==SMBH.mass]
        star = binary_set-smbh_temp
        star.position -= smbh_temp.position
        star.velocity -= smbh_temp.velocity
        star.type = name[star_] 
        particle_set.add_particle(star)
    
    nsyst = 0
    for star_ in particle_set[1:]:
        temp_planets = Particles(no_planets)
        tot_mass = (1e-3*star_.mass)
        while True:
            planet_masses = new_salpeter_mass_distribution(no_planets, 
                                                     0.1|units.MEarth, 
                                                     tot_mass, 
                                                     alpha=-1.3  # arXiv:1105.3544
                                                     )
            if np.sum(planet_masses)/tot_mass >= 0.5:
                break
        temp_planets.mass = np.sort(planet_masses)

        k = 10
        max_hill_radius = (semi_major[nsyst]*1|units.au)*((star_.mass)/(3*SMBH.mass))**(1./3.)
        prev_semi = 0.5 | units.au
        prev_mass = temp_planets[1].mass
        eccentricity = 0
        inclination = np.arccos(1-2*np.random.uniform(0,1, 1)) | units.rad

        planetary_system = Particles()
        piter = 0 
        nsyst += 1
        for p in temp_planets:
            numerator = prev_semi*(k/2*((prev_mass+p.mass)/(3*star_.mass))**(1/3)+1)
            denominator = (1-k/2*((prev_mass+p.mass)/(3*star_.mass))**(1/3))
            semimajor = (numerator+prev_semi)/denominator 
            long_asc_node = np.random.uniform(0, 2*np.pi, 1) | units.rad
            true_anomaly = np.random.uniform(0, 2*np.pi, 1)
            arg_periapsis = np.random.uniform(0, 2*np.pi, 1) | units.rad
            if semimajor <= 0.75*max_hill_radius:
                piter += 1
                binary_set = new_binary_from_orbital_elements(
                                    mass1=p.mass, mass2=star_.mass,
                                    semimajor_axis=semimajor,
                                    eccentricity=eccentricity,
                                    inclination=inclination,
                                    longitude_of_the_ascending_node=long_asc_node,
                                    true_anomaly=true_anomaly,
                                    argument_of_periapsis=arg_periapsis,
                                    G=constants.G)
                host = binary_set[binary_set.mass==binary_set.mass.max()]
                binary_set.position += (star_.position - host.position)
                binary_set.velocity += (star_.velocity - host.velocity)
                binary_set.type = "Planet "+str(piter)
                binary_set -= host
                planetary_system.add_particle(binary_set)
            prev_mass = p.mass
            prev_semi = semimajor
        from amuse.community.mercury.interface import Mercury
        from amuse.units import nbody_system
        from amuse.ext.orbital_elements import orbital_elements_from_binary
                
        conv = nbody_system.nbody_to_si(planetary_system.mass.sum(), 100 | units.AU)
        code = Mercury(conv)
        planetary_system.add_particle(star_)
        planetary_system.move_to_center()
        code.particles.add_particle(planetary_system)
        
        from amuse.ext.orbital_elements import orbital_elements_from_binary
        print(code.particles.position.in_(units.au))
        for p in planetary_system:
            bin_sys = Particles()
            bin_sys.add_particle(p)
            bin_sys.add_particle(planetary_system[-1])
            kepler_elements = orbital_elements_from_binary(bin_sys, G=constants.G)
            print(kepler_elements[2].in_(units.AU), kepler_elements[3])
            
        import matplotlib.pyplot as plt
        plt.scatter(code.particles.x.value_in(units.AU), code.particles.y.value_in(units.AU))
        plt.show()
        code.evolve_model(0.1 | units.Gyr)
        plt.scatter(code.particles.x.value_in(units.AU), code.particles.y.value_in(units.AU))
        plt.scatter(code.particles[code.particles.mass == code.particles.mass.max()].x.value_in(units.AU), 
                    code.particles[code.particles.mass == code.particles.mass.max()].y.value_in(units.AU))
        plt.show()
        STOP
            
        planetary_system.syst_id = nsyst
        star_.syst_id = nsyst
        particle_set.add_particle(planetary_system)
        
    
    output_dir = os.path.join("initial_particles", "init_particle_set")
    write_set_to_file(particle_set, output_dir, "amuse", 
                      close_file=True, overwrite_file=True)


make_systems(5)