import glob
from natsort import natsorted
import numpy as np
import os
import time as cpu_time

from amuse.lab import constants, Particles, read_set_from_file, write_set_to_file
from amuse.units import units, nbody_system
from amuse.units.optparse import OptionParser

from src.environment_functions import galactic_frame, set_parent_radius
from src.hierarchical_particles import HierarchicalParticles
from src.nemesis import Nemesis

# Constants
START_TIME = cpu_time.time()
MIN_EVOL_MASS = 0.08 | units.MSun


def create_output_directories(sim_dir, run_idx) -> str:
    """
    Creates relevant directories for output data
    
    Args:
        sim_dir (str): Simulation directory path
        run_idx (int):  Index of specific run
    Returns:
        String: Directory path for specific run
    """
    config_name = f"Nrun{run_idx}"
    dir_path = os.path.join(sim_dir, config_name)
    if not os.path.exists(os.path.join(dir_path, "*")):
        os.mkdir(dir_path+"/")
        subdir = ["event_data", "collision_snapshot", 
                  "data_process", "simulation_stats", 
                  "simulation_snapshot", "ejected_particles"]
        for path in subdir:
            os.makedirs(os.path.join(dir_path, path))
            
    return dir_path

def load_particle_set(sim_dir, run_idx) -> Particles:
    """
    Load particle set from file
    
    Args:
        sim_dir (String):  Path to initial conditions
        run_idx (Int):  Index of run
    Returns:
        Particles:  Initial conditions of particle set wished to simulate
    """
    particle_set_dir = os.path.join(sim_dir, "initial_particles", "*")
    particle_sets = natsorted(glob.glob(particle_set_dir))
    if run_idx >= len(particle_sets):
        raise IndexError(f"Error: Run index {run_idx} out of range.")
    
    file = particle_sets[run_idx]
    if not os.path.isfile(file):
        raise FileNotFoundError(f"Error: No particle set found in {particle_set_dir}")
    
    particle_set = read_set_from_file(file)
    particle_set.coll_events = 0
    particle_set.move_to_center()
    
    if len(particle_set) == 0:
        raise Exception(f"Error: Particle set {particle_set_dir} is empty.")
    
    return particle_set

def configure_galactic_frame(particle_set) -> Particles:
    """
    Shift particle set to galactic frame. Currently assumes solar orbit.
    
    Args:
        particle_set (Particles):  The particle set
    Returns:
        Particles: Particle set with galactocentric coordinates
    """
    return galactic_frame(particle_set, 
                          dx=-8.4 | units.kpc, 
                          dy=0.0 | units.kpc, 
                          dz=17  | units.pc,
                          dvx=11.352 | units.kms,
                          dvy=12.25 | units.kms,
                          dvz=7.41 | units.kms)
    
def identify_parents(particle_set) -> list:
    """
    Identify parents in particle set. These are either:
        - Isolated particles (syst_id < 0)
        - Hosts of children system (max mass in system)
    
    Args:
        particle_set (Particles):  The particle set
    Returns:
        List:  Index 0 contains major bodies, index 1 test particles
    """
    isolated_particles = particle_set[particle_set.syst_id <= 0]
    major_bodies = isolated_particles[isolated_particles.mass != (0. | units.kg)]
    test_particles = isolated_particles - major_bodies
    for system_id in np.unique(particle_set.syst_id):
        if system_id > 0:
            system = particle_set[particle_set.syst_id == system_id]
            hosting_body = system[system.mass.argmax()]
            major_bodies += hosting_body
            
    return [major_bodies, test_particles]

def run_simulation(sim_dir, tend, code_dt, dtbridge, 
                   dt_diag, par_nworker, dE_track, 
                   gal_field, star_evol, verbose) -> None:
    """
    Run simulation and output data.
    
    Args:
        sim_dir (String):  Path to initial conditions
        tend (Float):  Simulation end time
        code_dt (Float):  Gravitational integrator internal timestep
        dtbridge (Float):  Bridge timestep
        dt_diag (Float):  Diagnostic time step
        par_nworker (Int):  Number of workers for parent code
        dE_track (Boolean):  Flag turning on energy error tracker
        gal_field (Boolean):  Flag turning on galactic field or not
        star_evol (Boolean):  Flag turning on stellar evolution or not
        verbose (Boolean):  Flag turning on print statements or not
    """
    run_idx = len(glob.glob(sim_dir+"/Nrun*"))
    EPS = 1.e-8
    
    # Setup directory paths
    dir_path = create_output_directories(sim_dir, run_idx)
    coll_path = os.path.join(dir_path, "collision_snapshot")
    ejected_dir = os.path.join(dir_path, "ejected_particles")
    snapdir_path = os.path.join(dir_path, "simulation_snapshot")

    particle_set = load_particle_set(sim_dir, run_idx)
    if (gal_field):
        particle_set = configure_galactic_frame(particle_set)
        
    parent_particles = particle_set[(particle_set.type != "JMO") &
                                    (particle_set.type != "ASTEROID") &
                                    (particle_set.type != "PLANET")]
    Rvir = parent_particles.virial_radius()
    vdisp = np.sqrt((constants.G*parent_particles.mass.sum())/Rvir)
    conv_par = nbody_system.nbody_to_si(np.sum(major_bodies.mass), Rvir)
    
    major_bodies, test_particles = identify_parents(particle_set)
    isolated_systems = major_bodies[major_bodies.syst_id < 0]
    bounded_systems = major_bodies[major_bodies.syst_id > 0]
    parents = HierarchicalParticles(isolated_systems)
    
    # Setting up system
    nemesis = Nemesis(MIN_EVOL_MASS, conv_par, 
                      dtbridge, coll_path, 
                      ejected_dir, code_dt, EPS,
                      par_nworker, dE_track, 
                      star_evol, gal_field,
                      verbose)
    for id_ in np.unique(bounded_systems.syst_id):
        children = particle_set[particle_set.syst_id == id_]
        newparent = nemesis.particles.add_subsystem(children)
        newparent.radius = set_parent_radius(newparent.mass)
    nemesis.particles.add_particles(parents)
    nemesis.asteroids = test_particles
    nemesis.commit_particles()
    
    par_nworker = int(len(major_bodies) // 500 + 1)
    min_radius = nemesis.particles.radius.min()
    typical_crosstime = 2.*(min_radius/vdisp)
    if (verbose):
        print(f"dt= {dtbridge.in_(units.yr)}")
        print(f"Number of steps= {tend/dtbridge}")
        print(f"Minimum children system radius= {min_radius.in_(units.au)}")
        print(f"Dispersion velocity= {vdisp.in_(units.kms)}")
        print(f"Min. system crossing time= {typical_crosstime.in_(units.kyr)}")
        print(f"Total number of snapshots: {tend/dt_diag}")
    if dtbridge > (10/code_dt)*typical_crosstime:
        raise ValueError("!!! Warning: dt > (10/code_dt)*Typical System Crossing Time !!!")
        
    if (nemesis._dE_track):
        energy_arr = [ ]
    
    allparts = nemesis.particles.all()
    allparts.add_particles(nemesis.asteroids)
    write_set_to_file(
        allparts.savepoint(0 | units.Myr), 
        os.path.join(snapdir_path, "snap_0"), 
        'amuse', close_file=True, overwrite_file=True
    )
    
    t = 0. | units.yr
    t_diag = dt_diag
    snapshot_no = 0
    prev_step = nemesis.dt_step
    snap_cpu_time = cpu_time.time()
    while t < tend:
        if (verbose):
            t0 = cpu_time.time()
        
        t += dtbridge
        if t == dtbridge and (dE_track):
            E0 = nemesis._calculate_total_energy()
        
        while nemesis.model_time < t*(1. - EPS):
            nemesis.evolve_model(t)
            
        if (nemesis.model_time >= t_diag) and (nemesis.dt_step != prev_step):
            print(f"Saving snap. T= {t.in_(units.yr)}")
            print(f"Time taken: {cpu_time.time() - snap_cpu_time}")
            snap_cpu_time = cpu_time.time()
            
            t_diag += dt_diag
            snapshot_no += 1
            allparts = nemesis.particles.all()
            allparts.add_particles(nemesis.asteroids)
            
            fname = os.path.join(snapdir_path, f"snap_{snapshot_no}")
            write_set_to_file(
                allparts.savepoint(0 | units.Myr),  fname, 
                'amuse', close_file=True, overwrite_file=True
            )
          
        if (dE_track) and (prev_step != nemesis.dt_step):
            E1 = nemesis.calculate_total_energy()
            E1 += nemesis.corr_energy
            energy_arr.append(abs((E1-E0)/E0))
            
            prev_step = nemesis.dt_step
            print(f"t = {t.in_(units.Myr)}, dE = {abs((E1-E0)/E0)}")
            
        prev_step = nemesis.dt_step
        
        if (verbose):
            t1 = cpu_time.time()
            print(f"Step took {t1-t0} seconds")
        
    allparts = nemesis.particles.all()
    allparts.add_particles(nemesis.asteroids)
    write_set_to_file(
        allparts.savepoint(0 | units.Myr), 
        os.path.join(snapdir_path, f"snap_{snapshot_no+1}"), 
        'amuse', close_file=True, overwrite_file=True
    )
    
    print("...Simulation Ended...")
    nemesis._stellar_code.stop()  
    nemesis._parent_code.stop()
    for code in nemesis.subcodes.values():
        code.stop()

    # Store simulation statistics
    sim_time = (cpu_time.time() - START_TIME)/60.
    fname = os.path.join(dir_path, 'simulation_stats', f'sim_stats_{run_idx}.txt')
    with open(fname, 'w') as f:
        f.write(f"Total CPU Time: {sim_time} minutes \
                \nEnd Time: {t.in_(units.Myr)} \
                \nTime step: {dtbridge.in_(units.Myr)}")
    f.close()
    
    with open(os.path.join(dir_path, "energy_error.csv"), 'w') as f:
        f.write(f"Energy error: {energy_arr}")
    f.close()
     
def new_option_parser():
    result = OptionParser()
    result.add_option("--par_nworker", 
                      dest="par_nworker", 
                      type="int", 
                      default=1,
                      help="Number of workers for parent code")
    result.add_option("--tend", 
                      dest="tend", 
                      type="float", 
                      unit=units.Myr, 
                      default=20. | units.Myr,
                      help="End time of simulation")
    result.add_option("--tbridge", 
                      dest="tbridge", 
                      type="float", 
                      unit=units.yr, 
                      default=1000. | units.yr,
                      help="Bridge timestep")
    result.add_option("--code_dt", 
                      dest="code_dt", 
                      type="float", 
                      default=2.**-3. ,
                      help="Gravitational integrator internal timestep")
    result.add_option("--dt_diag", 
                      dest="dt_diag", 
                      type="int", 
                      unit=units.kyr, 
                      default=100 | units.kyr,
                      help="Diagnostic time step")
    result.add_option("--gal_field", 
                      dest="gal_field", 
                      action="store_true", 
                      default=True,
                      help="Flag to turn on galactic field")
    result.add_option("--dE_track", 
                      dest="dE_track", 
                      action="store_true", 
                      default=False,
                      help="Flag to turn on energy error tracker")
    result.add_option("--star_evol", 
                      dest="star_evol", 
                      action="store_true", 
                      default=True,
                      help="Flag to turn on stellar evolution")
    result.add_option("--verbose", 
                      dest="verbose", 
                      action="store_true", 
                      default=True,
                      help="Flag to turn on print statements")
    
    return result
        
if __name__ == "__main__":
    # data_dir = "examples/S-Stars"
    data_dir = "examples/ejecting_suns"
    config_idx = 3
    configurations = glob.glob(os.path.join(data_dir, "sim_data", "*"))
    config_choice = natsorted(configurations)[config_idx]
    
    o, args = new_option_parser().parse_args()
    
    run_simulation(
        sim_dir=config_choice, 
        tend=o.tend, 
        dtbridge=o.tbridge,
        code_dt=o.code_dt,
        dt_diag=o.dt_diag,
        par_nworker=o.par_nworker, 
        gal_field=o.gal_field, 
        dE_track=o.dE_track, 
        star_evol=o.star_evol, 
        verbose=o.verbose
    )
    