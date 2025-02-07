import glob
from natsort import natsorted
import numpy as np
import os
import time as cpu_time

from amuse.io import read_set_from_file, write_set_to_file
from amuse.lab import Particles
from amuse.units import units, nbody_system
from amuse.units.optparse import OptionParser

from src.environment_functions import galactic_frame, set_parent_radius
from src.hierarchical_particles import HierarchicalParticles
from src.nemesis import Nemesis

# Constants
START_TIME = cpu_time.time()
MIN_EVOL_MASS = 0.08 | units.MSun

    
def create_output_directories(sim_dir: str) -> str:
    """
    Creates directories for output
    
    Args:
        sim_dir (str):  Simulation directory path
    Returns:
        String: Directory path for specific run
    """
    config_name = f"Nrun{RUN_IDX}"
    directory_path = os.path.join(sim_dir, config_name)
    if not os.path.exists(directory_path):
        os.mkdir(directory_path+"/")
        subdir = ["event_data", "collision_snapshot",
                  "data_process", "simulation_stats",
                  "simulation_snapshot"]
        for path in subdir:
            os.makedirs(os.path.join(directory_path, path), exist_ok=True)
        
        collision_output = os.path.join(directory_path, "collision_snapshot", "output")
        os.makedirs(collision_output, exist_ok=True)
            
    return directory_path

def load_particle_set(sim_dir: str) -> Particles:
    """
    Load particle set from file
    
    Args:
        sim_dir (str):  Path to initial conditions
    Returns:
        Particles:  Initial particle set properties
    """
    particle_set_dir = os.path.join(sim_dir, "initial_particles", "*")
    particle_sets = natsorted(glob.glob(particle_set_dir))
    
    file = particle_sets[RUN_IDX]
    particle_set = read_set_from_file(file)
    if len(particle_set) == 0:
        raise ValueError(f"Error: Particle set {file} is empty.")
    
    particle_set.coll_events = 0
    particle_set.move_to_center()
    
    return particle_set

def configure_galactic_frame(particle_set: Particles) -> Particles:
    """
    Shift particle set to galactic frame. Currently assumes solar orbit.
    
    Args:
        particle_set (particles):  The particle set
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
    
def identify_parents(particle_set: Particles) -> list:
    """
    Identify parents in particle set. These are either:
        - Isolated particles (syst_id < 0)
        - Hosts of children system (max mass in system)
    
    Args:
        particle_set (particles):  The particle set
    Returns:
        list:  Index 0 contains major bodies, index 1 test particles
    """
    major_bodies = particle_set[particle_set.syst_id <= 0]
    for system_id in np.unique(particle_set.syst_id):
        if system_id > 0:
            system = particle_set[particle_set.syst_id == system_id]
            hosting_body = system[system.mass.argmax()]
            major_bodies += hosting_body
            
    return major_bodies

def setup_simulation(particle_set: Particles) -> tuple:
    """
    Setup simulation directories and load particle set.

    Args:
        particle_set (particles): The particle set

    Returns:
        tuple: (directory_path, snapshot_path, particle_set)
    """
    sim_dir = particle_set.split("initial_particles/")[0]
    directory_path = create_output_directories(sim_dir)
    snapshot_path = os.path.join(directory_path, "simulation_snapshot")
    particle_set = load_particle_set(sim_dir)
    
    return directory_path, snapshot_path, particle_set

def run_simulation(particle_set: Particles, tend: float, 
                   code_dt: float, dtbridge, dt_diag: float, 
                   dE_track: bool, gal_field: bool, star_evol: bool, 
                   verbose: bool) -> None:
    """
    Run simulation and output data.
    
    Args:
        particle_set (String):  Path to initial conditions
        tend (float):  Simulation end time
        code_dt (float):  Gravitational integrator internal timestep
        dtbridge (float):  Bridge timestep
        dt_diag (float):  Diagnostic time step
        dE_track (boolean):  Flag turning on energy error tracker
        gal_field (boolean):  Flag turning on galactic field or not
        star_evol (boolean):  Flag turning on stellar evolution or not
        verbose (boolean):  Flag turning on print statements or not
    """
    EPS = 1.e-8
    
    directory_path, snapshot_path, particle_set = setup_simulation(particle_set)
    coll_dir = os.path.join(directory_path, "collision_snapshot")
    if (gal_field):
        particle_set = configure_galactic_frame(particle_set)
        
    Rvir = particle_set[particle_set.mass > (0. | units.kg)].virial_radius()
    
    major_bodies = identify_parents(particle_set)
    isolated_systems = major_bodies[major_bodies.syst_id < 0]
    bounded_systems = major_bodies[major_bodies.syst_id > 0]
    conv_par = nbody_system.nbody_to_si(np.sum(major_bodies.mass), Rvir)
    par_nworker = max(1, len(isolated_systems) // 500)
    
    # Setting up system
    parents = HierarchicalParticles(isolated_systems)
    nemesis = Nemesis(min_stellar_mass=MIN_EVOL_MASS, 
                      par_conv=conv_par, 
                      dtbridge=dtbridge, 
                      coll_dir=coll_dir,
                      eps=EPS, 
                      code_dt=code_dt, 
                      par_nworker=par_nworker, 
                      dE_track=dE_track, 
                      star_evol=star_evol, 
                      gal_field=gal_field,
                      verbose=verbose)
    for id_ in np.unique(bounded_systems.syst_id):
        children = particle_set[particle_set.syst_id == id_]
        newparent = nemesis.particles.add_subsystem(children)
        newparent.radius = set_parent_radius(newparent.mass)
        
    nemesis.particles.add_particles(parents)
    nemesis.commit_particles()
    nemesis._split_subcodes()  # Check for any splits at t=0
        
    if (nemesis._dE_track):
        energy_arr = [ ]
        E0 = nemesis._calculate_total_energy()
    
    allparts = nemesis.particles.all()
    write_set_to_file(
        allparts.savepoint(0 | units.Myr), 
        os.path.join(snapshot_path, "snap_0"), 
        'amuse', close_file=True, overwrite_file=True
    )
    
    if (verbose):
        print(
            f"Simulation Parameters:\n"
            f"  Total number of particles: {len(particle_set)}\n"
            f"  Total number of initial children: {id_}\n"
            f"  Bridge timestep: {dtbridge.in_(units.yr)}\n"
            f"  End time: {tend.in_(units.Myr)}\n"
            f"  Galactic field: {gal_field}"
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
        while nemesis.model_time < t*(1. - EPS):
            nemesis.evolve_model(t)
            
        if (nemesis.model_time >= t_diag) and (nemesis.dt_step != prev_step):
            if (verbose):
                print(
                    f"Saving snapshot {snapshot_no} at time {t.in_(units.yr)}\n"
                    f"Time taken: {cpu_time.time() - snap_cpu_time}"
                )
                snap_cpu_time = cpu_time.time()
            
            t_diag += dt_diag
            snapshot_no += 1
            allparts = nemesis.particles.all()
            
            fname = os.path.join(snapshot_path, f"snap_{snapshot_no}")
            write_set_to_file(
                allparts.savepoint(nemesis.model_time),  
                fname, 'amuse', 
                close_file=True, 
                overwrite_file=True
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
    write_set_to_file(
        allparts.savepoint(nemesis.model_time), 
        os.path.join(snapshot_path, f"snap_{snapshot_no+1}"), 
        'amuse', close_file=True, overwrite_file=True
    )
    
    print("...Simulation Ended...")
    
    # Store simulation statistics
    sim_time = (cpu_time.time() - START_TIME)/60.
    fname = os.path.join(directory_path, 'simulation_stats', f'sim_stats_{RUN_IDX}.txt')
    with open(fname, 'w') as f:
        f.write(f"Total CPU Time: {sim_time} minutes \
                \nEnd Time: {t.in_(units.Myr)} \
                \nTime step: {dtbridge.in_(units.Myr)}")
    
    # Kill all workers
    for parent, code in nemesis.subcodes.items():
        pid = nemesis._pid_workers[parent]
        nemesis.resume_worker(pid)
        code.stop()
        
    nemesis._stellar_code.stop()  
    nemesis._parent_code.stop()
    
    if (dE_track):
        with open(os.path.join(directory_path, "energy_error.csv"), 'w') as f:
            f.write(f"Energy error: {energy_arr}")
     
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
                      default=1.e-1,
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
    result.add_option("--run_idx",
                      dest="run_idx",
                      type="int",
                      default=0,
                      help="Index of specific run")
    
    return result
        
if __name__ == "__main__":
    o, args = new_option_parser().parse_args()
    
    RUN_IDX = o.run_idx
    initial_particles = natsorted(glob.glob("examples/basic_cluster/initial_particles/*"))
    try:
        particle_set = initial_particles[RUN_IDX]
    except IndexError:
        raise IndexError(f"Error: Run index {RUN_IDX} out of range. \n"
                         f"Available particle sets: {initial_particles}.")
    
    run_simulation(
        particle_set=particle_set, 
        tend=o.tend, 
        dtbridge=o.tbridge,
        code_dt=o.code_dt,
        dt_diag=o.dt_diag,
        gal_field=o.gal_field, 
        dE_track=o.dE_track, 
        star_evol=o.star_evol, 
        verbose=o.verbose
    )