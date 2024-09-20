import glob
import itertools
from natsort import natsorted
import numpy as np
import os
import sys
import time as cpu_time

from amuse.lab import constants, read_set_from_file, write_set_to_file
from amuse.units import units, nbody_system
from amuse.units.optparse import OptionParser

from src.environment_functions import galactic_frame
from src.hierarchical_particles import HierarchicalParticles
from src.nemesis import Nemesis


def create_output_directories(sim_dir, run_idx) -> str:
    """
    Creates relevant directories for output data
    
    Args:
        sim_dir (String): Simulation directory path
        run_idx (Int):    Index of specific run
    
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

def run_simulation(sim_dir, tend, code_dt, eta, 
                   dt_diag, par_nworker, dE_track, 
                   gal_field, star_evol):
    """
    Run simulation and output data.
    
    Args:
        sim_dir (String):  Path to initial conditions
        tend (Float):  Simulation end time
        code_dt (Float):  Gravitational integrator internal timestep
        eta (Float):  Parameter tuning dt
        dt_diag (Float):  Diagnostic time step
        par_nworker (Int):  Number of workers for parent code
        dE_track (Boolean):  Flag turning on energy error tracker
        gal_field (Boolean):  Flag turning on galactic field or not
        star_evol (Boolean):  Flag turning on stellar evolution or not
    """
    run_idx = len(glob.glob(sim_dir+"/Nrun*"))
    
    # Setup directory paths
    dir_path = create_output_directories(sim_dir, run_idx)
    coll_path = os.path.join(dir_path, "collision_snapshot")
    ejected_dir = os.path.join(dir_path, "ejected_particles")
    snapdir_path = os.path.join(dir_path, "simulation_snapshot")

    # Load particle set
    particle_set_dir = os.path.join(sim_dir, "initial_particles", "*")
    particle_sets = natsorted(glob.glob(particle_set_dir))
    particle_set = read_set_from_file(particle_sets[run_idx])
    particle_set -= particle_set[particle_set.syst_id > 3]  # Testing purpose
    particle_set.coll_events = 0

    parent_particles = particle_set[(particle_set.type != "JMO") 
                                    & (particle_set.type != "ASTEROID") 
                                    & (particle_set.type != "PLANET")]
    Rvir = parent_particles.virial_radius()
    vdisp = np.sqrt((constants.G*parent_particles.mass.sum())/Rvir)
    
    if (gal_field): # Sun's orbit
        particle_set = galactic_frame(particle_set, 
                                      dx=-8.4 | units.kpc, 
                                      dy=0.0 | units.kpc, 
                                      dz=17  | units.pc,
                                      dvx=11.352 | units.kms,
                                      dvy=12.25 | units.kms,
                                      dvz=7.41 | units.kms)
    
    isolated_particles = particle_set[particle_set.syst_id < 0]
    major_bodies = isolated_particles[isolated_particles.mass != (0. | units.kg)]
    test_particles = isolated_particles - major_bodies
    for system_id in np.unique(particle_set.syst_id):  # Identify parents
        if system_id > -1:
            system = particle_set[particle_set.syst_id == system_id]
            hosting_body = system[system.mass == max(system.mass)]
            major_bodies += hosting_body
    
    parents = HierarchicalParticles(major_bodies)
    initial_systems = parents[parents.syst_id > 0]
    for id_ in np.unique(initial_systems.syst_id):
        children = particle_set[particle_set.syst_id == id_]
        host = parents[parents.syst_id == id_][0]
        parents.assign_subsystem(children, host, 
                                 relative=True, 
                                 recenter=False)
    
    par_nworker = int(len(major_bodies) // 1000 + 1)
    conv_par = nbody_system.nbody_to_si(np.sum(major_bodies.mass), 
                                        major_bodies.virial_radius())
    conv_child = nbody_system.nbody_to_si(np.mean(parents.mass), 
                                          np.mean(parents.radius))
    dt = eta * tend

    # Setting up system
    nemesis = Nemesis(conv_par, conv_child, dt, 
                      code_dt, par_nworker, dE_track, 
                      star_evol, gal_field)
    nemesis.min_mass_evol = MIN_EVOL_MASS
    nemesis.particles.add_particles(parents)
    nemesis.commit_particles()
    nemesis.coll_dir = coll_path
    nemesis.ejected_dir = ejected_dir
    nemesis.test_particles = test_particles
    
    min_radius = nemesis.particles.radius.min()
    typical_crosstime = 2.*(min_radius/vdisp)
    print(f"dt= {dt.in_(units.yr)}", end="  ")
    print(f"Minimum children system radius= {min_radius.in_(units.au)}", end="  ")
    print(f"Dispersion velocity= {vdisp.in_(units.kms)}", end="  ")
    print(f"Min. system crossing time= {typical_crosstime.in_(units.kyr)}")
    print(f"Total number of snapshots: {tend/dt_diag}")
    if dt > 5*typical_crosstime:
        print("!!! Warning: dt > 5*Typical System Crossing Time !!!")
        sys.exit(1)
        
    if (nemesis.dE_track):
        energy_arr = [ ]
        E0 = nemesis.calculate_total_energy()
        nemesis.E0 = E0
    
    allparts = nemesis.particles.all()
    write_set_to_file(
        allparts.savepoint(0 | units.Myr), 
        os.path.join(snapdir_path, "snap_0"), 
        'amuse', close_file=True, overwrite_file=True
    )
    
    t = 0. | units.yr
    t_diag = 0. | units.yr
    snapshot_no = 0
    prev_step = nemesis.dt_step
    while t < tend:
        t += dt
        
        while nemesis.parent_code.model_time < t:
            nemesis.evolve_model(t)
            
        if nemesis.model_time > t_diag:
            print(f"Saving snap. T= {t.in_(units.yr)}")
            t_diag += dt_diag
            snapshot_no += 1
            allparts = nemesis.particles.all()
            
            fname = os.path.join(snapdir_path, f"snap_{nemesis.dt_step}")
            write_set_to_file(
                allparts.savepoint(0 | units.Myr),  fname, 
                'amuse', close_file=True, overwrite_file=True
            )
          
        if (nemesis.dE_track) and (prev_step != nemesis.dt_step):
            E1 = nemesis.calculate_total_energy()
            E1 += nemesis.corr_energy
            energy_arr.append(abs((E1-E0)/E0))
            
            prev_step = nemesis.dt_step
            print(f"t = {t.in_(units.Myr)}, dE = {abs((E1-E0)/E0)}")
        prev_step = nemesis.dt_step
      
    print("...Simulation Ended...")
    nemesis.stellar_code.stop()  
    nemesis.parent_code.stop()
    for code in nemesis.subcodes.values():
        code.stop()

    # Store simulation statistics
    sim_time = (cpu_time.time() - START_TIME)/60.
    fname = os.path.join(dir_path, 'simulation_stats', f'sim_stats_{run_idx}.txt')
    with open(fname, 'w') as f:
        f.write(f"Total CPU Time: {sim_time} minutes \
                \nEnd Time: {t.in_(units.Myr)} \
                \nTime step: {dt.in_(units.Myr)} \
                \nInitial Typical tcross: {typical_crosstime.in_(units.yr)}")
     
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
    result.add_option("--eta", 
                      dest="eta", 
                      type="float", 
                      default=1e-5 ,
                      help="Parameter tuning dt")
    result.add_option("--code_dt", 
                      dest="code_dt", 
                      type="float", 
                      default=0.01 ,
                      help="Gravitational integrator internal timestep")
    result.add_option("--dt_diag", 
                      dest="dt_diag", 
                      type="int", 
                      unit=units.kyr, 
                      default=1 | units.kyr,
                      help="Diagnostic time step")
    result.add_option("--gal_field", 
                      dest="gal_field", 
                      action="store_true", 
                      default=False,
                      help="Flag to turn on galactic field")
    result.add_option("--dE_track", 
                      dest="dE_track", 
                      action="store_true", 
                      default=False,
                      help="Flag to turn on energy error tracker")
    result.add_option("--star_evol", 
                      dest="star_evol", 
                      action="store_true", 
                      default=False,
                      help="Flag to turn on stellar evolution")
    
    return result
        
if __name__ == "__main__":
    START_TIME = cpu_time.time()
    MIN_EVOL_MASS = 0.08 | units.MSun
    
    # data_dir = "examples/S-Stars"
    data_dir = "examples/ejecting_suns"
    config_idx = 1
    configurations = glob.glob(os.path.join(data_dir, "sim_data", "*"))
    config_choice = natsorted(configurations)[config_idx]
    
    o, args = new_option_parser().parse_args()
    
    run_simulation(
        sim_dir=config_choice, 
        tend=o.tend, 
        eta=o.eta,
        code_dt=o.code_dt,
        dt_diag=o.dt_diag,
        par_nworker=o.par_nworker, 
        gal_field=o.gal_field, 
        dE_track=o.dE_track, 
        star_evol=o.star_evol, 
    )
    
