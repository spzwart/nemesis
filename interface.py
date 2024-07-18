import glob
from natsort import natsorted
import numpy as np
import os
import time as cpu_time

from amuse.lab import Particles, read_set_from_file, write_set_to_file
from amuse.units import units, nbody_system
from src.environment_functions import galactic_frame
from src.hierarchical_particles import HierarchicalParticles
from src.nemesis import Nemesis


def run_simulation(sim_dir, tend, eta, code_dt,
                   par_n_worker, dE_track, gal_field, 
                   star_evol
                   ):
    """Function to run simulation.
    Inputs:
    sim_dir:  Initial condition file directory path
    tend:  Simulation end time
    eta:  Parent system simulation step-time
    code_dt:  Child integrator internal timestep
    par_n_worker:  Number of workers for parent code
    dE_track:  Flag turning on energy error tracker
    gal_field:  Flag turning on galactic field or not
    star_evol:  Flag turning on stellar evolution
    """
    
    # Creating output directories
    run_idx = len(glob.glob(sim_dir+"/*")) - 1
    config_name = "Nrun"+str(run_idx)
    dir_path = os.path.join(sim_dir, config_name)
    if os.path.exists(os.path.join(dir_path, "*")):
        pass
    else:
        os.mkdir(dir_path+"/")
        subdir = ["event_data", "collision_snapshot", 
                  "data_process", "simulation_stats", 
                  "simulation_snapshot", "ejected_particles"
                  ]
        for path in subdir:
            os.makedirs(os.path.join(dir_path, path))
    coll_path = os.path.join(dir_path, "collision_snapshot")
    ejected_dir = os.path.join(dir_path, "ejected_particles")
    snapdir_path = os.path.join(dir_path, "simulation_snapshot")

    # Organise particle set
    particle_set_dir = os.path.join(sim_dir, "initial_particles",
                                    "run_{:}".format(run_idx)
                                    )
    particle_set = read_set_from_file(particle_set_dir)
    particle_set -= particle_set[particle_set.syst_id > 8]
    particle_set.coll_events = 0
    
    isolated_particles = particle_set[particle_set.syst_id < 0]
    major_bodies = isolated_particles[isolated_particles.mass != (0 | units.kg)]
    test_particles = isolated_particles - major_bodies
    for system_id in np.unique(particle_set.syst_id):
        if system_id > -1:
            system = particle_set[particle_set.syst_id == system_id]
            hosting_body = system[system.mass == max(system.mass)]
            major_bodies += hosting_body
        
    if (gal_field): # Sun's orbit
        particle_set = galactic_frame(particle_set, 
                                      dx=-8.4 | units.kpc, 
                                      dy=0.0 | units.kpc, 
                                      dz=17  | units.pc,
                                      dvx=11.352 | units.kms,
                                      dvy=12.25 | units.kms,
                                      dvz=7.41 | units.kms
                                      )
    conv_par = nbody_system.nbody_to_si(np.sum(major_bodies.mass), 
                                        major_bodies.virial_radius()
                                        )
    dt = eta*tend
    

    # Setting up parents + children
    nmajor = len(major_bodies)
    par_n_worker = nmajor // 1000 + 1

    parents = Particles(nmajor)
    parents = major_bodies.copy()
    parents.sub_worker_radius = parents.radius
    parents = HierarchicalParticles(parents)
    
    initial_systems = parents[parents.syst_id > 0]
    for id_ in np.unique(initial_systems.syst_id):
        children = particle_set[particle_set.syst_id == id_]
        host = parents[parents.syst_id == id_][0]
        parents.assign_subsystem(children, host, 
                                 relative=True, 
                                 recenter=False
                                 )

    # Setting up system
    conv_child = nbody_system.nbody_to_si(np.mean(parents.mass), 
                                          np.mean(parents.radius)
                                          )
    nemesis = Nemesis(conv_par, conv_child, dt, code_dt, 
                      par_n_worker, dE_track, star_evol, 
                      gal_field
                      )
    nemesis.min_mass_evol = MIN_EVOL_MASS
    nemesis.particles.add_particles(parents)
    nemesis.commit_particles(conv_child)
    nemesis.coll_dir = coll_path
    nemesis.ejected_dir = ejected_dir
    nemesis.test_particles = test_particles
    
    allparts = nemesis.particles.all()
    if (nemesis.dE_track):
        energy_arr = [ ]
        E0 = allparts.potential_energy() + allparts.kinetic_energy()
        nemesis.E0 = E0
    write_set_to_file(allparts.savepoint(0|units.Myr), 
                      os.path.join(snapdir_path, "snap_0"), 
                      'amuse', close_file=True, overwrite_file=True
                      )
    
    t = 0 | units.yr
    snapshot_no = 0
    SNAP_PER_ITER = 100
    dt_snapshot = dt * SNAP_PER_ITER
    prev_step = nemesis.dt_step
    while t < tend:
        t += dt
        
        while nemesis.parent_code.model_time < t:
            nemesis.dt_step += 1
            nemesis.evolve_model(t)
            
        if dt_snapshot <= nemesis.parent_code.model_time:
            print("Saving snap @ time: ", t.in_(units.yr))
            dt_snapshot += SNAP_PER_ITER * dt
            snapshot_no += 1
            allparts = nemesis.particles.all()
            
            fname = os.path.join(snapdir_path, "snap_"+str(snapshot_no))
            write_set_to_file(
                allparts.savepoint(0|units.Myr),
                fname, 'amuse', close_file=True, 
                overwrite_file=True
            )
          
        if (nemesis.dE_track) and (prev_step != nemesis.dt_step):
            print("...Check {:}...".format(t.in_(units.Myr)))
            allparts = nemesis.particles.all()
            E1 = allparts.potential_energy() + allparts.kinetic_energy() 
            E1 += nemesis.corr_energy
            energy_arr.append(abs((E1-E0)/E0))
            
            prev_step = nemesis.dt_step
      
    print("...Simulation Ended...")
    nemesis.stellar_code.stop()  
    nemesis.parent_code.stop()
    for code in nemesis.subcodes.values():
        code.stop()

    # Store simulation statistics
    sim_time = (cpu_time.time() - START_TIME)/60
    fname = os.path.join(dir_path, 'simulation_stats', 'sim_stats_'+str(run_idx)+'.txt')
    with open(fname, 'w') as f:
        f.write("Total CPU Time: {} minutes".format(sim_time))
        f.write("\nEnd Time: {}".format(t.in_(units.Myr)))
        f.write("\nTime step: {}".format(dt.in_(units.Myr)))
    f.close()
        
if __name__ == "__main__":
    START_TIME = cpu_time.time()
    MIN_EVOL_MASS = 0.08 | units.MSun
    
    # data_dir = "examples/S-Stars"
    data_dir = "examples/ejecting_suns"
    if data_dir == "examples/ejecting_suns":
        config_idx = 1
        configurations = glob.glob(os.path.join(data_dir, "sim_data", "*"))
        config_choice = natsorted(configurations)[config_idx]
    
    run_simulation(
        sim_dir=config_choice, 
        par_n_worker=1, 
        tend=10 | units.Myr, 
        eta=1e-5, 
        code_dt=5e-2, 
        gal_field=True, 
        dE_track=True, 
        star_evol=True, 
    )