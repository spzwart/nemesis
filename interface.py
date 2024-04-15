import glob
from natsort import natsorted
import numpy as np
import os
import pandas as pd
import time as cpu_time

from amuse.lab import Particles, read_set_from_file, write_set_to_file
from amuse.units import units, nbody_system
from src.environment_functions import galactic_frame, set_parent_radius
from src.hierarchical_particles import HierarchicalParticles
from src.nemesis import Nemesis

import matplotlib.pyplot as plt

def run_simulation(sim_dir, tend, eta, code_dt, 
                   par_n_worker, dE_track, gal_field, 
                   star_evol):
    """Function to run simulation.
    Inputs:
    sim_dir:  Initial condition file directory path
    tend:  Simulation end time
    eta:  Parent system simulation step-time
    code_dt:  Internal timestep parameter
    par_n_worker:  Number of workers for parent code
    dE_track:  Flag turning on energy error tracker
    gal_field:  Flag turning on galactic field or not
    star_evol:  Flag turning on stellar evolution
    """
    def smaller_nbody_power_of_two(dt, conv):
        """Function scaling dt relative to N-body units"""
        nbdt = conv.to_nbody(dt).value_in(nbody_system.time)
        idt = np.floor(np.log2(nbdt))
        return conv.to_si(2**idt | nbody_system.time)

    START_TIME = cpu_time.time()
    MIN_EVOL_MASS = 0.01 | units.MSun
    SNAP_PER_ITER = 10
    
    # Creating output directories
    RUN_CHOICE = max(0, len(glob.glob(sim_dir+"/*")) - 1)
    config_name = "Nrun"+str(RUN_CHOICE)
    dir_path = os.path.join(sim_dir, config_name)
    if os.path.exists(os.path.join(dir_path, "*")):
        pass
    else:
        os.mkdir(dir_path+"/")
        subdir = ["event_data", "collision_snapshot", 
                  "data_process", "simulation_stats", 
                  "simulation_snapshot", "system_changes"
                 ]
        for path in subdir:
            os.makedirs(os.path.join(dir_path, path))
    coll_path = os.path.join(dir_path, "collision_snapshot")
    syst_change_path = os.path.join(dir_path, "system_changes")
    snapdir_path = os.path.join(dir_path, "simulation_snapshot")

    # Organise particle set
    # Change line to directory hosting particle set
    particle_set_dir = os.path.join(sim_dir, "initial_particles",
                                    "run_"+str(RUN_CHOICE)
                                    )
    particle_set = read_set_from_file(particle_set_dir)
    particle_set -= particle_set[particle_set.name=="EARTHMOO"]
    
    particle_set.coll_events = 0
    major_bodies = particle_set[particle_set.syst_id < 0]
    for system_id in np.unique(particle_set.syst_id):
        if system_id > -1:
            system = particle_set[particle_set.syst_id == system_id]
            hosting_body = system[system.mass == max(system.mass)]
            major_bodies += hosting_body
        
    if (gal_field):
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
    dt = smaller_nbody_power_of_two(dt, conv_par)
    
    # Setting up parents
    NSYSTS = len(major_bodies)
    parents = Particles(NSYSTS)
    parents = major_bodies.copy()
    parents.sub_worker_radius = parents.radius
    parents.radius = set_parent_radius(parents.mass, dt)
    RVIR_INIT = parents.virial_radius().in_(units.pc)
    Q_INIT = abs(parents.kinetic_energy()/parents.potential_energy())

    # Setting up children
    parents = HierarchicalParticles(parents)
    initial_systems = parents[parents.syst_id > 0]
    for id_ in np.unique(initial_systems.syst_id):
        children = particle_set[particle_set.syst_id == id_]
        host = parents[parents.syst_id == id_][0]
        parents.assign_subsystem(children, host)

    # Setting up system
    conv_child = nbody_system.nbody_to_si(np.mean(parents.mass), 
                                          np.mean(parents.radius)
                                          )
    nemesis = Nemesis(conv_par, conv_child, 
                      dt, code_dt, par_n_worker, 
                      dE_track, star_evol, 
                      gal_field
                      )
    nemesis.min_mass_evol = MIN_EVOL_MASS
    nemesis.timestep = dt
    nemesis.particles.add_particles(parents)
    nemesis.parents = parents.copy_to_memory()
    nemesis.radius = set_parent_radius(parents.mass, code_dt)
    nemesis.commit_particles(conv_child)
    nemesis.channel_makers()
    nemesis.coll_dir = coll_path
    
    allparts = nemesis.particles.all()
    if (dE_track):
        E0_all = nemesis.energy_track()
        if (gal_field):
            PE = nemesis.grav_bridge.potential_energy
            KE = nemesis.grav_bridge.kinetic_energy
            E0_all += (PE+KE)
        
    t = 0 | units.yr
    dt_iter = 0
    write_set_to_file(allparts.savepoint(0|units.Myr), 
                      os.path.join(snapdir_path, "snap_"+str(dt_iter)), 
                      'amuse', close_file=True, overwrite_file=True
                      )
    
    # Run code
    dt_iter = 0
    dt_snapshot = SNAP_PER_ITER * dt
    while t < tend*(1-1e-12):
        t += dt
        while nemesis.parent_code.model_time < t*(1 - 1e-12):
            nemesis.evolve_model(t)

            if (dE_track):
                E1_all = nemesis.energy_track()
                if (gal_field):
                    PE = nemesis.grav_bridge.potential_energy
                    KE = nemesis.grav_bridge.kinetic_energy
                    E1_all += (PE+KE)
                    E0_all += nemesis.dEa
                print("Energy error: ", abs(E0_all-E1_all)/E0_all)

        if (nemesis.save_snap):
            path = os.path.join(syst_change_path,
                                "par_chng_"+str(len(nemesis.event_time))
                                )
            write_set_to_file(allparts.savepoint(0|units.Myr), path,
                              'amuse', close_file=True, overwrite_file=True
                              )
        if dt_snapshot <= t:
          dt_iter += 50
          dt_snapshot += SNAP_PER_ITER * dt
          print("Saving snap @ time: ", t.in_(units.yr))
          allparts = nemesis.particles.all()
          write_set_to_file(allparts.savepoint(0|units.Myr),
                            os.path.join(snapdir_path, "snap_"+str(dt_iter)),
                            'amuse', close_file=True, overwrite_file=True
                            )
      
    print("...Simulation Ended...")
    RVIR_FIN = nemesis.particles.virial_radius()
    Q_FIN = -(nemesis.particles.kinetic_energy()) \
            /(nemesis.particles.potential_energy())
    nemesis.stellar_code.stop()  
    nemesis.parent_code.stop()
    for code in nemesis.subcodes.values():
        code.stop()


    # Store data files
    path = os.path.join(dir_path, "event_data", "event_"+str(RUN_CHOICE)+".h5")
    data_arr = pd.DataFrame([nemesis.event_time, 
                            nemesis.event_key, 
                            nemesis.event_type]
                            )
    data_arr.to_hdf(path, key="df", mode="w")
    SIM_TIME = cpu_time.time()-START_TIME

    # Store simulation statistics
    lines = ["Total CPU Time: {} minutes".format(SIM_TIME),
            "End Time: {}".format(t.in_(units.Myr)), 
            "Time step: {}".format(dt.in_(units.Myr)),
            "Initial Rvirial: {}".format(RVIR_INIT.in_(units.pc)),
            "Final Rvirial: {}".format(RVIR_FIN.in_(units.pc)),
            "Initial Q: {}".format(Q_INIT), 
            "Final Q: {}".format(Q_FIN),
            "Init No. major_bodies: {}".format(NSYSTS)
            ]
    with open(os.path.join(dir_path, 'simulation_stats', 
              'simulation_stats_'+str(RUN_CHOICE)+'.txt'), 'w') as f:
        for line_ in lines:
            f.write(line_)
            f.write('\n')

if __name__=="__main__":
    # sim_dir = "examples/realistic_cluster/"
    # sim_dir = "examples/S-Stars"
    sim_dir = "examples/ejecting_suns"
    if sim_dir == "examples/ejecting_suns":
        config_idx = 0
        run_idx = 0
        
        configurations = glob.glob(os.path.join(sim_dir, "sim_data", "*"))
        config_choice = natsorted(configurations)[config_idx]
        sim_dir = natsorted(glob.glob(config_choice))[run_idx]
        
    run_simulation(sim_dir=sim_dir, tend=30 | units.Myr, 
                   eta=1e-4, code_dt=1e-2, par_n_worker=1, 
                   gal_field=False, dE_track=False, 
                   star_evol=True,
                   )