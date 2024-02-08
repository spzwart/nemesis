import glob
import natsort
import numpy as np
import os
import pandas as pd
import time as cpu_time

from amuse.lab import Particles, read_set_from_file, write_set_to_file
from amuse.units import units, nbody_system
from src.environment_functions import EnvironmentFunctions
from src.hierarchical_particles import HierarchicalParticles
from src.nemesis import Nemesis

def snap_name(dpath, dt_iter):
  return dpath+"/simulation_snapshot/snap_dt"+str(dt_iter)

def run_code(sim_dir, tend, eta, code_dt, 
             par_nworker, dE_track, gal_field):
  """Function to run simulation.

     Inputs:
     sim_dir:      Directory of particle set simulated
     tend:         Simulation end time
     eta:          Parent system simulation step-time
     par_nworker:  Number of workers for parent code
     dE_track:     Flag turning on energy error tracker
     gal_field:    Flag turning on galactic field or not
  """

  def smaller_nbody_power_of_two(dt, conv):
    """Function scaling dt relative to N-body units"""
    nbdt = conv.to_nbody(dt).value_in(nbody_system.time)
    idt = np.floor(np.log2(nbdt))
    return conv.to_si(2**idt | nbody_system.time)

  start_time = cpu_time.time()
  
  #Making data files
  data_direc = os.path.join(sim_dir, "sim_data")
  if sim_dir.lower()=="examples/realistic_cluster/":
    star_evol = True

    Nconfig = len(glob.glob(data_direc+"/*"))
    config_name = "Nrun"+str(Nconfig)
    dir_path = os.path.join(data_direc, config_name)
    if os.path.exists(os.path.join(dir_path, "*")):
      pass
    else:
      os.mkdir(dir_path+"/")
      subdir = ["coll_orbital", "ejec_snapshots", "energy_data", "event_data",
                "data_process", "initial_binaries", "lagrangian_data",
                "simulation_stats", "simulation_snapshot", "system_changes"]
      for path in subdir:
          os.makedirs(os.path.join(dir_path, path))
      dir_changes = os.path.join(dir_path, "system_changes")
    particle_set = read_set_from_file(os.path.join(sim_dir, "initial_particles/init_particle_set"))
    major_bodies = particle_set[particle_set.type=="STAR"]

  elif sim_dir.lower()=="examples/runaway_bh/":
    star_evol = False
    resume=False

    configuration = glob.glob(data_direc+"/*")
    config = configuration[0]
    if not (resume):
        init_psets = glob.glob(config+"/init_snapshot/*")
        sorted_psets = natsort.natsorted((init_psets))
        init_pset = sorted_psets[0]
        resume_iter = 0 
    else:
        run_idx = 0
        config_dir = (glob.glob(config+"/simulation_resume/*"))
        sorted_config = natsort.natsorted(config_dir)
        psets = sorted_config[run_idx]
        init_psets = (glob.glob(psets+"/*"))
        sorted_psets = natsort.natsorted(init_psets)
        init_pset = sorted_psets[-1]
    dir_count = len(glob.glob(config+"/simulation_stats/*"))
    fname = "config_"+str(dir_count)
    ejec_dir = os.path.join(config, "ejec_snapshots", fname)
    ep_dir = os.path.join(config, "ejec_par_change", fname)
    merge_dir = os.path.join(config, "child_merge", fname)
    pchange_dir = os.path.join(config, "parent_change", fname)
    simsnap_dir = os.path.join(config, "simulation_snapshot", fname)

    print("Simulating ", init_pset)
    particle_set = read_set_from_file(init_pset)
    major_bodies = particle_set

  env_func = EnvironmentFunctions()
  if (gal_field):
    particle_set = env_func.galactic_frame(particle_set, 
                                     dx=8.3 | units.kpc, 
                                     dy=0.0 | units.kpc, 
                                     dz=17  | units.pc
                                     )
  conv_par = nbody_system.nbody_to_si(
                      np.sum(major_bodies.mass), 
                      major_bodies.virial_radius()
                      )
  dt = eta*tend
  dt = smaller_nbody_power_of_two(dt, conv_par)
  
  #Setting up parents
  Nsysts = len(major_bodies)
  hosts = Particles(len(major_bodies))
  hosts.mass = major_bodies.mass
  hosts.velocity = major_bodies.velocity
  hosts.position = major_bodies.position
  hosts.syst_id = major_bodies.syst_id
  hosts.type = major_bodies.type
  hosts.sub_worker_radius = major_bodies.radius
  hosts.radius = env_func.parent_radius(hosts.mass, dt)

  #Setting up children
  parents = HierarchicalParticles(hosts)
  Rvir_init = parents.virial_radius().in_(units.pc)
  Q_init = abs(parents.kinetic_energy()/parents.potential_energy())
  initial_systems = hosts[hosts.syst_id>0]
  for id_ in np.unique(initial_systems.syst_id):
    children = particle_set[particle_set.syst_id==id_]
    host = hosts[hosts.syst_id==id_][0]
    parents.assign_subsystem(children, host)
  print("Number of children: ", max(particle_set.syst_id))
  print("Rvir: ", Rvir_init)
  print("Qvir: ", Q_init)

  conv_child = nbody_system.nbody_to_si(np.mean(parents.mass), 
                                        np.mean(parents.radius))
  nemesis = Nemesis(conv_par, conv_child, dt, code_dt, 
                    par_nworker, dE_track, star_evol, 
                    gal_field)
  nemesis.timestep = dt
  nemesis.particles.add_particles(parents)
  nemesis.parents = parents.copy_to_memory()
  nemesis.radius = env_func.parent_radius(parents.mass, code_dt)
  nemesis.commit_particles(conv_child)
  nemesis.channel_makers()
  allparts = nemesis.particles.all()
  if (dE_track):
    E0a = allparts.kinetic_energy()+allparts.potential_energy()
    if (gal_field):
      E0a+=nemesis.grav_bridge.potential_energy+nemesis.grav_bridge.kinetic_energy

  t = 0 | units.yr
  time = [0.]
  totalE = [0.]
  event_iter = [ ]
  dt_iter = 0
  write_set_to_file(allparts.savepoint(0|units.Myr), 
                    snap_name(dir_path, str(dt_iter)), 
                    'amuse', close_file=True, 
                    overwrite_file=True)
  while t < tend:
    print("T=", t.in_(units.Myr))
    t+=dt
    time.append(t.value_in(units.yr))
    nemesis.evolve_model(t)  
    if (dE_track):
      E1a = nemesis.energy_track()
      if (gal_field):
        E1a+=nemesis.grav_bridge.potential_energy+nemesis.grav_bridge.kinetic_energy
      E0a+=nemesis.dEa
      totalE.append(abs((E0a-E1a)/E0a))
      print("Energy error:     ", abs(E0a-E1a)/E0a)

    if dt_iter%10==1:
      print("Time: ", t.in_(units.yr), end=' ')
      write_set_to_file(allparts.savepoint(0|units.Myr), 
                        snap_name(dir_path, str(dt_iter)), 
                        'amuse', close_file=True, 
                        overwrite_file=True)

    if (nemesis.save_snap):
      event_iter = np.concatenate((event_iter, dt_iter), axis=None)
      path = os.path.join(dir_changes, "parent_change_"+str(nemesis.event_time[-1]))
      write_set_to_file(allparts.savepoint(0|units.Myr),
                        path, 'amuse', close_file=True, 
                        overwrite_file=True)
    dt_iter+=1
  
  Rvir_fin = nemesis.particles.virial_radius()
  Qfinal = -(nemesis.particles.kinetic_energy())/(nemesis.particles.potential_energy())

  nemesis.parent_code.stop()
  nemesis.stellar_code.stop()  
  for parent, code in list(nemesis.subcodes.items()):
    code.stop()

  path = os.path.join(dir_path, "event_data", "event_data_"+str(Nconfig)+".h5")
  data_arr = pd.DataFrame([nemesis.event_time, nemesis.event_key, 
                           nemesis.event_type, nemesis.event_mass])
  data_arr.to_hdf(path, key="df", mode="w")
  end_time = cpu_time.time()
  tot_time = end_time-start_time

  lines = ["Total CPU Time: {} minutes".format(tot_time),
           "End Time: {}".format(t.in_(units.Myr)), 
           "Time step: {}".format(dt.in_(units.Myr)),
           "Initial Rvirial: {}".format(Rvir_init.in_(units.pc)),
           "Final Rvirial: {}".format(Rvir_fin.in_(units.pc)),
           "Initial Q: {}".format(Q_init), 
           "Final Q: {}".format(Qfinal),
           "Init No. major_bodies: {}".format(Nsysts)]
  with open(os.path.join(dir_path, 'simulation_stats', 
            'simulation_stats_'+str(Nconfig)+'.txt'), 'w') as f:
      for line_ in lines:
          f.write(line_)
          f.write('\n')

if __name__=="__main__":
  sim_dir="examples/realistic_cluster/"
  #sim_dir ="examples/runaway_bh/"
  run_code(sim_dir=sim_dir, tend=50 | units.Myr, 
           eta=1e-4, code_dt=5e-2, par_nworker=1,
           gal_field=False, dE_track=False,)
