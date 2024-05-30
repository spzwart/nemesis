import fnmatch
import glob
import numpy as np
import os

from amuse.community.seba.interface import SeBa
from amuse.ext.orbital_elements import orbital_elements_from_binary
from amuse.lab import Particles, units, write_set_to_file, constants
from parti_initialiser import ClusterInitialise

def run_code(rvir, vejec, target_nimbh):
    """Run script to get initial conditions. \n
    Inputs:
    rvir:  Initial virial radius
    vejec:  Rough velocity of ejected SMBH
    target_nimbh:  Target # IMBH
    """
    TARGET_MASS = 5600 | units.MSun #1700 | units.MSun
    TARGET_NSIMS = 10
    SMBH_MASS = 4e5 | units.MSun
    
    print("Configuration: "+str(nIMBH))
    data_direc = "examples/runaway_bh/data/"
    config_name = "Nimbh"+str(nIMBH)+"_RA_BH_Run"
    dir_path = data_direc+config_name
    nsims = len(glob.glob(dir_path+"/init_conds/*"))
    if dir_path in glob.glob("data/*"):
        None
    else:
        os.mkdir(dir_path+"/")
        os.mkdir(dir_path+"/coll_orbital/")
        os.mkdir(dir_path+"/data_process/")
        os.mkdir(dir_path+"/data_process/gw_calcs_all/")
        os.mkdir(dir_path+"/data_process/gw_calcs_evolved_10Myr/")
        os.mkdir(dir_path+"/data_process/ejec_calcs/")
        os.mkdir(dir_path+"/init_conds/")
        os.mkdir(dir_path+"/init_snapshot/")
        os.mkdir(dir_path+"/merge_snapshots/")
        os.mkdir(dir_path+"/simulation_resume/")
        os.mkdir(dir_path+"/simulation_snapshot/")
        os.mkdir(dir_path+"/simulation_stats/")
    
    while nsims < (TARGET_NSIMS):
        print("...Running simulation...")

        sbh_code = ClusterInitialise()
        pset = sbh_code.init_cluster(SMBH_MASS, rvir)
        minor = pset[pset.type != "smbh"]
        SMBH = pset - minor
        SMBH.velocity += [1,1,1] * (vejec/np.sqrt(3))
        
        pre_evol_mass = minor.mass.sum()
        print("Mass pre-evolution:", pre_evol_mass.in_(units.MSun))
        stellar_code = SeBa()
        stellar_code.particles.add_particle(minor)
        stellar_code.evolve_model(10 | units.Myr)
        channel = stellar_code.particles.new_channel_to(minor)
        channel.copy_attributes(["mass", "radius"])
        stellar_code.stop()
        post_evol_mass = minor.mass.sum()
        print("Mass post-evolution: ", (post_evol_mass.in_(units.MSun)))

        final_bound = pset.copy_to_memory()
        particle = 0
        for parti_ in minor:
            particle += 1
            if particle%1000 == 0:
                print("Particle #: ", particle)
            bin_sys = Particles()
            bin_sys.add_particle(parti_)
            bin_sys.add_particle(SMBH)

            kepler_elements = orbital_elements_from_binary(bin_sys, 
                                                           G=constants.G
                                                           )
            ecc = abs(kepler_elements[3])
            if ecc >= 1:
                final_bound -= parti_

        bound_stars = final_bound[final_bound.type == "star"]
        print("Mean stellar masses: ", np.mean(bound_stars.mass.in_(units.MSun)))
        print("Total mass: ", np.sum(bound_stars.mass.in_(units.MSun)))
        print("#Stars", len(final_bound))
        if abs(TARGET_MASS - np.sum(bound_stars.mass)) <= (150 | units.MSun):
            print("...successful initial conditions...")
            print(len(bound_stars))
            
            imbh = bound_stars.random_sample(target_nimbh)
            imbh.mass = (4000 | units.MSun)/target_nimbh
            imbh.type = "IMBH"
            imbh.radius = sbh_code.isco_radius(imbh.mass)
            imbh.collision_radius = sbh_code.coll_radius(imbh.radius)
            
            type, counts = (np.unique(final_bound.type, return_counts=True))

            nsims = len(fnmatch.filter(os.listdir(dir_path+"/init_conds/"), "*"))
            fname = "config_"+str(nsims)
            os.mkdir(dir_path+"/coll_orbital/"+fname)
            os.mkdir(dir_path+"/merge_snapshots/"+fname)
            os.mkdir(dir_path+"/simulation_resume/"+fname)
            os.mkdir(dir_path+"/simulation_snapshot/"+fname)

            write_set_to_file(final_bound, dir_path+"/init_snapshot/"+fname, 
                              "hdf5", close_file=True, overwrite_file=False
                              )

            lines = ['Initial Virial Radius: '+str(rvir.in_(units.pc)),
                     'Total Number of Particles: '+str(len(pset)),
                     'Pops & Counts: '+str(type)+" "+str(counts),
                     #'IMBH Mass: '+str(np.mean(imbh.mass.in_(units.MSun))),
                     'Ejection Velocity: '+str(vejec),
                     'SMBH Mass: '+str(SMBH_MASS.in_(units.MSun)),
                     'Stellar Mass Pre-Evolution: '+str(pre_evol_mass),
                     'Stellar Mass Post-Evolution: '+str(post_evol_mass)
                     ]
            with open(os.path.join(dir_path+str("/init_conds/"), 
                'simulation_stats_'+str(nsims)+'.txt'), 'w') as f:
                for line in lines:
                    f.write(line)
                    f.write('\n')

nIMBH = 0
run_code(rvir=1 | units.pc,
         vejec=185 | units.kms,
         target_nimbh=nIMBH
         )
