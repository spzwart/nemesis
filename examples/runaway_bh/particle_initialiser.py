import fnmatch
import glob
import numpy as np
import os

from amuse.ext.orbital_elements import orbital_elements_from_binary
from amuse.lab import Particles, units, write_set_to_file, constants
from examples.runaway_bh.parti_setup import IMBH_init

def run_code(pops, fstar, imbh_mass, stellar_mass, 
            bin_star, rvir, vejec, nsims):
    """Function to run the code

        Inputs:
        pops:         Cluster population (BH + Stars)
        fstar:        Fraction of cluster to be stars
        imbh_mass:    Mass distribution for IMBH
        stellar_mass: Mass distribution for stars
        bin_star:     Initialise with IMBH-Star systems
        rvir:         Initial virial radius
        vejec:        Rough velocity of ejected SMBH
        nsims:        Number of simulations
    """

    masses = [4e5 | units.MSun, 
              4e6 | units.MSun, 
              4e7 | units.MSun, 
              4e8 | units.MSun]
    for SMBH_mass in masses:
        print("Configuration: 4e"+str(int(np.log10(SMBH_mass.value_in(units.MSun)/4))))
        data_direc = "examples/runaway_bh/sim_data"
        config_name = "Ncluster"+str(pops)+"_fStar"+str(fstar) \
                    +"_msmbh4e"+str(np.log10(SMBH_mass.value_in(units.MSun)/4)) \
                        +"_rvir"+str(rvir.value_in(units.pc))+"pc_vejec1000"
        dir_path = data_direc+config_name
        if dir_path in glob.glob("examples/runaway_bh/sim_data*"):
            None
        else:
            os.mkdir(dir_path+"/")
            os.mkdir(dir_path+"/child_merge/")
            os.mkdir(dir_path+"/coll_orbital/")
            os.mkdir(dir_path+"/data_process/")
            os.mkdir(dir_path+"/data_process/gw_calcs/")
            os.mkdir(dir_path+"/ejec_snapshots/")
            os.mkdir(dir_path+"/energy_data/")
            os.mkdir(dir_path+"/event_data/")
            os.mkdir(dir_path+"/init_conds/")
            os.mkdir(dir_path+"/init_snapshot/")
            os.mkdir(dir_path+"/lagrangian_data/")
            os.mkdir(dir_path+"/merge_snapshots/")
            os.mkdir(dir_path+"/parent_change/")
            os.mkdir(dir_path+"/simulation_resume/")
            os.mkdir(dir_path+"/simulation_snapshot/")
            os.mkdir(dir_path+"/simulation_stats/")
        
        for run in range(nsims):
            print("...Running simulation ", run, "...")
            dir_count = len(fnmatch.filter(os.listdir(dir_path+"/init_conds/"), "*"))
            fname = "config_"+str(dir_count)
            os.mkdir(dir_path+"/child_merge/"+fname)
            os.mkdir(dir_path+"/ejec_snapshots/"+fname)
            os.mkdir(dir_path+"/parent_change/"+fname)
            os.mkdir(dir_path+"/simulation_resume/"+fname)
            os.mkdir(dir_path+"/simulation_snapshot/"+fname)

            IMBH_code = IMBH_init()
            pset = IMBH_code.IMBH_first(pops, fstar, SMBH_mass, 
                                        imbh_mass, stellar_mass, 
                                        bin_star, rvir)
            minor = pset[pset.type!="smbh"]
            imbh = minor[minor.type!="star"]
            star = minor[minor.type=="star"]
            SMBH = pset-minor
            SMBH.velocity += [1,1,1]*(vejec/np.sqrt(3))

            Qimbh = abs(imbh.kinetic_energy()/imbh.potential_energy())
            Qstar = abs(star.kinetic_energy()/star.potential_energy())
            Rvimbh = imbh.virial_radius()
            Rvstar = star.virial_radius()

            final_bound = pset.copy_to_memory()
            particle = 0
            for parti_ in minor:
                particle+=1
                if particle%1000==0:
                    print("Particle #: ", particle)
                bin_sys = Particles()
                bin_sys.add_particle(parti_)
                bin_sys.add_particle(SMBH)

                kepler_elements = orbital_elements_from_binary(
                                         bin_sys, G=constants.G)
                ecc = abs(kepler_elements[3])
                if ecc >= 1:
                    final_bound -= parti_
            no_imbh = len(final_bound[final_bound.type=="imbh"])
            no_star = len(final_bound[final_bound.type=="star"])
            print("Total: ", len(final_bound), "IMBH: ", no_imbh)
            write_set_to_file(final_bound, dir_path+"/init_snapshot/"+fname, "hdf5",
                              close_file=True, overwrite_file=False)

            lines = ['Initial Virial Radius: '+str(rvir.in_(units.pc)),
                     'Total Number of Particles: '+str(len(pset)),
                     'Total Number of IMBH: '+str(pops),
                     'IMBH Initial Q:'+str(Qimbh)+" Initial Rvir "+str(Rvimbh.in_(units.pc)),
                     'Total Number of Stars: '+str(pops*fstar),
                     'Star Initial Q:'+str(Qstar)+" Initial Rvir "+str(Rvstar.in_(units.pc)),
                     'Total Number of Final Bound IMBH: '+str(no_imbh),
                     'Total Number of Final Bound Star: '+str(no_star),
                     'Ejection Velocity: '+str(vejec),
                     'SMBH Mass: '+str(SMBH_mass.in_(units.MSun))]

            with open(os.path.join(dir_path+str("/init_conds/"), 
                                'simulation_stats_'+str(run)+'.txt'), 'w') as f:
                for line in lines:
                    f.write(line)
                    f.write('\n')

imbh_masses = ["Equal", "Power"]
stellar_masses = ["Equal", "Kroupa"]
run_code(pops=50, 
         fstar=1000, 
         imbh_mass=imbh_masses[0],
         stellar_mass=stellar_masses[0],
         bin_star=False, 
         rvir=1 | units.pc,
         vejec=1000 | units.kms,
         nsims=10)
