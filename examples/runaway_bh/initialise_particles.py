import fnmatch
import glob
import numpy as np
import os

from amuse.ext.orbital_elements import orbital_elements_from_binary
from amuse.lab import Particles, units, write_set_to_file, constants
from parti_initialiser import sbh_init

def run_code(pops, fstar, sbh_mass, star_mass, 
            bin_star, rvir, vejec, target_nsbh):
    """Function to run the code

        Inputs:
        pops:         Cluster population (BH + Stars)
        fstar:        Fraction of cluster to be stars
        sbh_mass:     Mass distribution for sbh
        star_mass:    Mass distribution for stars
        bin_star:     Initialise with sbh-Star systems
        rvir:         Initial virial radius
        vejec:        Rough velocity of ejected SMBH
        target_nsbh:  Target # stellar-mass black holes
    """
    target_nsims = 16/max(1,target_nsbh)

    # 
    masses = [4e5 | units.MSun]#, 4e6 | units.MSun, 4e7 | units.MSun, 4e8 | units.MSun]
    nsims = 0 
    for SMBH_mass in masses:
        print("Configuration: 4e"+str(int(np.log10(SMBH_mass.value_in(units.MSun)/4))))
        data_direc = "examples/runaway_bh/data/"
        config_name = "Nsbh"+str(target_nsbh)+"_msmbh4e"+str(np.log10(SMBH_mass.value_in(units.MSun)/4)) \
                        +"_vejec"+str(vejec.value_in(units.kms))+"kms"
        dir_path = data_direc+config_name
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
        
        while nsims<target_nsims:
            print("...Running simulation...")

            sbh_code = sbh_init()
            pset = sbh_code.sbh_first(pops, fstar, SMBH_mass, 
                                      sbh_mass, star_mass, 
                                      bin_star, rvir)
            minor = pset[pset.type!="smbh"]
            sbh = minor[minor.type!="star"]
            star = minor[minor.type=="star"]
            SMBH = pset-minor
            SMBH.velocity += [1,1,1]*(vejec/np.sqrt(3))

            Qsbh = abs(sbh.kinetic_energy()/sbh.potential_energy())
            Qstar = abs(star.kinetic_energy()/star.potential_energy())
            Rvsbh = sbh.virial_radius()
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

            no_sbh = len(final_bound[final_bound.type=="SBH"])
            no_star = len(final_bound[final_bound.type=="star"])
            print("#Stars: ", no_star, "#SBH: ", no_sbh)
            if no_sbh==target_nsbh and abs(no_star-400)<=20:
                print("...successful initial conditions...")
                nsims = len(fnmatch.filter(os.listdir(dir_path+"/init_conds/"), "*"))
                fname = "config_"+str(nsims)
                os.mkdir(dir_path+"/coll_orbital/"+fname)
                os.mkdir(dir_path+"/merge_snapshots/"+fname)
                os.mkdir(dir_path+"/simulation_resume/"+fname)
                os.mkdir(dir_path+"/simulation_snapshot/"+fname)

                write_set_to_file(final_bound, dir_path+"/init_snapshot/"+fname, "hdf5",
                                close_file=True, overwrite_file=False)

                lines = ['Initial Virial Radius: '+str(rvir.in_(units.pc)),
                        'Total Number of Particles: '+str(len(pset)),
                        'Total Number of SBH: '+str(pops),
                        'SBH Initial Q:'+str(Qsbh)+" Initial Rvir "+str(Rvsbh.in_(units.pc)),
                        'Total Number of Stars: '+str(pops*fstar),
                        'Star Initial Q:'+str(Qstar)+" Initial Rvir "+str(Rvstar.in_(units.pc)),
                        'Total Number of Final Bound sbh: '+str(no_sbh),
                        'Total Number of Final Bound Star: '+str(no_star),
                        'Ejection Velocity: '+str(vejec),
                        'SMBH Mass: '+str(SMBH_mass.in_(units.MSun))]
                with open(os.path.join(dir_path+str("/init_conds/"), 
                    'simulation_stats_'+str(nsims)+'.txt'), 'w') as f:
                    for line in lines:
                        f.write(line)
                        f.write('\n')
        
sbh_masses = ["Equal", "Power"]
star_masses = ["Equal", "Kroupa"]
run_code(pops=40, 
         fstar=150, 
         sbh_mass=sbh_masses[0],
         star_mass=star_masses[0],
         bin_star=False, 
         rvir=1 | units.pc,
         vejec=200 | units.kms,
         target_nsbh=8)
