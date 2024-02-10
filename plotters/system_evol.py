import glob
import matplotlib.pyplot as plt
from natsort import natsorted
import numpy as np
import os
import pandas as pd

from amuse.ext.LagrangianRadii import LagrangianRadii as LagRad
from amuse.ext.orbital_elements import orbital_elements_from_binary
from amuse.io.base import read_set_from_file
from amuse.lab import constants, Particles, units

from plotters.file_reader import ReadData
from plotters.plotter_func import med_iqr, moving_average
from plotters.plotter_setup import PlotterSetup

class SystPlot(object):
    def __init__(self):
        self.data_path = ReadData().path
        self.config_path = ReadData().config_path
        self.path_len = ReadData().path_len
        self.runs = ReadData().org_files("simulation_snapshot")
        self.clean = PlotterSetup()

        self.colours = ["black", "Red", "Blue", "dodgerblue", "darkviolet"]

    def event_tracker(self, proc_data):
        """Track events occuring in simulation"""

        def plotting_function(ptype):
            print("...Event Detections for ", ptype, "...")
            self.config_red = proc_data
            in_path = self.data_path+proc_data+"/event_data/"
            files = natsorted(glob.glob(in_path+"/*"))

            time_df = np.linspace(0,30,10**4)
            Nejec_df = [np.zeros(len(time_df)) for i in range(len(files))]
            Ndrif_df = [np.zeros(len(time_df)) for i in range(len(files))]
            Nmerg_df = [np.zeros(len(time_df)) for i in range(len(files))]

            riter = 0
            for f_ in files:
                data = pd.read_hdf(f_)
                time = data.iloc[0]
                event = data.iloc[2]
                parti = data.iloc[-1]

                for ev_, tm_, pt_ in zip(event, time, parti):
                    if ptype.lower()=="all":
                        process = True
                    else:
                        if ev_.lower()!="merger":
                            if pt_.lower()=="imbh":
                                process = True
                            else:
                                process = False
                        else:
                            process = False
                    
                    if (process):
                        idx = (np.abs(time_df-tm_)).argmin()
                        if ev_.lower()=="ejection":
                            Nejec_df[riter][idx:] += 1
                        elif ev_.lower()=="drifter":
                            Ndrif_df[riter][idx:] += 1
                        else:
                            Nmerg_df[riter][idx:] += 1
                riter += 1
            Nejec_df = np.asarray(Nejec_df)
            Ndrif_df = np.asarray(Ndrif_df)
            Nmerg_df = np.asarray(Nmerg_df)

            med_ejec, IQRL_ejec, IQRH_ejec = med_iqr(Nejec_df)
            med_drif, IQRL_drif, IQRH_drif = med_iqr(Ndrif_df)
            med_merg, IQRL_merg, IQRH_merg = med_iqr(Nmerg_df)
            
            lines = ["Median Ejection: {}".format(med_ejec[-1]), 
                     "IQR Ejection: {}, {}".format(med_ejec[-1]-IQRL_ejec[-1], IQRH_ejec[-1]-med_ejec[-1]),
                     "Median Drifter: {}".format(med_drif[-1]), 
                     "IQR Drifter: {}, {}".format(med_drif[-1]-IQRL_drif[-1], IQRH_drif[-1]-med_drif[-1]),
                     "Median Merger: {}".format(med_merg[-1]), 
                     "IQR Merger: {}, {}".format(med_merg[-1]-IQRL_merg[-1], IQRH_merg[-1]-med_merg[-1])]

            with open(os.path.join("plotters/output",
                    self.config_red+"_"+ptype+"_Nevents.txt"), 'w') as f:
                for line_ in lines:
                    f.write(line_)
                    f.write('\n')
            
            med_val = [med_ejec, med_drif, med_merg]
            IQRL = [IQRL_ejec, IQRL_drif, IQRL_merg]
            IQRH = [IQRH_ejec, IQRH_drif, IQRH_merg]
            labels = [r"Ejection", r"Drifter", r"Merger"]

            fig, ax = plt.subplots()
            self.clean.tickers(ax, "plot", False)
            ax.set_xlabel(r"$t$ [Myr]", fontsize=self.clean.axlabel_size)
            ax.set_ylabel(r"$\langle N_{\mathrm{event}}\rangle$", 
                        fontsize=self.clean.axlabel_size)
            time_df = moving_average(time_df, 200)
            for k_ in range(len(med_val)):
                med_df = moving_average(med_val[k_], 200)
                IQRL_df = moving_average(IQRL[k_], 200)
                IQRH_df = moving_average(IQRH[k_], 200)
                ax.plot(time_df, med_df, color=self.colours[k_], label=labels[k_])
                ax.plot(time_df, IQRL_df, color=self.colours[k_], alpha=0.3)
                ax.plot(time_df, IQRH_df, color=self.colours[k_], alpha=0.3)
                ax.fill_between(time_df, IQRL_df, IQRH_df, color=self.colours[k_], alpha=0.3)
            ax.legend(fontsize=self.clean.axlabel_size)
            ax.set_xlim(time_df[0],time_df[-1])
            ax.set_ylim(0,1.01*max(IQRH_ejec))
            plt.savefig("plotters/figures/"+ptype+"Nevents"+self.config_red+".pdf", 
                        dpi=300, bbox_inches='tight')
            plt.close()

        plotting_function("all")
        plotting_function("imbh")

    def lagrange_evol(self, proc_data):
        """Plot evolution of Lagrangian radii"""

        print("...Lagrangian Data...")

        self.config_red = proc_data
        in_path = self.data_path+proc_data+"/simulation_snapshot/"
        config_files = natsorted(glob.glob(in_path+"/*"))[0] #FIX THIS

        lag25_star_arr = [ ]
        lag50_star_arr = [ ]
        lag75_star_arr = [ ]

        lag25_imbh_arr = [ ]
        lag50_imbh_arr = [ ]
        lag75_imbh_arr = [ ]
        if (True):#for run_ in config_files:
            snapshot = natsorted(glob.glob(config_files+"/*"))#natsorted(glob.glob(run_+"/*"))
            lag25 = [[ ], [ ]]
            lag50 = [[ ], [ ]]
            lag75 = [[ ], [ ]]
            for dt_ in snapshot:
                print(dt_)
                pset = read_set_from_file(dt_, "hdf5")
                IMBH = pset[pset.type=="imbh"]
                star = pset[pset.type=="star"]

                lag25[0].append(LagRad(star)[5].value_in(units.pc))
                lag50[0].append(LagRad(star)[6].value_in(units.pc))
                lag75[0].append(LagRad(star)[-1].value_in(units.pc))
                lag25[1].append(LagRad(IMBH)[5].value_in(units.pc))
                lag50[1].append(LagRad(IMBH)[6].value_in(units.pc))
                lag75[1].append(LagRad(IMBH)[-1].value_in(units.pc))

            lag25_star_arr.append(lag25[0])
            lag50_star_arr.append(lag50[0])
            lag75_star_arr.append(lag75[0])
            lag25_imbh_arr.append(lag25[1])
            lag50_imbh_arr.append(lag50[1])
            lag75_imbh_arr.append(lag75[1])

        lag25_star = med_iqr(lag25_star_arr)
        lag50_star = med_iqr(lag50_star_arr)
        lag75_star = med_iqr(lag75_star_arr)
        stardf = [lag25_star, lag50_star, lag75_star]

        lag25_imbh = med_iqr(lag25_imbh_arr)
        lag50_imbh = med_iqr(lag50_imbh_arr)
        lag75_imbh = med_iqr(lag75_imbh_arr)
        imbhdf = [lag25_imbh, lag50_imbh, lag75_imbh]

        time = [0.03*i for i in range(len(lag25_star[0]))]
        labels = [r"$r_{L,25}$", r"$r_{L,50}$", r"$r_{L,100}$"]
        
        fig, ax = plt.subplots()
        self.clean.tickers(ax, "plot", False)
        ax.set_xlabel(r"$t$ [Myr]", fontsize=self.clean.axlabel_size)
        ax.set_ylabel(r"$\log_{10}r_{L,x}$ [pc]", fontsize=self.clean.axlabel_size)
        for data_ in range(len(labels)):
            ax.plot(time, np.log10(stardf[data_][2]), color=self.colours[data_], label=labels[data_])
            ax.plot(time, np.log10(stardf[data_][1]), color=self.colours[data_], alpha=0.35)
            ax.plot(time, np.log10(stardf[data_][0]), color=self.colours[data_], alpha=0.35)
            ax.fill_between(time, stardf[data_][0], stardf[data_][1], 
                            alpha=0.35, color=self.colours[data_])
        ax.legend(fontsize=self.clean.axlabel_size, loc=4)
        ax.set_xlim(0,max(time))
        ax.set_ylim(0,1.02*max(np.log10(stardf[data_][2])))
        plt.savefig("plotters/figures/star_lagrangian"+self.config_red+".pdf", 
                    dpi=300, bbox_inches='tight')
        plt.close()
        
        fig, ax = plt.subplots()
        self.clean.tickers(ax, "plot", False)
        ax.set_xlabel(r"$t$ [Myr]", fontsize=self.clean.axlabel_size)
        ax.set_ylabel(r"$r_{L,x}$ [pc]", fontsize=self.clean.axlabel_size)
        for data_ in range(len(labels)):
            ax.plot(time, imbhdf[data_][2], color=self.colours[data_], label=labels[data_])
            ax.plot(time, imbhdf[data_][1], color=self.colours[data_], alpha=0.35)
            ax.plot(time, imbhdf[data_][0], color=self.colours[data_], alpha=0.35)
            ax.fill_between(time, imbhdf[data_][0], imbhdf[data_][1], 
                            alpha=0.35, color=self.colours[data_])
        ax.legend(fontsize=self.clean.axlabel_size, loc=4)
        ax.set_xlim(0,max(time))
        ax.set_ylim(0,1.02*max(imbhdf[data_][2]))
        plt.savefig("plotters/figures/IMBH_lagrangian"+self.config_red+".pdf", 
                    dpi=300, bbox_inches='tight')
        plt.close()

    def smbh_dist(self, proc_data):
        """Plot correlation function w.r.t SMBH position at final position"""

        print("...Two-Point Correlation Function...")
        self.config_red = proc_data
        in_path = self.data_path+proc_data+"/simulation_snapshot/"
        config_files = natsorted(glob.glob(in_path+"/*"))[0] #FIX THIS

        star_dist = [ ]
        imbh_dist = [ ]
        
        if (True):
        #for run_ in config_files:
            #print(run_)
            snapshot = natsorted(glob.glob(config_files+"/*"))#natsorted(glob.glob(run_+"/*"))
            dt_final = read_set_from_file(snapshot[-1], "hdf5")
            SMBH = dt_final[dt_final.type=="smbh"]
            for parti_ in dt_final[dt_final.type!="smbh"]:
                dist = (parti_.position-SMBH.position).length().value_in(units.pc)
                if np.isfinite(dist):
                    if parti_.type=="imbh":
                        imbh_dist.append(np.log10(dist))
                    else:
                        star_dist.append(np.log10(dist))
        
        star_dist = np.sort(star_dist)
        imbh_dist = np.sort(imbh_dist)
        
        fig = plt.figure(figsize=(5, 6))
        gs = fig.add_gridspec(2, 1, 
                              height_ratios=(2, 3), left=0.1, 
                              right=0.9, bottom=0.1, top=0.9, 
                              wspace=0.35, hspace=0.15)
        ax_bot = fig.add_subplot(gs[1, 0:2])
        ax_top = fig.add_subplot(gs[0, 0:2], sharex=ax_bot)
        ax_bot.set_xlabel(r'$\log_{10}\zeta(r_{i},r_{j})$ [pc]', fontsize=self.clean.axlabel_size)
        ax_bot.set_ylabel(r'$\log_{10}f_{r_{ij}<}$', fontsize=self.clean.axlabel_size)
        ax_top.set_ylabel(r'$\rho/\rho_{\mathrm{max}}$', fontsize=self.clean.axlabel_size)
        ax_bot, ax_top = self.clean.cdf_pdf(star_dist, ax_bot, ax_top, "red", 
                                            "Star-SMBH", "final", True, "Bot")
        ax_bot, ax_top = self.clean.cdf_pdf(imbh_dist, ax_bot, ax_top, "blue", 
                                            "IMBH-SMBH", "final", True, "Bot")
        plt.savefig("plotters/figures/SMBH_dist"+self.config_red+".pdf", 
                    dpi=300, bbox_inches='tight')
        plt.clf()

    def smbh_orb_elem(self, proc_data):
        """Plot final orbital parameters w.r.t SMBH"""

        print("...Orbital Elements w.r.t SMBH...")
        self.config_red = proc_data
        in_path = self.data_path+proc_data+"/simulation_snapshot/"
        config_files = natsorted(glob.glob(in_path+"/*"))[0] #TO FIX

        semi_fin = [[ ], [ ]]
        ecc_fin = [[ ], [ ]]
        if (True):
        #for run_ in config_files:
            snapshot = natsorted(glob.glob(config_files+"/*"))[-1]#natsorted(glob.glob(run_+"/*"))[-1]
            print("Processing: ", snapshot)
            dt_final = read_set_from_file(snapshot, "hdf5")
            SMBH_final = dt_final[dt_final.type=="smbh"]
            final_pset = dt_final-SMBH_final

            for parti_ in final_pset:
                if parti_.type == "star":
                    idx = 0
                else:
                    idx = 1
                bin_sys = Particles()
                bin_sys.add_particle(parti_)
                bin_sys.add_particle(SMBH_final)
                bin_sys.move_to_center()
                
                kepler_elements = orbital_elements_from_binary(bin_sys, G=constants.G)
                semi_fin[idx].append(np.log10(abs(kepler_elements[2]).value_in(units.pc)))
                ecc_fin[idx].append(np.log10(abs(kepler_elements[3])))
                
        fig = plt.figure(figsize=(10, 6))
        gs = fig.add_gridspec(2, 4,  width_ratios=(2, 2, 2, 2), 
                              height_ratios=(2, 3), left=0.1, 
                              right=0.9, bottom=0.1, top=0.9, 
                              wspace=0.5, hspace=0.15)
        axL = fig.add_subplot(gs[1, 0:2])
        axL1 = fig.add_subplot(gs[0, 0:2], sharex=axL)
        axR = fig.add_subplot(gs[1, 2:])
        axR1 = fig.add_subplot(gs[0, 2:], sharex=axR)
        axL.set_xlabel(r'$\log_{10}e_{\rm{SMBH}}$', fontsize=self.clean.axlabel_size)
        axL.set_ylabel(r'$\log_{10}f_{e<}$', fontsize=self.clean.axlabel_size)
        axL1.set_ylabel(r'$\rho/\rho_{\rm{max}}$', fontsize=self.clean.axlabel_size)
        axR.set_xlabel(r'$\log_{10}a_{\rm{SMBH}}$ [pc]', fontsize=self.clean.axlabel_size)
        axR.set_ylabel(r'$\log_{10}f_{a<}$', fontsize=self.clean.axlabel_size)
        axL, axL1 = self.clean.cdf_pdf(ecc_fin[0], axL, axL1, "red", "Star", "final", False, "Top")
        axL, axL1 = self.clean.cdf_pdf(ecc_fin[1], axL, axL1, "blue", "IMBH", "final", False, "Top")
        axR, axR1 = self.clean.cdf_pdf(semi_fin[0], axR, axR1, "red", "Star", "final", True, "Top")
        axR, axR1 = self.clean.cdf_pdf(semi_fin[1], axR, axR1, "blue", "IMBH", "final", True, "Top")
        axL.axvline(0, ls=":", color="black")
        axL1.axvline(0, ls=":", color="black")
        plt.savefig("plotters/figures/orb_sem_ecc"+self.config_red+".pdf", 
                    dpi=300, bbox_inches='tight')
        plt.close()

    def system_plotter(self, proc_data):
        """Plot spatial distribution of initial and final system"""

        print("...System Plotter...")
        self.config_red = proc_data
        path = self.data_path+proc_data+"/simulation_snapshot/"

        config_files = natsorted(glob.glob(path+"/*"))[0] #natsorted(glob.glob(path+"/*")) TO FIX
        dt_idx = [0,-1]
        fname_str = ["init", "final"]

        for idx_ in dt_idx:
            cfig = 0
            if (True):
            #for config_ in config_files:
                #print("Plotting for ", config_)
                cfig+=1
                fname = self.config_red+"_dt"+fname_str[idx_]+"_config"+str(cfig)+"_syst_evol.pdf"
                final_snap = natsorted(glob.glob(config_files+"/*"))[idx_]#natsorted(glob.glob(config_+"/*"))[idx_]
                snap_df = read_set_from_file(final_snap, "hdf5")
                SMBH = snap_df[snap_df.type=="smbh"]
                IMBH = snap_df[snap_df.type=="imbh"]
                star = snap_df[snap_df.type=="star"]

                fig, ax = plt.subplots(figsize=(6,6))
                ax.scatter(None, None, color="blue", label="IMBH")
                ax.scatter(None, None, color="red", label="Star")
                circle = plt.Circle((0,0), 0.01, color="black")
                ax.add_patch(circle)
                self.clean.tickers(ax, "plot", True)

                max_x = 0
                max_y = 0
                for parti_ in snap_df-SMBH:
                    max_x = max(abs(parti_.x-SMBH.x).value_in(units.pc), max_x)
                    max_y = max(abs(parti_.y-SMBH.y).value_in(units.pc), max_y)
                max_coord = max(max_x, max_y)

                imbh_pos = (IMBH.position-SMBH.position).value_in(units.pc)
                star_pos = (star.position-SMBH.position).value_in(units.pc)

                for pos, mass in zip(imbh_pos, IMBH.mass):
                    ax.scatter(pos[0], pos[1], color="blue",
                            s=0.5*(mass.value_in(units.MSun))**0.75)
                                
                for pos, mass in zip(star_pos, star.mass):
                    ax.scatter(pos[0], pos[1], color="red",
                            s=3*(mass.value_in(units.MSun))**0.75)
                                
                if idx_==-1:
                    ax.set_xlim(-1.05, 1.05)
                    ax.set_ylim(-1.05, 1.05)
                else:
                    ax.set_xlim(-1.1*max_coord, 1.1*max_coord)
                    ax.set_ylim(-1.1*max_coord, 1.1*max_coord)
                ax.set_xlabel(r"$x$ [pc]", fontsize=self.clean.axlabel_size)
                ax.set_ylabel(r"$y$ [pc]", fontsize=self.clean.axlabel_size)
                ax.legend(fontsize=self.clean.axlabel_size)
                plt.savefig("plotters/figures/system_layout/"+fname, dpi=300, bbox_inches='tight')
                
    def system_tester(self, proc_data):
        """Test various system aspects"""

        print("...System Plotter...")
        self.config_red = proc_data
        path = self.data_path+proc_data+"/simulation_snapshot/"
        config_files = natsorted(glob.glob(path+"/*"))

        cfig = 0
        for config_ in config_files:
            print("Plotting for ", config_)
            cfig+=1

            fname = self.config_red+"_dt_config"+str(cfig)+"_syst_evol.png"
            final_snap = natsorted(glob.glob(config_+"/*"))
            dt_iter = 0

            fig, ax = plt.subplots(figsize=(6,6))
            circle = plt.Circle((0,0), 0.01, color="black")
            self.clean.tickers(ax, "plot", True)
            c=["red","blue","green"]
            for dt_ in final_snap:
                print("Reading snapshot ", dt_)
                snap_df = read_set_from_file(dt_, "hdf5")
                SMBH = snap_df[snap_df.type=="smbh"]
                IMBH = snap_df[snap_df.type=="imbh"]
                star = snap_df[snap_df.type=="star"]
                minor = IMBH+star

                if dt_iter==0 or dt_iter==len(final_snap):
                    print("#Star=", len(star), "#IMBH=", len(IMBH))
                    print("Rvir=", snap_df.virial_radius().in_(units.pc))
                    print("Q=", snap_df.kinetic_energy()/snap_df.potential_energy())
                    print("Qimbh=", IMBH.kinetic_energy()/IMBH.potential_energy())
                    print("Qminor=", minor.kinetic_energy()/minor.potential_energy())
                    print("Star radius: ", star.radius.value_in(units.RSun))
                    print("Star mass: ", star.mass.value_in(units.MSun))
                    print("===========================")

                minor_pos = (star.position-SMBH.position).value_in(units.pc)
                for x_ in minor_pos:
                    ax.scatter(x_[0], x_[1], s=3)

                dt_iter+=1
                
            #ax.add_patch(circle)
            ax.set_xlabel(r"$x$ [pc]", fontsize=self.clean.axlabel_size)
            ax.set_ylabel(r"$y$ [pc]", fontsize=self.clean.axlabel_size)
            plt.savefig("plotters/figures/system_layout/"+fname, dpi=300, bbox_inches='tight')
            plt.close()

swag = SystPlot()
swag.lagrange_evol(swag.config_path[3][swag.path_len:])
#swag.smbh_dist(swag.config_path[3][swag.path_len:])
#swag.smbh_orb_elem(swag.config_path[3][swag.path_len:])
#swag.system_plotter(swag.config_path[3][swag.path_len:])
#swag.event_tracker(swag.config_path[3][swag.path_len:])
#swag.system_tester(swag.config_path[3][swag.path_len:])