import glob
import matplotlib.pyplot as plt
from natsort import natsorted
import numpy as np
import os
import pandas as pd

from amuse.ext.orbital_elements import orbital_elements_from_binary
from amuse.io.base import read_set_from_file
from amuse.lab import constants, units, Particles

from plotters.file_reader import ReadData
from plotters.gw_calcs import CalcGW
from plotters.plotter_setup import PlotterSetup


class EventCalc(object):
    def __init__(self):
        self.data_path = ReadData().path
        self.config_path = ReadData().config_path
        self.path_len = ReadData().path_len
        self.runs = ReadData().org_files("simulation_snapshot")
        self.clean = PlotterSetup()

        self.path = "/data_process/ejec_calcs"
        self.colours = ["black", "Red", "Blue", "dodgerblue", "darkviolet"]
        self.labels = ["SMBH-IMBH", "SMBH-Star", "IMBH-IMBH", "Star-Star", "Star-IMBH"]
        self.xticks = [None,None,None]
        self.yticks = self.xticks

    def read_data(self, proc_data):
        self.config_red = proc_data

        #Save ejection stats
        for initcond_ in self.config_path:
            ejec_dir = initcond_+"/ejec_snapshots/*"
            snap_dir = initcond_+"/simulation_snapshot/*"

            ejec_configs = natsorted(glob.glob(ejec_dir))
            sim_configs = natsorted(glob.glob(snap_dir))

            self.etype_df = [[ ] for i in range(len(ejec_configs))]
            self.evel_df = [[ ] for i in range(len(ejec_configs))]
            self.fparti_df = [[ ] for i in range(len(ejec_configs))]
            self.fsmbh_df = [[ ] for i in range(len(ejec_configs))]

            swag = [ ]

            citer = 0
            for econfig_, sconfig_ in zip(ejec_configs, sim_configs):
                ejec_snaps = natsorted(glob.glob(econfig_+"/*"))
                fin_snap = natsorted(glob.glob(sconfig_+"/*"))[-1]
                fin_snap = read_set_from_file(fin_snap, "hdf5")

                etype = [ ]
                evel = [ ]
                fparti = [ ]
                for snap_ in ejec_snaps:
                    parti_key = int(snap_.split("/ejec_")[-1])
                    snap_df = read_set_from_file(snap_, "hdf5")
                    ejecp = snap_df[snap_df.key==parti_key]
                    com_vel = snap_df.center_of_mass_velocity()

                    ejec_vel = (ejecp.velocity-com_vel).length()
                    evel.append(ejec_vel.value_in(units.kms))
                    etype.append(ejecp.type[0])
                    swag.append(ejecp)

                    finalp = fin_snap[fin_snap.key==parti_key]
                    fparti.append(finalp)

                SMBH = fin_snap[fin_snap.type=="smbh"]
                self.fsmbh_df[citer].append(SMBH)
                self.evel_df[citer].append(evel)
                self.etype_df[citer].append(etype)
                self.fparti_df[citer].append(fparti)
                citer+=1

            self.evel_df = np.asarray(self.evel_df, dtype=object)
            self.etype_df = np.asarray(self.etype_df, dtype=object)
            self.fparti_df = np.asarray(self.fparti_df, dtype=object)
            self.fsmbh_df = np.asarray(self.fsmbh_df, dtype=object)
        
            #Save merger stats
            #merg_dir = initcond_+"/ejec_snapshots/*"
            snap_dir = initcond_+"/simulation_snapshot/*"
            sim_configs = natsorted(glob.glob(snap_dir))
            
            self.mecc_df = [[ ] for i in range(len(ejec_configs))]
            self.msem_df = [[ ] for i in range(len(ejec_configs))]
            self.mtyp_df = [[ ] for i in range(len(ejec_configs))]

            citer = 0
            for config_ in sim_configs:
                snaps = natsorted(glob.glob(config_+"/*"))
                dt_iter = 0

                merger_keys = [ ]
                for dt_ in snaps:
                    parti_set = read_set_from_file(dt_, "hdf5")
                    if dt_iter==0:
                        prev_set = parti_set
                        prev_coll = np.sum(prev_set.coll_events)
                    if np.sum(parti_set.coll_events)>prev_coll:
                        mergers = prev_set
                        for p_ in parti_set:
                            if p_.key in mergers.key:
                                mergers -= p_
                        merger_keys.append([mergers[0].key, mergers[1].key])
                        prev_coll = np.sum(parti_set.coll_events)
                    dt_iter += 1

                sem_df = [ ]
                ecc_df = [ ]
                mrg_df = [ ]
                for merger_ in merger_keys:
                    for dt_ in snaps:
                        parti_set = read_set_from_file(dt_, "hdf5")
                        mergers = parti_set[parti_set.key==merger_[0]]
                        mergers += parti_set[parti_set.key==merger_[1]]
                        if len(mergers)>0:
                            kepler_elements = orbital_elements_from_binary(mergers, G=constants.G)
                            sem_df = np.concatenate((sem_df, np.log10(abs(kepler_elements[2].value_in(units.au)))), axis=None)
                            ecc_df = np.concatenate((ecc_df, np.log10(1-abs(kepler_elements[3]))), axis=None)
                            mrg_df = np.concatenate((mrg_df, [mergers[0].type, mergers[1].type]), axis=None)
                    prev_set = parti_set
                    prev_coll = np.sum(prev_set.coll_events)
                    
                self.mecc_df[citer] = np.concatenate((self.mecc_df[citer], ecc_df), axis=None)
                self.msem_df[citer] = np.concatenate((self.msem_df[citer], sem_df), axis=None)
                self.mtyp_df[citer] = np.concatenate((self.mtyp_df[citer], mrg_df), axis=None)
                citer+=1

            self.msem_df = np.asarray(self.msem_df, dtype=object)
            self.mecc_df = np.asarray(self.mecc_df, dtype=object)
            self.mtyp_df = np.asarray(self.mtyp_df, dtype=object)

    def ejec_hist_cdf(self):
        """Plot final distance histogram and save statistics"""

        def plotter_func(df, xlabel, ylabel, df_label, fname):
            """
               Plot histogram and CDF of data values
               
               Inputs:
               df:          Data values
               (x/y) label: Labels for (x/y) axis
               df_label:    Label for legend
               fname:       File output name
            """
            Nbins = 10

            xdata_star = np.sort(df[0])
            ydata_star = [i/len(xdata_star) for i in range(len(xdata_star))]
            xdata_imbh = np.sort(df[1])
            ydata_imbh = [i/len(xdata_imbh) for i in range(len(xdata_imbh))]

            n_vals = [ ]
            fig, ax = plt.subplots()
            for x_ in [xdata_star, xdata_imbh]:
                n, bins, patches = ax.hist(x_, Nbins)
                n_vals.append(n)
            ax.clear()

            fig = plt.figure(figsize=(6, 7))
            gs = fig.add_gridspec(2, 1, height_ratios=(2, 3), left=0.1, 
                                right=0.9, bottom=0.1, top=0.9, 
                                wspace=0.5, hspace=0.15)
            ax_bot = fig.add_subplot(gs[1, 0:2])
            ax_top = fig.add_subplot(gs[0, 0:2], sharex=ax_bot)
            ax_bot.set_ylabel(r'$\rho/\rho_{\rm{max}}$', fontsize=self.clean.axlabel_size)
            ax_bot.set_xlabel(xlabel, fontsize=self.clean.axlabel_size)
            ax_top.set_ylabel(ylabel, fontsize=self.clean.axlabel_size)
            iter=0
            for x_, y_ in zip([xdata_star, xdata_imbh], [ydata_star, ydata_imbh]):
                iter+=1
                n, bins, patches = ax_bot.hist(x_, Nbins, histtype='step', 
                                              color=self.colours[iter], 
                                              weights=[1/n_vals[iter-1].max()]*len(x_))
                n, bins, patches = ax_bot.hist(x_, Nbins, color=self.colours[iter], 
                                               alpha=0.4, weights=[1/n_vals[iter-1].max()]*len(x_))
                ax_top.plot(x_, y_, color=self.colours[iter], label=df_label[iter-1])
            self.clean.tickers(ax_bot, 'plot', False)
            self.clean.tickers(ax_top, 'plot', False)
            ax_top.legend(fontsize=self.clean.axlabel_size)
            ax_bot.set_xlim(0,1.01*max(xdata_star))
            ax_top.set_ylim(0.01, 1.01)
            plt.savefig(fname, dpi = 300, bbox_inches='tight')
            plt.close()

        star_dist_df = [ ]
        imbh_dist_df = [ ]
        for run, typ, SMBH in zip(self.fparti_df, self.etype_df, self.fsmbh_df):
            run=run[0]
            typ=typ[0]
            SMBH=SMBH[0]
            for parti_, typ_ in zip(run, typ):
                dist = (parti_.position-SMBH.position).length()
                if typ_.lower()=="star":
                    star_dist_df.append(dist.value_in(units.pc))
                else:
                    imbh_dist_df.append(dist.value_in(units.pc))
        plotter_func([star_dist_df, imbh_dist_df], r"$r_{\rm SMBH}$ [pc]", 
                     r"$f_{r_{\rm SMBH}<}$", ["Star", "IMBH"],
                     "plotters/figures/EjecDist_Hist_"+self.config_red+".pdf")

        star_vel_df = [ ]
        imbh_vel_df = [ ]
        for run, typ in zip(self.evel_df, self.etype_df):
            run=run[0]
            typ=typ[0]
            for vparti_, typ_ in zip(run, typ):
                if typ_.lower()=="star":
                    star_vel_df.append(vparti_)
                else:
                    imbh_vel_df.append(vparti_)
        plotter_func([star_vel_df, imbh_vel_df], r"$v_{\rm ejec}$ [km s$^{-1}$]", 
                     r"$f_{v_{\rm ejec}<}$", ["Star", "IMBH"],
                     "plotters/figures/EjecVel_Hist_"+self.config_red+".pdf")
                    
    def merger_evol(self, xdata, ydata, xlabel, ylabel, fname, plot_type):
        """Plot (a,(1-e)) and (f,h) diagram for merging events"""
        

        fig, ax = plt.subplots()
        if plot_type.lower()=="strain_freq":
            GW_plotter = CalcGW()
            GW_plotter.interferometer_plotter(ax)
            freq = [ ]
            strain = [ ]

        for x_, y_ in zip(xdata, ydata):
            if len(x_)>0:
                c = np.linspace(0,1,len(x_))
                ax.scatter(x_, y_, alpha=c)
        ax.set_xlabel(xlabel, fontsize=self.clean.axlabel_size)
        ax.set_ylabel(ylabel, fontsize=self.clean.axlabel_size)
        self.clean.tickers(ax, 'plot', False)
        plt.savefig(fname, dpi=300)
        plt.close()

    def stats_tracker(self):
        """Track and output statistics on events"""

        print("""...Event statistics...""")
        imbh_vel = [ ]
        star_vel = [ ]
        for vel_, typ_ in zip(self.evel_df, self.etype_df):
            typ_ = typ_[0]
            vel_ = vel_[0]
            for velocity, parti_type in zip(vel_, typ_):
                if parti_type.lower()=="star":
                    star_vel.append(velocity)
                else:
                    imbh_vel.append(velocity)

        med_vel_imbh = np.median(imbh_vel)
        med_vel_star = np.median(star_vel)
        IQRL_vel_imbh = med_vel_imbh - np.percentile(imbh_vel,25)
        IQRH_vel_imbh = np.percentile(imbh_vel,75) - med_vel_imbh
        IQRL_vel_star = med_vel_star - np.percentile(star_vel,25)
        IQRH_vel_star = np.percentile(star_vel,75) - med_vel_star

        ejec_type = np.asarray([np.zeros(2) for i in range(np.shape(self.evel_df)[0])])
        citer = 0
        for typ_ in self.etype_df:
            typ_ = typ_[0]
            for parti_type in typ_:
                if parti_type.lower()=="star":
                    ejec_type[citer][0]+=1
                else:
                    ejec_type[citer][1]+=1
            citer+=1
        med_type = [np.median(data) for data in zip(*ejec_type)]
        IQRL_type = [med_type[0] - np.percentile(ejec_type[:,0],25), 
                     med_type[1] - np.percentile(ejec_type[:,1],25)]
        IQRH_type = [np.percentile(ejec_type[:,0],75) - med_type[0], 
                     np.percentile(ejec_type[:,1],75) - med_type[1]]

        lines = ["IMBH Median Ejection Velocity [kms/s]: {}".format(med_vel_imbh), 
                 "IQR Ejection Velocity [kms/s]: {}, {}".format(IQRL_vel_imbh, IQRH_vel_imbh),
                 "Star Median Ejection Velocity [kms/s]: {}".format(med_vel_star), 
                 "IQR Ejection Velocity [kms/s]: {}, {}".format(IQRL_vel_star, IQRH_vel_star),
                 "\nEjection types (Stars || IMBH)",
                 "Median # Events: {}".format(med_type),
                 "IQR Events: {}, {}".format(IQRL_type, IQRH_type)]

        with open(os.path.join("plotters/output",
                  self.config_red+"_Events.txt"), 'w') as f:
            for line_ in lines:
                f.write(line_)
                f.write('\n')

swag = EventCalc()
swag.read_data(swag.config_path[0][swag.path_len:])
#swag.ejec_fparams()
#swag.ejec_hist_cdf()
swag.merger_evol(swag.msem_df, swag.mecc_df, 
                 r"$\log_{10}a$ [au]", r"$\log_{10}(1-e)$",
                 "plotters/figures/Merger_orb_params_"+swag.config_red+".pdf",
                 "orb_params")
#swag.stats_tracker()