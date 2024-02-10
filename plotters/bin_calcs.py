import glob
from itertools import combinations
import matplotlib.pyplot as plt
from natsort import natsorted
import numpy as np
import os
import pandas as pd
import pickle as pkl

from amuse.ext.orbital_elements import orbital_elements_from_binary
from amuse.io.base import read_set_from_file
from amuse.lab import constants, Particles, Particle, units

from plotters.file_reader import ReadData
from plotters.gw_calcs import CalcGW
from plotters.plotter_func import med_iqr, vdisp
from plotters.plotter_setup import PlotterSetup

np.seterr(divide='ignore', invalid='ignore')


class BinCalcs(object):
    def __init__(self):
        self.data_path = ReadData().path
        self.config_path = ReadData().config_path
        self.path_len = ReadData().path_len
        self.runs = ReadData().org_files("simulation_snapshot")
        self.clean = PlotterSetup()

        self.path = "/data_process/bin_calcs"

        self.colours = ["black", "Red", "Blue", "dodgerblue", "darkviolet"]
        self.labels = ["SMBH-IMBH", "SMBH-Star", "IMBH-IMBH", "Star-Star", "Star-IMBH"]
        self.xticks = [None,None,None]
        self.yticks = self.xticks

    def process_data(self):
        """Process snapshots into Binary/Hierarchical data files"""

        for config_ in self.config_path:
            if config_+self.path in glob.glob(config_+"/data_process/*"):
                None
            else:
                os.mkdir(config_+self.path)
        
        for initcond_ in self.config_path:
            configs = natsorted(glob.glob(initcond_+"/simulation_snapshot/*"))
            citer = 0
            for run_ in configs:
                print("Reading config: ", run_)
                fname = "bin_config"+run_[-2:]+".pkl"
                snapshot = natsorted(glob.glob(run_+"/*"))
                citer+=1
                dt_iter = 0
                for dt_ in snapshot:
                    print("Processing snapshot #: ", dt_iter)
                    snap_df = read_set_from_file(dt_, "hdf5")
                    snap_df = vdisp(snap_df)
                    
                    components = snap_df.connected_components(threshold=(1 | units.pc))
                    bin_no = 0
                    dt_iter+=1
                    for c in components:
                        if len(c)>1:
                            bin_combo = list(combinations(c,2))
                            msyst_keys = [ ]
                            for bin_ in bin_combo:
                                bin_sys = Particles()  
                                bin_sys.add_particle(bin_[0])
                                bin_sys.add_particle(bin_[1])
                                self.bin_dist = bin_sys.center_of_mass()-snap_df[snap_df.type=="smbh"].position
                                bin_sys.move_to_center()
                                
                                KE = bin_sys.kinetic_energy()
                                PE = bin_sys.potential_energy()
                                if KE<abs(PE):
                                    self.detect_bin(bin_sys)
                                    self.detect_hier(components, bin_sys)
                                    if self.bin_type.lower()!="false":
                                        bin_no+=1
                                        msyst_keys = np.concatenate((msyst_keys, [bin_[0].key, bin_[1].key]), axis=None)
                                        fname = "Binary_sim"+str(citer)+"_dt"+str(dt_iter)+"_bin"+str(bin_no)+".pkl"
                                        df_arr = pd.DataFrame()
                                        df_vals = pd.Series({"Simulation": citer, 
                                                             "Snapshot": dt_iter,
                                                             "Keys ": [bin_[0].key, bin_[1].key], 
                                                             "Type" : [bin_[0].type, bin_[1].type],
                                                             "Bin Hard": [self.bin_type], 
                                                             "Mass" : [bin_[0].mass, bin_[1].mass], 
                                                             "Semi-major" : [self.semi_inner.value_in(units.au)], 
                                                             "Inclinate" : [self.incl_inner], 
                                                             "Arg. of Pericenter" : [self.arg_peri_inner], 
                                                             "Ascending Node" : [self.asc_node_inner], 
                                                             "True anomaly" : [self.true_anom_inner], 
                                                             "Eccentricity" : [self.ecc_inner],
                                                             "Hierarchical Key" : [self.outer_key],
                                                             "Hierarchical Mass" : [self.outer_mass],
                                                             "Hierarchical Type" : [self.outer_type],
                                                             "Hierarchical Semi" : [self.semi_out],
                                                             "Hierarchical Ecc" : [self.ecc_out],
                                                             "Bin. Distance" : [self.bin_dist]
                                                             })
                                        df_arr = df_arr._append(df_vals, ignore_index=True)
                                        df_arr.to_pickle(os.path.join(initcond_+self.path, fname))

    def detect_bin(self, bin_sys):
        """Check if binary is tight enough relative to local neighbourhood
        
           Inputs:
           bin_sys:  Prospective binary system
           vdisp:  Velocity dispersion at 25 Lagrangian radii
        """

        self.bin_type = "False"
        kepler_elements = orbital_elements_from_binary(bin_sys, G=constants.G)
        self.semi_inner = kepler_elements[2]
        self.ecc_inner = kepler_elements[3]
        self.incl_inner = kepler_elements[4]
        self.arg_peri_inner = kepler_elements[5]
        self.asc_node_inner = kepler_elements[6]
        self.true_anom_inner = kepler_elements[7]

        vdisp = max(bin_sys.vdisp)
        soft_lim = constants.G*min(bin_sys.mass)/(2*vdisp**2)
        hard_lim = constants.G*min(bin_sys.mass)/(100*vdisp**2)
        if (self.ecc_inner<1):
            if self.semi_inner<hard_lim:
                self.bin_type = "Hard"
            elif self.semi_inner<soft_lim:
                self.bin_type = "Soft"
            elif self.semi_inner<(1|units.pc):
                self.bin_type = "Orbiting"

    def detect_hier(self, components, bin_sys):
        """Check if stable hierarchical system using Mardling & Aarseth 2001
        
           Inputs:
           components: Inner and outer component of system
           bin_sys:    Individual components of inner system
        """

        self.outer_key = [ ]
        self.semi_out = [ ]
        self.ecc_out = [ ]
        self.outer_mass = [ ]
        self.outer_type = [ ]
        for c in components:
            c = c[c.key!=bin_sys[0].key]
            c = c[c.key!=bin_sys[1].key]
            if len(c)>0:
                for c_ in c:
                    inner_p = Particle()
                    inner_p.mass = bin_sys.mass.sum()
                    inner_p.position = bin_sys.center_of_mass()
                    inner_p.velocity = bin_sys.center_of_mass_velocity()
                    hier_sys = Particles()
                    hier_sys.add_particle(inner_p)
                    hier_sys.add_particle(c_)

                    kepler_elements = orbital_elements_from_binary(hier_sys, G=constants.G)
                    semi_out = kepler_elements[2]
                    ecc_out = kepler_elements[3]
                    if (ecc_out)<1 and semi_out<(1e4 | units.au):
                        semi_ratio = abs(semi_out/self.semi_inner)
                        equality = 2.8*((1+hier_sys[1].mass/(inner_p.mass))*(1+ecc_out)/(1-ecc_out)**0.5)**0.4

                        if semi_ratio>equality:
                            self.outer_key.append(hier_sys[1].key)
                            self.semi_out.append(semi_out.value_in(units.au))
                            self.ecc_out.append(ecc_out)
                            self.outer_mass.append(hier_sys[1].mass)
                            self.outer_type.append(hier_sys[1].type)

    def read_data(self, proc_data):
        """Read pickle files"""

        self.config_red = proc_data
        path = self.data_path+proc_data+"/data_process/bin_calcs"

        self.bin_sim = [ ]
        self.bin_dt = [ ]
        self.bin_keys = [ ]
        self.bin_type = [ ]
        self.bin_hard = [ ]
        self.bin_mass = [ ]
        self.bin_semi = [ ]
        self.bin_incl = [ ]
        self.bin_aop = [ ]
        self.bin_ascn = [ ]
        self.bin_tanom = [ ]
        self.bin_ecc = [ ]
        self.bin_hkey = [ ]
        self.bin_hmass = [ ]
        self.bin_htype = [ ]
        self.bin_hsemi = [ ]
        self.bin_hecc = [ ]
        self.bin_dist = [ ]

        files = natsorted(glob.glob(path+"/*"))
        for file_ in files:
            with open(file_, 'rb') as bin_data:
                bin_data = pkl.load(bin_data)
                self.bin_sim.append(bin_data.iloc[0][0])
                self.bin_dt.append(bin_data.iloc[0][1])
                self.bin_keys.append(bin_data.iloc[0][2])
                self.bin_type.append(bin_data.iloc[0][3])
                self.bin_hard.append(bin_data.iloc[0][4])
                self.bin_mass.append(bin_data.iloc[0][5])
                self.bin_semi.append(bin_data.iloc[0][6])
                self.bin_incl.append(bin_data.iloc[0][7])
                self.bin_aop.append(bin_data.iloc[0][8])
                self.bin_ascn.append(bin_data.iloc[0][9])
                self.bin_tanom.append(bin_data.iloc[0][10])
                self.bin_ecc.append(bin_data.iloc[0][11])
                self.bin_hkey.append(bin_data.iloc[0][12])
                self.bin_hmass.append(bin_data.iloc[0][13])
                self.bin_htype.append(bin_data.iloc[0][14])
                self.bin_hsemi.append(bin_data.iloc[0][15])
                self.bin_hecc.append(bin_data.iloc[0][16])
                self.bin_dist.append(bin_data.iloc[0][17])
        self.bin_dt = np.asarray(self.bin_dt)
        self.bin_sim = np.asarray(self.bin_sim)

    def gw_freq_strain(self):
        """Plot binary f vs. h GW diagram"""
        
        star_imbh = [[ ], [ ]]
        imbh_imbh = [[ ], [ ]]
        star_star = [[ ], [ ]]
        star_smbh = [[ ], [ ]]
        imbh_smbh = [[ ], [ ]]

        star_imbh_type = [ ]
        imbh_imbh_type = [ ]
        star_star_type = [ ]
        star_smbh_type = [ ]
        imbh_smbh_type = [ ]

        GW_calc = CalcGW(dvals="all")
        for style, type, sem, ecc, mass in zip(self.bin_hard, self.bin_type, self.bin_semi, self.bin_ecc, self.bin_mass):
            ecc = ecc[0]
            sem = sem[0]*(1 | units.au)
            
            bin_sys = Particles(2)
            bin_sys[0].mass = mass[0]
            bin_sys[1].mass = mass[1]

            nharm = GW_calc.GW_harmonic(ecc)
            freq = GW_calc.GW_freq(sem, nharm, bin_sys)
            strain = np.log10(GW_calc.GW_strain(sem, ecc, freq, bin_sys, nharm))
            freq = np.log10(freq.value_in(units.Hz))
            if "smbh" in type:
                if "imbh" in type:
                    imbh_smbh[0].append(freq)
                    imbh_smbh[1].append(strain)
                    imbh_smbh_type = np.concatenate((imbh_smbh_type, style), axis=None)
                else:
                    star_smbh[0].append(freq)
                    star_smbh[1].append(strain)
                    star_smbh_type = np.concatenate((star_smbh_type, style), axis=None)
            elif "imbh" in type:
                if type[0]==type[1]:
                    imbh_imbh[0].append(freq)
                    imbh_imbh[1].append(strain)
                    imbh_imbh_type = np.concatenate((imbh_imbh_type, style), axis=None)
                else:
                    star_imbh[0].append(freq)
                    star_imbh[1].append(strain)
                    star_imbh_type = np.concatenate((star_imbh_type, style), axis=None)
            else:
                star_star[0].append(freq)
                star_star[1].append(strain)
                star_star_type = np.concatenate((star_star_type, style), axis=None)

        data_arr = [imbh_smbh, star_smbh, imbh_imbh, star_star, star_imbh]
        styl_arr = [imbh_smbh_type, star_smbh_type, imbh_imbh_type, 
                    star_star_type, star_imbh_type]
        fname = ["plotters/figures/AllBin_GW_freq_strain"+self.config_red+".pdf", 
                 "plotters/figures/HardCritBin_GW_freq_strain"+self.config_red+".pdf"]

        for p_ in range(2):
            fig = plt.figure(figsize=(8, 6))
            gs = fig.add_gridspec(2, 2,  width_ratios=(4, 2), height_ratios=(2, 4),
                                left=0.1, right=0.9, bottom=0.1, top=0.9,
                                wspace=0.05, hspace=0.05)
            ax = fig.add_subplot(gs[1, 0])
            ax1 = fig.add_subplot(gs[0, 0], sharex=ax)
            ax2 = fig.add_subplot(gs[1, 1], sharey=ax)
            for ax_ in [ax, ax1, ax2]:
                self.clean.tickers(ax_, "plot", False)
            for k_ in range(len(data_arr)):
                if p_ == 0:
                    freq = data_arr[k_][0]
                    strain = data_arr[k_][1]
                else:
                    data_arr[k_][0] = np.asarray(data_arr[k_][0])
                    data_arr[k_][1] = np.asarray(data_arr[k_][1])

                    freq = data_arr[k_][0][styl_arr[k_]=="Hard"]
                    strain = data_arr[k_][1][styl_arr[k_]=="Hard"]
                    
                ax.scatter(None, None, label=self.labels[k_], color=self.colours[k_])
                ax.scatter(freq, strain, color=self.colours[k_], s=3)
                if len(freq)>2:
                    KDE_x, KDE_y = GW_calc.KDE_plotter([freq, strain], False)
                    ax1.tick_params(axis="x", labelbottom=False)
                    ax2.tick_params(axis="y", labelleft=False)

                    ax1.plot(KDE_x[0], KDE_x[1], color=self.colours[k_])
                    ax1.fill_between(KDE_x[0], KDE_x[1], alpha=0.35, color=self.colours[k_])
                    ax2.plot(KDE_y[0], KDE_y[1], color=self.colours[k_])
                    ax2.fill_between(KDE_y[0], KDE_y[1], alpha=0.35, color=self.colours[k_])
            GW_calc.interferometer_plotter(ax)
            
            ax.set_xlim(-13,0)
            ax.set_ylim(-33,-12)
            ax.set_xlabel(r"$\log_{10}f$ [Hz]", fontsize=self.clean.axlabel_size)
            ax.set_ylabel(r"$\log_{10}h$", fontsize=self.clean.axlabel_size)
            
            ax1.set_ylabel(r'$\rho/\rho_{\rm{max}}$', 
                        fontsize=self.clean.axlabel_size)
            ax.legend(fontsize=self.clean.axlabel_size, 
                    bbox_to_anchor=(1.53,1.53))
            ax1.set_ylim(0.01,1.04)
            ax2.set_xlabel(r'$\rho/\rho_{\rm{max}}$', 
                        fontsize=self.clean.axlabel_size)
            ax2.set_xlim(0.01,1.04)

            plt.savefig(fname[p_], dpi=300, bbox_inches='tight')
            plt.close()

    def pop_evol(self):
        """Plot binary demographic evolution over sim. time"""

        configs = '/home/erwanh/Runaway_BH/'+self.config_red
        Nfiles = len(glob.glob(configs+"/simulation_snapshot/*")[:3])
        Nsnaps = len(glob.glob(configs+"/simulation_snapshot/config_3*/*"))
        
        smbh_imbh = np.asarray([np.zeros(Nsnaps) for i in range(Nfiles)], dtype=object)
        smbh_star = np.asarray([np.zeros(Nsnaps) for i in range(Nfiles)], dtype=object)
        star_imbh = np.asarray([np.zeros(Nsnaps) for i in range(Nfiles)], dtype=object)
        star_star = np.asarray([np.zeros(Nsnaps) for i in range(Nfiles)], dtype=object)
        imbh_imbh = np.asarray([np.zeros(Nsnaps) for i in range(Nfiles)], dtype=object)

        for type_, run_, dt_ in zip(self.bin_type, self.bin_sim, self.bin_dt):
            dt_ -= 1
            run_ -= 1
            if "smbh" in type_:
                if "imbh" in type_:
                    smbh_imbh[run_][dt_] += 1
                else:
                    smbh_star[run_][dt_] += 1
            elif "imbh" in type_:
                if type_[0]==type_[1]:
                    imbh_imbh[run_][dt_] += 1
                else:
                    star_imbh[run_][dt_] += 1
            else:
                star_star[run_][dt_] += 1
                
        med_sm_im, IQRL_sm_im, IQRH_sm_im = med_iqr(smbh_imbh)
        med_sm_st, IQRL_sm_st, IQRH_sm_st = med_iqr(smbh_star)
        med_im_im, IQRL_im_im, IQRH_im_im = med_iqr(imbh_imbh)
        med_st_st, IQRL_st_st, IQRH_st_st = med_iqr(star_star)
        
        time = [0.03*i for i in range(Nsnaps)]

        fig, ax = plt.subplots()
        ax.set_ylabel(r'$\langle N\rangle$', fontsize=self.clean.axlabel_size)
        ax.set_xlabel(r'$t$ [Myr]', fontsize=self.clean.axlabel_size)
        ax.plot(time, med_sm_im, color=self.colours[0], alpha=0.7)
        ax.plot(time, med_sm_st, color=self.colours[1], alpha=0.7)
        ax.plot(time, med_im_im, color=self.colours[2], alpha=0.7)
        ax.plot(time, med_st_st, color=self.colours[3], alpha=0.7)
        ax.scatter(time, med_sm_im, color=self.colours[0], label=self.labels[0])
        ax.scatter(time, med_sm_st, color=self.colours[1], label=self.labels[1])
        ax.scatter(time, med_im_im, color=self.colours[2], label=self.labels[2])
        ax.scatter(time, med_st_st, color=self.colours[3], label=self.labels[3])
        ax.legend(fontsize=self.clean.axlabel_size)
        self.clean.tickers(ax, "plot", 2)
        plt.savefig("plotters/figures/Bin_Evol"+self.config_red+".pdf", 
                    dpi=300, bbox_inches='tight')
        plt.close()

    def pop_stats(self):
        """Extract statistics on binary population"""

        configs = '/home/erwanh/Runaway_BH/'+self.config_red
        Nfiles = len(glob.glob(configs+"/simulation_snapshot/*"))
        Nsnaps = len(glob.glob(configs+"/simulation_snapshot/config_0/*"))
        
        smbh_imbh = np.zeros(Nfiles)
        smbh_star = np.zeros(Nfiles)
        star_imbh = np.zeros(Nfiles)
        star_star = np.zeros(Nfiles)
        imbh_imbh = np.zeros(Nfiles)

        smbh_imbh_style = np.zeros(Nfiles)
        smbh_star_style = np.zeros(Nfiles)
        star_imbh_style = np.zeros(Nfiles)
        star_star_style = np.zeros(Nfiles)
        imbh_imbh_style = np.zeros(Nfiles)
        for style_, type_, run_, dt_ in zip(self.bin_hard, self.bin_type, self.bin_sim, self.bin_dt):
            if dt_ == max(self.bin_dt[self.bin_sim==run_]):
                if "smbh" in type_:
                    if "imbh" in type_:
                        smbh_imbh[run_] += 1
                        if style_[0].lower()!="extra soft":
                            smbh_imbh_style[run_] += 1
                    else:
                        smbh_star[run_] += 1
                        if style_[0].lower()!="extra soft":
                            smbh_star_style[run_] += 1
                elif "imbh" in type_:
                    if type_[0]==type_[1]:
                        imbh_imbh[run_] += 1
                        if style_[0].lower()!="extra soft":
                            imbh_imbh_style[run_] += 1
                    else:
                        star_imbh[run_] += 1
                        if style_[0].lower()!="extra soft":
                            star_imbh_style[run_] += 1
                else:
                    star_star[run_] += 1
                    if style_[0].lower()!="extra soft":
                        star_star_style[run_] += 1
        
        bin_hard = np.zeros(Nfiles)
        bin_soft = np.zeros(Nfiles)
        for btype_, run_ in zip(self.bin_hard, self.bin_sim):
            run_ -= 1
            if btype_[0].lower=="hard":
                bin_hard[run_]+=1
            else:
                bin_soft[run_]+=1

        with open(os.path.join("plotters/output",
                  self.config_red+"_BinHier.txt"), 'w') as f:

            f.write("All Bins: \n")
            df_all = [smbh_imbh, smbh_star, star_imbh, star_star, imbh_imbh]
            self.write_pop_stats(df_all, f)
            hard_ratio = [ ]
            for dfh_, dfs_ in zip(bin_hard, bin_soft):
                hard_ratio.append(dfh_/(dfh_+dfs_))
            med = np.median(hard_ratio)
            IQRL = med-np.percentile(hard_ratio, 25)
            IQRH = np.percentile(hard_ratio, 75)-med

            f.write("Median Hard %: {}\n".format(med))
            f.write("IQR [{}, {}]\n\n".format(IQRL, IQRH))
            f.write("=====================\n")
            f.write("Hardness Criteria: \n")
            df_crit = [smbh_imbh_style, smbh_star_style, star_imbh_style, star_star_style, imbh_imbh_style]
            self.write_pop_stats(df_crit, f)
            
    def sem_ecc(self):
        """Plot (a,e) diagram of detected binaries"""
        
        star_imbh = [[ ], [ ]]
        imbh_imbh = [[ ], [ ]]
        star_star = [[ ], [ ]]
        star_smbh = [[ ], [ ]]
        imbh_smbh = [[ ], [ ]]
        imbh_smbh_style = [[ ], [ ]]

        for sim, dt, style, type, sem, ecc, dt in zip(self.bin_sim, self.bin_dt, 
                                                      self.bin_hard, self.bin_type, 
                                                      self.bin_semi, self.bin_ecc, 
                                                      self.bin_dt):
            if dt == max(self.bin_dt[self.bin_sim==sim]):
                ecc = np.log10(1-ecc[0])
                sem = np.log10(sem[0])
                if "smbh" in type:
                    if "star" in type:
                        star_smbh[0].append(sem)
                        star_smbh[1].append(ecc)
                    else:
                        imbh_smbh[0].append(sem)
                        imbh_smbh[1].append(ecc)
                        if style[0].lower()!="extra soft":
                            imbh_smbh_style[0].append(sem)
                            imbh_smbh_style[1].append(ecc)
                elif type[0] == type[1]:
                    if type[0]=="star":
                        star_star[0].append(sem)
                        star_star[1].append(ecc)
                    else:
                        imbh_imbh[0].append(sem)
                        imbh_imbh[1].append(ecc)
                else:
                    star_imbh[0].append(sem)
                    star_imbh[1].append(ecc)
        
        fig = plt.figure(figsize=(10, 6))
        gs = fig.add_gridspec(2, 4,  width_ratios=(2, 2, 2, 2), 
                              height_ratios=(2, 3), left=0.1, 
                              right=0.9, bottom=0.1, top=0.9, 
                              wspace=0.5, hspace=0.15)
        axL = fig.add_subplot(gs[1, 0:2])
        axL1 = fig.add_subplot(gs[0, 0:2], sharex=axL)
        axR = fig.add_subplot(gs[1, 2:])
        axR1 = fig.add_subplot(gs[0, 2:], sharex=axR)
        axL.set_xlabel(r'$\log_{10}(1-e)$', fontsize=self.clean.axlabel_size)
        axL.set_ylabel(r'$\log_{10}f_{(1-e)<}$', fontsize=self.clean.axlabel_size)
        axL1.set_ylabel(r'$\rho/\rho_{\rm{max}}$', fontsize=self.clean.axlabel_size)
        axR.set_xlabel(r'$\log_{10}a$ [au]', fontsize=self.clean.axlabel_size)
        axR.set_ylabel(r'$\log_{10}f_{a<}$', fontsize=self.clean.axlabel_size)
        axR1.legend(fontsize=self.clean.axlabel_size)
        if len(star_imbh[0])>0:
            axL, axL1 = self.clean.cdf_pdf(star_imbh[1], axL, axL1, "black", "Star-IMBH", "final", False, "Top")
            axR, axR1 = self.clean.cdf_pdf(star_imbh[0], axR, axR1, "black", "Star-IMBH", "final", True, "Top")
        axL, axL1 = self.clean.cdf_pdf(imbh_smbh[1], axL, axL1, "blue", "IMBH-SMBH", "final", False, "Top")
        axL, axL1 = self.clean.cdf_pdf(star_smbh[1], axL, axL1, "red", "Star-SMBH", "final", False, "Top")
        axR, axR1 = self.clean.cdf_pdf(imbh_smbh[0], axR, axR1, "blue", "IMBH-SMBH", "final", True, "Top")
        axR, axR1 = self.clean.cdf_pdf(star_smbh[0], axR, axR1, "red", "Star-SMBH", "final", True, "Top")
        plt.savefig("plotters/figures/sem_ecc_bin"+self.config_red+".pdf", 
                    dpi=300, bbox_inches='tight')
        plt.close()
    
    def write_pop_stats(self, array, file):
        """Function to write population statistics into file
        
           Inputs:
           array:  Array hosting data values
           file:   Output file
        """

        iloop = 0
        for df_ in array:
            med = np.median(df_)
            IQRL = med-np.percentile(df_, 25)
            IQRH = np.percentile(df_, 75)-med

            if iloop==0:
                file.write("SMBH-IMBH: \n")
            elif iloop==1:
                file.write("SMBH-Star: \n")
            elif iloop==2:
                file.write("Star-IMBH: \n")
            elif iloop==3:
                file.write("Star-Star: \n")
            elif iloop==4:
                file.write("IMBH-IMBH: \n")

            file.write("Median # {} \n".format(med))
            file.write("IQR [{},{}] \n\n".format(IQRL, IQRH))
            iloop += 1

swag = BinCalcs()
swag.process_data()
swag.read_data(swag.config_path[3][swag.path_len:])
swag.pop_evol()
swag.gw_freq_strain()
swag.pop_stats()
swag.sem_ecc()