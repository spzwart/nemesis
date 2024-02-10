import glob
from itertools import combinations
import LISA_Curves.LISA as li
import matplotlib.pyplot as plt
from natsort import natsorted
import numpy as np
import os
import pandas as pd
import pickle as pkl
from scipy import stats
from scipy.special import jv
import statsmodels.api as sm

from amuse.ext.orbital_elements import orbital_elements_from_binary
from amuse.io.base import read_set_from_file
from amuse.lab import constants, Particles, units

from plotters.file_reader import ReadData
from plotters.plotter_func import moving_average
from plotters.plotter_setup import PlotterSetup


class CalcGW(object):
    def __init__(self, dvals):
        self.data_path = ReadData().path
        self.config_path = ReadData().config_path
        self.path_len = ReadData().path_len
        self.runs = ReadData().org_files("simulation_snapshot")
        self.clean = PlotterSetup()

        self.path = "/data_process/gw_calcs"
        self.rlim_sec = 1e4 | units.au
        self.lum_dist = 1 | units.Gpc
        self.redshift = 0.19724       #z(dL=1 Gpc) using Ned Wright & arXiv:1807.06209

        self.dvals = dvals
        self.colours = ["Blue", "Red", "Black", "dodgerblue", "darkviolet"]
        self.labels = ["SMBH-IMBH", "SMBH-Star", "IMBH-IMBH", "Star-Star", "Star-IMBH"]

    def process_data(self):
        """Process snapshots into GW data files"""

        for config_ in self.config_path:
            if config_+self.path in glob.glob(config_+"/data_process/*"):
                None
            else:
                os.mkdir(config_+self.path)
        
        for initcond_ in self.config_path:
            configs = natsorted(glob.glob(initcond_+"/simulation_snapshot/*"))
            for run_ in configs:
                if "DONE" not in run_:
                    print("Reading config: ", run_)
                    fname = "gw_config"+run_[-2:]+".pkl"
                    snapshot = natsorted(glob.glob(run_+"/*"))
                    dt_iter = 0

                    gw_type = [ ]
                    gw_key = [ ]
                    gw_semi = [ ]
                    gw_ecc = [ ]
                    gw_freq = [ ]
                    gw_strain = [ ]

                    dtiter = 0
                    for dt_ in snapshot:
                        print("Reading snapshot #: ", dtiter)
                        dtiter+=1 

                        gw_type_dt = [ ]
                        gw_key_dt = [ ]
                        gw_semi_dt = [ ]
                        gw_ecc_dt = [ ]
                        gw_freq_dt = [ ]
                        gw_strain_dt = [ ]

                        snap_df = read_set_from_file(dt_, "hdf5")
                        secondaries = snap_df[snap_df.type!="smbh"]
                        SMBH = snap_df[snap_df.type=="smbh"][0]
                            
                        for parti_ in secondaries:
                            bin_sys = Particles()
                            bin_sys.add_particle(parti_)
                            bin_sys.add_particle(SMBH)

                            kepler_elements = orbital_elements_from_binary(bin_sys, G=constants.G)
                            semimajor = abs(kepler_elements[2])
                            eccentric = abs(kepler_elements[3])
                            if eccentric<1:
                                nharm = self.GW_harmonic(eccentric)
                                frequency = self.GW_freq(semimajor, nharm, bin_sys)
                                strain = self.GW_strain(semimajor, eccentric, 
                                                        frequency, bin_sys, nharm)
                                if (strain>1e-30):
                                    gw_type_dt.append([parti_.type, SMBH.type])
                                    gw_key_dt.append([parti_.key, SMBH.key])
                                    gw_semi_dt.append(semimajor)
                                    gw_ecc_dt.append(eccentric)
                                    gw_freq_dt.append(frequency)
                                    gw_strain_dt.append(strain)

                        sec_neigh = secondaries.connected_components(threshold=self.rlim_sec)
                        for sys_ in sec_neigh:
                            if len(sys_)>1:
                                bin_systs = list(combinations(sys_,2))
                                for bin_ in bin_systs:
                                    bin_sys = Particles()
                                    bin_sys.add_particle(bin_[0])
                                    bin_sys.add_particle(bin_[1])

                                    kepler_elements = orbital_elements_from_binary(bin_sys, G=constants.G)
                                    semimajor = abs(kepler_elements[2])
                                    eccentric = abs(kepler_elements[3])
                                    if eccentric<1:
                                        nharm = self.GW_harmonic(eccentric)
                                        frequency = self.GW_freq(semimajor, nharm, bin_sys)
                                        strain = self.GW_strain(semimajor, eccentric, 
                                                                frequency, bin_sys, nharm)
                                        if (strain>1e-30):
                                            gw_key_dt.append([bin_[0].key, bin_[1].key])
                                            gw_type_dt.append([bin_[0].type, bin_[1].type])
                                            gw_semi_dt.append(semimajor)
                                            gw_ecc_dt.append(eccentric)
                                            gw_freq_dt.append(frequency)
                                            gw_strain_dt.append(strain)

                        gw_key.append(gw_key_dt)
                        gw_type.append(gw_type_dt)
                        gw_semi.append(gw_semi_dt)
                        gw_ecc.append(gw_ecc_dt)
                        gw_freq.append(gw_freq_dt)
                        gw_strain.append(gw_strain_dt)
                        dt_iter += 1

                    gw_data = pd.Series({"Particle Key": gw_key, 
                                         "Particle Type": gw_type,
                                         "GW Semi" : gw_semi,
                                         "GW Ecc" : gw_ecc, 
                                         "GW Freq" : gw_freq, 
                                         "GW Strain" : gw_strain})
                    gw_data.to_pickle(os.path.join(initcond_+self.path+"/", fname))

                if "DONE" not in run_:
                    os.rename(run_, run_+"DONE")
                    
    def GW_freq(self, semi, nharm, bin_parti):
        """
        Frequency equation is based on Samsing et al. 2014 eqn (43). 
        
        Inputs:
        semi:      Binary semi-major axis
        nharm:     GW harmonic mode
        bin_parti: Binary particle set
        """

        m1 = bin_parti[0].mass
        m2 = bin_parti[1].mass

        freq =  (2*np.pi)**-1*np.sqrt(constants.G*(m1+m2)/abs(semi)**3)*nharm
        return freq
    
    def GW_dfreq(self, semi, nharm, bin_parti, chirp_mass, ecc_func):
        """
        Function to take into account the limited LISA observation time, T ~ 5yrs
        Based on equation (6) of Kremer et al. 2019.
        
        Inputs:
        semi:       Binary semi-major axis
        nharm:      GW harmonic mode
        bin_parti:  Binary particle set
        chirp_mass: Binary chirp mass
        ecc_func:   Value of eccentricity func. for binary
        """

        m1 = bin_parti[0].mass
        m2 = bin_parti[1].mass

        forb = np.sqrt(constants.G*(m1+m2))/(2*np.pi)*abs(semi)**-1.5*(self.redshift+1)**-1
        dfreq = (96*nharm)/(10*np.pi)*(constants.G*chirp_mass)**(5/3)/(constants.c**5)*(2*np.pi*forb)**(11/3)*abs(ecc_func)
        return dfreq

    def GW_strain(self, semi, ecc, freq, bin_parti, nharm):
        """
        Use of eqn (7) Kremer et al. 2018.
        Use of eqn (20) of Peters and Matthews (1963).
        
        Inputs:
        semi:      Binary semi-major axis
        ecc:       Binary eccentricity
        freq:      Binary GW frequency
        bin_parti: Binary particle set
        nharm:     GW harmonic mode
        """

        m1 = bin_parti[0].mass
        m2 = bin_parti[1].mass

        chirp_mass = (m1*m2)**0.6/((1+self.redshift)*(m1+m2)**0.2)
        cfactor = 2/(3*np.pi**(4/3))*(constants.G**(5/3))/(constants.c**3)*(self.lum_dist*(1+self.redshift))**-2
        ecc_func = (1+(73/24)*ecc**2+(37/96)*ecc**4)*(1-ecc**2)**-3.5
        dfreq = self.GW_dfreq(semi, nharm, bin_parti, chirp_mass, ecc_func)
        factor = min(1, dfreq*(5 | units.yr)/freq)

        strain = factor*cfactor*chirp_mass**(5/3)*freq**(-1/3)*(2/nharm)**(2/3)*(self.GW_gfunc(ecc, nharm)/ecc_func)
        strain = (strain.value_in(units.s**-1.6653345369377348e-16))**0.5

        return strain

    def GW_gfunc(self, ecc, nharm):
        return nharm**4/32*((jv(nharm-2, nharm*ecc)-2*ecc*jv(nharm-1, nharm*ecc) + 2/nharm*jv(nharm, nharm*ecc) \
               + 2*ecc*jv(nharm+1, nharm*ecc) - jv(nharm+2, nharm*ecc))**2 + (1-ecc**2)*(jv(nharm-2, nharm*ecc) \
               - 2*jv(nharm, nharm*ecc) + jv(nharm+2, nharm*ecc))**2 + 4/(3*nharm**2)*(jv(nharm, nharm*ecc)**2))

    def GW_harmonic(self, ecc):
        """
        Finding the peak harmonic of gravitational frequency
        Equation 36 of Wen (2003)
        """ 

        nharm = 2*(1+ecc)**1.1954/(1-ecc**2)**1.5
        return nharm

    def GW_time(self, semi, ecc, m1, m2):
        """
        Calculate the GW timescale based on Peters (1964).
        
        Inputs:
        semi:    The semi-major axis of the binary
        ecc:     The eccentricity of the binary
        m1/m2:   The binary component masses
        outputs: The gravitational wave timescale
        """

        red_mass = (m1*m2)/(m1+m2)
        tot_mass = m1 + m2
        tgw = (5/256)*(constants.c)**5/(constants.G**3)*(semi**4*(1-ecc**2)**3.5)/(red_mass*tot_mass**2)
        return tgw

    def read_data(self, proc_data):
        """Read pickle files"""
        
        self.sim = [ ]
        self.gw_key = [ ]
        self.gw_typ = [ ]
        self.gw_sem = [ ]
        self.gw_ecc = [ ]
        self.gw_frq = [ ]
        self.gw_str = [ ]

        self.config_red = proc_data
        path = self.data_path+proc_data+"/data_process/gw_calcs"

        gw_files = natsorted(glob.glob(path+"/*"))
        sim = 0
        for file_ in gw_files:
            with open(file_, 'rb') as gw_data:
                sim += 1
                gw_data = pkl.load(gw_data)
                if self.dvals.lower()=="final":
                    dt_idx = -1
                    self.sim.append(sim)
                    self.gw_key.append(gw_data.iloc[0][dt_idx][0])
                    self.gw_typ.append(gw_data.iloc[1][dt_idx][0])
                    self.gw_sem.append(gw_data.iloc[2][dt_idx][0])
                    self.gw_ecc.append(gw_data.iloc[3][dt_idx][0])
                    self.gw_frq.append(gw_data.iloc[4][dt_idx][0])
                    self.gw_str.append(gw_data.iloc[5][dt_idx][0])
                elif self.dvals.lower()=="all":
                    self.sim.append(sim)
                    self.gw_key.append(gw_data.iloc[0])
                    self.gw_typ.append(gw_data.iloc[1])
                    self.gw_sem.append(gw_data.iloc[2])
                    self.gw_ecc.append(gw_data.iloc[3])
                    self.gw_frq.append(gw_data.iloc[4])
                    self.gw_str.append(gw_data.iloc[5])

        self.sim = np.asarray(self.sim, dtype=object)
        self.gw_typ = np.asarray(self.gw_typ, dtype=object)
        self.gw_sem = np.asarray(self.gw_sem, dtype=object)
        self.gw_ecc = np.asarray(self.gw_ecc, dtype=object)
        self.gw_frq = np.asarray(self.gw_frq, dtype=object)
        self.gw_str = np.asarray(self.gw_str, dtype=object)

    def gw_stats(self, config):
        """Extract GW event statistics"""

        def data_track(typ_, idx):
            """
            Track event types
            
            Inputs:
            typ_:   Array hosting GW event constituents
            idx:    Integer identifying if event is observable
            """

            if 'smbh' in typ_:
                if 'star' in typ_:
                    SMBH_star[idx]+=1
                else:
                    SMBH_IMBH[idx]+=1
            elif 'imbh' not in typ_:
                star_star[idx]+=1
            elif 'star' not in typ_:
                IMBH_IMBH[idx]+=1
            else:
                IMBH_star[idx]+=1

        def data_stats(array):
            """Track median and IQR of data across all sims"""

            median = [np.median(config_val) for config_val in zip(*array)]
            IQRL = [med-np.percentile(config_val, 25) for med, config_val in zip(median, zip(*array))]
            IQRH = [np.percentile(config_val, 75)-med for med, config_val in zip(median, zip(*array))]

            return [median, IQRL, IQRH]

        lisa = li.LISA()
        ares = np.load('SGWB/files/S_h_muAres_nofgs.npz')
        Ares_freq = ares['x']
        Ares_stra = ares['y']

        new_dist = 0.1 | units.Mpc
        dist_factor = self.lum_dist/new_dist
        new_z = (1+0.5292)

        SMBH_IMBH_arr = [[ ] for i in range(len(np.unique(self.sim)))]
        SMBH_star_arr = [[ ] for i in range(len(np.unique(self.sim)))]
        IMBH_star_arr = [[ ] for i in range(len(np.unique(self.sim)))]
        IMBH_IMBH_arr = [[ ] for i in range(len(np.unique(self.sim)))]
        star_star_arr = [[ ] for i in range(len(np.unique(self.sim)))]
        for x_ in np.unique(self.sim):
            #Tables are in Total/Myr || Lisa/Myr || muAres/Myr
            SMBH_IMBH = np.zeros(3)
            SMBH_star = np.zeros(3)
            IMBH_star = np.zeros(3)
            IMBH_IMBH = np.zeros(3)
            star_star = np.zeros(3)

            gw_type = self.gw_typ[self.sim==x_]
            gw_freq = self.gw_frq[self.sim==x_]
            gw_strain = self.gw_str[self.sim==x_]
            for frq_run, str_run, typ_run in zip(gw_freq, gw_strain, gw_type):
                for frq_snap, str_snap, typ_snap in zip(frq_run, str_run, typ_run):
                    for frq_, str_, typ_ in zip(frq_snap, str_snap, typ_snap):
                        data_track(typ_, 0)
                        frq_ = frq_.value_in(units.Hz)*new_z
                        str_ = str_*dist_factor
                        idx = np.abs(np.asarray(Ares_freq-frq_)).argmin()
                        if frq_>min(Ares_freq):
                            if str_>(np.sqrt(frq_*lisa.Sn(frq_))):
                                data_track(typ_, 1)
                            if str_>np.sqrt(frq_*Ares_stra[idx]):
                                data_track(typ_, 2)
            SMBH_IMBH_arr[x_-1] = SMBH_IMBH
            SMBH_star_arr[x_-1] = SMBH_star
            IMBH_star_arr[x_-1] = IMBH_star
            IMBH_IMBH_arr[x_-1] = IMBH_IMBH
            star_star_arr[x_-1] = star_star
        SMBH_IMBH_arr = np.asarray(SMBH_IMBH_arr)
        SMBH_star_arr = np.asarray(SMBH_star_arr)
        IMBH_star_arr = np.asarray(IMBH_star_arr)
        IMBH_IMBH_arr = np.asarray(IMBH_IMBH_arr)
        star_star_arr = np.asarray(star_star_arr)

        SMBH_IMBH = data_stats(SMBH_IMBH_arr)
        SMBH_star = data_stats(SMBH_star_arr)
        IMBH_star = data_stats(IMBH_star_arr)
        IMBH_IMBH = data_stats(IMBH_IMBH_arr)
        star_star = data_stats(star_star_arr)

        
        with open(os.path.join("plotters/output",
                  config+"_GWStats.txt"), 'w') as f:
            f.write("GW Events\nOrganised as follows: Total Events || Lisa Observables || muAres Observables\n\n")
            f.write("SMBH-IMBH:\n")
            f.write("Median: {:}".format(str(SMBH_IMBH[0])))
            f.write("\nIQR:    [{:}, {:}]".format(str(SMBH_IMBH[1]), str(SMBH_IMBH[2])))
            f.write("\n\nSMBH-Star:\n")
            f.write("Median: {:}".format(str(SMBH_star[0])))
            f.write("\nIQR:    [{:}, {:}]".format(str(SMBH_star[1]), str(SMBH_star[2])))
            f.write("\n\nIMBH-IMBH:\n")
            f.write("Median: {:}".format(str(IMBH_IMBH[0])))
            f.write("\nIQR:    [{:}, {:}]".format(str(IMBH_IMBH[1]), str(IMBH_IMBH[2])))
            f.write("\n\nStar-Star:\n")
            f.write("Median: {:}".format(str(star_star[0])))
            f.write("\nIQR:    [{:}, {:}]".format(str(star_star[1]), str(star_star[2])))
            f.write("\n\nIMBH-Star:\n")
            f.write("Median: {:}".format(str(IMBH_star[0])))
            f.write("\nIQR:    [{:}, {:}]".format(str(IMBH_star[1]), str(IMBH_star[2])))
                        
    def gw_event_types(self):
        """Track and plot events based on their demographic"""

        sim_vals = [[ ] for i in range(np.shape(self.gw_typ[3])[0])]
        for run_ in self.gw_typ:
            idx = 0
            for data_ in run_:
                dt_vals = np.zeros(5)
                types = np.array(data_).T
                if len(types)>0:
                    prim = types[0]
                    sec = types[1]

                    smbhs = np.asarray([smbh=="smbh" for smbh in sec])
                    imbhs_p = np.asarray([imbh=="imbh" for imbh in prim])
                    imbhs_s = np.asarray([imbh=="imbh" for imbh in sec])
                    stars_p = np.asarray([imbh=="star" for imbh in prim])
                    stars_s = np.asarray([imbh=="star" for imbh in sec])

                    dt_vals[0] += np.sum(smbhs&imbhs_p)
                    dt_vals[1] += np.sum(smbhs&stars_p)
                    dt_vals[2] += np.sum(imbhs_p&imbhs_s)
                    dt_vals[3] += np.sum(stars_p&stars_s)
                    dt_vals[4] += np.sum((stars_p&imbhs_s))
                    dt_vals[4] += np.sum((stars_s&imbhs_p))
                sim_vals[idx].append(dt_vals.tolist())
                idx += 1

        med_vals = [[ ] for i in range(5)]
        IQR_high = [[ ] for i in range(5)]
        IQR_low = [[ ] for i in range(5)]
        for data_ in sim_vals:
            for k_ in range(5):
                med_vals[k_].append(np.median(np.asarray(data_).T[k_]))
                q1, q3 = np.percentile(np.asarray(data_).T[k_], [25,75])
                IQR_high[k_].append(q3)
                IQR_low[k_].append(q1)

        fig, ax = plt.subplots()
        self.clean.tickers(ax, "plot", False)
        for i_ in med_vals:
            print(len(i_))
        time = [0.03*i for i in range(len(med_vals[0]))]
        for k_ in range(2):
            time_smooth = moving_average(time, 5)
            IQRL_smooth = moving_average(IQR_low[k_], 5)
            IQRH_smooth = moving_average(IQR_high[k_], 5)
            med_smooth = moving_average(med_vals[k_], 5)

            ax.plot(time_smooth, med_smooth, label=self.labels[k_], color=self.colours[k_])
            ax.fill_between(time_smooth, IQRL_smooth, IQRH_smooth, color=self.colours[k_], alpha=0.3)
        ax.set_xlabel(r"$t$ [Myr]", fontsize=self.clean.axlabel_size)
        ax.set_ylabel(r"$\langle N_{\mathrm{event}}\rangle$", fontsize=self.clean.axlabel_size)
        ax.legend(fontsize=self.clean.axlabel_size)
        ymax = 1.04*max(med_vals[0])
        dy = np.floor(ymax)/4
        yticks = np.arange(0, ymax+dy, dy)
        ax.set_yticks(yticks)
        #ax.set_ylim(0,dy/5+ymax)
        ax.set_xlim(0,max(time))
        plt.savefig("plotters/figures/GWevent_type"+self.config_red+".pdf", dpi=300, bbox_inches='tight')
        plt.close()
    
    def KDE_plotter(self, data, data_filt):
        """
        Function to plot the frequency/strain histogram along its scatter plot.
        Use of: https://arxiv.org/pdf/2007.04241.pdf
        
        Inputs:
        data:        The data array
        data_filt:   To crop data files too large to estimate KDE
        """

        x = data[0]
        y = data[1]
        if (data_filt):
            no_data = round(len(x)**0.9)
        else:
            no_data = len(x)

        kde_freq = sm.nonparametric.KDEUnivariate(x[:no_data])
        kde_freq.fit()
        kde_freq.density = (kde_freq.density/max(kde_freq.density))

        kde_strain = sm.nonparametric.KDEUnivariate(y[:no_data])
        kde_strain.fit()
        kde_strain.density = (kde_strain.density/max(kde_strain.density))

        return [[kde_freq.support, kde_freq.density],
                [kde_strain.density, kde_strain.support]]

    def interferometer_plotter(self, ax):
        """
           Overplot interferometers on f vs. h diagram
           Observation data uses scripts from: 
            - https://github.com/eXtremeGravityInstitute/LISA_Sensitivity/tree/master
            - https://github.com/pcampeti/SGWBProbe
        """

        # LISA
        lisa = li.LISA() 
        x_temp = np.linspace(1e-5, 1, 1000)
        Sn = lisa.Sn(x_temp)

        # SKA
        SKA = np.load(os.path.join(os.getcwd(),'SGWB/files/hc_SKA.npz'))
        SKA_freq = SKA['x']
        SKA_hc = SKA['y']
        SKA_strain = SKA_hc**2/SKA_freq

        # muAres 
        Ares = np.load(os.path.join(os.getcwd(), 'SGWB/files/S_h_muAres_nofgs.npz'))
        Ares_freq = Ares['x']
        Ares_strain = Ares['y']

        ax.plot(np.log10(x_temp), np.log10(np.sqrt(x_temp*Sn)), color='slateblue')
        ax.plot(np.log10(Ares_freq), np.log10(np.sqrt(Ares_freq*Ares_strain)), linewidth='1.5', color='red')
        ax.plot(np.log10(SKA_freq), np.log10(np.sqrt(SKA_freq*SKA_strain)), linewidth='1.5', color='orangered')
        ax.text(-9.4, -16, 'SKA', fontsize=self.clean.axlabel_size, rotation=327, color='orangered')
        ax.text(-4.5, -18.2, 'LISA',fontsize=self.clean.axlabel_size, rotation=315, color='slateblue')
        ax.text(-6.6, -18.5, r'$\mu$Ares', fontsize=self.clean.axlabel_size, rotation=315, color='red')

    def gw_freq_strain(self):
        """Plot (f,h) diagram for GW events"""
        
        freq = [ ]
        strain = [ ]
        typ = [ ]
        for run_ in range(len(self.gw_typ)):
            for data_ in range(len(self.gw_typ[run_])):
                for dt_ in range(len(self.gw_frq[run_][data_])):
                    p1 = self.gw_typ[run_][data_][dt_][0]
                    p2 = self.gw_typ[run_][data_][dt_][1]
                    if p2=="smbh":
                        if p1=="imbh":
                            typ.append("SMBH-IMBH")
                        elif p1=="star":
                            typ.append("SMBH-Star")
                    else:
                        if p1=="imbh" and p1==p2:
                            typ.append("IMBH-IMBH")
                        elif p1=="star" and p1==p2:
                            typ.append("Star-Star")
                        else:
                            typ.append("Star-IMBH")
                    freq_val = np.log10(self.gw_frq[run_][data_][dt_].value_in(units.Hz))
                    strain_val = np.log10(self.gw_str[run_][data_][dt_])
                    freq.append(freq_val)
                    strain.append(strain_val)
        typ = np.array(typ)
        strain = np.array(strain)
        freq = np.array(freq)
        
        fig = plt.figure(figsize=(8, 6))
        gs = fig.add_gridspec(2, 2,  width_ratios=(4, 2), height_ratios=(2, 4),
                              left=0.1, right=0.9, bottom=0.1, top=0.9,
                              wspace=0.05, hspace=0.05)
        ax = fig.add_subplot(gs[1, 0])
        ax1 = fig.add_subplot(gs[0, 0], sharex=ax)
        ax2 = fig.add_subplot(gs[1, 1], sharey=ax)
        for ax_ in [ax, ax1, ax2]:
            self.clean.tickers(ax_, "plot", False)
        for k_ in range(len(self.labels)):
            x = freq[typ==self.labels[k_]]
            y = strain[typ==self.labels[k_]]
            ax.scatter(x, y, color=self.colours[k_], s=3, alpha=0.2)
            if len(x)>0:
                KDE_x, KDE_y = self.KDE_plotter([x,y], False)
                ax1.tick_params(axis="x", labelbottom=False)
                ax2.tick_params(axis="y", labelleft=False)

                ax1.plot(KDE_x[0], KDE_x[1], 
                         color=self.colours[k_], 
                         label=self.labels[k_])
                ax1.fill_between(KDE_x[0], KDE_x[1], 
                                 alpha=0.35, 
                                 color=self.colours[k_])
                ax2.plot(KDE_y[0], KDE_y[1], 
                         color=self.colours[k_])
                ax2.fill_between(KDE_y[0], KDE_y[1], 
                                 alpha=0.35, 
                                 color=self.colours[k_])
        self.interferometer_plotter(ax)
        
        ax.set_xlim(-12,0)
        ax.set_ylim(-30,-12)
        ax.set_xlabel(r"$\log_{10}f$ [Hz]", fontsize=self.clean.axlabel_size)
        ax.set_ylabel(r"$\log_{10}h$", fontsize=self.clean.axlabel_size)
        
        ax1.set_ylabel(r'$\rho/\rho_{\rm{max}}$', 
                       fontsize=self.clean.axlabel_size)
        ax1.legend(fontsize=self.clean.axlabel_size, 
                   bbox_to_anchor=(1.53,1.05))
        ax1.set_ylim(0,1.04)
        ax2.set_xlabel(r'$\rho/\rho_{\rm{max}}$', 
                       fontsize=self.clean.axlabel_size)
        ax2.set_xlim(0,1.04)

        plt.savefig("plotters/figures/GW_freq_strain"+self.config_red+".png", 
                    dpi=700, bbox_inches='tight')
        plt.close()

    def gw_sem_ecc(self):
        """Plot (a,(1-e)) diagram for GW events"""

        sem = [ ]
        ecc = [ ]
        typ = [ ]
        for run_ in range(len(self.gw_typ)):
            for data_ in range(len(self.gw_typ[run_])):
                for dt_ in range(len(self.gw_sem[run_][data_])):
                    p1 = self.gw_typ[run_][data_][dt_][0]
                    p2 = self.gw_typ[run_][data_][dt_][1]
                    if p2=="smbh":
                        if p1=="imbh":
                            typ.append("SMBH-IMBH")
                        else:
                            typ.append("SMBH-Star")
                    else:
                        if p1=="imbh" and p1==p2:
                            typ.append("IMBH-IMBH")
                        elif p1=="star" and p1==p2:
                            typ.append("Star-Star")
                        else:
                            typ.append("Star-IMBH")
                    sem_val = np.log10(self.gw_sem[run_][data_][dt_].value_in(units.au))
                    ecc_val = np.log10(1-self.gw_ecc[run_][data_][dt_])
                    sem.append(sem_val)
                    ecc.append(ecc_val)
        typ = np.array(typ)
        ecc = np.array(ecc)
        sem = np.array(sem)
        
        fig = plt.figure(figsize=(10, 5))
        gs = fig.add_gridspec(2, 4,  width_ratios=(2, 2, 2, 2), 
                            height_ratios=(2, 3), left=0.1, 
                            right=0.9, bottom=0.1, top=0.9, 
                            wspace=0.5, hspace=0.15)
        axL = fig.add_subplot(gs[1, 0:2])
        axL1 = fig.add_subplot(gs[0, 0:2], sharex=axL)
        axR = fig.add_subplot(gs[1, 2:])
        axR1 = fig.add_subplot(gs[0, 2:], sharex=axR)
        axL.set_xlabel(r'$\log_{10}(1-e)_{\rm{SMBH}}$', fontsize=self.clean.axlabel_size)
        axL.set_ylabel(r'$\log_{10}f_{e<}$', fontsize=self.clean.axlabel_size)
        axL1.set_ylabel(r'$\rho/\rho_{\rm{max}}$', fontsize=self.clean.axlabel_size)
        axR.set_xlabel(r'$\log_{10}a_{\rm{SMBH}}$ [au]', fontsize=self.clean.axlabel_size)
        axR.set_ylabel(r'$\log_{10}f_{a<}$', fontsize=self.clean.axlabel_size)
        for k_ in range(len(self.labels)):
            ecc_df = ecc[typ==self.labels[k_]]
            sem_df = sem[typ==self.labels[k_]]
            if len(sem_df)>0:
                axL, axL1 = self.clean.cdf_pdf(ecc_df, axL, axL1, self.colours[k_], 
                                               self.labels[k_], "Final", True, "Top")
                axR, axR1 = self.clean.cdf_pdf(sem_df, axR, axR1, self.colours[k_], 
                                               self.labels[k_], "Final", False, "Top")
        plt.savefig("plotters/figures/gw_sem_ecc_CDF"+self.config_red+".pdf", 
                    dpi=300, bbox_inches='tight')
        plt.close()

        minx = 100
        maxx = 0
        miny = 100
        maxy = 0
        
        fig = plt.figure(figsize=(8, 6))
        gs = fig.add_gridspec(2, 2,  width_ratios=(4, 2), height_ratios=(2, 4),
                                left=0.1, right=0.9, bottom=0.1, top=0.9,
                                wspace=0.05, hspace=0.05)
        ax = fig.add_subplot(gs[1, 0])
        ax1 = fig.add_subplot(gs[0, 0], sharex=ax)
        ax2 = fig.add_subplot(gs[1, 1], sharey=ax)
        ax1.tick_params(axis="x", labelbottom=False)
        ax2.tick_params(axis="y", labelleft=False)
        ax.set_ylabel(r'$\log_{10}(1-e)$', fontsize=self.clean.axlabel_size)
        ax.set_xlabel(r'$\log_{10} a$ [au]', fontsize=self.clean.axlabel_size)
        ax1.set_ylabel(r'$\rho/\rho_{\rm{max}}$', fontsize=self.clean.axlabel_size)
        ax2.set_xlabel(r'$\rho/\rho_{\rm{max}}$', fontsize=self.clean.axlabel_size)
        for k_ in range(len(self.labels)):
            ecc_df = ecc[typ==self.labels[k_]]
            sem_df = sem[typ==self.labels[k_]]
            if len(sem_df)>0:
                minx = min(minx, min(sem_df))
                maxx = max(maxx, max(sem_df))
                miny = min(miny, min(ecc_df))
                maxy = max(maxy, max(ecc_df))

                ax.scatter(None, None, color=self.colours[k_], label=self.labels[k_])
                ax.scatter(sem_df, ecc_df, color=self.colours[k_], alpha=0.2, s=0.75)

                kde_ecc = sm.nonparametric.KDEUnivariate(ecc_df)
                kde_ecc.fit()
                kde_ecc.density = (kde_ecc.density/max(kde_ecc.density))
                ax2.plot(kde_ecc.density, kde_ecc.support, color=self.colours[k_])
                ax2.fill_between(kde_ecc.density, kde_ecc.support, 
                                 alpha=0.35, color=self.colours[k_])
            
                kde_sem = sm.nonparametric.KDEUnivariate(sem_df)
                kde_sem.fit()
                kde_sem.density = (kde_sem.density/max(kde_sem.density))
                ax1.plot(kde_sem.support, kde_sem.density, 
                         color=self.colours[k_], label=self.labels[k_])
                ax1.fill_between(kde_sem.support, kde_sem.density, 
                                 alpha=0.35, color=self.colours[k_])
        ax.set_xlim(0.95*minx, 1.05*maxx)
        ax.set_ylim(1.02*miny, 0)
        ax1.set_ylim(0.001, 1.04)
        ax2.set_xlim(0.001, 1.04)
        ax.legend(fontsize=self.clean.axlabel_size, bbox_to_anchor=(1.52,1.56))
        for ax_ in [ax, ax1, ax2]:
            self.clean.tickers(ax_, "plot", False)
        plt.savefig("plotters/figures/gw_sem_ecc_scatter_"+self.config_red+".png", 
                    dpi=300, bbox_inches='tight')
        plt.close()
 
swag = CalcGW(dvals="all")
#swag.process_data()
swag.read_data(swag.config_path[0][swag.path_len:])
#swag.gw_stats(swag.config_path[0][swag.path_len:])
#swag.gw_sem_ecc()
#swag.gw_freq_strain()
#swag.gw_event_types()