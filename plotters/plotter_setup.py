import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.ticker import StrMethodFormatter
import numpy as np
import statsmodels.api as sm


class PlotterSetup(object):
    def __init__(self):
        plt.rcParams["font.family"] = "Times New Roman"
        plt.rcParams["mathtext.fontset"] = "cm"
        self.axlabel_size = 14
        self.tick_size = 14
        
        left = .45
        width = .5
        bottom = .58
        height = .5
        self.right = left + width
        self.top = bottom + height

    def tickers(self, ax, ptype, sig_fig):
        """Function to setup axis
        
           Inputs:
           ax:       Axis cleaning up
           ptype:    Plot type (hist || plot)
           sig_fig:  Number of sig. figs. on axis ticks
        """

        ax.yaxis.set_ticks_position('both')
        ax.xaxis.set_ticks_position('both')
        ax.xaxis.set_minor_locator(mtick.AutoMinorLocator())
        ax.yaxis.set_minor_locator(mtick.AutoMinorLocator())

        if (sig_fig):
            formatter = StrMethodFormatter("{x:.1f}")
            ax.xaxis.set_major_formatter(formatter)
            ax.yaxis.set_major_formatter(formatter)

        if ptype == "hist":
            ax.tick_params(axis="y", labelsize=self.tick_size)
            ax.tick_params(axis="x", labelsize=self.tick_size)
            return ax
        else:
            ax.tick_params(axis="y", which='both', direction="in", labelsize=self.tick_size)
            ax.tick_params(axis="x", which='both', direction="in", labelsize=self.tick_size)
            return ax

    def cdf_pdf(self, data_arr, ax_bot, ax_top, dcolour, 
                dlabels, in_fin, labelb, label_plac):
        """
        Function to plot the CDF and KDE of orbital properties.

        Inputs:
        data_arr:  The flattened data set
        ax_bot:    The CDF plot
        ax_top:    The KDE plot
        dcolours:  The colour scheme
        dlabels:   The labels for the plot
        in_fin:    Distinguish (i)nitial or (f)inal data
        labelb:    Plotting with labels (1 = True || 0 = False)
        lab_plot:  Which plot hosts the label
        """

        plot_ini = PlotterSetup()
        data_sort = np.sort(data_arr)
        data_idx = np.asarray([i for i in enumerate(data_sort)])

        kde = sm.nonparametric.KDEUnivariate(data_sort)
        kde.fit()
        kde.density /= max(kde.density)
        if (labelb):
            if label_plac.lower()=="top":
                if in_fin.lower()=="initial":
                    ax_bot.plot(data_sort, np.log10(data_idx[:,0]/data_idx[-1,0]), 
                                color=dcolour, linestyle=":")
                else:
                    ax_bot.plot(data_sort, np.log10(data_idx[:,0]/data_idx[-1,0]), 
                                color=dcolour)
                ax_top.plot(kde.support, kde.density, color=dcolour, label=dlabels)
                ax_top.legend(loc='upper left')
                ax_top.legend(prop={'size': self.axlabel_size})
                
            else:
                if in_fin.lower()=="initial":
                    ax_bot.plot(data_sort, np.log10(data_idx[:,0]/data_idx[-1,0]), 
                                color=dcolour, linestyle=":", label=dlabels)
                else:
                    ax_bot.plot(data_sort, np.log10(data_idx[:,0]/data_idx[-1,0]), 
                                color=dcolour, label=dlabels)
                ax_top.plot(kde.support, kde.density, color=dcolour)
                ax_bot.legend(loc='lower right')
                ax_bot.legend(prop={'size': self.axlabel_size})
        else:
            ax_bot.plot(data_sort, np.log10(data_idx[:,0]/data_idx[-1,0]), 
                        color=dcolour, label=dlabels)
            ax_top.plot(kde.support, kde.density, color=dcolour)

        ax_top.fill_between(kde.support, kde.density, alpha=0.35, color=dcolour)

        for ax_ in [ax_bot, ax_top]:
            plot_ini.tickers(ax_, 'plot', False)
            ax_.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))
        ax_top.set_ylim(1e-2, 1.1)

        return ax_bot, ax_top