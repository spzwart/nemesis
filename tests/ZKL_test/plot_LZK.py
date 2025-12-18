import glob
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import natsort

from amuse.ext.orbital_elements import orbital_elements
from amuse.lab import constants, units, Particles, read_set_from_file



# Plot parameters
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "cm"
axlabel_size = 14
tick_size = 14


def tickers(ax) -> plt.axis:
    """
    Function to setup axis
    Args:
        ax (axis):  Axis needing cleaning up
    Returns:
        ax (axis):  The cleaned axis
    """
    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('both')
    ax.xaxis.set_minor_locator(mtick.AutoMinorLocator())
    ax.yaxis.set_minor_locator(mtick.AutoMinorLocator())
    ax.tick_params(
        axis="y", which='both', 
        direction="in", 
        labelsize=tick_size
    )
    ax.tick_params(
        axis="x", which='both', 
        direction="in", 
        labelsize=tick_size
    )
    return ax

def plot_LZK():
    nem_1e0au = natsort.natsorted(glob.glob(f"tests/ZKL_test/data/ZKL_Rpar0au/simulation_snapshot/*"))[::10]
    nem_1e3au = natsort.natsorted(glob.glob(f"tests/ZKL_test/data/ZKL_Rpar1000au/simulation_snapshot/*"))[::10]
    nem_1e2au = natsort.natsorted(glob.glob(f"tests/ZKL_test/data/ZKL_Rpar100au/simulation_snapshot/*"))[::10]

    inc_df = [[ ], [ ], [ ]]
    ecc_df = [[ ], [ ], [ ]]
    for i, data in enumerate([nem_1e0au, nem_1e3au, nem_1e2au]):
        print(f"Analysing {len(data)} snapshots for data #{i}")
        for j, snapshot in enumerate(data):
            if len(snapshot) == 0:
                continue
            bodies = read_set_from_file(snapshot, format='hdf5')
            if j == 0:
                Etot = bodies.potential_energy() + bodies.kinetic_energy()
            elif j%500==0:
                E = bodies.potential_energy() + bodies.kinetic_energy()
                dE = (E - Etot)/Etot
                print(f"\nEnergy error: {dE:.2e}")

            print(f"\rProgress: {100*j/len(data):.2f}%", end=" ")
            sun = bodies[bodies.mass.argmax()]
            inner_Jup = bodies[bodies.mass.argmin()]
            outer_Jup = bodies - inner_Jup - sun

            inner_bin = inner_Jup + sun
            kepler_inner = orbital_elements(inner_bin, G=constants.G)
            inner_ecc = kepler_inner[3]
            inner_inc = kepler_inner[5].value_in(units.deg)
            
            ibin = Particles(1)
            ibin.mass = inner_bin.mass.sum()
            ibin.position = inner_bin.center_of_mass()
            ibin.velocity = inner_bin.center_of_mass_velocity()
            
            new_set = Particles()
            new_set.add_particle(outer_Jup)
            new_set.add_particles(ibin)
            kepler_outer = orbital_elements(new_set, G=constants.G)
            outer_inc = kepler_outer[5].value_in(units.deg)
            inc_tot = (inner_inc + outer_inc)
            outer_ecc = kepler_outer[3]
            
            ecc_df[i].append(1-inner_ecc)
            inc_df[i].append(inc_tot)   
    
    color = ["black", "dodgerblue", "white"]
    label = ["0 au", "1000 au", "100 au"]
    ls = ["-", "-", "-"]
    lw = [7, 3, 1]
    
    data_dic = {
        "ecc": [ecc_df, r"$1 - e_{\rm in}$"],
        "inc": [inc_df, r"$i_{\rm tot}$ [$\degree$]"]
    }
    time = [5000/1e6 * j for j in range(len(nem_1e0au))]  # In Myr
    for fig_name, data in data_dic.items():
        df = data[0]
        ylabel = data[1]

        fig, ax = plt.subplots(figsize=(6, 5))
        tickers(ax)
        ax.set_ylabel(ylabel, fontsize=axlabel_size)
        ax.set_xlabel(r"$t$ [Myr]", fontsize=axlabel_size)
        for i in range(3):
            ax.plot(time[:len(df[i])], df[i],
                    color=color[i],
                    zorder=i,
                    lw=lw[i],
                    ls=ls[i])
            ax.scatter(
                [], [], 
                label=label[i], 
                color=color[i], 
                edgecolor="black"
                )

        if fig_name == "ecc":
            # Create inset plot
            axins = inset_axes(ax, width="35%", height="35%", loc="lower right", borderpad=2)
            for i in range(3):
                axins.plot(
                    time[:len(ecc_df[i])], ecc_df[i], 
                    color=color[i], 
                    lw=lw[i], 
                    ls=ls[i], 
                    zorder=i
                    )
            axins.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
            axins.set_xlim(0., 0.12)
            axins.set_ylim(0.9965, 0.9995)
            ax.set_yscale("log")
            
        ax.set_xlim(time[0], time[-1])
        ax.legend(fontsize=axlabel_size, frameon=False)
        plt.savefig(f"tests/ZKL_test/ZKL_plot_{fig_name}_alt.pdf", bbox_inches="tight")
        plt.clf()
        plt.close(fig)


plot_LZK() 