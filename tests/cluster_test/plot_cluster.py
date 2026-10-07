from amuse.lab import read_set_from_file, units, Particles, constants
from amuse.ext.orbital_elements import orbital_elements

from natsort import natsorted
import glob
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.colors as colors
import matplotlib.patheffects as pe
import numpy as np
import os
from scipy import stats


# Set some globals
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "cm"

SAVE_DIR = "tests/cluster_test"
COLOURS = ["black", "dodgerblue"]
LABELS = ["Ph4", "Nemesis"]
LINESTYLE = ["-", "--"]
MARKER_SIZE = [50, 20, 40, 10]
LW = [8, 3]
AXLABEL_SIZE = TICK_SIZE = 22

DATA_DIR = "tests/cluster_test/data"


def tickers(ax) -> None:
    """
    Function to setup axis
    Args:
        ax (axis):  Axis needing cleaning up.
    """
    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('both')
    ax.xaxis.set_minor_locator(mtick.AutoMinorLocator())
    ax.yaxis.set_minor_locator(mtick.AutoMinorLocator())
    ax.tick_params(
        axis="y",
        which='both',
        direction="in",
        labelsize=TICK_SIZE
    )
    ax.tick_params(
        axis="x",
        which='both',
        direction="in",
        labelsize=TICK_SIZE
    )


def compare_visually(dir_data, nem_data) -> None:
    """
    Compare the direct and Nemesis data visually.
    Args:
        dir_data (list):  List of direct integration data files.
        nem_data (list):  List of Nemesis integration data files.
    """
    out_dir = f"{SAVE_DIR}/visual_comparison/"
    os.makedirs(out_dir, exist_ok=True)
    colours_pastel = ["black", "dodgerblue", "red", "blue"]
    for step, (direct, nem_) in enumerate(zip(dir_data, nem_data)):
        dir_particles = read_set_from_file(direct)
        nem_particles = read_set_from_file(nem_)
        for particles in [dir_particles, nem_particles]:
            particles.move_to_center()

        # Extract stars
        star_nem = nem_particles[nem_particles.mass >= 0.08 | units.MSun]
        star_dir = dir_particles[dir_particles.mass >= 0.08 | units.MSun]

        dr = 0 | units.au
        for s in star_nem:
            target = star_dir[star_dir.original_key == s.original_key]
            dr = max(dr, (s.position - target.position).lengths())
        print(f"    Maximum star drij = {dr.max().in_(units.au)}")

        # Extract asteroids
        dir_ast = dir_particles[dir_particles.mass == (0. | units.kg)]
        nem_ast = nem_particles[nem_particles.mass == (0. | units.kg)]
        dir_particles -= dir_ast
        nem_particles -= nem_ast

        fig, ax = plt.subplots(figsize=(7, 6))
        for i, major in enumerate([dir_particles, nem_particles]):
            ax.scatter(
                major.x.value_in(units.au),
                major.y.value_in(units.au),
                color=COLOURS[i],
                label=LABELS[i],
                s=MARKER_SIZE[i],
                zorder=2
            )

        for i, minor in enumerate([dir_ast, nem_ast]):
            ax.scatter(
                minor.x.value_in(units.au),
                minor.y.value_in(units.au),
                color=colours_pastel[i+2],
                s=5, alpha=0.25,
                zorder=3
            )

        tickers(ax)
        ax.set_xlabel(r"$x$ [au]", fontsize=AXLABEL_SIZE)
        ax.set_ylabel(r"$y$ [au]", fontsize=AXLABEL_SIZE)
        plt.savefig(f"{out_dir}/{step}.pdf", dpi=300, bbox_inches='tight')
        plt.clf()
        plt.close()


def process_data(dir_data, nem_data):
    """
    Save direct N-body and Nemesis integration eccentricity
    and semi-major axis data of asteroids.
    """
    def _save_kepler(binary, sma_df, ecc_df, rtide):
        kepler_elements = orbital_elements(binary, G=constants.G)
        sma, ecc = kepler_elements[2], kepler_elements[3]
        if kepler_elements[3] <= 1. and sma * (1 + ecc) <= rtide:
            sma_df.append(sma.value_in(units.au))
            ecc_df.append(ecc)

    initial_nem = read_set_from_file(nem_data[0])
    host_dic = {}
    for system_id in np.unique(initial_nem.syst_id):
        system = initial_nem[initial_nem.syst_id == system_id]
        if system_id > 0:
            host = system[system.mass.argmax()]
            asteroids = system[system.mass == 0. | units.kg]
            host_dic[host.original_key] = asteroids.original_key

    dt_dir = read_set_from_file(dir_data[-1])
    dt_nem = read_set_from_file(nem_data[-1])
    dt_dir.move_to_center()
    dt_nem.move_to_center()

    nem_stars = dt_nem[dt_nem.mass > 0.08 | units.MSun]
    ast_dir = dt_dir[dt_dir.mass == 0. | units.kg]

    temp_dir_df = [[] for _ in range(2)]  # SMA, Ecc
    temp_nem_df = [[] for _ in range(2)]
    for host_key, system_key in host_dic.items():
        host_dir = dt_dir[dt_dir.original_key == host_key]
        host_nem = dt_nem[dt_nem.original_key == host_key]

        for a_key in system_key:
            ext = nem_stars - host_nem
            drij = (ext.position - host_nem.position).lengths()
            rtide = drij.min() * (host.mass / ext[drij.argmin()].mass)**(1/3)

            # First deal with Direct data
            target = ast_dir[ast_dir.original_key == a_key]
            binary = Particles(particles=[host_dir, target])
            _save_kepler(binary, temp_dir_df[0], temp_dir_df[1], rtide)

            # Then deal with Nemesis data
            target = dt_nem[dt_nem.original_key == a_key]
            binary = Particles(particles=[host_nem, target])
            _save_kepler(binary, temp_nem_df[0], temp_nem_df[1], rtide)
    print(f"#Sample Direct={len(temp_dir_df[0])}", end=", ")
    print(f"#Sample Nemesis={len(temp_nem_df[0])}")
    return temp_dir_df, temp_nem_df


def plot_cluster_cdf(dir_data, nem_data):
    fin_dir = read_set_from_file(dir_data[-1], "hdf5")
    fin_nem = read_set_from_file(nem_data[-1], "hdf5")
    for particles in [fin_dir, fin_nem]:
        particles.move_to_center()

    dir_stars = fin_dir[fin_dir.mass > 0.08 | units.MSun]
    nem_stars = fin_nem[fin_nem.mass > 0.08 | units.MSun]
    dir_ast = fin_dir[fin_dir.mass == 0. | units.kg]
    nem_ast = fin_nem[fin_nem.mass == 0. | units.kg]

    velocity_df = [[], []]
    fig, ax = plt.subplots(figsize=(7, 6))
    for i, data in enumerate([dir_stars, nem_stars, dir_ast, nem_ast]):
        velocities = data.velocity.lengths().value_in(units.kms)
        sorted_vel = np.sort(velocities)
        sample = np.arange(1, len(sorted_vel)+1) / len(sorted_vel)
        ax.plot(
            sorted_vel, sample,
            color=COLOURS[i % 2],
            ls=LINESTYLE[i // 2],
            lw=LW[i % 2]
            )
        if i < 2:  # Store velocity data of massive bodies
            velocity_df[i] = sorted_vel
            ax.scatter(None, None, color=COLOURS[i], label=LABELS[i], s=50)
    tickers(ax)
    ax.set_xscale("log")
    ax.set_xlabel(r"$v$ [km/s]", fontsize=AXLABEL_SIZE)
    ax.set_ylabel(r"$f_{<}$", fontsize=AXLABEL_SIZE)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=AXLABEL_SIZE, frameon=False)
    plt.savefig(f"{SAVE_DIR}/cdf_final_vel.pdf", dpi=300, bbox_inches='tight')
    plt.close()

    cramer = stats.cramervonmises_2samp(velocity_df[0], velocity_df[1])
    print(f"    Test for stellar velocity: {cramer}")

    dr = [[], []]
    fig, ax = plt.subplots(figsize=(7, 6))
    for i, data in enumerate([dir_stars, nem_stars, dir_ast, nem_ast]):
        positions = data.position.lengths().value_in(units.au)
        sorted_pos = np.sort(positions)
        sample = np.arange(1, len(sorted_pos)+1) / len(sorted_pos)
        ax.plot(
            sorted_pos, sample,
            color=COLOURS[i % 2],
            ls=LINESTYLE[i // 2],
            lw=LW[i % 2]
            )
        if i < 2:
            ax.scatter(None, None, color=COLOURS[i], label=LABELS[i], s=50)
            dr[i] = sorted_pos
    tickers(ax)
    ax.set_xscale("log")
    ax.set_xlabel(r"$r$ [pc]", fontsize=AXLABEL_SIZE)
    ax.set_ylabel(r"$f_{<}$", fontsize=AXLABEL_SIZE)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=AXLABEL_SIZE, frameon=False)
    plt.savefig(f"{SAVE_DIR}/cdf_final_pos.pdf", dpi=300, bbox_inches='tight')
    plt.close()

    cramer = stats.cramervonmises_2samp(dr[0], dr[1])
    print(f"   Test for star positions: {cramer}")


def plot_ast_cdf(dir_sma_ast, dir_ecc_ast, nem_sma_ast, nem_ecc_ast):
    """
    Plot CDFs of asteroid semi-major axis and eccentricity from
    direct N-body and Nemesis simulations.
    Args:
        dir_sma_ast (list):  List of semi-major axis from direct N-body.
        dir_ecc_ast (list):  List of eccentricities from direct N-body.
        nem_sma_ast (list):  List of semi-major axis from Nemesis.
        nem_ecc_ast (list):  List of eccentricities from Nemesis.
    """
    data_dic = {
        "sma": [[dir_sma_ast, nem_sma_ast], r"$a$ [au]", [2, 1e5]],
        "ecc": [[dir_ecc_ast, nem_ecc_ast], r"$e$", [1e-3, 1]]
    }
    res_df = [[] for _ in range(2)]  # SMA, Ecc
    for i, (fname, (df, xlabel, xlims)) in enumerate(data_dic.items()):
        fig, axs = plt.subplots(
            2, 1,
            figsize=(6, 5),
            sharex=True,
            gridspec_kw={
                "height_ratios": [1, 3],
                "wspace": 0.0,
                "hspace": 0.05
            }
        )

        axs[0].axhline(0, color="black", lw=1, zorder=0)
        if i == 0:
            axs[0].axvline(2*100*2**(1/3), color="red", lw=1)
            axs[0].axvline(100*2**(1/3), color="black", lw=1)
            axs[0].axvline(100*0.5**(1/3), color="black", lw=1)

        for j, data in enumerate(df):
            sorted_data = np.sort(data)
            sample_data = np.arange(1, len(sorted_data)+1) / len(sorted_data)
            axs[1].plot(sorted_data, sample_data, lw=3, color=COLOURS[j])
            axs[1].scatter([], [], color=COLOURS[j], label=LABELS[j], s=50)

        sorted_nem = np.sort(df[1])
        sample_nem = np.arange(1, len(sorted_nem)+1) / len(sorted_nem)
        sorted_dir = np.sort(df[0])
        sample_dir = np.arange(1, len(sorted_dir)+1) / len(sorted_dir)
        xmin = max(sorted_dir.min(), sorted_nem.min())
        xmax = min(sorted_dir.max(), sorted_nem.max())

        x_common = np.linspace(xmin, xmax, 1000)
        dir_i = np.interp(x_common, sorted_dir, sample_dir)
        nem_i = np.interp(x_common, sorted_nem, sample_nem)
        residuals = nem_i - dir_i

        res_df[0].append(x_common)
        res_df[1].append(residuals)

        axs[0].plot(x_common, residuals, color="gray", lw=3)
        axs[0].set_ylabel(r"$\Delta y$", fontsize=AXLABEL_SIZE)

        axs[1].set_xlabel(xlabel, fontsize=AXLABEL_SIZE)
        axs[1].set_ylabel(r"$f_{<}$", fontsize=AXLABEL_SIZE)
        axs[1].legend(fontsize=AXLABEL_SIZE, frameon=False)
        axs[1].set_ylim(0, 1.)
        axs[1].set_xlim(xlims)
        for ax in axs:
            tickers(ax)
        if i == 0:
            for ax in axs:
                ax.set_xscale("log")
            axs[0].set_xlim(xmin, 2. * xmax)
        res_lims = axs[0].get_ylim()
        max_y = max(abs(res_lims[0]), abs(res_lims[1]))
        axs[0].set_ylim(-max_y, max_y)
        plt.savefig(
            f"{SAVE_DIR}/cdf_{fname}.pdf",
            dpi=300, bbox_inches='tight'
            )
        plt.clf()
        plt.close()

        dir_sorted = np.asarray(np.sort(df[0]))
        nem_sorted = np.asarray(np.sort(df[1]))

        anderson = stats.anderson_ksamp(
            [dir_sorted, nem_sorted],
            method=stats.PermutationMethod()
            )
        cramer = stats.cramervonmises_2samp(dir_sorted, nem_sorted)
        ks = stats.ks_2samp(dir_sorted, nem_sorted)
        print(f"Anderson-Darling test for {fname}: {anderson}")
        print(f"Cramer-von Mises test for {fname}: {cramer}")
        print(f"KS test for {fname}: {ks}")

        D_obs = stats.ks_2samp(
            dir_sorted, nem_sorted,
            alternative='two-sided',
            mode='asymp'
            ).statistic
        n1, n2 = len(dir_sorted), len(nem_sorted)

        alpha = 0.05
        c_alpha = np.sqrt(-0.5 * np.log(alpha / 2))
        D_alpha = c_alpha * np.sqrt((n1 + n2) / (n1 * n2))
        print(f"D_obs = {D_obs}, D_alpha (alpha={alpha})= {D_alpha}")


def plot_ast_residual(dir_data, nem_data):
    """
    Save direct N-body and Nemesis integration eccentricity and
    semi-major axis data of asteroids.
    Args:
        dir_data (list):  List of data files associated to
                          direct integration runs.
        nem_data (list):  List of data files associated to Nemesis runs.
    """
    initial_nem = read_set_from_file(nem_data[0])
    host_dic = {}
    for system_id in np.unique(initial_nem.syst_id):
        system = initial_nem[initial_nem.syst_id == system_id]
        if system_id > 0:
            host = system[system.mass.argmax()]
            asteroids = system[system.mass == 0. | units.kg]
            host_dic[host.original_key] = asteroids.original_key

    dt_dir = read_set_from_file(dir_data[-1])
    dt_nem = read_set_from_file(nem_data[-1])
    ast_dir = dt_dir[dt_dir.mass == 0. | units.kg]

    data = []
    for i, (host_key, system_key) in enumerate(host_dic.items()):
        host_dir = dt_dir[dt_dir.original_key == host_key]
        host_nem = dt_nem[dt_nem.original_key == host_key]

        for a_key in system_key:
            # First deal with Direct data
            target = ast_dir[ast_dir.original_key == a_key]
            binary = Particles(particles=[host_dir, target])
            kepler_elements = orbital_elements(binary, G=constants.G)
            sma, ecc = kepler_elements[2], kepler_elements[3]
            if kepler_elements[3] <= 1.:
                sma_value_dir = sma.value_in(units.au)
                ecc_value_dir = ecc

            # Then deal with Nemesis data
            target = dt_nem[dt_nem.original_key == a_key]
            binary = Particles(particles=[host_nem, target])
            kepler_elements = orbital_elements(binary, G=constants.G)
            sma, ecc = kepler_elements[2], kepler_elements[3]
            if kepler_elements[3] <= 1.:
                sma_value_nem = sma.value_in(units.au)
                ecc_value_nem = ecc

            if ecc_value_nem <= 1. and ecc_value_dir <= 1.:
                data.append((
                    sma_value_dir, ecc_value_dir,
                    sma_value_nem, ecc_value_nem
                ))

    arr = np.asarray(data)
    a_dir, e_dir, a_nem, e_nem = arr.T

    sma_nem_log = np.log10(a_nem)
    sma_dir_log = np.log10(a_dir)
    xbins = np.linspace(np.log10(4), np.log10(1000), 15)
    ybins = np.linspace(0, 1.0, 15)

    decc = e_nem - e_dir
    N, _, _ = np.histogram2d(sma_nem_log, e_nem, bins=[xbins, ybins])

    Z, xedges, yedges, _ = stats.binned_statistic_2d(
        sma_dir_log, e_dir, decc,
        statistic='median',
        bins=[xbins, ybins]
    )
    Z = np.where(N >= 1, Z, np.nan)

    fig, ax = plt.subplots(figsize=(7, 6))
    tickers(ax)
    pcm = ax.pcolormesh(
        10**xedges, yedges, Z.T,
        cmap='coolwarm_r',
        norm=colors.CenteredNorm(0)
        )

    # Overplot number counts
    x_centers = 0.5 * (xedges[:-1] + xedges[1:])
    y_centers = 0.5 * (yedges[:-1] + yedges[1:])
    for ix, xc in enumerate(x_centers):
        for iy, yc in enumerate(y_centers):
            count = int(N[ix, iy])
            if count == 0 or count >= 10:
                continue
            ax.text(
                10**xc, yc, f"{count}",
                ha="center", va="center",
                color="white",
                fontsize=AXLABEL_SIZE,
                path_effects=[pe.withStroke(linewidth=1.5, foreground="black")]
            )

    cbar = plt.colorbar(pcm, ax=ax)
    cbar.set_label(label=r"$\Delta e$", fontsize=AXLABEL_SIZE)
    cbar.ax.tick_params(labelsize=AXLABEL_SIZE)
    ax.set_xlabel(r"$a [{\rm au}]$", fontsize=AXLABEL_SIZE)
    ax.set_ylabel(r"$e$", fontsize=AXLABEL_SIZE)
    ax.set_ylim(0, 1)
    ax.set_xscale("log")
    ax.set_xlim(10**xbins[0], 10**xbins[-1])

    y = np.linspace(0, 1, 100)
    Rlink_max = [2*100*2**(1/3) / (1 + i) for i in y]
    Rpar_max = [100*2**(1/3) / (1 + i) for i in y]
    Rpar_min = [100*0.5**(1/3) / (1 + i) for i in y]
    ax.plot(Rlink_max, y, color="red")
    ax.plot(Rpar_max, y, color="black")
    ax.plot(Rpar_min, y, color="black")
    plt.savefig(f"{SAVE_DIR}/residual_ecc.pdf", dpi=300, bbox_inches='tight')
    plt.clf()
    plt.close()


def plot_energy(dir_df, nem_df):
    """Plot the energy evolution of the system"""
    dE_array = [[] for _ in range(2)]  # Direct, Nemesis
    for i, data_files in enumerate([dir_df, nem_df]):
        for j, snapshot in enumerate(data_files):
            p = read_set_from_file(snapshot)
            massives = p[p.mass > 0. | units.kg]
            if j == 0:
                E0 = massives.kinetic_energy() + massives.potential_energy()
            else:
                Et = massives.kinetic_energy() + massives.potential_energy()
                dE = abs((Et - E0)/E0)
                dE_array[i].append(dE)

    fig, ax = plt.subplots(figsize=(7, 6))
    for i, df in enumerate(dE_array):
        time = np.linspace(0, 0.1, len(df))
        ax.plot(time, df, color=COLOURS[i], lw=3-i)

    ax.scatter(None, None, color=COLOURS[0], label=LABELS[0], s=50)
    ax.scatter(None, None, color=COLOURS[1], label=LABELS[1], s=50)
    ax.set_xlabel(r"$t$ [Myr]", fontsize=AXLABEL_SIZE)
    ax.set_ylabel(r"$\Delta E / E_0$", fontsize=AXLABEL_SIZE)
    tickers(ax)
    ax.set_yscale("log")
    ax.set_xlim(0, 0.05)
    ax.legend(fontsize=AXLABEL_SIZE, frameon=False)
    plt.savefig(f"{SAVE_DIR}/energy.pdf", dpi=300, bbox_inches='tight')
    plt.clf()


def validate_runs(dir_data, nem_data):
    """
    Validate that both tested runs have identical initial conditions
    and same particle counts for each snapshot.
    Args:
        dir_data (list):  List of direct integration data files.
        nem_data (list):  List of Nemesis integration data files.
    """
    direct_initial = read_set_from_file(dir_data[0])
    Nemesis_initial = read_set_from_file(nem_data[0])
    dr = (direct_initial.position - Nemesis_initial.position).lengths()
    dv = (direct_initial.velocity - Nemesis_initial.velocity).lengths()
    dm = direct_initial.mass - Nemesis_initial.mass

    # Ensure initial conditions are the same for both simulations
    if dr.max() > (0. | units.m):
        raise AssertionError(
            f"Error: Particle not initialised the same."
            f" Maximum dr = {dr.max().in_(units.m)}"
            )
    if dv.max() > (0. | units.kms):
        raise AssertionError(
            f"Error: Particle not initialised the same. "
            f" Maximum dv = {dv.max().in_(units.kms)}"
            )
    if dm.max() > (0. | units.MSun):
        raise AssertionError(
            f"Error: Particle not initialised the same. "
            f" Maximum dm = {dm.max().in_(units.MSun)}"
        )
    print("Simulation has same IC confirmed.")

    # Ensure particle counts match for each snapshot
    for i, (p, q) in enumerate(zip(dir_data, nem_data)):
        px = read_set_from_file(p)
        qx = read_set_from_file(q)
        if len(px) != len(qx):
            raise AssertionError(
                "!!! Particle count mismatch !!!\n"
                f"    Pair #{i}\n"
                f"    Direct snapshot: {p} ----> {len(px)}"
                f"    Nemesis snapshot: {q} ---> {len(qx)}"
                )
        assert len(px) == len(qx)
    print("Simulation has same particle counts confirmed.")


path = "{}/{}/simulation_snapshot/snap_*.hdf5"
dir_data = natsorted(glob.glob(path.format(DATA_DIR, "cluster_run_direct")))
nem_data = natsorted(glob.glob(path.format(DATA_DIR, "cluster_run_nemesis")))

Nsnaps = min(len(dir_data), len(nem_data))
dir_data = dir_data[:Nsnaps]
nem_data = nem_data[:Nsnaps]

print("...Validating runs...")
validate_runs(dir_data, nem_data)

print("...Plotting dE...")
plot_energy(dir_data, nem_data)

print("...Comparing visually...")
#compare_visually(dir_data, nem_data)

print("...Cluster Overall CDF Plots...")
plot_cluster_cdf(dir_data, nem_data)

print("...Processing Asteroids...")
direct_df, nemesis_df = process_data(dir_data, nem_data)

print("...Asteroids CDF Plots...")
plot_ast_cdf(
    direct_df[0],
    direct_df[1],
    nemesis_df[0],
    nemesis_df[1]
)

print("...Asteroids Residuals Plot...")
plot_ast_residual(dir_data, nem_data)
