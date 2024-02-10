import glob
import matplotlib.pyplot as plt
import numpy as np
import os
from natsort import natsorted
from numpy import random

from amuse.datamodel import Particles, Particle
from amuse.ext.orbital_elements import new_binary_from_orbital_elements
from amuse.lab import read_set_from_file, write_set_to_file
from amuse.units import units, constants

def clean_plot(ax, plot_type):
    import matplotlib.ticker as mtick

    label_size = 16
    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('both')
    ax.xaxis.set_minor_locator(mtick.AutoMinorLocator())
    ax.yaxis.set_minor_locator(mtick.AutoMinorLocator())
    if plot_type=='hist':
        ax.tick_params(axis="y", labelsize=label_size)
        ax.tick_params(axis="x", labelsize=label_size)
    else:
        ax.tick_params(axis="y", which='both', direction="in", labelsize=label_size)
        ax.tick_params(axis="x", which='both', direction="in", labelsize=label_size)

    return ax, label_size

def ZAMS_radius(mass):
  """Define stellar radius at ZAMS"""

  mass = mass.value_in(units.MSun)
  mass_sq = (mass)**2
  r_zams = mass**1.25*(0.1148+0.8604*mass_sq)/(0.04651+mass_sq)
  return r_zams | units.RSun

def new_rotation_matrix_from_euler_angles(phi, theta, chi):
  cosp=np.cos(phi)
  sinp=np.sin(phi)
  cost=np.cos(theta)
  sint=np.sin(theta)
  cosc=np.cos(chi)
  sinc=np.sin(chi)
  return np.array(
      [[cost*cosc, -cosp*sinc + sinp*sint*cosc, sinp*sinc + cosp*sint*cosc], 
        [cost*sinc, cosp*cosc + sinp*sint*sinc, -sinp*cosc + cosp*sint*sinc],
        [-sint,  sinp*cost,  cosp*cost]])

def rotate(position, velocity, phi, theta, psi):
  """Rotate planetary system"""
  Runit = position.unit
  Vunit = velocity.unit
  matrix = new_rotation_matrix_from_euler_angles(phi, theta, psi)
  return (np.dot(matrix, position.value_in(Runit)) | Runit,
          np.dot(matrix, velocity.value_in(Vunit)) | Vunit)

def load_particles(plot_bool):
  """Load particles from previous data sets"""

  data_dir = os.path.join("examples", "realistic_cluster", 
                          "initial_particles", "data_files")
  env_files = natsorted(glob.glob(os.path.join(data_dir, "cluster_data/*")))[-1]
  env_data = read_set_from_file(env_files, "hdf5")

  stars = Particles(len(env_data))
  stars.mass = env_data.mass
  stars.radius = env_data.radius
  stars.velocity = env_data.velocity
  stars.position = env_data.position
  stars.type = "STAR"
  stars.syst_id = -1
  planet_data = natsorted(glob.glob(os.path.join(data_dir, "acc3_vis4_cluster/*_planet.npz")))
  planet_no = np.asarray([int(f_.split("_planet")[0][74:]) for f_ in planet_data])
  
  particle_set = Particles()
  nsyst = 0
  stars_copy = stars.copy_to_memory()
  semi = [ ]
  planet_mass = [ ]
  host_mass = [ ]
  for i in range(len(planet_data)):
    nsyst+=1

    planet = Particle()
    load_planet_data = np.load(planet_data[i], allow_pickle=True)
    load_planet_data.keys()
    planet.mass = load_planet_data['Mc'][-1]+load_planet_data['Me'][-1] | units.MJupiter
    planet.type = "PLANET"
    planet.radius = (1|units.RJupiter)*(planet.mass.value_in(units.MJupiter))**(-0.02) #From Baron et al. 2023

    host = stars_copy[int(planet_no[i])]
    host.type = "STAR"
    host.mass = load_planet_data['star_mass'] | units.MSun
    host.radius = ZAMS_radius(host.mass)
    
    semimajor = load_planet_data['a'][-1] | units.au
    eccentricity = np.random.uniform(0, 0.05)
    inclination = np.arccos(1-2*np.random.uniform(0,1, 1)) | units.rad
    long_asc_node = np.random.uniform(0, 2*np.pi, 1) | units.rad
    true_anomaly = np.random.uniform(0, 2*np.pi, 1)
    arg_periapsis = np.random.uniform(0, 2*np.pi, 1) | units.rad

    semi.append(semimajor.value_in(units.au))
    planet_mass.append(planet.mass.value_in(units.MSun))
    host_mass.append(host.mass.value_in(units.MSun))

    phi = np.radians(random.uniform(0.0, 90.0))        #rotate under x
    theta0 = np.radians((random.normal(-90.0,90.0)))   #rotate under y
    theta_inclination = np.radians(random.normal(0, 1.0))
    theta = theta0 + theta_inclination
    psi = np.radians(random.uniform(0.0, 180.0))

    binary_set = new_binary_from_orbital_elements(
                    mass1=planet.mass, mass2=host.mass,
                    semimajor_axis=semimajor,
                    eccentricity=eccentricity,
                    inclination=inclination,
                    longitude_of_the_ascending_node=long_asc_node,
                    true_anomaly=true_anomaly,
                    argument_of_periapsis=arg_periapsis,
                    G=constants.G)
    planet.position = binary_set[0].position
    planet.velocity = binary_set[0].velocity

    planet.position, planet.velocity = rotate(planet.position, 
                                              planet.velocity, 
                                              phi, theta, psi)
    planet.position+=host.position
    planet.velocity+=host.velocity

    host.syst_id = nsyst
    planet.syst_id = nsyst
    particle_set.add_particle(host)
    particle_set.add_particle(planet)

    stars-=host

  isol = stars[stars.syst_id==-1]
  isol.radius = ZAMS_radius(isol.mass)
  particle_set.add_particles(isol)
  output_dir = os.path.join("examples", "realistic_cluster", 
                    "initial_particles", "init_particle_set")
  write_set_to_file(particle_set, output_dir, "amuse", 
         close_file=True, overwrite_file=True)

  if (plot_bool):
    from scipy import stats

    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["mathtext.fontset"] = "cm"
    fig, ax = plt.subplots()
    ax, label_size = clean_plot(ax, "scatter")
    ax.set_ylabel(r"$\log_{10}m_{\rm p}$ [M$_{\odot}$]", fontsize=label_size)
    ax.set_xlabel(r"$\log_{10}a$ [AU]", fontsize=label_size)
    ax.set_ylim(-5.2,0)
    ax.set_xlim(-2.4, 2.4)
    colour_axes = ax.scatter(np.log10(semi), np.log10(planet_mass), 
                            edgecolors="black", c=np.log10(host_mass), 
                            cmap="Blues")
    colour_bar = plt.colorbar(colour_axes, ax=ax)
    colour_bar.set_label(label=r"$\log_{10}M_{\rm *}$ [M$_{\odot}$]", fontsize=label_size)
    plt.savefig(os.path.join(data_dir, "initial_planet_parameters.pdf"), 
                dpi=300, bbox_inches="tight")
    plt.clf()


    xx, yy = np.mgrid[-2.4:2.4:200j, -5.2:0:200j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    values = np.vstack([np.log10(semi)[:,0], np.log10(planet_mass)])
    kernel = stats.gaussian_kde(values, bw_method = "silverman")
    f = np.reshape(kernel(positions).T, xx.shape)

    fig, ax = plt.subplots()
    ax, label_size = clean_plot(ax, "hist")
    ax.set_ylabel(r"$\log_{10}m_{\rm p}$ [M$_{\odot}$]", fontsize=label_size)
    ax.set_xlabel(r"$\log_{10}a$ [AU]", fontsize=label_size)
    ax.set_ylim(-5.2,0)
    ax.set_xlim(-2.4, 2.4)
    cfset = ax.contourf(xx, yy, f, cmap="Blues", levels=7, zorder=1)
    cset = ax.contour(xx, yy, f, colors="black", levels=7, zorder=2)
    ax.clabel(cset, inline=1, fontsize=10)
    plt.savefig(os.path.join(data_dir, "initial_planet_parameters_contours.pdf"), 
                dpi=300, bbox_inches="tight")
    plt.clf()

    xx, yy = np.mgrid[-5.2:0:200j, -1.4:0.4:200j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    values = np.vstack([np.log10(planet_mass), np.log10(host_mass)])
    kernel = stats.gaussian_kde(values, bw_method = "silverman")
    f = np.reshape(kernel(positions).T, xx.shape)

    x = np.linspace(-5.2,0)
    fig, ax = plt.subplots()
    ax, label_size = clean_plot(ax, "hist")
    ax.set_ylabel(r"$\log_{10}M_{*}$ [M$_{\odot}$]", fontsize=label_size)
    ax.set_xlabel(r"$\log_{10}m_{\rm p}$ [M$_{\odot}$]", fontsize=label_size)
    ax.set_xlim(-5.2,0)
    ax.set_ylim(-1.4, 0.4)
    ax.text(-1.6,-1.3,r"$m_{\rm p} < M_{*}$", rotation=65, fontsize=label_size)
    ax.text(-1.2,-1.3,r"$m_{\rm p} > M_{*}$", rotation=65, fontsize=label_size)
    cfset = ax.contourf(xx, yy, f, cmap="Blues", levels=7, zorder=1)
    cset = ax.contour(xx, yy, f, colors="black", levels=7, zorder=2)
    ax.plot(x, x, linestyle=":", color="black")
    ax.clabel(cset, inline=1, fontsize=10)
    plt.savefig(os.path.join(data_dir, "initial_mass_parameters_contours.pdf"), 
                dpi=300, bbox_inches="tight")
    plt.clf()
  
load_particles(plot_bool=True)