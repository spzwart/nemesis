import glob
import matplotlib.pyplot as plt
from natsort import natsorted
import os

from amuse.community.hop.interface import Hop
from amuse.lab import units, nbody_system
from amuse.lab import read_set_from_file, write_set_to_file

def load_particles(config_choice):
    """Load particles. Creates a single-planet planetary system."""
    # Load cluster data (Wilhem & Portegies Zwart (in works))
    data_dir = os.path.join("examples", "realistic_cluster", 
                            "initial_particles", "data_files",
                            config_choice)
    cluster_data_files = os.path.join(data_dir, "cluster_data/*")
    env_files = natsorted(glob.glob(cluster_data_files))[-10]
    print(env_files)
    env_data = read_set_from_file(env_files, "hdf5")
    env_data.move_to_center()
    
    unit_converter = nbody_system.nbody_to_si(1|units.MSun, 1|units.RSun)
    hop = Hop(unit_converter)
    hop.particles.add_particles(env_data)
    hop.calculate_densities()

    mean_densty = hop.particles.density.mean() 
    hop.parameters.peak_density_threshold = mean_densty
    hop.parameters.saddle_density_threshold = 0.9*mean_densty
    hop.parameters.outer_density_threshold = 0.01*mean_densty

    hop.do_hop()
    result = [x.get_intersecting_subset_in(env_data) for x in hop.groups()]
    hop.stop()
    fig = plt.figure(figsize=(12,6))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122)
    n=0
    ax1.scatter(env_data.x.value_in(units.pc),
                env_data.y.value_in(units.pc),
                env_data.z.value_in(units.pc),
                s=5
                )
    ax2.scatter(env_data.x.value_in(units.pc),
                env_data.y.value_in(units.pc),
                s=5)
    for sub_cluster in result:
        com = sub_cluster.center_of_mass()
        print(len(sub_cluster), len(env_data))
        if len(sub_cluster) < 1000:
            distances = (sub_cluster.position - com).lengths()
            sub_cluster -= sub_cluster[distances > (1.3|units.pc)]
        
        ax1.scatter(sub_cluster.x.value_in(units.pc),
                    sub_cluster.y.value_in(units.pc),
                    sub_cluster.z.value_in(units.pc),
                    s=5
                    )
        ax2.scatter(sub_cluster.x.value_in(units.pc),
                    sub_cluster.y.value_in(units.pc),
                    s=5)
    ax1.view_init(10, 20)
    for ax in [ax1, ax2]:
        ax.set_xlabel(r"$x$ [pc]")
        ax.set_ylabel(r"$y$ [pc]")
        ax.set_xlim(-6.4, 6.4)
        ax.set_ylim(-6.4, 6.4)
    ax1.set_zlim(-6.4, 6.4)
    ax1.set_zlabel(r"$z$ [pc]")
    plt.show()
    STOP
    write_set_to_file(sub_cluster, "examples/realistic_cluster/subcluster_pset_"+str(config_choice), "amuse", 
                      close_file=True, overwrite_file=True)

config_choices = ["config_1_vh", "config_2_vm", "config_3_vl"]
load_particles(config_choice=config_choices[0])