import glob
import matplotlib.pyplot as plt
from natsort import natsorted
import os

from amuse.community.hop.interface import Hop
from amuse.lab import units, nbody_system
from amuse.lab import read_set_from_file, write_set_to_file

def load_particles():
    """Load particles. Creates a single-planet planetary system."""
    # Load cluster data (Wilhem & Portegies Zwart (in works))
    data_dir = os.path.join("examples", "realistic_cluster", 
                            "initial_particles", "data_files")
    cluster_data_files = os.path.join(data_dir, "cluster_data/*")
    env_files = natsorted(glob.glob(cluster_data_files))[-1]
    env_data = read_set_from_file(env_files, "hdf5")
    
    unit_converter = nbody_system.nbody_to_si(1|units.MSun, 1|units.RSun)
    hop = Hop(unit_converter)
    hop.particles.add_particles(env_data)
    hop.calculate_densities()

    mean_densty = 0.02*hop.particles.density.mean() 
    hop.parameters.peak_density_threshold = mean_densty
    hop.parameters.saddle_density_threshold = 0.03*mean_densty
    hop.parameters.outer_density_threshold = 0.000001*mean_densty

    hop.do_hop()
    result = [x.get_intersecting_subset_in(env_data) for x in hop.groups()]
    hop.stop()
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    n=0
    for sub_cluster in result:
        ax.scatter(sub_cluster.x.value_in(units.pc),
                   sub_cluster.y.value_in(units.pc),
                   sub_cluster.z.value_in(units.pc),
                   s=5
                   )
    ax.view_init(10, 20)
    ax.set_xlabel(r"$x$ [pc]")
    ax.set_ylabel(r"$y$ [pc]")
    ax.set_zlabel(r"$z$ [pc]")
    plt.savefig("Maite_Subcluster.pdf", dpi=200, bbox_inches="tight")
    write_set_to_file(sub_cluster, "examples/realistic_cluster/subcluster_pset", "amuse", 
                      close_file=True, overwrite_file=True)
    
    
  
load_particles()