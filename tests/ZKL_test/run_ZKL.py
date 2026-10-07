"""
Run a Lidov-Kozai test simulation.
This is to check whether Nemesis can accurately reproduce
the Lidov-Kozai mechanism in hierarchical triple systems.
"""


import glob
from amuse.units import units
from main import run_simulation


IC_file = glob.glob("tests/ZKL_test/data/ICs/*")[0]
run_simulation(
    IC_file=IC_file,
    run_idx=0,
    tend=10 | units.Myr,
    dtbridge=500 | units.yr,
    code_dt=0.1,
    dt_diag=500 | units.yr,
    n_worker_parent=1,
    gal_field=0,
    dE_track=1,
    star_evol=0,
    verbose=1
)
