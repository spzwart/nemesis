import glob
import os
from amuse.units import units
from main import run_simulation


IC_file = glob.glob("tests/cluster_test/data/ICs/*")[0]
RUN_NEMESIS = 1

tend = 0.1 | units.Myr
dt = 500 | units.yr
dt_diag = 10000 | units.yr
code_dt = 0.1


if RUN_NEMESIS:
    run_simulation(
        IC_file=IC_file,
        run_idx=0,
        tend=tend,
        dtbridge=dt,
        n_worker_parent=1,
        code_dt=code_dt,
        dt_diag=dt_diag,
        gal_field=0,
        dE_track=1,
        star_evol=0,
        verbose=1
    )
else:
    from amuse.community.ph4.interface import Ph4
    from amuse.lab import nbody_system, read_set_from_file, write_set_to_file

    import time as cpu_time

    data_dir = "tests/cluster_test/data"
    out_dir = f"{data_dir}/cluster_run_direct/"
    dirs = ["simulation_snapshot", "sim_stats", "energy_error"]
    for d in dirs:
        os.makedirs(f"{out_dir}/{d}", exist_ok=True)

    # Load Nemesis particle set
    particle_set = read_set_from_file(
        f"{data_dir}/cluster_run_nemesis/simulation_snapshot/snap_0.hdf5"
    )
    major_bodies = particle_set[particle_set.mass > 0.08 | units.MSun]

    converter = nbody_system.nbody_to_si(
        major_bodies.mass.sum(),
        major_bodies.virial_radius()
        )
    code = Ph4(converter, number_of_workers=3)
    code.particles.add_particles(particle_set)
    code.parameters.epsilon_squared = (1 | units.au)**2.
    code.parameters.timestep_parameter = code_dt
    channel = code.particles.new_channel_to(particle_set)

    snapshot_dir = f"{out_dir}/simulation_snapshot/snap_{{}}.hdf5"
    write_set_to_file(
        particle_set,
        snapshot_dir.format(0),
        'hdf5',
        close_file=True,
        overwrite_file=True
    )

    time = 0. | units.yr
    step = 0
    diag_step = dt_diag // dt

    t0 = cpu_time.time()
    dE_arr = []
    while time < tend:
        print("Time = ", time.in_(units.yr), flush=True)

        time += dt
        code.evolve_model(time)
        channel.copy()

        step += 1
        if (step % diag_step) == 0:
            write_set_to_file(
                particle_set,
                snapshot_dir.format(step),
                'hdf5',
                close_file=True,
                overwrite_file=True
            )

    t1 = cpu_time.time()
    with open(f"{out_dir}/sim_stats/sim_stats.txt", 'w') as f:
        f.write(f"Total CPU Time: {(t1-t0)/60} minutes")
        f.write(f"\nEnd Time: {tend.in_(units.Myr)}")
        f.write(f"\nTime step: {dt.in_(units.Myr)}")

    with open(f"{out_dir}/energy_error/energy_error.csv", 'w') as f:
        f.write(f"Energy error: {dE_arr}")

    code.stop()
