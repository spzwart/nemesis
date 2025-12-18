# NEMESIS

**Nemesis** is a flexible, **multi-physics**, **multi-scale algorithm** for integrating hierarchical systems (e.g., planetary systems in star clusters, circumstellar disks, or binaries in galactic environments) embedded within the [AMUSE](https://amuse.readthedocs.io/en/latest/) library.  

Nemesis works by decoupling tight subsystems from the global environment, integrating them in isolation, and then synchronising the micro- and macroscales at regular intervals. This scheme allows:
- High parallelisability
- Accurate energy conservation compared to direct N-body only runs
- Seamless inclusion of the galactic tidal field and stellar evolution  

A full description is given in *Hochart & Portegies Zwart (in prep.)*.  
A demonstration video is available [here](https://youtu.be/cycIn8hDZKY).  
For AMUSE installation instructions, see [this guide](https://amuse.readthedocs.io/en/latest/install/installing.html).

At runtime, Nemesis automatically creates output directories for a given run. These are hosted under `data/`:
- **`simulation_snapshot/`** – HDF5 snapshots of particle phase-space and masses
- **`collision_snapshot/`** – plain-text files describing detected collisions
- **`sim_stats/`** – text files with run statistics  

Runs can be **resumed automatically**, provided diagnostic parameters (`dtbridge`, `dt_diag`) are unchanged.


### Installation & Running
1. **Install dependencies**: <br />
    ```conda install --file requirements.txt``` <br />
    or <br />
    ```pip install -r requirements.txt```
2. **Compile C++ files**. These are used to calculate the correction kicks between subsystems and the global environment, synchronising the micro- and macrostate: <br />
    ```cd src/cpp``` <br />
    ```make```
3. **Generate initial condiitons**. For instance: <br />
    ```cd examples/```
    ```python basic_cluster/particle_initialiser.py```
    This will create a particle set with several planetary systems. The particle set are always saved in a folder ```initial_particles/```.
4. **Run simulation**. From the project root: <br />
    ```python main.py```
   If, instead, you wish to simulate your system for 1 Myr with a bridge time step of 100 yr:
   ```python main.py --tend=1Myr --dtbridge=100yr```
   Command-line arguments are documented in main.py docstrings.

### Repository structure
- `main.py`: Run code to simulate your system.
- `src/environment_functions.py`: Script containing various functions to define different environment properties.
- `src/globals.py`: All global constants and magic numbers used in the simulation.
- `src/grav_correctors.py`: Force correction routines to synchronise the micro- and macrostates (synchronise parent with children).
- `src/hierarchical_particles.py`: Script to categorise particles into parents and children.
- `src/nemesis.py`: Script hosting the evolution procedure.
- `examples/`: Folders with several examples initialising particles set to be run.

### Free parameters: 
In addition to the input functions needed to execute `interface.py`, the following may vary depending on your simulation:

main.py:
- `galactic_frame()`: The phase-space coordinates. Default is centered about a Milky Way-like galaxy.
- `RUN_IDX`: The system realisation within your `initial_particles/` directory wished to simulate.

src/globals.py:
- `ASTEROID_RADIUS`: Collision radius for asteroid (test) particles.
- `CONNECTED_COEFF`: Threshold for detecting particles within a subsystem that are ejected.
- `EPS`: Tolerance with which models have successfully integrated to required time step.
- `GRAV_THRESHOLD`: Threshold for modifying the parent particle radius in case it is relatively isolated.
- `MIN_EVOL_MASS`: The minimum mass for a particle to be flagged for stellar evolution.
- `PARENT_NWORKER`: Number of workers for parent integrator.
- `PARENT_RADIUS_COEFF`: Pre-factor influencing the parent particle radius.
- `PARENT_RADIUS_MAX`: Maximum allowed parent particle radius.

src/nemesis.py:
- `_sub_worker()`: Number of child workers. Dedicated gravitational solver for subsystems.
- `_parent_worker`: Dedicated gravitational solver for parent code.

### EXAMPLE:
To run example script, execute `python basic_cluster/particle_initialiser.py` to create an AMUSE particle set. Afterwards, execute `python main.py`.

### TESTS:
An example Nemesis test is provided, comparing its performance relative to N-body integrators in capturing the von Zeipel-Lidov-Kozai effect.
To run this test follow:
- Set-up initial conditions: `python /tests/ZKL_test/initialise_LZK.py`
- To run Nemesis: `python /tests/ZKL_test/run_ZKL.py`
- To plot results: `python/tests/ZKL_test/plot_ZKL.py`

Some vital notes regarding the test:
- At times, `Huayno` is unable to capture the LZK effect. These are some suggested parameters:
    - In `nemesis._parent_worker`, use `Huayno` as parent integrator with mode `SHARED10_COLLISIONS`.
    - In `nemesis._sub_worker`, use `Huayno` as children integrator with mode `SHARED10_COLLISIONS`.
    - In `nemesis._sub_worker` set child converter with `scale_radius/10.`
    - End time: 10 Myr
    - Bridge time: 500 yr
    - Diagnostic time: 5000 yr
    - Code internal time-step: 0.1
    - Turn off galactic field + stellar evolution
    - Change `PARENT_RADIUS_COEFF` in `src/globals` to 1e-5 au, 100 au and 1000 au.
    - Turn off children collisions
Make sure that the parent and child code is the same integrator. This allows testing the performance of the child split algorithm vs. non-splitting scenarios.

### Example Scientific Runs
- [van Elteren et al. 2019: Survivability of planetary systems in young
and dense star clusters](https://www.aanda.org/articles/aa/full_html/2019/04/aa34641-18/aa34641-18.html)

### NOTES:
- To setup children at the initial time step, it is required that the particle set contains a `syst_id` attribute whose value is an integer. The set of particles with the same `syst_id` value will be flagged as a subsystem as long as `syst_id` > 0.
- Since Nemesis relies heavily on frequent stop/start (hibernate/resume) cycles for its child integrators, sockets are used instead of MPI. The persistent stop/start cycles conflict with MPI worker behaviour since MPI workers cannot safely handle repeated suspend/resume signals, especially in large-N simulations where hundreds of worker processes are active. Repeated stop/start operations can lead to workers being incorrectly terminated, crashing the simulation altogether. Socket-based channels, however, can tolerate stop/start cyles because they do not use the tightly coupled, state-sensitive collective semantics of MPI. The problem, however, is that children code are restricted to one core per.