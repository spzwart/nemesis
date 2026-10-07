# NEMESIS

A multi-scale integration framework for hierarchical astrophysical systems embedded within the [AMUSE](https://amuse.readthedocs.io/en/latest/) library.  

**The development is ongoing at:** 
- https://gitlab.strw.leidenuniv.nl/spz/nemesis 
- https://github.com/ErwanH29/nemesis <br />
<HR>
<HR>

### Overview
```Nemesis``` works by dynamically decoupling subsystems from their parent environment.
- Fast-evolving subsystems ("children", i.e binary stars or planetary systems) are integrated in isolation.
- The global system ("parents") is integrated with a cluster-optimised integrator (i.e ```Ph4```).
- Synchronisation between scales (children and parents) occurs through a second-order kick-drift-kick scheme.

This procedure enables:
- High parallelisability.
- Improved energy conservation vs direct N-body.
- Efficient treatment of extreme scale separations.
- Seamless inclusion of tidal fields and stellar evolution.  

A full description is given in [Hochart & Portegies Zwart 2026](https://ui.adsabs.harvard.edu/abs/2026arXiv260303077H/abstract).  
A demonstration video is available [here](https://youtu.be/cycIn8hDZKY).  
For AMUSE installation instructions, see [this guide](https://amuse.readthedocs.io/en/latest/install/installing.html).

### Requirements
- Python ≥ 3.10
- AMUSE framework
- C++ compiler with C++11 support
- OpenMP support
- Conda environment recommended

### Installation & Running
1. **Cloning**: <br />
    ```git clone https://github.com/ErwanH29/nemesis.git```
2. **Install dependencies**: <br />
    ```cd nemesis```
    ```conda install --file requirements.txt``` <br />
3. **Install recommended AMUSE packages**.  `huayno` `ph4` `seba` `symple` `kepler`. If not, this command can be used during your [AMUSE installation](https://amuse.readthedocs.io/en/latest/install/installing.html)  `./setup install amuse-framework amuse-huayno amuse-ph4 amuse-seba amuse-symple amuse-kepler` <br />
4. **Compile C++ files**. These are used to calculate the correction kicks between children systems and parents, synchronising the micro- and macro-state: <br />
    ```cd src/cpp && make``` <br />
5. **Generate initial conditions**. For instance: <br />
    ```cd examples/```
    ```python basic_cluster/particle_initialiser.py```
    This will create a particle set with several planetary systems. The particle set is always saved in a folder ```ICs/```.
6. **Run the simulation**. From the project root: <br />
    ```python main.py```
   If, instead, you wish to simulate your system for 1 Myr with a bridge time step of 100 yr:
   ```python main.py --tend=1Myr --dtbridge=100yr```
   Command-line arguments are documented in main.py.

NOTES: 
- Runs can be resumed automatically. However, if any simulation parameters are changed when resuming, while a warning message is printed, the simulation will proceed.
- ```Nemesis``` must be run from the same Python environment used to compile the C++ extension.

### Output Structure
At runtime, ```Nemesis``` automatically creates output directories for a given run. These are hosted under `data/`:
- **`simulation_snapshot/`** – HDF5 particle snapshots.
- **`collision_snapshot/`** – Files logging collision events.
- **`sim_stats/`** – Runtime diagnostics.

### Repository Structure
- `main.py`: Main interfacing allowing to run simulation.
- `examples/`: Folders with an example in generating a cluster with ```AMUSE```.
- `src/environment_functions.py`: Script containing several functions which define different particle attributes and environmental properties.
- `src/globals.py`: All global constants and magic numbers used in the simulation.
- `src/grav_correctors.py`: Force correction routines to synchronise the children systems with the set of parents (synchronising the local and macro scales).
- `src/hierarchical_particles.py`: Setting up the particle sets for ```Nemesis```. Namely, categorising parents and children.
- `src/nemesis.py`: Script hosting the evolution procedure.
- `src/split_children.py`: Script to handle fragmentation of children system.
- `tests/`: Folders with several test examples.

### Free Parameters:
src/globals.py:
- `ASTEROID_RADIUS`: Collision radius for asteroid (test) particles.
- `EPS`: Integration tolerance.
- `MIN_EVOL_MASS`: Stellar evolution mass threshold.
- `PARENT_RADIUS_COEFF`: Parent radius scaling coefficient.
- `PARENT_RADIUS_MIN`: Minimum pareent radius.
- `PARENT_RADIUS_MAX`: Maximum parent radius.
- `SPLIT_PARAM`: Child system linking length.

src/nemesis.py:
- `_init_()`: Number of cores used when calculating correction kicks.
- `_parent_worker()`: Parent integrator.
- `_stellar_worker()`: Stellar evolution integrator.
- `_sub_worker()`: Number of child workers and child integrator. 
NOTE: Any worker change must be reflected in `_worker_list` to ensure proper handling at run time.

### Tests:
```Nemesis``` includes validation tests comparing performance against direct N-body integrations.

##### Von Zeipel–Lidov–Kozai Test:
To run this test follow:
- Set-up initial conditions: `python tests/ZKL_test/initialise_ZKL.py`
- To run ```Nemesis```: `PYTHONPATH=".:$PYTHONPATH" python tests/ZKL_test/run_ZKL.py`
- To plot results: `python tests/ZKL_test/plot_ZKL.py`

Suggested parameters:
- In `nemesis._parent_worker`, use `Huayno` as parent integrator with mode `SHARED10_COLLISIONS`.
- In `nemesis._sub_worker`, use `Huayno` as children integrator with mode `SHARED10_COLLISIONS`.
- Code internal time-step: 0.1.
- Turn off galactic field + stellar evolution.
- Change `PARENT_RADIUS_COEFF` in `src/globals` to 1e-5 au, 100 au and 1000 au.
- Turn off children collisions.
Make sure that the parent and child code is the same integrator so comparison between the 1e-5 au, 100 au and 1000 au models is possible.

The test allows validation of:
- Subsystem splitting.
- Ensure accurate secular evolution.

##### Asteroids in Cluster Test:
To run this test follow:
- Set-up initial conditions: `python tests/cluster_test/initialise_cluster.py`
- To run ```Nemesis```: `python tests/cluster_test/run_cluster.py` with the flag `RUN_NEMESIS = 1`
- To run direct N-body code: `python tests/cluster_test/run_cluster.py` with the flag `RUN_NEMESIS = 0`
- To plot results: `python tests/cluster_test/plot_cluster.py`

Suggested parameters:
- `Ph4` as parent, child and direct N-body integrator.
- Code parameters (i.e, internal time-step) are identical between parent, child and direct N-body integrator.
- End time: 0.1 Myr. This is a short enough time to allow any systematic errors to emerge while also remaining below the cluster's crossing time.
- Bridge time: 500 yr.
- Diagnostic time: 10000 yr.
- Code internal time-step: 0.1.
- Turn off galactic field + stellar evolution -- This will allow a better comparison in performance.
- Turn off child collisions.

### Example Scientific Runs
- [van Elteren et al. 2019: Survivability of planetary systems in young and dense star clusters](https://www.aanda.org/articles/aa/full_html/2019/04/aa34641-18/aa34641-18.html)
- [Hochart & Portegies Zwart 2026: Oort Cloud ecology: IV. Exchanging asteroids](https://ui.adsabs.harvard.edu/abs/arXiv:2604.19413)

### Notes:
- Children require a ```syst_id``` particle attribute, with ```syst_id > 0``` forming children systems.
- ```Nemesis``` uses sockets instead of MPI for child subsystems. This is because ```Nemesis``` relies heavily on hibernation/resumption cycles for its child integrators. The persistent stop/start cycles go against MPI philosophy of requiring processes to remain synchronised and active constantly. As such, interrupting or suspending workers mid-communication may lead to crashes. When a large number of children is present, this is bound to happen. Instead, sockets work as independent processes, making them more stable to the stop/start cycles.
- Stellar radii in snapshots correspond to their ZAMS radius.

### Citation:
If you use ```Nemesis``` in scientific work, please cite:
2019A&A...624A.120V, 2026arXiv260303077H