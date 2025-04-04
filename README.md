# NEMESIS

This project generalises the Nemesis algorithm using [AMUSE](https://amuse.readthedocs.io/en/latest/). A small example video is seen [here](https://youtu.be/cycIn8hDZKY). For AMUSE install see the following [GitHub repository](https://github.com/LourensVeen/amuse-course).

Nemesis is a flexible and efficient multi-scale algorithm used to integrate hierarchical systems, for instance planetary systems in stellar clusters. It does so by decoupling smaller systems from one another to allow their integration to run in parallel and with the use of bridge and considers the galactic tidal field and stellar evolution.

### Running instructions
1. Install the required libraries: <br />
    ```conda install --file requirements.txt``` <br />
    or <br />
    ```pip install -r requirements.txt```
2. Compile C++ files: <br />
    ```cd src/cpp``` <br />
    ```make```
3. Create a cluster particle set. For instance, in ```examples/``` execute: <br />
    ```python basic_cluster/particle_initialiser.py```
    This will create a particle set with several planetary systems. The particle set are always saved in a folder ```initial_particles/```.
4. Execute script from the root directory. Make sure that the ```initial_particles``` in the main directs the code to the target ```initial_particles/``` directory.To run with default values execute: <br />
    ```python main.py```
Ensure the main script is correctly pointed to the initial_particles/ directory. Data (including collision events, snapshots, and statistics) will be saved in a designated output folder.


### Script contents
- `main.py`: Run code to simulate your system
- `src/environment_functions.py`: Script containing various functions to define different environment properties
- `src/grav_correctors.py`: Script correcting the magnitude and projection of force felt by parents due to children, and children due to parents
- `src/hierarchical_particles.py`: Script to categorise particles into parents and children
- `src/nemesis.py`: Script hosting the evolution procedure
- `examples/`: Folders with several examples initialising particles set to be run.

### Free parameters: 
In addition to the input functions needed to execute `interface.py`, the following may vary depending on your simulation:

main.py:
- `galactic_frame()`: The phase-space coordinates centered about a Milky Way-like galaxy.

src/globals.py:
- `CONNECTED_COEFF`: Threshold for detecting what constitutes an ejected children.
- `EPS`: Tolerance with which models have successfully integrated to required timestep.
- `GRAV_THRESHOLD`: Threshold for modifying the parent radius in case it is relatively isolated.
- `MIN_EVOL_MASS`: The minimum mass for a particle to be flagged for stellar evolution.
- `PARENT_RADIUS_COEFF`: Pre-factor influencing the parent system radius.
- `PARENT_RADIUS_MAX`: Maximum allowed parent radius.

src/nemesis.py:
- In `__init__`, `maximum_radius`: Maximum parent radius
- In `__init__`, `minimum_radius`: Minimum parent radius
- In `_sub_worker()`: Number of child workers.

### NOTES:
- Children are identified as particles with attribute `syst_id > 0`. Their parents are identified with the same `syst_id` value.