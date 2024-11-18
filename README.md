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
4. Execute script from the root directory. To run with default values execute: <br />
    ```python main.py```


### Script contents
- `main.py`: Run code to simulate your system
- `src/environment_functions.py`: Script containing various functions to define different environment properties
- `src/grav_correctors.py`: Script correcting the magnitude and projection of force felt by parents due to children, and children due to parents
- `src/hierarchical_particles.py`: Script to categorise particles into parents and children
- `src/nemesis.py`: Script hosting the evolution procedure
- `examples/`: Folders with several examples initialising particles set to be run.

NOTE: Children are identified as particles with attribute `syst_id > 0`. Their corresponding parents are identified with the same `syst_id` value.

### Free parameters: 
In addition to the input functions needed to execute `interface.py`, the following may vary depending on your simulation:

main.py:
- `MIN_EVOL_MASS`: The minimum mass for a particle to be flagged for stellar evolution.
- `galactic_frame()`: The phase-space coordinates centered about a Milky Way-like galaxy.
- `typical_crosstime`: The typical crossing time of a parent system. Tendency is to keep diagnostic timestep below 10 crossing times of the smallest system.

src/environment_functions.py:
- `threshold`: If no second-nearest parent system is within `DIST_THRESHOLD` from iterated parent, it is 'isolated' and a possible ejector.
- In `set_parent_radius`, there is a pre-factor influencing the parent system radius.

src/nemesis.py:
- In `__init__`, `maximum_radius`: Maximum parent radius
- In `__init__`, `minimum_radius`: Minimum parent radius
- In `__init__`, `kick_ast_iter`: How many iterations occur between kicking isolated asteroids
- In `_parent_worker()` time step parameter.
- In `_split_subcodes()`, there is a coefficient influencing the parent radius to detect 'dettached' objects.
- In `__drift_test_particles()`, there is a coefficient tuning distance for which isolated asteroids are captured by a parent.