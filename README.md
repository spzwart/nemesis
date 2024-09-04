This project generalises the nemesis algorithm using [AMUSE](https://amuse.readthedocs.io/en/latest/).

### Running instructions
1. Install the required libraries: conda install --file requirements.txt <br />
2. Compile cpp for gravitational kicks: g++ -shared -o src/gravity.so -g -fPIC src/grav_kicks.cpp <br />
3. Compile cpp file tracking ejections: g++ -shared -o src/ejection.so -g -fPIC src/ejector_calc.cpp <br />
4. Execute script from the root directory: python main.py


### Script contents
- main.py: Run code to simulate your system
- src/environment\_functions.py: Script containing various functions to define different environment properties
- src/grav_correctors.py: Script correcting the magnitude and projection of force felt by parents due to children, and children due to parents
- src/hierarchical\_particles.py: Script to categorise particles into parents and children
- src/nemesis.py: Script hosting the evolution procedure
- examples/: Folders with several examples initialising particles set to be run.

 Branch | Application | Examples 
:---|:---|:---
refactor |  Planetary Systems | realistic\_cluster
PN\_global | AGN Environment | S-Stars 
 
Children are identified as particles with attribute syst\_id > 0. Their corresponding parents are identified with the same syst_id value.

### Free parameters: 
In addition to the input functions needed to execute interface.py, the following may vary depending on your simulation:

main.py:
- galactic_frame() input position and velocity coordinates.
- ITER_PER_SNAP: Number of time steps per snapshot.
- typical_crosstime: The typical crossing time of a parent system. Tendency is to keep diagnostic timestep below 5 times the crossing time.

src/environment_functions.py:
- threshold: If no second-nearest parent system is within DIST_THRESHOLD from iterated parent, it is 'isolated' and a possible ejector.

src/nemesis.py:
- In commit_particles(), particles flagged for stellar evolution depend on their 'type' attribute.
- parent_worker() time step parameter. Make sure it is small enough that the parent worker's model time ~ required time step.
- If test particles are present, they are drifted every 10 steps.
- In split_subcodes(), there is a coefficient influencing the parent radius to detect 'dettached' objects.