Code generalising the nemesis algorithm. Run python interface.py from root directory to execute code.

- examples/: Folders with several examples initialising particles set to be run.
Children systems are those detected with attribute syst_id>0, with equivalent syst_id particles being children of a parent system.
Only systems with planets can be children (Two nearby stars will not form a children system)

- environment_functions.py: Script with various functions to define different environment properties
- grav_correctors.py: Script correcting the magnitude and projection of force felt by parents due to children, and children due to parents
- hierarchical_particles.py: Script to categorise particles into parents and children
- interface.py: Run code to simulate your system
- nemesis.py: Script hosting the evolution procedure
