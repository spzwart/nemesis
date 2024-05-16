Code generalising the nemesis algorithm using [AMUSE](https://amuse.readthedocs.io/en/latest/) functionalities. \\
Run conda install --file requirements.txt to install the required libraries. \\
Run python interface.py from root directory to execute code.

- interface.py: Run code to simulate your system
- src/environment\_functions.py: Script with various functions to define different environment properties
- src/grav_correctors.py: Script correcting the magnitude and projection of force felt by parents due to children, and children due to parents
- src/hierarchical\_particles.py: Script to categorise particles into parents and children
- src/nemesis.py: Script hosting the evolution procedure
- examples/: Folders with several examples initialising particles set to be run.

To compile src/grav_kicks.cpp execute: g++ -shared -o src/gravity.so -g -fPIC src/grav_kicks.cpp \\
To compile src/ejector_calc.cpp execute: g++ -shared -o src/ejection.so -g -fPIC src/ejector_calc.cpp

Unless in branch 'one\_child\_system', children identified as particles with attribute syst\_id>0. Corresponding parents have the same syst_id value.

 Branch | Application | Examples 
:---|:---:|---:
 refactor |  Planetary Systems | realistic\_cluster, ejecting\_suns 
 one\_child\_system |   Democratic AGN | runaway\_bh 
 PN\_global |  Hierarchical AGN | S-Stars 
