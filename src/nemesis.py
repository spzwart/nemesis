############################################# NOTES ##################################
# 1. Evolving subsystems in _evolve_coll_offset --> ONE-BY-ONE, NOT PARALLELISED. 
#    Main bottle neck and requires a change of Nemesis logic to circumvent.
# 2. split_subcodes, process_parent_merger is not parallelised. However due to requiring
#    shared memory, this is not possible.
############### OTHER ROOM FOR IMPROVEMENT:
# 1. Flexible bridge times. One can use a physical-based one like classical N-body integrations
#    or utilise machine learning.
# 2. Better defined parent radius --> Currently ignores cluster density. Again, can use a 
#    physically motivated one (some fraction of external parent gravitational force) or
#    use machine learning.
# 3. Asteroids require Ph4 as symplectic codes with test particles do not seem to allow collisions.
######################################################################################

 
from concurrent.futures import ThreadPoolExecutor, as_completed
import ctypes
import gc
import numpy as np
from numpy.ctypeslib import ndpointer
import os
import psutil
import signal
import threading
import traceback

from amuse.community.huayno.interface import Huayno
from amuse.community.ph4.interface import Ph4
from amuse.community.seba.interface import SeBa

from amuse.couple import bridge
from amuse.datamodel import Particles, Particle
from amuse.ext.basicgraph import UnionFind
from amuse.ext.composition_methods import SPLIT_4TH_S_M6
from amuse.ext.galactic_potentials import MWpotentialBovy2015
from amuse.ext.orbital_elements import orbital_elements
from amuse.lab import write_set_to_file
from amuse.units import units, constants, nbody_system

from src.environment_functions import connected_components_kdtree, set_parent_radius
from src.environment_functions import planet_radius, ZAMS_radius
from src.globals import ASTEROID_RADIUS, CONNECTED_COEFF, EPS
from src.globals import MIN_EVOL_MASS, PARENT_RADIUS_MAX, PARENT_NWORKER
from src.grav_correctors import CorrectionFromCompoundParticle
from src.grav_correctors import CorrectionForCompoundParticle
from src.hierarchical_particles import HierarchicalParticles


class Nemesis(object):
    def __init__(self, par_conv, dtbridge, coll_dir, 
                 available_cpus=os.cpu_count(), nmerge=0,
                 resume_time=0. | units.yr, code_dt=0.03, 
                 dE_track=False, star_evol=False, 
                 gal_field=True, verbose=True):
        """
        Class setting up the simulation.
        Args:
            par_conv (converter):  Parent N-body converter
            dtbridge (units.time):  Diagnostic time step
            coll_dir (str):  Path to store collision data
            available_cpus(int):  Number of available CPUs
            nmerge (int):  Number of mergers loaded particle set has
            resume_time (units.time):  Time which simulation is resumed at
            code_dt (float):  Internal time step
            dE_track (bool):  Flag turning on/off energy error tracker
            star_evol (bool):  Flag turning on/off stellar evolution
            gal_field (bool):  Flag turning on/off galactic field
            verbose (bool):  Flag turning on/off verbose output
        """
        # Private attributes
        self.__dt = dtbridge
        self.__coll_dir = coll_dir
        self.__code_dt = code_dt
        self.__par_nworker = PARENT_NWORKER
        self.__star_evol = star_evol
        self.__gal_field = gal_field
        self.__total_free_cpus = available_cpus
        self.__main_process = psutil.Process(os.getpid())
        self.__nmerge = nmerge
        self.__resume_offset = resume_time
        self.__dE_track = dE_track

        # Protected attributes
        self._verbose = verbose
        self._parent_conv = par_conv
        self._lock = threading.Lock()
        self._MWG = MWpotentialBovy2015()
        self._time_offsets = dict()
        self._pid_workers = dict()
        self._child_channels = dict()
        self._new_systems = dict()
        self._new_offsets = dict()

        self.parent_code = self._parent_worker()
        self.grav_coll = self.parent_code.stopping_conditions.collision_detection
        self.grav_coll.enable()
        if (self.__star_evol):
            self.stellar_code = self._stellar_worker()
            self.SN_detection = self.stellar_code.stopping_conditions.supernova_detection
            self.SN_detection.enable()
        self.particles = HierarchicalParticles(self.parent_code.particles)
        self.subcodes = dict()
        self.dt_step = 0
        
        self._major_channel_maker()
        self._validate_initialization()
        self.lib = self._load_grav_lib()

    def _validate_initialization(self) -> None:
        """Validate initialised variables of the class"""
        if self.__dt is None or self.__dt <= 0 | units.s:
            raise ValueError("Error: dt must be a positive float")
        if not isinstance(self.__code_dt, (int, float)) or self.__code_dt <= 0:
            raise ValueError("Error: code_dt must be a positive float")
        if not isinstance(self.__par_nworker, int) or self.__par_nworker <= 0:
            raise ValueError("Error: par_nworker must be a positive integer")
        if not isinstance(MIN_EVOL_MASS.value_in(units.MSun), float) \
            or MIN_EVOL_MASS <= 0 | units.kg:
                raise ValueError(f"Error: minimum stellar mass {MIN_EVOL_MASS} must be a positive float")
        if not isinstance(self.__coll_dir, str):
            raise ValueError("Error: coll_dir must be a string")

    def _load_grav_lib(self) -> ctypes.CDLL:
        """Setup library to allow Python and C++ communication"""
        py_to_c_types = ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS')
        lib = ctypes.CDLL('./src/build/kick_particles_worker.so')
        lib.find_gravity_at_point.argtypes = [
            py_to_c_types,
            py_to_c_types,
            py_to_c_types,
            py_to_c_types,
            py_to_c_types,
            py_to_c_types,
            py_to_c_types,
            py_to_c_types,
            py_to_c_types,
            py_to_c_types,
            ctypes.c_int,
            ctypes.c_int
        ]
        lib.find_gravity_at_point.restype = None
        return lib
    
    def commit_particles(self) -> None:
        """
        Commit particle system by:
            - recentering the system
            - setting the parent radius
            - initialising children codes
            - channels for children
            - defining stellar code particles
            - setting up the galactic field
        """
        particles = self.particles
        subsystem = self.subsystems
        subcodes = self.subcodes
        
        # Clean-up any old subcodes if needed
        for parent_key, (parent, code) in subcodes.items():
            if ((parent_key in subsystem) and \
                (subsystem[parent_key] is subcodes[parent_key].particles)):
                continue
            self._time_offsets.pop(code)
            del code

        if (self.__star_evol):
            particles = particles.all()
            star_mask = particles.mass > MIN_EVOL_MASS
            self.stars = particles[star_mask]
            self.stellar_code.particles.add_particle(self.stars)

        particles.radius = set_parent_radius(particles.mass)
        ntotal = len(self.subsystems.keys())
        for nsyst, (parent_key, (parent, sys)) in enumerate(self.subsystems.items()):
            sys.move_to_center()
            
            scale_mass = parent.mass
            scale_radius = set_parent_radius(scale_mass)
            parent.radius = scale_radius
            if parent not in self.subcodes:
                code, number_of_workers = self._sub_worker(children=sys, 
                                                           scale_mass=scale_mass, 
                                                           scale_radius=scale_radius)

                self._time_offsets[code] = self.model_time
                self.subsystems[parent_key] = (parent, sys)
                self.subcodes[parent_key] = code
                self._child_channel_maker(
                    parent_key=parent_key, 
                    code_particles=code.particles, 
                    children=sys
                )

                # Store children PID to allow hibernation
                worker_pid = self.get_child_pids(number_of_workers)
                self._pid_workers[parent_key] = worker_pid
                self.hibernate_workers(worker_pid)

            if self._verbose:
                print(f"System {nsyst+1}/{ntotal}, radius: {parent.radius.in_(units.au)}")

        self.particles.recenter_subsystems(max_workers=self.avail_cpus)
        overly_massive = particles.radius > PARENT_RADIUS_MAX
        particles[overly_massive].radius = PARENT_RADIUS_MAX

        if (self.__gal_field):
            self._setup_bridge()
        else:
            self._evolve_code = self.parent_code

    def _setup_bridge(self) -> None:
        """Embed system into galactic potential"""
        gravity = bridge.Bridge(use_threading=True, method=SPLIT_4TH_S_M6,)
        gravity.add_system(self.parent_code, (self._MWG, ))
        gravity.timestep = self.__dt
        self._evolve_code = gravity

    def _stellar_worker(self) -> SeBa:
        """Define stellar evolution integrator"""
        return SeBa()

    def _parent_worker(self):
        """Define global integrator"""
        code = Ph4(self._parent_conv, number_of_workers=self.__par_nworker)
        code.parameters.epsilon_squared = (0. | units.au)**2.
        code.parameters.timestep_parameter = self.__code_dt
        return code

    def _sub_worker(self, children: Particles, scale_mass, scale_radius):
        """
        Initialise children integrator.
        Args:
            children (Particles):  Children systems
            scale_mass (units.mass):  Mass of the system
            scale_radius (units.length):  Radius of the system
        Returns:
            Code:  Gravitational integrator with particle set
        """
        if len(children) == 0:
            raise ValueError("Error: No children provided.")

        converter = nbody_system.nbody_to_si(scale_mass, scale_radius)

        # Most efficient to keep the number of workers to 1 for each children.
        # Change only if number of cores available > number of children systems.
        number_of_workers = 1
        if (0. | units.kg) in children.mass:
            code = Ph4(converter, number_of_workers=number_of_workers)
            code.parameters.epsilon_squared = (0. | units.au)**2.
            code.parameters.timestep_parameter = self.__code_dt
            code.particles.add_particles(children)

        else:
            code = Huayno(converter, number_of_workers=number_of_workers)
            code.particles.add_particles(children)
            code.parameters.epsilon_squared = (0. | units.au)**2.
            code.parameters.timestep_parameter = self.__code_dt
            code.set_integrator("SHARED8_COLLISIONS")

        return code, number_of_workers

    def get_child_pids(self, number_of_workers) -> list:
        """Returns the PID of the most recently spawned children worker"""
        child_pids = []
        for child in self.__main_process.children(recursive=True):
            if len(child_pids) >= number_of_workers:
                break
            try:
                if 'ph4_worker' in child.name() or 'huayno_worker' in child.name():
                    child_pids.append(child.pid)
            except Exception as e:
                print(f"Error extracting PID: {e}")
                print("Check your children integrator matches the expected name")
                exit(-1)

        return child_pids

    def hibernate_workers(self, pid_list: list) -> None:
        """
        Hibernate workers to reduce CPU usage and optimise performance.
        Args:
            pid (list):  List of process ID of worker
        """
        for pid in pid_list:
            try:
                os.kill(pid, signal.SIGSTOP)
            except ProcessLookupError:
                print(f"Warning: Process {pid} not found. It may have exited.")
            except PermissionError:
                print(f"Error: Insufficient permissions to stop process {pid}.")

    def resume_workers(self, pid_list: list) -> None:
        """
        Resume workers to continue simulation.
        Args:
            pid (int):  Process ID of worker
        """
        for pid in pid_list:
            try:
                os.kill(pid, signal.SIGCONT)
            except ProcessLookupError:
                print(f"Warning: Process {pid} not found. It may have exited.")
            except PermissionError:
                print(f"Error: Insufficient permissions to stop process {pid}.")

    def _major_channel_maker(self) -> None:
        """Create channels for communication between codes"""
        parent_particles = self.parent_code.particles
        if self.__star_evol:
            self.channels = {
                "from_stellar_to_gravity":
                    self.stellar_code.particles.new_channel_to(
                        parent_particles,
                        attributes=["mass"],
                        target_names=["mass"]
                    ),
                "from_gravity_to_parents":
                    parent_particles.new_channel_to(
                        self.particles,
                        attributes=["x", "y", "z", "vx", "vy", "vz"],
                        target_names=["x", "y", "z", "vx", "vy", "vz"]
                    ),
                "from_parents_to_gravity":
                    self.particles.new_channel_to(
                        parent_particles,
                        attributes=["x", "y", "z", "vx", "vy", "vz"],
                        target_names=["x", "y", "z", "vx", "vy", "vz"]
                    )
            }
        else:
            self.channels = {
                "from_gravity_to_parents":
                    parent_particles.new_channel_to(
                        self.particles,
                        attributes=["x", "y", "z", "vx", "vy", "vz"],
                        target_names=["x", "y", "z", "vx", "vy", "vz"]
                    ),
                "from_parents_to_gravity":
                    self.particles.new_channel_to(
                        parent_particles,
                        attributes=["x", "y", "z", "vx", "vy", "vz"],
                        target_names=["x", "y", "z", "vx", "vy", "vz"]
                    )
            }

    def _child_channel_maker(self, parent_key: int, code_particles: Particles, children: Particles) -> None:
        """
        Create communication channel between codes and specified children system
        Args:
            parent_key (int):  Parent particle key
            code_particles (Particles):  Children particle set in grav. code
            children (Particles): Children particle set in local memory
        """
        grav_copy_variables = [
            "x", "y", "z", 
            "vx", "vy", "vz", 
            "radius", "mass"
        ]

        if (self.__star_evol):
            self._child_channels[parent_key] = {
                "from_star_to_gravity":
                    self.stellar_code.particles.new_channel_to(
                        code_particles,
                        attributes=["mass", "radius"],
                        target_names=["mass", "radius"]
                        ),
                "from_star_to_children":
                    self.stellar_code.particles.new_channel_to(
                        children,
                        attributes=["age", "relative_age"],
                        target_names=["age", "relative_age"]
                        ),
                "from_gravity_to_children": 
                    code_particles.new_channel_to(
                        children,
                        attributes=grav_copy_variables,
                        target_names=grav_copy_variables
                        ),
                "from_children_to_gravity": 
                    children.new_channel_to(
                        code_particles,
                        attributes=grav_copy_variables,
                        target_names=grav_copy_variables
                        )
                }
        else:
            self._child_channels[parent_key] = {
                "from_gravity_to_children": 
                    code_particles.new_channel_to(
                        children,
                        attributes=grav_copy_variables,
                        target_names=grav_copy_variables
                        ),
                "from_children_to_gravity": 
                    children.new_channel_to(
                        code_particles,
                        attributes=grav_copy_variables,
                        target_names=grav_copy_variables
                        )
                }

    def calculate_total_energy(self) -> float:
        """Calculate systems total energy."""
        all_parts = self.particles.all()
        Ek = all_parts.kinetic_energy()
        Ep = all_parts.potential_energy()
        Etot = Ek + Ep
        all_parts.remove_particles(all_parts)
        return Etot

    def _star_channel_copier(self) -> None:
        """Copy attributes from stellar code to grav. particle set"""
        self.channels["from_stellar_to_gravity"].copy()
        pid_dictionary = self._pid_workers
        for parent_key, channel in self._child_channels.items():
            pid = pid_dictionary[parent_key]
            self.resume_workers(pid)
            channel["from_star_to_gravity"].copy()
            channel["from_star_to_children"].copy()
            self.hibernate_workers(pid)

    def _sync_grav_to_local(self) -> None:
        """Sync gravity particles to local set"""
        self.channels["from_gravity_to_parents"].copy()
        pid_dictionary = self._pid_workers
        for parent_key, channel in self._child_channels.items():
            pid = pid_dictionary[parent_key]
            self.resume_workers(pid)
            channel["from_gravity_to_children"].copy()
            self.hibernate_workers(pid)

    def _sync_local_to_grav(self) -> None:
        """Sync local particle set to global integrator"""
        self.particles.recenter_subsystems(max_workers=self.avail_cpus)
        self.channels["from_parents_to_gravity"].copy()
        pid_dictionary = self._pid_workers
        for parent_key, channel in self._child_channels.items():
            pid = pid_dictionary[parent_key]
            self.resume_workers(pid)
            channel["from_children_to_gravity"].copy()
            self.hibernate_workers(pid)

    def evolve_model(self, tend, timestep=None):
        """
        Evolve the system until tend.
        Args:
            tend (units.time):  Time to simulate till
            timestep (units.time):  Timestep to simulate
        """
        if timestep is None:
            timestep = tend - self.model_time

        evolve_time = self.model_time
        while self.model_time < (evolve_time + timestep) * (1. - EPS):
            self.corr_energy = 0. | units.J
            self.dt_step += 1

            self._sync_grav_to_local()
            if (self.__star_evol):
                self._stellar_evolution(evolve_time + timestep/2.)
                self._star_channel_copier()

            self._drift_global(evolve_time + timestep)
            if self.subcodes:
                self._drift_child(self.model_time)

            self._sync_grav_to_local()
            self._correction_kicks(
                self.particles, 
                self.subsystems,
                self.model_time - evolve_time
                )

            self._sync_local_to_grav()
            if (self.__star_evol):
                self._stellar_evolution(self.model_time)

            self.split_subcodes()
            self.channels["from_parents_to_gravity"].copy()
        gc.collect()

        if self._verbose:
            print(f"Time: {self.model_time.in_(units.Myr)}")
            print(f"Parent code time: {self.parent_code.model_time.in_(units.Myr)}")
            for parent_key, code in self.subcodes.items():
                pid = self._pid_workers[parent_key]
                self.resume_workers(pid)
                total_time = code.model_time + self._time_offsets[code]

                if not abs(total_time - self.model_time)/self.__dt < 0.01:
                    print(f"Parent: {parent_key}, Kids: {code.particles.mass.in_(units.MJupiter)}", end=", ")
                    print(f"Excess simulation: {(total_time - self.model_time).in_(units.kyr)}")

                self.hibernate_workers(pid)
            if self.__star_evol:
                print(f"Stellar code time: {self.stellar_code.model_time.in_(units.Myr)}")
            print(f"#Children: {len(self.subsystems.keys())}")
            print("===" * 50)

    def split_subcodes(self) -> None:
        """Check for any isolated children"""
        if self._verbose:
            print("...Checking Splits...")

        new_isolated = Particles()
        for parent_key, (parent, subsys) in list(self.subsystems.items()):
            host = subsys[subsys.mass.argmax()]
            par_rad = parent.radius
            par_pos = parent.position
            furthest = (subsys.position - host.position).lengths().max()
            criteria = CONNECTED_COEFF * par_rad
            if furthest > criteria / 2.:
                components = connected_components_kdtree(subsys, threshold=criteria)
                if len(components) > 1:
                    if self._verbose:
                        print("...Split Detected...")

                    par_vel = parent.velocity

                    pid = self._pid_workers.pop(parent_key)
                    self.resume_workers(pid)
                    self.particles.remove_particle(parent)
                    code = self.subcodes.pop(parent_key)
                    self._time_offsets.pop(code)
                    self._child_channels.pop(parent_key)
                    for c in components:
                        sys = c.copy()
                        sys.position += par_pos
                        sys.velocity += par_vel

                        if len(sys) > 1 and max(sys.mass) > (0. | units.kg):
                            newparent = self.particles.add_subsystem(sys)
                            newparent_key = newparent.key
                            scale_mass = newparent.mass
                            scale_radius = set_parent_radius(scale_mass)
                            newparent.radius = scale_radius

                            newcode, number_of_workers = self._sub_worker(children=sys,
                                                                          scale_mass=scale_mass,
                                                                          scale_radius=scale_radius)
                            worker_pid = self.get_child_pids(number_of_workers)

                            self.subcodes[newparent_key] = newcode
                            self._time_offsets[newcode] = self.model_time
                            self._child_channel_maker(
                                parent_key=newparent_key,
                                code_particles=newcode.particles,
                                children=sys
                            )
                            self._pid_workers[newparent_key] = worker_pid
                            self.hibernate_workers(worker_pid)

                        else:
                            new_isolated.add_particles(sys)

                    code.cleanup_code()
                    code.stop()

                    del subsys, parent

        if len(new_isolated) > 0:
            new_isolated.radius = set_parent_radius(new_isolated.mass)
            overly_massive = new_isolated.radius > PARENT_RADIUS_MAX
            new_isolated[overly_massive].radius = PARENT_RADIUS_MAX
            self.particles.add_particles(new_isolated)

    def _process_parent_mergers(self) -> None:
        """Process merging of parents from previous timestep in parallel"""
        self.channels["from_gravity_to_parents"].copy()

        nmergers = len(self._new_systems.keys())
        for parent_key, children in self._new_systems.items():
            parent_local = self.particles[self.particles.key == parent_key]
            time_offset = self._new_offsets[parent_key]

            children.position += parent_local.position
            children.velocity += parent_local.velocity

            newparent = self.particles.add_subsystem(children)
            newparent_key = newparent.key
            scale_mass = newparent.mass
            scale_radius = set_parent_radius(scale_mass)
            newparent.radius = scale_radius

            newcode, number_of_workers = self._sub_worker(children=children,
                                                          scale_mass=scale_mass, 
                                                          scale_radius=scale_radius)
            worker_pid = self.get_child_pids(number_of_workers)

            self._time_offsets[newcode] = time_offset
            self.subcodes[newparent_key] = newcode
            self.subsystems[newparent_key] = (newparent, children)
            self._child_channel_maker(
                parent_key=newparent_key, 
                code_particles=newcode.particles, 
                children=children
            )
            self._pid_workers[newparent_key] = worker_pid
            self.hibernate_workers(worker_pid)

            self.particles.remove_particle(parent_local)

        new_parents = self.particles[-nmergers:]
        overly_massive = new_parents.radius > PARENT_RADIUS_MAX
        new_parents[overly_massive].radius = PARENT_RADIUS_MAX

    def _asteroid_merger(self, local_mass, local_ast) -> None:
        """
        Resolve the merging of asteroid with a parent system.        
        Args:
            local_mass (Particle):  Parent particle
            local_ast (Particle):  Asteroid particles
        """
        local_mass_key = local_mass.key
        if local_mass_key not in self.subsystems \
            and local_mass_key not in self._new_systems:
                if self._verbose:
                    print("Merging to lonely parent...")

                mass_pos = local_mass.position
                mass_vel = local_mass.velocity

                system = Particles()
                system.add_particle(local_mass)
                system.add_particle(local_ast)
                system.position -= mass_pos
                system.velocity -= mass_vel

                newparent = Particle()
                newparent.mass = local_mass.mass
                newparent.position = mass_pos
                newparent.velocity = mass_vel
                newparent.radius = local_mass.radius

                self.particles.remove_particle(local_mass)
                self.particles.add_particle(newparent)

                self._new_systems[newparent.key] = system
                self._new_offsets[newparent.key] = self.model_time
                        
        elif local_mass_key in self.subsystems:
            if self._verbose:
                print("Subcode exists; no previous massive merger...")
            
            local_ast.position -= local_mass.position
            local_ast.velocity -= local_mass.velocity

            system = self.subsystems[local_mass_key][1]
            pid = self._pid_workers[local_mass_key]
            self.resume_workers(pid)
            code = self.subcodes[local_mass_key]
            system.add_particle(local_ast)
            code.particles.add_particle(local_ast)
            self.hibernate_workers(pid)
            
        elif local_mass_key in self._new_systems:
            if self._verbose:
                print("Previous merger occured...")

            local_ast.position -= local_mass.position
            local_ast.velocity -= local_mass.velocity

            system = self._new_systems[local_mass_key]
            system.add_particle(local_ast)

        else:
            print("Curious parent merger?")

    def _massive_merger(self, coll_time, coll_set: Particles) -> None:
        """
        Resolve the merging of two massive parents.
        Args:
            coll_time (units.time):  Time of collision
            coll_set (Particles):  Colliding particle set
        """
        collsubset = self._evolve_coll_offset(coll_set, coll_time)
        newparts = Particles()
        for particle in collsubset:
            local_parent = particle.as_particle_in_set(self.particles)
            local_parent_key = local_parent.key
            if local_parent_key in self.subcodes:
                pid = self._pid_workers.pop(local_parent_key)
                self.resume_workers(pid)

                code = self.subcodes.pop(local_parent_key)
                self._time_offsets.pop(code)
                self._child_channels.pop(local_parent_key)
                system = self.subsystems.pop(local_parent_key)

                sys = system[1]
                sys.position += local_parent.position
                sys.velocity += local_parent.velocity

                newparts.add_particles(sys)

                code.cleanup_code()
                code.stop()

            else:
                new_particle = newparts.add_particle(local_parent)
                if new_particle.mass < MIN_EVOL_MASS:
                    new_particle.radius = planet_radius(new_particle.mass)
                else:
                    new_particle.radius = ZAMS_radius(new_particle.mass)

            self.particles.remove_particle(local_parent)

        # Temporary new parent particle
        newparent = Particle()
        newparent.mass = collsubset.total_mass()
        newparent.radius = set_parent_radius(newparent.mass)
        if newparent.radius > PARENT_RADIUS_MAX:
            newparent.radius = PARENT_RADIUS_MAX
        newparent.position = collsubset.center_of_mass()
        newparent.velocity = collsubset.center_of_mass_velocity()
        self.particles.add_particle(newparent)
        newparts.move_to_center()

        self._new_systems[newparent.key] = newparts
        self._new_offsets[newparent.key] = self.model_time

    def _parent_merger(self, coll_time, coll_set: Particles):
        """
        Resolve the merging of two parent systems.
        Args:
            coll_time (units.time):  Time of collision
            coll_set (Particles):  Colliding particle set
        """
        self.channels["from_gravity_to_parents"].copy()

        # Ugly workaround
        colliders = Particles()
        for collider in coll_set:
            colliders.add_particle(collider)

        masses = colliders.mass
        try:
            if min(masses) == (0. | units.kg):
                if self._verbose:
                    print(f"... Asteroid merger @ T={coll_time.value_in(units.Myr):.6f}", end=" Myr: ")

                asteroid = colliders[masses.argmin()]
                massive = colliders[masses.argmax()]
                local_mass = massive.as_particle_in_set(self.particles)
                local_ast = asteroid.as_particle_in_set(self.particles)
                local_ast.radius = ASTEROID_RADIUS

                self._asteroid_merger(local_mass, local_ast)
                self.particles.remove_particle(local_ast)

            else:
                if self._verbose:
                    print(f"... Massive merger:@ T={coll_time.value_in(units.Myr):.6f} Myr")
                self._massive_merger(coll_time, coll_set)

        except Exception as e:
            print(f"Error while merging {colliders}: {e}")
            print("Traceback:", traceback.format_exc())
            exit(-1)

    def _evolve_coll_offset(self, coll_set: Particles, coll_time) -> list:
        """
        Function to evolve and/or resync the final moments of collision.
        Args:
            coll_set (Particles):  Attributes of colliding particle
            coll_time (units.time):  Time of simulation where collision occurs
        Returns:
            List: Index 0 contains parent colliders, index 1 childrens of merging parents
        """
        collsubset = Particles()
        for particle in coll_set:
            particle_key = particle.key
            collsubset.add_particle(particle)

            # Check if this parent already merged in this time loop
            code = None
            if particle_key in self._new_systems:
                if self._verbose:
                    print("...Parent Merging Again...")

                collsubset.remove_particle(particle)
                evolved_parent = particle.as_particle_in_set(self.particles)

                offset = self._new_offsets.pop(particle_key)
                children = self._new_systems.pop(particle_key)
                children.position += evolved_parent.position
                children.velocity += evolved_parent.velocity
                
                newparent = self.particles.add_subsystem(children)
                newparent_key = newparent.key
                scale_mass = newparent.mass
                scale_radius = set_parent_radius(scale_mass)
                newparent.radius = scale_radius
                newcode, number_of_workers = self._sub_worker(children=children, 
                                                              scale_mass=scale_mass, 
                                                              scale_radius=scale_radius)
                newcode.particles.move_to_center()

                self.subcodes[newparent_key] = newcode
                self.subsystems[newparent_key] = (newparent, children)
                self._time_offsets[newcode] = offset
                self._child_channel_maker(
                    parent_key=newparent_key, 
                    code_particles=newcode.particles, 
                    children=children
                )

                worker_pid = self.get_child_pids(number_of_workers)
                self._pid_workers[newparent_key] = worker_pid
                self.particles.remove_particle(particle)
                collsubset.add_particle(newparent)
                particle = newparent  # Redefine parent for further processing

                code = newcode

            elif particle_key in self.subcodes:
                worker_pid = self._pid_workers[particle_key]
                self.resume_workers(worker_pid)

                code = self.subcodes[particle_key]
                offset = self._time_offsets[code]
                newparent = particle.copy()

            if (code):
                if self._verbose:
                    dt = (coll_time - offset - code.model_time)
                    print(f"Evolving {len(code.particles)} for: {dt.in_(units.kyr)}")

                stopping_condition = code.stopping_conditions.collision_detection
                stopping_condition.enable()
                while code.model_time < (coll_time - offset) * (1. - EPS):
                    code.evolve_model(coll_time - offset)

                    if stopping_condition.is_set():
                        collsubset.remove_particle(newparent)

                        coll_a_particles = stopping_condition.particles(0)
                        coll_b_particles = stopping_condition.particles(1)
                        Nmergers = max(len(np.unique(coll_a_particles.key)),  
                                       len(np.unique(coll_b_particles.key)))
                        
                        resolved_keys = dict()
                        Nresolved = 0
                        for coll_a, coll_b in zip(coll_a_particles, coll_b_particles):
                            if Nresolved < Nmergers:  # Stop recursive loop
                                particle_dict = {p.key: p for p in code.particles}
                                if coll_a.key in resolved_keys.keys():
                                    coll_a = particle_dict.get(resolved_keys[coll_a.key])
                                if coll_b.key in resolved_keys:
                                    coll_b = particle_dict.get(resolved_keys[coll_b.key])

                                if coll_b.key == coll_a.key:
                                    print("Curious?")
                                    continue

                                children = self.subsystems[newparent.key][1]
                                colliding_particles = Particles(particles=[coll_a, coll_b])
                                newparent, resolved_keys = self._handle_collision(
                                                                children, newparent, 
                                                                colliding_particles, 
                                                                code, resolved_keys
                                                                )
                                Nresolved += 1

                        collsubset.add_particle(newparent)

                if len(code.particles) > 1:
                    self._child_channels[newparent.key]["from_gravity_to_children"].copy()
                    self.hibernate_workers(self._pid_workers[newparent.key])

                else:  #  Require dereferencing subsys as well?
                    code = self.subcodes.pop(newparent.key)
                    self._time_offsets.pop(code)
                    self._child_channels.pop(newparent.key)
                    self._pid_workers.pop(newparent.key)
                    self.subsystems.pop(newparent.key)

                    code.stop()
                    del code
                
        return collsubset
    
    def _handle_collision(self, children: Particles, parent: Particle, enc_parti: Particles, code, resolved_keys: dict):
        """
        Merge two particles if the collision stopping condition is met
        Args:
            children (Particles):  The children particle set
            parent (Particle):  The parent particle
            enc_parti (Particles): The particles in the collision
            code (Code):  The integrator used
            resolved_keys (dict):  Dictionary holding {Collider i Key: Remnant Key}
        Returns:
            Particles:  New parent particle alongside dictionary of merging particles keys
        """
        # Save properties
        self.__nmerge += 1
        print(f"...Collision #{self.__nmerge} Detected...")
        parent_key = parent.key
        self._child_channels[parent_key]["from_gravity_to_children"].copy()
        
        coll_a = enc_parti[0].as_particle_in_set(children)
        coll_b = enc_parti[1].as_particle_in_set(children)
        collider = Particles(particles=[coll_a, coll_b])
        kepler_elements = orbital_elements(collider, G=constants.G)
        sma = kepler_elements[2]
        ecc = kepler_elements[3]
        inc = kepler_elements[4]
        
        tcoll = code.model_time + self._time_offsets[code] + self.__resume_offset
        with open(os.path.join(self.__coll_dir, f"merger{self.__nmerge}.txt"), 'w') as f:
            f.write(f"Tcoll: {tcoll.in_(units.yr)}")
            f.write(f"\nParent Key: {parent.key}")
            f.write(f"\nKey1: {enc_parti[0].key}")
            f.write(f"\nKey2: {enc_parti[1].key}")
            f.write(f"\nType1: {coll_a.type}")
            f.write(f"\nType2: {coll_b.type}")
            f.write(f"\nM1: {enc_parti[0].mass.in_(units.MSun)}")
            f.write(f"\nM2: {enc_parti[1].mass.in_(units.MSun)}")
            f.write(f"\nSemi-major axis: {abs(sma).in_(units.au)}")
            f.write(f"\nEccentricity: {ecc}")
            f.write(f"\nInclination: {inc.in_(units.deg)}")
        
        if not self.__merged:
            write_set_to_file(
                self.particles.all().savepoint(0 | units.Myr), 
                os.path.join(self.__coll_dir, f"Cluster_Merger{self.__nmerge}.hdf5"),
                'amuse', close_file=True, overwrite_file=True
            )
            self.__merged = True
        
        # Create merger remnant
        most_massive = collider[collider.mass.argmax()]
        collider_mass = collider.mass
        if min(collider_mass) == (0. | units.kg):
            remnant = Particles(particles=[most_massive])

        elif max(collider_mass) > (0 | units.kg):
            remnant  = Particles(1)
            remnant.mass = collider.total_mass()
            remnant.position = collider.center_of_mass()
            remnant.velocity = collider.center_of_mass_velocity()

            if remnant.mass > MIN_EVOL_MASS:
                remnant.radius = ZAMS_radius(remnant.mass)
                if self.__star_evol:
                    self.stellar_code.particles.add_particle(remnant)
                    self.stars.add_particle(remnant)

                    if coll_a.mass > MIN_EVOL_MASS:
                        self.stellar_code.particles.remove_particle(coll_a)
                        self.stars.remove_particle(coll_a)

                    if coll_b.mass > MIN_EVOL_MASS:
                        self.stellar_code.particles.remove_particle(coll_b)
                        self.stars.remove_particle(coll_b)

            else:
                remnant.radius = planet_radius(remnant.mass)

        else:
            raise ValueError("Error: Asteroid - Asteroid collision")
        
        print(f"{coll_a.type}, {coll_b.type}")
        print(f"{coll_a.mass.in_(units.MSun)} + {coll_b.mass.in_(units.MSun)} --> {remnant.mass.in_(units.MSun)}")
        print(f"{coll_a.radius.in_(units.RSun)} + {coll_b.radius.in_(units.RSun)} --> {remnant.radius.in_(units.RSun)}")

        remnant.coll_events = max(collider.coll_events) + 1
        remnant.type = most_massive.type
        remnant.original_key = most_massive.original_key
        
        # Deal with simultaneous mergers
        changes = [ ]
        coll_a_change = 0
        coll_b_change = 0
        if not resolved_keys:
            resolved_keys[coll_a.key] = remnant.key[0]
            resolved_keys[coll_b.key] = remnant.key[0]
        else: 
            # If the current collider is a remnant of past event, remap
            for prev_collider, resulting_remnant in resolved_keys.items():
                if coll_a.key == resulting_remnant:  
                    changes.append((prev_collider, remnant.key[0]))
                    coll_a_change = 1
                elif coll_b.key == resulting_remnant:
                    changes.append((prev_collider, remnant.key[0]))
                    coll_b_change = 1
                    
            if coll_a_change == 0:
                resolved_keys[coll_a.key] = remnant.key[0]
            if coll_b_change == 0:
                resolved_keys[coll_b.key] = remnant.key[0]
       
        for key, new_value in changes:
            resolved_keys[key] = new_value

        children.remove_particle(coll_a)
        children.remove_particle(coll_b)
        children.add_particles(remnant)
        
        if min(collider_mass) > (0. | units.kg):
            nearest_mass = abs(children.mass - parent.mass).argmin()
            if remnant.key == children[nearest_mass].key:  # If the remnant is the host
                children.position += parent.position
                children.velocity += parent.velocity
                
                newparent = self.particles.add_subsystem(children)
                newparent_key = newparent.key
                newparent.radius = parent.radius

                # Re-mapping dictionary to new parent
                old_code = self.subcodes.pop(parent_key)
                old_offset = self._time_offsets.pop(old_code)
                old_channel = self._child_channels.pop(parent_key)
                
                self.subcodes[newparent_key] = old_code
                new_code = self.subcodes[newparent_key]
                self._time_offsets[new_code] = old_offset
                self._child_channel_maker(
                    parent_key=newparent_key, 
                    code_particles=new_code.particles, 
                    children=children
                    )
                child_pid = self._pid_workers.pop(parent_key)
                self._pid_workers[newparent_key] = child_pid
                
                self.particles.remove_particle(parent)
                
                del old_channel  # Check if this breaks

            else:
                newparent = parent
        else:
            newparent = parent
            
        children.synchronize_to(self.subcodes[newparent.key].particles)
          
        return newparent, resolved_keys
    
    def _handle_supernova(self, SN_detect, bodies: Particles) -> None:
        """Handle SN events"""
        if self.__dE_track:
            E0 = self.calculate_total_energy()
            
        SN_particle = SN_detect.particles(0)
        for ci in range(len(SN_particle)):
            SN_parti = Particles(particles=SN_particle)
            natal_kick_x = SN_parti.natal_kick_x
            natal_kick_y = SN_parti.natal_kick_y
            natal_kick_z = SN_parti.natal_kick_z
            
            SN_parti = SN_parti.get_intersecting_subset_in(bodies)
            SN_parti.vx += natal_kick_x
            SN_parti.vy += natal_kick_y
            SN_parti.vz += natal_kick_z
            
        if self.__dE_track:
            E1 = self.calculate_total_energy()
            self.corr_energy += E1 - E0

    def _find_coll_sets(self, p1: Particle, p2: Particle) -> UnionFind:
        """
        Find encountering particle sets
        Args:
            p1 (Particle):  Particle a of merger
            p2 (Particle):  Particle b of merger
        Returns:
            UnionFind: Set of colliding particles
        """
        coll_sets = UnionFind()
        for p,q in zip(p1, p2):
            coll_sets.union(p, q)
            
        return coll_sets.sets()
    
    def _stellar_evolution(self, dt) -> None:
        """
        Evolve stellar evolution.
        Args:
            dt (units.time):  Time to evolve till
        """
        while self.stellar_code.model_time < dt * (1. - EPS):
            self.stellar_code.evolve_model(dt)
            
            if self.SN_detection.is_set():
                print("...Detection: SN Explosion...")
                self._handle_supernova(self.SN_detection, self.stars)
                    
    def _drift_global(self, dt) -> None:
        """
        Evolve parent system until dt.
        Args:
            dt (units.time):  Time to evolve till
        """
        if self._verbose:
            print("...Drifting Global...")
        
        self.__merged = False
        coll_time = None
        while self._evolve_code.model_time < dt * (1. - EPS):
            self._evolve_code.evolve_model(dt)
            
            if self.grav_coll.is_set():
                if self.__dE_track:
                    E0 = self.calculate_total_energy()
                coll_time = self.parent_code.model_time
                coll_sets = self._find_coll_sets(
                                self.grav_coll.particles(0), 
                                self.grav_coll.particles(1)
                                )
                for cs in coll_sets:
                    self._parent_merger(coll_time, cs)
                if self.__dE_track:
                    E1 = self.calculate_total_energy()
                    self.corr_energy += E1 - E0

        if (self.__gal_field):
            while self.parent_code.model_time < dt * (1. - EPS):
                self.parent_code.evolve_model(dt)
                if self.grav_coll.is_set():
                    if self.__dE_track:
                        E0 = self.calculate_total_energy()
                    
                    coll_time = self.parent_code.model_time
                    coll_sets = self._find_coll_sets(
                                    self.grav_coll.particles(0), 
                                    self.grav_coll.particles(1)
                                    )
                    for cs in coll_sets:
                        self._parent_merger(coll_time, cs)
                    if self.__dE_track:
                        E1 = self.calculate_total_energy()
                        self.corr_energy += E1 - E0

        if coll_time:
            self._process_parent_mergers()
            self._new_offsets.clear()
            self._new_systems.clear()

    def _drift_child(self, dt) -> None:
        """
        Evolve children system until dt.
        Args:
            dt (units.time): Time to evolve till.
        """
        def resolve_collisions(code, parent: Particle, children: Particles, stopping_condition):
            """
            Function to resolve collisions
            Args:
                code (Code):  Code with collision
                parent (Particle):  Parent particle
                stopping_condition (StoppingCondition):  Stopping condition to resolve
            """
            coll_a_particles = stopping_condition.particles(0)
            coll_b_particles = stopping_condition.particles(1)
                        
            resolved_keys = dict()
            Nmergers = max(len(np.unique(coll_a_particles.key)),
                           len(np.unique(coll_b_particles.key)))
            Nresolved = 0
            for coll_a, coll_b in zip(coll_a_particles, coll_b_particles):
                if Nresolved < Nmergers:  # Stop recursive loop
                    particle_dict = {p.key: p for p in code.particles}
                    if coll_a.key in resolved_keys.keys():
                        coll_a = particle_dict.get(resolved_keys[coll_a.key])
                    if coll_b.key in resolved_keys:
                        coll_b = particle_dict.get(resolved_keys[coll_b.key])

                    if coll_b.key == coll_a.key:
                        print("Curious?")
                        continue

                    colliding_particles = Particles(particles=[coll_a, coll_b])
                    parent, resolved_keys = self._handle_collision(
                                                children, parent, 
                                                colliding_particles, 
                                                code, resolved_keys
                                                )
                    Nresolved += 1

            del resolved_keys

            return parent

        def evolve_code(parent_key: Particle):
            """
            Evolve children code until dt
            Args:
                parent_key (int):  Parent particle key
            """
            self.resume_workers(self._pid_workers[parent_key])
            code = self.subcodes[parent_key]
            parent = self.subsystems[parent_key][0]
            children = self.subsystems[parent_key][1]
            stopping_condition = code.stopping_conditions.collision_detection
            stopping_condition.enable()

            evol_time = dt - self._time_offsets[code]
            while code.model_time < evol_time * (1. - EPS):
                code.evolve_model(evol_time)

                if stopping_condition.is_set():
                    with self._lock:
                        if self.__dE_track:
                            KE = code.particles.kinetic_energy()
                            PE = code.particles.potential_energy()
                            E0 = KE + PE

                        parent = resolve_collisions(
                                    code, parent, 
                                    children,
                                    stopping_condition
                                    )

                        if self.__dE_track:
                            KE = code.particles.kinetic_energy()
                            PE = code.particles.potential_energy()
                            E1 = KE + PE
                            self.corr_energy += E1 - E0

            self.hibernate_workers(self._pid_workers[parent.key])

        if self._verbose:
            print("...Drifting Children...")
        self.__merged = False
        with ThreadPoolExecutor(max_workers=self.avail_cpus) as executor:
            futures = {
                executor.submit(evolve_code, parent_key): parent_key 
                for parent_key in self.subcodes.keys()
            }
            for future in as_completed(futures):  # Iterate over to ensure no silent failures
                parent_key = futures[future]
                try:
                    future.result()
                except Exception as e:
                    print(f"Error while evolving parent {parent_key}: {e}")
                    write_set_to_file(
                        self.subsystems[parent_key][1], 
                        "temp/error.hdf5", 'amuse', 
                        close_file=True, 
                        overwrite_file=True
                        )
                    exit(-1)

        for parent_key in list(self.subcodes.keys()):  # Remove single children systems:
            pid = self._pid_workers[parent_key]
            self.resume_workers(pid)
            if len(self.subcodes[parent_key].particles) == 1:
                old_subcode = self.subcodes.pop(parent_key)
                self._time_offsets.pop(old_subcode)
                self._child_channels.pop(parent_key)
                self._pid_workers.pop(parent_key)

                old_subcode.cleanup_code()
                old_subcode.stop()

                del old_subcode

            else:
                self.hibernate_workers(pid)

    def _kick_particles(self, particles: Particles, corr_code, dt) -> None:
        """
        Apply correction kicks onto target particles
        Args:
            particles (Particles):  Particles whose accelerations are corrected
            corr_code (Code):  Object providing the difference in gravity
            dt (units.time):  Time-step of correction kick
        """
        ax, ay, az = corr_code.get_gravity_at_point(
                        particles.radius,
                        particles.x, 
                        particles.y, 
                        particles.z
                        )

        particles.velocity[:,0] += dt * ax
        particles.velocity[:,1] += dt * ay
        particles.velocity[:,2] += dt * az

    def _correct_children(self, perturber_mass, perturber_x, perturber_y, perturber_z,
                          parent_x, parent_y, parent_z, subsystem: Particles, dt) -> None:
        """
        Apply correcting kicks onto children particles.
        Args:
            perturber_mass (units.mass):  Mass of perturber
            perturber_x (units.length):  X-position of perturber
            perturber_y (units.length):  Y-position of perturber
            perturber_z (units.length):  Z-position of perturber
            parent_x (units.length):  X-position of parent
            parent_y (units.length):  Y-position of parent
            parent_z (units.length):  Z-position of parent
            subsystem (Particles):  Children particle set
            dt (units.time):  Time interval for applying kicks
        """
        subsystem_x = subsystem.x + parent_x
        subsystem_y = subsystem.y + parent_y
        subsystem_z = subsystem.z + parent_z
        
        corr_par = CorrectionForCompoundParticle(
                        grav_lib=self.lib,
                        parent_x=parent_x,
                        parent_y=parent_y,
                        parent_z=parent_z, 
                        system=subsystem,
                        system_x=subsystem_x,
                        system_y=subsystem_y,
                        system_z=subsystem_z, 
                        perturber_mass=perturber_mass,
                        perturber_x=perturber_x,
                        perturber_y=perturber_y,
                        perturber_z=perturber_z,
                        )
        self._kick_particles(subsystem, corr_par, dt)
       
    def _correction_kicks(self, particles: Particles, subsystems: dict, dt) -> None:
        """
        Apply correcting kicks onto children and parent particles
        Args:
            particles (Particles):  Parent particle set
            subsystems (dict):  Dictionary of children system
            dt (units.time):  Time interval for applying kicks
        """
        def process_children_jobs(parent, children):
            removed_idx = abs(particles_mass - parent.mass).argmin()
            mask = np.ones(len(particles_mass), dtype=bool)
            mask[removed_idx] = False
            
            pert_mass = particles_mass[mask]
            pert_xpos = particles_x[mask]
            pert_ypos = particles_y[mask]
            pert_zpos = particles_z[mask]

            future = executor.submit(
                        self._correct_children,
                        perturber_mass=pert_mass,
                        perturber_x=pert_xpos,
                        perturber_y=pert_ypos,
                        perturber_z=pert_zpos,
                        parent_x=parent.x,
                        parent_y=parent.y,
                        parent_z=parent.z,
                        subsystem=children,
                        dt=dt
                        )

            return future
        
        if subsystems and len(particles) > 1:
            # Setup array for CorrectionFor
            particles_mass = particles.mass
            particles_x = particles.x
            particles_y = particles.y
            particles_z = particles.z
            
            corr_chd = CorrectionFromCompoundParticle(
                            grav_lib=self.lib,
                            particles=particles,
                            particles_x=particles_x,
                            particles_y=particles_y,
                            particles_z=particles_z,
                            subsystems=subsystems,
                            num_of_workers=self.avail_cpus
                            )
            self._kick_particles(particles, corr_chd, dt)

            futures = []
            with ThreadPoolExecutor(max_workers=self.avail_cpus) as executor:
                for parent, children in subsystems.values():
                    try:
                        future = process_children_jobs(parent, children)
                        futures.append(future)
                    except Exception as e:
                        print(f"Error submitting job for parent {parent.key}: {e}")
                        print(subsystems[parent.key][1])
                        print(f"Traceback: {traceback.format_exc()}")
                        exit(-1)

                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:
                        print(f"Error in CorrectionFor result: {e}")
                        print(f"Traceback: {traceback.format_exc()}")
                        exit(-1)
               
    @property
    def model_time(self) -> float:  
        """Extract the global integrator model time"""
        return self.parent_code.model_time

    @property
    def subsystems(self) -> dict:
        """Extract the children system"""
        return self.particles.collection_attributes.subsystems

    @property
    def avail_cpus(self) -> int:
        """
        Extract the number of available CPUs, computed by considering: 
            - Number of parent workers
            - One SeBa worker
            - One worker for main process
            - Leave one free for good measure
        """
        ncpu = min(len(self.subsystems), self.__total_free_cpus - self.__par_nworker - 3)
        if ncpu < 1:
            return 1
        return ncpu