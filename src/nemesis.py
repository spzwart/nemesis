############################################# NOTES ###############################################################
####################################################################################################################
####################################################################################################################
# 1. Although split_subcodes can be parallelised, the benefit is very little and complexity
#    outweighs it. No use in putting in lower-level language given dictionary look ups.
####################################  OTHER ROOM FOR IMPROVEMENT  #################################################
# 1. Flexible bridge times.
# 2. Flexible parent radius.
# 3. Use old workers instead of spawning new ones on splitting.
# 4. In _handle_collision(), the condition if remnant.key == children[nearest_mass].key 
#    could be removed entirely. But needs testing to ensure no bugs.
# 5. Logic of split_subcodes. Main bottleneck is in spawning codes, but parallelising is made difficult due to
#    needing PID management. Could be significantly improved if a technique identifying which thread has which PID.
# 6. When removing single particle children in _drift_child(), could be prone to error (Not encountered in tests).
# 7. In _process_parent_mergers(), recycling old codes would be more efficient.
####################################################################################################################
####################################################################################################################
####################################################################################################################

 
from concurrent.futures import ThreadPoolExecutor, as_completed
import ctypes
import gc
import numpy as np
from numpy.ctypeslib import ndpointer
import os
import psutil
import signal
import threading
import time
import traceback
import sys

from amuse.community.huayno.interface import Huayno
from amuse.community.ph4.interface import Ph4
from amuse.community.seba.interface import SeBa
from amuse.community.symple.interface import Symple

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
    def __init__(
            self, par_conv, dtbridge, coll_dir, 
            available_cpus=os.cpu_count(), nmerge=0,
            resume_time=0. | units.yr, code_dt=0.03, 
            dE_track=False, star_evol=False, 
            gal_field=True, verbose=True
            ):
        """
        Class setting up the simulation.

        Args:
            par_conv (converter):      Parent N-body converter
            dtbridge (units.time):     Diagnostic time step
            coll_dir (str):            Path to store collision data
            available_cpus(int):       Number of available CPUs
            nmerge (int):              Number of mergers loaded particle set has
            resume_time (units.time):  Time which simulation is resumed at
            code_dt (float):           Internal time step
            dE_track (bool):           Flag turning on/off energy error tracker
            star_evol (bool):          Flag turning on/off stellar evolution
            gal_field (bool):          Flag turning on/off galactic field
            verbose (bool):            Flag turning on/off verbose output
        """
        # Private attributes
        self.__dt = dtbridge
        self.__coll_dir = coll_dir
        self.__code_dt = code_dt
        self.__gal_field = gal_field
        self.__lock = threading.Lock()
        self.__main_process = psutil.Process(os.getpid())
        self.__nmerge = nmerge
        self.__par_nworker = PARENT_NWORKER
        self.__resume_offset = resume_time
        self.__star_evol = star_evol
        self.__total_free_cpus = available_cpus

        # Protected attributes
        self._verbose = verbose
        self._parent_conv = par_conv
        self._MWG = MWpotentialBovy2015()
        self._coll_parents = dict()
        self._coll_children = dict()
        self._isolated_mergers = dict()
        
        # Children dictionaries
        self._child_channels = dict()
        self._cpu_time = dict()
        self._pid_workers = dict()
        self.subcodes = dict()
        self._time_offsets = dict()

        self.parent_code = self._parent_worker()
        self.grav_coll = self.parent_code.stopping_conditions.collision_detection
        self.grav_coll.enable()
        if (self.__star_evol):
            self.stellar_code = self._stellar_worker()
            self.SN_detection = self.stellar_code.stopping_conditions.supernova_detection
            self.SN_detection.enable()
        self.particles = HierarchicalParticles(self.parent_code.particles)
        self.dt_step = 0
        self.dE_track = dE_track

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
    
    def cleanup_code(self, first_clean=1) -> None:
        """
        Cleanup all codes and processes
        Args:
            first_clean (bool):  If true, cleans parent and stellar codes
        """
        if self._verbose:
            print("...Cleaning up Nemesis...")

        if first_clean:
            self.parent_code.cleanup_code()
            self.parent_code.stop()
            if (self.__star_evol):
                self.stellar_code.cleanup_code()
                self.stellar_code.stop()

        for parent_key, code in self.subcodes.items():
            pid = self._pid_workers[parent_key]
            self.resume_workers(pid)
            code.cleanup_code()
            code.stop()

        gc.collect()
        if self._verbose:
            print("...Nemesis cleaned up...")
    
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
            parent.radius = min(PARENT_RADIUS_MAX, scale_radius)
            if parent not in self.subcodes:
                code, child_PID = self._sub_worker(
                    children=sys, 
                    scale_mass=scale_mass, 
                    scale_radius=scale_radius
                    )

                self._set_worker_affinity(child_PID)
                self._time_offsets[code] = self.model_time
                self.subsystems[parent_key] = (parent, sys)
                self.subcodes[parent_key] = code
                self._child_channel_maker(
                    parent_key=parent_key, 
                    code_particles=code.particles, 
                    children=sys
                )
                self._cpu_time[parent_key] = 0

                # Store children PID to allow hibernation
                self._pid_workers[parent_key] = child_PID
                self.hibernate_workers(child_PID)

            if self._verbose:
                print(f"System {nsyst+1}/{ntotal}, radius: {parent.radius.in_(units.au)}")

        self.particles.recenter_subsystems(max_workers=self.avail_cpus)
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
        code.parameters.force_sync = True
        return code

    def _sub_worker(self, children: Particles, scale_mass, scale_radius, number_of_workers=1):
        """
        Initialise children integrator.
        Args:
            children (Particles):         Children systems
            scale_mass (units.mass):      Mass of the system
            scale_radius (units.length):  Radius of the system
            number_of_workers (int):      Number of workers to use
        Returns:
            Code:  Gravitational integrator with particle set
        """
        if len(children) == 0:
            self.cleanup_code()
            print("Error: No children provided.")
            print(f"Traceback: {traceback.format_exc()}")
            sys.exit()

        converter = nbody_system.nbody_to_si(scale_mass, scale_radius)
        PIDs_before = self._snapshot_worker_pids()
        code = Huayno(
            converter, 
            number_of_workers=number_of_workers, 
            channel_type="sockets"
            )
        code.particles.add_particles(children)
        code.parameters.epsilon_squared = (0. | units.au)**2.
        code.parameters.timestep_parameter = self.__code_dt
        code.set_integrator("SHARED8_COLLISIONS")

        PIDs_after = self._snapshot_worker_pids()
        worker_PID = list(PIDs_after - PIDs_before)

        return code, worker_PID
    
    def _set_worker_affinity(self, pid_list):
        """Ensure child workers have access to all visible CPUs."""
        try:
            ncpu = os.cpu_count()
            if ncpu is None:
                return
            all_cores = list(range(ncpu))

            for pid in pid_list:
                p = psutil.Process(pid)
                p.cpu_affinity(all_cores)

        except Exception as e:
            if self._verbose:
                print(f"Warning: could not set affinity for workers {pid_list}: {e}")

    def _snapshot_worker_pids(self) -> set[int]:
        """Return the set of PIDs of all children workers"""
        pids = set()
        for child in self.__main_process.children(recursive=True):
            try:
                name = child.name()
            except Exception:
                self.cleanup_code()
                error = f"Error extracting PID from child process: {child.name()} \n" \
                         "Check if child worker name matches expected name in get_child_pids."
                print(error)
                print(f"Traceback: {traceback.format_exc()}")
                sys.exit()
            if "huayno_worker" in name \
                or "ph4_worker" in name or \
                    "symple_worker" in name:
                pids.add(child.pid)
        return pids

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
                self.cleanup_code()
                print(f"Warning: Process {pid} not found. It may have exited.")
                print(f"Traceback: {traceback.format_exc()}")
                sys.exit()
            except PermissionError:
                self.cleanup_code()
                print(f"Error: Insufficient permissions to stop process {pid}.")
                print(f"Traceback: {traceback.format_exc()}")
                sys.exit()

    def resume_workers(self, pid_list: list) -> None:
        """
        Resume workers to continue simulation.

        Args:
            pid_list (list):  List of process IDs for worker
        """
        for pid in pid_list:
            try:
                os.kill(pid, signal.SIGCONT)
            except ProcessLookupError:
                print(f"Warning: Process {pid} not found. It may have exited.")
                print(f"Traceback: {traceback.format_exc()}")
                self.cleanup_code()
                sys.exit()
            except PermissionError:
                print(f"Error: Insufficient permissions to stop process {pid}.")
                print(f"Traceback: {traceback.format_exc()}")
                self.cleanup_code()
                sys.exit()

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

    def _child_channel_maker(
            self, parent_key: int, 
            code_particles: Particles, 
            children: Particles
            ) -> None:
        """
        Create communication channel between codes and specified children system

        Args:
            parent_key (int):            Parent particle key
            code_particles (Particles):  Children particle set in grav. code
            children (Particles):        Children particle set in local memory
        """
        grav_copy_variables = [
            "x", "y", "z", 
            "vx", "vy", "vz", 
            "radius", "mass"
        ]

        if self.__star_evol:
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
            tend (units.time):      Time to simulate till
            timestep (units.time):  Time step to simulate
        """
        if timestep is None:
            timestep = tend - self.model_time

        self.evolve_time = self.model_time
        kick_corr = timestep
        while self.model_time < (self.evolve_time + timestep) * (1. - EPS):
            self.corr_energy = 0. | units.J
            self.dt_step += 1

            if self.evolve_time == 0. | units.yr \
                and self.__resume_offset == 0. | units.yr:
                    if self.__star_evol:
                        self._stellar_evolution(timestep/2.)
                        self._star_channel_copier()

                    self._sync_grav_to_local()
                    self._correction_kicks(
                        self.particles, 
                        self.subsystems,
                        dt=timestep/2.
                        )
                    kick_corr = timestep/2.

            self.old_copy = self.particles.copy()
            self._drift_global(
                self.model_time + timestep,
                corr_time=kick_corr
                )
            if self.subcodes:
                self._drift_child(self.model_time)

            self._sync_grav_to_local()
            self._correction_kicks(
                self.particles, 
                self.subsystems,
                dt=timestep
                )
            self._sync_local_to_grav()
            if (self.__star_evol):
                self._stellar_evolution(self.model_time + timestep/2.)
                self._star_channel_copier()
            self.split_subcodes()
            self.channels["from_parents_to_gravity"].copy()

        del self.old_copy
        gc.collect()

        if self._verbose:
            print(f"Time: {self.model_time.in_(units.Myr)}")
            print(f"Parent code time: {self.parent_code.model_time.in_(units.Myr)}")
            Nkids = 0
            for parent_key, code in self.subcodes.items():
                Nkids += 1
                pid = self._pid_workers[parent_key]
                self.resume_workers(pid)
                total_time = code.model_time + self._time_offsets[code]

                if not abs(total_time - self.model_time)/self.__dt < 0.01:
                    print(f"Parent: {parent_key}, # Children: {len(code.particles)}", end=", ")
                    print(f"Excess simulation: {(total_time - self.model_time).in_(units.kyr)}")

                self.hibernate_workers(pid)
            if self.__star_evol:
                print(f"Stellar code time: {self.stellar_code.model_time.in_(units.Myr)}")
            print(f"#Children: {Nkids}")
            print("===" * 50)

    def split_subcodes(self) -> None:
        """Check for any isolated children"""
        def get_linking_info(radius, subsys):
            components = connected_components_kdtree(
                system=subsys, 
                threshold=CONNECTED_COEFF/2.*radius
                )
            return components
        
        if self._verbose:
            print("...Checking Splits...")    
        if self.dE_track:
            if self.model_time == 0. | units.yr:
                self.corr_energy = 0. | units.J
            E0 = self.calculate_total_energy()

        new_pids = [ ]
        new_isolated = Particles()
        to_process = [ ]
        with ThreadPoolExecutor(max_workers=max(1, self.avail_cpus//2)) as executor:
            futures = {
                executor.submit(get_linking_info, parent.radius, children) : parent_key
                for parent_key, (parent, children) in self.subsystems.items()
            }
            for future in as_completed(futures):
                parent_key = futures[future]
                try:
                    components = future.result()
                    if len(components) <= 1:
                        continue
                    to_process.append((parent_key, components))
                except Exception as e:
                    self.cleanup_code()
                    print(f"Error during split detection for parent {parent_key}: {e}")
                    print(f"Traceback: {traceback.format_exc()}")
                    sys.exit()

        for parent_key, components in to_process:
            if self._verbose:
                print("...Split Detected...")
            rework_code = False

            parent, _ = self.subsystems[parent_key]
            par_pos = parent.position
            par_vel = parent.velocity

            pid = self._pid_workers.pop(parent_key)
            self.resume_workers(pid)
            self.particles.remove_particle(parent)

            code = self.subcodes.pop(parent_key)
            offset = self._time_offsets.pop(code)
            self._child_channels.pop(parent_key)
            cpu_time = self._cpu_time.pop(parent_key)
            for c in components:
                sys = c.as_set()
                sys.position += par_pos
                sys.velocity += par_vel

                if len(sys) > 1 and max(sys.mass) > (0. | units.kg):
                    newparent = self.particles.add_subsystem(sys)
                    newparent_key = newparent.key
                    scale_mass = newparent.mass
                    scale_radius = set_parent_radius(scale_mass)
                    newparent.radius = scale_radius
                    if not rework_code:  # Recycle old code
                        rework_code = True
                        newcode = code
                        newcode.particles.remove_particles(code.particles)
                        newcode.particles.add_particles(sys)
                        self._time_offsets[newcode] = offset
                        worker_pid = pid
                        
                    else:  # Could be optimised if done in parallel.
                        newcode, worker_pid = self._sub_worker(
                            children=sys,
                            scale_mass=scale_mass,
                            scale_radius=scale_radius
                            )
                        self._set_worker_affinity(worker_pid)
                        self._time_offsets[newcode] = self.model_time
                    
                    self._cpu_time[newparent_key] = cpu_time
                    self.subcodes[newparent_key] = newcode

                    self._child_channel_maker(
                        parent_key=newparent_key,
                        code_particles=newcode.particles,
                        children=sys
                    )
                    self._child_channels[newparent_key]["from_children_to_gravity"].copy()  # More precise
                    
                    self._pid_workers[newparent_key] = worker_pid
                    new_pids.append(worker_pid)
                    if len(new_pids) > int(self.avail_cpus // 2):
                        for pid in new_pids:
                            self.hibernate_workers(pid)
                        new_pids.clear()

                else:
                    new_isolated.add_particles(sys)
            
            if not rework_code:  # Only triggered if pure ionisation
                code.cleanup_code()
                code.stop()

        for pid in new_pids:
            self.hibernate_workers(pid)
        new_pids.clear()

        if len(new_isolated) > 0:
            new_isolated.radius = set_parent_radius(new_isolated.mass)
            mask = new_isolated.radius > PARENT_RADIUS_MAX
            new_isolated[mask].radius = PARENT_RADIUS_MAX
            self.particles.add_particles(new_isolated)

        if self.dE_track:
            E1 = self.calculate_total_energy()
            self.corr_energy += E1 - E0

    def _process_parent_mergers(self, corr_time) -> None:
        """
        Process parent mergers by merging children systems into new parents.
        Reverse kicks are applied to children systems, and new offset is.
        previous timestep.
        """
        def process_single_merger(parent_key_oldparent):
            parent_key, old_parent_set = parent_key_oldparent
            result = {
                "parent_key": parent_key,
                "old_parent_set": old_parent_set,
                }

            if parent_key in self._isolated_mergers:
                offset, newparts = self._isolated_mergers.pop(parent_key)
            else:
                offset = self.evolve_time
                children = self._coll_children[parent_key]

                # Apply reverse kicks (this is thread-safe if internal state is not shared)
                self._correction_kicks(
                    old_parent_set,
                    children,
                    dt=-corr_time,
                    kick_par=False,
                )

                newparts = Particles()
                newparts.add_particles(old_parent_set)
                for _, (old_parent, child) in children.items():
                    newparts.remove_particle(old_parent)
                    # Shift children positions and velocities
                    child.position += old_parent.position
                    child.velocity += old_parent.velocity
                    newparts.add_particles(child)

            newparts.move_to_center()
            change_rad = newparts[newparts.radius > 1 | units.au]
            for p in change_rad:
                if p.mass > MIN_EVOL_MASS:
                    p.radius = ZAMS_radius(p.mass)
                else:
                    p.radius = planet_radius(p.mass)

            scale_mass = newparts.mass.sum()
            scale_radius = set_parent_radius(scale_mass)
            with self.__lock:
                newcode, child_PID = self._sub_worker(
                    children=newparts,
                    scale_mass=scale_mass,
                    scale_radius=scale_radius,
                    )
                self._set_worker_affinity(child_PID)
                self._pid_workers[parent_key] = child_PID

            result.update({
                "parent_key": parent_key,
                "newcode": newcode,
                "offset": offset,
                "scale_radius": scale_radius,
                "worker_pid": child_PID,
                "children": newparts,
                })
            return result

        self.channels["from_gravity_to_parents"].copy()
        if self.dE_track:
            E0 = self.calculate_total_energy()

        particle_keys = self.particles.key
        merger_results = [ ]
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = list(executor.map(
                                process_single_merger, 
                                self._coll_parents.items()
                                )
                           )
            merger_results.extend(futures)

        for result in merger_results:
            parent_key = result["parent_key"]
            newparts = result["children"]
            newcode = result["newcode"]
            offset = result["offset"]
            scale_radius = result["scale_radius"]
            worker_pid = result["worker_pid"]

            new_parent = self.particles[particle_keys == parent_key][0]
            newparent_key = new_parent.key
            new_parent.radius = min(scale_radius, PARENT_RADIUS_MAX)
            self.particles.assign_subsystem(new_parent, newparts)

            self.resume_workers(worker_pid)
            self._cpu_time[newparent_key] = 0
            self._time_offsets[newcode] = offset
            self.subcodes[newparent_key] = newcode
            self._child_channel_maker(
                parent_key=newparent_key,
                code_particles=newcode.particles,
                children=newparts,
            )
            self._pid_workers[newparent_key] = worker_pid
            self.hibernate_workers(worker_pid)

        if self.dE_track:
            E1 = self.calculate_total_energy()
            self.corr_energy += E1 - E0

    def _parent_merger(self, coll_set: Particles) -> None:
        """
        Resolve the merging of two massive parents.

        Args:
            coll_set (Particles):    Colliding particle set
        """
        colliders = Particles()
        for collider in coll_set:
            colliders.add_particle(collider)

        ### Store particles merger history
        old_key = self.old_copy.key
        coll_parents_temp = Particles()
        coll_children_temp = dict()
        new_par = Particles()
        temp_set = Particles()
        isol_system = True

        for particle in colliders:
            par_key = particle.key
            new_par.add_particle(particle)

            if par_key in self.subsystems:  # Not merged yet and hosts children
                if self._verbose:
                    print(f"First time Parent {par_key} merging. Has children...")

                ## Remove all references, keep system dynamically younger
                _, children = self.subsystems.pop(par_key)
                pid = self._pid_workers.pop(par_key)
                self.resume_workers(pid)

                code = self.subcodes.pop(par_key)
                self._time_offsets.pop(code)
                self._child_channels.pop(par_key)
                self._cpu_time.pop(par_key)
                code.cleanup_code()
                code.stop()

                ## Store old attributes for reverse kicks
                old_parent = self.old_copy[old_key == par_key]
                coll_parents_temp.add_particle(old_parent) 
                coll_children_temp[par_key] = (old_parent[0], children)  # Index to make object

            elif par_key in old_key: # Not merged yet and has no children
                if self._verbose:
                    print(f"First time Parent {par_key} merges. Is isolated...")

                p = particle.as_particle_in_set(self.particles)
                temp_set.add_particle(p)  # Updated set

                particle = self.old_copy[old_key == par_key]
                if particle.mass == (0. | units.kg):
                    if self._verbose:
                        print(f"Merging particle is an asteroid")

                    particle.radius = ASTEROID_RADIUS
                elif particle.mass < MIN_EVOL_MASS:
                    particle.radius = planet_radius(particle.mass)
                else:
                    particle.radius = ZAMS_radius(particle.mass)
                coll_parents_temp.add_particle(particle)
                
            elif par_key in self._coll_parents:  # Already merged
                if self._verbose:
                    print(f"Parent {par_key} merged before...")

                isol_system = False
                if par_key in self._isolated_mergers:
                    self._isolated_mergers.pop(par_key)

                ## All references already popped in earlier merger
                old_particle = self._coll_parents.pop(par_key)
                coll_parents_temp.add_particle(old_particle)

                prev_coll_children = self._coll_children.pop(par_key)
                for prev_key, (old_parent, children) in prev_coll_children.items():
                    if prev_key not in coll_children_temp:
                        coll_children_temp[prev_key] = (old_parent, children)
                    else:
                        existing_old_part, existing_children = coll_children_temp[prev_key]
                        children.add_particles(existing_children)
                        coll_children_temp[prev_key] = (existing_old_part, children)

            else:
                print(f"Curious particle {particle} in coll_set?")
                
            self.particles.remove_particle(particle)

        newparent = Particles(1)
        newparent.mass = new_par.mass.sum()
        newparent.position = new_par.center_of_mass()
        newparent.velocity = new_par.center_of_mass_velocity()
        newparent.radius = set_parent_radius(newparent.mass)
        
        if not coll_children_temp and isol_system:  
            # Only isolated particles, preserve dynamical history
            self._isolated_mergers[newparent.key[0]] = (self.model_time, temp_set)
        
        self.particles.add_particles(newparent)
        self._coll_parents[newparent.key[0]]  = coll_parents_temp
        self._coll_children[newparent.key[0]] = coll_children_temp
        
        del new_par

    def _handle_collision(
            self, children: Particles, parent: Particle,
            enc_parti: Particles, code, resolved_keys: dict
            ):
        """
        Merge two particles if the collision stopping condition is met.

        Args:
            children (Particles):  The children particle set
            parent (Particle):     The parent particle
            enc_parti (Particles): The particles in the collision
            code (Code):           The integrator used
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
                old_cpu_time = self._cpu_time.pop(parent_key)

                self.subcodes[newparent_key] = old_code
                new_code = self.subcodes[newparent_key]
                self._time_offsets[new_code] = old_offset
                self._child_channel_maker(
                    parent_key=newparent_key,
                    code_particles=new_code.particles,
                    children=children
                    )
                self._cpu_time[newparent_key] = old_cpu_time
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
        """
        Handle SN events

        Args:
            SN_detect (StoppingCondition):  Stopping condition for SN detection
            bodies (Particles):             Particles in the parent system
        """
        if self.dE_track:
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
            
        if self.dE_track:
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
                    
    def _drift_global(self, dt, corr_time) -> None:
        """
        Evolve parent system until dt.

        Args:
            dt (units.time):         Time to evolve till
            corr_time (units.time):  Time to correct for drift
        """
        if self._verbose:
            print("...Drifting Global...")
            print(f"Evolving until: {dt.in_(units.Myr)}")

        self.__merged = False
        coll_time = None
        while self._evolve_code.model_time < dt * (1. - EPS):
            self._evolve_code.evolve_model(dt)
            if self.grav_coll.is_set():
                if self.dE_track:
                    E0 = self.calculate_total_energy()
                coll_time = self.parent_code.model_time
                coll_sets = self._find_coll_sets(
                                self.grav_coll.particles(0), 
                                self.grav_coll.particles(1)
                                )
                try:
                    if self._verbose:
                        print(f"... Merger @ T={coll_time.value_in(units.Myr):.6f} Myr")
                    for cs in coll_sets:
                        self._parent_merger(cs)
                except Exception as e:
                    print(f"Error while merging {coll_sets}: {e}")
                    print("Traceback:", traceback.format_exc())
                    self.cleanup_code()
                    sys.exit()
                    
                if self.dE_track:
                    E1 = self.calculate_total_energy()
                    self.corr_energy += E1 - E0

        if (self.__gal_field):
            while self.parent_code.model_time < dt * (1. - EPS):
                self.parent_code.evolve_model(dt)
                if self.grav_coll.is_set():
                    if self.dE_track:
                        E0 = self.calculate_total_energy()
                    
                    coll_time = self.parent_code.model_time
                    coll_sets = self._find_coll_sets(
                                    self.grav_coll.particles(0), 
                                    self.grav_coll.particles(1)
                                    )
                    try:
                        if self._verbose:
                            print(f"... Merger @ T={coll_time.value_in(units.Myr):.6f} Myr")
                        for cs in coll_sets:
                            self._parent_merger(cs)
                    except Exception as e:
                        print(f"Error while merging {coll_sets}: {e}")
                        print("Traceback:", traceback.format_exc())
                        self.cleanup_code()
                        sys.exit()

                    if self.dE_track:
                        E1 = self.calculate_total_energy()
                        self.corr_energy += E1 - E0

        if coll_time:
            self._process_parent_mergers(corr_time)
            self._coll_parents.clear()
            self._coll_children.clear()
            self._isolated_mergers.clear()

    def _drift_child(self, dt) -> None:
        """
        Evolve children system until dt.

        Args:
            dt (units.time):  Time to evolve till.
        """
        def resolve_collisions(
                code, 
                parent: Particle, 
                children: Particles, 
                stopping_condition
                ):
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

            t0 = time.time()
            while code.model_time < evol_time * (1. - EPS):
                code.evolve_model(evol_time)
                if stopping_condition.is_set():
                    with self.__lock:
                        if self.dE_track:
                            KE = code.particles.kinetic_energy()
                            PE = code.particles.potential_energy()
                            E0 = KE + PE

                        parent = resolve_collisions(
                                    code, parent,
                                    children,
                                    stopping_condition
                                    )

                        if self.dE_track:
                            KE = code.particles.kinetic_energy()
                            PE = code.particles.potential_energy()
                            E1 = KE + PE
                            self.corr_energy += E1 - E0
            t1 = time.time()
            self._cpu_time[parent.key] = t1 - t0
            self.hibernate_workers(self._pid_workers[parent.key])

        if self._verbose:
            print("...Drifting Children...")
        
        sorted_cpu_time = sorted(
            self.subcodes.keys(),
            key=lambda x: self._cpu_time[x],
            reverse=True
        )
        with ThreadPoolExecutor(max_workers=self.avail_cpus) as executor:
            futures = {
                executor.submit(evolve_code, parent_key): parent_key 
                for parent_key in sorted_cpu_time
            }
            for ifut, future in enumerate(as_completed(futures)):  # Iterate over to ensure no silent failures
                parent_key = futures[future]
                try:
                    future.result()
                except Exception as e:
                    if ifut == 0:
                        self.cleanup_code()
                    else:
                        self.cleanup_code(first_clean=0)
                    print(f"Error while evolving parent {parent_key}: {e}")
                    print(f"Traceback: {traceback.format_exc()}")
                    sys.exit()

        for parent_key in list(self.subcodes.keys()):  # Remove single children systems:
            pid = self._pid_workers[parent_key]
            self.resume_workers(pid)
            if len(self.subcodes[parent_key].particles) == 1:
                old_subcode = self.subcodes.pop(parent_key)
                self._time_offsets.pop(old_subcode)
                self._child_channels.pop(parent_key)
                self._pid_workers.pop(parent_key)
                self._cpu_time.pop(parent_key)

                old_subcode.cleanup_code()
                old_subcode.stop()

                del old_subcode

            else:
                self.hibernate_workers(pid)

    def _kick_particles(self, particles: Particles, corr_code, dt) -> None:
        """
        Apply correction kicks onto target particles/

        Args:
            particles (Particles):  Particles whose accelerations are corrected
            corr_code (Code):       Object providing the difference in gravity
            dt (units.time):        Time-step of correction kick
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
        
        del corr_code, ax, ay, az

    def _correct_children(
            self, perturber_mass, 
            perturber_x, perturber_y, perturber_z,
            parent_x, parent_y, parent_z, 
            subsystem: Particles, dt
            ) -> None:
        """
        Apply correcting kicks onto children particles.

        Args:
            perturber_mass (units.mass):  Mass of perturber
            perturber_x (units.length):  X-position of perturber
            perturber_y (units.length):  Y-position of perturber
            perturber_z (units.length):  Z-position of perturber
            parent_x (units.length):     X-position of parent
            parent_y (units.length):     Y-position of parent
            parent_z (units.length):     Z-position of parent
            subsystem (Particles):       Children particle set
            dt (units.time):             Time interval for applying kicks
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
       
    def _correction_kicks(
            self, particles: Particles, 
            subsystems: dict, dt, 
            kick_par=True
            ) -> None:
        """
        Apply correcting kicks onto children and parent particles

        Args:
            particles (Particles):  Parent particle set
            subsystems (dict):      Dictionary of children system
            dt (units.time):        Time interval for applying kicks
            kick_par (boolean):     Whether to apply correction to parents
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
            
            if kick_par:
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
                del corr_chd

            futures = []
            with ThreadPoolExecutor(max_workers=self.avail_cpus) as executor:
                try:
                    for parent, children in subsystems.values():
                        future = process_children_jobs(parent, children)
                        futures.append(future)
                    for future in as_completed(futures):
                        future.result()
                except Exception as e:
                    print(f"Error submitting job for parent: {e}")
                    print(f"Traceback: {traceback.format_exc()}")
                    self.cleanup_code()
                    sys.exit()

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
        return max(1, ncpu)  # Ensure at least one CPU is available