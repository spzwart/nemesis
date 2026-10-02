"""
Possible Room for Improvements:
1. Flexible bridge times [Global].
2. Adaptable individual time-step depending on proximity of parents
   with one another.
3. Flexible parent radius.
4. In _handle_collision(), the logic remnant.key == children[nearest_mass].key
   could be removed entirely. But needs testing to ensure no bugs.
5. In _process_parent_mergers() (and splitting subcodes), ideally one would
   parallelise the routine. However, parallelising is made difficult due to
   needing PID list for management. Could be improved if one trackers workers
   memory address and not their PID. Also, even when many mergers occur, it is
   not a bottle neck, so not a priority.
6. _parent_merger() is relatively slow due to dictionary manipulation.
7. Stellar code in AMUSE can't handle different metallicities and/or
   stellar ages,requires one per group.

Extending the Physics:
1. Handling of binary stellar evolution with SeBa. Currently, if a binary
   is formed, the two stars are evolved as single stars.
2. Allowing post-Newtonian codes to have particles constructed in the
   Nemesis fashion.
3. Inclusion of hydro code in child worker to allow for gas accretion
   and disc evolution within embedded environments.

Other:
- Huayno can solve Kepler equations. This would yield energy errors at
  numerical precision and give quicker results, but the difference is
  minimal since a two-body children will not be the bottle neck.
  Nevertheless, if you wish to implement this, you will need to change
  how the recycling of children is done in _process_parent_mergers()
  and split_subcodes() since if you go from an N -> 2 or a 2 -> N-body system,
  recycling will no longer work.
"""
from __future__ import annotations

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
from amuse.units import units, nbody_system

from src.environment_functions import (
    set_parent_radius, 
    planet_radius, 
    ZAMS_radius
)
from src.globals import (
    ASTEROID_RADIUS, 
    EPS, 
    GRAV_CONST,
    MIN_EVOL_MASS
)
from src.grav_correctors import CorrectionKicks
from src.hierarchical_particles import HierarchicalParticles
from src.split_children import split_subcodes


class Nemesis(object):
    def __init__(
        self,
        dtbridge: units.time,
        code_dt: units.time,
        n_worker_parent: int,
        par_conv: nbody_system.nbody_to_si,
        coll_dir: str,
        free_cpus=os.cpu_count(),
        nmerge=0,
        resume_time=0. | units.yr,
        dE_track=False,
        star_evol=False,
        gal_field=True,
        verbose=True,
    ):
        """
        Class setting up the simulation.
        Args:
            dtbridge (units.time):     Diagnostic time step.
            code_dt (float):           Internal time step.
            n_worker_parent (int):     Number of workers for parent code.
            par_conv (converter):      Parent N-body converter.
            coll_dir (str):            Path to store collision data.
            free_cpus(int):            Number of available CPUs.
            nmerge (int):              Number of mergers in initial conditions.
            resume_time (units.time):  Time which simulation is resumed at.
            dE_track (bool):           Flag for energy error tracker.
            star_evol (bool):          Flag for stellar evolution.
            gal_field (bool):          Flag for galactic field.
            verbose (bool):            Flag for verbose output.
        """
        # Private attributes
        self.__dt = dtbridge
        self.__coll_dir = coll_dir
        self.__code_dt = code_dt
        self.__gal_field = gal_field
        self.__main_process = psutil.Process(os.getpid())
        self.__nmerge = nmerge
        self.__par_nworker = n_worker_parent
        self.__resume_offset = resume_time
        self.__star_evol = star_evol
        self.__total_free_cpus = free_cpus
        self.__worker_list = [
            "huayno_worker",
            "ph4_worker",
            "symple_worker"
            ]

        # Protected attributes
        self._verbose = verbose
        self._parent_conv = par_conv
        self._MWG = MWpotentialBovy2015()
        self._coll_parents = dict()
        self._coll_children = dict()
        self._isolated_mergers = dict()
        self._lock = threading.Lock()

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
        self.corr_energy = 0. | units.J

        if self.__star_evol:
            self._star_to_parents_chnl()

        self._validate_initialization()
        self.CorrKicks = CorrectionKicks(
            grav_lib=self._load_grav_lib(),
            nworkers=self.num_workers
            )

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
            raise ValueError(f"Error: minimum stellar mass {MIN_EVOL_MASS} must be positive")
        if not isinstance(self.__coll_dir, str):
            raise ValueError("Error: coll_dir must be a string")

    def _load_grav_lib(self) -> ctypes.CDLL:
        """Setup library to allow Python and C++ communication"""
        path = os.getcwd()
        py_to_c_types = ndpointer(
            dtype=np.float64,
            ndim=1,
            flags='C_CONTIGUOUS'
            )

        lib = ctypes.CDLL(f'{path}/src/build/kick_particles_worker.so')
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

    def cleanup_code(self, first_clean=True) -> None:
        """
        Cleanup all codes and processes
        Args:
            first_clean (bool):  If true, cleans parent and stellar codes.
        """
        if self._verbose:
            print("...Cleaning up Nemesis...", flush=True)

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
            print("...Nemesis cleaned up...", flush=True)

    def commit_particles(self) -> None:
        """
        Commit particle system by:
            - Recentering the children.
            - Setting the parent radius.
            - Initialising children codes.
            - Creating channels for children.
            - Defining stellar code particles.
            - Setting up the galactic field.
        """
        if (self.__star_evol):
            p = self.particles.all()
            star_mask = p.mass > MIN_EVOL_MASS
            self.stars = p[star_mask]
            self.stellar_code.particles.add_particles(self.stars)

        self.particles.radius = set_parent_radius(self.particles.mass)
        ntotal = len(self.children.keys())
        for nchild, (parent_key, (parent, child)) in enumerate(self.children.items()):
            if self._verbose:
                print(f"\rSystem {nchild+1}/{ntotal}...", end="", flush=True)

            scale_mass = parent.mass
            scale_radius = set_parent_radius(scale_mass)
            parent.radius = scale_radius

            child.move_to_center()
            if parent not in self.subcodes:
                code, child_pid = self._sub_worker(
                    children=child,
                    scale_mass=scale_mass,
                    scale_radius=scale_radius
                    )

                self._set_worker_affinity(child_pid)
                self._time_offsets[code] = self.model_time
                self.children[parent_key] = (parent, child)
                self.subcodes[parent_key] = code
                self._child_channel_maker(
                    parent_key=parent_key,
                    code_set=code.particles,
                    children=child
                )
                self._cpu_time[parent_key] = 0

                # Store children PID to allow hibernation
                self._pid_workers[parent_key] = child_pid
                self.hibernate_workers(child_pid)

        self.particles.recenter_children(max_workers=self.avail_cpus)
        if (self.__gal_field):
            self._setup_bridge()
        else:
            self._evolve_code = self.parent_code

    def _setup_bridge(self) -> None:
        """Embed cluster into galactic potential"""
        gravity = bridge.Bridge(use_threading=True, method=SPLIT_4TH_S_M6,)
        gravity.add_system(self.parent_code, (self._MWG, ))
        gravity.timestep = self.__dt
        self._evolve_code = gravity

    def _stellar_worker(self) -> object:
        """Define stellar evolution integrator"""
        return SeBa()

    def _parent_worker(self) -> object:
        """Define global integrator"""
        code = Ph4(self._parent_conv, number_of_workers=self.__par_nworker)
        code.parameters.epsilon_squared = (0. | units.au)**2.
        code.parameters.timestep_parameter = self.__code_dt
        code.parameters.force_sync = True
        return code

    def _sub_worker(
        self,
        children: Particles,
        scale_mass: units.mass,
        scale_radius: units.length,
        number_of_workers=1
    ) -> tuple[object, list[int]]:
        """
        Initialise children integrator.
        Args:
            children (Particles):         Child particle set.
            scale_mass (units.mass):      Mass of the system.
            scale_radius (units.length):  Radius of the system.
            number_of_workers (int):      Number of workers to use.
        Returns:
            Code:  Gravitational integrator with particle set
        """
        if len(children) == 0:
            self.cleanup_code()
            raise ReferenceError("No children provided")

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
        code.set_integrator("SHARED10_COLLISIONS")

        PIDs_after = self._snapshot_worker_pids()
        worker_PID = list(PIDs_after - PIDs_before)

        return code, worker_PID

    def _set_worker_affinity(self, pid_list: list[int]) -> None:
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
            raise RuntimeError("Warning: could not set affinity for workers")

    def _snapshot_worker_pids(self) -> set[int]:
        """Return the set of PIDs of all children workers"""
        pids = set()
        try:
            children = self.__main_process.children(recursive=True)
        except (psutil.NoSuchProcess, FileNotFoundError):
            return pids

        for child in children:
            try:
                info = child.as_dict(attrs=["pid", "name"])
            except (psutil.Error, FileNotFoundError):
                continue

            name = info.get("name") or ""
            if any(tag in name for tag in self.__worker_list):
                pids.add(info["pid"])

        return pids

    def hibernate_workers(self, pid_list: list[int]) -> None:
        """
        Hibernate workers to reduce CPU usage and optimise performance.
        Args:
            pid (list):  List of process ID of worker.
        """
        for pid in pid_list:
            try:
                os.kill(pid, signal.SIGSTOP)
            except ProcessLookupError:
                self.cleanup_code()
                raise ProcessLookupError(f"Process {pid} not found. It may have exited.")
            except PermissionError:
                self.cleanup_code()
                raise PermissionError(f"Insufficient permissions to stop process {pid}.")

    def resume_workers(self, pid_list: list[int]) -> None:
        """
        Resume workers to continue simulation.
        Args:
            pid_list (list):  List of process IDs for worker.
        """
        for pid in pid_list:
            try:
                os.kill(pid, signal.SIGCONT)
            except ProcessLookupError:
                self.cleanup_code()
                raise ProcessLookupError(f"Process {pid} not found. It may have exited.")
            except PermissionError:
                self.cleanup_code()
                raise PermissionError(f"Insufficient permissions to stop process {pid}.")

    def _star_to_parents_chnl(self) -> None:
        """Create channels to copy stellar attributes to parent code."""
        parent_particles = self.parent_code.particles
        self.channels = {
            "from_stellar_to_gravity":
                self.stellar_code.particles.new_channel_to(
                    parent_particles,
                    attributes=["mass"],
                    target_names=["mass"]
                ),
        }

    def _child_channel_maker(
            self,
            parent_key: int,
            code_set: Particles,
            children: Particles
    ) -> dict:
        """
        Create communication channel between codes and specified
        children system.
        Args:
            parent_key (int):      Parent particle key.
            code_set (Particles):  Children particle set in grav. code.
            children (Particles):  Children particle set in local memory.
        Returns:
            dict:  Dictionary of newly created channel
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
                        code_set,
                        attributes=["mass", "radius"],
                        target_names=["mass", "radius"]
                        ),
                "from_gravity_to_children":
                    code_set.new_channel_to(
                        children,
                        attributes=grav_copy_variables,
                        target_names=grav_copy_variables
                        ),
                "from_children_to_gravity":
                    children.new_channel_to(
                        code_set,
                        attributes=grav_copy_variables,
                        target_names=grav_copy_variables
                        )
                }
        else:
            self._child_channels[parent_key] = {
                "from_gravity_to_children":
                    code_set.new_channel_to(
                        children,
                        attributes=grav_copy_variables,
                        target_names=grav_copy_variables
                        ),
                "from_children_to_gravity":
                    children.new_channel_to(
                        code_set,
                        attributes=grav_copy_variables,
                        target_names=grav_copy_variables
                        )
                }

        return self._child_channels[parent_key]

    def calculate_total_energy(self) -> units.energy:
        """Calculate systems total energy."""
        all_parts = self.particles.all()
        Ek = all_parts.kinetic_energy()
        Ep = all_parts.potential_energy()
        return Ek + Ep

    def _star_channel_copier(self) -> None:
        """Copy attributes from stellar code to grav. particle set"""
        self.channels["from_stellar_to_gravity"].copy()
        pid_dictionary = self._pid_workers
        for parent_key, channel in self._child_channels.items():
            pid = pid_dictionary[parent_key]
            self.resume_workers(pid)
            channel["from_star_to_gravity"].copy()
            self.hibernate_workers(pid)

    def _sync_grav_to_local(self) -> None:
        """Sync gravity particles to local child set"""
        pid_dictionary = self._pid_workers
        for parent_key, channel in self._child_channels.items():
            pid = pid_dictionary[parent_key]
            self.resume_workers(pid)
            channel["from_gravity_to_children"].copy()
            self.hibernate_workers(pid)

    def _sync_local_to_grav(self) -> None:
        """Sync local child set set to gravity particles"""
        pid_dictionary = self._pid_workers
        for parent_key, channel in self._child_channels.items():
            pid = pid_dictionary[parent_key]
            self.resume_workers(pid)
            channel["from_children_to_gravity"].copy()
            self.hibernate_workers(pid)

    def evolve_model(self, tend: units.time, timestep=None) -> None:
        """
        Evolve the system of particles until tend.
        Args:
            tend (units.time):      Time to simulate till
            timestep (units.time):  Time step to simulate
        """
        if timestep is None:
            timestep = tend - self.model_time

        self.evolve_time = self.model_time
        kick_corr = timestep
        while self.model_time < (self.evolve_time + timestep) * (1. - EPS):
            self.dt_step += 1

            if (self.evolve_time + self.__resume_offset) == 0. | units.yr:
                if self.__star_evol:
                    self._stellar_evolution(0.5 * timestep)
                    self._star_channel_copier()
            else:
                self.CorrKicks._correction_kicks(
                    self.particles,
                    self.children,
                    dt=timestep
                )
                self.particles.recenter_children(
                    self.num_workers
                )
                self._sync_local_to_grav()

            # Drift step
            self._drift_global(
                self.model_time + timestep,
                corr_time=kick_corr
                )
            if self.subcodes:
                self._drift_child(self.model_time)
            self._sync_grav_to_local()

            # Star evolution
            if (self.__star_evol):
                self._stellar_evolution(
                    self.model_time + 0.5 * timestep
                )
                self._star_channel_copier()
            split_subcodes(nem_class=self)

        if self._verbose:  # For diagnostics
            Nkids = len(self.children.keys())
            print(
                f"Time: {self.model_time.in_(units.Myr)}\n"
                f"Global time: {self.parent_code.model_time.in_(units.Myr)}", 
                flush=True
                )
            if self.__star_evol:
                print(f"Stellar code time: {self.stellar_code.model_time.in_(units.Myr)}", flush=True)
            print(f"#Children: {Nkids}")
            print("===" * 50, flush=True)

    def _process_parent_mergers(self, corr_time: units.time) -> None:
        """
        Process all parent mergers in previous global drift step.
        - New children system constructed.
        - Apply reverse kicks on new children so they are consistent.
          with the previous timestep. Required for ZKL.
        """
        particle_keys = self.particles.key
        for parent_key, old_parent_set in self._coll_parents.items():
            code = None
            cpu_time = 0.0

            if parent_key in self._isolated_mergers:
                offset, newparts = self._isolated_mergers.pop(parent_key)
            else:  # At least one parent had children before merger.
                offset = self.evolve_time
                children_set = self._coll_children[parent_key]

                # Apply reverse kicks
                self.CorrKicks._correction_kicks(
                    old_parent_set,
                    children_set,
                    dt=-corr_time,
                )

                newparts = Particles()
                newparts.add_particles(old_parent_set)
                for old_key, (old_parent, children) in children_set.items():
                    newparts.remove_particle(old_parent)

                    # Shift children positions and velocities
                    children.position += old_parent.position
                    children.velocity += old_parent.velocity
                    newparts.add_particles(children)

                    child_pid = self._pid_workers.pop(old_key, None)
                    if child_pid is not None:  # Specific parent hosted children.
                        self.resume_workers(child_pid)
                        if code is None:
                            code = self.subcodes.pop(old_key)
                            old_offset = self._time_offsets.pop(code)
                            worker_pid = child_pid

                        else:
                            if self._verbose:
                                print("Cleaning up old code...", flush=True)
                            old_code = self.subcodes.pop(old_key)
                            self._time_offsets.pop(old_code)
                            old_code.cleanup_code()
                            old_code.stop()

                        cpu_time += self._cpu_time.pop(old_key)
                        self._child_channels.pop(old_key)

                    else:  # Specific parent was isolated.
                        None

            newparts.move_to_center()
            change_rad = newparts[newparts.radius > 1 | units.au]  # Hand-wavey
            for p in change_rad:
                if p.mass > MIN_EVOL_MASS:
                    p.radius = ZAMS_radius(p.mass)
                else:
                    p.radius = planet_radius(p.mass)

            scale_mass = newparts.mass.sum()
            scale_radius = set_parent_radius(scale_mass)
            if code is None:
                newcode, worker_pid = self._sub_worker(
                    children=newparts,
                    scale_mass=scale_mass,
                    scale_radius=scale_radius,
                    )
            else:
                code.particles.remove_particles(code.particles)
                code.particles.add_particles(newparts)
                newcode = code

            self._set_worker_affinity(worker_pid)
            new_parent = self.particles[particle_keys == parent_key][0]
            new_parent.radius = scale_radius
            self.particles.assign_children(new_parent, newparts)

            if code is None:  # Parent merger consisted of isolated particles.
                self._time_offsets[newcode] = offset
            else:  # Required, otherwise code will fall behind dynamically.
                self._time_offsets[newcode] = old_offset

            self._cpu_time[parent_key] = cpu_time
            self.subcodes[parent_key] = newcode
            self._child_channel_maker(
                parent_key=parent_key,
                code_set=newcode.particles,
                children=newparts,
            )
            self._pid_workers[parent_key] = worker_pid
            self.hibernate_workers(worker_pid)

    def _parent_merger(self, coll_set: Particles) -> None:
        """
        Resolve the merging of two massive parents.
        Args:
            coll_set (Particles):  Colliding particle set.
        """
        colliders = Particles()
        for collider in coll_set:
            colliders.add_particle(collider)

        # Store particles merger history
        coll_parents_temp = Particles()
        coll_children_temp = dict()
        new_par = Particles()
        temp_set = Particles()
        isol_system = True

        for particle in colliders:
            par_key = particle.key
            new_par.add_particle(particle)

            if par_key in self.children:
                if self._verbose:
                    print("First time Parent merges. Has children...", flush=True)

                # Store old attributes for reverse kicks and ZKL
                _, children = self.children.pop(par_key)
                old_parent = self.old_copy_map[par_key]
                coll_parents_temp.add_particle(old_parent)
                coll_children_temp[par_key] = (old_parent, children)

            elif par_key in self.old_keys:
                if self._verbose:
                    print("First time Parent merges. Isolated...", flush=True)

                # Store updated and old set in case other parent has children.
                p = particle.as_particle_in_set(self.particles)
                temp_set.add_particle(p)  # Updated set.
                particle = self.old_copy_map[par_key]  # Old set.
                if particle.mass == (0. | units.kg):
                    if self._verbose:
                        print("Merging particle is an asteroid", flush=True)
                    particle.radius = ASTEROID_RADIUS
                elif particle.mass < MIN_EVOL_MASS:
                    particle.radius = planet_radius(particle.mass)
                else:
                    particle.radius = ZAMS_radius(particle.mass)
                coll_parents_temp.add_particle(particle)

            elif par_key in self._coll_parents:
                if self._verbose:
                    print("Parent merged before...", flush=True)

                isol_system = False
                if par_key in self._isolated_mergers:
                    self._isolated_mergers.pop(par_key)

                # Default Nemesis references already popped in earlier merger
                old_particle = self._coll_parents.pop(par_key)
                coll_parents_temp.add_particle(old_particle)

                # Re-map references of previous merger
                prev_coll_children = self._coll_children.pop(par_key)
                for prev_key, (old_par, children) in prev_coll_children.items():
                    if prev_key not in coll_children_temp:
                        coll_children_temp[prev_key] = (old_par, children)
                    else:
                        ex_old_par, ex_children = coll_children_temp[prev_key]
                        children.add_particles(ex_children)
                        coll_children_temp[prev_key] = (
                            ex_old_par, children
                            )

            else:
                print(f"Curious particle {particle} in coll_set?", flush=True)

            self.particles.remove_particle(particle)

        newparent = Particles(1)
        newparent.mass = new_par.mass.sum()
        newparent.position = new_par.center_of_mass()
        newparent.velocity = new_par.center_of_mass_velocity()
        newparent.radius = set_parent_radius(newparent.mass)

        if not coll_children_temp and isol_system:
            # Only isolated particles, preserve dynamical history
            self._isolated_mergers[newparent.key[0]] = (
                self.model_time, temp_set
                )

        self.particles.add_particles(newparent)
        self._coll_parents[newparent.key[0]] = coll_parents_temp
        self._coll_children[newparent.key[0]] = coll_children_temp

    def _handle_collision(
            self,
            children: Particles,
            parent: Particle,
            enc_parti: Particles,
            code: object,
            resolved_keys: dict,
            first_merger: bool
    ) -> tuple[Particles, dict]:
        """
        Merge two particles if the collision stopping condition is met.
        Args:
            children (Particles):  The children particle set.
            parent (Particle):     The parent particle.
            enc_parti (Particles): The particles in the collision.
            code (object):         The integrator used.
            resolved_keys (dict):  Dictionary, {Collider i Key: Remnant Key}.
            first_merger (bool):   Whether it's the first merger in children during integration step.
        Returns:
            Particles: New parent.
            dict: Updated resolved keys dictionary.
        """
        if self.dE_track:
            E0 = self.calculate_total_energy()

        # Save properties
        self.__nmerge += 1
        parent_key = parent.key
        self._child_channels[parent_key]["from_gravity_to_children"].copy()

        coll_a = enc_parti[0].as_particle_in_set(children)
        coll_b = enc_parti[1].as_particle_in_set(children)
        collider = Particles(particles=[coll_a, coll_b])
        if self._verbose:
            print(
                f"...Collision #{self.__nmerge} Detected...\n"
                f"Collider A: {coll_a}"
                f"Collider B: {coll_b}",
                flush=True
            )

        kepler_elements = orbital_elements(collider, G=GRAV_CONST)
        sma = kepler_elements[2]
        ecc = kepler_elements[3]
        inc = kepler_elements[5]

        tcoll = code.model_time + self._time_offsets[code] + self.__resume_offset
        file_name = os.path.join(self.__coll_dir, f"merger{self.__nmerge}.txt")
        with open(file_name, "w") as f:
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
            if max(collider_mass) == (0. | units.kg):
                raise ValueError(
                    "Collision between two zero-mass particles."
                    "Something went wrong..."
                    )

            remnant = Particles(particles=[most_massive])

        elif max(collider_mass) > (0 | units.kg):
            remnant = Particles(1)
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

        if self._verbose:
            print(f"{coll_a.type}, {coll_b.type}", flush=True)

        remnant.coll_events = max(collider.coll_events) + 1
        remnant.type = most_massive.type
        remnant.original_key = most_massive.original_key

        # Deal with simultaneous collision events being detected in code.
        changes = []
        coll_a_change = 0
        coll_b_change = 0
        if not resolved_keys:
            resolved_keys[coll_a.key] = remnant.key[0]
            resolved_keys[coll_b.key] = remnant.key[0]

        else:  # If the current collider is a remnant of past event, remap
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

            if remnant.key == children[nearest_mass].key:  # If remnant is host
                children.position += parent.position
                children.velocity += parent.velocity

                newparent = self.particles.add_children(children)
                newparent_key = newparent.key
                newparent.radius = parent.radius

                # Re-mapping dictionary to new parent
                old_code = self.subcodes.pop(parent_key)
                old_offset = self._time_offsets.pop(old_code)
                old_cpu_time = self._cpu_time.pop(parent_key)
                self._child_channels.pop(parent_key)

                self.subcodes[newparent_key] = old_code
                new_code = self.subcodes[newparent_key]
                self._time_offsets[new_code] = old_offset
                self._child_channel_maker(
                    parent_key=newparent_key,
                    code_set=new_code.particles,
                    children=children
                    )
                self._cpu_time[newparent_key] = old_cpu_time
                child_pid = self._pid_workers.pop(parent_key)
                self._pid_workers[newparent_key] = child_pid
                self.particles.remove_particle(parent)

            else:
                newparent = parent
        else:
            newparent = parent

        children.synchronize_to(self.subcodes[newparent.key].particles)
        if self.dE_track:
            E1 = self.calculate_total_energy()
            self.corr_energy += E1 - E0

        return newparent, resolved_keys

    def _handle_supernova(self, SN_detect) -> None:
        """
        Handle SN events.
        Args:
            SN_detect (StoppingCondition):  SN stopping conidtion.
        """
        if self._verbose:
            print("...Supernova Detected...", flush=True)

        if self.dE_track:
            E0 = self.calculate_total_energy()

        sn_particles = SN_detect.particles(0)
        p = self.particles.all()
        for ci in range(len(sn_particles)):
            if self._verbose:
                print(f"...Supernova Detected...")

            sn = sn_particles[ci]
            child_particle = False
            local_p = sn.as_particle_in_set(p)
            if local_p in self.parent_code.particles:
                if self._verbose:
                    print("Supernova in parent code", flush=True)

                gravity_particle = local_p.as_particle_in_set(
                    self.parent_code.particles
                    ).as_set()
                code = self.parent_code

            else:
                for parent_key, code in self.subcodes.items():
                    self.resume_workers(self._pid_workers[parent_key])
                    if local_p in code.particles:
                        if self._verbose:
                            print("Supernova in  children code", flush=True)

                        gravity_particle = local_p.as_particle_in_set(
                            code.particles
                            ).as_set()
                        break

                    self.hibernate_workers(self._pid_workers[parent_key])
                    child_particle = True

            # Ugly, but AMUSE doesn't allow direct assignment for particle subsets
            parts = gravity_particle.copy()
            parts.vx = gravity_particle.vx + sn.natal_kick_x
            parts.vy = gravity_particle.vy + sn.natal_kick_y
            parts.vz = gravity_particle.vz + sn.natal_kick_z

            channel = parts.new_channel_to(gravity_particle)
            channel.copy_attributes(["vx", "vy", "vz"])

            if child_particle:
                self.hibernate_workers(self._pid_workers[parent_key])

            if self.dE_track:
                E1 = self.calculate_total_energy()
                self.corr_energy += E1 - E0

    def _find_coll_sets(self, p1: Particle, p2: Particle) -> UnionFind:
        """
        Find encountering particle sets.
        Args:
            p1 (Particle):  Particle a of merger.
            p2 (Particle):  Particle b of merger.
        Returns:
            UnionFind: Set of colliding particles.
        """
        coll_sets = UnionFind()
        for p, q in zip(p1, p2):
            coll_sets.union(p, q)
        return coll_sets.sets()

    def _stellar_evolution(self, dt: units.time) -> None:
        """
        Evolve stellar evolution.
        Args:
            dt (units.time):  Time to evolve till
        """
        while self.stellar_code.model_time < dt * (1. - EPS):
            self.stellar_code.evolve_model(dt)

            if self.SN_detection.is_set():
                print("...Detection: SN Explosion...", flush=True)
                self._handle_supernova(self.SN_detection)

    def _drift_global(self, dt, corr_time) -> None:
        """
        Evolve parent system until dt.
        Args:
            dt (units.time):         Time to evolve till
            corr_time (units.time):  Time to correct for drift
        """
        if self._verbose:
            print(
                "...Drifting Global...\n"
                f"Evolving {len(self.particles)} until: {dt.in_(units.Myr)}",
                flush=True
            )

        self.old_copy = self.particles.copy()
        self.old_copy_map = {p.key: p for p in self.old_copy}
        self.old_keys = self.old_copy.key

        coll_time = None
        while self._evolve_code.model_time < dt * (1. - EPS):
            self._evolve_code.evolve_model(dt)
            if self.grav_coll.is_set():
                coll_time = self.parent_code.model_time
                coll_sets = self._find_coll_sets(
                    self.grav_coll.particles(0),
                    self.grav_coll.particles(1)
                    )
                try:
                    if self._verbose:
                        print(f"... Merger @ T={coll_time.in_(units.Myr)}", flush=True)
                    for cs in coll_sets:
                        self._parent_merger(cs)
                except Exception:
                    self.cleanup_code()
                    raise Exception("Error during parent merger.")

        if (self.__gal_field):
            while self.parent_code.model_time < dt * (1. - EPS):
                self.parent_code.evolve_model(dt)
                if self.grav_coll.is_set():
                    coll_time = self.parent_code.model_time
                    coll_sets = self._find_coll_sets(
                        self.grav_coll.particles(0),
                        self.grav_coll.particles(1)
                        )
                    try:
                        if self._verbose:
                            print(f"... Merger @ T={coll_time.in_(units.Myr)}", flush=True)
                        for cs in coll_sets:
                            self._parent_merger(cs)
                    except Exception:
                        self.cleanup_code()
                        raise Exception("Error during parent merger.")

        if coll_time:
            self._process_parent_mergers(corr_time)
            self._coll_parents.clear()
            self._coll_children.clear()
            self._isolated_mergers.clear()

    def _drift_child(self, dt: units.time) -> None:
        """
        Evolve children system until dt.
        Args:
            dt (units.time):  Time to evolve till.
        """
        def resolve_collisions(
                code: object,
                parent: Particle,
                children: Particles,
                stopping_condition,
                first_merger: bool
        ) -> Particle:
            """
            Function to resolve collisions
            Args:
                code (object):      Code with collision.
                parent (Particle):  Parent particle.
                stopping_condition (StoppingCondition): Stopping condition.
                first_merger (bool): Whether it's the first merger in children during integration step.
            Returns:
                Particle: Updated parent particle.
            """
            coll_a_particles = stopping_condition.particles(0)
            coll_b_particles = stopping_condition.particles(1)

            resolved_keys = dict()
            Nmergers = max(
                len(np.unique(coll_a_particles.key)),
                len(np.unique(coll_b_particles.key))
                )
            Nresolved = 0
            for coll_a, coll_b in zip(coll_a_particles, coll_b_particles):
                if Nresolved < Nmergers:  # Stop recursive loop
                    particle_dict = {p.key: p for p in code.particles}
                    if coll_a.key in resolved_keys.keys():
                        coll_a = particle_dict.get(resolved_keys[coll_a.key])
                    if coll_b.key in resolved_keys:
                        coll_b = particle_dict.get(resolved_keys[coll_b.key])

                    if coll_b.key == coll_a.key:
                        print("Curious?", flush=True)
                        continue

                    colliding_particles = Particles(particles=[coll_a, coll_b])
                    parent, resolved_keys = self._handle_collision(
                        children,
                        parent,
                        colliding_particles,
                        code,
                        resolved_keys,
                        first_merger
                        )
                    Nresolved += 1

            del resolved_keys

            return parent

        def evolve_code(parent_key: Particle) -> None:
            """
            Evolve children code.
            Args:
                parent_key (int):  Parent particle key
            """
            self.resume_workers(self._pid_workers[parent_key])
            code = self.subcodes[parent_key]
            parent = self.children[parent_key][0]
            children = self.children[parent_key][1]
            first_merger = True

            sc = code.stopping_conditions.collision_detection
            sc.enable()  # A bit dirty
            evol_time = dt - self._time_offsets[code]

            t0 = time.time()
            while code.model_time < evol_time * (1. - EPS):
                code.evolve_model(evol_time)
                if sc.is_set():
                    with self._lock:
                        parent = resolve_collisions(
                            code, parent, children, sc, first_merger
                            )
                        first_merger = False

            t1 = time.time()
            self._cpu_time[parent.key] = t1 - t0
            self.hibernate_workers(self._pid_workers[parent.key])

        if self._verbose:
            print("...Drifting Children...", flush=True)

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
            for ifut, future in enumerate(as_completed(futures)):
                parent_key = futures[future]
                try:
                    future.result()
                except Exception as e:
                    if ifut == 0:
                        self.cleanup_code()
                    else:
                        self.cleanup_code(first_clean=False)
                    raise Exception(f"Error while evolving parent {parent_key}: {e}")

        for parent_key in list(self.subcodes.keys()):
            pid = self._pid_workers[parent_key]
            if len(self.subcodes[parent_key].particles) == 1:
                self.resume_workers(pid)
                old_subcode = self.subcodes.pop(parent_key)
                self._time_offsets.pop(old_subcode)
                self._child_channels.pop(parent_key)
                self._pid_workers.pop(parent_key)
                self._cpu_time.pop(parent_key)

                old_subcode.cleanup_code()
                old_subcode.stop()

            else:
                self.hibernate_workers(pid)

    @property
    def model_time(self) -> float:
        """Extract the global integrator model time"""
        return self.parent_code.model_time

    @property
    def children(self) -> dict:
        """Extract the children system"""
        return self.particles.collection_attributes.subsystems

    @property
    def avail_cpus(self) -> int:
        """
        Extract the number of available CPUs, computed by considering:
            - Number of parent workers.
            - One SeBa worker.
            - One worker for main process.
            - Buffer so no overloading.
        """
        Nchildren = len(self.children)
        Nworkers = self.__total_free_cpus - self.__par_nworker - 3
        ncpu = min(Nchildren, Nworkers)
        return max(1, ncpu)  # Ensure at least one CPU is available

    @property
    def num_workers(self) -> int:
        """Extract the number of workers currently active."""
        nworker = max(1, len(self.children) // 5)
        return min(nworker, self.avail_cpus)
