from concurrent.futures import ThreadPoolExecutor, as_completed
import ctypes
import gc
import numpy as np
from numpy.ctypeslib import ndpointer
import os
import queue
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

from src.environment_functions import set_parent_radius
from src.environment_functions import planet_radius, ZAMS_radius
from src.grav_correctors import CorrectionFromCompoundParticle
from src.grav_correctors import CorrectionForCompoundParticle
from src.hierarchical_particles import HierarchicalParticles


############################## CURRENT BOTTLE NECKS ##################################
# 1. Evolving subsystems in _evolve_coll_offset --> ONE-BY-ONE, NOT PARALLELISED
# 2. SPLIT_SUBCODES --> SPLITTING SYSTEMS, NOT PARALLELISED (MEMORY ISSUES)
#### OTHER ROOM FOR IMPROVEMENT:
# 1. DEFINE BRIDGE TIME BETTER --> CURRENTLY FIXED
# 2. DEFINE PARENT SYSTEM BETTER --> CURRENTLY IGNORES NEAREST NEIGHBOURS
# 3. FOR ASTEROIDS, USE A DIFFERENT INTEGRATOR --> CURRENTLY USING PH4
# 4. FOR ASTEROIDS, TOO MANY SPLITS AND RE-MERGES --> HAVE A FILTER TO REDUCE THE `COMETS`
# 5. REMOVE EJECTED SYSTEMS
######################################################################################



class Nemesis(object):
    def __init__(self, min_stellar_mass, par_conv, dtbridge, 
                 coll_dir, available_cpus=os.cpu_count(),
                 eps=1.e-8, code_dt=0.03, par_nworker=1, dE_track=False, 
                 star_evol=False, gal_field=True, verbose=True):
        """
        Class setting up the simulation.
        
        Args:
            min_stellar_mass (units.mass):  Minimum stellar mass for stellar evolution
            par_conv (converter):  Parent N-body converter
            dtbridge (units.time):  Diagnostic time step
            coll_dir (str):  Path to store collision data
            available_cpus(int):  Number of available CPUs
            eps (float):  Threshold for evolution time
            code_dt (float):  Internal time step
            par_nworker (int):  Number of workers for global integrator
            dE_track (bool):  Flag turning on/off energy error tracker
            star_evol (bool):  Flag turning on/off stellar evolution
            gal_field (bool):  Flag turning on/off galactic field
            verbose (bool):  Flag turning on/off verbose output
        """
        
        # Private attributes
        self.__child_conv = nbody_system.nbody_to_si(1. | units.MSun, 500. | units.au)
        self.__parent_conv = par_conv
        self.__min_mass_evol_evol = min_stellar_mass
        self.__dt = dtbridge
        self.__coll_dir = coll_dir
        self.__code_dt = code_dt
        self.__par_nworker = par_nworker
        self.__star_evol = star_evol
        self.__gal_field = gal_field
        self.__verbose = verbose
        self.__eps = eps
        self.__total_free_cpus = available_cpus
        self.__lock = threading.Lock()
        self.__main_process = psutil.Process(os.getpid())
        self.__nmerge = 0
        
        # Protected attributes
        self._dE_track = dE_track
        self._time_offsets = dict()
        self._pid_workers = dict()
        self._child_channels = dict()
        self._max_radius = 1500. | units.au
        self._MWG = MWpotentialBovy2015()
        self._parent_code = self._parent_worker()
        self.grav_coll = self._parent_code.stopping_conditions.collision_detection
        self.grav_coll.enable()

        if (self.__star_evol):
            self._stellar_code = self._stellar_worker()
            self.SN_detection = self._stellar_code.stopping_conditions.supernova_detection
            self.SN_detection.enable()
        
        self.particles = HierarchicalParticles(self._parent_code.particles)
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
        if not isinstance(self.__min_mass_evol_evol.value_in(units.MSun), float) \
            or self.__min_mass_evol_evol <= 0 | units.kg:
                raise ValueError(f"Error: minimum stellar mass {self.__min_mass_evol_evol} must be a positive float")
        if not isinstance(self.__coll_dir, str):
            raise ValueError("Error: coll_dir must be a string")

    def _load_grav_lib(self) -> ctypes.CDLL:
        """Setup library to allow Python and C++ communication"""
        lib = ctypes.CDLL('./src/build/kick_particles_worker.so')
        lib.find_gravity_at_point.argtypes = [
            ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
            ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
            ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
            ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
            ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
            ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
            ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
            ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
            ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
            ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
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
        length_unit = particles.radius.unit
        if not hasattr(particles, "sub_worker_radius"):
            particles.sub_worker_radius = 0. | length_unit

        # Clean-up any old subcodes if needed
        for parent, code in self.subcodes.items():
            if ((parent in self.subsystems) and \
                (self.subsystems[parent] is self.subcodes[parent].particles)):
                continue
            self._time_offsets.pop(code)
            del code

        if (self.__star_evol):
            parti = particles.all()
            star_mask = parti.mass > self.__min_mass_evol_evol
            self.stars = parti[star_mask]
            self._stellar_code.particles.add_particle(self.stars)
                    
        particles.radius = set_parent_radius(particles.mass)
        ntotal = len(self.subsystems.keys())
        for nsyst, (parent, sys) in enumerate(self.subsystems.items()):
            sys.move_to_center()
            parent.radius = set_parent_radius(parent.mass)
            
            if parent not in self.subcodes:
                code = self._sub_worker(sys)
                
                self._time_offsets[code] = self.model_time
                self.subsystems[parent] = sys
                self.subcodes[parent] = code
                self._child_channel_maker(
                    parent=parent, 
                    code_particles=code.particles, 
                    children=sys
                    )
                
                # Store children PID to allow hibernation
                worker_pid = self.get_child_pid()
                self._pid_workers[parent] = worker_pid
                self.hibernate_workers(worker_pid)
                
            if self.__verbose:
                print(f"System {nsyst+1}/{ntotal}, radius: {parent.radius.in_(units.au)}")
        
        self.particles.recenter_subsystems(max_workers=self.avail_cpus)
        overly_massive = particles.radius > self._max_radius
        particles[overly_massive].radius = self._max_radius

        if (self.__gal_field):
            self._setup_bridge()
            
        else:
            self._evolve_code = self._parent_code
        
    def _setup_bridge(self) -> None:
        """Embed system into galactic potential"""
        gravity = bridge.Bridge(use_threading=True, method=SPLIT_4TH_S_M6,)
        gravity.add_system(self._parent_code, (self._MWG, ))
        gravity.timestep = self.__dt
        self._evolve_code = gravity
    
    def _stellar_worker(self) -> SeBa:
        """Define stellar evolution integrator"""
        return SeBa()

    def _parent_worker(self):
        """Define global integrator"""
        code = Ph4(self.__parent_conv, number_of_workers=self.__par_nworker)
        code.parameters.epsilon_squared = (0. | units.au)**2.
        code.parameters.timestep_parameter = self.__code_dt
        return code
      
    def _sub_worker(self, children: Particles):
        """
        Initialise children integrator.
        
        Args:
            children (Particles):  Children systems
        Returns:
            Code:  Gravitational integrator with particle set
        """
        if len(children) == 0:
            raise ValueError("Error: No children provided.")
    
        if (0. | units.kg) in children.mass:
            code = Ph4(self.__child_conv)
            code.parameters.epsilon_squared = (0. | units.au)**2.
            code.parameters.timestep_parameter = self.__code_dt
            code.particles.add_particles(children)
            return code
            
        else:
            code = Huayno(self.__child_conv)
            code.particles.add_particles(children)
            code.parameters.timestep_parameter = self.__code_dt
            code.set_integrator("SHARED8_COLLISIONS")
            return code
    
    def get_child_pid(self) -> int:
        """Returns the PID of the most recently spawned children worker"""
        for child in self.__main_process.children(recursive=True):
            if 'huayno_worker' in child.name():
                return child.pid
    
    def hibernate_workers(self, pid: int) -> None:
        """
        hibernate workers to reduce CPU usage
        
        Args:
            pid (int):  Process ID of worker
        """
        os.kill(pid, signal.SIGSTOP)
        
    def resume_workers(self, pid: int) -> None:
        """
        Resume workers to continue simulation
        
        Args:
            pid (int):  Process ID of worker
        """
        os.kill(pid, signal.SIGCONT)
     
    def _major_channel_maker(self) -> None:
        """Create channels for communication between codes"""
        parent_particles = self._parent_code.particles
        stellar_particles = self._stellar_code.particles
        self.channels = {
            "from_stellar_to_gravity":
                stellar_particles.new_channel_to(
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
    
    def _child_channel_maker(self, parent: Particle, code_particles: Particles, children: Particles) -> None:
        """
        Create communication channel between codes and specified children system
        
        Args:
            parent (Particle):  Parent particle
            code_particles (Particles):  Children particle set in grav. code
            children (Particles): Children particle set in local memory
        """
        grav_copy_variables = [
            "x", "y", "z", 
            "vx", "vy", "vz", 
            "radius", "mass"
            ]
        
        self._child_channels[parent] = {
            "from_star_to_gravity":
                self._stellar_code.particles.new_channel_to(
                    code_particles,
                    attributes=["mass", "radius"],
                    target_names=["mass", "radius"]
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
    
    def _calculate_total_energy(self) -> float:
        """
        Calculate systems total energy
        
        Returns:
            float:  Cluster total energy
        """
        all_parts = self.particles.all()
        Ek = all_parts.kinetic_energy()
        Ep = all_parts.potential_energy()
        Etot = Ek + Ep
        all_parts.remove_particles(all_parts)
        return Etot
      
    def _star_channel_copier(self) -> None:
        """Copy attributes from stellar code to grav. integrator particle set"""
        self.channels["from_stellar_to_gravity"].copy()
        for parent, channel in self._child_channels.items():
            pid = self._pid_workers[parent]
            self.resume_workers(pid)
            channel["from_star_to_gravity"].copy()
            self.hibernate_workers(pid)
        
    def _sync_grav_to_local(self) -> None:
        """Sync gravity particles to local set"""
        self.channels["from_gravity_to_parents"].copy()
        for parent, channel in self._child_channels.items():
            pid = self._pid_workers[parent]
            self.resume_workers(pid)
            channel["from_gravity_to_children"].copy()
            self.hibernate_workers(pid)
        
    def _sync_local_to_grav(self) -> None:
        """Sync local particle set to global integrator"""
        self.particles.recenter_subsystems(max_workers=self.avail_cpus)
        self.channels["from_parents_to_gravity"].copy()
        for parent, channel in self._child_channels.items():
            pid = self._pid_workers[parent]
            self.resume_workers(pid)
            channel["from_children_to_gravity"].copy()
            self.hibernate_workers(pid)
       
    def evolve_model(self, tend, timestep=None):
        """
        Evolve the system until tend
        
        Args:
            tend (units.time):  Time to simulate till
            timestep (units.time):  Timestep to simulate
        """
        if timestep is None:
            timestep = tend - self.model_time
        
        evolve_time = self.model_time
        while self.model_time < (evolve_time + timestep) * (1. - self.__eps):
            self.corr_energy = 0. | units.J
            self.dt_step += 1
            
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
                
            self._split_subcodes()
            self.channels["from_parents_to_gravity"].copy()   
            
        gc.collect()
        
        if self.__verbose:
            print(f"Time: {self.model_time.in_(units.Myr)}")
            print(f"Parent code time: {self._parent_code.model_time.in_(units.Myr)}")
            for parent, code in self.subcodes.items():
                pid = self._pid_workers[parent]
                self.resume_workers(pid)
                total_time = code.model_time + self._time_offsets[code]
                
                if not abs(total_time - self.model_time)/self.__dt < 0.01:
                    print(f"Parent: {parent.key}, Kids: {code.particles.mass.in_(units.MJupiter)}", end=", ")
                    print(f"Excess simulation: {(total_time - self.model_time).in_(units.kyr)}")
                    
                self.hibernate_workers(pid)
                
            print(f"Stellar code time: {self._stellar_code.model_time.in_(units.Myr)}")
            print(f"#Children: {len(self.subsystems.keys())}")
            print("===" * 50)
            
    def _split_subcodes(self) -> None:
        """Check for any isolated children"""    
        if self.__verbose:
            print("...Checking Splits...")
        
        Nnew_parents = 0
        new_isolated_parents = Particles()
        for parent, subsys in list(self.subsystems.items()):
            radius = parent.radius
            host = subsys[subsys.mass.argmax()]
            child_pos = subsys.position
            host_pos = host.position
            furthest = (child_pos - host_pos).lengths().max()
            
            if furthest > radius:
                components = subsys.connected_components(threshold=2 * radius)

                if len(components) > 1:
                    if self.__verbose:
                        print("...Split Detected...")
                    parent_pos = parent.position
                    parent_vel = parent.velocity
                    
                    pid = self._pid_workers.pop(parent)
                    self.resume_workers(pid)
                    self.particles.remove_particle(parent)
                    code = self.subcodes.pop(parent)
                    offset = self._time_offsets.pop(code)
                    channel = self._child_channels.pop(parent)

                    for c in components:
                        sys = c.copy()
                        sys.position += parent_pos
                        sys.velocity += parent_vel
                        
                        if len(sys) > 1 and max(sys.mass) > (0. | units.kg):
                            Nnew_parents += 1
                            newparent = self.particles.add_subsystem(sys)
                                
                            newcode = self._sub_worker(sys)
                            worker_pid = self.get_child_pid()
                            
                            self.subcodes[newparent] = newcode
                            self._time_offsets[newcode] = self.model_time
                            self._child_channel_maker(
                                parent=newparent, 
                                code_particles=newcode.particles, 
                                children=sys
                                )
                            self._pid_workers[newparent] = worker_pid
                            self.hibernate_workers(worker_pid)

                        else:
                            Nnew_parents += len(sys)
                            new_isolated_parents.add_particles(sys)

                    code.cleanup_code()
                    code.stop()

                    del code, subsys, parent, pid, offset, channel
        
        Nnew_isolated = len(new_isolated_parents)
        if Nnew_parents > 0:
            self.particles.add_particles(new_isolated_parents)
            new_parents = self.particles[-Nnew_parents:]
            
            self.particles[-Nnew_isolated:].syst_id = -1
            new_parents.radius = set_parent_radius(new_parents.mass)
            overly_massive = new_parents.radius > self._max_radius
            new_parents[overly_massive].radius = self._max_radius

    def _create_new_children(self, job_queue: queue) -> None:
        """
        Create new children systems based on parent mergers.
        
        Args:
            job_queue (queue):  Queue of jobs, each hosting new parent systems
        """
        new_children, time_offset = job_queue.get()
        
        asteroids = new_children[new_children.mass == (0. | units.kg)]
        stellar_masses = new_children.mass > self.__min_mass_evol_evol
        stars = new_children[stellar_masses]
        stars.radius = ZAMS_radius(stars.mass)
        
        planets = new_children[~stellar_masses]
        if asteroids:
            planets -= asteroids
        for p in planets:
            p.radius = planet_radius(p.mass)
            
        if self.__verbose:
            print(f"New Children has: {len(asteroids)} asteroids, {len(planets)} planets, {len(stars)} stars")
        
        with self.__lock:
            newparent = self.particles.add_subsystem(new_children)
            newparent.radius = set_parent_radius(newparent.mass)
            newcode = self._sub_worker(new_children)
            worker_pid = self.get_child_pid()
            
            self._time_offsets[newcode] = time_offset
            self.subcodes[newparent] = newcode
            self.subsystems[newparent] = new_children
            self._child_channel_maker(
                parent=newparent, 
                code_particles=newcode.particles, 
                children=new_children
                )
            self._pid_workers[newparent] = worker_pid
            self.hibernate_workers(worker_pid)
            
    def _process_parent_mergers(self) -> None:
        """Process merging of parents from previous timestep in parallel"""
        self.channels["from_gravity_to_parents"].copy()
        
        job_queue = queue.Queue()
        nmergers = len(self._new_systems.keys())
        for temp_parent, new_children in self._new_systems.items():
            n = temp_parent.as_particle_in_set(self._parent_code.particles)
            time_offset = self._new_offsets[temp_parent]
                    
            new_children.position += n.position
            new_children.velocity += n.velocity
            
            job_queue.put((new_children, time_offset))
            self.particles.remove_particle(temp_parent)
        
        threads = [ ]
        for _ in range(nmergers):
            th = threading.Thread(target=self._create_new_children, 
                                  args=(job_queue, ))
            th.start()
            threads.append(th)
            
        for th in threads:
            th.join()
            
        job_queue.queue.clear()
        del job_queue, self._new_systems, self._new_offsets
        
        new_parents = self.particles[-nmergers:]
        overly_massive = new_parents.radius > self._max_radius
        new_parents[overly_massive].radius = self._max_radius
    
    def _parent_merger(self, coll_time, coll_set: Particles) -> Particle:
        """
        Resolve the merging of two parent systems.
        
        Args:
            coll_time (units.time):  Time of collision
            coll_set (Particles):  Colliding particle set
        Returns:
            Particle:  Superset containing new parent and children
        """
        self.channels["from_gravity_to_parents"].copy()
        coll_array = self._evolve_coll_offset(coll_set, coll_time)
        collsubset = coll_array[0] 
        
        newparts = HierarchicalParticles(Particles())
        for parti_ in collsubset:
            parti_ = parti_.as_particle_in_set(self.particles)
            
            if parti_ in self.subcodes:
                pid = self._pid_workers.pop(parti_)
                self.resume_workers(pid)
                code = self.subcodes.pop(parti_)
                offset = self._time_offsets.pop(code)
                channel = self._child_channels.pop(parti_)
                
                sys = self.subsystems.pop(parti_)
                sys.position += parti_.position
                sys.velocity += parti_.velocity
                newparts.add_particles(sys)

                code.cleanup_code()
                code.stop()

                del code, pid, sys, offset, channel
              
            else:
                new_parti = newparts.add_particle(parti_)
                new_parti.radius = parti_.sub_worker_radius
                
            self.particles.remove_particle(parti_)
            
        # Temporary new parent particle
        newparent = Particle()
        newparent.mass = collsubset.total_mass()
        if min(collsubset.mass) > 0. | units.kg:
            newparent.position = collsubset.center_of_mass()
            newparent.velocity = collsubset.center_of_mass_velocity()
        else:
            most_massive = collsubset[collsubset.mass.argmax()]
            newparent.position = most_massive.position
            newparent.velocity = most_massive.velocity
        
        self.particles.add_particle(newparent)
        newparts.move_to_center()
        
        self._new_systems[newparent] = newparts
        self._new_offsets[newparent] = self.model_time
        
        return newparent

    def _evolve_coll_offset(self, coll_set: Particles, coll_time) -> list:
        """
        Function to evolve and/or resync the final moments of collision.
        
        Args:
            coll_set (Particles):  Attributes of colliding particle
            coll_time (units.time):  Time of simulation where collision occurs
        Returns:
            List: Index 0 contains parent colliders, index 1 childrens of merging parents
        """
        ##### HUGE BOTTLE NECK WHEN OFTEN CALLED
        
        collsubset = Particles()
        collsyst = dict()
        for parti_ in coll_set:
            collsubset.add_particle(parti_)

            # If a recently merged parent merges in the same time-loop, you need to give it children
            # in case of children collisions occuring during the evolution.
            code = None
            if parti_ in self._new_systems:  
                if self.__verbose:
                    print("...Parent Merging Again...")

                collsubset.remove_particle(parti_)
                evolved_parent = parti_.as_particle_in_set(self._parent_code.particles)
               
                offset = self._new_offsets.pop(parti_)
                children = self._new_systems.pop(parti_)
                children.position += evolved_parent.position
                children.velocity += evolved_parent.velocity

                newcode = self._sub_worker(children)
                newcode.particles.move_to_center()

                newparent = self.particles.add_subsystem(children)
                newparent.radius = set_parent_radius(newparent.mass)

                self.subcodes[newparent] = newcode
                self.subsystems[newparent] = children
                self._time_offsets[newcode] = offset
                self._child_channel_maker(
                    parent=newparent, 
                    code_particles=newcode.particles, 
                    children=children
                    )

                worker_pid = self.get_child_pid()
                self._pid_workers[newparent] = worker_pid
                self._new_systems.pop(parti_)
                self.particles.remove_particle(parti_)
                collsubset.add_particle(newparent)
                parti_ = newparent  # Redefine parent for further processing
                
                code = newcode
            
            elif parti_ in self.subcodes:
                worker_pid = self._pid_workers[parti_]
                self.resume_workers(worker_pid)
                
                code = self.subcodes[parti_]
                offset = self._time_offsets[code]
                newparent = parti_.copy()
            
            if (code):
                if self.__verbose:
                    dt = (coll_time - offset - code.model_time)
                    print(f"Evolving {len(code.particles)} for: {dt.in_(units.kyr)}")
                
                stopping_condition = code.stopping_conditions.collision_detection
                stopping_condition.enable()
                while code.model_time < (coll_time - offset)*(1. - self.__eps):
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
                                
                                colliding_particles = Particles(particles=[coll_a, coll_b])
                                newparent, resolved_keys = self._handle_collision(self.subsystems[newparent], 
                                                                                  newparent, colliding_particles, 
                                                                                  code, resolved_keys)
                                Nresolved += 1
                        
                        collsubset.add_particle(newparent)
                
                if len(code.particles) > 1:
                    self._child_channels[newparent]["from_gravity_to_children"].copy()
                    self.hibernate_workers(self._pid_workers[newparent])
                
                else:  #  Require dereferencing subsys as well?
                    code = self.subcodes.pop(newparent)
                    offset = self._time_offsets.pop(code)
                    channel = self._child_channels.pop(newparent)

                    code.stop()
                    del code, offset, channel

        for parti_ in collsubset:              
            if parti_ in self.subsystems:
                collsyst[parti_] = self.subsystems[parti_]
                
        return [collsubset, collsyst]
    
    def _handle_collision(self, children: Particles, parent: Particle, enc_parti: Particles, code, resolved_keys: dict):
        """
        Merge two particles if the collision stopping condition is met
        
        Args:
            children (Particles):  The children particle set
            parent (Particle):  The parent particle
            enc_parti (Particles): The particles in the collision
            code (Code):  The integrator used
            resolved_keys (dict):  Dictionary holding {Collider i Key: Remnant Key}
            end_time (float):  The time the old code will evolve to. Used for new codes offset
        Returns:
            Particles:  New parent particle alongside dictionary of merging particles keys
        """
        # Save properties
        self.__nmerge += 1
        print(f"...Collision #{self.__nmerge} Detected...")
        self._child_channels[parent]["from_gravity_to_children"].copy()
        
        coll_a = children[children.key == enc_parti[0].key]
        coll_b = children[children.key == enc_parti[1].key]
        
        collider = Particles(particles=[coll_a, coll_b])
        kepler_elements = orbital_elements(collider, G=constants.G)
        sma = kepler_elements[2]
        ecc = kepler_elements[3]
        inc = kepler_elements[4]
        
        tcoll = code.model_time + self._time_offsets[code]
        with open(os.path.join(self.__coll_dir, f"merger{self.__nmerge}.txt"), 'w') as f:
            f.write(f"Tcoll: {tcoll.in_(units.yr)}")
            f.write(f"\nSystem ID: {parent.syst_id}")
            f.write(f"\nKey1: {enc_parti[0].key}")
            f.write(f"\nKey2: {enc_parti[1].key}")
            f.write(f"\nType1: {enc_parti[0].type}")
            f.write(f"\nType2: {enc_parti[1].type}")
            f.write(f"\nM1: {enc_parti[0].mass.in_(units.MSun)}")
            f.write(f"\nM2: {enc_parti[1].mass.in_(units.MSun)}")
            f.write(f"\nSemi-major axis: {abs(sma).in_(units.au)}")
            f.write(f"\nEccentricity: {ecc}")
            f.write(f"\nInclination: {inc} deg")
            
        write_set_to_file(
            self.particles, 
            os.path.join(self.__coll_dir, f"Cluster_Merger{self.__nmerge}.hdf5"),
            'amuse', close_file=True, overwrite_file=True
        )
          
        write_set_to_file(
            children, 
            os.path.join(self.__coll_dir, f"System_Merger{self.__nmerge}.hdf5"),
            'amuse', close_file=True, overwrite_file=True
        )
        
        # Create merger remnant
        most_massive = collider[collider.mass.argmax()]
        if min(collider.mass) == (0. | units.kg):
            remnant = Particles(particles=[most_massive])
            
        elif max(collider.mass) > (0 | units.kg):
            remnant  = Particles(1)
            remnant.mass = collider.total_mass()
            remnant.position = collider.center_of_mass()
            remnant.velocity = collider.center_of_mass_velocity()
            
            if remnant.mass > self.__min_mass_evol_evol:
                remnant.radius = ZAMS_radius(remnant.mass)
                self._stellar_code.particles.add_particle(remnant)
                self.stars.add_particle(remnant)
            else:
                remnant.radius = planet_radius(remnant.mass)
                
            if coll_a.mass > self.__min_mass_evol_evol:
                self._stellar_code.particles.remove_particle(coll_a)
                self.stars.remove_particle(coll_a)
            if coll_b.mass > self.__min_mass_evol_evol:
                self._stellar_code.particles.remove_particle(coll_b)
                self.stars.remove_particle(coll_b)
                
            remnant.sub_worker_radius = remnant.radius
            
        else:
            raise ValueError("Error: Asteroid - Asteroid collision")
                
        remnant.coll_events = max(collider.coll_events) + 1
        remnant.type = most_massive.type
        
        # Deal with simultaneous mergers
        changes = [ ]
        coll_a_change = coll_b_change = 0
        if not resolved_keys:
            resolved_keys[coll_a.key[0]] = remnant.key[0]
            resolved_keys[coll_b.key[0]] = remnant.key[0]
        else: 
            # If the current collider is a remnant of past event, remap
            for prev_collider, resulting_remnant in resolved_keys.items():
                if coll_a.key[0] == resulting_remnant:  
                    changes.append((prev_collider, remnant.key[0]))
                    coll_a_change = 1
                elif coll_b.key[0] == resulting_remnant:
                    changes.append((prev_collider, remnant.key[0]))
                    coll_b_change = 1
                    
            if coll_a_change == 0:
                resolved_keys[coll_a.key[0]] = remnant.key[0]
            if coll_b_change == 0:
                resolved_keys[coll_b.key[0]] = remnant.key[0]
       
        for key, new_value in changes:
            resolved_keys[key] = new_value
        
        print(f"{coll_a.type}, {coll_b.type}")
        print(f"{coll_a.mass.in_(units.MSun)} + {coll_b.mass.in_(units.MSun)} --> {remnant.mass.in_(units.MSun)}")
        
        children.remove_particles(coll_a)
        children.remove_particles(coll_b)
        children.add_particles(remnant)
        if min(collider.mass) > (0. | units.kg):
            nearest_mass = abs(children.mass - parent.mass).argmin()
            if remnant.key == children[nearest_mass].key:  # If the remnant is the host
                children.position += parent.position
                children.velocity += parent.velocity
                
                newparent = self.particles.add_subsystem(children)
                newparent.radius = parent.radius

                # Re-mapping dictionary to new parent
                old_code = self.subcodes.pop(parent)
                old_offset = self._time_offsets.pop(old_code)
                old_channel = self._child_channels.pop(parent)
                
                self.subcodes[newparent] = old_code
                new_code = self.subcodes[newparent]
                self._time_offsets[new_code] = old_offset
                self._child_channel_maker(
                    parent=newparent, 
                    code_particles=new_code.particles, 
                    children=children
                    )
                child_pid = self._pid_workers.pop(parent)
                self._pid_workers[newparent] = child_pid
                
                self.particles.remove_particle(parent)
                
                del old_channel  # Check if this breaks

            else:
                newparent = parent
        else:
            newparent = parent
            
        children.synchronize_to(self.subcodes[newparent].particles)
          
        return newparent, resolved_keys
    
    def _handle_supernova(self, SN_detect, bodies: Particles) -> None:
        """
        Handle SN events
        
        Args:
            SN_detect (StoppingCondition):  Detected particle set undergoing SN
            bodies (Particles):  All bodies undergoing stellar evolution
        """
        if (self._dE_track):
            E0 = self._calculate_total_energy()
            
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
            
        if (self._dE_track):
            E1 = self._calculate_total_energy()
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
        Evolve stellar evolution
        
        Args:
            dt (units.time):  Time to evolve till
        """
        while self._stellar_code.model_time < dt * (1. - self.__eps):
            self._stellar_code.evolve_model(dt)
            
            if self.SN_detection.is_set():
                print("...Detection: SN Explosion...")
                self._handle_supernova(self.SN_detection, self.stars)
                    
    def _drift_global(self, dt) -> None:
        """
        Evolve parent system until dt
        
        Args:
            dt (units.time):  Time to evolve till
        """
        if self.__verbose:
            print("...Drifting Global...")
        
        self._new_systems = dict()
        self._new_offsets = dict()
        coll_time = None
        while self._evolve_code.model_time < dt * (1. - self.__eps):
            self._evolve_code.evolve_model(dt)
            
            if self.grav_coll.is_set():
                if (self._dE_track):
                    E0 = self._calculate_total_energy()
                    
                coll_time = self._parent_code.model_time
                coll_sets = self._find_coll_sets(self.grav_coll.particles(0), 
                                                 self.grav_coll.particles(1))
                
                if self.__verbose:
                    print(f"Bridge: Parent Merger. T = {coll_time.in_(units.Myr)}")
                for cs in coll_sets:
                    self._parent_merger(coll_time, cs)
                if (self._dE_track):
                    E1 = self._calculate_total_energy()
                    self.corr_energy += E1 - E0

        if (self.__gal_field):
            while self._parent_code.model_time < dt * (1. - self.__eps):
                self._parent_code.evolve_model(dt)
                if self.grav_coll.is_set():
                    if (self._dE_track):
                        E0 = self._calculate_total_energy()
                    
                    coll_time = self._parent_code.model_time
                    coll_sets = self._find_coll_sets(self.grav_coll.particles(0), 
                                                     self.grav_coll.particles(1))

                    if self.__verbose:
                        print(f"Parent: Parent Merger. T = {coll_time.in_(units.Myr)}")
                    for cs in coll_sets:
                        self._parent_merger(coll_time, cs)
                        
                    if (self._dE_track):
                        E1 = self._calculate_total_energy()
                        self.corr_energy += E1 - E0
                        
        if (coll_time):
            self._process_parent_mergers()
            
    def _drift_child(self, dt) -> None:
        """
        Evolve children system until dt.
        
        Args:
            dt (units.time): Time to evolve till.
        """
        def resolve_collisions(code, parent: Particle, stopping_condition):
            """
            Function to resolve collisions
            
            Args:
                code (Code):  Code with collision
                parent (Particle):  Parent particle
                stopping_condition (StoppingCondition):  Stopping condition to resolve
            """
            children = self.subsystems[parent]
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
        
        def evolve_code(parent: Particle):
            """
            Evolve children code until dt
            
            Args:
                parent (Particle):  Parent particle
            """
            try:
                self.resume_workers(self._pid_workers[parent])
                code = self.subcodes[parent]
                stopping_condition = code.stopping_conditions.collision_detection
                stopping_condition.enable()
                
                evol_time = dt - self._time_offsets[code]
                while code.model_time < evol_time * (1. - self.__eps):
                    code.evolve_model(evol_time)
                    
                    if stopping_condition.is_set():
                        with self.__lock:
                            if (self._dE_track):
                                KE = code.particles.kinetic_energy()
                                PE = code.particles.potential_energy()
                                E0 = KE + PE

                            parent = resolve_collisions(
                                        code, parent,
                                        stopping_condition
                                        )

                            if (self._dE_track):
                                KE = code.particles.kinetic_energy()
                                PE = code.particles.potential_energy()
                                E1 = KE + PE
                                self.corr_energy += E1 - E0

                self.hibernate_workers(self._pid_workers[parent])

            except Exception as e:
                print(f"Error while evolving parent {parent.key}: {e}")
                print(f"Traceback: {traceback.format_exc()}")

        if self.__verbose:
            print("...Drifting Children...")

        with ThreadPoolExecutor(max_workers=self.avail_cpus) as executor:
            futures = {executor.submit(evolve_code, parent):
                       parent for parent in self.subcodes.keys()}
            for future in as_completed(futures):  # Iterate over to ensure no silent failures
                parent = futures.pop(future)
                try:
                    future.result()
                except Exception as e:
                    print(f"Error while evolving parent {parent.key}: {e}")
        futures = None
        gc.collect()
        
        for parent in list(self.subcodes.keys()):  # Remove single children systems:
            pid = self._pid_workers[parent]
            self.resume_workers(pid)
            if len(self.subcodes[parent].particles) == 1:
                old_subcode = self.subcodes.pop(parent)
                old_offset = self._time_offsets.pop(old_subcode)
                old_channel = self._child_channels.pop(parent)
                pid = self._pid_workers.pop(parent)
                
                old_subcode.cleanup_code()
                old_subcode.stop()
                
                del old_subcode, old_offset, old_channel, pid 
            
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
        Apply correcting kicks onto children particles

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
                for parent, children in subsystems.items():
                    try:
                        future = process_children_jobs(parent, children)
                        futures.append(future)
                    except Exception as e:
                        print(f"Error submitting job for parent {parent.key}: {e}")
                        print(f"Traceback: {traceback.format_exc()}")

                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:
                        print(f"Error in CorrectionFor result: {e}")
                        print(f"Traceback: {traceback.format_exc()}")
                            
    @property
    def model_time(self) -> float:  
        """Extract the global integrator model time"""
        return self._parent_code.model_time
    
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
        return min(len(self.subsystems.keys()), self.__total_free_cpus - self.__par_nworker - 3)