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
from amuse.community.rebound.interface import Rebound
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


class Nemesis(object):
    def __init__(self, min_stellar_mass, par_conv, dtbridge, 
                 coll_dir, available_cpus=os.cpu_count(),
                 eps=1.e-8, code_dt=0.03, par_nworker=1, dE_track=False, 
                 star_evol=False, gal_field=True, verbose=True):
        """
        Class setting up the simulation.
        
        Args:
            min_stellar_mass (Float):  Minimum stellar mass for stellar evolution
            par_conv (Converter):  Parent N-body converter
            dtbridge (Float):  Diagnostic time step
            coll_dir (String):  Path to store collision data
            available_cpus(Int):  Number of available CPUs
            eps (Float):  Threshold for evolution time
            code_dt (Float):  Internal time step
            par_nworker (Int):  Number of workers for global integrator
            dE_track (Boolean):  Flag turning on/off energy error tracker
            star_evol (Boolean):  Flag turning on/off stellar evolution
            gal_field (Boolean):  Flag turning on/off galactic field
            verbose (Boolean):  Flag turning on/off verbose output
        """
        
        # Private attributes
        self.__child_conv = nbody_system.nbody_to_si(1. | units.MSun, 100. | units.au)
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
        self.__pid_workers = dict()
        self.__lock = threading.Lock()
        self.__main_process = psutil.Process(os.getpid())
        self.__nmerge = 0
        
        # Protected attributes
        self._max_radius = 200. | units.au
        self._nejec = 0
        self._dE_track = dE_track
        self._time_offsets = dict()
        
        self._MWG = MWpotentialBovy2015()
        self._parent_code = self._parent_worker()
        if (self.__star_evol):
            self._stellar_code = self._stellar_worker()
        
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
            ndpointer(dtype=np.float128, ndim=1, flags='C_CONTIGUOUS'),
            ndpointer(dtype=np.float128, ndim=1, flags='C_CONTIGUOUS'),
            ndpointer(dtype=np.float128, ndim=1, flags='C_CONTIGUOUS'),
            ndpointer(dtype=np.float128, ndim=1, flags='C_CONTIGUOUS'),
            ndpointer(dtype=np.float128, ndim=1, flags='C_CONTIGUOUS'),
            ndpointer(dtype=np.float128, ndim=1, flags='C_CONTIGUOUS'),
            ndpointer(dtype=np.float128, ndim=1, flags='C_CONTIGUOUS'),
            ndpointer(dtype=np.float128, ndim=1, flags='C_CONTIGUOUS'),
            ndpointer(dtype=np.float128, ndim=1, flags='C_CONTIGUOUS'),
            ndpointer(dtype=np.float128, ndim=1, flags='C_CONTIGUOUS'),
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
            - defining stellar code particles
            - setting up the galactic field
        """
        particles = self.particles
        length_unit = particles.radius.unit
        if not hasattr(particles, "sub_worker_radius"):
            particles.sub_worker_radius = 0. | length_unit

        for parent, code in self.subcodes.items():
            if ((parent in self.subsystems) and \
                (self.subsystems[parent] is self.subcodes[parent].particles)):
                continue
            self._time_offsets.pop(code)
            del code
        
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
                
                # Store children PID to allow hibernation
                worker_pid = self.get_child_pid()
                self.__pid_workers[parent] = worker_pid
                self.hibernate_workers(worker_pid)
                
            if (self.__verbose):
                print(f"System {nsyst+1}/{ntotal}, radius: {parent.radius.in_(units.au)}")
        
        self.particles.recenter_subsystems(max_workers=int(2 * self.avail_cpus))
        particles[particles.radius > self._max_radius].radius = self._max_radius

        if (self.__star_evol):
            parti = particles.all()
            self.stars = parti[parti.mass > self.__min_mass_evol_evol]
            stellar_code = self._stellar_code
            stellar_code.particles.add_particle(self.stars)

        if (self.__gal_field):
            self._setup_bridge()
            
        else:
            self._evolve_code = self._parent_code
            self.__tidal_radius = None

    def _extract_tidal_radius(self) -> None:
        """Compute the tidal radius of the system"""
        particles = self.particles.copy()
        massives = particles[particles.mass > (0. | units.kg)]
        self.__tidal_radius = tidal_radius(massives)
        print(f"Tidal radius: {self.__tidal_radius.in_(units.pc)}")
        
    def _setup_bridge(self) -> None:
        """Embed system into galactic potential"""
        gravity = bridge.Bridge(use_threading=True,
                                method=SPLIT_4TH_S_M6,)
        gravity.add_system(self._parent_code, (self._MWG, ))
        gravity.timestep = self.__dt
        self._evolve_code = gravity
    
    def _stellar_worker(self) -> SeBa:
        """Define stellar evolution integrator"""
        return SeBa()

    def _parent_worker(self):
        """
        Define global integrator
        
        Args:
            par_conv (converter):  Converter for global integrator
        Returns:
            Code:  Gravitational integrator with particle set
        """
        code = Ph4(self.__parent_conv, number_of_workers=self.__par_nworker)
        code.parameters.epsilon_squared = (0. | units.au)**2.
        code.parameters.timestep_parameter = self.__code_dt
        return code
      
    def _sub_worker(self, children):
        """
        Initialise children integrator
        
        Args:
            children (Particle set):  Children systems
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
            
        # TO DO: Add REBOUND integrator
        """elif masses[-1] > 100.*masses[-2]:
            code = Rebound(self.__child_conv)
            code.particles.add_particles(children)
            code.set_integrator("WHFast")"""
        
        return code
    
    def get_child_pid(self) -> int:
        """Returns the PID of the most recently spawned children worker"""
        for child in self.__main_process.children(recursive=True):
            if 'ph4_worker' in child.name() or 'huayno_worker' in child.name():
                return child.pid
    
    def hibernate_workers(self, pid) -> None:
        """
        hibernate workers to reduce CPU usage
        
        Args:
            pid (Int):  Process ID of worker
        """
        os.kill(pid, signal.SIGSTOP)
        
    def resume_workers(self, pid) -> None:
        """
        Resume workers to continue simulation
        
        Args:
            pid (Int):  Process ID of worker
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
                    attributes=["x","y","z","vx","vy","vz"],
                    target_names=["x","y","z","vx","vy","vz"]
                ),
            "from_parents_to_gravity":
                self.particles.new_channel_to(
                    parent_particles,
                    attributes=["x","y","z","vx","vy","vz"],
                    target_names=["x","y","z","vx","vy","vz"]
                )
        }
    
    def _calculate_total_energy(self) -> float:
        """
        Calculate systems total energy
        
        Returns:
            Float:  Cluster total energy
        """
        all_parts = self.particles.all()
        Ek = all_parts.kinetic_energy()
        Ep = all_parts.potential_energy()
        Etot = Ek + Ep
        return Etot
      
    def _star_channel_copier(self) -> None:
        """Copy attributes from stellar code to grav. integrator particle set"""
        stars = self._stellar_code.particles
        self.channels["from_stellar_to_gravity"].copy()
        
        for parent, code in self.subcodes.items():
            pid = self.__pid_workers[parent]
            self.resume_workers(pid)
            channel = stars.new_channel_to(code.particles)
            channel.copy_attributes(["mass", "radius"])
            self.hibernate_workers(pid)
            
    def _grav_channel_copier(self, transfer_data, receive_data, 
                             attributes=["x","y","z","vx","vy","vz","radius","mass"]) -> None:
        """
        Communicate information between grav. integrator and local particle set
        
        Args:
            transfer_data (Particle set):  Particle set to transfer data from
            receive_data (Particle set):  Particle set to update data
            attributes (Array):  Attributes wanting to copy
        """
        channel = transfer_data.new_channel_to(receive_data)
        channel.copy_attributes(attributes)
        
    def _sync_local_to_code(self) -> None:
        """Sync local particle set to global integrator"""
        self.particles.recenter_subsystems(max_workers=int(2 * self.avail_cpus))
        self.channels["from_parents_to_gravity"].copy()
        for parent, subsyst in self.subsystems.items():
            pid = self.__pid_workers[parent]
            self.resume_workers(pid)
            self._grav_channel_copier(
                subsyst,
                self.subcodes[parent].particles
            )
            self.hibernate_workers(pid)
       
    def evolve_model(self, tend, timestep=None):
        """
        Evolve the system until tend
        
        Args:
            tend (Float):  Time to simulate till
            timestep (Float):  Timestep to simulate
        """
        if timestep is None:
            timestep = tend - self.model_time
        
        evolve_time = self.model_time
        while self.model_time < (evolve_time + timestep)*(1. - self.__eps):
            self.corr_energy = 0. | units.J
            self.dt_step += 1

            if (self.__star_evol):
                self._stellar_evolution(evolve_time + timestep/2.)
                self._star_channel_copier()

            if evolve_time == 0. | units.yr:
                self._correction_kicks(
                    self.particles, 
                    self.subsystems,
                    timestep/2.
                )
                self._sync_local_to_code()

            self._drift_global(evolve_time + timestep)
            if self.subcodes:
                self._drift_child(self.model_time)

            if evolve_time == 0. | units.yr:
                self._correction_kicks(
                    self.particles, 
                    self.subsystems,
                    timestep/2.
                )
            else:
                self._correction_kicks(
                    self.particles, 
                    self.subsystems,
                    self.model_time - evolve_time
                )
            self._sync_local_to_code()
                
            if (self.__star_evol):
                self._stellar_evolution(self.model_time)
                
            self._split_subcodes()

        gc.collect()
        
        if (self.__verbose):
            print(f"Time: {self.model_time.in_(units.Myr)}")
            print(f"Parent code time: {self._parent_code.model_time.in_(units.Myr)}")
            for parent, code in self.subcodes.items():
                pid = self.__pid_workers[parent]
                self.resume_workers(pid)
                total_time = code.model_time + self._time_offsets[code]
                
                if not abs(total_time - self.model_time)/self.__dt < 0.01:
                    print(f"Parent: {parent.key}, Kids: {code.particles.mass.in_(units.MJupiter)}", end=", ")
                    print(f"Excess simulation: {(total_time - self.model_time).in_(units.kyr)}")
                    
                self.hibernate_workers(pid)
                
            print(f"Stellar code time: {self._stellar_code.model_time.in_(units.Myr)}")
            print(f"==================================================")
            
    def _split_subcodes(self) -> None:
        """Check for any isolated children"""    
        if (self.__verbose):
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
                components = subsys.connected_components(threshold=2. * radius)

                if len(components) > 1:  # Checking for dissolution of system
                    if (self.__verbose):
                        print("...Split Detected...")
                    parent_pos = parent.position
                    parent_vel = parent.velocity
                    
                    pid = self.__pid_workers.pop(parent)
                    self.resume_workers(pid)
                    self.particles.remove_particle(parent)
                    code = self.subcodes.pop(parent)
                    self._time_offsets.pop(code)

                    for c in components:
                        sys = c.copy_to_memory()
                        sys.position += parent_pos
                        sys.velocity += parent_vel
                        
                        if len(sys) > 1 and max(sys.mass) > (0. | units.kg):
                            Nnew_parents += 1
                            newparent = self.particles.add_subsystem(sys)
                                
                            newcode = self._sub_worker(sys)
                            self.subcodes[newparent] = newcode
                            self._time_offsets[newcode] = self.model_time

                            worker_pid = self.get_child_pid()
                            self.__pid_workers[newparent] = worker_pid
                            self.hibernate_workers(worker_pid)

                        else:
                            Nnew_parents += len(sys)
                            new_isolated_parents.add_particles(sys)

                    code.cleanup_code()
                    code.stop()

                    del code, subsys, parent, pid
        
        Nnew_isolated = len(new_isolated_parents)
        self.particles.add_particles(new_isolated_parents)
        new_parents = self.particles[-Nnew_parents:]
        
        self.particles[-Nnew_isolated:].syst_id = -1
        new_parents.radius = set_parent_radius(new_parents.mass)
        excessive_radius = new_parents.radius > self._max_radius
        new_parents[excessive_radius].radius = self._max_radius
        
        self.channels["from_parents_to_gravity"].copy()

    def _create_new_children(self, job_queue) -> None:
        """
        Create new children systems based on parent mergers.
        
        Args:
            job_queue (Queue):  Queue of jobs, each hosting new parent systems
        """
        new_children, time_offset = job_queue.get()
        
        asteroids = new_children[new_children.mass == (0. | units.kg)]
        stars = new_children[new_children.mass > self.__min_mass_evol_evol]
        stars.radius = ZAMS_radius(stars.mass)
        
        planets = new_children - stars
        if (asteroids):
            planets -= asteroids
        for p in planets:
            p.radius = planet_radius(p.mass)
            
        if (self.__verbose):
            print(f"New Children has: {len(asteroids)} asteroids, {len(planets)} planets, {len(stars)} stars")
        
        with self.__lock:
            newparent = self.particles.add_subsystem(new_children)
            newparent.radius = set_parent_radius(newparent.mass)
            newcode = self._sub_worker(new_children)
            
            self._time_offsets[newcode] = time_offset
            self.subcodes[newparent] = newcode
            self.subsystems[newparent] = new_children
            
            worker_pid = self.get_child_pid()
            self.__pid_workers[newparent] = worker_pid
            self.hibernate_workers(worker_pid)
            
    def _process_parent_mergers(self) -> None:
        """Process merging of parents from previous timestep in parallel"""
        self.channels["from_gravity_to_parents"].copy()
        
        job_queue = queue.Queue()
        nmergers = len(self.__new_systems.keys())
        for temp_parent, new_children in self.__new_systems.items():
            n = temp_parent.as_particle_in_set(self._parent_code.particles)
            time_offset = self.__new_offsets[temp_parent]
                    
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
        del job_queue
        
        # Modify radius
        new_parents = self.particles[-nmergers:]
        new_parents[new_parents.radius > self._max_radius].radius = self._max_radius
        self.particles.recenter_subsystems(max_workers=int(2 * self.avail_cpus))
    
    def _parent_merger(self, coll_time, coll_set) -> Particle:
        """
        Resolve the merging of two parent systems.
        
        Args:
            coll_time (Float):  Time of collision
            coll_set (Particle set):  Colliding particle set
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
                pid = self.__pid_workers.pop(parti_)
                self.resume_workers(pid)
                code = self.subcodes.pop(parti_)
                self._time_offsets.pop(code)
                
                sys = self.subsystems[parti_]
                sys.position += parti_.position
                sys.velocity += parti_.velocity
                newparts.add_particles(sys)

                code.cleanup_code()
                code.stop()

                del code, pid
              
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
        
        self.__new_systems[newparent] = newparts
        self.__new_offsets[newparent] = self.model_time
        
        return newparent

    def _evolve_coll_offset(self, coll_set, coll_time) -> list:
        """
        Function to evolve and/or resync the final moments of collision.
        
        Args:
            coll_set (Particle set):  Attributes of colliding particle
            coll_time (Float):  Time of simulation where collision occurs
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
            if parti_ in self.__new_systems:  
                if (self.__verbose):
                    print("...Parent Merging Again...")

                collsubset.remove_particle(parti_)
                evolved_parent = parti_.as_particle_in_set(self._parent_code.particles)
               
                offset = self.__new_offsets.pop(parti_)
                children = self.__new_systems.pop(parti_)
                children.position += evolved_parent.position
                children.velocity += evolved_parent.velocity

                newcode = self._sub_worker(children)
                newcode.particles.move_to_center()

                newparent = self.particles.add_subsystem(children)
                newparent.radius = set_parent_radius(np.sum(children.mass))

                self.subcodes[newparent] = newcode
                self.subsystems[newparent] = children
                self._time_offsets[newcode] = offset

                worker_pid = self.get_child_pid()
                self.__pid_workers[newparent] = worker_pid
                self.__new_systems.pop(parti_)
                self.particles.remove_particle(parti_)
                collsubset.add_particle(newparent)
                parti_ = newparent
                
                code = newcode
            
            elif parti_ in self.subcodes:
                worker_pid = self.__pid_workers[parti_]
                self.resume_workers(worker_pid)
                
                code = self.subcodes[parti_]
                offset = self._time_offsets[code]
            
            if (code):
                if (self.__verbose):
                    print("Evolving until: ", (coll_time - offset).in_(units.kyr))
                    print("Evolving for: ", (coll_time - offset - code.model_time).in_(units.kyr))
                stopping_condition = code.stopping_conditions.collision_detection
                stopping_condition.enable()
                newparent = parti_.copy()
                while code.model_time < (coll_time - offset)*(1. - self.__eps):
                    code.evolve_model(coll_time - offset)
                    
                    if stopping_condition.is_set():
                        coll_time = code.model_time
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
                    self._grav_channel_copier(
                        code.particles, 
                        self.subsystems[newparent]
                    )
                    self.hibernate_workers(self.__pid_workers[newparent])
                
                else:
                    code = self.subcodes.pop(newparent)
                    offset = self._time_offsets.pop(code)

                    code.stop()
                    del code, offset

        for parti_ in collsubset:              
            if parti_ in self.subsystems:
                collsyst[parti_] = self.subsystems[parti_]
                
        return [collsubset, collsyst]
    
    def _handle_collision(self, children, parent, enc_parti, code, resolved_keys):
        """
        Merge two particles if the collision stopping condition is met
        
        Args:
            children (Particle set):  The children particle set
            parent (Particle):  The parent particle
            enc_parti (Particle set): The particles in the collision
            code (Code):  The integrator used
            resolved_keys (Dict):  Dictionary holding {Collider i Key: Remnant Key}
            end_time (Float):  The time the old code will evolve to. Used for new codes offset
        Returns:
            Particles:  New parent particle alongside dictionary of merging particles keys
        """
        # Save properties
        self.__nmerge += 1
        print(f"...Collision #{self.__nmerge} Detected...")
        self._grav_channel_copier(code.particles, children)
        
        coll_a = children[children.key == enc_parti[0].key]
        coll_b = children[children.key == enc_parti[1].key]
        
        collider = Particles(particles=[coll_a, coll_b])
        kepler_elements = orbital_elements(collider, G=constants.G)
        sem = kepler_elements[2]
        ecc = kepler_elements[3]
        inc = kepler_elements[4]
        
        tcoll = code.model_time + self._time_offsets[code]
        with open(os.path.join(self.__coll_dir, f"merger{self.__nmerge}.txt"), 'w') as f:
            f.write(f"Tcoll: {tcoll.in_(units.yr)}")
            f.write(f"\nKey1: {enc_parti[0].key}")
            f.write(f"\nKey2: {enc_parti[1].key}")
            f.write(f"\nM1: {enc_parti[0].mass.in_(units.MSun)}")
            f.write(f"\nM2: {enc_parti[1].mass.in_(units.MSun)}")
            f.write(f"\nSemi-major axis: {abs(sem).in_(units.au)}")
            f.write(f"\nEccentricity: {ecc}")
            f.write(f"\nInclination: {inc} deg")
        
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
            else:
                remnant.radius = planet_radius(remnant.mass)
            remnant.sub_worker_radius = remnant.radius
            if coll_a.mass > self.__min_mass_evol_evol:
                self._stellar_code.particles.remove_particle(coll_a)
            if coll_b.mass > self.__min_mass_evol_evol:
                self._stellar_code.particles.remove_particle(coll_b)
            
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
            if remnant.key == children[nearest_mass].key:
                children.position += parent.position
                children.velocity += parent.velocity
                
                newparent = self.particles.add_subsystem(children)
                newparent.radius = parent.radius

                # Re-mapping dictionary to new parent
                old_code = self.subcodes.pop(parent)
                old_offset = self._time_offsets.pop(old_code)
                
                self.subcodes[newparent] = old_code
                new_code = self.subcodes[newparent]
                self._time_offsets[new_code] = old_offset
                child_pid = self.__pid_workers.pop(parent)
                self.__pid_workers[newparent] = child_pid
                
                self.particles.remove_particle(parent)

            else:
                newparent = parent
        else:
            newparent = parent
            
        children.synchronize_to(self.subcodes[newparent].particles)
          
        return newparent, resolved_keys
    
    def _handle_supernova(self, SN_detect, bodies) -> None:
        """
        Handle SN events
        
        Args:
            SN_detect (StoppingCondition):  Detected particle set undergoing SN
            bodies (Particle set):  All bodies undergoing stellar evolution
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
            
    def _find_coll_sets(self, p1, p2) -> UnionFind:
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
            dt (Float):  Time to evolve till
        """
        SN_detection = self._stellar_code.stopping_conditions.supernova_detection
        SN_detection.enable()
        while self._stellar_code.model_time < dt*(1. - self.__eps):
            self._stellar_code.evolve_model(dt)
            
            if SN_detection.is_set():
                print("...Detection: SN Explosion...")
                self._handle_supernova(SN_detection, self.stars)
                    
    def _drift_global(self, dt) -> None:
        """
        Evolve parent system until dt
        
        Args:
            dt (Float):  Time to evolve till
        """
        if (self.__verbose):
            print("...Drifting Global...")
        stopping_condition = self._parent_code.stopping_conditions.collision_detection
        stopping_condition.enable()
        
        self.__new_systems = dict()
        self.__new_offsets = dict()
        coll_time = None
        while self._evolve_code.model_time < dt*(1. - self.__eps):
            self._evolve_code.evolve_model(dt)
            
            if stopping_condition.is_set():
                if (self._dE_track):
                    E0 = self._calculate_total_energy()
                    
                coll_time = self._parent_code.model_time
                coll_sets = self._find_coll_sets(stopping_condition.particles(0), 
                                                 stopping_condition.particles(1))
                
                if (self.__verbose):
                    print(f"Bridge: Parent Merger. T = {coll_time.in_(units.Myr)}")
                for cs in coll_sets:
                    self._parent_merger(coll_time, cs)
                if (self._dE_track):
                    E1 = self._calculate_total_energy()
                    self.corr_energy += E1 - E0

        if (self.__gal_field):
            while self._parent_code.model_time < dt*(1. - self.__eps):
                self._parent_code.evolve_model(dt)
                if stopping_condition.is_set():
                    if (self._dE_track):
                        E0 = self._calculate_total_energy()
                    
                    coll_time = self._parent_code.model_time
                    coll_sets = self._find_coll_sets(stopping_condition.particles(0), 
                                                     stopping_condition.particles(1))

                    if (self.__verbose):
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
            dt (Float): Time to evolve till.
        """
        def resolve_collisions(code, parent, evol_time, stopping_condition):
            """
            Function to resolve collisions
            
            Args:
                code (Code):  Code with collision
                parent (Particle):  Parent particle
                evol_time (Float):  Time to evolve till
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
                                                    code, resolved_keys)
                    Nresolved += 1

            return parent
        
        def evolve_code(parent):
            """
            Evolve children code until dt
            
            Args:
                parent (Particle):  Parent particle
            """
            try:
                self.resume_workers(self.__pid_workers[parent])
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
                                        evol_time, 
                                        stopping_condition
                                        )

                            if (self._dE_track):
                                KE = code.particles.kinetic_energy()
                                PE = code.particles.potential_energy()
                                E1 = KE + PE
                                self.corr_energy += E1 - E0

                self.hibernate_workers(self.__pid_workers[parent])

            except Exception as e:
                print(f"Error while evolving parent {parent.key}: {e}")
                print(f"Traceback: {traceback.format_exc()}")

        if (self.__verbose):
            print("...Drifting Children...")

        with ThreadPoolExecutor(max_workers=self.avail_cpus) as executor:
            futures = {executor.submit(evolve_code, parent):
                       parent for parent in list(self.subcodes.keys())
                       }
            for future in as_completed(futures):  #Iterate over to ensure no silent failures
                parent = futures[future]
                try:
                    future.result()
                except Exception as e:
                    print(f"Error while evolving parent {parent.key}: {e}")

        for parent in list(self.subcodes.keys()): # Remove single children systems:
            pid = self.__pid_workers[parent]
            self.resume_workers(pid)
            if len(self.subcodes[parent].particles) == 1:
                old_subcode = self.subcodes.pop(parent)
                old_offset = self._time_offsets.pop(old_subcode)
                pid = self.__pid_workers.pop(parent)
                
                old_subcode.cleanup_code()
                old_subcode.stop()
                del old_subcode, old_offset, pid
            
            else:
                self.hibernate_workers(pid)

    def _kick_particles(self, particles, corr_code, dt) -> None:
        """
        Apply correction kicks onto target particles
        
        Args:
            particles (Particle set):  Particles to correct accelerations of
            corr_code (Code):  Code containing information on difference in gravitational field
            dt (Float):  Time-step of correction kick
        """
        parts = particles.copy_to_memory()
        ax, ay, az = corr_code.get_gravity_at_point(
                        parts.radius,
                        parts.x, 
                        parts.y, 
                        parts.z
                    )
        
        parts.vx += dt * ax
        parts.vy += dt * ay
        parts.vz += dt * az
        
        channel = parts.new_channel_to(particles)
        channel.copy_attributes(["vx","vy","vz"])

    def _correct_children(self, particles, parent, subsystem, dt) -> None:
        """
        Apply correcting kicks onto children particles
        
        Args:
            particles (Particle set):  Parent particle set
            parent (Particle):  Parent particle
            subsystem (Particle set):  Children particle set
            dt (Float):  Time interval for applying kicks
        """
        corr_par = CorrectionForCompoundParticle(particles, parent, subsystem, self.lib)
        self._kick_particles(subsystem, corr_par, dt)
       
    def _correction_kicks(self, particles, subsystems, dt) -> None:
        """
        Apply correcting kicks onto children and parent particles

        Args:
            particles (Particle set):  Parent particle set
            subsystems (Dictionary):  Dictionary of children system
            dt (Float):  Time interval for applying kicks
        """
        # Kick parent particles
        if self.model_time != 0. | units.yr:
            self.channels["from_gravity_to_parents"].copy()
            for parent, subsyst in subsystems.items():
                pid = self.__pid_workers[parent]
                self.resume_workers(pid)
                self._grav_channel_copier(
                    self.subcodes[parent].particles,
                    subsyst
                )
                self.hibernate_workers(pid)

        if subsystems and len(particles) > 1:
            corr_chd = CorrectionFromCompoundParticle(particles,
                                                      subsystems,
                                                      self.lib,
                                                      self.avail_cpus)
            self._kick_particles(particles, corr_chd, dt)
            
            with ThreadPoolExecutor(max_workers=self.avail_cpus) as executor:
                futures = [executor.submit(self._correct_children, particles.copy(), parent.copy(), subsystem, dt)
                           for parent, subsystem in subsystems.items()]
                for future in as_completed(futures):
                    future.result()
            
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