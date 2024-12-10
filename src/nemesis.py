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

from src.environment_functions import ejection_checker, set_parent_radius, tidal_radius
from src.environment_functions import planet_radius, ZAMS_radius
from src.grav_correctors import CorrectionFromCompoundParticle
from src.grav_correctors import CorrectionForCompoundParticle
from src.hierarchical_particles import HierarchicalParticles


class Nemesis(object):
    def __init__(self, min_stellar_mass, par_conv, dtbridge, 
                 coll_dir, ejected_dir, available_cpus=os.cpu_count(),
                 eps=1.e-8, code_dt=0.03, par_nworker=1, dE_track=False, 
                 star_evol=False, gal_field=True, verbose=True):
        """
        Class setting up the simulation.
        
        Args:
            min_stellar_mass (Float):  Minimum stellar mass for stellar evolution
            par_conv (Converter):  Parent N-body converter
            dtbridge (Float):  Diagnostic time step
            coll_dir (String):  Path to store collision data
            ejected_dir (String):  Path to store ejected particles
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
        self.__ejected_dir = ejected_dir
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
        self._min_radius = 10. | units.au
        self._nejec = 0
        self._dE_track = dE_track
        self._time_offsets = dict()
        
        self._MWG = MWpotentialBovy2015()
        self._parent_code = self._parent_worker()
        if (self.__star_evol):
            self._stellar_code = self._stellar_worker()
        
        self.asteroids = Particles()
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
        if not isinstance(self.__ejected_dir, str):
            raise ValueError("Error: ejected_dir must be a string")

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
            - initialising children codes when needed
            - defining particles to evolve with stellar codes
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
            parent.radius = set_parent_radius(np.sum(sys.mass))
            if parent not in self.subcodes:
                code = self._sub_worker(sys)
                code.particles.move_to_center()
                
                self._time_offsets[code] = self.model_time
                self.subsystems[parent] = sys
                self.subcodes[parent] = code
                
                # Store children PID to allow hibernation
                worker_pid = self.get_child_pid()
                self.__pid_workers[parent] = worker_pid
                self.hibernate_workers(worker_pid)
                
            if (self.__verbose):
                print(f"System {nsyst+1}/{ntotal}, radius: {parent.radius.in_(units.au)}")
                
        particles[particles.radius > self._max_radius].radius = self._max_radius
        particles[particles.radius < self._min_radius].radius = self._min_radius

        if (self.__star_evol):
            parti = particles.all()
            self.stars = parti[parti.mass > self.__min_mass_evol_evol]
            stellar_code = self._stellar_code
            stellar_code.particles.add_particle(self.stars)

        if (self.__gal_field):
            self._setup_bridge()
            self._extract_tidal_radius()
            
        else:
            self._evolve_code = self._parent_code
            self.__tidal_radius = None

    def _extract_tidal_radius(self) -> None:
        """Compute the tidal radius of the system"""
        particles = self.particles[self.particles.mass > (0. | units.kg)].copy()
        self.__tidal_radius = tidal_radius(particles)
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
        code.parameters.epsilon_squared = (0. | units.au)**2
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
            if (self.__verbose):
                print("...Asteroid System Detected...")
            code = Huayno(self.__child_conv)
            code.particles.add_particles(children)
            code.parameters.timestep_parameter = self.__code_dt
            code.set_integrator("OK")
        else:
            if (self.__verbose):
                print("...Only Massive Children...")
            code = Huayno(self.__child_conv)
            code.particles.add_particles(children)
            code.parameters.timestep_parameter = self.__code_dt
            code.set_integrator("SHARED8_COLLISIONS")
            
        # TO DO: Add REBOUND integrator
        """elif masses[-1] > 100.*masses[-2]:
            code = Rebound(self.__child_conv)
            code.particles.add_particles(children)
            code.set_integrator("WHFast")"""
        
        return code
    
    def get_child_pid(self) -> int:
        """Returns the PID of the most recently spawned children worker"""
        for child in self.__main_process.children(recursive=True):
            if 'huayno_worker' in child.name():
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
     
    def _major_channel_maker(self):
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
                    attributes=["vx","vy","vz"],
                    target_names=["vx","vy","vz"]
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
      
    def _star_channel_copier(self):
        """Copy attributes from stellar code to grav. integrator particle set"""
        stars = self._stellar_code.particles
        self.channels["from_stellar_to_gravity"].copy()
        for parent, code in self.subcodes.items():
            self.resume_workers(self.__pid_workers[parent])
            channel = stars.new_channel_to(code.particles)
            channel.copy_attributes(["mass", "radius"])
            self.hibernate_workers(self.__pid_workers[parent])
            
    def _grav_channel_copier(self, transfer_data, receive_data, 
                             attributes=["x","y","z","vx","vy","vz", "radius", "mass"]) -> None:
        """
        Communicate information between grav. integrator and local particle set
        
        Args:
            transfer_data (Particle set):  Particle set to transfer data from
            receive_data (Particle set):  Particle set to update data
            attributes (Array):  Attributes wanting to copy
        """
        channel = transfer_data.new_channel_to(receive_data)
        channel.copy_attributes(attributes)
       
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
                    timestep
                )
                
            if self.asteroids:
                if (self.__verbose):
                    print("...Drifting asteroids...")
                self._drift_test_particles(self.model_time - evolve_time)
                
            if (self.__star_evol):
                self._stellar_evolution(self.model_time)
                
            self._split_subcodes()
            
            if self.asteroids:  # Account for asteroids
                particles = self.particles.copy()
                particles.add_particles(self.asteroids)
            else:
                particles = self.particles.copy()
                
            ejected_idx = ejection_checker(particles, self.__tidal_radius)
            self._ejection_remover(particles, ejected_idx)
            
        gc.collect()
        if (self.__verbose):
            print(f"Time: {self.model_time.in_(units.Myr)}")
            print(f"Parent code time: {self._parent_code.model_time.in_(units.Myr)}")
            for parent, code in self.subcodes.items():
                self.resume_workers(self.__pid_workers[parent])
                total_time = code.model_time + self._time_offsets[code]
                if not abs(total_time - self.model_time) < 0.01 * self.__dt:
                    print(f"Parent: {parent.key}, Excess simulation: {total_time - self.model_time}")
                self.hibernate_workers(self.__pid_workers[parent])
                
            print(f"Stellar code time: {self._stellar_code.model_time.in_(units.Myr)}")
            print(f"==================================================")
            
    def _split_subcodes(self) -> None:
        """Remove child from system if object is isolated by rij > 1.5*radius"""
        def split_algorithm(job_queue):
            parent, subsys = job_queue.get()
            radius = parent.radius
            host = subsys[subsys.mass.argmax()]
            furthest = (subsys.position - host.position).lengths().max()
            
            if furthest > 0.75*radius:
                components = subsys.connected_components(threshold=1.5*radius)
                
                if len(components) > 1:  # Checking for dissolution of system
                    if (self.__verbose):
                        print("...Split Detected...")
                    parent_pos = parent.position
                    parent_vel = parent.velocity
                    
                    with self.__lock:
                        self.resume_workers(self.__pid_workers[parent])
                        self.particles.remove_particle(parent)
                        code = self.subcodes.pop(parent)
                        self._time_offsets.pop(code)
                    
                        for c in components:
                            sys = c.copy_to_memory()
                            if sys.mass.max() == (0. | units.kg):
                                sys.position += parent_pos
                                sys.velocity += parent_vel
                                sys.syst_id = -1
                                
                                asteroid_batch.append(sys)
                                
                            else:
                                if len(sys) > 1:
                                    sys.position += parent_pos
                                    sys.velocity += parent_vel
                                    
                                    newparent = self.particles.add_subsystem(sys)
                                    newparent.radius = set_parent_radius(np.sum(sys.mass))
                                    if newparent.radius > self._max_radius:
                                        newparent.radius = self._max_radius
                                    elif newparent.radius < self._min_radius:
                                        newparent.radius = self._min_radius
                                        
                                    sys.move_to_center()
                                    newcode = self._sub_worker(sys)
                                    self.subcodes[newparent] = newcode
                                    self._time_offsets[newcode] = self.model_time
                                    
                                    worker_pid = self.get_child_pid()
                                    self.__pid_workers[newparent] = worker_pid
                                    self.hibernate_workers(worker_pid)
                                    
                                else:
                                    sys.position += parent_pos
                                    sys.velocity += parent_vel
                                    
                                    newparent = self.particles.add_subsystem(sys)
                                    newparent.radius = set_parent_radius(np.sum(sys.mass))
                                    if newparent.radius > self._max_radius:
                                        newparent.radius = self._max_radius
                                    elif newparent.radius < self._min_radius:
                                        newparent.radius = self._min_radius
                            
                        code.cleanup_code()
                        code.stop()
                        del code, subsys, parent
                    
        if (self.__verbose):
            print("...Checking Splits...")
        
        asteroid_batch = [ ]
        job_queue = queue.Queue()
        for nsyst, (host, subsystem) in enumerate(self.subsystems.items()):
            job_queue.put((host.copy(), subsystem))
        
        threads = [ ]
        for _ in range(nsyst):
            th = threading.Thread(target=split_algorithm, 
                                  args=(job_queue, ))
            th.start()
            threads.append(th)
        
        for th in threads:
            th.join()
            
        job_queue.queue.clear()
        del job_queue
                    
        if asteroid_batch:
            for isolated_asteroids in asteroid_batch:
                self.asteroids.add_particles(isolated_asteroids)
            if (self.__verbose):
                print(f"# Isolated asteroids: {len(self.asteroids)} asteroids")
               
    def _ejection_remover(self, parent_systems, ejected_idx) -> None:
        """
        Remove ejected particles from system. Save their properties to a file.
        
        Args:
            ejected_idx (array):  Array containing booleans flagging for particle ejections
        """
        #TO DO: FUNCTION STILL REQUIRES TESTING
        if (self._dE_track):
            E0 = self._calculate_total_energy()
        
        ejected_particles = parent_systems[ejected_idx]
        n_massive = len(ejected_particles[ejected_particles.mass > (0. | units.kg)])
        n_massless = len(ejected_particles[ejected_particles.mass == 0. | units.kg])
        print(f"# Ejected Massive: {n_massive}, # Ejected Massless: {n_massless}")
        
        ejected_asteroids = Particles()
        ejected_massive = Particles()
        ejected_stars = Particles()
        for ejected_particle in ejected_particles:
            if ejected_particle in self.asteroids:
                ejected_asteroids.add_particle(ejected_particle)
            else:
                self._nejec += 1
                ejected_massive.add_particle(ejected_particle)
                if ejected_particle in self.subcodes:
                    self.resume_workers(self.__pid_workers[ejected_particle])
                    code = self.subcodes.pop(ejected_particle)
                    
                    sys = self.subsystems[ejected_particle]
                    filename = os.path.join(self.__ejected_dir, f"escapers_{self.model_time.in_(units.Myr)}_n{self._nejec}")
                    if (self.__verbose):
                        print(f"System pop: {len(sys)}")
                                        
                    write_set_to_file(
                        sys, filename, 'amuse', 
                        close_file=True, 
                        overwrite_file=True
                    )
                    
                    code.cleanup_code()
                    code.stop()
                    del code
                    
                    stars = sys[sys.mass > self.__min_mass_evol_evol]
                    ejected_stars.add_particles(stars)
                    
                elif ejected_particle in self._stellar_code.particles:
                    ejected_stars.add_particle(ejected_particle)
                    
        
        if len(ejected_particles) > 0:
            self._stellar_code.particles.remove_particles(ejected_stars)
            self.particles.remove_particle(ejected_massive)
            self.particles.synchronize_to(self._parent_code.particles)
            self.asteroids.remove_particles(ejected_asteroids)
            
            if (self._dE_track):
                E1 = self._calculate_total_energy()
                self.corr_energy += E1 - E0
    
    def _create_new_children(self, job_queue) -> None:
        """
        Create new children systems based on parent mergers.
        
        Args:
            job_queue (Queue):  Queue of jobs, each hosting new parent systems
        """
        new_children, time_offset = job_queue.get()
        
        no_radius = new_children[new_children.radius == (0. | units.au)]
        planets = no_radius[no_radius.mass <= self.__min_mass_evol_evol]
        stars = no_radius[no_radius.mass > self.__min_mass_evol_evol]
        planets.radius = planet_radius(planets.mass)
        stars.radius = ZAMS_radius(stars.mass)
        with self.__lock:
            newparent = self.particles.add_subsystem(new_children)
            
        newparent.radius = set_parent_radius(np.sum(new_children.mass))
        new_children.move_to_center()    
        newcode = self._sub_worker(new_children)
    
        with self.__lock:
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
        
        no_new_parents = 0
        parent_particles = self._parent_code.particles
        for new_parent, new_children in self.__new_systems.items():
            n = parent_particles[parent_particles.key == new_parent.key]
            time_offset = self.__new_offsets[new_parent]
                    
            pos_shift = n.position - new_parent.position
            vel_shift = n.velocity - new_parent.velocity
            new_children.position += pos_shift
            new_children.velocity += vel_shift 
            
            job_queue.put((new_children, time_offset))
            self.particles.remove_particle(new_parent)
            no_new_parents += 1
        
        threads = [ ]
        for worker in range(no_new_parents):
            th = threading.Thread(target=self._create_new_children, 
                                  args=(job_queue, ))
            th.start()
            threads.append(th)
            
        for th in threads:
            th.join()
            
        job_queue.queue.clear()
        del job_queue
        
        # Modify radius
        new_parents = self.particles[-no_new_parents:]
        new_parents[new_parents.radius > self._max_radius].radius = self._max_radius
        new_parents[new_parents.radius < self._min_radius].radius = self._min_radius
        self.particles.recenter_subsystems(max_workers=2.*self.avail_cpus)
    
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
        radius = 0. | units.au
        for parti_ in collsubset:
            parti_ = parti_.as_particle_in_set(self.particles)
            radius = max(radius, parti_.radius)
            
            if parti_ in self.subcodes:
                self.resume_workers(self.__pid_workers[parti_])
                code = self.subcodes.pop(parti_)
                self._time_offsets.pop(code)
                
                sys = self.subsystems[parti_]
                sys.position += parti_.position
                sys.velocity += parti_.velocity
                newparts.add_particles(sys)

                code.cleanup_code()
                code.stop()
                del code
              
            else:
                new_parti = newparts.add_particle(parti_)
                new_parti.radius = parti_.sub_worker_radius
                
            self.particles.remove_particle(parti_)
            
        # Temporary new parent particle
        newparent = Particle()
        newparent.mass = newparts.total_mass()
        newparent.position = newparts.center_of_mass()
        newparent.velocity = newparts.center_of_mass_velocity()
        newparent.radius = radius
        
        self.particles.add_particle(newparent)
        self.__new_systems[newparent] = newparts
        self.__new_offsets[newparent] = self.model_time
        
        worker_pid = self.get_child_pid()
        self.__pid_workers[newparent] = worker_pid
        self.hibernate_workers(worker_pid)
        
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
        collsubset = Particles()
        collsyst = dict()
        for parti_ in coll_set:
            collsubset.add_particle(parti_)
            
            # If a recently merged parent merges in the same time-loop, you need to give it children
            if parti_ in self.__new_systems:  
                collsubset.remove_particle(parti_)

                evolved_parent = parti_.as_particle_in_set(self._parent_code.particles)
                children = self.__new_systems[parti_]
                offset = self.__new_offsets[parti_]
                
                pos_shift = evolved_parent.position - parti_.position
                vel_shift = evolved_parent.velocity - parti_.velocity
                children.position += pos_shift
                children.velocity += vel_shift

                newcode = self._sub_worker(children)
                newcode.particles.move_to_center()

                newparent = self.particles.add_subsystem(children)
                newparent.radius = set_parent_radius(np.sum(children.mass))

                self.subcodes[newparent] = newcode
                self.subsystems[newparent] = children
                self._time_offsets[newcode] = offset

                worker_pid = self.get_child_pid()
                self.__pid_workers[newparent] = worker_pid
                self.hibernate_workers(worker_pid)
                
                self.__new_systems.pop(parti_)
                self.particles.remove_particle(parti_)
                collsubset.add_particle(newparent)
                parti_ = newparent
            
            if parti_ in self.subcodes:
                worker_pid = self.__pid_workers[parti_]
                self.resume_workers(worker_pid)
                
                code = self.subcodes[parti_]
                offset = self._time_offsets[code]
                stopping_condition = code.stopping_conditions.collision_detection
                stopping_condition.enable()
                newparent = parti_.copy()
                
                if (self.__verbose):
                    print("Evolving for: ", (coll_time - offset).in_(units.Myr))
                     
                while code.model_time < (coll_time - offset)*(1. - self.__eps):
                    code.evolve_model(coll_time-offset)
                    
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
                                
                                end_time = (coll_time - offset)
                                colliding_particles = Particles(particles=[coll_a, coll_b])
                                newparent, resolved_keys = self._handle_collision(self.subsystems[newparent], newparent, 
                                                                                  colliding_particles, code, resolved_keys, 
                                                                                  end_time)
                                Nresolved += 1
                        
                        collsubset.add_particle(newparent)
                self._grav_channel_copier(
                    code.particles, 
                    self.subsystems[parti_]
                )
                self.hibernate_workers(worker_pid)
                
        for parti_ in collsubset:              
            if parti_ in self.subsystems:
                collsyst[parti_] = self.subsystems[parti_]
                
        return [collsubset, collsyst]
    
    def _handle_collision(self, children, parent, enc_parti, code, resolved_keys, end_time):
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
        
        output_dir = os.path.join(self.__coll_dir, f"merger_Step{self.dt_step}_SystemID{parent.key}.hdf5")
        write_set_to_file(
            code.particles,
            output_dir, 'hdf5', 
            close_file=True, 
            overwrite_file=True
        )
        
        coll_a = children[children.key == enc_parti[0].key]
        coll_b = children[children.key == enc_parti[1].key]
        
        collider = Particles(particles=[coll_a, coll_b])
        kepler_elements = orbital_elements(collider, G=constants.G)
        sem = kepler_elements[2]
        ecc = kepler_elements[3]
        inc = kepler_elements[4]
        
        tcoll = code.model_time + self._time_offsets[code]
        with open(os.path.join(self.__coll_dir, "output", f"merger{self.__nmerge}.txt"), 'w') as f:
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
                self._time_offsets[new_code] = end_time
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
    
    def _drift_test_particles(self, dt) -> None:
        """
        Kick and evolve isolated test particles
        
        Args:
            dt (float):  Time to drift test particles
        """
        if (self.__verbose):
            print(f"...Drifting {len(self.asteroids)} Asteroids...")
            print(f"TIME = {dt.in_(units.kyr)}")
        
        ax, ay, az = self._evolve_code.get_gravity_at_point(
                            0.*self.asteroids.radius,
                            self.asteroids.x,
                            self.asteroids.y,
                            self.asteroids.z
                            )
        half_step = dt/2.
        self.asteroids.vx += ax * half_step
        self.asteroids.vy += ay * half_step
        self.asteroids.vz += az * half_step
        self.asteroids.position += self.asteroids.velocity*dt
        self.asteroids.vx += ax * half_step
        self.asteroids.vy += ay * half_step
        self.asteroids.vz += az * half_step
        
        for particle in self.particles:  # Check if any asteroids lies within a parent's radius
            distances = (self.asteroids.position - particle.position).lengths()
            within_distance = distances < 1.25*particle.radius
            newsystem = self.asteroids[within_distance]
            
            if newsystem:
                newparts = HierarchicalParticles(Particles())
                if particle in self.subcodes:
                    if (self.__verbose):
                        print("...Merging asteroid with parent...")
                        
                    self.resume_workers(self.__pid_workers[particle])
                    code = self.subcodes.pop(particle)
                    offset = self._time_offsets.pop(code)
                    subsys = self.subsystems[particle]

                    subsys.position += particle.position
                    subsys.velocity += particle.velocity 
                    newparts.add_particles(subsys)
                    newparts.add_particle(newsystem)
                    
                    code.cleanup_code()
                    code.stop()
                    del code

                else:
                    if (self.__verbose):
                        print("...Merging asteroid with isolated body...")
                    newparts.add_particle(particle)
                    newparts.add_particles(newsystem)
                    
                self.asteroids.remove_particle(newsystem)
                self.particles.remove_particle(particle)
                
                newcode = self._sub_worker(newparts)
                newcode.particles.move_to_center()
                
                newparent = self.particles.add_subsystem(newparts)
                newparent.radius = set_parent_radius(np.sum(newparts.mass))
                if newparent.radius > self._max_radius:
                    newparent.radius = self._max_radius
                elif newparent.radius < self._min_radius:
                    newparent.radius = self._min_radius
                    
                self._time_offsets[newcode] = self.model_time - newcode.model_time
                self.subcodes[newparent] = newcode
                    
                worker_pid = self.get_child_pid()
                self.__pid_workers[newparent] = worker_pid
                self.hibernate_workers(worker_pid)
                    
    def _drift_global(self, dt) -> None:
        """
        Evolve parent system until dt
        
        Args:
            dt (Float):  Time to evolve till
            corr_time (Float): Time to correct for drift
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
            
        if (self.__verbose):
            print(f"Parent code time: {self._parent_code.model_time.in_(units.Myr)}")
            print(f"Bridge code time: {self._evolve_code.model_time.in_(units.Myr)}")
            print(f"Stellar code time: {self._stellar_code.model_time.in_(units.Myr)}")
            
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
            self._grav_channel_copier(code.particles, children)
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
                                                    colliding_particles, code, 
                                                    resolved_keys, evol_time
                                                    )
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
                evol_time = dt - self._time_offsets[code]
                if evol_time <= 0 | units.s:
                    raise ValueError("Error: subcode dt must be positive")
                
                stopping_condition = code.stopping_conditions.collision_detection
                stopping_condition.enable()
                
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

        for parent in self.subcodes.keys(): # Remove single children systems:
            self.resume_workers(self.__pid_workers[parent])
            if len(self.subcodes[parent].particles) == 1:
                old_subcode = self.subcodes.pop(parent)
                old_offset = self._time_offsets.pop(old_subcode)
                
                old_subcode.cleanup_code()
                old_subcode.stop()
                del old_subcode
                del old_offset
                
            else:
                self.hibernate_workers(self.__pid_workers[parent])
        
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
        self.resume_workers(self.__pid_workers[parent])
        self._grav_channel_copier(
                subsystem, self.subcodes[parent].particles,
                ["vx","vy","vz"]
        )
        self.hibernate_workers(self.__pid_workers[parent])

    def _correction_kicks(self, particles, subsystems, dt) -> None:
        """
        Apply correcting kicks onto children and parent particles
        
        Args:
            particles (Particle set):  Parent particle set
            subsystems (Dictionary):  Dictionary of children system
            dt (Float):  Time interval for applying kicks
        """
        # Kick parent particles
        self.channels["from_gravity_to_parents"].copy()
        for parent, subsyst in subsystems.items():
            self.resume_workers(self.__pid_workers[parent])    
            self._grav_channel_copier(
                self.subcodes[parent].particles, 
                subsyst
            )
            self.hibernate_workers(self.__pid_workers[parent])  
            
        if subsystems and len(particles) > 1:      
            corr_chd = CorrectionFromCompoundParticle(particles, 
                                                      subsystems,
                                                      self.lib,
                                                      self.avail_cpus)
            self._kick_particles(particles, corr_chd, dt)
            self.channels["from_parents_to_gravity"].copy()

            with ThreadPoolExecutor(max_workers=self.avail_cpus) as executor:
                futures = [executor.submit(self._correct_children, particles.copy(), parent.copy(), subsystem, dt)
                           for parent, subsystem in subsystems.items()]
                for future in as_completed(futures):
                    future.result()
                    
        self.particles.recenter_subsystems(max_workers=2.*self.avail_cpus)
        for parent, subsyst in subsystems.items():
            self.resume_workers(self.__pid_workers[parent])
            self._grav_channel_copier(
                subsyst,
                self.subcodes[parent].particles
            )
            self.hibernate_workers(self.__pid_workers[parent])  
        
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
        return self.__total_free_cpus - self.__par_nworker - 3