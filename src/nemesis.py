import ctypes
import gc
import numpy as np
from numpy.ctypeslib import ndpointer
import os
import queue
import threading

from amuse.community.huayno.interface import Huayno
from amuse.community.ph4.interface import ph4
from amuse.community.rebound.interface import Rebound
from amuse.community.seba.interface import SeBa

from amuse.couple import bridge
from amuse.datamodel import Particles, Particle
from amuse.ext.basicgraph import UnionFind
from amuse.ext.composition_methods import SPLIT_4TH_S_M6, SPLIT_4TH_S_M4
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
    def __init__(self, min_stellar_mass, par_conv, 
                 dt, coll_path, ejected_dir, eps=1.e-8, 
                 code_dt=0.03, par_nworker=1, dE_track=False, 
                 star_evol=False, gal_field=False, verbose=True):
        """
        Class setting up the simulation.
        
        Args:
            min_stellar_mass (Float):  Minimum stellar mass for stellar evolution
            par_conv (Converter):  Parent N-body converter
            dt (Float):  Diagnostic time step
            coll_path (String):  Path to store collision data
            ejected_dir (String):  Path to store ejected particles
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
        self.__min_mass_evol_evol = min_stellar_mass
        self.__dt = dt
        self.__coll_dir = coll_path
        self.__ejected_dir = ejected_dir
        self.__code_dt = code_dt
        self.__par_nworker = par_nworker
        self.__star_evol = star_evol
        self.__gal_field = gal_field
        self.__verbose = verbose
        self.__eps = eps
        
        # Protected attributes
        self._max_radius = 120. | units.au
        self._min_radius = 10. | units.au
        self._kick_ast_iter = 1  # Kick isolated asteroids every step
        self._nejec = 0
        self._dE_track = dE_track
        self._time_offsets = dict()
        
        self._MWG = MWpotentialBovy2015()
        self._parent_code = self._parent_worker(par_conv)
        self._asteroid_offset = 0. | units.yr
        self._asteroid_code = self._test_worker()
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
        if not isinstance(self.__min_mass_evol_evol.value_in(units.MSun), float) or self.__min_mass_evol_evol <= 0 | units.kg:
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
        for parent, sys in self.subsystems.items():
            sys.move_to_center()
            if parent not in self.subcodes:
                code = self._sub_worker(sys)
                code.particles.move_to_center()
                
                self._time_offsets[code] = self.model_time
                self.subsystems[parent] = sys
                self.subcodes[parent] = code
            parent.radius = set_parent_radius(np.sum(sys.mass))
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
        self.__tidal_radius = tidal_radius(self.particles)
        print(f"Tidal radius: {self.__tidal_radius.in_(units.pc)}")
        
    def _setup_bridge(self) -> None:
        """Embed system into galactic potential"""
        gravity = bridge.Bridge(use_threading=True,
                                method=SPLIT_4TH_S_M6,)
        gravity.add_system(self._parent_code, (self._MWG, ))
        gravity.timestep = self.__dt
        self.grav_bridge = gravity
        self._evolve_code = self.grav_bridge
    
    def _stellar_worker(self) -> SeBa:
        """Define stellar evolution integrator"""
        return SeBa()

    def _parent_worker(self, par_conv) -> ph4:
        """
        Define global integrator
        
        Args:
            par_conv (converter):  Converter for global integrator
        Returns:
            Code:  Gravitational integrator with particle set
        """
        code = ph4(par_conv, number_of_workers=self.__par_nworker)
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
            code = Huayno(self.__child_conv)
            code.particles.add_particles(children)
            code.parameters.timestep_parameter = self.__code_dt
            code.set_integrator("OK")
        
        else:
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
     
    def _test_worker(self) -> bridge.Bridge:
        """
        Kick integrator for isolated test particles
        
        Returns:
            Code:  Test particle integrator
        """
        code = Huayno(self.__child_conv)
        code.set_integrator("OK")
        
        gravity = bridge.Bridge(use_threading=False, 
                                method=SPLIT_4TH_S_M4)
        gravity.timestep = self.__dt
        if (self.__gal_field):
            gravity.add_system(code, (self._parent_code, self._MWG))
            
        else:
            gravity.add_system(code, (self._parent_code, ))
            
        return gravity

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
        # STILL TO DO: ADD CORRECTION FOR MW ENERGY WHEN BRIDGED
        all_parts = self.particles.all()
        Ek = all_parts.kinetic_energy()
        Ep = all_parts.potential_energy()
        Etot = Ek + Ep
        return Etot
      
    def _star_channel_copier(self):
        """Copy attributes from stellar code to grav. integrator particle set"""
        stars = self._stellar_code.particles
        self.channels["from_stellar_to_gravity"].copy()
        for children in self.subcodes.values():
            channel = stars.new_channel_to(children.particles)
            channel.copy_attributes(["mass", "radius"])
            
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

            if (self.__star_evol):
                self._stellar_evolution(self.model_time)
            self._split_subcodes()
            ejected_idx = ejection_checker(self.particles.copy(), 
                                           self.__tidal_radius)
            self._ejection_remover(ejected_idx)
            
        gc.collect()
        if (self.__verbose):
            print(f"Time: {self.model_time.in_(units.Myr)}")
            print(f"Parent code time: {self._parent_code.model_time.in_(units.Myr)}")
            for code in self.subcodes.values():
                print(f"Subcode time: {code.model_time.in_(units.Myr)}")
            print(f"Stellar code time: {self._stellar_code.model_time.in_(units.Myr)}")
            
    def _split_subcodes(self) -> None:
        """Remove child from system if object is isolated by rij > 1.25*radius"""
        #TO DO: ASTEROID SPLITS STILL REQUIRES TESTING
        if (self.__verbose):
            print("...Checking Splits...")
            
        asteroid_batch = [ ]
        for parent, subsys in list(self.subsystems.items()):
            radius = parent.radius
            furthest = (subsys.position - subsys.center_of_mass()).lengths().max()
            
            if furthest > 0.6*radius:
                components = subsys.connected_components(threshold=1.25*radius)
                
                if len(components) > 1:  # Checking for dissolution of system
                    if (self.__verbose):
                        print("...Split Detected...")
                    parent_pos = parent.position
                    parent_vel = parent.velocity
                    
                    self.particles.remove_particle(parent)
                    code = self.subcodes.pop(parent)
                    self._time_offsets.pop(code)
                    
                    for c in components:
                        sys = c.copy_to_memory()
                        if sys.mass.max() == (0. | units.kg):
                            if (self.__verbose):
                                print(f"Removing {len(sys)} asteroids")
                            sys.position += parent_pos
                            sys.velocity += parent_vel
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
        
        if asteroid_batch:
            for isolated_asteroids in asteroid_batch:
                self.asteroids.add_particles(isolated_asteroids)
                self._asteroid_code.particles.add_particles(isolated_asteroids)
               
    def _ejection_remover(self, ejected_idx) -> None:
        """
        Remove ejected particles from system. Save their properties to a file.
        
        Args:
            ejected_idx (array):  Array containing booleans flagging for particle ejections
        """
        #TO DO: FUNCTION STILL REQUIRES TESTING
        if (self._dE_track):
            E0 = self._calculate_total_energy()
        
        ejected_particles = self.particles[ejected_idx]
        for ejected_particle in ejected_particles:
            self._nejec += 1
            
            print(f"...Ejection #{self._nejec} Detected...")
            if ejected_particle in self.subcodes:
                code = self.subcodes.pop(ejected_particle)
                
                sys = self.subsystems[ejected_particle]
                filename = os.path.join(self.__ejected_dir, f"cluster_escapers_{self._nejec}")
                if (self.__verbose):
                    print(f"System pop: {len(sys)}")
                                    
                write_set_to_file(
                    sys.savepoint(0. | units.Myr), 
                    filename, 'amuse', close_file=True, 
                    overwrite_file=True
                )
                
                code.cleanup_code()
                code.stop()
                del code
        
        if len(ejected_particles) > 0:
            ejected_stars = ejected_particles[ejected_particles.mass > self.__min_mass_evol_evol]
            
            self._stellar_code.particles.remove_particles(ejected_stars)
            self.particles.remove_particle(ejected_particles)
            self.particles.synchronize_to(self._parent_code.particles)
            
            if (self._dE_track):
                E1 = self._calculate_total_energy()
                self.corr_energy += E1 - E0
    
    def _create_new_children_from_mergers(self, job_queue, lock) -> None:
        """
        Create new children systems based on parent mergers.
        
        Args:
            job_queue (Queue):  Queue of jobs, each hosting new parent systems
            lock (Lock):  Lock to prevent interfering communication
        """
        new_children, time_offset = job_queue.get()
        
        no_radius = new_children[new_children.radius == 0. | units.au]
        planets = no_radius[no_radius.mass <= self.__min_mass_evol_evol]
        stars = no_radius[no_radius.mass > self.__min_mass_evol_evol]
        planets.radius = planet_radius(planets.mass)
        stars.radius = ZAMS_radius(stars.mass)
        
        with lock:
            newparent = self.particles.add_subsystem(new_children)
            newparent.radius = set_parent_radius(np.sum(new_children.mass))
            
        new_children.move_to_center()    
        newcode = self._sub_worker(new_children)
    
        # TO DO: Check if this is a problem (multi mergers in same time-loop)
        self._time_offsets[newcode] = time_offset
        self.subcodes[newparent] = newcode
        self.subsystems[newparent] = new_children
            
    def _process_parent_mergers(self) -> None:
        """Process merging of parents from previous timestep in parallel"""
        self.channels["from_gravity_to_parents"].copy()
        
        lock = threading.Lock()
        job_queue = queue.Queue()
        
        no_new_parents = 0
        for new_parent, new_children in self.__new_systems.items():
            n = next((p for p in self._parent_code.particles if p.key == new_parent.key), None)
            time_offset = self.__new_offsets[new_parent]
                    
            # Synchronise children with updated parent phase-space coordinates
            pos_shift = n.position - new_parent.position
            vel_shift = n.velocity - new_parent.velocity
            new_children.position += pos_shift
            new_children.velocity += vel_shift 
            
            job_queue.put((new_children, time_offset))
            self.particles.remove_particle(new_parent)
            no_new_parents += 1
        
        threads = [ ]
        for worker in range(no_new_parents):
            th = threading.Thread(target=self._create_new_children_from_mergers, 
                                  args=(job_queue, lock))
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
        self.particles.recenter_subsystems()
    
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

                # TO CHECK IF THIS IS THE SAME
                evolved_parent = next((p for p in self._parent_code.particles if p.key == parti_.key), None)
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

                self.__new_systems.pop(parti_)
                self.particles.remove_particle(parti_)
                collsubset.add_particle(newparent)
                parti_ = newparent
            
            if parti_ in self.subcodes:
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
                                if coll_a.key in resolved_keys.keys():
                                    coll_a = next((p for p in code.particles if p.key == resolved_keys[coll_a.key]), None)
                                if coll_b.key in resolved_keys.keys():
                                    coll_b = next((p for p in code.particles if p.key == resolved_keys[coll_b.key]), None)

                                if coll_b.key == coll_a.key:
                                    print("Curious?")
                                    continue
                                
                                end_time = (coll_time - offset)
                                colliding_particles = Particles(particles=[coll_a, coll_b])
                                newparent, resolved_keys = self._handle_collision(self.subsystems[newparent], newparent, 
                                                                                  colliding_particles, coll_time, 
                                                                                  code, resolved_keys, end_time)
                                Nresolved += 1
                        
                        collsubset.add_particle(newparent)
                self._grav_channel_copier(
                    code.particles, 
                    self.subsystems[parti_]
                )
                
        for parti_ in collsubset:              
            if parti_ in self.subsystems:
                collsyst[parti_] = self.subsystems[parti_]
                
        return [collsubset, collsyst]
    
    def _handle_collision(self, children, parent, enc_parti, tcoll, code, resolved_keys, end_time):
        """
        Merge two particles if the collision stopping condition is met
        
        Args:
            children (Particle set):  The children particle set
            parent (Particle):  The parent particle
            enc_parti (Particle set): The particles in the collision
            tcoll (Float):  The time-stamp for which the particles collide at
            code (Code):  The integrator used
            resolved_keys (Dict):  Dictionary holding {Collider i Key: Remnant Key}
            end_time (Float):  The time the old code will evolve to. Used for new codes offset
        Returns:
            Particles:  New parent particle alongside dictionary of merging particles keys
        """
        # TO DO: Test function {(Pos, Vels) of parent, (Pos, Vels) of children, New offset, New code}
        # Save properties
        allparts = self.particles.all()
        nmerge = np.sum(allparts.coll_events) + 1
        print(f"...Collision #{nmerge} Detected...")
        write_set_to_file(allparts.savepoint(0. | units.Myr),
            os.path.join(self.__coll_dir, f"merger{nmerge}"),
            'amuse', close_file=True, overwrite_file=True
        )
        
        coll_a = next((p for p in children if p.key == enc_parti[0].key), None)
        coll_b = next((p for p in children if p.key == enc_parti[1].key), None)
        collider = coll_a + coll_b
        
        kepler_elements = orbital_elements(collider, G=constants.G)
        sem = kepler_elements[2]
        ecc = kepler_elements[3]
        inc = kepler_elements[4]
        with open(os.path.join(self.__coll_dir, f"merger{nmerge}.txt"), 'w') as f:
            f.write(f"Tcoll: {tcoll.in_(units.yr)}")
            f.write(f"\nKey1: {enc_parti[0].key}")
            f.write(f"\nKey2: {enc_parti[1].key}")
            f.write(f"\nM1: {enc_parti[0].mass.in_(units.MSun)}")
            f.write(f"\nM2: {enc_parti[1].mass.in_(units.MSun)}")
            f.write(f"\nSemi-major axis: {abs(sem).in_(units.au)}")
            f.write(f"\nEccentricity: {ecc}")
            f.write(f"\nInclination: {inc} deg")
        
        # Create merger remnant
        if max(collider.mass) > 0 | units.kg:
            remnant  = Particles(1)
            remnant.mass = collider.total_mass()
            remnant.position = collider.center_of_mass()
            remnant.velocity = collider.center_of_mass_velocity()
        else:
            raise ValueError("Error: Asteroid - Asteroid collision")
            
        if "HOST" in coll_a.type or "HOST" in coll_b.type:
            remnant.type = "HOST"
            remnant.radius = ZAMS_radius(remnant.mass)
            self._stellar_code.particles.add_particle(remnant)
        elif remnant.mass > self.__min_mass_evol_evol:
            remnant.type = "STAR"
            remnant.radius = ZAMS_radius(remnant.mass)
            self._stellar_code.particles.add_particle(remnant)
        else:
            most_massive = collider.mass.argmax()
            remnant.type = collider[most_massive].type
            remnant.radius = planet_radius(remnant.mass)
            
        remnant.coll_events = nmerge
        remnant.sub_worker_radius = remnant.radius
        
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
        
        children.add_particles(remnant)
        children.remove_particles(coll_a)
        children.remove_particles(coll_b)
        nearest_mass = abs(children.mass - parent.mass).argmin()
        
        if remnant.key == children[nearest_mass].key:
            children.position += parent.position
            children.velocity += parent.velocity
            
            # Create new parent particle
            newparent = self.particles.add_subsystem(children)
            newparent.radius = parent.radius

            # Re-mapping dictionary to new parent
            old_code = self.subcodes.pop(parent)
            old_offset = self._time_offsets.pop(old_code)
            self.subcodes[newparent] = old_code
            new_code = self.subcodes[newparent]
            self._time_offsets[new_code] = end_time

            self.particles.remove_particle(parent)
            children.synchronize_to(self.subcodes[newparent].particles)

        else:
            newparent = parent
            children.synchronize_to(code.particles)
        
        if coll_a.mass > self.__min_mass_evol_evol:
          self._stellar_code.particles.remove_particle(coll_a)
        if coll_b.mass > self.__min_mass_evol_evol:
          self._stellar_code.particles.remove_particle(coll_b)
          
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
        
        dt = dt - self._asteroid_offset
        self._asteroid_code.evolve_model(dt)
        self._asteroid_code.particles.new_channel_to(self.asteroids).copy()
        
        for particle in self.particles:  # Check if any asteroids lies within a parent's radius
            distances = (self.asteroids.position - particle.position).lengths()
            newsystem = self.asteroids[distances < 1.25*particle.radius]
            
            if newsystem:
                newparts = HierarchicalParticles(Particles())
                if particle in self.subcodes:
                    if (self.__verbose):
                        print("...Merging asteroid with parent...")
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
                    newparts.add_particle(newsystem)
                    
                self.asteroids.remove_particle(newsystem)
                self._asteroid_code.particles.remove_particle(newsystem)
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
        
        if len(self._asteroid_code.particles) == 0:
            self._asteroid_code.stop()
                    
    def _drift_global(self, dt) -> None:
        """
        Evolve parent system until dt
        
        Args:
            dt (Float):  Time to evolve till
            corr_time (Float): Time to correct for drift
        """
        if (self.__verbose):
            print("...Drifting Global...")
            dist = (self._parent_code.particles[0].position - self._parent_code.particles[1].position).lengths()
            print(f"# Parents {len(self._parent_code.particles)}, Distance {dist.in_(units.au)}")
            for p, c in self.subsystems.items():
                print(f"# Kids {len(c)}")
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
            dt (Float):  Time to evolve till
        """     
        def resolve_collisions(code, parent, evol_time, stopping_condition):
            """Function to resolve collisions"""
            
            self._grav_channel_copier(
                code.particles, 
                self.subsystems[parent]
            )
            coll_time = code.model_time
            coll_a_particles = stopping_condition.particles(0)
            coll_b_particles = stopping_condition.particles(1)
                        
            resolved_keys = dict()
            Nmergers = max(len(np.unique(coll_a_particles.key)),  
                           len(np.unique(coll_b_particles.key)))
            Nresolved = 0
            for coll_a, coll_b in zip(coll_a_particles, coll_b_particles):
                if Nresolved < Nmergers:  # Stop recursive loop
                    if coll_a.key in resolved_keys.keys():
                        coll_a = next((p for p in code.particles if p.key == resolved_keys[coll_a.key]), None)
                    if coll_b.key in resolved_keys.keys():
                        coll_b = next((p for p in code.particles if p.key == resolved_keys[coll_b.key]), None)
                        
                    if coll_b.key == coll_a.key:
                        print("Curious?")
                        continue
                    
                    colliding_particles = Particles(particles=[coll_a, coll_b])
                    parent, resolved_keys = self._handle_collision(self.subsystems[parent], parent, 
                                                                   colliding_particles, coll_time, 
                                                                   code, resolved_keys, evol_time)
                    Nresolved += 1
            
            return parent
        
        def evolve_code():
            """Algorithm to evolve individual children codes"""
            try:
                parent = job_queue.get(timeout=1)  # Timeout to prevent blocking indefinitely
            except queue.Empty:
                raise ValueError("Error: No children in system")
                
            code = self.subcodes[parent]
            evol_time = dt - self._time_offsets[code]
            stopping_condition = code.stopping_conditions.collision_detection
            stopping_condition.enable()
            
            while code.model_time < evol_time*(1. - self.__eps):
                code.evolve_model(evol_time)
                
                if stopping_condition.is_set():
                    with lock:  # All threads stop until resolve collision
                        if (self._dE_track):
                            E0 = self._calculate_total_energy()
                        
                        parent = resolve_collisions(code, 
                                                    parent, 
                                                    evol_time, 
                                                    stopping_condition)
                        if (self._dE_track):
                           E1 = self._calculate_total_energy()
                           self.corr_energy += E1 - E0
                                
            job_queue.task_done()
        
        if (self.__verbose):
            print("...Drifting Children...")
        job_queue = queue.Queue()
        for parent in self.subcodes.keys():
            job_queue.put(parent)
            self.key = parent.key
            
        lock = threading.Lock()
        threads = []
        for worker in range(len(self.subcodes.values())):
            th = threading.Thread(target=evolve_code)
            th.start()
            threads.append(th)
        
        for th in threads:
            th.join()  # Wait for all threads to finish
        
        changes = [ ]
        for parent in self.subcodes.keys(): # Remove single children systems:
            if len(self.subcodes[parent].particles) == 1:
                changes.append(parent)
        
        for parent in changes:
            old_subcode = self.subcodes.pop(parent)
            old_offset = self._time_offsets.pop(old_subcode)
            
            old_subcode.cleanup_code()
            old_subcode.stop()
            del old_subcode
            del old_offset
                
    def _kick_particles(self, particles, corr_code, dt) -> None:
        """
        Apply correction kicks onto target particles
        
        Args:
            particles (Particle set):  Particles to correct accelerations of
            corr_code (Code):  Code containing information on difference in gravitational field
            dt (Float):  Time-step of correction kick
        """
        parts = particles.copy_to_memory()
        ax, ay, az = corr_code.get_gravity_at_point(parts.radius,
                                                    parts.x, 
                                                    parts.y, 
                                                    parts.z)
        parts.vx += dt*ax
        parts.vy += dt*ay
        parts.vz += dt*az
        
        channel = parts.new_channel_to(particles)
        channel.copy_attributes(["vx","vy","vz"])
        
    def _correct_children(self, job_queue, dt, lock) -> None:
        """
        Apply correcting kicks onto children particles
        
        Args:
            job_queue (Queue):  Queue of jobs, each hosting unique parent system
            dt (Float):  Time interval for applying kicks
            lock (Lock):  Lock to prevent interfering communication
        """
        particles, parent, subsyst = job_queue.get()
        
        corr_par = CorrectionForCompoundParticle(particles, parent, subsyst, self.lib)
        with lock:
            self._kick_particles(subsyst, corr_par, dt)
        self._grav_channel_copier(
            subsyst, self.subcodes[parent].particles, 
            ["vx","vy","vz"]
        )

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
            self._grav_channel_copier(
                self.subcodes[parent].particles, 
                subsyst
            )
            
        if subsystems and len(particles) > 1:                    
            corr_chd = CorrectionFromCompoundParticle(particles, 
                                                      subsystems,
                                                      self.lib)
            self._kick_particles(particles, corr_chd, dt)
            self.channels["from_parents_to_gravity"].copy()
            
            # Kick children
            job_queue = queue.Queue()
            lock = threading.Lock()
            for parent, subsyst in subsystems.items():
                p = particles.copy()
                job_queue.put((p, parent, subsyst))
                
            threads = [ ]
            for worker in range(len(subsystems.keys())):
                thread = threading.Thread(target=self._correct_children,
                                          args=(job_queue, dt, lock))
                thread.start()
                threads.append(thread)
                
            for thread in threads:
                thread.join()
                
        self.particles.recenter_subsystems()
        for parent, subsyst in subsystems.items():
            self._grav_channel_copier(
                subsyst,
                self.subcodes[parent].particles
            )
        
    @property
    def model_time(self) -> float:  
        """Extract the global integrator model time"""
        return self._parent_code.model_time
    
    @property
    def subsystems(self) -> dict:
        """Extract the children system"""
        return self.particles.collection_attributes.subsystems