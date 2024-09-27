import gc
import numpy as np
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
from amuse.units import units, constants

from src.environment_functions import ejection_checker, set_parent_radius
from src.environment_functions import planet_radius, ZAMS_radius
from src.grav_correctors import CorrectionFromCompoundParticle
from src.grav_correctors import CorrectionForCompoundParticle
from src.hierarchical_particles import HierarchicalParticles


class Nemesis(object):
    def __init__(self, min_stellar_mass, par_conv, child_conv, 
                 dt, coll_path, ejected_dir, code_dt=0.03, 
                 par_nworker=1, dE_track=False, 
                 star_evol=False, gal_field=False):
        """
        Class setting up the simulation.
        
        Args:
            min_stellar_mass (Float):  Minimum stellar mass for stellar evolution
            par_conv (Converter):  Parent N-body converter
            child_conv (Converter):  Children N-body converter
            dt (Float):  Diagnostic time step
            coll_path (String):  Path to store collision data
            __ejected_dir (String):  Path to store ejected particles
            snapdir_path (String):  Path to store snapshots
            code_dt (Float):  Internal time step
            par_nworker (Int):  Number of workers for global integrator
            dE_track (Boolean):  Flag turning on/off energy error tracker
            star_evol (Boolean):  Flag turning on/off stellar evolution
            gal_field (Boolean):  Flag turning on/off galactic field
        """
        
        # Private attributes
        self.__min_mass_evol_evol = min_stellar_mass
        self.__child_conv = child_conv
        self.__dt = dt
        self.__coll_dir = coll_path
        self.__ejected_dir = ejected_dir
        self.__code_dt = code_dt
        self.__par_nworker = par_nworker
        self.__star_evol = star_evol
        self.__gal_field = gal_field
        
        # Protected attributes
        self._max_radius = 1000. | units.au
        self._min_radius = 50. | units.au
        self._kick_ast_iter = 0
        self._nejec = 0
        self._dE_track = dE_track
        self._MWG = MWpotentialBovy2015()
      
        self._parent_code = self._parent_worker(par_conv)
        self._asteroid_offset = 0. | units.yr
        self._asteroid_code = self._test_worker()
        if (self.__star_evol):
            self._stellar_code = self._stellar_worker()
        
        self.asteroids = Particles()
        self.particles = HierarchicalParticles(self._parent_code.particles)
        self.subcodes = dict()
        self._time_offsets = dict()
        
        self.E0 = None
        self.dt_step = 0
        
        self._major_channel_maker()
        
        # Validation
        self._validate_initialization()

    def _validate_initialization(self):
        if self.__dt is None or self.__dt <= 0 | units.s:
            raise ValueError("Error: dt must be a positive float")
        if not isinstance(self.__code_dt, (int, float)) or self.__code_dt <= 0:
            raise ValueError("Error: code_dt must be a positive float")
        if not isinstance(self.__par_nworker, int) or self.__par_nworker <= 0:
            raise ValueError("Error: par_nworker must be a positive integer")
        if self.__min_mass_evol_evol is None or self.__min_mass_evol_evol <= 0 | units.kg:
            raise ValueError("Error: __min_mass_evol_evol must be a positive float")
        if not isinstance(self.__coll_dir, str):
            raise ValueError("Error: coll_dir must be a string")
        if not isinstance(self.__ejected_dir, str):
            raise ValueError("Error: ejected_dir must be a string")

    def commit_particles(self):
        """Commit particle system by:
            - recentering the system
            - setting the parent radius
            - initialising children codes when needed
            - defining particles to evolve with stellar codes
            - setting up the galactic field
        """
        particles = self.particles
        particles.recenter_subsystems()
        length_unit = particles.radius.unit
        if not hasattr(particles, "sub_worker_radius"):
            particles.sub_worker_radius = 0. | length_unit

        for parent, code in self.subcodes.items():
            if ((parent in self.subsystems) and \
                (self.subsystems[parent] is self.subcodes[parent].particles)):
                continue
            self._time_offsets.pop(code)
            del code
        
        particles.radius = set_parent_radius(particles.mass, self.__dt, 1)
        for parent, sys in self.subsystems.items():
            parent.radius = set_parent_radius(np.sum(sys.mass), self.__dt, len(sys))
            if parent not in self.subcodes:
                code = self._sub_worker(sys)
                self._time_offsets[code] = self.model_time - code.model_time
                self.subsystems[parent] = sys
                self.subcodes[parent] = code
        particles[particles.radius > self._max_radius].radius = self._max_radius
        particles[particles.radius < self._min_radius].radius = self._min_radius
        particles.radius = 1000 | units.au
        particles[particles.syst_id > 0].radius = 50000 | units.au

        if (self.__star_evol):
            parti = particles.all()
            self.stars = parti[parti.mass > self.__min_mass_evol_evol]
            stellar_code = self._stellar_code
            stellar_code.particles.add_particle(self.stars)

        if (self.__gal_field):
            self._setup_bridge()
        else:
            self._evolve_code = self._parent_code
        
    def _setup_bridge(self):
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
            code (Code):  Gravitational integrator with particle set
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
            code (Code):  Gravitational integrator with particle set
        """
        if len(children) == 0:
            raise ValueError("Error: No children to evolve")
        
        if (0. | units.kg) in children.mass:
            code = Huayno(self.__child_conv)
            code.particles.add_particles(children)
            code.parameters.timestep_parameter = 0.1
            code.set_integrator("OK")
        
        else:
            code = Huayno(self.__child_conv)
            code.particles.add_particles(children)
            code.parameters.timestep_parameter = 0.1
            code.set_integrator("SHARED8_COLLISIONS")
            
        # TO DO: Add REBOUND integrator
        """elif masses[-1] > 100.*masses[-2]:
            code = Rebound(self.__child_conv)
            code.particles.add_particles(children)
            code.set_integrator("WHFast")"""
        
        return code
     
    def _test_worker(self):
        """
        Kick integrator for isolated test particles
        
        Returns:
            gravity (code):  Test particle integrator
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
        self.channels = {"from_stellar_to_gravity":
                          stellar_particles.new_channel_to(parent_particles,
                          attributes=["mass"],
                          target_names=["mass"]),
                         "from_gravity_to_parents":
                          parent_particles.new_channel_to(self.particles,
                          attributes=["x","y","z","vx","vy","vz"],
                          target_names=["x","y","z","vx","vy","vz"])}
    
    def _calculate_total_energy(self) -> float:
        """
        Calculate systems total energy
        
        Returns:
            Etot (Float):  Cluster total energy
        """
        all_parts = self.particles.all()
        Ek = all_parts.kinetic_energy()
        Ep = all_parts.potential_energy()
        Etot = Ek + Ep
        return Etot
    
    def _energy_tracker(self) -> float:
        """
        Calculate system energy error
        
        Returns:
            dE (Float):  Relative energy error
        """
        Etot = self._calculate_total_energy()
        Etot += self.corr_energy
        dE = abs((Etot - self.E0)/self.E0)
        return dE  
      
    def _star_channel_copier(self):
        """Copy attributes from stellar code to grav. integrator particle set"""
        stars = self._stellar_code.particles
        self.channels["from_stellar_to_gravity"].copy()
        for children in self.subcodes.values():
            channel = stars.new_channel_to(children.particles)
            channel.copy_attributes(["mass", "radius"])
            
    def _grav_channel_copier(self, transfer_data, receive_data, attributes=["x","y","z","vx","vy","vz"]):
        """
        Communicate information between grav. integrator and local particle set
        
        Args:
            transfer_data (Code):  Particle set to transfer data from
            receive_data (Code):  Particle set to update data
            attributes (Array):  Attributes wanting to copy
        Returns:
            channel (Channel): Channel to communicate between two codes
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
        while self.model_time < (tend - timestep/2.):
            self.corr_energy = 0. | units.J
            self.dt_step += 1
            updated_step = self.model_time
            
            if (self.__star_evol):
                self._stellar_evolution(evolve_time + timestep/2.)
                self._star_channel_copier()
            
            if (self._dE_track):
                E0 = self._calculate_total_energy()
            if self.model_time != (0. | units.s):
                self._correction_kicks(
                    self.particles, 
                    self.subsystems,
                    timestep/2.
                )
            if (self._dE_track):
                E1 = self._calculate_total_energy()
                self.corr_energy += E1 - E0
                
            self._drift_global(
                evolve_time + timestep, 
                evolve_time + timestep/2.
            )
            self._drift_child(self.model_time)
            if len(self.asteroids) > 0:# and (self.dt_step % self._kick_ast_iter) == 0:
                self._drift_test_particles(self.model_time)

            if (self.__star_evol):
                self._stellar_evolution(self.model_time)

            if (self._dE_track):
                E0 = self._calculate_total_energy()
            
            kick_dt = self.model_time - updated_step - timestep/2.
            self._correction_kicks(
                self.particles, 
                self.subsystems,
                kick_dt
            )
            if (self._dE_track):
                E1 = self._calculate_total_energy()
                self.corr_energy += E1 - E0
            
            self._split_subcodes()
            ejected_idx = ejection_checker(self.particles.copy(), 
                                           self.__gal_field)
            self._ejection_remover(ejected_idx)
        gc.collect()
            
    def _split_subcodes(self):
        """Track parent system dissolution. Satisfied if a particles nearest neighbour rij > 2*radius"""
        print("...Checking Splits...")
            
        asteroid_batch = [ ]
        for parent, subsys in list(self.subsystems.items()):
            
            radius = parent.radius
            furthest = (subsys.position - subsys.center_of_mass()).lengths().max()
            if furthest > radius:
                
                components = subsys.connected_components(threshold=2.*radius)
                if len(components) > 1:  # Checking for dissolution of system
                    parent_pos = parent.position
                    parent_vel = parent.velocity
                    
                    self.particles.remove_particle(parent)
                    code = self.subcodes.pop(parent)
                    self._time_offsets.pop(code)
                    
                    for c in components:
                        sys = c.copy_to_memory()
                        if sys.mass.max() == (0. | units.kg):
                            print(f"Removing {len(sys)} asteroids")
                            sys.position += parent_pos
                            sys.velocity += parent_vel
                            asteroid_batch.append(sys)
                            
                        else:
                            if len(sys) > 1:
                                sys.position += parent_pos
                                sys.velocity += parent_vel
                                newcode = self._sub_worker(sys)
                                newcode.particles.move_to_center()
                                
                                self._time_offsets[newcode] = self.model_time - newcode.model_time
                                newparent = self.particles.add_subsystem(sys)
                                self.subcodes[newparent] = newcode
                                newparent.radius = set_parent_radius(np.sum(sys.mass), 
                                                                     self.__dt, 
                                                                     len(sys))
                                if newparent.radius > self._max_radius:
                                    newparent.radius = self._max_radius
                                elif newparent.radius < self._min_radius:
                                    newparent.radius = self._min_radius
                            
                            else:
                                sys.position += parent_pos
                                sys.velocity += parent_vel
                                newparent = self.particles.add_subsystem(sys)
                                newparent.radius = set_parent_radius(np.sum(sys.mass), 
                                                                     self.__dt, 
                                                                     len(sys))
                                if newparent.radius > self._max_radius:
                                    newparent.radius = self._max_radius
                                elif newparent.radius < self._min_radius:
                                    newparent.radius = self._min_radius
                        
                    del code, subsys, parent
        
        if asteroid_batch:
            for isolated_asteroids in asteroid_batch:
                self.asteroids.add_particles(isolated_asteroids)
                self.asteroid_code.particles.add_particles(isolated_asteroids)
               
    def _ejection_remover(self, ejected_idx):
        """
        Remove ejected particles from system. Save their properties to a file.
        
        Args:
            ejected_idx (array):  Array containing booleans flagging for particle ejections
        """
        if (self._dE_track):
            E0 = self._calculate_total_energy()
        
        ejected_particles = self.particles[ejected_idx]
        for ejected_particle in ejected_particles:
            self._nejec += 1
            
            print(f"...Ejection #{self._nejec} Detected...")
            if ejected_particle in self.subcodes:
                code = self.subcodes.pop(ejected_particle)
                
                sys = self.subsystems[ejected_particle]
                filename = os.path.join(self.__ejected_dir, f"cluster_escapers")
                print(f"System pop: {len(sys)}")
                                    
                write_set_to_file(
                    sys.savepoint(0. | units.Myr), 
                    filename, 'amuse', close_file=True, 
                    append_to_file=True
                )
                
                del code
        
        if len(ejected_particles) > 0:
            ejected_stars = ejected_particles[ejected_particles.mass > self.__min_mass_evol_evol]
            self._stellar_code.particles.remove_particles(ejected_stars)
            
            self.particles.remove_particle(ejected_particles)
            self.particles.synchronize_to(self._parent_code.particles)
            
            if (self._dE_track):
                E1 = self._calculate_total_energy()
                self.corr_energy += E1 - E0
    
    def _create_new_children_from_mergers(self, job_queue, model_time, lock):
        """Create new children systems based on parent mergers.
        
        Args:
            job_queue (Queue):  Queue of jobs, each hosting new parent systems
            model_time (Float):  Time of simulation
            lock (Lock):  Lock to prevent interfering communication
        """
        new_children, time_offset = job_queue.get()
        
        newcode = self._sub_worker(new_children)
        newcode.particles.move_to_center()
        with lock:
            newparent = self.particles.add_subsystem(new_children)
            newparent.radius = set_parent_radius(np.sum(new_children.mass), 
                                                 self.__dt, 
                                                 len(new_children))
    
        self._time_offsets[newcode] = model_time - time_offset
        self.subcodes[newparent] = newcode
        self.subsystems[newparent] = new_children
            
    def _process_parent_mergers(self):
        model_time = self.model_time
        self.channels["from_gravity_to_parents"].copy()
        
        lock = threading.Lock()
        new_parent_queue = queue.Queue()
        nitems = 0
        for new_parent, new_children in self.__new_systems.items():
            n = self._parent_code.particles[self._parent_code.particles.key == new_parent.key]
            pos_shift = n.position - new_parent.position
            vel_shift = n.velocity - new_parent.velocity
            new_children.position += pos_shift
            new_children.velocity += vel_shift 
            time_offset = self.__new_offsets[new_parent]
            
            new_parent_queue.put((new_children, time_offset))
            self.particles.remove_particle(new_parent)
            nitems += 1
        
        threads = [ ]
        for worker in range(nitems):
            th = threading.Thread(target=self._create_new_children_from_mergers, 
                                  args=(new_parent_queue, model_time, lock, ))
            th.daemon = True
            th.start()
            threads.append(th)
            
        for th in threads:
            th.join()
            
        # Modify radius
        new_parents = self.particles[-nitems:]
        new_parents[new_parents.radius > self._max_radius].radius = self._max_radius
        new_parents[new_parents.radius < self._min_radius].radius = self._min_radius
    
    def _parent_merger(self, coll_time, corr_time, coll_set) -> Particle:
        """Resolve the merging of two parent systems.
        
        Args:
            coll_time (Float):  Time of collision
            corr_time (Float):  Collision correction time
            coll_set (Particle set):  Colliding particle set
        Returns:
            newparent (ParticleSuperset):  Superset containing new parent and children
        """
        self.channels["from_gravity_to_parents"].copy()
        
        
        ### Delay this until children loop to resolve all mergers in parallel?
        coll_array = self._evolve_coll_offset(coll_set, coll_time)
        collsubset = coll_array[0] 
        collsyst = coll_array[1] 
        
        dt = coll_time - corr_time
        self._correction_kicks(collsubset, collsyst, dt)
        
        newparts = HierarchicalParticles(Particles())
        radius = 0. | units.au
        for parti_ in collsubset:
            parti_ = parti_.as_particle_in_set(self.particles)
            radius = max(radius, parti_.radius)
            if parti_ in self.subcodes:  # If collider is a parent with children
                code = self.subcodes.pop(parti_)
                self._time_offsets.pop(code)
                
                parts = code.particles.copy_to_memory()
                sys = self.subsystems[parti_]
                self._grav_channel_copier(parts, sys)
                
                sys.position += parti_.position
                sys.velocity += parti_.velocity
                newparts.add_particles(sys)

                code.stop()
                del code
              
            else:  # If collider is isolated prior
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
        """Function to evolve and/or resync the final moments of collision.
        
        Args:
            coll_set (Particle set):  Attributes of colliding particle
            coll_time (Float):  Time of simulation where collision occurs
        Returns:
            coll_array (List): List with one element hosting particle set 
            and the other with the childrens of merging particles
        """
        collsubset = Particles()
        collsyst = dict()
        for parti_ in coll_set:
            collsubset.add_particle(parti_)
            
            # If a recently merged parent merges in the same time-loop, you need to give it children
            if parti_ in self.__new_systems:  
                collsubset.remove_particle(parti_)

                evolved_parent = self._parent_code.particles[self._parent_code.particles.key == parti_.key]
                children = self.__new_systems[parti_]
                offset = self.__new_offsets[parti_]

                pos_shift = evolved_parent.position - parti_.position
                vel_shift = evolved_parent.velocity - parti_.velocity
                children.position += pos_shift
                children.velocity += vel_shift

                newcode = self._sub_worker(children)
                newcode.particles.move_to_center()

                newparent = self.particles.add_subsystem(children)
                newparent.radius = set_parent_radius(np.sum(children.mass),
                                                      self.__dt,
                                                      len(children))

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
                
                print("Evolving for: ", (coll_time - offset).in_(units.Myr)) 
                while code.model_time < (coll_time - offset)*(1. - 1e-12):
                    code.evolve_model(coll_time-offset)
                    
                    if stopping_condition.is_set():
                        print("!!! COLLIDING CHILDREN !!!")
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
                                    coll_a = code.particles[code.particles.key == resolved_keys[coll_a.key]]
                                if coll_b.key in resolved_keys.keys():
                                    coll_b = code.particles[code.particles.key == resolved_keys[coll_b.key]]

                                if coll_b.key == coll_a.key:
                                    print("Curious?")
                                    continue

                                colliding_particles = Particles(particles=[coll_a, coll_b])
                                newparent, resolved_keys = self._handle_collision(self.subsystems[newparent], newparent, 
                                                                                 colliding_particles, coll_time, 
                                                                                 code, resolved_keys)
                                Nresolved += 1
                        
                        collsubset.add_particle(newparent)
                
        for parti_ in collsubset:              
            if parti_ in self.subsystems:
                collsyst[parti_] = self.subsystems[parti_]
                
        return [collsubset, collsyst]
    
    def _handle_collision(self, children, parent, enc_parti, tcoll, code, resolved_keys):
        """
        Merge two particles if the collision stopping condition is met
        
        Args:
            children (Particle set):  The children particle set
            parent (Particle):  The parent particle
            enc_parti (Particle set): The particles in the collision
            tcoll (Float):  The time-stamp for which the particles collide at
            code (Code):  The integrator used
            resolved_keys (Dict):  Dictionary holding {Collider i Key: Remnant Key}
        Returns:
            newparent (ParticleSuperset):  New parent particle
            resolved_keys (Dictionary):  Keys of merging particles
        """
        # Save properties
        allparts = self.particles.all()
        nmerge = np.sum(allparts.coll_events) + 1
        print(f"...Collision #{nmerge} Detected...")
        write_set_to_file(allparts.savepoint(0. | units.Myr),
            os.path.join(self.__coll_dir, f"merger{nmerge}"),
            'amuse', close_file=True, overwrite_file=True
        )
        
        coll_a = children[children.key == enc_parti[0].key]
        coll_b = children[children.key == enc_parti[1].key]
        
        collider = Particles()
        collider.add_particle(coll_a)
        collider.add_particle(coll_b)
        
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
        f.close()
        
        # Create merger remnant
        if max(collider.mass) > 0 | units.kg:
            remnant  = Particles(1)
            remnant.mass = collider.total_mass()
            remnant.position = collider.center_of_mass()
            remnant.velocity = collider.center_of_mass_velocity()
        else:
            raise ValueError("Error: Asteroid - Asteroid collision")
            
        if "HOST" in coll_a.type or "STAR" in coll_b.type:
            remnant.type = "HOST"
            remnant.radius = ZAMS_radius(remnant.mass)
        elif remnant.mass > self.__min_mass_evol_evol:
            remnant.type = "STAR"
            remnant.radius = ZAMS_radius(remnant.mass)
        else:
            most_massive = collider.mass.argmax()
            remnant.type = collider[most_massive].type
            remnant.radius = planet_radius(remnant.mass)
            
        if remnant.mass > self.__min_mass_evol_evol: #Lower limit for star evolution
            self._stellar_code.particles.add_particle(remnant)
            
        remnant.coll_events = collider.coll_events.sum() + 1
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
            self._time_offsets[newparent] = self.model_time - old_code.model_time
            self.subcodes[newparent] = old_code
        
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
    
    def _handle_supernova(self, SN_detect, bodies):
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
            coll_sets (Particle set): Set of colliding particles
        """
        coll_sets = UnionFind()
        for p,q in zip(p1, p2):
            coll_sets.union(p, q)
        return coll_sets.sets()

    def _stellar_evolution(self, dt):
        """
        Evolve stellar evolution
        
        Args:
            dt (Float):  Time to evolve till
        """
        SN_detection = self._stellar_code.stopping_conditions.supernova_detection
        SN_detection.enable()
        while self._stellar_code.model_time < dt:
            self._stellar_code.evolve_model(dt)
            
            if SN_detection.is_set():
                print("...Detection: SN Explosion...")
                self._handle_supernova(SN_detection, self.stars)
    
    def _drift_test_particles(self, dt):
        """
        Kick and evolve isolated test particles
        
        Args:
            dt (float):  Time to drift test particles
        """
        print(f"...Drifting {len(self.asteroids)} Asteroids...")
        dt = dt - self._asteroid_offset
        self._asteroid_code.evolve_model(dt)
        self._asteroid_code.particles.new_channel_to(self.asteroids).copy()
        
        print(self._asteroid_code.model_time.in_(units.Myr))
        for particle in self.particles:  # Check if any asteroids lies within a parent's radius
            distances = (self.asteroids.position - particle.position).lengths()
            newsystem = self.asteroids[distances < 1.5*particle.radius]
            
            if newsystem:
                newparts = HierarchicalParticles(Particles())
                if particle in self.subcodes:
                    print("...Merging asteroid with parent...")
                    code = self.subcodes.pop(particle)
                    offset = self._time_offsets.pop(code)
                    subsys = self.subsystems[particle]

                    subsys.position += particle.position
                    subsys.velocity += particle.velocity 
                    newparts.add_particles(subsys)
                    newparts.add_particle(newsystem)
                    
                    code.stop()
                    del code

                else:
                    print("...Merging asteroid with isolated body...")
                    newparts.add_particle(particle)
                    newparts.add_particle(newsystem)
                    
                self.asteroids.remove_particle(newsystem)
                self._asteroid_code.particles.remove_particle(newsystem)
                self.particles.remove_particle(particle)
                
                newcode = self._sub_worker(newparts)
                newcode.particles.move_to_center()  # Prevent energy drift
                newparent = self.particles.add_subsystem(newparts)
                newparent.radius = set_parent_radius(np.sum(newparts.mass), self.__dt, len(newparts))
                if newparent.radius > self._max_radius:
                    newparent.radius = self._max_radius
                elif newparent.radius < self._min_radius:
                    newparent.radius = self._min_radius
                    
                self._time_offsets[newcode] = self.model_time - newcode.model_time
                self.subcodes[newparent] = newcode
        
        if len(self._asteroid_code.particles) == 0:
            self._asteroid_code.stop()
            self._asteroid_offset = self.model_time
                    
    def _drift_global(self, dt, corr_time):
        """
        Evolve parent system until dt
        
        Args:
            dt (Float):  Time to evolve till
            corr_time (Float): Time to correct for drift
        """
                
        print("...Drifting Global...")
        stopping_condition = self._parent_code.stopping_conditions.collision_detection
        stopping_condition.enable()
        self.__new_systems = dict()
        self.__new_offsets = dict()
        coll_time = None
        
        print(f"Goal: {dt.in_(units.Myr)}")
        while self._evolve_code.model_time < dt*(1. - 1.1e-12):
            print(f"Current: {self._parent_code.model_time.in_(units.Myr)}", end=" ")
            print(f"# Particles {len(self._parent_code.particles)}")
            
            self._evolve_code.evolve_model(dt)
            if stopping_condition.is_set():
                if (self._dE_track):
                    E0 = self._calculate_total_energy()
                    
                coll_time = self._parent_code.model_time
                coll_sets = self._find_coll_sets(stopping_condition.particles(0), 
                                                 stopping_condition.particles(1))
                
                print(f"Bridge: Parent Merger. T = {coll_time.in_(units.Myr)}")
                for cs in coll_sets:
                    self._parent_merger(coll_time, corr_time, cs)
                    
                if (self._dE_track):
                    E1 = self._calculate_total_energy()
                    self.corr_energy += E1 - E0

        if (self.__gal_field):
            while self._parent_code.model_time < dt*(1. - 1.1e-12):
                self._parent_code.evolve_model(dt)
                if stopping_condition.is_set():
                    if (self._dE_track):
                        E0 = self._calculate_total_energy()
                    
                    coll_time = self._parent_code.model_time
                    coll_sets = self._find_coll_sets(stopping_condition.particles(0), 
                                                    stopping_condition.particles(1))
                    
                    print(f"Parent: Parent Merger. T = {coll_time.in_(units.Myr)}")
                    for cs in coll_sets:
                        self._parent_merger(coll_time, corr_time, cs)
                        
                    if (self._dE_track):
                        E1 = self._calculate_total_energy()
                        self.corr_energy += E1 - E0
        if (coll_time):
            self._process_parent_mergers()
            STOP
            
        if (1):
            print(f"Parent code time: {self._parent_code.model_time.in_(units.Myr)}")
            print(f"Bridge code time: {self._evolve_code.model_time.in_(units.Myr)}")
            print(f"Stellar code time: {self._stellar_code.model_time.in_(units.Myr)}")
        
    def _drift_child(self, dt):
        """
        Evolve children system until dt.
        
        Args:
            dt (Float):  Time to evolve till
        """     
        def resolve_collisions(code, parent, stopping_condition):
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
                        coll_a = code.particles[code.particles.key == resolved_keys[coll_a.key]]
                    if coll_b.key in resolved_keys.keys():
                        coll_b = code.particles[code.particles.key == resolved_keys[coll_b.key]]
                        
                    if coll_b.key == coll_a.key:
                        print("Curious?")
                        continue
                    
                    colliding_particles = Particles(particles=[coll_a, coll_b])
                    parent, resolved_keys = self._handle_collision(self.subsystems[parent], parent, 
                                                                  colliding_particles, coll_time, 
                                                                  code, resolved_keys)
                    Nresolved += 1
            
            return parent
        
        def evolve_code(lock):
            """
            Algorithm to evolve individual children codes
            
            Args:
                lock:  Lock to prevent simultaneous access to shared resources
            """
            try:
                parent = parent_queue.get(timeout=1)  # Timeout to prevent blocking indefinitely
            except queue.Empty:
                raise ValueError("Error: No children in system")
                
            code = self.subcodes[parent]
            evol_time = dt - self._time_offsets[code]
            stopping_condition = code.stopping_conditions.collision_detection
            stopping_condition.enable()
            
            while code.model_time < evol_time*(1. - 1.e-12):
                code.evolve_model(evol_time)
                
                if stopping_condition.is_set():
                    print("!!! COLLIDING CHILDREN !!!")
                    
                    with lock:  # All threads stop until resolve collision
                        if (self._dE_track):
                            E0 = self._calculate_total_energy()
                            
                        parent = resolve_collisions(code, parent, stopping_condition)
                        if (self._dE_track):
                           E1 = self._calculate_total_energy()
                           self.corr_energy += E1 - E0
                                
            parent_queue.task_done()
            
        print("...Drifting Children...")
        parent_queue = queue.Queue()
        for parent in self.subcodes.keys():
            parent_queue.put(parent)
            self.key = parent.key
            
        lock = threading.Lock()
        threads = []
        for worker in range(len(self.subcodes.values())):
            th = threading.Thread(target=evolve_code, args=(lock,))
            th.daemon = True
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
            
            del old_subcode
            del old_offset      
                
    def _kick_particles(self, particles, corr_code, dt):
        """Kick particle set
        
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

    def _correction_kicks(self, particles, subsystems, dt):
        """
        Apply correcting kicks onto children and parent particles
        
        Args:
            particles (Particle set):  Parent particle set
            subsystems (Dictionary):  Dictionary of children system
            dt (Float):  Time interval for applying kicks
        """
        if subsystems and len(particles) > 1:
            # Kick parent particles
            corr_chd = CorrectionFromCompoundParticle(particles, 
                                                      subsystems)
            self.channels["from_gravity_to_parents"].copy()
            self._kick_particles(particles, corr_chd, dt)
            self._grav_channel_copier(
                particles, 
                self._parent_code.particles,
                ["vx","vy","vz"]
            )
            
            # Kick children
            corr_par = CorrectionForCompoundParticle(particles, 
                                                     parent=None, 
                                                     system=None)
            for parent, subsyst in subsystems.items():
                corr_par.parent = parent
                corr_par.system = subsyst
                
                self._grav_channel_copier(
                    self.subcodes[parent].particles, 
                    subsyst
                )
                self._kick_particles(subsyst, corr_par, dt)
                self._grav_channel_copier(
                    subsyst, self.subcodes[parent].particles, 
                    ["vx","vy","vz"]
                )
    
    @property
    def model_time(self):  
        """Extract the global integrator model time"""
        return self._parent_code.model_time
    
    @property
    def subsystems(self):
        """Extract the children system"""
        return self.particles.collection_attributes.subsystems
    