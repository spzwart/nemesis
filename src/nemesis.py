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
from amuse.ext.orbital_elements import orbital_elements_from_binary
from amuse.lab import write_set_to_file
from amuse.units import units, constants

from src.environment_functions import ejection_checker, set_parent_radius
from src.environment_functions import planet_radius, ZAMS_radius
from src.grav_correctors import CorrectionFromCompoundParticle
from src.grav_correctors import CorrectionForCompoundParticle
from src.hierarchical_particles import HierarchicalParticles


class Nemesis(object):
    def __init__(self, par_conv, child_conv, dt,
                 code_dt=0.03, par_nworker=1, 
                 dE_track=False, star_evol=False, 
                 gal_field=False):
        """
        Class setting up the simulation.
        
        Args:
            par_conv (Converter):  Parent N-body converter
            child_conv (Converter):  Children N-body converter
            dt (Float):  Diagnostic time step
            code_dt (Float):  Internal time step
            par_nworker (Int):  Number of workers for global integrator
            dE_track (Boolean):  Flag turning on/off energy error tracker
            star_evol (Boolean):  Flag turning on/off stellar evolution
            gal_field (Boolean):  Flag turning on/off galactic field
        """
        self.code_dt = code_dt
        self.par_nworker = par_nworker
        self.star_evol = star_evol
      
        self.parent_code = self.parent_worker(par_conv)
        if (self.star_evol):
            self.stellar_code = self.stellar_worker()

        self.child_conv = child_conv
        self.dt = dt
        self.dE_track = dE_track
        self.gal_field = gal_field
        self.test_particles = Particles()
      
        self.particles = HierarchicalParticles(self.parent_code.particles)
        self.subcodes = dict()
        self.time_offsets = dict()
        
        self.min_mass_evol = None
        self.coll_dir = None
        self.ejected_dir = None
        self.E0 = None
        self.max_radius = 1500. | units.au
        self.nejec = 0
        self.dt_step = 0
        
        if not isinstance(dt, (int, float)) or dt <= 0:
            raise ValueError("Error: dt must be a positive float")
        if not isinstance(code_dt, (int, float)) or code_dt <= 0:
            raise ValueError("Error: code_dt must be a positive float")
        if not isinstance(par_nworker, int) or par_nworker <= 0:
            raise ValueError("Error: par_nworker must be a positive integer")
        if not isinstance(self.min_mass_evol, (int, float)) or self.min_mass_evol <= 0:
            raise ValueError("Error: min_mass_evol must be a positive float")
        if not isinstance(self.coll_dir, str):
            raise ValueError("Error: coll_dir must be a string")
        if not isinstance(self.ejected_dir, str):
            raise ValueError("Error: ejected_dir must be a string")

    def commit_particles(self):
        """Commit particle system by:
            - recentering the system
            - setting the parent radius
            - initialising children codes when needed
            - defining particles to evolve with stellar codes
            - setting up the galactic field
        """
        self.particles.recenter_subsystems()
        length_unit = self.particles.radius.unit
        if not hasattr(self.particles, "sub_worker_radius"):
            self.particles.sub_worker_radius = 0. | length_unit

        for parent, code in self.subcodes.items():
            if ((parent in self.subsystems) and \
                (self.subsystems[parent] is self.subcodes[parent].particles)):
                continue
            self.time_offsets.pop(code)
            del code
        
        self.particles.radius = set_parent_radius(self.particles.mass, self.dt, 1)
        for parent, sys in self.subsystems.items():
            parent.radius = set_parent_radius(np.sum(sys.mass), self.dt, len(sys))
            if parent not in self.subcodes:
                code = self.sub_worker(sys)
                self.time_offsets[code] = self.model_time - code.model_time
                self.subsystems[parent] = sys
                self.subcodes[parent] = code
        self.particles[self.particles.radius > self.max_radius].radius = self.max_radius

        if (self.star_evol):
            parti = self.particles.all()
            self.stars = parti[parti.mass > self.min_mass_evol]
            stellar_code = self.stellar_code
            stellar_code.particles.add_particle(self.stars)

        if (self.gal_field):
            self.setup_bridge()
        else:
            self.evolve_code = self.parent_code

    def setup_bridge(self):
        """Embed system into galactic potential"""
        self.MWG = MWpotentialBovy2015()
        gravity = bridge.Bridge(use_threading=True,
                                method=SPLIT_4TH_S_M6,)
        gravity.add_system(self.parent_code, (self.MWG, ))
        gravity.timestep = self.dt
        self.grav_bridge = gravity
        self.evolve_code = self.grav_bridge
    
    def stellar_worker(self) -> SeBa:
        """Define stellar evolution integrator"""
        return SeBa()

    def parent_worker(self, par_conv) -> ph4:
        """
        Define global integrator
        
        Args:
            par_conv (converter):  Converter for global integrator
        Returns:
            code (Code):  Gravitational integrator with particle set
        """
        code = ph4(par_conv, number_of_workers=self.par_nworker)
        code.parameters.epsilon_squared = (0. | units.au)**2
        code.parameters.timestep_parameter = self.code_dt
        return code
      
    def sub_worker(self, children):
        """
        Initialise children integrator
        
        Args:
            children (Particle set):  Children systems
        Returns:
            code (Code):  Gravitational integrator with particle set
        """
        if len(children) == 0:
            raise ValueError("Error: No children to evolve")
        
        if children.mass.min() == (0. | units.kg):
            print("...Integrator: Test Particle...")
            code = Huayno(self.child_conv)
            code.particles.add_particles(children)
            code.parameters.timestep_parameter = 0.1
            code.set_integrator("OK")
        
        else:
            print("...Integrator: Huayno...")
            code = Huayno(self.child_conv)
            code.particles.add_particles(children)
            code.parameters.timestep_parameter = 0.1
            code.set_integrator("SHARED8_COLLISIONS")
            
        # TO DO: Add REBOUND integrator
        """elif masses[-1] > 100.*masses[-2]:
            print("...Integrator: Rebound...")
            code = Rebound(self.child_conv)
            code.particles.add_particles(children)
            code.set_integrator("WHFast")"""
        
        return code
            
    def test_worker(self):
        """
        Kick integrator for isolated test particles
        
        Returns:
            gravity (Code):  Gravitational integrator with particle set
        """
        code = Huayno(self.child_conv)
        code.particles.add_particles(self.test_particles)
        code.set_integrator("OK")
        self.test_code_to_local = code.particles.new_channel_to(self.test_particles)
        
        if (self.gal_field):
            gravity = bridge.Bridge(use_threading=False, method=SPLIT_4TH_S_M4)
            gravity.timestep = self.dt
            gravity.add_system(code, (self.parent_code, self.MWG))
            
        else:
            gravity = code
            
        return gravity

    def star_channel_copier(self):
        """Copy attributes from stellar code to grav. integrator particle set"""
        stars = self.stellar_code.particles
        stars.new_channel_to(self.parent_code.particles).copy_attributes(["mass"])
        for children in self.subcodes.values():
            channel = stars.new_channel_to(children.particles)
            channel.copy_attributes(["mass", "radius"])
            
    def grav_channel_copier(self, transfer_data, receive_data, attributes):
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
    
    def energy_tracker(self) -> float:
        """
        Calculate system energy error
        
        Returns:
            dE (Float):  Relative energy error
        """
        Etot = self.calculate_total_energy()
        Etot += self.corr_energy
        dE = abs((Etot - self.E0)/self.E0)
        return dE
    
    def calculate_total_energy(self) -> float:
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
        
    def evolve_model(self, tend, timestep=None):
        """
        Evolve the system until tend
        
        Args:
            tend (Float):  Time to simulate till
            timestep (Float):  Timestep to simulate
        """
        if timestep is None:
            timestep = tend - self.model_time
        
        while self.model_time < (tend - timestep/2.):
            evolve_time = self.model_time
            self.corr_energy = 0. | units.J
            self.dt_step += 1
            
            if (self.star_evol):
                self.stellar_evolution(self.model_time + timestep/2.)
                self.star_channel_copier()
            
            if (self.dE_track):
                E0 = self.calculate_total_energy()
            if self.model_time != (0. | units.s):
                self.correction_kicks(
                    self.particles, 
                    self.subsystems,
                    timestep/2.
                )
            if (self.dE_track):
                E1 = self.calculate_total_energy()
                self.corr_energy += E1 - E0
                
            self.drift_global(evolve_time + timestep, 
                              evolve_time + timestep/2.
                              )
            self.drift_child(self.parent_code.model_time)
            if len(self.test_particles) > 0: #and self.dt_step % 10 == 0:
                dt = self.parent_code.model_time - evolve_time
                self.drift_test_particles(dt)

            if (self.star_evol):
                self.stellar_evolution(self.parent_code.model_time)

            if (self.dE_track):
                E0 = self.calculate_total_energy()
            kick_dt = self.parent_code.model_time - self.model_time - timestep/2.
            self.correction_kicks(
                self.particles, 
                self.subsystems,
                kick_dt
            )
            if (self.dE_track):
                E1 = self.calculate_total_energy()
                self.corr_energy += E1 - E0
            
            self.split_subcodes()
            ejected_idx = ejection_checker(self.particles.copy(), 
                                           self.gal_field)
            self.ejection_remover(ejected_idx)
            
    def split_subcodes(self):
        """
        Track parent system dissolution. Children are removed from parent 
        if their distance is greater than twice the parent radius
        """
        print("...Checking Splits...")
            
        for parent, subsys in list(self.subsystems.items()):
            radius = parent.radius
            self.grav_channel_copier(
                self.subcodes[parent].particles, subsys,
                ["x","y","z","vx","vy","vz"]
            )
            components = subsys.connected_components(threshold=2.*radius)
            
            if len(components) > 1:  # Checking for dissolution of system
                parent_pos = parent.position
                parent_vel = parent.velocity
                
                self.particles.remove_particle(parent)
                code = self.subcodes.pop(parent)
                self.time_offsets.pop(code)
                
                for c in components:
                    sys = c.copy_to_memory()
                    if len(sys) > 1:
                        if max(sys.mass) == (0. | units.kg):
                            print("...new asteroid conglomerate...")
                            for asteroids in [self.asteroids, self.asteroid_code.particles]:
                                asteroids.add_particles(sys)
                                asteroids[-len(sys):].position += parent_pos
                                asteroids[-len(sys):].velocity += parent_vel
                                
                        else:
                            print("...new children system...")
                            sys.position += parent_pos
                            sys.velocity += parent_vel
                            newcode = self.sub_worker(sys)
                            newcode.particles.move_to_center()
                            
                            self.time_offsets[newcode] = self.model_time - newcode.model_time
                            newparent = self.particles.add_subsystem(sys)
                            self.subcodes[newparent] = newcode
                            newparent.radius = set_parent_radius(np.sum(sys.mass), self.dt, len(sys))
                            if newparent.radius > self.max_radius:
                                newparent.radius = self.max_radius
                        
                    else:
                        if sys.mass == (0. | units.kg):
                            print("...new isolated asteroid...")
                            for asteroids in [self.asteroids, self.asteroid_code.particles]:
                                asteroids.add_particles(sys)
                                asteroids[-len(sys):].position += parent_pos
                                asteroids[-len(sys):].velocity += parent_vel
                            
                        else:
                            print("...new isolated particle...")
                            sys.position += parent_pos
                            sys.velocity += parent_vel
                            newparent = self.particles.add_subsystem(sys)
                            newparent.radius = set_parent_radius(np.sum(sys.mass), self.dt, len(sys))
                            if newparent.radius > self.max_radius:
                                newparent.radius = self.max_radius
                    
                del code
               
    def ejection_remover(self, ejected_idx):
        """
        Output and remove ejected particles from system
        
        Args:
            ejected_idx (array):  Array containing booleans flagging for particle ejections
        """
        print("...Checking Ejections...")
        if (self.dE_track):
            E0 = self.calculate_total_energy()
        
        ejected_particles = self.particles[ejected_idx]
        for ejected_particle in ejected_particles:
            self.nejec += 1
            
            print(f"...Ejection #{self.nejec} Detected...")
            if ejected_particle in self.subcodes:
                code = self.subcodes.pop(ejected_particle)
                
                sys = self.subsystems[ejected_particle]
                filename = os.path.join(self.ejected_dir, f"cluster_escapers")
                print(f"System pop: {len(sys)}")
                                    
                write_set_to_file(
                    sys.savepoint(0. | units.Myr), 
                    filename, 'amuse', close_file=True, 
                    append_to_file=True
                )
                
                del code
        
        self.particles.remove_particle(ejected_particles)    
        if (self.dE_track):
            E1 = self.calculate_total_energy()
            self.corr_energy += E1 - E0
    
    def parent_merger(self, coll_time, corr_time, coll_set) -> Particle:
        """Resolve the merging of two parent systems.
        
        Args:
            coll_time (Float):  Time of collision
            corr_time (Float):  Collision correction time
            coll_set (Particle set):  Colliding particle set
        Returns:
            newparent (ParticleSuperset):  Superset containing new parent and children
        """
        collsubset, collsyst = self.evolve_coll_offset(coll_set, coll_time)
        dt = coll_time - corr_time
        self.correction_kicks(collsubset, collsyst, dt)
        
        newparts = HierarchicalParticles(Particles())
        self.grav_channel_copier(
            self.parent_code.particles,
            self.particles,
            ["x","y","z","vx","vy","vz"]
        )
        
        for parti_ in collsubset:
            parti_ = parti_.as_particle_in_set(self.particles)
            
            if parti_ in self.subcodes:  # Check if collider is a parent with children
                print("...Merging into some existing system...")
                code = self.subcodes.pop(parti_)
                self.time_offsets.pop(code)
                parts = code.particles.copy_to_memory()
                sys = self.subsystems[parti_]
                
                channel = parts.new_channel_to(sys)
                channel.copy_attributes(["x","y","z","vx","vy","vz"])
                
                sys.position += parti_.position
                sys.velocity += parti_.velocity 
                newparts.add_particles(sys)

                code.stop()
                del code
              
            else:  # Loop for two parent particle collisions
                print("...Merging with some isolated object...")
                new_parti = newparts.add_particle(parti_)
                new_parti.radius = parti_.sub_worker_radius
                
            self.particles.remove_particle(parti_)
        
        newcode = self.sub_worker(newparts)
        newcode.particles.move_to_center()  # Prevent energy drift
        newparent = self.particles.add_subsystem(newparts)
        newparent.radius = set_parent_radius(np.sum(newparts.mass), 
                                             self.dt, 
                                             len(newparts))
        if newparent.radius > self.max_radius:
            newparent.radius = self.max_radius
            
        self.time_offsets[newcode] = self.model_time - newcode.model_time
        self.subcodes[newparent] = newcode
        
        if (self.gal_field):
            self.setup_bridge()
        
        return newparent
        
    def evolve_coll_offset(self, coll_set, coll_time):
        """Function to evolve and/or resync the final moments of collision.
        
        Args:
            coll_set (Particle set):  Attributes of colliding particle
            coll_time (Float):  Time of simulation where collision occurs
        Returns:
            collsubset (ParticleSet):  Particle set with merging particles \n
            collsyst (Dictionary):  Childrens of merging particles
        """
        collsubset = Particles()
        collsyst = dict()
        
        for parti_ in coll_set:
            collsubset.add_particle(parti_)
            
            if parti_ in self.subcodes:
                code = self.subcodes[parti_]
                offset = self.time_offsets[code]
                stopping_condition = code.stopping_conditions.collision_detection
                stopping_condition.enable()
                newparent = parti_.copy()
                
                print("Evolving for: ", (coll_time - offset).in_(units.Myr)) 
                while code.model_time < (coll_time - offset)*(1. - 1e-12):
                    code.evolve_model(coll_time-offset)
                    
                    if stopping_condition.is_set():
                        print("!!! COLLIDING CHILDREN !!!")
                        coll_time = code.model_time
                        coll_a_particles = stopping_condition.particles(0)
                        coll_b_particles = stopping_condition.particles(1)
                        
                        resolved_keys = dict()
                        Nmergers = max(len(np.unique(coll_a_particles.key)),  
                                       len(np.unique(coll_b_particles.key)))
                        Nresolved = 0

                        collsubset.remove_particle(newparent)
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
                                newparent, resolved_keys = self.handle_collision(self.subsystems[newparent], newparent, 
                                                                                 colliding_particles, coll_time, 
                                                                                 code, resolved_keys)
                                Nresolved += 1
                        
                        collsubset.add_particle(newparent)

        for parti_ in collsubset:              
            if parti_ in self.subsystems:
                collsyst[parti_] = self.subsystems[parti_]
        
        return collsubset, collsyst
    
    def handle_collision(self, children, parent, enc_parti, tcoll, code, resolved_keys):
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
            os.path.join(self.coll_dir, f"merger{nmerge}"),
            'amuse', close_file=True, overwrite_file=True
        )
        
        coll_a = children[children.key == enc_parti[0].key]
        coll_b = children[children.key == enc_parti[1].key]
        
        collider = Particles()
        collider.add_particle(coll_a)
        collider.add_particle(coll_b)
        
        kepler_elements = orbital_elements_from_binary(collider, G=constants.G)
        sem = kepler_elements[2]
        ecc = kepler_elements[3]
        inc = kepler_elements[4]
        with open(os.path.join(self.coll_dir, f"merger{nmerge}.txt"), 'w') as f:
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
            
        if "STAR" in coll_a.type or "STAR" in coll_b.type:
            remnant.type = "STAR"
            remnant.radius = ZAMS_radius(remnant.mass)
        elif "HOST" in coll_a.type or "HOST" in coll_b.type:
            remnant.type = "HOST"
            remnant.radius = ZAMS_radius(remnant.mass)
        else:
            remnant.type = "PLANET"
            remnant.radius = planet_radius(remnant.mass)
            
        if remnant.mass > self.min_mass_evol: #Lower limit for star evolution
            self.stellar_code.particles.add_particle(remnant)
            
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
        
        print(f"{coll_a.mass.in_(units.MSun)} + {coll_b.mass.in_(units.MSun)} --> {remnant.mass.in_(units.MSun)}")
        
        children.add_particles(remnant)
        children.remove_particles(coll_a)
        children.remove_particles(coll_b)
        nearest_mass = abs(children.mass - parent.mass).argmin()
        
        if remnant.key == children[nearest_mass].key:
            print("...New parent particle...")
            children.position += parent.position
            children.velocity += parent.velocity
            
            # Create new parent particle
            newparent = self.particles.add_subsystem(children)
            newparent.radius = parent.radius

            # Re-mapping dictionary to new parent
            old_code = self.subcodes.pop(parent)
            self.time_offsets[newparent] = self.model_time - old_code.model_time
            self.subcodes[newparent] = old_code
        
            self.particles.remove_particle(parent)
            children.synchronize_to(self.subcodes[newparent].particles)

        else:
            newparent = parent
            children.synchronize_to(code.particles)
        
        if coll_a.mass > self.min_mass_evol:
          self.stellar_code.particles.remove_particle(coll_a)
        if coll_b.mass > self.min_mass_evol:
          self.stellar_code.particles.remove_particle(coll_b)

        return newparent, resolved_keys
    
    def handle_supernova(self, SN_detect, bodies):
        """
        Handle SN events
        
        Args:
            SN_detect (StoppingCondition):  Detected particle set undergoing SN
            bodies (Particle set):  All bodies undergoing stellar evolution
        """
        if (self.dE_track):
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
            
        if (self.dE_track):
            E1 = self.calculate_total_energy()
            self.corr_energy += E1 - E0
            
    def find_coll_sets(self, p1, p2) -> UnionFind:
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

    def stellar_evolution(self, dt):
        """
        Evolve stellar evolution
        
        Args:
            dt (Float):  Time to evolve till
        """
        SN_detection = self.stellar_code.stopping_conditions.supernova_detection
        SN_detection.enable()
        while self.stellar_code.model_time < dt:
            self.stellar_code.evolve_model(dt)
            
            if SN_detection.is_set():
                print("...Detection: SN Explosion...")
                self.handle_supernova(SN_detection, self.stars)
    
    def drift_test_particles(self, dt):
        """
        Kick and evolve isolated test particles
        
        Args:
            dt (Float):  Time to evolve till
        """
        print(f"...Drifting {len(self.test_particles)} Asteroids...")
        gravity = self.test_worker()
        gravity.evolve_model(dt)
        self.test_code_to_local.copy()
        gravity.stop()
        
        new_system = False
        for particle in self.particles:  # Check if any asteroids lies within a parent's radius
            distances = (self.test_particles.position - particle.position).lengths()
            newsystem = self.test_particles[distances < 1.5*particle.radius]
            
            if newsystem:
                new_system = True
                newparts = HierarchicalParticles(Particles())
                if particle in self.subcodes:
                    print("...Merging asteroid with parent...")
                    code = self.subcodes.pop(particle)
                    offset = self.time_offsets.pop(code)
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
                    
                self.test_particles.remove_particle(newsystem)
                self.particles.remove_particle(particle)
                
                newcode = self.sub_worker(newparts)
                newcode.particles.move_to_center()  # Prevent energy drift
                newparent = self.particles.add_subsystem(newparts)
                newparent.radius = set_parent_radius(np.sum(newparts.mass), 
                                                     self.code_dt, 
                                                     len(newparts))
                if newparent.radius > self.max_radius:
                    newparent.radius = self.max_radius
                    
                self.time_offsets[newcode] = self.model_time - newcode.model_time
                self.subcodes[newparent] = newcode
                
        if (new_system) and (self.gal_field):
            self.setup_bridge()
                    
    def drift_global(self, dt, corr_time):
        """
        Evolve parent system until dt
        
        Args:
            dt (Float):  Time to evolve till
            corr_time (Float): Time to correct for drift
        """
        print("...Drifting Global...")
        stopping_condition = self.parent_code.stopping_conditions.collision_detection
        stopping_condition.enable()
        
        print(f"Goal: {dt.in_(units.Myr)}")
        timestep = dt - self.parent_code.model_time
        while self.parent_code.model_time < dt - timestep/2.:
            print(f"Current: {self.parent_code.model_time.in_(units.Myr)}", end=" ")
            print(f"# Particles {len(self.parent_code.particles)}")
            
            self.evolve_code.evolve_model(dt)
            if stopping_condition.is_set():
                if (self.dE_track):
                    E0 = self.calculate_total_energy()
                    
                print("!!! Parent Merger !!!")
                coll_time = self.parent_code.model_time
                coll_sets = self.find_coll_sets(stopping_condition.particles(0), 
                                                stopping_condition.particles(1)
                                                )
                for cs in coll_sets:
                    self.parent_merger(coll_time, corr_time, cs)
                    
                if (self.dE_track):
                    E1 = self.calculate_total_energy()
                    self.corr_energy += E1 - E0
                    
        print(f"Parent code time: {self.parent_code.model_time.in_(units.Myr)}")
        
    def drift_child(self, dt):
        """
        Evolve children system until dt.
        
        Args:
            dt (Float):  Time to evolve till
        """     
        def resolve_collisions(code, parent, stopping_condition):
            """Function to resolve collisions"""
            self.grav_channel_copier(
                code.particles, self.subsystems[parent],
                ["x","y","z","vx","vy","vz"]
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
                    parent, resolved_keys = self.handle_collision(self.subsystems[parent], parent, 
                                                                  colliding_particles, coll_time, 
                                                                  code, resolved_keys
                                                                  )
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
            evol_time = dt - self.time_offsets[code]
            stopping_condition = code.stopping_conditions.collision_detection
            stopping_condition.enable()
            
            while code.model_time < evol_time*(1. - 1.e-12):
                code.evolve_model(evol_time)
                
                if stopping_condition.is_set():
                    print("!!! COLLIDING CHILDREN !!!")
                    
                    with lock:  # All threads stop until resolve collision
                        if (self.dE_track):
                            E0 = self.calculate_total_energy()
                            
                        parent = resolve_collisions(code, parent, stopping_condition)
                        print("...collision resolved...")
                        
                        if (self.dE_track):
                           E1 = self.calculate_total_energy()
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
            old_offset = self.time_offsets.pop(old_subcode)
            
            del old_subcode
            del old_offset      
                
    def kick_particles(self, particles, corr_code, dt):
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

    def correction_kicks(self, particles, subsystems, dt):
        """
        Apply correcting kicks onto children and parent particles
        
        Args:
            particles (Particle set):  Parent particle set
            subsystems (Dictionary):  Dictionary of children system
            dt (Float):  Time interval for applying kicks
        """
        if subsystems and len(particles) > 1:
            attributes = ["x","y","z","vx","vy","vz"]
            
            # Kick parent particles
            corr_chd = CorrectionFromCompoundParticle(particles, 
                                                      subsystems)
            self.grav_channel_copier(
                self.parent_code.particles, 
                particles, attributes
            )
            self.kick_particles(particles, corr_chd, dt)
            self.grav_channel_copier(
                particles, 
                self.parent_code.particles,
                ["vx","vy","vz"]
            )
            
            # Kick children
            corr_par = CorrectionForCompoundParticle(particles, 
                                                     parent=None, 
                                                     system=None)
            for parent, subsyst in subsystems.items():
                corr_par.parent = parent
                corr_par.system = subsyst
                
                self.grav_channel_copier(
                    self.subcodes[parent].particles, 
                    subsyst, attributes
                )
                self.kick_particles(subsyst, corr_par, dt)
                self.grav_channel_copier(
                    subsyst, self.subcodes[parent].particles, 
                    ["vx","vy","vz"]
                )
    
    @property
    def model_time(self):  
        """Extract the global integrator model time"""
        return self.parent_code.model_time
    
    @property
    def subsystems(self):
        """Extract the children system"""
        return self.particles.collection_attributes.subsystems