
############################## NOTES ##############################
# Recentering subsystems is optimised. Even without GIL release, 
# overhead of multiprocessing is too consuming. Most likely the 
# AMUSE-based arrays (position/velocity) are already wrapped around 
# NumPy arrays.
###################################################################

from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

from amuse.datamodel import Particle, Particles, ParticlesOverlay
from amuse.units import units



class HierarchicalParticles(ParticlesOverlay):
    """Class to make particle set"""
    def __init__(self, *args, **kwargs):
        ParticlesOverlay.__init__(self,*args,**kwargs)
        self.collection_attributes.subsystems = dict()

    def add_particles(self, parts: Particles) -> Particles:  
        """
        Add particles to particle set.

        Args:
            parts (Particles):  The particle set to be added
        Returns:
            ParticlesOverlay:  The particle set
        """
        _parts = ParticlesOverlay.add_particles(self,parts)
        if hasattr(parts.collection_attributes, "subsystems"):
            for parent, sys in parts.collection_attributes.subsystems.values():
                parent = parent.as_particle_in_set(self)
                self.collection_attributes.subsystems[parent.key] = (parent, sys)

        return _parts

    def add_subsystem(self, sys: Particles, recenter=True) -> Particle:
        """
        Create a parent from particle subsytem

        Args:
            sys (Particles):  The subsystem set
            recenter (bool):  Flag to recenter the parent
        Returns:
            Particle:  The parent particle
        """
        if len(sys) == 1:
            return self.add_particles(sys)[0]

        parent = Particle()
        self.assign_parent_attributes(
            sys, parent, 
            relative=False, 
            recenter=recenter
        )
        parent = self.add_particle(parent)
        self.collection_attributes.subsystems[parent.key] = (parent, sys)

        return parent

    def assign_parent_attributes(self, sys: Particles, parent: Particle, relative=True, recenter=True) -> None:
        """
        Create parent from subsystem attributes

        Args:
            sys (Particles):  The subsystem set
            parent (Particle):  The parent particle
            relative (bool):  Flag to assign relative attributes
            recenter (bool):  Flag to recenter the parent
        """
        if not relative:
            parent.position = 0.*sys[0].position
            parent.velocity = 0.*sys[0].velocity

        massives = sys[sys.mass != (0. | units.kg)]
        if recenter:
            masses = massives.mass.value_in(units.kg)
            positions = massives.position.value_in(units.m)
            velocities = massives.velocity.value_in(units.ms)

            com = np.average(positions, weights=masses, axis=0)
            com_vel = np.average(velocities, weights=masses, axis=0)

            parent.position += com | units.m
            parent.velocity += com_vel | units.ms
            sys.position -= com | units.m
            sys.velocity -= com_vel | units.ms

        parent.mass = np.sum(massives.mass)

    def recenter_subsystems(self, max_workers) -> None:
        """
        Recenter subsystems

        Args:
            max_workers (int):  The number of cores to use
        """
        def calculate_com(parent_pos, parent_vel, system) -> tuple:
            """
            Calculate and shift system relative to center of mass

            Args:
                parent_pos (units.length):  Parent particle position
                parent_vel (units.velocity):  Parent particle velocity
                system (Particles):  The subsystem particle set

            Returns:
                tuple:  The shifted position and velocity
            """
            massives = system[system.mass != (0. | units.kg)]
            masses = massives.mass.value_in(units.kg)
            system_pos = massives.position.value_in(units.m)
            system_vel = massives.velocity.value_in(units.ms)

            com = np.average(system_pos, weights=masses, axis=0) | units.m
            com_vel = np.average(system_vel, weights=masses, axis=0) | units.ms

            system.position -= com 
            system.velocity -= com_vel
            parent_pos += com
            parent_vel += com_vel

            return parent_pos, parent_vel

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(calculate_com, parent.position, parent.velocity, sys): parent
                for parent, sys in self.collection_attributes.subsystems.values()
            }

            for future in as_completed(futures):
                parent = futures.pop(future)
                com_pos, com_vel = future.result()
                parent.position = com_pos
                parent.velocity = com_vel

    def remove_particles(self, parts: Particles) -> None:
        """
        Remove particles from particle set.

        Args:
            parts (Particles):  The particle to be removed
        """
        for p in parts:
            self.collection_attributes.subsystems.pop(p.key, None)

        ParticlesOverlay.remove_particles(self, parts)

    def all(self) -> Particles:
        """
        Get copy of complete particle set in galactocentric 
        or cluster frame of reference.

        Returns:
            Particles:  The complete particle set simulating
        """
        parts = self.copy()
        parts.syst_id = -1

        subsystems = self.collection_attributes.subsystems
        for system_id, (parent, sys) in enumerate(subsystems.values()):
            parts.remove_particle(parent)

            subsys = parts.add_particles(sys)
            subsys.position += parent.position
            subsys.velocity += parent.velocity
            subsys.syst_id = system_id + 1

        return parts