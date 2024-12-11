from concurrent.futures import ThreadPoolExecutor
import numpy as np

from amuse.datamodel import Particle, Particles, ParticlesOverlay
from amuse.units import units


class HierarchicalParticles(ParticlesOverlay):
    """Class to make particle set"""
    def __init__(self, *args, **kwargs):
        ParticlesOverlay.__init__(self,*args,**kwargs)
        self.collection_attributes.subsystems = dict()

    def add_particles(self, parts) -> ParticlesOverlay:  
        """
        Add particles to particle set.
        
        Args:
            parts (Particle set):  The particle set to be added
        Returns:
            ParticlesOverlay:  The particle set
        """
        _parts=ParticlesOverlay.add_particles(self,parts)
        if hasattr(parts.collection_attributes, "subsystems"):
            for parent, sys in parts.collection_attributes.subsystems.items():
                self.collection_attributes.subsystems[parent.as_particle_in_set(self)] = sys
        return _parts
    
    def add_subsystem(self, sys, recenter=True) -> Particle:
        """
        Create a parent from particle subsytem
        
        Args:
            sys (Particle set):  The children particle set
            recenter (Boolean):  Flag to recenter the parent
        Returns:
            Particle:  The parent particle
        """
        if len(sys) == 1:
            return self.add_particles(sys)[0]
        
        p = Particle()
        self.assign_parent_attributes(
            sys, p, relative=False, 
            recenter=recenter
        )
        parent = self.add_particle(p)
        self.collection_attributes.subsystems[parent] = sys
        return parent

    def assign_parent_attributes(self, sys, parent, relative=True, recenter=True) -> None:
        """
        Create parent from subsystem attributes
        
        Args:
            sys (Particle set):  The children particle set
            parent (Particle):  The parent particle
            relative (Boolean):  Flag to assign relative attributes
            recenter (Boolean):  Flag to recenter the parent
        """
        if not (relative):
            parent.position = 0.*sys[0].position
            parent.velocity = 0.*sys[0].velocity
        
        if (recenter):
            massives = sys[sys.mass != (0. | units.kg)]
            parent.position += massives.center_of_mass()
            parent.velocity += massives.center_of_mass_velocity()
            sys.move_to_center()
            
        parent.mass = np.sum(sys.mass)

    def recenter_subsystems(self, max_workers) -> None:
        """
        Recenter the children subsystems
        
        Args:
            max_workers (int):  The number of cores to use
        """
        def calculate_com(parent_copy, system):
            """
            Calculate and shift system relative to center of mass
            
            Args:
                parent_copy (Particle):  The copied parent particle
                system (Particle set):  The children particle set
            Returns:
                Particle:  The updated (copied) parent particle
            """
            massives = system[system.mass != (0. | units.kg)]
            
            center_of_mass = massives.center_of_mass()
            center_of_mass_velocity = massives.center_of_mass_velocity()
            
            system.position -= center_of_mass
            system.velocity -= center_of_mass_velocity
            parent_copy.position += center_of_mass
            parent_copy.velocity += center_of_mass_velocity
            
            return parent_copy
                 
        updated_parents = dict()
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                parent: executor.submit(calculate_com, parent.copy(), sys)
                for parent, sys in self.collection_attributes.subsystems.items()
            }
            for parent, future in futures.items():
                updated_parents[parent] = future.result()
            
        for parent, updated_parent in updated_parents.items():
            parent.position = updated_parent.position
            parent.velocity = updated_parent.velocity

    def remove_particles(self, parts) -> None:
        """
        Remove particles from particle set.
        
        Args:
            parts (object):  The particle to be removed
        """
        for p in parts:
            self.collection_attributes.subsystems.pop(p, None)
        ParticlesOverlay.remove_particles(self, parts)
    
    def all(self) -> Particles:
        """
        Get copy of complete particle set in galactocentric 
        or cluster frame of reference.
        
        Returns:
            Particles:  The complete particle set simulating
        """
        parts = self.copy_to_memory()
        parts.syst_id = -1
        
        subsystems = self.collection_attributes.subsystems
        for system_id, (parent, sys) in enumerate(subsystems.items()):
            parts.remove_particle(parent)
            subsys = parts.add_particles(sys)
            subsys.sub_worker_radius = subsys.radius
            subsys.position += parent.position
            subsys.velocity += parent.velocity
            subsys.syst_id = system_id + 1
        
        return parts