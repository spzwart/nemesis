import numpy as np
from amuse.datamodel import Particle, Particles, ParticlesOverlay


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
            parent.position += sys.center_of_mass()
            parent.velocity += sys.center_of_mass_velocity()
            sys.move_to_center()
            
        parent.mass = np.sum(sys.mass)
    
    def recenter_subsystems(self) -> None:
        """Recenter parents to children components"""
        for parent, sys in self.collection_attributes.subsystems.items():
            center_of_mass = sys.center_of_mass()
            center_of_mass_velocity = sys.center_of_mass_velocity()
            
            parent.position += center_of_mass
            parent.velocity += center_of_mass_velocity
            sys.position -= center_of_mass
            sys.velocity -= center_of_mass_velocity

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