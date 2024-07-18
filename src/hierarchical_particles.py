import numpy as np
from amuse.datamodel import Particle, ParticlesOverlay


class HierarchicalParticles(ParticlesOverlay):
    """Class to make particle set"""
    def __init__(self, *args, **kwargs):
        ParticlesOverlay.__init__(self,*args,**kwargs)
        self.collection_attributes.subsystems = dict()

    def add_particles(self, parts):  
        """Add particles to particle set."""
        _parts=ParticlesOverlay.add_particles(self,parts)
        if hasattr(parts.collection_attributes,"subsystems"):
            for parent,sys in list(parts.collection_attributes.subsystems.items()):
                self.collection_attributes.subsystems[parent.as_particle_in_set(self)] = sys
        return _parts
    
    def add_subsystem(self, sys, recenter=True):
        """Make a parent from particle subsytem"""
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

    def assign_parent_attributes(self, sys, parent, relative=True, recenter=True):
        """Create parent from subsystem attributes."""
        parent.mass = np.sum(sys.mass)
        if not (relative):
            parent.position = 0.*sys[0].position
            parent.velocity = 0.*sys[0].velocity
        if (recenter):
            parent.position += sys.center_of_mass()
            parent.velocity += sys.center_of_mass_velocity()
            sys.move_to_center()
    
    def assign_subsystem(self, sys, parent, relative=True, recenter=True):
        """Assign children to their parent particle."""
        self.assign_parent_attributes(sys, parent, relative, recenter)
        self.collection_attributes.subsystems[parent] = sys
    
    def recenter_subsystems(self):
        """Recenter parents to children components"""
        for parent, sys in list(self.collection_attributes.subsystems.items()): 
            sys.move_to_center()
            parent.position += sys.center_of_mass()
            parent.velocity += sys.center_of_mass_velocity()

    def remove_particles(self, parts):
        """Remove particles from particle set."""
        for p in parts:
            self.collection_attributes.subsystems.pop(p, None)
        ParticlesOverlay.remove_particles(self, parts)
    
    def all(self):
        """Get copy of complete particle set"""
        parts = self.copy_to_memory()
        for parent, sys in list(self.collection_attributes.subsystems.items()):
            parts.remove_particle(parent)
            subsys = parts.add_particles(sys)
            subsys.sub_worker_radius = subsys.radius
            subsys.position += parent.position
            subsys.velocity += parent.velocity
        return parts