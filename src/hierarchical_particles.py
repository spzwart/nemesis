import numpy as np
from amuse.datamodel import Particle, Particles, ParticlesOverlay
import threading
import queue


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
        def calculate_com(result_queue):
            try:
                parent_copy, system_copy = job_queue.get(timeout=1)
            except queue.Empty:
                raise ValueError("Error: No children system")

            center_of_mass = system_copy.center_of_mass()
            center_of_mass_velocity = system_copy.center_of_mass_velocity()
            system_copy.position -= center_of_mass
            system_copy.velocity -= center_of_mass_velocity
            parent_copy.position += center_of_mass
            parent_copy.velocity += center_of_mass_velocity

            result_queue.put((system_copy[0].syst_id, parent_copy))
            job_queue.task_done()

        import time as cpu_time
        t0 = cpu_time.time()

        result_queue = queue.Queue()
        job_queue = queue.Queue()
        threads = [ ]

        nworkers = 0
        for parent, sys in self.collection_attributes.subsystems.items():
            nworkers += 1
            sys.syst_id = nworkers
            job_queue.put((parent.copy(), sys))

        for worker in range(nworkers):
            th = threading.Thread(target=calculate_com, args=(result_queue, ))
            th.start()
            threads.append(th)

        for th in threads:
            th.join()
        
        changes = np.asarray([ ])
        while not result_queue.empty():
            system_id, parent_copy = result_queue.get()
            changes = np.concatenate((changes, parent_copy), axis=None)
         
        for parent in self.collection_attributes.subsystems.keys():
            updated_parent = changes[changes == parent]
            parent.position = updated_parent[0].position
            parent.velocity = updated_parent[0].velocity

        job_queue.queue.clear()
        del job_queue

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