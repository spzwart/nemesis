import numpy as np
from amuse.lab import constants

def find_gravity_at_point(particles, radius, x, y, z):
    """Find gravitational field at some point
    Input:
    particles:  Particle for which you want to find grav. field
    radius:  Radius of all other relevant particles
    x/y/z:  Position of all other relevant particles"""
    m1 = particles.mass
    result_ax = np.zeros_like(x)
    result_ay = np.zeros_like(y)
    result_az = np.zeros_like(z)

    dx = x[:, np.newaxis] - particles.x
    dy = y[:, np.newaxis] - particles.y
    dz = z[:, np.newaxis] - particles.z
    dr_squared = (dx**2 + dy**2 + dz**2 + radius[:, np.newaxis]**2)
    
    ax = -constants.G * (m1 * dx / dr_squared ** 1.5).sum(axis=1)
    ay = -constants.G * (m1 * dy / dr_squared ** 1.5).sum(axis=1)
    az = -constants.G * (m1 * dz / dr_squared ** 1.5).sum(axis=1)

    result_ax[:] = ax
    result_ay[:] = ay
    result_az[:] = az

    return result_ax, result_ay, result_az

class CorrectionFromCompoundParticle(object):
    def __init__(self, system, subsystems, worker_code_factory):
        """Correct force vector exerted by parents on all 
        other particles present by that of its children.
        Inputs:
        system:  Parent particles to correct force of
        subsystems:  Collection of subsystems present
        worker_code_factory:  Calculate potential field
        """
        self.system = system
        self.subsystems = subsystems
        self.worker_code_factory = worker_code_factory
    
    def get_gravity_at_point(self, radius, x, y, z):
        """Compute difference in gravitational acceleration felt by parents
        due to force exerted by parents hosting children, and their children.
        dF = \sum_j (\sum_i F_{i} - F_{j}) where j is parent and i is children of parent j
        """
        particles = self.system.copy_to_memory()
        acc_units = (particles.vx.unit**2/particles.x.unit)
        particles.ax = 0. | acc_units
        particles.ay = 0. | acc_units
        particles.az = 0. | acc_units

        for parent,sys in list(self.subsystems.items()):
            code = self.worker_code_factory()
            code.particles.add_particles(sys.copy_to_memory())
            code.particles.position += parent.position
            code.particles.velocity += parent.velocity

            # Potential from children particles
            parts = particles - parent
            ax_sub, ay_sub, az_sub = find_gravity_at_point(sys, 
                                                           0*parts.radius, 
                                                           parts.x, parts.y, 
                                                           parts.z
                                                           )

            code = self.worker_code_factory()
            code.particles.add_particle(parent)
            ax, ay, az = find_gravity_at_point(parent, 0*parts.radius, 
                                               parts.x, parts.y, parts.z
                                               )
            for i in range(len(parts)):
                parts[i].ax += ax_sub[i] - ax[i]
                parts[i].ay += ay_sub[i] - ay[i]
                parts[i].az += az_sub[i] - az[i]
                
        return particles.ax, particles.ay, particles.az

    def get_potential_at_point(self, radius, x, y, z):
        """Compute difference in potential field felt by parents due 
        to parents hosting children, and their children.
        """
        particles = self.system.copy_to_memory()
        particles.phi = 0. | (particles.vx.unit**2)
        for parent, sys in list(self.subsystems.items()): 
            code = self.worker_code_factory()
            code.particles.add_particles(sys.copy())
            code.particles.position += parent.position
            code.particles.velocity += parent.velocity

            # Potential from children particles
            parts = particles-parent
            phi = code.get_potential_at_point(0.*parts.radius,
                                              parts.x,parts.y,parts.z
                                              )
            parts.phi += phi
            
            # Potential from parent particle
            code = self.worker_code_factory()
            code.particles.add_particle(parent)
            phi = code.get_potential_at_point(0.*parts.radius,
                                              parts.x,parts.y,parts.z
                                              )
            parts.phi -= phi
        return particles.phi
  
  
class CorrectionForCompoundParticle(object):  
    def __init__(self, system, parent, worker_code_factory):
        """Correct force vector exerted by global particles on childrens"""
        self.system = system
        self.parent = parent
        self.worker_code_factory = worker_code_factory
    
    def get_gravity_at_point(self,radius,x,y,z):
        """Compute gravitational acceleration felt by children via 
        all other parents present in the simulation.
        dF_j = F_{i} - F_{j} where i is parent and j is children of parent i
        """
        parent = self.parent
        parts = self.system - parent
        instance = self.worker_code_factory()
        instance.particles.add_particles(parts)
        ax,ay,az = instance.get_gravity_at_point(0.*radius,
                                                parent.x+x,
                                                parent.y+y,
                                                parent.z+z
                                                )
        _ax,_ay,_az = instance.get_gravity_at_point([0.*parent.radius],
                                                    [parent.x],
                                                    [parent.y],
                                                    [parent.z]
                                                    )
        instance.cleanup_code()
        return (ax-_ax[0]), (ay-_ay[0]), (az-_az[0])


    def get_potential_at_point(self, radius, x, y, z):
        """Compute gravitational potential of children due to 
        all other parents present in the simulation.
        """
        parent = self.parent
        parts = self.system - parent
        instance = self.worker_code_factory()
        instance.particles.add_particles(parts)
        phi = instance.get_potential_at_point(0.*radius,
                                              parent.x+x, 
                                              parent.y+y, 
                                              parent.z+z
                                              )
        _phi = instance.get_potential_at_point([0.*parent.radius],
                                              [parent.x],
                                              [parent.y],
                                              [parent.z]
                                              )
        instance.cleanup_code()
        return (phi-_phi[0])
