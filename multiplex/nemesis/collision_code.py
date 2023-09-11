from amuse.support import code
from amuse import datamodel
from amuse.units import nbody_system
from amuse.units import quantities

class CollisionCode(code.ObjectWithState):
    
    def __init__(self, gravity_code, G = nbody_system.G):
        code.ObjectWithState.__init__(self)
        
        self.code = gravity_code
        self.particles = datamodel.ParticlesOverlay(self.code.particles)
        self.particles.ancestors = None
        
        self.collision_detection = self.code.stopping_conditions.collision_detection
        self.collision_detection.enable()
        #self.channel_to_code =  self.particles.new_channel_to(self.code.particles)
        #self.channel_from_code =  self.code.particles.new_channel_to(self.particles)
        self.collided_particles = datamodel.Particles()
        self.stopping_conditions = CollisionCodeStoppingConditions()
        self.stopping_conditions.timeout_detection = self.code.stopping_conditions.timeout_detection
        self.stopping_conditions.out_of_box_detection = self.code.stopping_conditions.out_of_box_detection
        self.G = G
        self.evolve_model.async = self.evolve_model_async
        
        self.total_delta_kinetic_energy = quantities.zero
        self.total_delta_potential_energy = quantities.zero
        



    def evolve_model_async(self, endtime):
        #self.particles.synchronize_to(self.code.particles)
        #self.channel_to_code.copy()
        self.stopping_conditions.unset()
        
        def handle_result(function):
            result = function()
            #self.channel_from_code.copy()
            if self.collision_detection.is_set():
                collision = datamodel.Particles(particles = [self.collision_detection.particles(0)[0], self.collision_detection.particles(1)[0]])
                #collision = p.get_intersecting_subset_in(self.particles)
                self.handle_collision(collision)
                #self.particles.synchronize_to(self.code.particles)
                #self.channel_to_code.copy()
                if not self.stopping_conditions.is_set():
                    raise Exception("TODO handle when stopping condition is not enabled")
                    
            return result
       
        request = self.code.evolve_model.async(endtime)
        request.add_result_handler(handle_result)
        return request
                

    @code.state_method('RUN')
    def evolve_model(self, endtime):
        #self.particles.synchronize_to(self.code.particles)
        #self.channel_to_code.copy()
        self.stopping_conditions.unset()
        while True:
            
            if len(self.particles) == 1:
                self.evolve_single_particle_to(endtime)
            else: 
                self.code.evolve_model(endtime)
                #self.channel_from_code.copy()
            if self.collision_detection.is_set():
                p = datamodel.Particles(particles = [self.collision_detection.particles(0)[0], self.collision_detection.particles(1)[0]])
                collision = p#.get_intersecting_subset_in(self.particles)
                self.handle_collision(collision)
                #self.particles.synchronize_to(self.code.particles)
                #self.channel_to_code.copy()
                if self.stopping_conditions.is_set():
                    break
            else:
                break
                




    @code.state_transition('START', 'RUN')
    def commit_particles(self):
        #self.code.particles.add_particles(self.particles)
        #self.code.commit_particl
        pass



    def handle_collision(self, collision):
        p = datamodel.Particles()
        
        #collision = self.collided_particles.add_particles(collision)
        collision = p.add_particles(collision)
        k0 = self.kinetic_energy
        p0 = self.potential_energy
        self.particles.remove_particles(collision)
        total_mass = collision.mass.sum()
        heaviest = collision[0] if collision[0].mass > collision[1].mass else collision[1]
        merger_product = heaviest.copy()
        merger_product.mass = total_mass
        merger_product.position = (collision.position.sum(0)) / 2.0
        merger_product.velocity = ((collision.velocity * collision.mass.reshape((len(collision),1)) / total_mass)).sum(0)
        merger_product.radius = (collision.radius**3).sum()**(1.0/3.0)                                                                                                                                              
        merger_product.ancestors = collision         
                                                                                                                                 
        self.particles.add_particle(merger_product)
        
        delta_kinetic_energy = -0.5 * (collision.mass.prod()/total_mass) * (collision.velocity[1] - collision.velocity[0]).length_squared()
        delta_potential_energy =  self.G * collision.mass.prod() /  (collision.position[1] - collision.position[0]).length()
        k1 = self.kinetic_energy
        p1 = self.potential_energy
        print(p1 - p0, delta_potential_energy, ( p1 - p0)/ delta_potential_energy)
        print(k1 - k0, delta_kinetic_energy, ( k1 - k0)/ delta_kinetic_energy)
        self.total_delta_kinetic_energy += delta_kinetic_energy
        self.total_delta_potential_energy += p1-p0
        
        merger_product.delta_kinetic_energy = delta_kinetic_energy
        merger_product.delta_potential_energy = delta_potential_energy
        
        if self.stopping_conditions.collision_detection.is_enabled():
            self.stopping_conditions.collision_detection.set(
                collision[0].as_set(),
                collision[1].as_set(),
                merger_product.as_set(),
            )
        



    @property
    def model_time(self):
        return self.code.model_time




    @property
    def kinetic_energy(self):
        return self.code.kinetic_energy - self.total_delta_kinetic_energy






    @property
    def potential_energy(self):
        return self.code.potential_energy - self.total_delta_potential_energy






    @property
    def total_energy(self):
        return self.kinetic_energy + self.potential_energy





    def stop(self):
        self.code.stop()

    def new_subset(self):
        return self.code.new_subset()
    def stop_subset(self, subset):
        return self.code.stop_subset(subset)
    def get_kinetic_energy(self, *args):
        return self.code.get_kinetic_energy(*args)
    def get_potential_energy(self, *args):
        return self.code.get_potential_energy(*args)
    def get_time(self, *args):
        return self.code.get_time(*args)
    def evolve_single_particle_to(self, endtime):
        particle = self.particles[0]
        particle.position += particle.velocity * (endtime - self.model_time)
                



class CollisionCodeStoppingConditions(object):
    
    def __init__(self):
        self.collision_detection = code.StoppingCondition('collision_detection')
    
    def unset(self):
        self.collision_detection.unset()
        
    def disable(self):
        self.collision_detection.disable()
        
    def is_set(self):
        return (
            self.collision_detection.is_set()
        )
        


