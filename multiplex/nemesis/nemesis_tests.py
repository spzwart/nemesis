import threading
import numpy
from amuse.community.hermite0.interface import Hermite
from amuse.test.amusetest import TestWithMPI
from amuse import datamodel
from amuse.units import nbody_system
from amuse.units import quantities
from amuse.units import units

from .nemesis import CalculateFieldForParticles
from .nemesis import Nemesis
from . import nemesis

from . import collision_code


class TestNemesis(TestWithMPI):
    
    def test01(self):
        binary1 = new_binary( 1 | nbody_system.mass, 1 | nbody_system.mass, 1 | nbody_system.length)
        binary2 = new_binary( 1 | nbody_system.mass, 1 | nbody_system.mass, 1 | nbody_system.length)
        binary3 = new_binary( 2 | nbody_system.mass, 2 | nbody_system.mass, 100 | nbody_system.length)
        
        binary3[0].subsystem = binary1
        binary3[1].subsystem = binary2
        binary3.dt = 0.5 | nbody_system.time
        
        status = {'number_of_codes': 0}
        def make_code(particles):
            status['number_of_codes'] += 1
            result = Hermite()
            result.particles.add_particles(particles)
            return result
        
        def make_gravity_code(particles):
            return CalculateFieldForParticles(particles, G = nbody_system.G)
        
        x = Nemesis(Hermite(), make_code, make_gravity_code, 0.1 | nbody_system.time, G = nbody_system.G, calculate_radius = lambda x : 0.1 | nbody_system.length)
        x.particles.add_particles(binary3)
        x.commit_particles()
        self.assertEqual(status['number_of_codes'], 2)
        x.evolve_model(1 | nbody_system.time)
        self.assertAlmostRelativeEquals((x.particles[0].position - x.particles[1].position).length(), 100.0 | nbody_system.length, 5)
        
        ref_code = Hermite()
        ref_code.particles.add_particles(binary3)
        ref_code.evolve_model(x.model_time)
        self.assertAlmostRelativeEquals(x.particles.position, ref_code.particles.position, 5)
        ref_code.stop()
        x.stop()


    def test02(self):
        binary1 = new_binary( 1 | nbody_system.mass, 1 | nbody_system.mass, 1 | nbody_system.length)
        binary3 = new_binary( 2 | nbody_system.mass, 1 | nbody_system.mass, 4 | nbody_system.length)
        
        binary3[0].subsystem = binary1
        binary3[1].subsystem = None
        binary3.dt = 0.5 | nbody_system.time
        
        status = {'number_of_codes': 0}
        def make_code(particles):
            status['number_of_codes'] += 1
            result = Hermite()
            result.particles.add_particles(particles)
            return result
        
        def make_gravity_code(particles):
            return CalculateFieldForParticles(particles, G = nbody_system.G)
        
        code = Hermite()
        code.parameters.end_time_accuracy_factor = 0.0
        x = Nemesis(code, make_code, make_gravity_code, 0.1 | nbody_system.time, G = nbody_system.G, calculate_radius = lambda x : 0.1 | nbody_system.length)
        x.timestep = 0.5 | nbody_system.time
        x.particles.add_particles(binary3)
        x.commit_particles()
        self.assertEqual(status['number_of_codes'], 1)
        x.evolve_model(3 | nbody_system.time)
        self.assertAlmostRelativeEquals((x.particles[0].position - x.particles[1].position).length(), 4.0 | nbody_system.length, 1)
        
        ref_code = Hermite()
        ref_code.parameters.end_time_accuracy_factor = 0.0
        ref_code.particles.add_particles(binary3)
        ref_code.evolve_model(x.model_time)
        self.assertAlmostRelativeEquals(x.model_time, ref_code.model_time)
        self.assertAlmostRelativeEquals(x.particles.position, ref_code.particles.position, 2)
        ref_code.stop()
        x.stop()


    def test03(self):
        binary1 = new_binary( 1 | nbody_system.mass, 1 | nbody_system.mass, 0.5 | nbody_system.length)
        binary3 = new_binary( 2 | nbody_system.mass, 1 | nbody_system.mass, 4 | nbody_system.length)
        
        binary3[0].subsystem = binary1
        binary3[1].subsystem = None
        binary3.dt = 0.5 | nbody_system.time
        
        status = {'number_of_codes': 0}
        def make_code(particles):
            status['number_of_codes'] += 1
            result = Hermite()
            result.parameters.end_time_accuracy_factor = 0.0
            result.particles.add_particles(particles)
            return result
        
        def make_gravity_code(particles):
            return CalculateFieldForParticles(particles, G = nbody_system.G)

        def all_parts(p):
            result = datamodel.Particles()
            result.add_particles(p[0].subsystem)
            result.position += p[0].position
            result.velocity += p[0].velocity
            result.add_particle(p[1])
            return result
        
        code = Hermite()
        code.parameters.end_time_accuracy_factor = 0.0
        x = Nemesis(code, make_code, make_gravity_code, 0.1 | nbody_system.time, G = nbody_system.G, calculate_radius = lambda x : 0.1 | nbody_system.length)
        x.timestep = 0.5 | nbody_system.time
        x.particles.add_particles(binary3)
        x.commit_particles()
        
        ref_code = Hermite()
        ref_code.parameters.end_time_accuracy_factor = 0.0
        ref_code.particles.add_particles(all_parts(binary3))
        
        
        self.assertEqual(status['number_of_codes'], 1)
        x.evolve_model(3 | nbody_system.time)
        self.assertAlmostRelativeEquals((x.particles[0].position - x.particles[1].position).length(), 4.0 | nbody_system.length, 2)
        
        
        ref_code.evolve_model(x.model_time)
        self.assertAlmostRelativeEquals(x.model_time, ref_code.model_time)
        
        
        self.assertAlmostRelativeEquals(all_parts(x.particles).position, ref_code.particles.position, 1)
        ref_code.stop()
        x.stop()


    def test04(self):
        binary1 = new_binary( 1 | nbody_system.mass, 1 | nbody_system.mass, 0.5 | nbody_system.length)
        binary3 = new_binary( 2 | nbody_system.mass, 1 | nbody_system.mass, 4 | nbody_system.length)
        
        binary3[0].subsystem = binary1
        binary3[1].subsystem = None
        binary3.dt = 0.5 | nbody_system.time
        
        status = {'number_of_codes': 0}
        def make_code(particles):
            status['number_of_codes'] += 1
            result = Hermite()
            result.parameters.end_time_accuracy_factor = 0.0
            result.particles.add_particles(particles)
            return result
        
        def make_gravity_code(particles):
            return CalculateFieldForParticles(particles, G = nbody_system.G)

        def all_parts(p):
            result = datamodel.Particles()
            result.add_particles(p[0].subsystem)
            result.position += p[0].position
            result.velocity += p[0].velocity
            result.add_particle(p[1])
            return result
        
        code = Hermite()
        code.parameters.end_time_accuracy_factor = 0.0
        x = Nemesis(code, make_code, make_gravity_code, 0.1 | nbody_system.time, G = nbody_system.G, calculate_radius = lambda x : 0.1 | nbody_system.length)
        x.timestep = 0.5 | nbody_system.time
        x.particles.add_particles(binary3)
        x.commit_particles()
        
        ref_code = Hermite()
        ref_code.parameters.end_time_accuracy_factor = 0.0
        ref_code.particles.add_particles(all_parts(binary3))
        
        
        self.assertEqual(status['number_of_codes'], 1)
        x.evolve_model(3 | nbody_system.time)
        self.assertAlmostRelativeEquals((x.particles[0].position - x.particles[1].position).length(), 4.0 | nbody_system.length, 2)
        
        
        ref_code.evolve_model(x.model_time)
        self.assertAlmostRelativeEquals(x.model_time, ref_code.model_time)
        
        
        self.assertAlmostRelativeEquals(all_parts(x.particles).position, ref_code.particles.position, 1)
        
        x.particles[0].subsystem = None
        x.recommit_particles()
        x.evolve_model(4 | nbody_system.time)
        self.assertAlmostRelativeEquals((x.particles[0].position - x.particles[1].position).length(), 4.0 | nbody_system.length, 2)
        ref_code.stop()
        x.stop()


    def test05(self):
        
        
        status = {'number_of_codes': 0}
        def make_code(particles):
            status['number_of_codes'] += 1
            result = Hermite()
            result.parameters.end_time_accuracy_factor = 0.0
            result.particles.add_particles(particles)
            return result
        
        def make_gravity_code(particles):
            return CalculateFieldForParticles(particles, G = nbody_system.G)

        def all_parts(p):
            result = datamodel.Particles()
            result.add_particles(p[0].subsystem)
            result.position += p[0].position
            result.velocity += p[0].velocity
            result.add_particle(p[1])
            return result
        
        stars = datamodel.Particles(3)
        stars.mass = 1 | nbody_system.mass
        stars.position = [
            [0,0,0],
            [1.2, 0, 0],
            [0, 0, 5]
        ]|nbody_system.length
        stars.velocity = [
            [0,0,0],
            [0,0.1, 0],
            [1, 0, 0],
        ]|nbody_system.speed
        stars.radius = 0.5 | nbody_system.length
        stars.dt =  0.1 | nbody_system.time
        print(stars)

        code = Hermite()
        code.parameters.end_time_accuracy_factor = 0.0
        x = Nemesis(code, make_code, make_gravity_code, timestep = 0.1 | nbody_system.time, G = nbody_system.G, calculate_radius = lambda x : 0.5 | nbody_system.length)
        x.particles.add_particles(stars)
        x.commit_particles()
        
        self.assertEqual(status['number_of_codes'], 0)
        x.evolve_model(1 | nbody_system.time)
        self.assertEqual(status['number_of_codes'], 1)
        
        x.stop()


    def test06(self):
        return 1
        all_subsystems = []
        class MySubSystem(nemesis.Subsystem):
            
            def __init__(self, *args, **kwargs):
                nemesis.Subsystem.__init__(self, *args, **kwargs)
                all_subsystems.append(self)
                
            def get_events(self):
                if not hasattr(self, 'events'):
                    self.events = []
                return self.events
            
            def kick(self, system, dt, min_dt, reuse_kick = False):
                self.get_events().append((self.model_time, 'kick', dt))
                nemesis.Subsystem.kick(self, system, dt, min_dt)
                self.next_dt = self.parent.dt
                
            def drift(self, end_time):
                self.get_events().append((self.model_time, 'drift', end_time - self.model_time))
                nemesis.Subsystem.drift(self, end_time)
             
            def drift_async(self, end_time):
                self.get_events().append((self.model_time, 'drift', end_time - self.model_time))
                nemesis.Subsystem.drift(self, end_time)
                return None
            
            def calculate_dt_from_cluster_acceleration(self, ax, ay, az, min_dt):
                return self.dt
            
        class Code(object):
            def __init__(self):
                self.model_time = 0 | nbody_system.time
                
            def evolve_model(self, end_time):
                self.model_time = end_time
            
            def stop(self):
                pass
            
        all_codes = []
        def make_code(particles):
            all_codes.append(code)
            result = Code()
            result.particles = particles.copy()
            return result
        
        def make_gravity_code(particles):
            return CalculateFieldForParticles(particles, G = nbody_system.G)

        def all_parts(p):
            result = datamodel.Particles()
            result.add_particles(p[0].subsystem)
            result.position += p[0].position
            result.velocity += p[0].velocity
            result.add_particle(p[1])
            return result
        
        stars = datamodel.Particles(3)
        stars.mass = 1 | nbody_system.mass
        stars.position = [
            [0,0,0],
            [1, 0, 0],
            [2, 0, 0]]|nbody_system.length
        stars.velocity = [0,0,0] | nbody_system.speed
        stars.radius = 0.5 | nbody_system.length
        stars.dt = [1, 2, 4] | nbody_system.time
        stars[0].subsystem = datamodel.Particles(2)
        stars[0].subsystem.mass = 0.5 | nbody_system.mass
        stars[0].subsystem.position = [[0,-0.1,0],[0,0.1,0]] | nbody_system.length
        stars[0].subsystem.velocity = [0,0,0] | nbody_system.speed
        stars[1].subsystem = datamodel.Particles(4)
        stars[1].subsystem.mass = 0.25 | nbody_system.mass
        stars[1].subsystem.position = [[1,-0.1,0],[1,0.1,0], [1,0,-0.1],[1, 0, 0.1]] | nbody_system.length
        stars[1].subsystem.velocity = [0,0,0] | nbody_system.speed
        stars[2].subsystem = datamodel.Particles(2)
        stars[2].subsystem.mass = 0.5 | nbody_system.mass
        stars[2].subsystem.position = [[1,-0.1,0],[1,0.1,0]] | nbody_system.length
        stars[2].subsystem.velocity = [0,0,0] | nbody_system.speed
        

        code = Code()
        code.particles = datamodel.Particles()
        x = Nemesis(code, make_code, make_gravity_code, 0.1 | nbody_system.time, G = nbody_system.G, subsystem_factory = MySubSystem, must_kick_subsystems = False)
        x.timestep = 1 | nbody_system.time
        x.particles.add_particles(stars)
        self.assertEqual(len(all_codes), 0)
        x.commit_particles()
        self.assertEqual(len(all_codes), 3)
        self.assertEqual(len(all_subsystems), 3)
        x.evolve_model(4 | nbody_system.time)
        for y in all_subsystems[2].events:
            print(y)
        self.assertEqual(len(all_subsystems[0].events), 4 * 3)
        self.assertEqual(len(all_subsystems[1].events), 4 * 3)
        self.assertEqual(len(all_subsystems[2].events), 2 * 3)
        x.stop()



    def test07(self):
        particles = datamodel.Particles(4)
        particles.mass = 1 | nbody_system.mass
        particles.position = [
            [0,0,0], 
            [1,0,0], 
            [0,1,0], 
            [1,1,0] 
         ] | nbody_system.length
        particles.velocity = [0,0,0] | nbody_system.speed
        particles.radius = 0.1 | nbody_system.length
        
        status = {'number_of_codes': 0}
        def make_code(particles):
            status['number_of_codes'] += 1
            result = Hermite()
            result.particles.add_particles(particles)
            return result
        
        def make_gravity_code(particles):
            return CalculateFieldForParticles(particles, G = nbody_system.G)
        
        def dr(ipart,jpart, *args):
            dx=ipart.x-jpart.x                                                                                                                                                                                                                                                                                                                                                                                  
            dy=ipart.y-jpart.y
            dz=ipart.z-jpart.z
            dr2=dx**2+dy**2+dz**2
            return dr2
        x = Nemesis(Hermite(), make_code, make_gravity_code, 0.1 | nbody_system.time, G = nbody_system.G, calculate_radius = lambda x : 0.1 | nbody_system.length)
        x.timestep = 0.1 | nbody_system.time
        x.distfunc = dr
        x.threshold = (0.5 | nbody_system.length)**2
        x.particles.add_particles(particles)
        x.commit_particles()
        print(x.kinetic_energy, x.potential_energy)
        e0 = x.kinetic_energy+ x.potential_energy
        x.handle_collision_in_cluster(x.particles[0:2], x.model_time, x.model_time)
        
        e1 = x.kinetic_energy+ x.potential_energy
        print(x.kinetic_energy , x.potential_energy)
        print((e1-e0) / e0)
        self.assertAlmostRelativeEquals(e0, e1)
        self.assertTrue(abs((e1-e0) / e0) < 1e-15)
        
        self.assertEqual(len(x.particles), 3)
        self.assertEqual(len(x.subsystems), 1)
        x.split_subsystems(x.subsystems)
        self.assertEqual(len(x.particles), 4)
        self.assertEqual(len(x.subsystems), 0)
        e2 = x.kinetic_energy+ x.potential_energy
        self.assertAlmostRelativeEquals(e0, e2)
        self.assertTrue(abs((e2-e0) / e0) < 1e-15)
        print(x.kinetic_energy , x.potential_energy)
        
        x.stop()


    def test08(self):
        particles = datamodel.Particles(4)
        particles.mass = 1 | nbody_system.mass
        particles.position = [
            [0,0,0], 
            [1,0,0], 
            [0,1,0], 
            [1,1,0] 
         ] | nbody_system.length
        particles.velocity = [
            [1,0,0], 
            [1,0,0], 
            [0,1,0], 
            [0,1,0] 
         ] | nbody_system.speed
        particles.radius = 0.1 | nbody_system.length
        
        status = {'number_of_codes': 0}
        def make_code(particles):
            status['number_of_codes'] += 1
            result = Hermite()
            result.particles.add_particles(particles)
            return result
        
        def make_gravity_code(particles):
            return CalculateFieldForParticles(particles, G = nbody_system.G)
        
        def dr(ipart,jpart, *args):
            dx=ipart.x-jpart.x                                                                                                                                                                                                                                                                                                                                                                                          
            dy=ipart.y-jpart.y
            dz=ipart.z-jpart.z
            dr2=dx**2+dy**2+dz**2
            return dr2
        x = Nemesis(Hermite(), make_code, make_gravity_code, 0.1 | nbody_system.time, G = nbody_system.G, calculate_radius = lambda x : 0.1 | nbody_system.length)
        x.timestep = 0.1 | nbody_system.time
        x.distfunc = dr
        x.threshold = (0.5 | nbody_system.length)**2
        x.particles.add_particles(particles)
        x.commit_particles()
        print(x.kinetic_energy, x.potential_energy)
        e0 = x.kinetic_energy+ x.potential_energy
        x.handle_collision_in_cluster(x.particles[0:2], x.model_time, x.model_time)
        
        e1 = x.kinetic_energy+ x.potential_energy
        print((e1-e0) / e0)
        print(x.kinetic_energy , x.potential_energy)
        self.assertAlmostRelativeEquals(e0, e1)
        self.assertTrue(abs((e1-e0) / e0) < 1e-15)
        
        self.assertEqual(len(x.particles), 3)
        self.assertEqual(len(x.subsystems), 1)
        x.split_subsystems(x.subsystems)
        self.assertEqual(len(x.particles), 4)
        self.assertEqual(len(x.subsystems), 0)
        e2 = x.kinetic_energy+ x.potential_energy
        print(x.kinetic_energy , x.potential_energy)

        self.assertAlmostRelativeEquals(e0, e2)
        self.assertTrue(abs((e2-e0) / e0) < 1e-15)
        
        x.stop()


    def test09(self):
        particles = datamodel.Particles(4)
        particles.mass = 1 | nbody_system.mass
        particles.position = [
            [0,0,0], 
            [0.2,0.2,0], 
            [0.5,0.5,0], 
            [1,1,0] 
         ] | nbody_system.length
        particles.velocity = [
            [0,0,0], 
            [0,-0.1,0], 
            [0,-0.1,0], 
            [0,-0.1,0] 
         ] | nbody_system.speed
        particles.radius = 0.1 | nbody_system.length
        
        status = {'number_of_codes': 0}
        def make_code(particles):
            status['number_of_codes'] += 1
            result = Hermite()
            result.particles.add_particles(particles)
            return result
        
        def make_gravity_code(particles):
            return CalculateFieldForParticles(particles, G = nbody_system.G)
        
        def dr(ipart,jpart, *args):
            dx=ipart.x-jpart.x                                                                                                                                                                                                                                                                                                                                                                                                      
            dy=ipart.y-jpart.y
            dz=ipart.z-jpart.z
            dr2=dx**2+dy**2+dz**2
            return dr2
        x = Nemesis(Hermite(), make_code, make_gravity_code, 0.1 | nbody_system.time, G = nbody_system.G, calculate_radius = lambda x : 0.1 | nbody_system.length)
        x.timestep = 0.1 | nbody_system.time
        x.distfunc = dr
        x.threshold = (0.05 | nbody_system.length)**2
        x.particles.add_particles(particles)
        x.commit_particles()
        print(x.kinetic_energy, x.potential_energy)
        e0 = x.kinetic_energy+ x.potential_energy
        x.handle_collision_in_cluster(x.particles[0:2], x.model_time, x.model_time)
        
        e1 = x.kinetic_energy+ x.potential_energy
        print((e1-e0) / e0)
        print(x.kinetic_energy , x.potential_energy)
        self.assertAlmostRelativeEquals(e0, e1)
        self.assertTrue(abs((e1-e0) / e0) < 1e-15)
        
        self.assertEqual(len(x.particles), 3)
        self.assertEqual(len(x.subsystems), 1)
        x.split_subsystems(x.subsystems)
        self.assertEqual(len(x.particles), 4)
        self.assertEqual(len(x.subsystems), 0)
        e2 = x.kinetic_energy+ x.potential_energy
        print(x.kinetic_energy , x.potential_energy)

        self.assertAlmostRelativeEquals(e0, e2)
        self.assertTrue(abs((e2-e0) / e0) < 1e-15)
        
        x.stop()



    def test10(self):

        binary1 = new_binary( 1 | nbody_system.mass, 1 | nbody_system.mass, 4 | nbody_system.length)
        binary3 = new_binary( 2 | nbody_system.mass, 1 | nbody_system.mass, 10 | nbody_system.length)
        
        binary3[0].subsystem = binary1
        binary3[1].subsystem = None
        
        
        code = Hermite(redirection="null")
        code.parameters.end_time_accuracy_factor = 0.0
        def make_code(particles):
            code = Hermite(redirection="null")
            
            code.parameters.stopping_conditions_timeout = 120 | units.s                                                                                                                                 
            if code.stopping_conditions.timeout_detection.is_supported():
                code.stopping_conditions.timeout_detection.enable()
                #print "with timeout detection enabled"
            result = collision_code.CollisionCode(code, G = nbody_system.G)
            result.particles.add_particles(particles)
            result.commit_particles()
            return result
        

        def make_gravity_code(particles):
            return nemesis.CalculateFieldForParticles(particles, G = nbody_system.G)

        
        
        code = Hermite(redirection="null")
        code.parameters.end_time_accuracy_factor = 0.0
        y = nemesis.Nemesis(code, make_code,  make_gravity_code, 0.1 | nbody_system.time, G = nbody_system.G)
        y.particles.add_particles(binary3)
        y.commit_particles()
        self.assertEqual(len(y.particles), 2)
        y.evolve_model(1 | nbody_system.time)
        self.assertEqual(len(y.particles), 3)
        y.stop()




    def test11(self):

        binary1 = new_binary( 1 | nbody_system.mass, 1 | nbody_system.mass, 4 | nbody_system.length)
        binary3 = new_binary( 2 | nbody_system.mass, 1 | nbody_system.mass, 10 | nbody_system.length)
        binary4 = new_binary( 0.5 | nbody_system.mass, 0.5 | nbody_system.mass, 1 | nbody_system.length)
        
        binary3[0].subsystem = binary1
        binary3[1].subsystem = None
        
        
        code = Hermite(redirection="null")
        code.parameters.end_time_accuracy_factor = 0.0
        def make_code(particles):
            code = Hermite(redirection="null")
            
            code.parameters.stopping_conditions_timeout = 120 | units.s                                                                                                                                                 
            if code.stopping_conditions.timeout_detection.is_supported():
                code.stopping_conditions.timeout_detection.enable()
                #print "with timeout detection enabled"
            result = collision_code.CollisionCode(code, G = nbody_system.G)
            result.particles.add_particles(particles)
            result.commit_particles()
            return result
        

        def make_gravity_code(particles):
            return nemesis.CalculateFieldForParticles(particles, G = nbody_system.G)

        
        
        code = Hermite(redirection="null")
        code.parameters.end_time_accuracy_factor = 0.0
        y = nemesis.Nemesis(code, make_code,  make_gravity_code, 0.1 | nbody_system.time, G = nbody_system.G)
        y.particles.add_particles(binary3)
        y.commit_particles()
        
        binary3[1].subsystem = binary4
        binary3.synchronize_to(code.particles)
        synchronize_subsystems(binary3, y.particles)
        self.assertEqual(len(y.particles[1].subsystem), 2)


    def test12(self):

        binary1 = new_binary( 1 | nbody_system.mass, 1 | nbody_system.mass, 0.1 | nbody_system.length)
        binary2 = new_binary( 1 | nbody_system.mass, 1 | nbody_system.mass, 0.1 | nbody_system.length)
        binary3 = new_binary( 2 | nbody_system.mass, 2 | nbody_system.mass, 1 | nbody_system.length, 0.5)
        binary3[0].subsystem = binary1
        binary3[1].subsystem = binary2
          
        
        
        code = Hermite(redirection="null")
        code.parameters.end_time_accuracy_factor = 0.0
        def make_code(particles):
            code = Hermite(redirection="null")
            
            code.parameters.stopping_conditions_timeout = 120 | units.s                                                                                                                                                     
            if code.stopping_conditions.timeout_detection.is_supported():
                code.stopping_conditions.timeout_detection.enable()
                #print "with timeout detection enabled"
            result = collision_code.CollisionCode(code, G = nbody_system.G)
            result.particles.add_particles(particles)
            result.commit_particles()
            return result
        

        def make_gravity_code(particles):
            return nemesis.CalculateFieldForParticles(particles, G = nbody_system.G)

        
        
        code = Hermite(redirection="null")
        code.parameters.end_time_accuracy_factor = 0.0
        y = nemesis.Nemesis(code, make_code,  make_gravity_code, 0.1 | nbody_system.time, G = nbody_system.G)
        y.particles.add_particles(binary3)
        y.commit_particles()
        self.assertEqual(len(y.subsystems), 2)
        y.particles.remove_particle(y.particles[1])
        y.recommit_particles()
        self.assertEqual(len(y.subsystems), 1)


def new_binary(mass1, mass2, semi_major_axis,
               eccentricity = 0, keyoffset = -1, radius = 0.1 | nbody_system.length,G = nbody_system.G):
    total_mass = mass1 + mass2
    mu = G * total_mass
    
    velocity_perihelion = numpy.sqrt( mu / semi_major_axis  * ((1.0 + eccentricity)/(1.0 - eccentricity)))
    radius_perihelion = semi_major_axis * (1.0 - eccentricity)
    
    mass_fraction_particle_1 = mass1 / (total_mass)
    
    if keyoffset > 0:
        binary = datamodel.Particles(keys=list(range(keyoffset, keyoffset+2)))
    else:
        binary = datamodel.Particles(2)
        
    binary[0].mass = mass1
    binary[1].mass = mass2
    binary[0].position = ((1.0 - mass_fraction_particle_1) * radius_perihelion * [1.0,0.0,0.0])
    binary[1].position = -(mass_fraction_particle_1 * radius_perihelion * [1.0,0.0,0.0])
    
    binary[0].velocity = ((1.0 - mass_fraction_particle_1) * velocity_perihelion * [0.0,1.0,0.0])
    binary[1].velocity = -(mass_fraction_particle_1 * velocity_perihelion * [0.0,1.0,0.0])
    binary.radius = radius
    return binary
    

def synchronize_subsystems(from_set, to_set):
    from_with_subsystem = from_set[~numpy.equal(from_set.subsystem, None)]
    to_without_subsystem = to_set[numpy.equal(to_set.subsystem, None)]
    subsystems_not_copied_yet = to_without_subsystem.get_intersecting_subset_in(from_with_subsystem)
    subsystems_not_copied_yet.new_channel_to(to_without_subsystem).copy_attribute("subsystem")

if __name__ == "__main__":
    x = multiplexing_gd.MultiplexingGravitationalDynamicsCode()
    print(x.particles)

