class correction_from_compound_particles(object):
  def __init__(self, system, subsystems,worker_code_factory):
    self.system=system
    self.subsystems=subsystems
    self.worker_code_factory=worker_code_factory
    
  def get_gravity_at_point(self,radius,x,y,z):
    particles=self.system.copy_to_memory()
    particles.ax=0. | (particles.vx.unit**2/particles.x.unit)
    particles.ay=0. | (particles.vx.unit**2/particles.x.unit)
    particles.az=0. | (particles.vx.unit**2/particles.x.unit)
    for parent,sys in list(self.subsystems.items()): 
      code=self.worker_code_factory()
      code.particles.add_particles(sys.copy_to_memory())
      code.particles.position+=parent.position
      code.particles.velocity+=parent.velocity
      parts=particles-parent
      ax,ay,az=code.get_gravity_at_point(0.*parts.radius,parts.x,parts.y,parts.z)
      parts.ax+=ax
      parts.ay+=ay
      parts.az+=az
      code=self.worker_code_factory()
      code.particles.add_particle(parent)
      ax,ay,az=code.get_gravity_at_point(0.*parts.radius,parts.x,parts.y,parts.z)
      parts.ax-=ax
      parts.ay-=ay
      parts.az-=az
    return particles.ax,particles.ay,particles.az

  def get_potential_at_point(self,radius,x,y,z):
    particles=self.system.copy_to_memory()
    particles.phi=0. | (particles.vx.unit**2)
    for parent,sys in list(self.subsystems.items()): 
      code=self.worker_code_factory()
      code.particles.add_particles(sys.copy())
      code.particles.position+=parent.position
      code.particles.velocity+=parent.velocity
      parts=particles-parent
      phi=code.get_potential_at_point(0.*parts.radius,parts.x,parts.y,parts.z)
      parts.phi+=phi
      code=self.worker_code_factory()
      code.particles.add_particle(parent)
      phi=code.get_potential_at_point(0.*parts.radius,parts.x,parts.y,parts.z)
      parts.phi-=phi
    return particles.phi
  
  
class correction_for_compound_particles(object):  
  def __init__(self,system, parent, worker_code_factory):
    self.system=system
    self.parent=parent
    self.worker_code_factory=worker_code_factory
  
  def get_gravity_at_point(self,radius,x,y,z):
    parent=self.parent
    parts=self.system - parent
    instance=self.worker_code_factory()
    instance.particles.add_particles(parts)
    ax,ay,az=instance.get_gravity_at_point(0.*radius,parent.x+x,parent.y+y,parent.z+z)
    _ax,_ay,_az=instance.get_gravity_at_point([0.*parent.radius],[parent.x],[parent.y],[parent.z])
    instance.cleanup_code()
    return (ax-_ax[0]),(ay-_ay[0]),(az-_az[0])

  def get_potential_at_point(self,radius,x,y,z):
    parent=self.parent
    parts=self.system - parent
    instance=self.worker_code_factory()
    instance.particles.add_particles(parts)
    phi=instance.get_potential_at_point(0.*radius,parent.x+x,parent.y+y,parent.z+z)
    _phi=instance.get_potential_at_point([0.*parent.radius],[parent.x],[parent.y],[parent.z])
    instance.cleanup_code()
    return (phi-_phi[0])
  
  