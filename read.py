import numpy as np

from amuse.ext.orbital_elements import orbital_elements_from_binary as kepler_elements
from amuse.lab import Particles, Particle, write_set_to_file, read_set_from_file, units, constants

def crit_binding_energy(mass):
    return constants.G*mass/(vel_disp)**2

fname = "/home/erwanh/shuo_data/cluster_data/viscous_particles_plt_i00273.hdf5"
cluster_parti = read_set_from_file(fname, "hdf5")

vel_disp = 0 | (units.kms)**2
for parti_ in cluster_parti:
    dx = (parti_.vx-np.mean(cluster_parti.vx))**2
    dy = (parti_.vy-np.mean(cluster_parti.vy))**2
    dz = (parti_.vz-np.mean(cluster_parti.vz))**2
    vel_disp+=(dx+dy+dz)
vel_disp = np.sqrt(vel_disp/len(cluster_parti))

system = 0
proc_keys = [ ]
bin_set = Particles()
with open("init_bins.txt", "r") as f:
    text = f.readlines()[15:]
    line_no = 0
    for line_ in text:
        text = line_.split("=")
        if "Star1" in line_:
            if system>0:
                uq_keys, counts = np.unique(proc_keys, return_counts=True)
                bin_set.add_particle(new_particles)
                for key in [star_key1, star_key2]:
                    if key in uq_keys[counts>1]:
                        host = bin_set[bin_set.key_tracker==key]
                        min_syst = min(host.system_id)
                        for syst_ in host.system_id:
                            chd = bin_set[bin_set.system_id==syst_]
                            chd.system_id = min_syst
            line_no = 0
            new_particles = Particles(2)
        if line_no==0:
            system+=1
            star_key1 = np.float64(text[1].split("Star")[0])
            star_key2 = np.float64(text[-1][:-1])
            proc_keys = np.concatenate((proc_keys, [star_key1, star_key2]), axis=None)
            new_particles.system_id = system
            new_particles[0].key_tracker = star_key1
            new_particles[1].key_tracker = star_key2
            
            idx1 = abs(cluster_parti.key-star_key1).argmin()
            idx2 = abs(cluster_parti.key-star_key2).argmin()
            new_particles[0].mass = cluster_parti[idx1].mass
            new_particles[0].position = cluster_parti[idx1].position
            new_particles[0].velocity = cluster_parti[idx1].velocity
            new_particles[1].mass = cluster_parti[idx2].mass
            new_particles[1].position = cluster_parti[idx2].position
            new_particles[1].velocity = cluster_parti[idx2].velocity
        elif line_no==1:
            semi_major = float(text[1][:-4]) * (1|units.au)
            new_particles.semi = semi_major
            new_particles.Eb = -(constants.G*new_particles[0].mass*new_particles[1].mass)/(2*new_particles.semi)
        elif line_no==2:
            ecc = float(text[1])*(1|units.deg)
            new_particles.ecc = ecc
        line_no+=1

key_id = 1
nbin=0
for key in np.unique(bin_set.system_id):
    particles = bin_set[bin_set.system_id==key]
    particles.system_id = key_id
    key_id+=1

for key_ in np.unique(bin_set.system_id):
    particles = bin_set[bin_set.system_id==key_]
    if len(particles)>2:
        central_bin = particles[particles.Eb==min(particles.Eb)]
        rem = particles-central_bin
        bin_set-=rem
        
        host = Particle()
        host.mass = central_bin.mass.sum()
        host.position = central_bin.center_of_mass()
        host.velocity = central_bin.center_of_mass_velocity()
        
        semi_arr = [ ]
        ecc_arr = [ ]
        for parti_ in rem:
            filter_p = bin_set[bin_set.key_tracker==parti_.key_tracker]
            if len(filter_p)!=0:
                minimum_id = min(key_id, filter_p.system_id)
        for parti_ in rem:
            filter_p = bin_set[bin_set.key_tracker==parti_.key_tracker]
            if len(filter_p)==0:
                ke = kepler_elements((host, parti_), G=constants.G)
                semi_major = ke[2]
                ecc = ke[3]
                parti_.semi = semi_major
                parti_.ecc = ecc
                parti_.system_id = minimum_id
                bin_set.add_particle(parti_)
        sorted_semi = np.sort(semi_arr)

        key_id+=1
    else:
        nbin+=1

#All "bound" objects
systems, counts = np.unique(bin_set.system_id, return_counts=True)
total=0
nice=0
for syst_ in systems:
    total+=1
    system = bin_set[bin_set.system_id==syst_]
    host = system[system.Eb==min(system.Eb)]
    if min(host.semi)<crit_binding_energy(host.mass.min()):
        system.system_id *= -1

import matplotlib.pyplot as plt

plt.scatter(bin_set.x.value_in(units.pc), bin_set.y.value_in(units.pc), c = bin_set.system_id)
plt.show()

print("#Bin: ", nbin)
print("#Trip: ", len(systems[counts==3]))
print("#Hier: ", len(systems[counts>3]))
print(len(bin_set), bin_set.key_tracker)
STOP

