from amuse.lab import *
import numpy as np
p1 = read_set_from_file("latest_snapshot.amuse", 'amuse')
p2 = read_set_from_file("latest_snapshot2.amuse", 'amuse')

for mass in np.unique(p1.mass):
    a = p1[p1.mass==mass]
    b = p2[p2.mass==mass]
    dr = (a.position - b.position).lengths()
    dv = (a.velocity - b.velocity).lengths()
    print(f"Mass: {mass}, dR: {dr}, dV: {dv}")