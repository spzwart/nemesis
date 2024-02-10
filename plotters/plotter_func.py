import numpy as np
from amuse.lab import units

def med_iqr(array):
    """Track median and IQR of various binary pops."""
    median = [np.median(i) for i in zip(*array)]
    IQRL = [np.percentile(i, 25) for i in zip(*array)]
    IQRH = [np.percentile(i, 75) for i in zip(*array)]

    return median, IQRL, IQRH

def moving_average(array, smoothing):
    """
    Conduct running average of some variable
    
    Inputs:
    array:     Array hosting values
    smoothing: Number of elements to average over
    """

    value = np.cumsum(array, dtype=float)
    value[smoothing:] = value[smoothing:] - value[:-smoothing]

    return value[smoothing-1:]/smoothing


def vdisp(pset):
    """Compute velocity dispersion
    
        Input:
        pset:  Particle set (contains particle < some defined rij)
    """
    SMBH = pset[pset.type=="smbh"]
    SMBH.vdisp = 0 | units.kms
    for p_ in pset[pset.type!="smbh"]:
        dist = (p_.position-SMBH.position).length()
        v0 = 200*(SMBH.mass/(10**8.29 | units.MSun))**(1/5.12) | units.kms #arXiv:1112.1078
        if dist>=(1 | units.pc):
            p_.vdisp = v0
        else:
            p_.vdisp = v0*np.sqrt((1|units.pc)/(dist))
            
    return pset

def hierarchical_detect(pset, fname):
    None