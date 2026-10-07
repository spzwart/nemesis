from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from amuse.datamodel import Particles
from amuse.units import units

from src.globals import (
    MWG, 
    PARENT_RADIUS_COEFF,
    PARENT_RADIUS_MAX,
    PARENT_RADIUS_MIN
)


def connected_components_kdtree(
    child_set: Particles,
    threshold: units.length
) -> list:
    """
    Returns a list of particles within some threshold linking length.
    Args:
        child_set (Particles):     The particle set.
        threshold (units.length):  Linking length.
    Returns:
        list: Connected component particle sets.
    """
    def find(index: int) -> int:
        while nodes[index] != index:
            nodes[index] = nodes[nodes[index]]
            index = nodes[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            nodes[right_root] = left_root


    rad_m = threshold.value_in(units.m)
    pos_m = child_set.position.value_in(units.m)

    count = len(child_set)
    nodes = np.arange(count, dtype=np.intp)  # Each particle is its own node initially

    tree = cKDTree(pos_m)
    # Join every pair whose distance is within the linking length
    for left, right in tree.query_pairs(rad_m):
        union(int(left), int(right))

    # Resolve every particle to its final connected-component root
    labels = np.fromiter((find(i) for i in range(count)), dtype=np.intp)
    roots, first_indices = np.unique(labels, return_index=True)
    order = np.argsort(first_indices)
    return [child_set[labels == root] for root in roots[order]]   # Find all particles that share the same root (new child)


def galactic_frame(
    parent_set: Particles,
    dpos: units.length,
    dvel: units.velocity
) -> Particles:
    """
    Shift particle set to galactic frame.
    Args:
        parent_set (Particles):  The particle set.
        dpos (units.length):     Position in the galactic frame.
        dvel (units.velocity):   Velocity in the galactic frame.
    Returns:
        Particles: Particle set shifted to galactocentric coordinates
    """
    dx, dy, dz = dpos
    dvx, dvy, dvz = dvel

    parent_set.x += dx
    parent_set.y += dy
    parent_set.z += dz
    distance = np.sqrt(dx**2 + dy**2 + dz**2)

    parent_set.vx += dvx
    parent_set.vy += dvy + MWG.circular_velocity(distance)
    parent_set.vz += dvz

    return parent_set


def set_parent_radius(system_mass: units.mass) -> units.length:
    """
    Merging radius of parent systems. Based on system crossing time.
        - Too large → Poor angular momentum conservation
        and inaccurate center of mass.
        - Too small → Excessive computation due to frequent
        small timesteps and poor resolution of wider orbits.
    Args:
        system_mass (units.mass):  Total mass of the system
    Returns:
        units.length: Merging radius of the parent system
    """
    mass_msun = system_mass.value_in(units.MSun)
    radius = PARENT_RADIUS_COEFF * mass_msun**(1.0 / 3.0)
    if np.ndim(mass_msun) == 0:
        return min(max(radius, PARENT_RADIUS_MIN), PARENT_RADIUS_MAX)
    
    mask_over = radius > PARENT_RADIUS_MAX
    mask_under = radius < PARENT_RADIUS_MIN
    mask_zero = mass_msun <= 1e-15  # Asteroids given zero radius

    radius[mask_over] = PARENT_RADIUS_MAX
    radius[mask_under] = PARENT_RADIUS_MIN
    radius[mask_zero] = 0.0 | units.au
    return radius


def hill_radius(
    m1: units.mass,
    m2: units.mass,
    dr: units.length
) -> units.length:
    """
    Compute the Hill radius between two bodies.
    Args:
        m1 (units.mass):    Mass of the first body
        m2 (units.mass):    Mass of the second body
        dr (units.length):  Distance between the two bodies
    Returns:
        units.length: The Hill radius between the two bodies
    """
    return dr * (m1 / (3 * m2))**(1./3.)


def planet_radius(planet_mass: units.mass) -> units.radius:
    """
    Compute planet radius (arXiv:2311.12593).
    Args:
        planet_mass (units.mass):  Planet mass.
    Returns:
        units.mass:  Planet radius.
    """
    mass_in_earth = planet_mass.value_in(units.MEarth)

    if mass_in_earth < 7.8:
        return (1. | units.REarth)*(mass_in_earth)**0.41
    elif mass_in_earth < 125:
        return (0.55 | units.REarth)*(mass_in_earth)**0.65
    return (14.3 | units.REarth)*(mass_in_earth)**(-0.02)


def ZAMS_radius(star_mass) -> units.radius:
    """
    Define stellar radius at ZAMS.
    Args:
        star_mass (units.mass):  Mass of star.
    Returns:
        units.length:  The ZAMS radius of the star.
    """
    mass_in_sun = star_mass.value_in(units.MSun)
    mass_sq = (mass_in_sun)**2.

    numerator = mass_in_sun**1.25 * (0.1148 + 0.8604 * mass_sq)
    denominator = (0.04651 + mass_sq)
    r_zams = numerator / denominator

    return r_zams | units.RSun
