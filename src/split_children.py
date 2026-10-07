""""
Possible Room for Improvements:
    1. Parallelise algorithm. Dictionary look ups and
       current recycling of old code complicates this.
    2. Remove hand-wavey dependence (Rpar and SPLIT_PARAMS)
       from parent radius. Ideally, this would be some fraction
       of Hill radius with a similar calculation to _get_dr_threshold,
       but preliminary tests show a pure-physics approach is too
       agressive and some hybrid model was needed. For future work.
            - Machine learning adaptable parent radius?
            - If you wish to see tested implementation of Hill radius
              please contact Erwan Hochart.
                *  In the edge-case where asteroids get ejected and the only new parent
                   is also rogue, the current method will not flag any splits. This will
                   be dealt with in the next drift, but could be optimised by checking
                   for this scenario here, although it requires further complexity in
                   the logic.
                *  Since the number of asteroids per system is typically
                   small, grid-based calculations can be used to check for splits.
                   For a large number of asteroids (>10^4), this will be memory-heavy
                   and the logic will need to be modified.
"""
from __future__ import annotations

import numpy as np

from amuse.datamodel import Particles
from amuse.units import units

from src.environment_functions import connected_components_kdtree, set_parent_radius
from src.globals import SPLIT_PARAM, PARENT_RADIUS_MAX


def split_subcodes(nem_class) -> None:
    """
    Check for any isolated children
    Args:
        nem_class (object):  Nemesis instance.
    """
    if nem_class._verbose:
        print("...Checking Splits...", flush=True)

    new_rogue = Particles()
    for parent_key, (parent, subsys) in list(nem_class.children.items()):
        components = connected_components_kdtree(
            child_set=subsys,
            threshold=SPLIT_PARAM / 2. * parent.radius
            )
        if len(components) <= 1:
            continue

        rework_code = False
        par_vel = parent.velocity
        par_pos = parent.position

        pid = nem_class._pid_workers.pop(parent_key)
        nem_class.resume_workers(pid)
        nem_class.particles.remove_particle(parent)

        code = nem_class.subcodes.pop(parent_key)
        offset = nem_class._time_offsets.pop(code)
        nem_class._child_channels.pop(parent_key)
        cpu_time = nem_class._cpu_time.pop(parent_key)

        for c in components:
            sys = c.as_set()
            sys.position += par_pos
            sys.velocity += par_vel

            ast_mask = sys.mass == 0.0 | units.kg
            has_massive = (len(sys) > 1) and (np.sum(~ast_mask) > 0)
            if has_massive:
                newparent = nem_class.particles.add_children(sys)
                newparent_key = newparent.key

                scale_mass = newparent.mass
                scale_radius = set_parent_radius(scale_mass)
                newparent.radius = scale_radius
                if not rework_code:  # Recycle old code
                    rework_code = True
                    newcode = code
                    newcode.particles.remove_particles(code.particles)
                    newcode.particles.add_particles(sys)
                    nem_class._time_offsets[newcode] = offset
                    worker_pid = pid

                else:
                    newcode, worker_pid = nem_class._sub_worker(
                        children=sys,
                        scale_mass=scale_mass,
                        scale_radius=scale_radius
                        )
                    nem_class._set_worker_affinity(worker_pid)
                    nem_class._time_offsets[newcode] = nem_class.model_time

                nem_class._cpu_time[newparent_key] = cpu_time
                nem_class.subcodes[newparent_key] = newcode

                channel = nem_class._child_channel_maker(
                    parent_key=newparent_key,
                    code_set=newcode.particles,
                    children=sys
                )
                channel["from_children_to_gravity"].copy()  # More precise

                nem_class._pid_workers[newparent_key] = worker_pid
                nem_class.hibernate_workers(worker_pid)

            else:
                new_rogue.add_particles(sys)

        if not rework_code:  # Only triggered if pure ionisation
            code.cleanup_code()
            code.stop()

    if len(new_rogue) > 0:
        if nem_class._verbose:
            print(f"{len(new_rogue)} new rogue bodies...", flush=True)

        new_rogue.radius = set_parent_radius(new_rogue.mass)
        mask = new_rogue.radius > PARENT_RADIUS_MAX
        new_rogue[mask].radius = PARENT_RADIUS_MAX
        nem_class.particles.add_particles(new_rogue)
