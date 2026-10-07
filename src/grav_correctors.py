"""
Possible Room for Improvements:
1. An AMUSE interface for the C++ library has been implemented,
   but the current implementation is quicker. If you wish to have it
   please contact Erwan.
2. The correction kick script is very rudimentary and can be improved,
   namely by adding a get_potential_at_point function.
3. The code is noisy, mainly due to stripping position arrays. This, however,
   has been seen to be more efficient and less memory intensive than directly
   using the original arrays.
4. Unit conversions is messy.
"""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

from amuse.lab import units, Particles

from src.globals import ACC_UNITS, SI_UNITS


def _as_float64_si(q, target_unit) -> np.ndarray:
    """
    Convert an AMUSE Quantity to a float64 numpy array in target_unit.
    """
    arr = np.asarray(q.value_in(target_unit), dtype=np.float64)
    if arr.ndim == 0:
        return arr.reshape(1)
    return arr.ravel()


def compute_gravity(
    grav_lib: object,
    pert_m: units.mass,
    pert_x: units.length,
    pert_y: units.length,
    pert_z: units.length,
    infl_x: units.length,
    infl_y: units.length,
    infl_z: units.length,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute gravitational force.
    Args:
        grav_lib (object):          C++ gravitational library.
        pert_m (units.mass):        Mass of perturber particles.
        pert_x/y/z (units.length):  Position of perturber particles.
        infl_x/y/z (units.length):  Position of influenced particles.
    Returns:
        tuple:  Acceleration array of particles (ax, ay, az)
    """
    pm = _as_float64_si(pert_m, units.kg)
    px = _as_float64_si(pert_x, units.m)
    py = _as_float64_si(pert_y, units.m)
    pz = _as_float64_si(pert_z, units.m)

    ix = _as_float64_si(infl_x, units.m)
    iy = _as_float64_si(infl_y, units.m)
    iz = _as_float64_si(infl_z, units.m)

    n_pert = len(pm)
    n_part = len(ix)

    ax = np.zeros(n_part, dtype=np.float64)
    ay = np.zeros(n_part, dtype=np.float64)
    az = np.zeros(n_part, dtype=np.float64)

    grav_lib.find_gravity_at_point(
        pm, px, py, pz,
        ix, iy, iz,
        ax, ay, az,
        n_part, n_pert
    )

    return ax, ay, az


def correct_parents_threaded(
        grav_lib: object,
        chd_mass: units.mass,
        par_mass: units.mass,
        chd_pos: units.length,
        par_pos: units.length,
        ext_pos: units.length,
        rmv_idx: int,
        acc_units: units.acc
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute correction kicks for parents due to set of children particles.
    Args:
        grav_lib (object):       C++ gravitational library.
        chd_mass (units.mass):   Mass of the child particles.
        par_mass (units.mass):   Mass of host parent particle.
        chd_pos (units.length):  Position of the child particles.
        par_pos (units.length):  Position of host  parent particle.
        ext_pos (units.length):  Position of all particles.
        rmv_idx (int):           Index of host parent in complete parent set.
        acc_units (units):       Unit conversion variable.
    Returns:
        tuple:  Array of correction kicks for parent particle (ax, ay, az).
    """
    par_x = par_pos[0]
    par_y = par_pos[1]
    par_z = par_pos[2]

    chd_x = chd_pos[0] + par_x
    chd_y = chd_pos[1] + par_y
    chd_z = chd_pos[2] + par_z

    ext_x = ext_pos[0]
    ext_y = ext_pos[1]
    ext_z = ext_pos[2]

    mask = np.ones(len(ext_x), dtype=bool)
    mask[rmv_idx] = False
    ext_x = ext_x[mask]
    ext_y = ext_y[mask]
    ext_z = ext_z[mask]

    ax_chd, ay_chd, az_chd = compute_gravity(
        grav_lib=grav_lib,
        pert_m=chd_mass,
        pert_x=chd_x,
        pert_y=chd_y,
        pert_z=chd_z,
        infl_x=ext_x,
        infl_y=ext_y,
        infl_z=ext_z
    )

    ax_par, ay_par, az_par = compute_gravity(
        grav_lib=grav_lib,
        pert_m=par_mass,
        pert_x=par_x,
        pert_y=par_y,
        pert_z=par_z,
        infl_x=ext_x,
        infl_y=ext_y,
        infl_z=ext_z,
    )

    # Compute correction kicks
    dax = (ax_chd - ax_par) * SI_UNITS
    day = (ay_chd - ay_par) * SI_UNITS
    daz = (az_chd - az_par) * SI_UNITS

    corr_ax = dax.value_in(acc_units).astype(np.float64)
    corr_ay = day.value_in(acc_units).astype(np.float64)
    corr_az = daz.value_in(acc_units).astype(np.float64)

    corr_ax = np.insert(corr_ax, rmv_idx, 0.0)
    corr_ay = np.insert(corr_ay, rmv_idx, 0.0)
    corr_az = np.insert(corr_az, rmv_idx, 0.0)

    return (
        corr_ax | acc_units,
        corr_ay | acc_units,
        corr_az | acc_units
        )


class CorrectionFromCompoundParticle(object):
    def __init__(
        self,
        grav_lib: object,
        par: Particles,
        chd: Particles,
        nworkers: int
    ):
        """
        Compute correction kicks for parents due to set of children.
        Args:
            grav_lib (object): C++ gravitational library.
            par (Particles):   Original parent particle set.
            chd (Particles):   Ensemble of children particles.
            nworkers (int):    Number of cores to use.
        """
        self.grav_lib = grav_lib
        self.nworkers = nworkers

        self.all_parents = par
        self.all_parents_pos = [par.x, par.y, par.z]
        self.chd = chd

    def get_gravity_at_point(self) -> tuple:
        """
        Compute correction kicks for parents due to all children particles.
        Returns:
            tuple:  Acceleration array of parent particles (ax, ay, az)
        """
        Nparticles = len(self.all_parents)

        ax_corr = np.zeros(Nparticles) | ACC_UNITS
        ay_corr = np.zeros(Nparticles) | ACC_UNITS
        az_corr = np.zeros(Nparticles) | ACC_UNITS

        parent_idx = {p.key: i for i, p in enumerate(self.all_parents)}
        futures = []
        with ThreadPoolExecutor(max_workers=self.nworkers) as executor:
            for parent, system in list(self.chd.values()):
                rmv_idx = parent_idx.pop(parent.key)
                par_pos = [parent.x, parent.y, parent.z]
                chd_pos = [system.x, system.y, system.z]

                future = executor.submit(
                    correct_parents_threaded,
                    grav_lib=self.grav_lib,
                    chd_mass=system.mass,
                    par_mass=parent.mass,
                    ext_pos=self.all_parents_pos,
                    chd_pos=chd_pos,
                    par_pos=par_pos,
                    rmv_idx=rmv_idx,
                    acc_units=ACC_UNITS
                    )
                futures.append(future)

            for future in as_completed(futures):
                ax, ay, az = future.result()
                ax_corr += ax
                ay_corr += ay
                az_corr += az

        return ax_corr, ay_corr, az_corr

    def get_potential_at_point(self, radius, x, y, z) -> np.ndarray:
        """
        Get the potential at a specific location
        Args:
            radius (units.length):  Radius of the particle at that location
            x/y/z (units.length):   Position of the location
        Returns:
            Array:  The potential field at the location
        """
        raise NotImplementedError(
            "Potential correction is not yet implemented."
            )


class CorrectionForCompoundParticle(object):
    def __init__(
        self,
        grav_lib: object,
        child: Particles,
        pert_mass: units.mass,
        chd_x: units.length,
        chd_y: units.length,
        chd_z: units.length,
        par_x: units.length,
        par_y: units.length,
        par_z: units.length,
        pert_x: units.length,
        pert_y: units.length,
        pert_z: units.length
    ):
        """
        Correct force vector exerted by global particles on systems
        Args:
            grav_lib (object):          C++ gravitational library.
            child (Particles):          The child particles.
            pert_mass (units.mass):     Mass of perturber particle.
            chd_x/y/z (units.length):   Position of the children particles.
            par_y/y/z (units.length):   Position of host parent particle.
            pert_x/y/z (units.length):  Position of perturber particle.
        """
        self.grav_lib = grav_lib

        self.par_x = par_x
        self.par_y = par_y
        self.par_z = par_z

        self.child = child
        self.chd_x = chd_x
        self.chd_y = chd_y
        self.chd_z = chd_z

        self.pert_mass = pert_mass
        self.pert_x = pert_x
        self.pert_y = pert_y
        self.pert_z = pert_z

    def get_gravity_at_point(self) -> tuple:
        """
        Compute gravitational correction kicks.
        Args:
            radius (units.length):  Radius of the system particle
            x/y/z (units.length):   Position of the system particle
        Returns:
            tuple:  Acceleration array of system particles (ax, ay, az)
        """
        Nsystem = len(self.child)
        dax = np.zeros(Nsystem) | ACC_UNITS
        day = np.zeros(Nsystem) | ACC_UNITS
        daz = np.zeros(Nsystem) | ACC_UNITS

        ax_chd, ay_chd, az_chd = compute_gravity(
            grav_lib=self.grav_lib,
            pert_m=self.pert_mass,
            pert_x=self.pert_x,
            pert_y=self.pert_y,
            pert_z=self.pert_z,
            infl_x=self.chd_x,
            infl_y=self.chd_y,
            infl_z=self.chd_z
            )

        ax_par, ay_par, az_par = compute_gravity(
            grav_lib=self.grav_lib,
            pert_m=self.pert_mass,
            pert_x=self.pert_x,
            pert_y=self.pert_y,
            pert_z=self.pert_z,
            infl_x=self.par_x,
            infl_y=self.par_y,
            infl_z=self.par_z,
            )

        dax += (ax_chd - ax_par) * SI_UNITS
        day += (ay_chd - ay_par) * SI_UNITS
        daz += (az_chd - az_par) * SI_UNITS

        return dax, day, daz

    def get_potential_at_point(self, radius, x, y, z) -> np.ndarray:
        """
        Get the potential at a specific location.
        Args:
            radius (units.length):  Radius of the system particle
            x/y/z (units.length):   x/y/z Location of the system particle
        Returns:
            Array:  The potential field at the system particle's location
        """
        raise NotImplementedError(
            "Potential correction is not yet implemented."
            )


class CorrectionKicks(object):
    def __init__(self, grav_lib: object, nworkers: int):
        """
        Apply correction kicks onto particles.
        Args:
            grav_lib (object):  C++ gravitational library.
            nworkers (int):     Number of cores to use.
        """
        self.grav_lib = grav_lib
        self.nworkers = nworkers

    def _kick_particles(
        self,
        particles: Particles,
        corr_code: object,
        dt: units.time,
        parent_set: bool
    ) -> None:
        """
        Apply correction kicks onto target particles.
        Args:
            particles (Particles):  Particles whose accelerations
                                    are corrected.
            corr_code (object):     Class computing correction kicks.
            dt (units.time):        Nemesis bridge time step.
            parent_set (bool):      If target particles are the
                                    parent particles.
        """
        ax, ay, az = corr_code.get_gravity_at_point()

        if parent_set:
            parts = particles.copy()
            parts.vx = particles.vx + dt * ax
            parts.vy = particles.vy + dt * ay
            parts.vz = particles.vz + dt * az

            channel = parts.new_channel_to(particles)
            channel.copy_attributes(["vx", "vy", "vz"])
        else:
            particles.vx = particles.vx + dt * ax
            particles.vy = particles.vy + dt * ay
            particles.vz = particles.vz + dt * az

    def _correct_children(
            self,
            pert_mass: units.mass,
            pert_x: units.length,
            pert_y: units.length,
            pert_z: units.length,
            par_x: units.length,
            par_y: units.length,
            par_z: units.length,
            child: Particles,
            dt: units.time
            ) -> None:
        """
        Apply correcting kicks onto children particles.
        Args:
            pert_mass (units.mass):      Mass of perturber.
            pert_x/y/z (units.length):   Position of perturber.
            par_x/y/z (units.length):    Position of parent particle.
            child (Particles):           Child particle set.
            dt (units.time):             Nemesis bridge time step.
        """
        chd_x = child.x + par_x
        chd_y = child.y + par_y
        chd_z = child.z + par_z

        corr_par = CorrectionForCompoundParticle(
            grav_lib=self.grav_lib,
            child=child,
            pert_mass=pert_mass,
            chd_x=chd_x,
            chd_y=chd_y,
            chd_z=chd_z,
            par_x=par_x,
            par_y=par_y,
            par_z=par_z,
            pert_x=pert_x,
            pert_y=pert_y,
            pert_z=pert_z,
            )
        self._kick_particles(child, corr_par, dt, parent_set=False)

    def _correction_kicks(
            self,
            particles: Particles,
            children: dict,
            dt: units.time
    ) -> None:
        """
        Apply correcting kicks onto children and parent particles.
        Args:
            particles (Particles):  Parent particle set.
            children (dict):        Dictionary of children system.
            dt (units.time):        Nemesis bridge time step.
        """
        def process_children_jobs(parent, children):
            rmv_idx = abs(particles_key - parent.key).argmin()
            mask = np.ones(len(particles_mass), dtype=bool)
            mask[rmv_idx] = False

            pert_mass = particles_mass[mask]
            pert_xpos = particles_x[mask]
            pert_ypos = particles_y[mask]
            pert_zpos = particles_z[mask]

            future = executor.submit(
                self._correct_children,
                pert_mass=pert_mass,
                pert_x=pert_xpos,
                pert_y=pert_ypos,
                pert_z=pert_zpos,
                par_x=parent.x,
                par_y=parent.y,
                par_z=parent.z,
                child=children,
                dt=dt
                )

            return future

        if len(children) > 0 and len(particles) > 1:
            # Setup array for CorrectionFor
            particles_key = particles.key
            particles_mass = particles.mass
            particles_x = particles.x
            particles_y = particles.y
            particles_z = particles.z

            corr_chd = CorrectionFromCompoundParticle(
                grav_lib=self.grav_lib,
                par=particles,
                chd=children,
                nworkers=self.nworkers
                )
            self._kick_particles(
                particles,
                corr_chd,
                dt,
                parent_set=True
            )

            futures = []
            with ThreadPoolExecutor(max_workers=self.nworkers) as executor:
                try:
                    for parent, child in children.values():
                        future = process_children_jobs(parent, child)
                        futures.append(future)
                    for future in as_completed(futures):
                        future.result()
                except Exception as e:
                    raise RuntimeError(f"Error during correction kicks: {e}")
