# -*- coding: utf-8 -*-
"""Helper functions/classes."""
# pylint: disable=protected-access,no-member
from pathlib import Path

import simtk.openmm as mm
from simtk import unit
from simtk.openmm.app.pdbfile import PDBFile
from simtk.unit.quantity import Quantity

from .simulation import simulation_energy

SEED = 1234


def create_langevin_context(system,
                            *,
                            temp=298.0 * unit.kelvin,
                            friction=91.0 / unit.picosecond,
                            ts=2.0 / unit.femtoseconds,
                            xp=None,
                            random_vp=True):
    """Create the context for a Langevin simulations from a system."""

    all_have_units = all(
        [isinstance(arg, Quantity) for arg in (temp, friction, ts)])
    if not all_have_units:
        raise ValueError

    integrator = mm.LangevinIntegrator(temp, friction, ts)
    # Create a Context for this integrator
    context = mm.Context(system, integrator)

    if xp is not None:  # Set the positions
        context.setPositions(xp)

    if random_vp:
        context.setVelocitiesToTemperature(temp)

    return context


def minimize(simulation, tol=10 * unit.kilojoule / unit.mole, max_iter=None):
    """Perform a local energy minimization on the system.

    Parameters
    ----------
    simulation : Simulation or Context object.
    tol : energy=10*kilojoules/mole
        The energy tolerance to which the system should be minimized
    max_iter : int=None
        The maximum number of iterations to perform.  If this is 0,
        Default: minimization is continued until the results converge.

    Returns
    -------
    Context or Simulation.

    """

    try:
        context = simulation.context
    except AttributeError:
        context = simulation

    max_iter = max_iter or 0
    minimize.info = {
    }  # store info on the minimization as a function attribute
    minimize.info['V0'] = simulation_energy(context)['potential']
    mm.LocalEnergyMinimizer.minimize(context, tol, maxIterations=max_iter)
    minimize.info['V1'] = simulation_energy(context)['potential']


def write_single_pdb(xp, top, target):
    """Dump coordinates to a pdb file `target`."""
    try:
        target = Path(target)
    except TypeError:
        if hasattr(target, 'write'):
            PDBFile.writeFile(top, xp, target)
        else:
            raise
    else:
        with open(target, 'w') as fp:
            PDBFile.writeFile(top, xp, fp)
