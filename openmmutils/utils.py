# -*- coding: utf-8 -*-
"""Helper functions/classes."""
# pylint: disable=protected-access,no-member
import numpy as np
from simtk import openmm, unit
from simtk.unit.quantity import Quantity


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

    integrator = openmm.LangevinIntegrator(temp, friction, ts)
    # Create a Context for this integrator
    context = openmm.Context(system, integrator)

    if xp is not None:  # Set the positions
        context.setPositions(xp)

    if random_vp:
        context.setVelocitiesToTemperature(temp)

    return context


def compute_energy(context, forces=False):
    """Return the potential energy and optionally, the forces."""
    if forces:
        state = context.getState(getEnergy=True, getForces=True)
        fp = np.asarray(state.getForces()._value)
    else:
        state = context.getState(getEnergy=True)
    vp = state.getPotentialEnergy()

    return vp if not forces else (vp, fp)


def context_data(context, data=None, **kwargs):
    """
    Return a context state.

    Parameters
    ----------
    context : context obj
    data : list or str
        List of quantities to include in the context state. If a string, split
        into a list.
    kwargs : pass to context.getState

    Returns
    -------
    state object

    """

    if isinstance(data, str):
        data = data.split()

    if data:
        data = {'get' + a.title().replace('_', ''): True for a in data}
    else:
        data = {}
    return context.getState(**data, **kwargs)
