# -*- coding: utf-8 -*-
"""Simulation utils."""

from simtk.openmm.openmm import State

from mmlite import SEED


def simulation_state(simulation, data=None, pbc=False, groups=-1):
    """
    Return a context state containing the quantities defined in `data`.

    Parameters
    ----------
    simulation : Simulation or Context or State object.
    data : list or str
        List of quantities to include in the context state.
        If a string, split into a list.
        Valid values are: {'positions', 'velocities', 'forces', 'energy',
        'parameters', 'parameter_derivatives'}
    pbc : bool=False
        Center molecules in the same cell.
    groups : set=set(range(32))
        Set of force groups indices to include when computing forces and
        energies. Default: include all energies.

    Returns
    -------
    state object

    """
    def camelcase(a):
        """Convert string to camelcase."""
        return a.title().replace('_', '')

    if isinstance(simulation, State):  # if a State object, just return
        return simulation

    try:
        context = simulation.context
    except AttributeError:
        context = simulation

    if isinstance(data, str):
        data = data.split()

    if data:
        data = {'get' + camelcase(a): True for a in data}
    else:
        data = {}

    return context.getState(**data, enforcePeriodicBox=pbc, groups=groups)


def simulation_data(simulation, data=None, pbc=False, groups=-1):
    """
    Return data from simulation state.

    Parameters
    ----------
    simulation : Simulation or Context object.
    data : list or str
        List of quantities to include in the context state.
        If a string, split into a list.
        Valid values are: {positions, velocities, forces, energy, parameters}
    pbc : bool=False
        Center molecules in the same cell.
    groups : set=set(range(32))
        Set of force groups indices to include when computing forces and
        energies. Default: include all energies.

    Returns
    -------
    dict
        A dict containing the potential and kinetic energy

    """

    # state = simulation_state(simulation, data=data, pbc=pbc, groups=-1)

    raise NotImplementedError


def simulation_energy(simulation):
    """
    Return the potential and kinetic energy.

    Parameters
    ----------
    simulation : Simulation or Context object.

    Returns
    -------
    dict
        A dict containing the potential and kinetic energy

    """

    state = simulation_state(simulation, 'energy')

    return {
        'potential': state.getPotentialEnergy(),
        'kinetic': state.getKineticEnergy()
    }


def simulation_positions(simulation):
    """
    Return atomic coordinates.

    Parameters
    ----------
    simulation : Simulation or Context object.

    Returns
    -------
    ndarray

    """

    state = simulation_state(simulation, 'positions')

    return state.getPositions(asNumpy=True)


def simulation_forces(simulation):
    """
    Return atomic forces.

    Parameters
    ----------
    simulation : Simulation or Context object.

    Returns
    -------
    ndarray

    """

    state = simulation_state(simulation, 'forces')

    return state.getForces(asNumpy=True)


def simulation_velocities(simulation):
    """
    Return atomic velocities.

    Parameters
    ----------
    simulation : Simulation or Context object.

    Returns
    -------
    ndarray

    """

    state = simulation_state(simulation, 'velocities')

    return state.getVelocities(asNumpy=True)


def set_simulation_temperature(simulation, temperature=298):
    """Initialize velocities according to temperature `t`."""
    try:
        context = simulation.context
    except AttributeError:
        context = simulation

    context.setVelocitiesToTemperature(temperature, SEED)


def set_simulation_positions(simulation, xp):
    """Set positions to `xp`."""
    try:
        context = simulation.context
    except AttributeError:
        context = simulation
    context.setPositions(xp)
