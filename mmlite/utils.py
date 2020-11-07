# -*- coding: utf-8 -*-
"""Helper functions/classes."""
import sys
# pylint: disable=protected-access,no-member
from pathlib import Path

import mdtraj
import nglview
import numpy as np
import simtk.openmm as mm
from simtk import unit
from simtk.openmm.app.pdbfile import PDBFile
from simtk.openmm.app.pdbreporter import PDBReporter
from simtk.openmm.app.statedatareporter import StateDataReporter
from simtk.openmm.openmm import State
from simtk.unit.quantity import Quantity

SEED = 1234


class Output:  # pylint: disable=too-few-public-methods
    """
    Helper class for reporters.

    Parameters
    ----------
    kind : str
        The reporter label, valid values are `pdb`, `data`, `screen`
    target : path or file-like
        The output file.
    dt : int
        Output frequency as number of time steps.

    """
    minimal_data = {'step': True, 'totalEnergy': True, 'temperature': True}
    full_data = {
        'step': True,
        'time': True,
        'potentialEnergy': True,
        'totalEnergy': True,
        'temperature': True
    }
    default_targets = {
        'pdb': 'traj.pdb',
        'data': 'data.csv',
        'screen': sys.stdout
    }

    def __init__(self, kind, dt, *, target=None):
        self.kind = kind
        self.dt = dt
        self.target = target or self.default_targets[kind]
        self._reporter = None

    @property
    def reporter(self):
        """Reporter object."""
        if self.kind == 'pdb':
            self._reporter = PDBReporter(self.target, self.dt)
        elif self.kind == 'data':
            self._reporter = StateDataReporter(self.target, self.dt,
                                               **self.full_data)
        elif self.kind == 'screen':
            self._reporter = StateDataReporter(sys.stdout, self.dt,
                                               **self.minimal_data)
        else:
            raise ValueError('Unkown reporter kind %s' % self.kind)
        return self._reporter


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


def compute_energy(context, forces=False):
    """Return the potential energy and optionally, the forces."""
    if forces:
        state = context.getState(getEnergy=True, getForces=True)
        fp = np.asarray(state.getForces()._value)
    else:
        state = context.getState(getEnergy=True)
    vp = state.getPotentialEnergy()

    return vp if not forces else (vp, fp)


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


def view_traj(xp, top):
    """Create an MDTraj Trajectory object and return a nglview view.

    Parameters
    ----------
    xp : array-like
    top : topology

    Return
    ------
    nglview view object

    """

    mdtop = mdtraj.Topology.from_openmm(top)
    traj = mdtraj.Trajectory(xp / unit.nanometers, mdtop)
    view = nglview.show_mdtraj(traj)
    if len(xp) < 10000:
        view.add_ball_and_stick('all')
    view.center(zoom=True)
    return view


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


def set_simulation_temperature(simulation, t=298):
    """Initialize velocities according to temperature `t`."""
    simulation.context.setVelocitiesToTemperature(t, 1)


def add_reporters(simulation,
                  outs=(Output('pdb', 1), Output('data',
                                                 1), Output('screen', 100))):
    """Define the simulation reporters.

    Parameters
    ----------
    simulation : Simulation object.
    outs : list of outputs

    """

    simulation.reporters = []
    for out in outs:
        simulation.reporters.append(out.reporter)
