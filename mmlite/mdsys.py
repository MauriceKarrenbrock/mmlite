# -*- coding: utf-8 -*-
"""Test systems."""
# pylint: disable=unused-import, too-few-public-methods, no-member
import logging
from abc import ABC  # python >= 3.4

import mdtraj
import nglview
import numpy as np
from openmmtools.testsystems import TestSystem
from simtk import openmm as mm
from simtk import unit
from simtk.openmm import app
from simtk.openmm.openmm import LangevinIntegrator, VerletIntegrator

from .output import add_screen_output, add_state_output, add_trajectory_output
from .simulation import (set_simulation_positions, set_simulation_temperature,
                         simulation_energy)

logger = logging.getLogger(__name__)


class MdSys(TestSystem, ABC):
    """Base class for storing data for a simulation.

    Basic attributes are:
    * topology
    * positions
    * system  # system = topology + forcefield; contains forces
    * simulation  # simulation = system + topology + integrator

    Parameters
    ----------
    integrator_class : Integrator object
    dt : Quantity
    friction : Quantity
    temperature : Quantity
    *args, **kwargs : to TestSystem

    """
    def __init__(self, *args, **kwargs):
        self.integrator_class = kwargs.pop('integrator', VerletIntegrator)
        self.dt = kwargs.pop('dt', 2.0 * unit.femtoseconds)
        self.friction = kwargs.pop('friction', 91.0 / unit.picosecond)
        self.temperature = kwargs.pop('temperature', 298.0 * unit.kelvin)
        super().__init__(*args, **kwargs)
        self._simulation = None

    @property
    def integrator(self):
        """Return a fresh integrator."""
        if self.integrator_class.__name__ == 'VerletIntegrator':
            return VerletIntegrator(self.dt)
        if self.integrator_class.__name__ == 'LangevinIntegrator':
            return LangevinIntegrator(self.temperature, self.friction, self.dt)
        raise ValueError(self.integrator_class)

    def langevin_integrator(self):
        """Return a fresh Langevin integrator."""
        return mm.LangevinIntegrator(self.temperature, self.friction, self.dt)

    def verlet_integrator(self):
        """Return a fresh Verlet integrator."""
        return mm.VerletIntegrator(self.dt)

    @property
    def simulation(self):
        """Return a Simulation object."""
        if not self._simulation:
            sim = app.Simulation(self.topology, self.system, self.integrator)
            set_simulation_positions(sim, self.positions)
            set_simulation_temperature(sim, temperature=self.temperature)
            # add reporters
            add_trajectory_output(sim)
            add_screen_output(sim)
            add_state_output(sim)
            self._simulation = sim
        return self._simulation

    def reset(self):
        """Reinitialize the simulation attribute."""
        self._simulation = None

    @property
    def mdtraj(self):
        """mdtraj object from actual positions."""
        top = mdtraj.Topology.from_openmm(self.topology)
        return mdtraj.Trajectory([self.positions / unit.nanometers], top)

    @property
    def view(self):
        """Return a nglview view for the actual positions."""
        view = nglview.show_mdtraj(self.mdtraj)
        if len(self.positions) < 10000:
            view.add_ball_and_stick('all')
        view.center(zoom=True)
        return view

    def minimize(self, tol=10 * unit.kilojoule / unit.mole, max_iter=None):
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

        ctx = self.simulation.context
        max_iter = max_iter or 0
        logger.info('Energy before minimization: %s',
                    simulation_energy(ctx)['potential'])
        mm.LocalEnergyMinimizer.minimize(ctx, tol, maxIterations=max_iter)
        logger.info('Energy after minimization: %s',
                    simulation_energy(ctx)['potential'])

    def step(self, *args, **kwargs):
        """Advance the system by a specified number of time steps."""
        self.simulation.step(*args, **kwargs)

    @property
    def reporters(self):
        """List the simulation reporters."""
        return self.simulation.reporters


class Water(MdSys):
    """Create a single tip3pfb water molecule."""
    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        # load the forcefield for tip3pfb
        ff = app.ForceField('amber14/tip3pfb.xml')

        self.topology = self.def_topology()
        self.positions = unit.Quantity(
            np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]]),
            unit.angstroms)

        # Create OpenMM System.
        self.system = ff.createSystem(
            self.topology,
            nonbondedCutoff=mm.NonbondedForce.NoCutoff,
            constraints=None,
            rigidWater=False,
            removeCMMotion=True)

        n_atoms = self.system.getNumParticles()
        self.ndof = 3 * n_atoms

    @staticmethod
    def def_topology():
        """Water molecule topology."""
        topology = app.Topology()
        # add `chain` to the topology
        chain = topology.addChain()
        # add a residue named "water" to the chain
        residue = topology.addResidue('water', chain)
        oxigen = app.Element.getByAtomicNumber(8)
        hydrogen = app.Element.getByAtomicNumber(1)
        atom0 = topology.addAtom('O', oxigen, residue)
        atom1 = topology.addAtom('H', hydrogen, residue)
        atom2 = topology.addAtom('H', hydrogen, residue)
        topology.addBond(atom0, atom1)
        topology.addBond(atom0, atom2)
        return topology
