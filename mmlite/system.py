# -*- coding: utf-8 -*-
"""Simulation utils."""
# pylint: disable=no-member
# pylint: disable=too-many-instance-attributes

import logging
from abc import ABC, abstractmethod  # python >= 3.4

import mdtraj
import nglview
import numpy as np
import simtk.openmm as mm
from simtk import unit

from mmlite import defaults

logger = logging.getLogger(__name__)


class System(ABC):
    """
    System class.

    abstract properties
    * topology
    * system
    * positions

    Main attributes:
    * temperature
    * friction
    * topology
    * system
    * positions

    """
    def __init__(self):

        self.temperature = defaults.temperature
        self.friction = defaults.friction
        self._topology = None
        self._mdtraj_topology = None
        self._system = None
        self._positions = None

    def read_system_from_xml(self, source):
        """Read system from file."""
        with open(source, 'r') as fp:
            self.system = mm.XmlSerializer.deserialize(fp.read())

    def write_system_to_xml(self, target):
        """Read system from file."""
        with open(target, 'w') as fp:
            print(mm.XmlSerializer.serialize(self.system), file=fp)

    @property
    @abstractmethod
    def topology(self):
        """Starting positions."""
        return self._topology

    @topology.setter
    def topology(self, value):
        """Set the starting positions."""
        self._topology = value
        self._mdtraj_topology = None

    @property
    @abstractmethod
    def system(self):
        """Starting positions."""

    @system.setter
    def system(self, value):
        """Set the starting positions."""
        self._system = value

    @property
    @abstractmethod
    def positions(self):
        """The positions of the system."""
        if self._positions is None:  # set to the starting coordinates
            pass  # implement with initial positions
        return self._positions

    @positions.setter
    def positions(self, value):
        self._positions = value
        return self._topology

    @property
    def mdtraj_topology(self):
        """The mdtraj.Topology object corresponding to the system."""
        if self._mdtraj_topology is None:
            self._mdtraj_topology = mdtraj.Topology.from_openmm(self._topology)
        return self._mdtraj_topology

    @property
    def name(self):
        """The name of the system."""
        return self.__class__.__name__

    @property
    def n_particles(self):
        """Total number of particles."""
        return self.system.getNumParticles()

    @property
    def masses(self):
        """List of particles masses."""
        return [
            self.system.getNumParticles(i) for i in range(self.n_particles)
        ]

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


class Water(System):
    """Create a single tip3pfb water molecule."""
    @property
    def topology(self):
        if self._topology is None:
            self._topology = self.def_topology()
        return self._topology

    @property
    def positions(self):
        if self._positions is None:
            self._positions = unit.Quantity(
                np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]]),
                unit.angstroms)
        return self._positions

    @property
    def system(self):
        """Create OpenMM System."""
        if self._system is None:
            # load the forcefield for tip3pfb
            ff = mm.app.ForceField('amber14/tip3pfb.xml')
            self._system = ff.createSystem(
                self.topology,
                nonbondedCutoff=mm.NonbondedForce.NoCutoff,
                constraints=None,
                rigidWater=False,
                removeCMMotion=True)
        return self._system

    @staticmethod
    def def_topology():
        """Water molecule topology."""
        topology = mm.app.Topology()
        # add `chain` to the topology
        chain = topology.addChain()
        # add a residue named "water" to the chain
        residue = topology.addResidue('water', chain)
        oxigen = mm.app.Element.getByAtomicNumber(8)
        hydrogen = mm.app.Element.getByAtomicNumber(1)
        atom0 = topology.addAtom('O', oxigen, residue)
        atom1 = topology.addAtom('H', hydrogen, residue)
        atom2 = topology.addAtom('H', hydrogen, residue)
        topology.addBond(atom0, atom1)
        topology.addBond(atom0, atom2)
        return topology
