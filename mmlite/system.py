# -*- coding: utf-8 -*-
"""Simulation utils."""
# pylint: disable=no-member
# pylint: disable=too-many-instance-attributes

import logging

import mdtraj
import nglview
import numpy as np
import simtk.openmm as mm
from simtk import unit

import mmlite.plot
from mmlite import defaults

logger = logging.getLogger(__name__)


class System:
    """
    System class.

    * topology
    * system
    * positions

    abstractmethod
    * setup()

    Main attributes:
    * temperature
    * friction
    * topology
    * system
    * positions

    """
    def __init__(  # pylint: disable=too-many-arguments
            self,
            topology=None,
            system=None,
            positions=None,
            temperature=None,
            friction=None):

        self.temperature = temperature or defaults.temperature
        self.friction = friction or defaults.friction
        self._system = system
        self._topology = topology
        self._mdtraj_topology = None
        self._positions = positions

        if not self._system and self.__class__ is not System:
            try:
                self.setup()
            except NotImplementedError as missing_setup:
                raise NotImplementedError(
                    'Inherited classes mut implement a setup method'
                ) from missing_setup

    def read_system_from_xml(self, source):
        """Read system from file."""
        with open(source, 'r') as fp:
            self.system = mm.XmlSerializer.deserialize(fp.read())

    def write_system_to_xml(self, target):
        """Read system from file."""
        with open(target, 'w') as fp:
            print(mm.XmlSerializer.serialize(self.system), file=fp)

    @property
    def topology(self):
        """Starting positions."""
        return self._topology

    @topology.setter
    def topology(self, value):
        """Set the starting positions."""
        self._topology = value
        self._mdtraj_topology = None

    @property
    def system(self):
        """Starting positions."""
        return self._system

    @system.setter
    def system(self, value):
        """Set the starting positions."""
        self._system = value

    @property
    def positions(self):
        """The positions of the system."""
        return self._positions

    @positions.setter
    def positions(self, value):
        self._positions = value

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

    def get_view(self, **kwargs):
        """Return a nglview view for the actual positions."""
        view = nglview.show_mdtraj(self.mdtraj)
        mmlite.plot.setup_view(view, **kwargs)
        return view

    @property
    def view(self):
        """Default ngl view."""
        return self.get_view()

    def setup(self):
        """Define self._system, self._topology, self._positions."""
        raise NotImplementedError

    def from_pdb(self, pdb, *, ff=('amber99sb.xml', 'tip3p.xml'), **kwargs):
        """
        Setup System object from pdb file.

        Optional kwargs are passed to forcefield createSystem method

        Parameters
        ----------
        pdb : filename
        ff : list
            List of forcefield files

        """
        pdb = mm.app.PDBFile(pdb)
        forcefield = mm.app.ForceField(*ff)
        if not kwargs:
            kwargs = {
                'nonbondedMethod': mm.app.PME,
                'nonbondedCutoff': 1 * unit.nanometer,
                'constraints': mm.app.HBonds
            }
        self._topology = pdb.getTopology()
        self._system = forcefield.createSystem(pdb.topology, **kwargs)
        self._positions = pdb.getPositions(asNumpy=True)


class Water(System):
    """Create a single tip3pfb water molecule."""
    def setup(self):
        self._topology = self.def_topology()
        self._positions = unit.Quantity(
            np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]]),
            unit.angstroms)
        # load the forcefield for tip3pfb
        ff = mm.app.ForceField('amber14/tip3pfb.xml')
        self._system = ff.createSystem(
            self.topology,
            nonbondedCutoff=mm.NonbondedForce.NoCutoff,
            constraints=None,
            rigidWater=False,
            removeCMMotion=True)

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

    def __call__(self):
        return self._topology, self._system, self._positions

    def __repr__(self):
        name = self.__class__.__name__
        return '%s(topology=%r,\n\tsystem=%r,\n\tposition=%r)' % (name,
                                                                  *self())


class Villin(System):
    """Solvated villin."""

    pdbfile = '/home/simo/scr/mmlite/data/villin.pdb'

    def setup(self):
        self.from_pdb(self.pdbfile)
