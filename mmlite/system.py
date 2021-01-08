# -*- coding: utf-8 -*-
"""Simulation utils."""
# pylint: disable=no-member
# pylint: disable=too-many-instance-attributes

import copy
import logging

import mdtraj
import nglview
import simtk.openmm as mm
from parmed import load_file
from simtk import unit

import mmlite.plot

logger = logging.getLogger(__name__)

# https://github.com/openmm/openmm/issues/2330
SYSTEM_DEFAULTS = {
    'nonbondedMethod': mm.app.PME,
    'nonbondedCutoff': 1.0 * unit.nanometer,
    'constraints': mm.app.HBonds,
    'rigidWater': True,
    'ewaldErrorTolerance': 5.e-4
}


class SystemMixin:
    """Add methods to TestSystem."""
    def read_system_from_xml(self, source):
        """Read system from file."""
        with open(source, 'r') as fp:
            self.system = mm.XmlSerializer.deserialize(fp.read())

    def write_system_to_xml(self, target):
        """Read system from file."""
        with open(target, 'w') as fp:
            print(mm.XmlSerializer.serialize(self.system), file=fp)

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
        args = copy.deepcopy(SYSTEM_DEFAULTS)
        args.update(kwargs)
        self._topology = pdb.getTopology()
        self._system = forcefield.createSystem(pdb.topology, **args)
        self._positions = pdb.getPositions(asNumpy=True)

    def from_gro(self, gro, top, **kwargs):
        """
        Setup System object from Gromacs .gro and .top file.

        Optional kwargs are passed to parmed topology createSystem method

        Parameters
        ----------
        gro : filename
        top : filename
        ff : list
            Dict of createSystem parameters.

        """
        coords = load_file(gro)
        top = load_file(top)
        top.box = coords.box

        args = copy.deepcopy(SYSTEM_DEFAULTS)
        args.update(kwargs)

        self._topology = top.topology
        self._system = top.createSystem(**args)
        self._positions = coords.coordinates

    def __call__(self):
        return self._topology, self._system, self._positions

    def __repr__(self):
        name = self.__class__.__name__
        return '%s(topology=%r,\n\tsystem=%r,\n\tposition=%r)' % (name,
                                                                  *self())

    @property
    def thermodynamic_state(self):
        """openmmtools ThermodynamicState object."""
        return self._thermodynamic_state

    @property
    def default_box_vectors(self):
        """System default box vectors."""
        return self.system.getDefaultPeriodicBoxVectors()

    @property
    def box_vectors(self):
        """Box vectors."""
        if self._box_vectors is None:
            self._box_vectors = self.default_box_vectors
        return self._box_vectors
