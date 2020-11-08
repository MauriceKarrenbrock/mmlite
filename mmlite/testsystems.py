# -*- coding: utf-8 -*-
"""Test systems."""
# pylint: disable=unused-import, too-few-public-methods
import numpy as np
from openmmtools.testsystems import TestSystem
from simtk import openmm as mm
from simtk import unit
from simtk.openmm import app


class Water(TestSystem):
    """Create a single tip3pfb water molecule."""
    def __init__(self, **kwargs):

        TestSystem.__init__(self, **kwargs)

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
