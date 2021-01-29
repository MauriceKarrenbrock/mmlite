# -*- coding: utf-8 -*-
"""MdSys systems."""
import logging

import numpy as np
import simtk.openmm as mm
from openmmtools import testsystems
from simtk import unit

from mmlite.system import SystemMixin

logger = logging.getLogger(__name__)


class Water(SystemMixin, testsystems.TestSystem):
    """Create a single tip3pfb water molecule."""
    def __init__(self):
        super().__init__()
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


class Villin(SystemMixin, testsystems.TestSystem):
    """Solvated villin."""

    pdbfile = '/home/simo/scr/mmlite/data/villin.pdb'

    def __init__(self):
        super().__init__()
        self.from_pdb(self.pdbfile)


class HostGuestExplicit(SystemMixin, testsystems.HostGuestExplicit):
    """CB7:B2 host-guest system in TIP3P explicit solvent."""


class HostGuestImplicit(SystemMixin, testsystems.HostGuestImplicit):
    """CB7:B2 host-guest system in TIP3P explicit solvent."""


class LysozymeImplicit(SystemMixin, testsystems.LysozymeImplicit):
    """
    T4 lysozyme L99A (AMBER ff96) with p-xylene ligand (GAFF + AM1-BCC) in
    implicit OBC GBSA solvent.
    """


class AlanineDipeptideVacuum(SystemMixin, testsystems.AlanineDipeptideVacuum):
    """CB7:B2 host-guest system in TIP3P explicit solvent."""


class AlanineDipeptideImplicit(SystemMixin,
                               testsystems.AlanineDipeptideImplicit):
    """Alanine dipeptide ff96 in OBC GBSA implicit solvent."""


class AlanineDipeptideExplicit(SystemMixin,
                               testsystems.AlanineDipeptideExplicit):
    """Alanine dipeptide ff96 in OBC GBSA implicit solvent."""


class DiatomicFluid(SystemMixin, testsystems.DiatomicFluid):
    """Create a diatomic fluid."""


class LennardJonesFluid(SystemMixin, testsystems.LennardJonesFluid):
    """Periodic fluid of Lennard-Jones particles."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Create topology.
        topology = mm.app.Topology()
        element = mm.app.Element.getBySymbol('Ar')
        chain = topology.addChain()
        residue = topology.addResidue('Alc', chain)
        topology.addAtom('Ar', element, residue)
        for _ in range(1, self.system.getNumParticles()):
            residue = topology.addResidue('Ar', chain)
            topology.addAtom('Ar', element, residue)
        self.topology = topology
