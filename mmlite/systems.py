# -*- coding: utf-8 -*-
"""MdSys systems."""
import logging

import numpy as np
import simtk.openmm as mm
from simtk import unit

from .mdsys import MdSys

logger = logging.getLogger(__name__)


class Water(MdSys):
    """Create a single tip3pfb water molecule."""
    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        # load the forcefield for tip3pfb
        ff = mm.app.ForceField('amber14/tip3pfb.xml')

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


class PeriodicLJ(MdSys):
    """Create a periodic Lennard-Jones system."""
    def __init__(self,
                 *args,
                 n=512,
                 m=39.9 * unit.amu,
                 density=0.05,
                 q=0 * unit.elementary_charge,
                 sigma=3.4 * unit.angstroms,
                 eps=0.238 * unit.kilocalories_per_mole,
                 **kwargs):
        super().__init__(*args, **kwargs)

        # init the system
        self.system = mm.System()

        # add the particles
        for _ in range(n):
            self.system.addParticle(m)

        # set the periodic box
        v = n * sigma**3 / density  # box volume
        edge = v**(1 / 3) / unit.angstroms
        self.box = np.diag([edge] * 3) * unit.angstroms  # box vectors
        self.system.setDefaultPeriodicBoxVectors(*self.box)

        # add LJ interactions
        force = mm.NonbondedForce()
        force.setNonbondedMethod(mm.NonbondedForce.CutoffPeriodic)
        # add force parameters to each particle in the System object
        for _ in range(n):
            force.addParticle(q, sigma, eps)
        # set cutoff (truncation) distance at 3*sigma
        force.setCutoffDistance(3.0 * sigma)
        # use a smooth switching function at cutoff
        force.setUseSwitchingFunction(True)
        # turn on switch at 2.5*sigma
        force.setSwitchingDistance(2.5 * sigma)
        # use long-range isotropic dispersion correction
        force.setUseDispersionCorrection(True)

        # system takes ownership of the NonbondedForce objec
        _ = self.system.addForce(force)

        # set positions
        self.positions = edge * np.random.rand(self.n_particles, 3)
