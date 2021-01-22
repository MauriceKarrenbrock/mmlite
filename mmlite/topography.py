# -*- coding: utf-8 -*-
"""Topography class."""
# pylint: disable=import-error

import mdtraj
import yank


class Topography(yank.Topography):  # pylint: disable=too-few-public-methods
    """yank.Topography subclass."""
    def __init__(self, topology, ligand_atoms=None, solvent_atoms='auto'):
        if not isinstance(topology, mdtraj.Topology):
            topology = mdtraj.Topology.from_openmm(topology)
            # for openmm topologies, fix serial numbers
            for a in topology.atoms:
                a.serial = a.index if a.serial is None else a.serial
            super().__init__(topology=topology,
                             ligand_atoms=ligand_atoms,
                             solvent_atoms=solvent_atoms)
