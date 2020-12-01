# -*- coding: utf-8 -*-
"""AlchemicalMatrix class."""
import copy
from collections.abc import MutableMapping

from yank import Topography


class AlchemicalMatrix(Topography, MutableMapping):  # pylint: disable=too-many-ancestors
    """Define the alchemical regions and their interactions."""
    def __init__(self, topology, ligand_atoms=None, solvent_atoms='auto'):
        super().__init__(topology,
                         ligand_atoms=ligand_atoms,
                         solvent_atoms=solvent_atoms)
        self._regions[0] = list(range(self.topology.n_atoms))

    def __getitem__(self, region):
        return copy.copy(self._regions[region])

    def __iter__(self):
        return iter(self._regions)

    def __len__(self):
        return len(self._regions)

    def __setitem__(self, region, selection):
        self._check_existing_regions(region)
        self._check_reserved_words(region)
        atom_selection = self.select(selection)
        # remove selection from pre-existing regions
        for name, sele in self._regions.items():
            self._regions[name] = sorted(set(sele) - set(atom_selection))
        # add new region
        self._regions[region] = atom_selection

    def __delitem__(self, region):
        self.remove_region(region)
