# -*- coding: utf-8 -*-
"""AlchemicalMatrix class."""
import copy
from collections.abc import MutableMapping

import mdtraj
import simtk.openmm as mm
import yank


class RegionsMixin:
    """Topography methods for regions."""
    def __repr__(self):
        return '%s(topology=%r)' % (self.__class__.__name__, self.topology)

    def __str__(self):
        return str({key: '%s atoms' % len(val) for key, val in self.items()})

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

    @property
    def regions(self):
        """Return a mapping from selection names to atom indices."""
        if not self._regions:  # define base region
            self._regions['default'] = list(range(self.topology.n_atoms))
        return self._regions

    @regions.setter
    def regions(self, mapping):
        """Check and set non-overlapping regions."""
        if 'default' not in mapping:
            raise ValueError('Mapping must contain a "default" region')
        n_unique = sum(set().union(*mapping.values()))
        n_total = sum(len(x) for x in mapping.values())
        if n_unique != n_total:
            raise ValueError('Selections must be mutually exclusive.')
        if n_total != self.topology.n_atoms:
            raise ValueError('Selections must be collectively exhaustive.')
        self._regions = copy.copy(mapping)


class AlchemicalMatrix(RegionsMixin, yank.Topography, MutableMapping):  # pylint: disable=too-many-ancestors
    """Define the alchemical regions and their interactions.

    Parameters
    ----------
    topology : mdtraj.Topology or simtk.openmm.app.Topology or dict
        The topology object specifying the system.
        If a Topography, define the new regions using the _regions attribute.
        If a dict, use the keys as region names and the values as list of
        atomic indices. In both cases, no overlap is allowed between the
        atomic indices of different regions.
    ligand_atoms : iterable of int or str, optional
        The atom indices of the ligand. A string is interpreted as an mdtraj
        DSL specification of the ligand atoms.
    solvent_atoms : iterable of int or str, optional
        The atom indices of the solvent. A string is interpreted as an mdtraj
        DSL specification of the solvent atoms. If 'auto', a list of common
        solvent residue names will be used to automatically detect solvent
        atoms (default is 'auto').

    """
    def __init__(self,
                 topology,
                 ligand_atoms=None,
                 solvent_atoms='auto',
                 regions=None):
        if isinstance(topology, Topography):
            ligand_atoms = ligand_atoms or topology.ligand_atoms
            solvent_atoms = solvent_atoms or topology.solvent_atoms
            regions = regions or topology._regions
            topology = topology.topology
        super().__init__(topology,
                         ligand_atoms=ligand_atoms,
                         solvent_atoms=solvent_atoms)


class Topography(RegionsMixin, yank.Topography, MutableMapping):  # pylint: disable=too-many-ancestors
    """
    Define the alchemical regions and their interactions.

    yank.Topography derived class. Regions are non-overlapping subsets of
    atomic indices. A default region named 'default' must always be present.

    Parameters
    ----------
    topology : mdtraj.Topology or simtk.openmm.app.Topology or Topography
        The topology object specifying the system.
        If a Topography, define the new regions using the _regions attribute.
        No overlap is allowed between the atomic indices of different regions.
    regions : dict
        Use the keys as region names and the values as list of
        atomic indices. In both cases, no overlap is allowed between the
        atomic indices of different regions.
        Override the definition in `topology` argument.
    ligand_atoms : iterable of int or str, optional
        The atom indices of the ligand. A string is interpreted as an mdtraj
        DSL specification of the ligand atoms.
        Override the definition in `topology` and `regions` argument.
    solvent_atoms : iterable of int or str, optional
        The atom indices of the solvent. A string is interpreted as an mdtraj
        DSL specification of the solvent atoms. If 'auto', a list of common
        solvent residue names will be used to automatically detect solvent
        atoms (default is 'auto').
        Override the definition in `topology` and `regions` argument.

    """
    def __init__(self,
                 topology,
                 ligand_atoms=None,
                 solvent_atoms='auto',
                 regions=None):
        if isinstance(topology, yank.Topography):  # if a topography
            topography = topology
            ligand_atoms = ligand_atoms or topography.ligand_atoms
            solvent_atoms = solvent_atoms or topography.solvent_atoms
            regions = regions or topography._regions
            topology = topography.topology
        if isinstance(topology, mm.app.Topology):
            topology = mdtraj.Topology.from_openmm(topology)
            # for openmm topologies, fix serial numbers; TODO: solve this
            for a in topology.atoms:
                a.serial = a.index + 1 if (a.serial is None) else a.serial
        super().__init__(topology=topology,
                         ligand_atoms=ligand_atoms,
                         solvent_atoms=solvent_atoms)
        # initialize regions
        self._regions['default'] = list(range(self.topology.n_atoms))
        if regions:
            self._regions.update(regions)
