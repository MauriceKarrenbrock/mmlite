# -*- coding: utf-8 -*-
"""Gromcas tools."""
# pylint: disable=import-error
import re
from collections import OrderedDict
from pathlib import Path

import mdtraj
import parmed
from mmlite.utils import split_trajectory


def _read_lines(path=None):
    """Yield non-empty lines."""
    path = Path(path).resolve()
    with open(path) as fp:
        for line in fp:
            line = line.strip()
            yield line


class InputFile(OrderedDict):
    """
    Store input lines into an ordered dict.

    Subclasses must override the stringify(), parse() methods.

    """

    COMMENT = re.compile(r'\s*;\s*(?P<value>.*)')

    def __init__(self, path=None, **kwargs):
        self.n_lines = 0  # init counter
        self.n_comment_lines = 0  # init comments counter
        self.n_blank_lines = 0  # init comments counter
        self._path = None
        if path:
            if isinstance(path, self.__class__):
                self.__dict__.update(path.__dict__)
            else:
                self.path = path
                self.read()  # if a path is given, read input and init dict
        super().__init__(**kwargs)  # update dict with kwargs

    def __setitem__(self, key, value):
        super().__setitem__(key, str(value))  # always store as strings

    @staticmethod
    def stringify(key, value):  # pylint: disable=unused-argument
        """Return a string for non-comment line."""
        return '%s' % value

    def __str__(self):
        lines = []
        for k, v in self.items():
            if isinstance(k, tuple):
                field = k[0]
                if field == 'comment':
                    lines.append('; %s' % v)
                elif field == 'blank':
                    lines.append('')
            else:  # non-comment
                lines.append(self.stringify(k, v))
        return '\n'.join(lines)

    def _check_path(self, path=None):
        fp = Path(path).resolve() if path else self.path
        if not fp:
            raise ValueError('Path not defined')
        return fp

    @property
    def path(self):
        """Path to file."""
        return self._path

    @path.setter
    def path(self, fp=None):
        self._path = Path(fp).resolve() if fp else None

    @property
    def lines(self):
        """Read lines. Remove empty lines and strip blanks."""
        if not self.path:
            raise ValueError('Path to input file is None')
        return _read_lines(self.path)

    def is_comment(self, line):
        """Return comment string if line is a comment line."""
        comment = self.COMMENT.match(line)
        if comment:
            return comment.group('value')
        return False

    def parse(self, line, data):
        """Parse line and update data."""
        data[self.n_lines] = line
        self.n_lines += 1

    def read(self, path=None):
        """Store info into the ordered dict."""

        if path:  # when fp is passed, update self.path
            self.path = path

        data = OrderedDict()
        for line in self.lines:
            if len(line) == 0:
                data[('blank', self.n_blank_lines)] = ''
                self.n_blank_lines += 1
            else:
                # check if comment
                comment = self.is_comment(line)
                if comment:
                    data[('comment', self.n_comment_lines)] = comment
                    self.n_comment_lines += 1
                else:
                    self.parse(line, data)
                    self.n_lines += 1
        self.update(data)

    def write(self, path):
        """Write to file."""
        path = Path(path).resolve()
        with open(path, 'w') as fp:
            print(self.__str__(), file=fp)


class Mdp(InputFile):
    """
    Store .mdp parameters into an OrderedDict.

    Usage:

    >>> mdp = Mdp(<path_to_file>)  # init from .mdp file
    >>> mdp['nsteps'] = 1000  # change some value
    >>> mdp.write(<path_to_new_file>)  # save new .mdp file

    """
    PARAMETER = re.compile(
        r'''
        \s*(?P<parameter>[^=]+?)\s*=\s*  # parameter (ws-stripped), before '='
        (?P<value>[^;]*)                # value (stop before comment=;)
        (?P<comment>\s*;.*)?            # optional comment
        ''', re.VERBOSE)

    def parse(self, line, data):
        """Parse parameters lines."""
        prm = self.PARAMETER.match(line)
        if prm:
            key = prm.group('parameter')
            val = prm.group('value').rstrip()
            data[key] = val
        else:
            raise ValueError('Cant parse line %s' % line)

    @staticmethod
    def stringify(key, value):
        """Return a string for non-comment line."""
        return '%s = %s' % (key, value)


class Top(InputFile):
    """
    Store .top file info as an OrderedDict.

    """


def save_top(topology, system, path='frames/system.top'):
    """Save gromacs .top file from openmm topology and system."""
    # get a parmed.structure.Structure object
    if isinstance(topology, mdtraj.Topology):
        topology = topology.to_openmm()
    structure = parmed.openmm.load_topology(topology, system=system)
    structure.save(str(Path(path)), overwrite=True)


def save_tpr(mdp, topology, tpr='system.tpr', positions=None, system=None):
    """
    Save gromacs .tpr file.

    Parameters
    ----------
    mdp : Mdp object or filepath
    topology : openmm Topology object or mmlite TestSytem object or filepath
        If a TestSystem object, positions and system are not needed.
    positions : filepath or array-like object, optional
        If passed, override positions from TestSystem object.
    system : openmm System object, optional
        If passed, override system from TestSystem object.
    tpr : output .tpr filename, optional

    """
    raise NotImplementedError


def generate_gromacs_input(trj, *, topology=None, system=None, split=None):
    """
    Create the files needed for alchemical tranformations via gromacs.

    The following files will be created in the same dir of the `trj` file:
    * a system.top file
    * (TODO) a .mdp file
    * (optionally) a file for each frame in the `trj` file

    Parameters
    ----------
    trj : filepath
        Path to multistate NetCDF file.
    topology : Topology object or filepath
        openmm/mdtraj Topology or filepath to .pdb file.
    system : System object or filepath
        openmm System or filepath to .pdb file.
    split : str or None
        If not None, split the trajectory into frames, using the argument to
        identify the format e.g. `split='.gro'`

    """

    trj = Path(trj)

    # if file has ref_system and topography in metadata
    # get system and topology
    # https://github.com/choderalab/yank/blob/bb2b8657896b73950a9f86cbb2c3d5832e0fa692/Yank/analyze.py#L1085

    trj_dir = trj.parent.resolve()

    if split:
        split_trajectory(trj, out=trj_dir, topology=topology, ext=split)
    save_top(topology, system, path=trj_dir / 'system.top')
