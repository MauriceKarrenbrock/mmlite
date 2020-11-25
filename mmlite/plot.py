# -*- coding: utf-8 -*-
"""Plotters."""
# pylint: disable=no-member
import copy

import matplotlib.pyplot as plt
import mdtraj
import nglview
import pandas
from simtk import unit

# see:
# http://nglviewer.org/ngl/api/manual/selection-language.html
# http://nglviewer.org/ngl/api/manual/molecular-representations.html
default_representations = {
    'protein': {
        'type': 'cartoon',
        'params': {
            'sele': 'protein'
        }
    },
    'ligand': {
        'type': 'ball+stick',
        'params': {
            'sele': 'ligand'
        }
    },
    'ion': {
        'type': 'ball+stick',
        'params': {
            'sele': 'ion'
        }
    },
    'water': {
        'type': 'line',
        'params': {
            'sele': 'water',
            'opacity': 0.1
        }
    },
}


def setup_view(view, **kwargs):
    """Setup view representations."""
    reps = copy.deepcopy(default_representations)
    for key, val in kwargs.items():
        if isinstance(val, str):  # set or override defaults
            reps[key] = {'type': val, 'params': {'sele': key}}
        if not val:  # remove if in defaults
            if key in reps:
                reps.pop(key)
    view.representations = list(reps.values())
    view.camera = 'orthographic'
    view.center(zoom=True)


def show_mdtraj(fp, stride=None, atom_indices=None, **kwargs):
    """
    Return a trajectory view.

    Parameters
    ----------
    fp : filename or list of filenames
    stride : int, optional
        Read every stride-th frame
    atom_indices : array, optional
        Read only a subset of the atoms coordinates from the file
    kwargs : dict
        Molecular representation of selections, e.g. protein='cartoon'

    Returns
    -------
    View object

    """
    traj = mdtraj.load(fp, stride=stride, atom_indices=atom_indices)
    view = nglview.show_mdtraj(traj)
    setup_view(view, **kwargs)
    return view


def show_file(fp, **kwargs):
    """
    Return a structure view.

    Parameters
    ----------
    fp : filename or list of filenames
    kwargs : dict
        Molecular representation of selections, e.g. protein='cartoon'

    Returns
    -------
    View object

    """
    view = nglview.show_file(fp)
    setup_view(view, **kwargs)
    return view


def plot_frame(xp, top):
    """Create an MDTraj Trajectory object and return a nglview view.

    Parameters
    ----------
    xp : array-like
    top : topology

    Return
    ------
    nglview view object

    """

    mdtop = mdtraj.Topology.from_openmm(top)
    traj = mdtraj.Trajectory(xp / unit.nanometers, mdtop)
    view = nglview.show_mdtraj(traj)
    if len(xp) < 10000:
        view.add_ball_and_stick('all')
    view.center(zoom=True)
    return view


def plot_simulation_data(fp, x='time', y='energy', stride=None, rolling=None):
    """Plot simulation data."""
    mapper = {
        'step': '#"Step"',
        'time': 'Time (ps)',
        'potential': 'Potential Energy (kJ/mole)',
        'kinetic': 'Kinetic Energy (kJ/mole)',
        'energy': 'Total Energy (kJ/mole)',
    }
    df = pandas.read_csv(fp)
    if rolling:
        df = df.rolling(rolling, center=True).mean()
    if stride:
        df = df.iloc[::stride]
    xs = df[mapper[x]]
    ys = df[mapper[y]]
    return plt.plot(xs, ys, '-o')
