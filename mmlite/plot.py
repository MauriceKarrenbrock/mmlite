# -*- coding: utf-8 -*-
"""Plotters."""
# pylint: disable=no-member
import copy

import matplotlib.pyplot as plt
import mdtraj
import nglview
import pandas
import simtk.openmm as mm
import yank
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
    'hetero': {
        'type': 'ball+stick',
        'params': {
            'sele': 'hetero'
        }
    },
    'sugar': {
        'type': 'ball+stick',
        'params': {
            'sele': 'sugar'
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
            'color': 'blue',
            'opacity': 0.1
        }
    },
}


def setup_view(view, top=None, **kwargs):
    """
    Setup view representations.

    Parameters
    ----------
    view : NGLWidget object
    trj : mdtraj Trajectory

    Returns
    -------
    NGLWidget object

    """
    # colors = ['green', 'red', 'blue']
    colors = {
        1: ['#ff727e'],
        2: '#7eff72 #ff727e'.split(),
        3: '#728aff #fff872 #ff727e'.split(),
        4: '#7eff72 #728aff #ffb172 #ff727e'.split()
    }
    reps = copy.deepcopy(default_representations)

    for key, val in kwargs.items():
        if isinstance(val, str):  # set or override defaults
            reps[key] = {'type': val, 'params': {'sele': key}}
        if not val:  # remove if in defaults
            if key in reps:
                reps.pop(key)
    view.representations = list(reps.values())

    if top:
        if isinstance(top, yank.Topography):
            topography = top
            topology = top.topology
            n_regions = len(topography.regions) - 1
            if n_regions > 4:
                raise ValueError('Cant show > 4 regions')
            if n_regions > 0:
                cls = list(colors[n_regions])
                for name, selection in topography.regions.items():
                    if name != 'default':
                        view.add_representation('ball+stick',
                                                selection=selection,
                                                color=cls.pop())
        else:
            topology = top
        n_protein_atoms = len(topology.select('protein'))
        if n_protein_atoms < 200:
            reps['protein'] = {
                'type': 'ball+stick',
                'params': {
                    'sele': 'protein'
                }
            }

    view.camera = 'orthographic'
    view.center(zoom=True)


def show_mdtraj(fp, stride=None, atom_indices=None, top=None, **kwargs):
    """
    Return a trajectory view.

    Parameters
    ----------
    fp : filename or list of filenames
    stride : int, optional
        Read every stride-th frame
    atom_indices : array, optional
        Read only a subset of the atoms coordinates from the file
    top : {str, Trajectory, Topology}
    kwargs : dict
        Molecular representation of selections, e.g. protein='cartoon'

    Returns
    -------
    View object

    """
    load_args = dict(stride=stride, atom_indices=atom_indices)
    if isinstance(top, mm.app.Topology):
        top = mdtraj.Topology.from_openmm(top)
    if top:
        load_args['top'] = top

    traj = mdtraj.load(fp, **load_args)
    view = nglview.show_mdtraj(traj)
    setup_view(view, top=top, **kwargs)  # TODO: fix
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
