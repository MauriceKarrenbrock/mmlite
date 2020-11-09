# -*- coding: utf-8 -*-
"""Plotters."""
# pylint: disable=no-member
import matplotlib.pyplot as plt
import mdtraj
import nglview
import pandas
from simtk import unit

nglview_defaults = [
    {
        'type': 'cartoon',
        'params': {
            'sele': 'protein'
        }
        # }, {
        # 'type': 'surface',
        # 'params': {
        # 'sele': 'protein',
        # 'opacity': 0.1
        # }
    },
    {
        'type': 'ball+stick',
        'params': {
            'sele': 'hetero'
        }
    }
]


def plot_traj(fp,
              stride=None,
              atom_indices=None,
              ball_and_stick=False,
              water=False):
    """Return a trajectory view."""
    traj = mdtraj.load(fp, stride=stride, atom_indices=atom_indices)
    view = nglview.show_mdtraj(traj)
    view.representations = nglview_defaults
    if ball_and_stick:
        try:
            ball_and_stick = ball_and_stick.split()
        except AttributeError:
            pass
        view.representations.extend([{
            'type': 'ball+stick',
            'params': {
                'sele': x
            }
        } for x in ball_and_stick])
    if water:
        view.representations.append({
            'type': 'ball+stick',
            'params': {
                'sele': 'water'
            }
        })
    # print(view.representations)
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
