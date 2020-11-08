# -*- coding: utf-8 -*-
"""Plotters."""
# pylint: disable=no-member
import matplotlib.pyplot as plt
import mdtraj
import nglview
import pandas
from simtk import unit


def pdb_traj(fp):
    """Return a trajectory view."""
    return nglview.show_mdtraj(mdtraj.load(fp))


def frame(xp, top):
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


def simulation_data(fp, x='time', y='potential', stride=None, rolling=None):
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
