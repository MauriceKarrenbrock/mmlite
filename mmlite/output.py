# -*- coding: utf-8 -*-
"""Output class for reporters."""
import sys
from pathlib import Path

from simtk.openmm.app.pdbreporter import PDBReporter
from simtk.openmm.app.statedatareporter import StateDataReporter

__all__ = ['add_reporters']


def add_reporters(
    simulation,
    outs='traj.pdb data.csv screen'.split(),
    freqs=(1, 1, 100),
    screen='step totalEnergy temperature'.split(),
    data='step time potentialEnergy totalEnergy temperature'.split()):
    """Define the simulation reporters.

    Parameters
    ----------
    simulation : Simulation object.
    outs : list
        Output files.
    freqs : list
        Frequencies in time-steps.
    screen : list
        Quantities for stdout.
    screen : list
        Quantities for data file.

    """

    simulation.reporters = []
    for fp, dt in zip(outs, freqs):
        if fp in 'screen stdout'.split():  # to stdout
            reporter = StateDataReporter(sys.stdout, dt,
                                         **{q: True
                                            for q in screen})
            simulation.reporters.append(reporter)
            continue
        try:
            fp = Path(fp)
        except TypeError as e:
            if hasattr(fp, 'write'):
                reporter = StateDataReporter(fp, dt,
                                             **{q: True
                                                for q in screen})
            else:
                raise ValueError('Not a valid file: %r' % fp) from e
        else:  # file path
            if fp.suffix == '.pdb':
                reporter = PDBReporter(str(fp), dt)
            elif fp.suffix == '.csv':
                reporter = StateDataReporter(str(fp), dt,
                                             **{q: True
                                                for q in data})
            simulation.reporters.append(reporter)
