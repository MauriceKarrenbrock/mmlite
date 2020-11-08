# -*- coding: utf-8 -*-
"""Output class for reporters."""
import sys

from simtk.openmm.app.pdbreporter import PDBReporter
from simtk.openmm.app.statedatareporter import StateDataReporter

__all__ = ['add_reporters']


class Output:  # pylint: disable=too-few-public-methods
    """
    Helper class for reporters.

    Parameters
    ----------
    kind : str
        The reporter label, valid values are `pdb`, `data`, `screen`
    target : path or file-like
        The output file.
    dt : int
        Output frequency as number of time steps.

    """
    minimal_data = {'step': True, 'totalEnergy': True, 'temperature': True}
    full_data = {
        'step': True,
        'time': True,
        'potentialEnergy': True,
        'totalEnergy': True,
        'temperature': True
    }
    default_targets = {
        'pdb': 'traj.pdb',
        'data': 'data.csv',
        'screen': sys.stdout
    }

    def __init__(self, kind, dt, *, target=None):
        self.kind = kind
        self.dt = dt
        self.target = target or self.default_targets[kind]
        self._reporter = None

    @property
    def reporter(self):
        """Reporter object."""
        if self.kind == 'pdb':
            self._reporter = PDBReporter(self.target, self.dt)
        elif self.kind == 'data':
            self._reporter = StateDataReporter(self.target, self.dt,
                                               **self.full_data)
        elif self.kind == 'screen':
            self._reporter = StateDataReporter(sys.stdout, self.dt,
                                               **self.minimal_data)
        else:
            raise ValueError('Unkown reporter kind %s' % self.kind)
        return self._reporter


def add_reporters(simulation,
                  outs=(Output('pdb', 1), Output('data',
                                                 1), Output('screen', 100))):
    """Define the simulation reporters.

    Parameters
    ----------
    simulation : Simulation object.
    outs : list of outputs

    """

    simulation.reporters = []
    for out in outs:
        simulation.reporters.append(out.reporter)
