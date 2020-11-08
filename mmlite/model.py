# -*- coding: utf-8 -*-
"""Model utils."""
from simtk import unit
from simtk.openmm import app


def empty_topology():
    """Empty topology."""
    return app.Topology()


def empty_positions():
    """Empty positions."""
    return unit.Quantity((), unit.angstroms)
