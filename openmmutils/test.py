# -*- coding: utf-8 -*-
"""Test systems."""
# pylint: disable=unused-import, too-few-public-methods
from abc import ABC, abstractmethod

from openmmtools.testsystems import TestSystem
from simtk.openmm.app.topology import Topology
from simtk.openmm.openmm import System
from simtk.unit.quantity import Quantity

test_register = {}


def register_class(cls, register):
    """Add a class to register."""
    register[cls.__name__] = cls
    return register


class TestType(type(ABC)):
    """Metaclass for entropy estimators."""
    def __new__(cls, name, bases, namespace, **kwargs):
        test_class = type.__new__(cls, name, bases, namespace, **kwargs)
        register_class(test_class, test_register)
        return test_class


class Test(ABC, metaclass=TestType):
    """
    Class for generic test system.

    Attributes
    ----------
    system : System object.
    positions : Quantity object.
    topology : Toplogy object.

    """
    def __init__(self, system, positions, topology):
        self._system = system
        self._positions = positions
        self._topology = topology
