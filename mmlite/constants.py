# -*- coding: utf-8 -*-
"""Constant."""
# pylint: disable=no-member
import numpy as np
from simtk import unit

PI = np.pi
KB = unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA

# OpenMM constant for Coulomb interactions in OpenMM units
# (openmm/platforms/reference/include/SimTKOpenMMRealType.h)
# TODO: Replace this with an import from simtk.openmm.constants once available
ONE_4PI_EPS0 = 138.935456

# Standard-state volume for a single molecule in a box of size (1 L) / (avogadros number).
LITER = 1000.0 * unit.centimeters**3
STANDARD_STATE_VOLUME = LITER / (unit.AVOGADRO_CONSTANT_NA * unit.mole)
