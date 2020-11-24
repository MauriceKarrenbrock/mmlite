# -*- coding: utf-8 -*-
# pylint: disable=no-member
"""Generic simulation defaults."""
from simtk import unit

ewald_error_tolerance = 1.e-5
cutoff_distance = 10.0 * unit.angstrom
switch_width = 1.5 * unit.angstrom
friction = 91.0 / unit.picosecond
temperature = 298.0 * unit.kelvin
time_step = 1.0 * unit.femtoseconds
integrator = 'VerletIntegrator'
