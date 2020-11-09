# -*- coding: utf-8 -*-
"""Helper functions/classes."""
# pylint: disable=protected-access,no-member
import logging
from pathlib import Path

import simtk.openmm as mm
from pdbfixer import PDBFixer
from simtk import unit
from simtk.openmm import app
from simtk.openmm.app.pdbfile import PDBFile
from simtk.unit.quantity import Quantity

from .simulation import simulation_energy

logger = logging.getLogger(__name__)


def create_langevin_context(system,
                            *,
                            temp=298.0 * unit.kelvin,
                            friction=91.0 / unit.picosecond,
                            ts=2.0 / unit.femtoseconds,
                            xp=None,
                            random_vp=True):
    """Create the context for a Langevin simulations from a system."""

    all_have_units = all(
        [isinstance(arg, Quantity) for arg in (temp, friction, ts)])
    if not all_have_units:
        raise ValueError

    integrator = mm.LangevinIntegrator(temp, friction, ts)
    # Create a Context for this integrator
    context = mm.Context(system, integrator)

    if xp is not None:  # Set the positions
        context.setPositions(xp)

    if random_vp:
        context.setVelocitiesToTemperature(temp)

    return context


def minimize(simulation, tol=10 * unit.kilojoule / unit.mole, max_iter=None):
    """Perform a local energy minimization on the system.

    Parameters
    ----------
    simulation : Simulation or Context object.
    tol : energy=10*kilojoules/mole
        The energy tolerance to which the system should be minimized
    max_iter : int=None
        The maximum number of iterations to perform.  If this is 0,
        Default: minimization is continued until the results converge.

    Returns
    -------
    Context or Simulation.

    """

    try:
        context = simulation.context
    except AttributeError:
        context = simulation

    max_iter = max_iter or 0
    minimize.info = {
    }  # store info on the minimization as a function attribute
    minimize.info['V0'] = simulation_energy(context)['potential']
    mm.LocalEnergyMinimizer.minimize(context, tol, maxIterations=max_iter)
    minimize.info['V1'] = simulation_energy(context)['potential']


def write_single_pdb(xp, top, target):
    """Dump coordinates to a pdb file `target`."""
    try:
        target = Path(target)
    except TypeError:
        if hasattr(target, 'write'):
            PDBFile.writeFile(top, xp, target)
        else:
            raise
    else:
        with open(target, 'w') as fp:
            PDBFile.writeFile(top, xp, fp)


def _write_to_file(file_name, a):
    with open(file_name, 'w') as fp:
        print(a, file=fp)


def serialize_system(context, system, integrator):
    """Save context info."""
    _write_to_file('system.xml', mm.XmlSerializer.serialize(system))
    _write_to_file('integrator.xml', mm.XmlSerializer.serialize(integrator))
    # pylint: disable=unexpected-keyword-arg, no-value-for-parameter
    state = context.getState(getPositions=True,
                             getVelocities=True,
                             getForces=True,
                             getEnergy=True,
                             getParameters=True,
                             enforcePeriodicBox=True)
    _write_to_file('state.xml', mm.XmlSerializer.serialize(state))


# pylint: disable=too-many-arguments
def fetch_pdb(pdb,
              chains='A',
              ff=('amber99sbildn.xml', 'tip3p.xml'),
              ph=7,
              pad=10 * unit.angstroms,
              nbonded=app.PME,
              constraints=app.HBonds,
              crystal_water=True):
    """
    Fetch, solvate and minimize a protein PDB structure.

    Parameters
    ----------
    pdb : str
        PDB Id.
    chains : str or list
        Chain(s) to keep in the system.
    ff : tuple of xml ff files.
        Forcefields for parametrization.
    ph : float
        pH value for adding missing hydrogens.
    pad: Quantity object
        Padding around macromolecule for filling box with water.
    nbonded : object
        The method to use for nonbonded interactions.  Allowed values are
        NoCutoff, CutoffNonPeriodic, CutoffPeriodic, Ewald, PME, or LJPME.
    constraints : object
        Specifies which bonds and angles should be implemented with
        constraints. Allowed values are None, HBonds, AllBonds, or HAngles.
    crystal_water : bool
        Keep crystal water.

    """

    # Load forcefield.
    logger.info('Retrieving %s from PDB...', pdb)
    ff = app.ForceField(*ff)

    # Retrieve structure from PDB.
    fixer = PDBFixer(pdbid=pdb)

    # Remove unselected chains.
    logger.info('Removing all chains but %s', chains)
    all_chains = [c.id for c in fixer.topology.chains()]
    fixer.removeChains(chainIds=set(all_chains) - set(chains))

    # Find missing residues.
    logger.info('Finding missing residues...')
    fixer.findMissingResidues()

    # Replace nonstandard residues.
    logger.info('Replacing nonstandard residues...')
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()

    # Add missing atoms.
    logger.info('Adding missing atoms...')
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()

    # Remove heterogens.
    logger.info('Removing heterogens...')
    fixer.removeHeterogens(keepWater=crystal_water)

    # Add missing hydrogens.
    logger.info('Adding missing hydrogens appropriate for pH %s', ph)
    fixer.addMissingHydrogens(ph)

    if nbonded in [app.PME, app.CutoffPeriodic, app.Ewald]:
        # Add solvent.
        logger.info('Adding solvent...')
        fixer.addSolvent(padding=pad)

    # Write PDB file.
    logger.info('Writing PDB file to "%s"...', '%s-pdbfixer.pdb' % pdb)
    app.PDBFile.writeFile(fixer.topology, fixer.positions,
                          open('%s-pdbfixer.pdb' % pdb, 'w'))

    # Create OpenMM System.
    logger.info('Creating OpenMM system...')
    system = ff.createSystem(fixer.topology,
                             nonbondedMethod=nbonded,
                             constraints=constraints,
                             rigidWater=True,
                             removeCMMotion=False)

    # Minimimze to update positions.
    logger.info('Minimizing...')
    integrator = mm.VerletIntegrator(1.0 * unit.femtosecond)
    context = mm.Context(system, integrator)
    context.setPositions(fixer.positions)
    mm.LocalEnergyMinimizer.minimize(context)
    # pylint: disable=unexpected-keyword-arg, no-value-for-parameter
    state = context.getState(getPositions=True)
    fixer.positions = state.getPositions()

    # Write final coordinates.
    logger.info('Writing PDB file to "%s"...', '%s-minimized.pdb' % pdb)
    with open('%s-minimized.pdb' % pdb, 'w') as fp:
        app.PDBFile.writeFile(fixer.topology, fixer.positions, fp)

    # Serialize final coordinates.
    logger.info('Serializing to XML...')
    serialize_system(context, system, integrator)
