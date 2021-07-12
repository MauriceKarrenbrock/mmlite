# -*- coding: utf-8 -*-
"""(multistate) sampling utils."""
# pylint: disable=no-member, protected-access, import-error
# pylint: disable=consider-using-with
import copy
import logging
import shutil
import tempfile
import time
from collections.abc import Sequence
from pathlib import Path

import mmdemux
import openmmtools as mmtools
import simtk.openmm as mm
import yank
from openmmtools import cache, mcmc, multistate
from openmmtools.states import SamplerState, ThermodynamicState
from simtk import unit

from mmlite import Topography

logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO)


def propagator(timestep=1.0 * unit.femtoseconds, n_steps=1000, platform=None):
    """Return a openmmtools mcmc move.

    Parameters
    ------------
    timestep : simtk.unit.Quantity, default=1.0 * unit.femtoseconds
    n_steps : int, default=1000
        steps between multistate moves
        therefore how many steps to do before attempting
        a replica swap
    platform : str, optional
        for default the openmmtools default will be used
        it is usually the fastest one that it can find
        it changes the global variable
        `openmmtools.cache.global_context_cache.platform`
        possible options are CUDA CPU Reference OpenCL

    Notes
    -----------
    reducing `n_steps` too much is going to drastically
    slow down the MD run
    """

    # ['Reference', 'CPU', 'CUDA', 'OpenCL']  # platforms
    if platform is not None:
        cache.global_context_cache.platform = mm.Platform.getPlatformByName(
            platform)

    return mcmc.LangevinDynamicsMove(
        timestep=timestep,
        collision_rate=5.0 / unit.picoseconds,
        n_steps=n_steps,  # steps between multistate moves
        reassign_velocities=True,
        n_restart_attempts=6)


def _check_restraint(restraint, topography=None):
    """Preprocess a ligand-receptor restraint.

    Parameters
    --------------
    restraint : yank.restraints.ReceptorLigandRestraint or simtk.unit.Quantity or True
    topography : mmlite.topography.Topography

    Returns
    -----------
    yank.restraints.Harmonic if a quantity or True was given as `restraint`
    or `restraint` if the imput was yank.restraints.ReceptorLigandRestraint
    """
    if not isinstance(restraint, yank.restraints.ReceptorLigandRestraint):
        if isinstance(restraint, unit.Quantity):
            k0 = restraint
        elif restraint is True:
            k0 = 120.0 * unit.kilojoule_per_mole / unit.nanometers**2
        else:
            raise ValueError('%r is not a valid restraint value' % restraint)
        if topography is None or 'ligand' not in topography:
            raise ValueError(
                'To apply a default restraint, a ligand must be defined')
        restraint = yank.restraints.Harmonic(
            spring_constant=k0,
            restrained_receptor_atoms=topography['receptor'],
            restrained_ligand_atoms=topography['ligand'])
    return restraint


class SamplerMixin:
    """Contains methods for derived multistate samplers.

    To define a derived multistate sampler from e.g. the ReplicaExchangeSampler

    ```
    class MySampler(SamplerMixin, multistate.ReplicaExchangeSampler):
        def __init__(self, **kwargs):
            # set defaults as explicit keyword arguments
            super().__init__(**kwargs)
            self.reporter = None
            self.ref_state = None
            self.topology = None
    ```

    """
    def from_testsystem(  # pylint: disable=too-many-arguments
            self,
            test,
            reference_state,
            thermodynamic_states,
            pressure=None,
            storage=None,
            target_state=0,
            metadata=None,
            **kwargs):
        """Initialize sampler from TestSystem object."""
        if not isinstance(thermodynamic_states, Sequence):  # a scalar
            thermodynamic_states = [thermodynamic_states]
        # check if temp or thermodynamic states or something else
        if not isinstance(thermodynamic_states[0], ThermodynamicState):
            # as temperatures
            temperatures = thermodynamic_states
            if not isinstance(temperatures[0], unit.Quantity):  # no units
                temperatures = [t * unit.kelvin for t in temperatures]
            thermodynamic_states = [
                ThermodynamicState(
                    system=test.system,
                    temperature=t,
                    pressure=pressure,
                ) for t in temperatures
            ]
        sampler_states = SamplerState(
            positions=test.positions,
            box_vectors=test.system.periodicBoxVectors)
        self.create(reference_state,
                    thermodynamic_states,
                    sampler_states,
                    test.topology,
                    target_state=target_state,
                    storage=storage,
                    metadata=metadata,
                    **kwargs)

    def from_pdb(self):
        """Initialize sampler from a single pdb structure."""
        raise NotImplementedError

    @staticmethod
    def _define_reporter(storage, stride=1):
        if isinstance(storage, multistate.MultiStateReporter):
            return storage
        return multistate.MultiStateReporter(storage,
                                             checkpoint_interval=stride)

    def create(  # pylint: disable=too-many-arguments
            self,
            reference_state,
            thermodynamic_states,
            sampler_states,
            top,
            target_state=0,
            initial_thermodynamic_states=None,
            unsampled_thermodynamic_states=None,
            storage=None,
            metadata=None,
            stride=1):
        """Create new multistate sampler simulation.

        Override MultistateSampler create.

        Parameters
        ----------
        thermodynamic_states : list of states.ThermodynamicState
            Thermodynamic states to simulate, where one replica is allocated
            per state.
            Each state must have a system with the same number of atoms.
        sampler_states : states.SamplerState or list
            One or more sets of initial sampler states.
            The number of replicas is taken to be the number of sampler states
            provided. If the sampler states do not have box_vectors attached
            and the system is periodic, an exception will be thrown.
        top : Topology or Topography object, optional
        target_state : int, optional
           The indef of the reference thermodynamic state. Defaults to 0.
        initial_thermodynamic_states : None or list or
            array-like of int of length len(sampler_states), optional,
            Default: None.
            Initial thermodynamic_state index for each sampler_state.
            If no initial distribution is chosen, ``sampler_states`` are
            distributed between the ``thermodynamic_states`` following these
            rules:

                * If ``len(thermodynamic_states) == len(sampler_states)``:
                  1-to-1 distribution
                * If ``len(thermodynamic_states) > len(sampler_states)``:
                  First and last state distributed first
                  Remaining ``sampler_states`` spaced evenly by index until
                  ``sampler_states`` are depleted.
                  If there is only one ``sampler_state``, then the only first
                  ``thermodynamic_state`` will be chosen
                * If ``len(thermodynamic_states) < len(sampler_states)``:
                  each ``thermodynamic_state`` receives an equal number of
                  ``sampler_states`` until there are insufficient number of
                  ``sampler_states`` remaining to give each
                  ``thermodynamic_state`` an equal number. Then the rules from
                  the previous point are followed.

        unsampled_thermodynamic_states : list of states.ThermodynamicState,
            optional, default=None
            These are ThermodynamicStates that are not propagated, but their
            reduced potential is computed at each iteration for each replica.
            These energy can be used as data for reweighting schemes (default
            is None).
        storage : str or instanced Reporter
            If str: the path to the storage file.
            Default checkpoint options from Reporter class are used.
            If Reporter: Uses the reporter options and storage path
            In the future this will be able to take a Storage class as well.
        metadata : dict, optional, default=None
           Simulation metadata to be stored in the file.

        """

        if isinstance(thermodynamic_states, ThermodynamicState):
            # a single state
            thermodynamic_states = [thermodynamic_states]
        if not isinstance(sampler_states,
                          (SamplerState, Sequence)):  # TODO: check this
            # as positions
            sampler_states = SamplerState(sampler_states)

        # Do not modify passed ref thermodynamic state.
        self._reference_thermodynamic_state = copy.deepcopy(reference_state)
        thermodynamic_state = copy.deepcopy(
            self._reference_thermodynamic_state)
        self._reference_system = thermodynamic_state.system

        if storage is None:
            storage = tempfile.NamedTemporaryFile(delete=False).name + '.nc'
        self.reporter = self._define_reporter(storage, stride)

        # set self.topography
        if isinstance(top, Topography):
            self.topography = top
        else:
            self.topography = Topography(top)

        self.ref_state = target_state

        if metadata is None:
            metadata = {}
        sampler_full_name = mmtools.utils.typename(self.__class__)
        metadata[
            'title'] = f'Created using {sampler_full_name} on {time.asctime(time.localtime())}'
        metadata['sampler_full_name'] = sampler_full_name
        metadata['topography'] = mmtools.utils.serialize(self.topography)
        metadata['reference_state'] = mmtools.utils.serialize(
            thermodynamic_state)  # the ref thermodynamic state

        super().create(
            thermodynamic_states=thermodynamic_states,
            sampler_states=sampler_states,
            storage=self.reporter,
            initial_thermodynamic_states=initial_thermodynamic_states,
            unsampled_thermodynamic_states=unsampled_thermodynamic_states,
            metadata=metadata)

    @property
    def reference_state(self):
        """System object for the reference state."""
        return self._reference_thermodynamic_state

    @property
    def reference_system(self):
        """System object for the reference state."""
        return self._reference_system

    @property
    def thermodynamic_states(self):
        """Thermodynamic states."""
        return self._thermodynamic_states

    @property
    def storage(self):
        """Path to storage file."""
        return self.reporter.filepath

    def demux(self,
              state_index=None,
              *,
              replica_index=None,
              to_file=None,
              **kwargs):
        """demux trajectories.

        Default: extract configurations for state_index = 0.

        """

        trj_file = Path(to_file) if to_file else Path('trj.nc')
        trj_dir = trj_file.parent.resolve()
        trj_dir.mkdir(parents=True, exist_ok=True)

        if (state_index is None) and (replica_index is None):
            state_index = self.ref_state
        top = kwargs.pop('top', None) or self.topology
        if not top:
            raise ValueError('Need a topology.')
        nc_path = kwargs.pop('nc_path', None) or self.reporter.filepath

        trj = mmdemux.extract_trajectory(ref_system=self.ref_system,
                                         top=top,
                                         nc_path=nc_path,
                                         state_index=state_index,
                                         replica_index=replica_index,
                                         to_file=trj_file,
                                         **kwargs)

        # save a ref pdb for topology
        out_dir = trj_file.parent
        filename = out_dir / 'system.pdb'
        trj[-1].save(str(filename))

        # save a ref pdb for topology
        out_dir = trj_file.parent
        filename = out_dir / 'system.pdb'
        trj[-1].save(str(filename))

        # save system as .xml
        serialized_system = mm.openmm.XmlSerializer.serialize(self.ref_system)
        with open('system.xml', 'w') as fp:
            print(serialized_system, file=fp)

        return trj


class SAMSSampler(SamplerMixin, multistate.SAMSSampler):  # pylint: disable=abstract-method
    """Sampler with multistate moves."""
    def __init__(
            self,
            number_of_iterations=1,  # total multistate moves
            mcmc_moves=None,
            **kwargs):
        if mcmc_moves is None:
            mcmc_moves = propagator()
        super().__init__(number_of_iterations=number_of_iterations,
                         mcmc_moves=mcmc_moves,
                         **kwargs)
        self.reporter = None
        self.ref_state = None
        self.topography = None


class ReplicaExchangeSampler(SamplerMixin, multistate.ReplicaExchangeSampler):  # pylint: disable=abstract-method
    """Sampler with multistate moves."""
    def __init__(
            self,
            number_of_iterations=1,  # total multistate moves
            mcmc_moves=None,
            **kwargs):
        if mcmc_moves is None:
            mcmc_moves = propagator()
        super().__init__(number_of_iterations=number_of_iterations,
                         mcmc_moves=mcmc_moves,
                         **kwargs)
        self.reporter = None
        self.ref_state = None
        self.topology = None


def _reference_compound_state(  # pylint: disable=too-many-locals
        reference_thermodynamic_state,
        top,
        region=None,
        restraint=None,
        ligand_atoms=None):
    """
    Return reference compound state.

    Parameters
    ----------
    reference_thermodynamic_state : ThermodynamicState object
    top : Topography or Topology object
    region : str or list(int)
        Atomic indices defining the alchemical region.
        a mdtraj selection string or a list of atom indexes (0 indexed)
    restraint : Quantity or yank LigandReceptorRestraint or False
        If ligand exists, restraint ligand and receptor movements.
        If True or a force constant, apply a Harmonic restraint.
        if None the function will add a default restraint if there is
        a ligand none otherwise
    ligand_atoms : str or list(int), optional
        a mdtraj selection string or a list of atom indexes (0 indexed)
        if given it will overwrite wathever may have been in the input
        `top` if it was a Topography.
        If it is not given it will be checked if something was defined in `top`
        if nothing was defined it will be taken for granted that there is no ligand

    Returns
    ---------
    reference compound state
    """

    if not isinstance(top, Topography):
        topography = Topography(top)
    else:
        topography = top

    if ligand_atoms:
        topography.ligand_atoms = ligand_atoms

    # If there is a ligand and restraint is None
    # Use the default
    if restraint is None and not topography.ligand_atoms:
        restraint = True

    thermodynamic_state = copy.deepcopy(reference_thermodynamic_state)

    if region is None:
        raise ValueError('An alchemical region is needed.')
    if region in topography:  # region is predefined
        alchemical_atoms = topography[region]
    else:
        alchemical_atoms = topography.select(region)

    # init an AlchemicalRegion object
    alchemical_region = mmtools.alchemy.AlchemicalRegion(
        alchemical_atoms=alchemical_atoms, alchemical_torsions=True)

    # Create an alchemically modified system using alchemical factory
    factory = mmtools.alchemy.AbsoluteAlchemicalFactory(
        consistent_exceptions=False)
    alchemical_system = factory.create_alchemical_system(
        thermodynamic_state.system, alchemical_region)

    # using the alchemical system, update the thermodynamic state and
    # init an alchemical state
    thermodynamic_state.system = alchemical_system
    alchemical_state = mmtools.alchemy.AlchemicalState.from_system(
        alchemical_system)

    restraint_state = None

    if restraint:
        restraint = _check_restraint(restraint, topography)

        # add the restraint force to the `System` of the thermodynamic state
        restraint.restrain_state(thermodynamic_state)
        # lambda_restraints is the strength of the restraint (betwenn 0 and 1)
        restraint_state = yank.restraints.RestraintState(lambda_restraints=1.0)

    # init the reference compund state (thermodynamic + alchemical)
    if restraint_state:
        composable_states = [alchemical_state, restraint_state]
    else:
        composable_states = [alchemical_state]

    compound_state = mmtools.states.CompoundThermodynamicState(
        thermodynamic_state=thermodynamic_state,
        composable_states=composable_states)

    return compound_state


def create_compound_states(reference_thermodynamic_state,
                           top,
                           protocol,
                           region=None,
                           restraint=None):
    """Return alchemically modified thermodynamic states.

    Parameters
    ----------
    reference_thermodynamic_state : ThermodynamicState object
    top : Topography or Topology object
    protocol : dict
        A dictionary ```{parameter_name: list_of_parameter_values}``` defining
        the protocol. All the parameter values list must have the same
        number of elements.
        the first value shall be 1. (reference state) and the last one
        the most scaled one.
        If for example you want to scale tosions on 8 replicas
        ```{'lambda_torsions' : list(numpy.logspace(0, -1, 8)))}```
        might be a reasonable starting point
    region : str or list
        Atomic indices defining the alchemical region.
    restraint : bool
        If ligand exists, restraint ligand and receptor movements.

    Returns
    ---------
    compound_states
    """

    compound_state = _reference_compound_state(reference_thermodynamic_state,
                                               top,
                                               region=region,
                                               restraint=restraint)

    # init the array of compound states
    compound_states = []
    protocol_keys, protocol_values = zip(*protocol.items())

    for state_id, state_values in enumerate(zip(*protocol_values)):
        compound_states.append(copy.deepcopy(compound_state))
        for lambda_key, lambda_value in zip(protocol_keys, state_values):
            if hasattr(compound_state, lambda_key):
                setattr(compound_states[state_id], lambda_key, lambda_value)
            else:
                raise AttributeError(
                    'CompoundThermodynamicState object does not '
                    f'have protocol attribute {lambda_key}')

    return compound_states


class SetupReplicaExchangeSimulation(object):  # pylint: disable=too-many-instance-attributes
    """A high level class to quickly set up a replica exchange simulation and run it

    Parameters
    --------------
    positions : openmm positions
    topology : openmm Topology
    system : openmm System
    alchemical_region : str or iterable(int)
        a mdtraj like selection string or an iterable of atom indexes that defines
        the alchemical region
    ligand_atoms : str or iterable(int), default=None
        a mdtraj like selection string or an iterable of atom indexes that defines
        the ligand. If omitted it means there is no ligand in the system
    restraint : openmm or yank restraint or simtk.Quantity False or None, default=None
        which kind of restraint to define between the protein and the ligand
        if a quantity is given an harmonic poteintial with K=`restraint` will be used
        if left None, if it is not a protein ligand system nothing happens if it is a protein
        ligand a default harmonic restraint is used with K=120kjmol/nanometers**2
        If False no restraint is used
    unrestrained_system : openmm System, optional, default=None
        an openmm system that is completely unrestrained (restrains=None, rigid_water=False)
        it is necessary if you want to use parmed to write a file for a different MD program
        (gromacs top file for example)
    replica_protocol : dict(thing_to_scale=list(float)), default=None
        the scaling values for the various replicas. If left None
        the `get_default_replica_protocol` method will be used
    temperature : simtik.Quantity, default=298.15*unit.kelvin
    pressure : simtik.Quantity, default=1.0*unit.atmosphere
    timestep : simtik.Quantity, default=2.0 * unit.femtoseconds
    number_of_iterations : int, default=1
        the number of attempted replica swaps
    steps_between_iterations : int, default=1000
        the number of MD steps between each replica swap attempt
    mcmc_moves : default=openmmtools.mcmc.LangevinDynamicsMove(
        collision_rate=5.0 / unit.picoseconds,
        reassign_velocities=True,
        n_restart_attempts=6)
    metadata : dict, default=None
        metadata to store in `storage`
    storage : str or path, default='trj.nc'
        the storage file in which the MD run will be written
    platform : str, optional, default=None
        for default the openmmtools default will be used
        it is usually the fastest one that it can find
        it changes the global variable
        `openmmtools.cache.global_context_cache.platform`
        possible options are CUDA CPU Reference OpenCL
    verbose=True,
    sampler_class default=ReplicaExchangeSampler,
    extend : bool, default=False
        if True and `storage` already exists instead of backing
        it up and restarting fresh the old `storage` will be used

    Methods
    -----------
    run(equilibration=0, extend=None, minimize=False)
        sets up and runs the simulation
    get_default_replica_protocol(n_replicas=8, base=0.2)
        static method to get a geometrical progression of
        scalig values for the torsional hamiltonian
    """
    def __init__(  # pylint: disable=too-many-arguments, too-many-locals
            self,
            positions,
            topology,
            system,
            alchemical_region,
            ligand_atoms=None,
            restraint=None,
            unrestrained_system=None,
            replica_protocol=None,
            temperature=298.15 * unit.kelvin,
            pressure=1.0 * unit.atmosphere,
            timestep=2.0 * unit.femtoseconds,
            number_of_iterations=1,
            steps_between_iterations=1000,
            mcmc_moves=None,
            metadata=None,
            storage='trj.nc',
            platform=None,
            verbose=True,
            sampler_class=ReplicaExchangeSampler,
            extend=False):

        self.positions = positions
        self.topology = topology
        self.system = system
        self.alchemical_region = alchemical_region
        self.ligand_atoms = ligand_atoms
        self.restraint = restraint
        self.unrestrained_system = unrestrained_system

        if replica_protocol is None:
            replica_protocol = self.get_default_replica_protocol()
        self.replica_protocol = replica_protocol
        self.temperature = temperature
        self.pressure = pressure
        self.timestep = timestep
        self.number_of_iterations = number_of_iterations
        self.steps_between_iterations = steps_between_iterations

        if mcmc_moves is None:
            mcmc_moves = propagator(timestep=timestep,
                                    n_steps=steps_between_iterations,
                                    platform=platform)
        self.mcmc_moves = mcmc_moves

        if metadata is None:
            metadata = {}
        self.metadata = metadata

        self.storage = Path(storage)
        # backup old storages if it doesn't have to be extended
        if self.storage.exists() and not extend:
            shutil.move(str(self.storage), self._get_backup_name(self.storage))
        self.metadata['storage'] = self.storage

        self.platform = platform

        if self.unrestrained_system:
            self.metadata['unrestrained_system'] = self.unrestrained_system

        if verbose:
            logging.getLogger('openmmtools.multistate').setLevel(logging.DEBUG)

        self.sampler_class = sampler_class
        self.metadata['sampler_class'] = self.sampler_class

    @staticmethod
    def get_default_replica_protocol(n_replicas=8, base=0.2):
        """
        Parameters
        ------------
        n_replicas : int, default=8
        base : float, default=0.2

        Returns
        ------------
        {'lambda_torsions' : [`base`**(i / (`n_replicas`-1)) for i in range(`n_replicas`)]}
        """
        scaling_values = [
            base**(i / (n_replicas - 1)) for i in range(n_replicas)
        ]

        return {'lambda_torsions': scaling_values}

    @staticmethod
    def _get_backup_name(file_name):
        """private"""
        file_name = str(file_name)

        i = 1

        while Path(file_name + f'({i})').exists():
            i += 1

        return file_name + f'({i})'

    def _setup_sampler(self):
        """private"""
        reference_thermodynamic_state = mmtools.states.ThermodynamicState(
            system=self.system,
            temperature=self.temperature,
            pressure=self.pressure)
        thermodynamic_states = create_compound_states(
            reference_thermodynamic_state,
            self.topology,
            self.replica_protocol,
            region=self.alchemical_region,
            restraint=self.restraint)

        try:
            sampler = self.metadata['sampler_class'].from_storage(
                self.metadata['storage'])
        except FileNotFoundError:

            class Dummy():  # pylint: disable=too-few-public-methods
                """Dummy class"""
                def __init__(self, system, positions, topology):
                    self.system = system
                    self.positions = positions
                    self.topology = topology

            test = Dummy(self.system, self.positions, self.topology)
            sampler = self.metadata['sampler_class'](
                number_of_iterations=0,
                mcmc_moves=self.mcmc_moves,
                online_analysis_interval=None)

            sampler.from_testsystem(
                test,
                reference_state=reference_thermodynamic_state,
                thermodynamic_states=thermodynamic_states,
                pressure=self.pressure,
                stride=self.steps_between_iterations,
                storage=self.storage,
                metadata=self.metadata)

        return sampler

    def run(self, equilibration=0, extend=None, minimize=False):
        """Sets up and runs the simulation

        Parameters
        ------------
        equilibration : int, default=0
            the number of equilibration steps to do before
            the replica exchange
        extend : int, optional, default=None
            if given will run the simulation for
            `extend` number of iterations
            useful when you want to extend the simulation over
            `self.number_of_iterations`
        minimize : bool, default=False
            if True a energy minimization will be done before the
            equilibration
        """
        if self.platform is not None:
            cache.global_context_cache.platform = mm.Platform.getPlatformByName(
                self.platform)

        if extend is None:
            extend = self.number_of_iterations

        sampler = self._setup_sampler()

        if minimize:
            sampler.minimize()

        if equilibration > 0:
            sampler.equilibrate(equilibration)
        if extend > 0:
            sampler.extend(extend)
