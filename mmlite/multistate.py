# -*- coding: utf-8 -*-
"""(multistate) sampling utils."""
# pylint: disable=no-member, protected-access, import-error
import logging
import tempfile
from collections.abc import Sequence
from pathlib import Path

import mmdemux
import openmmtools as mmtools
import simtk.openmm as mm
from mmlite import Topography
from openmmtools import mcmc, multistate
from openmmtools.states import SamplerState, ThermodynamicState
from simtk import unit

logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO)


def propagator(timestep=1.0 * unit.femtoseconds, n_steps=1000):
    """Return a openmmtools mcmc move."""
    return mcmc.LangevinDynamicsMove(
        timestep=timestep,
        collision_rate=5.0 / unit.picoseconds,
        n_steps=n_steps,  # steps between multistate moves
        reassign_velocities=True,
        n_restart_attempts=6)


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
            temperatures,
            pressure=None,
            storage=None,
            target_state=0,
            metadata=None,
            **kwargs):
        """Initialize sampler from TestSystem object."""
        if not isinstance(temperatures, Sequence):  # a scalar
            temperatures = [temperatures]
        if not isinstance(temperatures[0], unit.Quantity):  # no units
            temperatures = [t * unit.kelvin for t in temperatures]
        thermodynamic_states = [
            ThermodynamicState(
                system=test.system,
                temperature=t,
                pressure=pressure,
            ) for t in temperatures
        ]
        sampler_states = SamplerState(positions=test.positions,
                                      box_vectors=test.default_box_vectors)
        self.create(thermodynamic_states,
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
        if not isinstance(sampler_states, (SamplerState, Sequence)):
            # as positions
            sampler_states = SamplerState(sampler_states)

        if storage is None:
            storage = tempfile.NamedTemporaryFile(delete=False).name + '.nc'
        self.reporter = self._define_reporter(storage, stride)

        if metadata is None:
            metadata = {}

        # set self.topography
        if isinstance(top, Topography):
            self.topography = top
        else:
            self.topography = Topography(top)

        self.ref_state = target_state

        sampler_full_name = mmtools.utils.typename(self.__class__)
        metadata['sampler_full_name'] = sampler_full_name
        metadata['topography'] = mmtools.utils.serialize(self.topography)
        metadata['reference_state'] = mmtools.utils.serialize(
            thermodynamic_states[target_state])

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
        return self.thermodynamic_states[self.ref_state]

    @property
    def reference_system(self):
        """System object for the reference state."""
        return self.reference_state.system

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
            mcmc_moves=mcmc.LangevinDynamicsMove(
                timestep=1.0 * unit.femtoseconds,
                collision_rate=5.0 / unit.picoseconds,
                n_steps=1000,  # steps between multistate moves
                reassign_velocities=True,
                n_restart_attempts=6),
            **kwargs):
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
            mcmc_moves=mcmc.LangevinDynamicsMove(
                timestep=1.0 * unit.femtoseconds,
                collision_rate=5.0 / unit.picoseconds,
                n_steps=1000,  # steps between multistate moves
                reassign_velocities=True,
                n_restart_attempts=6),
            **kwargs):
        super().__init__(number_of_iterations=number_of_iterations,
                         mcmc_moves=mcmc_moves,
                         **kwargs)
        self.reporter = None
        self.ref_state = None
        self.topology = None
