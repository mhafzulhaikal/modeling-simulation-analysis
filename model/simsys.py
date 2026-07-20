"""Plug-and-play process simulation scenarios.

Provides three interchangeable simulation levels that share a common
``run(T, U, X0)`` interface and return a unified ``SimResult``:

  =====================  =====================================================
  Class                  Description
  =====================  =====================================================
  DynamicSimulation      Plant model only — all inputs supplied directly.
                         Use for model verification and open-loop step tests.
  OpenLoopSimulation     Plant + actuators (no setpoint / controller).
                         Use for actuator tuning and plant identification.
  ClosedLoopSimulation   All control elements wired via ``ct.interconnect``.
                         Use for controller design and performance analysis.
  =====================  =====================================================

Signal flow
-----------
::

    Dynamic:
        U (plant inputs directly) ──► Plant ──► states Y

    Open-loop:
        U_disturbance ──────────────────────────────────► Plant ──► Y
        U_M [%] ──► Actuator ──► F [m³/s] ─────────────►

    Closed-loop (per loop):
        SP ──► SetPoint ──► R[%] ──► Controller ──► M[%] ──► Actuator ──► F ──► Plant ──► Y
                                          ▲                                              │
                                          └── C[%] ◄── Sensor ◄── PV (plant state) ◄───┘
        Kc, tauI, tauD (external inputs, allow gain scheduling during simulation)

Plug-and-play usage
-------------------
All three simulation objects share the same run interface::

    # 1. Build elements once
    plant  = BiodieselPlant(**PROCESS_PARAMS)
    fcv    = ActuatorSystem('FCV_oil', tauV=12.0, f_max=6.59e-4)
    ctrl   = ControllerSystem('LC_100', bias=50.0)
    sp     = SetPointSystem('LSP_100', hi=3.0, low=0.0)
    sensor = SensorTransmitterSystem('LT_100', hi=3.0, low=0.0, tauT=0.0)

    # 2. Swap simulation levels — plant/elements stay the same
    sim_dyn  = DynamicSimulation(plant)
    sim_ol   = OpenLoopSimulation(plant, actuators={'f_oil': fcv})
    sim_cl   = ClosedLoopSimulation(plant, loops=[
        ControlLoop(mv='f_oil', pv='h',
                    actuator=fcv, controller=ctrl,
                    setpoint=sp,  sensor=sensor),
    ])

    # 3. Same run interface across all levels
    result = sim_cl.run(T, U, X0)
    result['h']          # level trajectory
    result['LC_100.M']   # controller output

Reproducibility
---------------
``SimResult`` stores all signal names alongside the data, making it
self-documenting.  Use ``sim.input_names``, ``sim.state_names``, and
``sim.output_names`` to inspect the signal layout before building U and X0.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import control as ct
import numpy as np

from model.actsys import ActuatorSystem
from model.ctrlsys import ControllerSystem
from model.plant import DynamicPlant
from model.spsys import SetPointSystem
from model.stsys import SensorTransmitterSystem

# =============================================================================
# Simulation result container
# =============================================================================


class SimResult:
    """Structured, self-documenting container for simulation output.

    Wraps ``ct.input_output_response`` output and provides named access to
    every signal — making results easy to reproduce, plot, and share.

    Attributes
    ----------
    t : ndarray, shape (n_steps,)
        Simulation time vector [s].
    y : ndarray, shape (n_outputs, n_steps)
        System output trajectories, in ``output_names`` order.
    x : ndarray, shape (n_states, n_steps)
        State trajectories, in ``state_names`` order.
    input_names : list of str
        Ordered external input names (matches columns of U passed to ``run()``).
    output_names : list of str
        Ordered output signal names (matches rows of ``y``).
    state_names : list of str
        Ordered state names (matches rows of ``x``).
    simulation_type : str
        One of ``'dynamic'``, ``'open_loop'``, ``'closed_loop'``.

    Examples
    --------
    >>> result = sim.run(T, U, X0)
    >>> result['h']           # level state trajectory
    >>> result['LC_100.M']    # controller output trajectory
    >>> result.t              # time vector
    """

    def __init__(
        self,
        response,
        *,
        state_names: list[str],
        output_names: list[str],
        input_names: list[str],
        simulation_type: str,
    ) -> None:
        self.t = np.asarray(response.t)
        self.y = np.asarray(response.y)
        self.x = np.asarray(response.x)
        self.state_names = list(state_names)
        self.output_names = list(output_names)
        self.input_names = list(input_names)
        self.simulation_type = str(simulation_type)
        self._response = response  # raw ct response for advanced use

    # --- Named signal access -------------------------------------------------

    def get_output(self, name: str) -> np.ndarray:
        """Return time trajectory of an output signal by name."""
        if name not in self.output_names:
            raise KeyError(f"Output '{name}' not found.\n  Available: {self.output_names}")
        return self.y[self.output_names.index(name)]

    def get_state(self, name: str) -> np.ndarray:
        """Return time trajectory of a state by name."""
        if name not in self.state_names:
            raise KeyError(f"State '{name}' not found.\n  Available: {self.state_names}")
        return self.x[self.state_names.index(name)]

    def __getitem__(self, name: str) -> np.ndarray:
        """Access any signal by name — outputs first, then states."""
        if name in self.output_names:
            return self.get_output(name)
        if name in self.state_names:
            return self.get_state(name)
        raise KeyError(
            f"Signal '{name}' not found.\n"
            f'  Outputs : {self.output_names}\n'
            f'  States  : {self.state_names}'
        )

    def __repr__(self) -> str:
        return (
            f'SimResult('
            f"type='{self.simulation_type}', "
            f't=[{self.t[0]:.1f}, {self.t[-1]:.1f}] s, '
            f'steps={len(self.t)}, '
            f'states={len(self.state_names)}, '
            f'outputs={len(self.output_names)})'
        )


# =============================================================================
# Control loop wiring specification
# =============================================================================


@dataclass
class ControlLoop:
    """Signal wiring for one feedback control loop.

    Declares which plant MV (manipulated variable) is driven by which
    actuator, which variable is measured by the sensor, and how the
    controller and setpoint elements are wired.

    Parameters
    ----------
    mv : str
        Plant ``INPUT_NAMES`` entry that is the manipulated variable
        (e.g., ``'f_oil'``).  This is what the actuator drives.
    pv : str
        The controlled/measured variable.  Meaning depends on ``pv_source``:

        - ``pv_source='plant'`` (default): a plant ``STATE_NAMES`` entry
          (e.g., ``'h'``, ``'T'``).  Sensor measures a reactor state.
        - ``pv_source='actuator'``: the port label on the actuator whose
          output is the measured PV (always ``'F'`` for ActuatorSystem).
          Use this for **flow control** loops where the transmitter
          measures the control valve output flow.

    actuator : ActuatorSystem
        Converts controller output M [%] → flow F [m³/s].
    controller : ControllerSystem
        Generates M from setpoint R and measurement C.
    setpoint : SetPointSystem
        Converts SP (engineering units) → normalized R [%].
    sensor : SensorTransmitterSystem
        Converts the PV signal → normalized C [%].
    pv_source : {'plant', 'actuator'}, optional
        Where the sensor reads its PV from.  Default: ``'plant'``.

    Examples
    --------
    Level/temperature control — PV is a plant state::

        loop = ControlLoop(
            mv='f_FAME', pv='h',
            actuator=..., controller=..., setpoint=..., sensor=...,
        )

    Flow control — PV is the actuator valve flow output::

        loop = ControlLoop(
            mv='f_oil', pv='F',
            pv_source='actuator',
            actuator=..., controller=..., setpoint=..., sensor=...,
        )
    """

    mv: str
    pv: str
    actuator: ActuatorSystem
    controller: ControllerSystem
    setpoint: SetPointSystem
    sensor: SensorTransmitterSystem
    pv_source: str = 'plant'  # 'plant' or 'actuator'

    _VALID_PV_SOURCES = ('plant', 'actuator')

    def __post_init__(self) -> None:
        if self.pv_source not in self._VALID_PV_SOURCES:
            raise ValueError(
                f'ControlLoop.pv_source must be one of {self._VALID_PV_SOURCES}, '
                f"got '{self.pv_source}'."
            )
        _check = [
            ('actuator', self.actuator, 'F', 'OUTPUT_NAMES'),
            ('actuator', self.actuator, 'M', 'INPUT_NAMES'),
            ('controller', self.controller, 'M', 'OUTPUT_NAMES'),
            ('controller', self.controller, 'R', 'INPUT_NAMES'),
            ('controller', self.controller, 'C', 'INPUT_NAMES'),
            ('setpoint', self.setpoint, 'R', 'OUTPUT_NAMES'),
            ('sensor', self.sensor, 'C', 'OUTPUT_NAMES'),
            ('sensor', self.sensor, 'PV', 'INPUT_NAMES'),
        ]
        for role, elem, port, attr in _check:
            if port not in getattr(elem, attr):
                raise ValueError(
                    f'ControlLoop: {role} ({type(elem).__name__}) must have '
                    f"port '{port}' in {attr}."
                )

    @property
    def tag(self) -> str:
        """Short identifier for this loop — the controller instrument tag."""
        return self.controller.system.name


# =============================================================================
# Abstract base class
# =============================================================================


class ProcessSimulation(ABC):
    """Abstract base for interchangeable process simulation scenarios.

    All concrete subclasses share the same ``run(T, U, X0)`` interface,
    so switching between dynamic, open-loop, and closed-loop simulation
    requires only changing the simulation object — not the plant or elements.

    Subclassing contract
    --------------------
    1. Call ``super().__init__(plant)`` from ``__init__``.
    2. Implement ``_build_system()`` to assemble and return the
       ``ct.NonlinearIOSystem`` (or interconnected system).
    3. Declare ``input_names``, ``output_names``, ``state_names`` as
       properties, returning lists in the order the signals appear in U/Y/X.

    The base class provides:

    - ``run(T, U, X0)``    — run and return ``SimResult``
    - ``make_X0(**kwargs)`` — build initial state vector by name
    - ``make_U(T, **kwargs)``  — build input matrix by name (constant or array)
    - ``__repr__``

    Attributes
    ----------
    plant : DynamicPlant
        The process plant model.
    system : ct.NonlinearIOSystem
        The fully assembled simulation system.
    """

    def __init__(self, plant: DynamicPlant) -> None:
        self.plant = plant
        self.system = self._build_system()

    # --- Abstract interface --------------------------------------------------

    @abstractmethod
    def _build_system(self) -> ct.NonlinearIOSystem:
        """Assemble and return the interconnected simulation system.

        Called once during ``__init__``.  The returned system is stored in
        ``self.system`` and is used by every call to ``run()``.
        """
        ...

    @property
    @abstractmethod
    def input_names(self) -> list[str]:
        """Ordered list of external input signal names.

        These correspond to the rows of the U matrix passed to ``run()``.
        Use ``make_U(T, **kwargs)`` to build U from named signals.
        """
        ...

    @property
    @abstractmethod
    def output_names(self) -> list[str]:
        """Ordered list of output signal names in the ``SimResult.y`` matrix."""
        ...

    @property
    @abstractmethod
    def state_names(self) -> list[str]:
        """Ordered list of state signal names in the ``SimResult.x`` matrix.

        Plant states use short names (e.g., ``'h'``, ``'T'``).
        Control element states use instrument-tag prefix
        (e.g., ``'FCV_100.vp'``, ``'LC_100.I_state'``).
        """
        ...

    @property
    @abstractmethod
    def simulation_type(self) -> str:
        """Short string identifying the simulation scenario."""
        ...

    # --- Shared helpers ------------------------------------------------------

    def make_X0(self, **kwargs) -> np.ndarray:
        """Build the initial state vector from named state values.

        Unspecified states default to zero.  Matches by exact name first,
        then by short name (last component after ``.``) when unambiguous.

        Parameters
        ----------
        **kwargs
            State name → initial value mappings.
            State names are listed in ``self.state_names``.

        Returns
        -------
        ndarray, shape (n_states,)

        Examples
        --------
        >>> X0 = sim.make_X0(h=1.5, T=338.15, T_coolant=310.0)
        >>> X0 = sim.make_X0(**{'FCV_100.vp': 50.0, 'h': 1.5})
        """
        X0 = np.zeros(len(self.state_names))
        for name, val in kwargs.items():
            idx = self._resolve_state(name)
            X0[idx] = float(val)
        return X0

    def make_U(self, T: np.ndarray, **kwargs) -> np.ndarray:
        """Build the input matrix from named constant or time-varying signals.

        Unspecified inputs default to zero.

        Parameters
        ----------
        T : ndarray, shape (n_steps,)
            Simulation time vector (used only for shape).
        **kwargs
            Input name → scalar value or array of shape (n_steps,).
            Input names are listed in ``self.input_names``.

        Returns
        -------
        ndarray, shape (n_inputs, n_steps)

        Examples
        --------
        >>> U = sim.make_U(T, f_oil=6.59e-4, T_coolant_in=298.15)
        >>> U = sim.make_U(T, SP_h=np.where(T < 600, 1.5, 2.0))  # step change
        """
        n_steps = len(T)
        U = np.zeros((len(self.input_names), n_steps))
        for name, val in kwargs.items():
            if name not in self.input_names:
                raise KeyError(f"Input '{name}' not found.\n  Available: {self.input_names}")
            idx = self.input_names.index(name)
            if np.isscalar(val):
                U[idx, :] = float(val)  # type: ignore
            else:
                arr = np.asarray(val, dtype=float)
                if arr.shape != (n_steps,):
                    raise ValueError(
                        f"Input '{name}': expected shape ({n_steps},), got {arr.shape}."
                    )
                U[idx, :] = arr
        return U

    def run(
        self,
        T: np.ndarray,
        U: np.ndarray,
        X0: np.ndarray,
        *,
        method: str = 'LSODA',
        **kwargs,
    ) -> SimResult:
        """Run the simulation and return a self-documenting ``SimResult``.

        Parameters
        ----------
        T : ndarray, shape (n_steps,)
            Simulation time vector [s].
        U : ndarray, shape (n_inputs, n_steps)
            External input matrix.  Build with ``make_U(T, **named_inputs)``
            to avoid ordering mistakes.
        X0 : ndarray, shape (n_states,)
            Initial state vector.  Build with ``make_X0(**named_states)``.
        method : str, optional
            ODE solver passed to ``solve_ivp``.  Default: ``'LSODA'``
            (stiff-aware; recommended for chemical reactor models).
        **kwargs
            Forwarded to ``ct.input_output_response``.

        Returns
        -------
        SimResult
            Use ``result['signal_name']`` to access any signal by name.
        """
        response = ct.input_output_response(
            self.system,
            T,
            U,  # type: ignore
            X0,  # type: ignore
            solve_ivp_kwargs={'method': method},
            **kwargs,
        )
        return SimResult(
            response,
            state_names=self.state_names,
            output_names=self.output_names,
            input_names=self.input_names,
            simulation_type=self.simulation_type,
        )

    def __repr__(self) -> str:
        return (
            f'{type(self).__name__}('
            f'plant={type(self.plant).__name__}, '
            f'inputs={len(self.input_names)}, '
            f'states={len(self.state_names)}, '
            f'outputs={len(self.output_names)})'
        )

    # --- Internal helpers ----------------------------------------------------

    def _resolve_state(self, name: str) -> int:
        """Return state index, matching by full name or unique short name."""
        # 1. Exact match
        if name in self.state_names:
            return self.state_names.index(name)
        # 2. Short-name match (last part after '.', if any)
        short_matches = [
            i
            for i, s in enumerate(self.state_names)
            if (s.rsplit('.', 1)[-1] if '.' in s else s) == name
        ]
        if len(short_matches) == 1:
            return short_matches[0]
        if len(short_matches) > 1:
            ambiguous = [self.state_names[i] for i in short_matches]
            raise KeyError(
                f"State short name '{name}' is ambiguous: matches {ambiguous}.\n"
                f"Use the full name (e.g., '{ambiguous[0]}')."
            )
        raise KeyError(f"State '{name}' not found.\n  Available: {self.state_names}")

    @staticmethod
    def _validate_unique_names(syslist: list, context: str) -> None:
        """Raise if any two systems in syslist share the same name."""
        seen: dict[str, str] = {}
        for sys in syslist:
            if sys.name in seen:
                raise ValueError(
                    f"{context}: duplicate system name '{sys.name}'. "
                    'Each instrument element must have a unique name tag.'
                )
            seen[sys.name] = sys.name


# =============================================================================
# Level 1 — Dynamic (plant only)
# =============================================================================


class DynamicSimulation(ProcessSimulation):
    """Plant-only simulation — all inputs supplied directly by the user.

    No control elements are involved.  Every plant input (disturbances and
    manipulated variables) must be provided in the U matrix.

    Use this level for:

    - Model verification against experimental data
    - Open-loop step response identification (FOPDT fitting)
    - Steady-state sensitivity analysis
    - Debugging plant model equations

    Input order: ``plant.INPUT_NAMES`` exactly (disturbances then MVs).
    Output order: ``plant.STATE_NAMES`` exactly.

    Parameters
    ----------
    plant : DynamicPlant
        The process plant model.

    Examples
    --------
    >>> sim = DynamicSimulation(plant)
    >>> U   = sim.make_U(T, f_oil=6.59e-4, f_MeOH=3.11e-4,
    ...                      c_TG_in=0.55, T_oil=298.15)
    >>> X0  = sim.make_X0(h=1.5, T=338.15, T_coolant=310.0)
    >>> res = sim.run(T, U, X0)
    >>> res['T']   # reactor temperature [K]
    """

    @property
    def simulation_type(self) -> str:
        return 'dynamic'

    def _build_system(self) -> ct.NonlinearIOSystem:
        return self.plant.system  # no assembly needed

    @property
    def input_names(self) -> list[str]:
        return list(self.plant.INPUT_NAMES)

    @property
    def output_names(self) -> list[str]:
        return list(self.plant.STATE_NAMES)

    @property
    def state_names(self) -> list[str]:
        return list(self.plant.STATE_NAMES)


# =============================================================================
# Level 2 — Open-loop (plant + actuators, no feedback)
# =============================================================================


class OpenLoopSimulation(ProcessSimulation):
    """Plant + actuators simulation — no setpoint station or controller.

    Actuators convert controller-output signals M [%] to volumetric flow
    rates F [m³/s].  Plant inputs that are NOT assigned an actuator are
    treated as disturbances and passed through directly.

    Use this level for:

    - Actuator valve sizing and time-constant tuning
    - Open-loop plant identification with realistic actuator dynamics
    - Step-test experiments (M step → observe plant response)
    - Decoupling and interaction analysis (RGA studies)

    Input order: ``[disturbance_1, ..., M_mv1, M_mv2, ...]``

    - Disturbances: plant inputs without an actuator (plant input name order)
    - M signals: one per actuator, in ``actuators`` dict insertion order,
      named ``M_{mv_name}`` (e.g., ``'M_f_oil'``).

    State order: ``[plant states..., actuator.name.vp...]``

    Output order: ``[plant states...]`` (same as DynamicSimulation)

    Parameters
    ----------
    plant : DynamicPlant
        The process plant model.
    actuators : dict[str, ActuatorSystem]
        Mapping of plant MV input name → ActuatorSystem.
        Example: ``{'f_oil': fcv_oil, 'f_MeOH': fcv_meoh}``.

    Examples
    --------
    >>> fcv_oil  = ActuatorSystem('FCV_oil',  tauV=12.0, f_max=6.59e-4)
    >>> fcv_meoh = ActuatorSystem('FCV_MeOH', tauV=8.0,  f_max=3.11e-4)
    >>> sim = OpenLoopSimulation(plant, {'f_oil': fcv_oil, 'f_MeOH': fcv_meoh})
    >>> print(sim.input_names)   # inspect signal layout before building U
    >>> U  = sim.make_U(T, c_TG_in=0.55, T_oil=298.15,
    ...                    M_f_oil=50.0, M_f_MeOH=45.0)
    >>> X0 = sim.make_X0(h=1.5, T=338.15)
    >>> res = sim.run(T, U, X0)
    """

    def __init__(
        self,
        plant: DynamicPlant,
        actuators: dict[str, ActuatorSystem],
        sensors: dict[str, SensorTransmitterSystem] | None = None,
    ) -> None:
        # Validate: every actuator key must be a plant input name
        for mv in actuators:
            if mv not in plant.INPUT_NAMES:
                raise ValueError(
                    f"OpenLoopSimulation: actuator key '{mv}' is not in "
                    f'plant.INPUT_NAMES.\n  Available: {plant.INPUT_NAMES}'
                )
        self.actuators: dict[str, ActuatorSystem] = dict(actuators)

        self.sensors: dict[str, SensorTransmitterSystem] = {}
        if sensors:
            for pv in sensors:
                if pv not in plant.STATE_NAMES and pv not in self.actuators:
                    raise ValueError(
                        f"OpenLoopSimulation: sensor key '{pv}' must be a plant state "
                        f'or an active actuator key.'
                    )
            self.sensors = dict(sensors)

        super().__init__(plant)

    @property
    def simulation_type(self) -> str:
        return 'open_loop'

    # --- Signal layout -------------------------------------------------------

    @property
    def _disturbance_names(self) -> list[str]:
        """Plant inputs that bypass actuators (passed directly by the user)."""
        return [inp for inp in self.plant.INPUT_NAMES if inp not in self.actuators]

    @property
    def input_names(self) -> list[str]:
        disturbances = self._disturbance_names
        M_signals = [f'M_{mv}' for mv in self.actuators]
        return disturbances + M_signals

    @property
    def output_names(self) -> list[str]:
        names = list(self.plant.STATE_NAMES)
        for sns in self.sensors.values():
            names.append(f'{sns.system.name}.C')
        return names

    @property
    def state_names(self) -> list[str]:
        # Plant states first (no prefix), then actuator states, then sensor states
        plant_states = list(self.plant.STATE_NAMES)
        act_states = [f'{act.system.name}.vp' for act in self.actuators.values()]
        sns_states = []
        for sns in self.sensors.values():
            for s in sns.STATE_NAMES:
                sns_states.append(f'{sns.system.name}.{s}')
        return plant_states + act_states + sns_states

    # --- System assembly -----------------------------------------------------

    def _build_system(self) -> ct.NonlinearIOSystem:
        syslist = (
            [self.plant.system]
            + [act.system for act in self.actuators.values()]
            + [sns.system for sns in self.sensors.values()]
        )
        self._validate_unique_names(syslist, 'OpenLoopSimulation')

        plt = self.plant.system.name

        # Wire: plant.mv input ← actuator.F output
        connections = [
            [f'{plt}.{mv}', f'{act.system.name}.F'] for mv, act in self.actuators.items()
        ]

        # Wire: sensor.pv input ← plant output or actuator output
        for pv, sns in self.sensors.items():
            if pv in self.plant.STATE_NAMES:
                connections.append([f'{sns.system.name}.PV', f'{plt}.{pv}'])
            elif pv in self.actuators:
                act = self.actuators[pv]
                connections.append([f'{sns.system.name}.PV', f'{act.system.name}.F'])

        # External inputs: disturbances (direct to plant) + M signals to actuators
        inplist = [f'{plt}.{d}' for d in self._disturbance_names] + [
            f'{act.system.name}.M' for act in self.actuators.values()
        ]

        # External outputs: all plant states + sensor outputs
        outlist = [f'{plt}.{s}' for s in self.plant.STATE_NAMES] + [
            f'{sns.system.name}.C' for sns in self.sensors.values()
        ]

        return ct.interconnect(
            syslist,
            connections=connections,
            inplist=inplist,
            outlist=outlist,
            name='open_loop',
        )


# =============================================================================
# Level 3 — Closed-loop (all control elements)
# =============================================================================


class ClosedLoopSimulation(ProcessSimulation):
    """Full closed-loop simulation with all control elements.

    Wires one or more feedback loops via ``ct.interconnect``.  Each loop
    is described by a ``ControlLoop`` dataclass.  Plant inputs without a
    ``ControlLoop`` assignment are treated as disturbances (external inputs).

    Signal routing per loop::

        SP [eng. units] ──► SetPoint ──► R [%] ──►┐
                                                    Controller ──► M [%] ──► Actuator ──► F ──► Plant
        C [%] ◄── Sensor ◄── PV [eng. units] ◄──────────────────────────────────────────────────────►
        ↑ (feedback to controller)

    External inputs per loop (in this order): ``SP_{pv}``, ``Kc_{pv}``,
    ``tauI_{pv}``, ``tauD_{pv}``.  Passing Kc/tauI/tauD as external inputs
    enables gain scheduling during simulation.

    Input order:
        ``[disturbance_1, ..., SP_pv1, Kc_pv1, tauI_pv1, tauD_pv1, SP_pv2, ...]``

    Output order:
        ``[plant states..., ctrl.name.M..., sp.name.R..., sensor.name.C...]``

    State order:
        ``[plant states..., (sensor.name.PVm)?, ctrl.name.I_state, ctrl.name.D_state,
           act.name.vp, ...]``  (per loop; sensor state only if ``tauT > 0``)

    Parameters
    ----------
    plant : DynamicPlant
        The process plant model.
    loops : list of ControlLoop
        One entry per feedback loop to close.

    Examples
    --------
    >>> loop = ControlLoop(
    ...     mv='f_oil', pv='h',
    ...     actuator=ActuatorSystem('FCV_100', tauV=12.0, f_max=6.59e-4),
    ...     controller=ControllerSystem('LC_100', bias=50.0),
    ...     setpoint=SetPointSystem('LSP_100', hi=3.0, low=0.0),
    ...     sensor=SensorTransmitterSystem('LT_100', hi=3.0, low=0.0, tauT=0.0),
    ... )
    >>> sim = ClosedLoopSimulation(plant, loops=[loop])
    >>> print(sim.input_names)     # inspect before building U
    >>> print(sim.state_names)     # inspect before building X0
    >>> U  = sim.make_U(T,
    ...     c_TG_in=0.55, T_oil=298.15,    # disturbances
    ...     SP_h=1.5,                       # level setpoint [m]
    ...     Kc_h=2.5, tauI_h=300.0, tauD_h=0.0,  # tuning
    ... )
    >>> X0 = sim.make_X0(h=1.5, T=338.15, T_coolant=310.0)
    >>> res = sim.run(T, U, X0)
    >>> res['h']           # level trajectory [m]
    >>> res['LC_100.M']    # controller output [%]
    """

    def __init__(
        self,
        plant: DynamicPlant,
        loops: list[ControlLoop],
    ) -> None:
        if not loops:
            raise ValueError("ClosedLoopSimulation: 'loops' must not be empty.")
        for i, lp in enumerate(loops):
            if lp.mv not in plant.INPUT_NAMES:
                raise ValueError(
                    f'ClosedLoopSimulation loop[{i}] ({lp.tag}): '
                    f"mv='{lp.mv}' is not in plant.INPUT_NAMES.\n"
                    f'  Available: {plant.INPUT_NAMES}'
                )
            if lp.pv_source == 'plant' and lp.pv not in plant.STATE_NAMES:
                raise ValueError(
                    f'ClosedLoopSimulation loop[{i}] ({lp.tag}): '
                    f"pv='{lp.pv}' is not in plant.STATE_NAMES.\n"
                    f'  Available: {plant.STATE_NAMES}\n'
                    f'  Tip: for flow control loops that measure actuator output,'
                    f" set pv_source='actuator'."
                )
        self.loops: list[ControlLoop] = list(loops)
        super().__init__(plant)

    @property
    def simulation_type(self) -> str:
        return 'closed_loop'

    # --- Signal layout -------------------------------------------------------

    @property
    def _controlled_mvs(self) -> list[str]:
        return [lp.mv for lp in self.loops]

    @property
    def _disturbance_names(self) -> list[str]:
        """Plant inputs not driven by any ControlLoop actuator."""
        return [inp for inp in self.plant.INPUT_NAMES if inp not in self._controlled_mvs]

    @property
    def input_names(self) -> list[str]:
        # Disturbances first, then per-loop: SP, Kc, tauI, tauD
        # Use controller tag (instrument name) as loop identifier — unique
        # and meaningful even when pv is an actuator port (flow loops).
        names = list(self._disturbance_names)
        for lp in self.loops:
            tag = lp.tag  # e.g., 'LC_100', 'FC_100'
            names += [f'SP_{tag}', f'Kc_{tag}', f'tauI_{tag}', f'tauD_{tag}']
        return names

    @property
    def output_names(self) -> list[str]:
        # Plant states + per-loop: M, R, C
        names = list(self.plant.STATE_NAMES)
        for lp in self.loops:
            names += [
                f'{lp.controller.system.name}.M',  # controller output
                f'{lp.setpoint.system.name}.R',  # normalized setpoint
                f'{lp.sensor.system.name}.C',  # normalized measurement
            ]
        return names

    @property
    def state_names(self) -> list[str]:
        # Plant states first (no prefix), then per-loop control-element states
        names = list(self.plant.STATE_NAMES)
        for lp in self.loops:
            # Sensor: 1 state (PVm) only if tauT > 0; static sensor has none
            for s in lp.sensor.STATE_NAMES:
                names.append(f'{lp.sensor.system.name}.{s}')
            # Controller: 2 states (I_state, D_state)
            for s in lp.controller.STATE_NAMES:
                names.append(f'{lp.controller.system.name}.{s}')
            # Setpoint: 0 states (static element — nothing to add)
            # Actuator: 1 state (vp)
            for s in lp.actuator.STATE_NAMES:
                names.append(f'{lp.actuator.system.name}.{s}')
        return names

    # --- System assembly -----------------------------------------------------

    def _build_system(self) -> ct.NonlinearIOSystem:
        # Collect all subsystems in deterministic order
        syslist = [self.plant.system]
        for lp in self.loops:
            # Sensor → Controller → Setpoint → Actuator (per loop)
            syslist += [
                lp.sensor.system,
                lp.controller.system,
                lp.setpoint.system,
                lp.actuator.system,
            ]
        self._validate_unique_names(syslist, 'ClosedLoopSimulation')

        plt = self.plant.system.name

        # Wire all feedback loops
        # ct.interconnect connection format: [input_spec, output_spec]
        # i.e., "this input port receives from that output port"
        connections = []
        for lp in self.loops:
            sp = lp.setpoint.system.name
            ctrl = lp.controller.system.name
            act = lp.actuator.system.name
            st = lp.sensor.system.name

            connections += [
                [f'{ctrl}.R', f'{sp}.R'],  # controller.R  ←  setpoint.R
                [f'{ctrl}.C', f'{st}.C'],  # controller.C  ←  sensor.C
                [f'{act}.M', f'{ctrl}.M'],  # actuator.M    ←  controller.M
                [f'{plt}.{lp.mv}', f'{act}.F'],  # plant.MV      ←  actuator.F
            ]
            # PV source: plant state OR actuator output (flow control)
            if lp.pv_source == 'actuator':
                # sensor measures actuator output flow F (flow control loop)
                connections.append([f'{st}.PV', f'{act}.{lp.pv}'])
            else:
                # sensor measures a plant state (level / temperature control)
                connections.append([f'{st}.PV', f'{plt}.{lp.pv}'])

        # External inputs: disturbances + [SP, Kc, tauI, tauD] per loop
        inplist = [f'{plt}.{d}' for d in self._disturbance_names]
        for lp in self.loops:
            sp = lp.setpoint.system.name
            ctrl = lp.controller.system.name
            inplist += [
                f'{sp}.SP',  # setpoint value (engineering units)
                f'{ctrl}.Kc',  # proportional gain
                f'{ctrl}.tauI',  # integral time constant
                f'{ctrl}.tauD',  # derivative time constant
            ]

        # External outputs: plant states + M, R, C per loop
        outlist = [f'{plt}.{s}' for s in self.plant.STATE_NAMES]
        for lp in self.loops:
            ctrl = lp.controller.system.name
            sp = lp.setpoint.system.name
            st = lp.sensor.system.name
            outlist += [
                f'{ctrl}.M',  # controller output (manipulated variable %)
                f'{sp}.R',  # normalized setpoint [%]
                f'{st}.C',  # normalized measurement [%]
            ]

        return ct.interconnect(
            syslist,
            connections=connections,
            inplist=inplist,
            outlist=outlist,
            name='closed_loop',
        )
