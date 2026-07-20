"""Abstract base class for all control loop element models.

Provides ``ControlElement``, a python-control compatible ABC that gives
every instrument model (actuator, controller, setpoint scaler,
sensor/transmitter) a uniform structure — mirroring ``DynamicPlant``
from ``plant.py``.

Design rules (same as DynamicPlant, adapted for control elements):

1. Declare ``STATE_NAMES``, ``INPUT_NAMES``, ``OUTPUT_NAMES`` as class-level
   lists.  Static (memoryless) elements set ``STATE_NAMES = []``.
2. In ``__init__``: build ``self.params``, then call ``super().__init__()``.
3. Override ``_update`` (ODE rhs) and ``_output`` (algebraic output).
   **Always use the forwarded** ``params`` **argument — never** ``self.params``
   — so per-simulation overrides via ``ct.input_output_response(..., params=)``
   stay active throughout the simulation.

Reproducibility note
--------------------
Every element stores all tuning constants in ``self.params``, which is
forwarded verbatim to ``ct.nlsys``.  To run a sensitivity study without
rebuilding the element, pass a ``params`` override to
``ct.input_output_response``::

    resp = ct.input_output_response(
        ctrl.system, T, U, X0,
        params={'bias': 45.0},   # override just one key
    )
"""

from abc import ABC, abstractmethod

import control as ct
import numpy as np

# =============================================================================
# Base class
# =============================================================================


class ControlElement(ABC):
    """Abstract base class for process control loop elements.

    Wraps ``ct.nlsys`` with a consistent OOP interface so that actuators,
    controllers, setpoint scalers, and sensor/transmitters all follow the
    same subclassing contract as ``DynamicPlant``.

    Subclassing contract
    --------------------
    1. Set ``STATE_NAMES``, ``INPUT_NAMES``, ``OUTPUT_NAMES`` as class-level
       lists (or instance attributes before calling ``super().__init__()``).
       For static, memoryless elements set ``STATE_NAMES = []``.
    2. Populate ``self.params`` dict before calling ``super().__init__()``.
    3. Implement ``_update`` (state derivatives) and ``_output`` (output
       equation).  Always read parameters from the *params* argument, not
       from ``self.params``, so per-simulation overrides remain effective.

    Example — minimal first-order lag
    ----------------------------------
    ::

        class FirstOrderLag(ControlElement):

            STATE_NAMES  = ['y']
            INPUT_NAMES  = ['u']
            OUTPUT_NAMES = ['y_out']

            def __init__(self, name: str, tau: float) -> None:
                self.params = {'name': name, 'tau': float(tau)}
                super().__init__()

            def _update(self, t, x, u, params):
                return [(float(u[0]) - float(x[0])) / params['tau']]

            def _output(self, t, x, u, params):
                return [float(x[0])]
    """

    STATE_NAMES: list[str] = []  # empty → static/memoryless element
    INPUT_NAMES: list[str] = []
    OUTPUT_NAMES: list[str] = []
    params: dict  # subclass must assign this before calling super().__init__()

    # -------------------------------------------------------------------------

    def __init__(self) -> None:
        if not self.INPUT_NAMES:
            raise ValueError(f'{type(self).__name__}.INPUT_NAMES must be a non-empty list.')
        if not self.OUTPUT_NAMES:
            raise ValueError(f'{type(self).__name__}.OUTPUT_NAMES must be a non-empty list.')
        if not hasattr(self, 'params'):
            raise AttributeError(
                f'{type(self).__name__}.__init__ must assign self.params '
                'before calling super().__init__().'
            )

        self._element_name = self.params.get('name', type(self).__name__.lower())

        # Prefix all parameters with the element name to prevent clashes in
        # ct.interconnect (which flattens all subsystem params into one dict).
        self.prefixed_params = {f'{self._element_name}_{k}': v for k, v in self.params.items()}

        if self.STATE_NAMES:
            # Dynamic element — register both update and output functions.
            self.system = ct.nlsys(
                self._update_wrapper,
                self._output_wrapper,
                name=self._element_name,
                states=self.STATE_NAMES,
                inputs=self.INPUT_NAMES,
                outputs=self.OUTPUT_NAMES,
                params=self.prefixed_params,
            )
        else:
            # Static (memoryless) element — no state equation, updfcn=None.
            self.system = ct.nlsys(
                None,
                self._output_wrapper,
                name=self._element_name,
                inputs=self.INPUT_NAMES,
                outputs=self.OUTPUT_NAMES,
                params=self.prefixed_params,
            )

    # --- Parameter Prefixing Wrappers -----------------------------------------

    def _unprefix_params(self, params: dict) -> dict:
        """Extract this element's parameters from a flattened interconnect dict."""
        prefix = f'{self._element_name}_'
        local_params = {}
        for k, v in params.items():
            if k.startswith(prefix):
                local_params[k[len(prefix) :]] = v
            # If the user passes an unprefixed override in input_output_response
            # (e.g., params={'tauT': 10.0}), it takes precedence globally.
            elif k in self.params:
                local_params[k] = v
        return local_params

    def _update_wrapper(self, t, x, u, params):
        """Wrapper to unprefix parameters before passing to subclass."""
        return self._update(t, x, u, self._unprefix_params(params))

    def _output_wrapper(self, t, x, u, params):
        """Wrapper to unprefix parameters before passing to subclass."""
        return self._output(t, x, u, self._unprefix_params(params))

    # --- Abstract interface ---------------------------------------------------

    def _update(
        self,
        t: float,
        x: np.ndarray,
        u: np.ndarray,
        params: dict,
    ) -> list:
        """Compute state derivatives (ODE right-hand side).

        Called by the ODE solver at every time step.

        IMPORTANT: Read all parameters from *params* (the argument forwarded
        by ``ct.nlsys``), **not** from ``self.params``.  This ensures that
        per-simulation overrides via ``ct.input_output_response(..., params=)``
        are correctly applied.

        Static elements (``STATE_NAMES = []``) do not need to override this
        method; the default implementation raises ``NotImplementedError`` as
        a safeguard if accidentally called.

        Parameters
        ----------
        t : float
            Current simulation time [s].
        x : ndarray, shape (n_states,)
            Current state vector.
        u : ndarray, shape (n_inputs,)
            Current input vector.
        params : dict
            Parameter dictionary forwarded by ``ct.nlsys``.  Equals
            ``self.params`` merged with any per-simulation override.

        Returns
        -------
        list of float
            State derivatives dx/dt, length n_states.
        """
        raise NotImplementedError(
            f'{type(self).__name__}._update() must be implemented for '
            'dynamic elements (STATE_NAMES is non-empty).'
        )

    @abstractmethod
    def _output(
        self,
        t: float,
        x: np.ndarray,
        u: np.ndarray,
        params: dict,
    ) -> list:
        """Compute the element output (algebraic output equation).

        IMPORTANT: Read all parameters from *params* (the argument forwarded
        by ``ct.nlsys``), **not** from ``self.params``.

        Parameters
        ----------
        t : float
            Current simulation time [s].
        x : ndarray, shape (n_states,)
            Current state vector (empty array for static elements).
        u : ndarray, shape (n_inputs,)
            Current input vector.
        params : dict
            Parameter dictionary forwarded by ``ct.nlsys``.

        Returns
        -------
        list of float
            Output vector, length n_outputs.
        """
        ...

    # -------------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f'{type(self).__name__}('
            f'states={len(self.STATE_NAMES)}, '
            f'inputs={len(self.INPUT_NAMES)}, '
            f'outputs={len(self.OUTPUT_NAMES)})'
        )
