"""Control valve actuator model for process control simulation.

Provides ``ActuatorSystem``, a first-order dynamic valve model compatible
with python-control (``ct.nlsys``).  Supports linear, equal-percentage, and
quick-opening valve characteristics, plus fail-closed / fail-open actions.

Signal flow
-----------
Controller output M [%]  →  valve position dynamics (1st-order lag)
    →  valve position vp [%]  →  characteristic curve  →  flow F [m³/s]
"""

import numpy as np

from model.ctrlbase import ControlElement

# =============================================================================
# Actuator system
# =============================================================================


class ActuatorSystem(ControlElement):
    """First-order dynamic model of a process control valve.

    Models valve position dynamics as a first-order lag driven by a
    controller output signal M [%], then converts the resulting valve
    position to a volumetric flow rate through a configurable
    characteristic curve.

    States  (1): vp — valve position [%]
    Inputs  (1): M  — controller output signal [%]
    Outputs (1): F  — actual volumetric flow rate [m³/s]

    Parameters
    ----------
    name : str
        Instrument tag for the valve (e.g., ``'FCV_100'``).
    tauV : float
        First-order valve time constant [s].  Must be >= 0.
        Set to 0 for an instantaneous (algebraic) valve.
    f_max : float
        Maximum volumetric flow rate at full-open position [m³/s].
        Must be > 0.
    vp_min : float, optional
        Minimum allowable valve position [%].  Default: 0.0.
    vp_max : float, optional
        Maximum allowable valve position [%].  Default: 100.0.
    valve_type : {'linear', 'equal_percentage', 'quick_opening'}, optional
        Inherent flow characteristic.  Default: ``'linear'``.
    valve_action : {'FC', 'FO'}, optional
        Failure mode: fail-closed (FC) or fail-open (FO) on signal loss.
        Default: ``'FC'``.

    Notes
    -----
    **Valve position dynamics** (first-order lag):

    .. code-block:: none

        FC action:  dvp/dt = (M        - vp) / tauV
        FO action:  dvp/dt = ((100-M)  - vp) / tauV

    **Flow rate calculation**::

        F = characteristic(clip(vp, vp_min, vp_max)) / 100 * f_max

    Safe keys for per-simulation override via
    ``ct.input_output_response(..., params=)``:
    ``tauV``, ``f_max``, ``vp_min``, ``vp_max``, ``valve_action``.

    Examples
    --------
    >>> act = ActuatorSystem('FCV_100', tauV=12.0, f_max=6.59e-4)
    >>> act.system   # ct.NonlinearIOSystem object
    """

    STATE_NAMES = ["vp"]
    INPUT_NAMES = ["M"]
    OUTPUT_NAMES = ["F"]

    VALVE_TYPES = ("linear", "equal_percentage", "quick_opening")
    VALVE_ACTIONS = ("FC", "FO")

    def __init__(
        self,
        name: str,
        tauV: float,
        f_max: float,
        vp_min: float = 0.0,
        vp_max: float = 100.0,
        valve_type: str = "linear",
        valve_action: str = "FC",
    ) -> None:
        if valve_type not in self.VALVE_TYPES:
            raise ValueError(
                f"valve_type must be one of {self.VALVE_TYPES}, got '{valve_type}'."
            )
        if valve_action not in self.VALVE_ACTIONS:
            raise ValueError(
                f"valve_action must be one of {self.VALVE_ACTIONS}, "
                f"got '{valve_action}'."
            )
        if tauV < 0:
            raise ValueError(f"tauV must be >= 0, got {tauV}.")
        if f_max <= 0:
            raise ValueError(f"f_max must be > 0, got {f_max}.")
        if vp_min >= vp_max:
            raise ValueError(f"vp_min ({vp_min}) must be less than vp_max ({vp_max}).")

        self.params = {
            "name": str(name),
            "tauV": float(tauV),
            "f_max": float(f_max),
            "vp_min": float(vp_min),
            "vp_max": float(vp_max),
            "valve_type": str(valve_type),
            "valve_action": str(valve_action),
        }
        super().__init__()

    # --- Valve characteristic curve ------------------------------------------

    @staticmethod
    def _valve_characteristic(vp: float, valve_type: str) -> float:
        """Map valve position [%] to normalized flow [%] via the inherent curve.

        Parameters
        ----------
        vp : float
            Valve position [%], expected in range [0, 100].
        valve_type : str
            One of ``'linear'``, ``'equal_percentage'``, ``'quick_opening'``.

        Returns
        -------
        float
            Normalized flow [%] in range [0, 100].
        """
        vp = float(vp)
        if valve_type == "linear":
            return vp
        if valve_type == "equal_percentage":
            # Rangeability R=50: flow = R^(vp/100 - 1) * 100
            return 100.0 * (50.0 ** (0.01 * vp - 1.0))
        if valve_type == "quick_opening":
            return 100.0 * np.sqrt(0.01 * vp)
        raise ValueError(f"Unknown valve_type: '{valve_type}'.")

    # --- ControlElement interface --------------------------------------------

    def _update(
        self,
        t: float,
        x: np.ndarray,
        u: np.ndarray,
        params: dict,
    ) -> list:
        """Valve position dynamics — ODE right-hand side.

        Parameters
        ----------
        t : float
            Current simulation time [s].
        x : ndarray, shape (1,)
            State [vp] — current valve position [%].
        u : ndarray, shape (1,)
            Input [M] — controller output signal [%].
        params : dict
            Parameter dict forwarded by ``ct.nlsys``.
            Use *params*, not ``self.params``, so per-simulation overrides
            via ``ct.input_output_response(..., params=)`` remain active.

        Returns
        -------
        list of float
            State derivative [dvp/dt].
        """
        vp = float(x[0])
        M = float(u[0])
        tauV = float(params["tauV"])

        if tauV <= 0.0:
            # Instantaneous valve: treat as algebraic (no dynamics).
            return [0.0]

        if params["valve_action"] == "FC":
            dvp_dt = (M - vp) / tauV
        else:  # FO — fail-open: signal drives valve toward (100 - M)
            dvp_dt = ((100.0 - M) - vp) / tauV

        return [dvp_dt]

    def _output(
        self,
        t: float,
        x: np.ndarray,
        u: np.ndarray,
        params: dict,
    ) -> list:
        """Convert valve position to volumetric flow rate.

        Parameters
        ----------
        t : float
            Current simulation time [s].
        x : ndarray, shape (1,)
            State [vp] — valve position [%].
        u : ndarray, shape (1,)
            Input [M] — controller output [%] (not used here).
        params : dict
            Parameter dict forwarded by ``ct.nlsys``.

        Returns
        -------
        list of float
            Output [F] — volumetric flow rate [m³/s].
        """
        vp = float(x[0]) if x is not None and len(x) > 0 else 0.0
        vp_clamped = np.clip(vp, params["vp_min"], params["vp_max"])
        flow_pct = self._valve_characteristic(vp_clamped, params["valve_type"])
        F = (flow_pct / 100.0) * params["f_max"]

        return [F]
