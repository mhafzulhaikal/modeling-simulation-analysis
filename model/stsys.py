"""Process sensor / transmitter model for control loop simulation.

Provides ``SensorTransmitterSystem``, which combines an optional first-order
lag (sensor response delay) with linear range scaling to produce a normalized
0–100 % measurement signal compatible with ``ControllerSystem``.

Signal flow
-----------
PV [engineering units]  →  1st-order lag (optional)  →  range scaling  →  C [%]

When ``tauT = 0`` the lag is bypassed and the output is algebraic
(instantaneous pass-through after scaling).
"""

import numpy as np

from model.ctrlbase import ControlElement

# =============================================================================
# Sensor / transmitter system
# =============================================================================


class SensorTransmitterSystem(ControlElement):
    """Process sensor with optional first-order lag and linear range scaling.

    Converts a raw process variable PV (in engineering units) to a normalized
    0–100 % signal C for feedback to the controller.  When ``tauT > 0``, a
    first-order lag emulates sensor response delay (e.g., thermowell lag for
    a thermocouple, or dampening in a pressure transmitter).

    States  (1 if tauT > 0, else 0): PVm — filtered / measured PV
    Inputs  (1): PV — true process variable (engineering units)
    Outputs (1): C  — normalized measurement signal [%]

    Parameters
    ----------
    name : str
        Instrument tag (e.g., ``'TT_100'``, ``'LT_100'``).
    hi : float
        Upper measurement range limit → 100 % output (engineering units).
    low : float
        Lower measurement range limit → 0 % output (engineering units).
    tauT : float
        Sensor / transmitter time constant [s].  Must be >= 0.
        Set to 0 for an instantaneous (algebraic) sensor.

    Notes
    -----
    **Sensor dynamics** (active only when ``tauT > 0``)::

        dPVm/dt = (PV - PVm) / tauT

    **Range scaling** (applied to PVm when dynamic; to PV directly when static)::

        C = (PVm - low) / (hi - low) * 100

    Safe keys for per-simulation override via
    ``ct.input_output_response(..., params=)``:
    ``tauT``, ``hi``, ``low`` (useful for transmitter calibration studies).

    Examples
    --------
    >>> # Temperature transmitter with 45 s thermowell lag
    >>> tt = SensorTransmitterSystem('TT_100', hi=368.15, low=298.15, tauT=45.0)

    >>> # Level transmitter — instantaneous (no lag)
    >>> lt = SensorTransmitterSystem('LT_100', hi=3.0, low=0.0, tauT=0.0)
    """

    INPUT_NAMES = ["PV"]
    OUTPUT_NAMES = ["C"]
    # STATE_NAMES is set at the instance level in __init__ based on tauT,
    # before super().__init__() is called.  Python's attribute lookup
    # (instance → class) ensures the base class sees the correct value.

    def __init__(
        self,
        name: str,
        hi: float,
        low: float,
        tauT: float,
    ) -> None:
        if hi <= low:
            raise ValueError(f"hi ({hi}) must be greater than low ({low}).")
        if tauT < 0:
            raise ValueError(f"tauT must be >= 0, got {tauT}.")

        # Dynamic if tauT > 0 (1 state: PVm); static (algebraic) if tauT == 0.
        self.STATE_NAMES = ["PVm"] if tauT > 0 else []

        self.params = {
            "name": str(name),
            "hi": float(hi),
            "low": float(low),
            "tauT": float(tauT),
        }
        super().__init__()

    # --- ControlElement interface --------------------------------------------

    def _update(
        self,
        t: float,
        x: np.ndarray,
        u: np.ndarray,
        params: dict,
    ) -> list:
        """Sensor lag dynamics — ODE right-hand side.

        Only registered with ``ct.nlsys`` when ``tauT > 0``.

        Parameters
        ----------
        t : float
            Simulation time [s].
        x : ndarray, shape (1,)
            State [PVm] — current filtered measurement.
        u : ndarray, shape (1,)
            Input [PV] — true process variable (engineering units).
        params : dict
            Parameter dict forwarded by ``ct.nlsys``.  Use *params*, not
            ``self.params``, so per-simulation overrides remain active.

        Returns
        -------
        list of float
            [dPVm/dt].
        """
        PVm = float(x[0])
        PV = float(u[0])
        tauT = float(params["tauT"])

        dPVm_dt = (PV - PVm) / tauT
        return [dPVm_dt]

    def _output(
        self,
        t: float,
        x: np.ndarray,
        u: np.ndarray,
        params: dict,
    ) -> list:
        """Scale measured value to 0–100 % normalized signal.

        Parameters
        ----------
        t : float
            Simulation time [s].
        x : ndarray
            State [PVm] when tauT > 0; empty array when tauT = 0.
        u : ndarray, shape (1,)
            Input [PV] — true process variable (engineering units).
        params : dict
            Parameter dict forwarded by ``ct.nlsys``.

        Returns
        -------
        list of float
            [C] — normalized measurement [%].
        """
        # Dynamic sensor: use filtered state PVm.
        # Static sensor: no state exists; scale the raw input PV directly.
        PVm = float(x[0]) if x is not None and len(x) > 0 else float(u[0])

        C = (PVm - params["low"]) / (params["hi"] - params["low"]) * 100.0
        return [C]
