"""PID controller model for process control loop simulation.

Provides ``ControllerSystem``, a flexible PID controller compatible with
python-control (``ct.nlsys``).  The controller operates on normalized
signals (0–100 %) and automatically selects its mode (P/PI/PD/PID) from
the live tuning inputs at every time step.

Signal flow
-----------
Normalized setpoint R [%]  ─┐
Normalized measurement C [%]─┤→  PID law  →  output M [%]
Tuning: Kc, tauI, tauD    ─┘

Control law features
--------------------
- Back-calculation anti-windup on the integral term
- Derivative action on the process variable only (no setpoint kick)
- First-order low-pass filter (tauF = Alpha * tauD) on the derivative
- Output saturation to [0, 100] %
- Direct and reverse acting modes
"""

import numpy as np

from model.ctrlbase import ControlElement

# =============================================================================
# Controller system
# =============================================================================


class ControllerSystem(ControlElement):
    """PID process controller with anti-windup and derivative filtering.

    Operates entirely on normalized (0–100 %) signals.  The active control
    mode (P, PI, PD, or PID) is resolved automatically from the live tuning
    inputs every time step, allowing seamless mode switching during a
    simulation — useful for gain-scheduling or robustness studies.

    States  (2): I_state — integral accumulator [%]
                 D_state — derivative filter state [%, same units as C]
    Inputs  (5): R     — normalized setpoint [%]
                 C     — normalized measurement [%]
                 Kc    — proportional gain [%/%]
                 tauI  — integral time constant [s]  (0 → no integral)
                 tauD  — derivative time constant [s] (0 → no derivative)
    Outputs (1): M     — controller output / manipulated variable [%]

    Parameters
    ----------
    name : str, optional
        Instrument tag (e.g., ``'TC_100'``).  Default: ``'ControllerSystem'``.
    bias : float, optional
        Output bias (initial steady-state output) [%].  Default: 50.0.
        Must be in [0, 100].
    mode : {'AUTO', 'P', 'PI', 'PD', 'PID'}, optional
        Operating mode.  ``'AUTO'`` selects the mode from the live tauI/tauD
        inputs at every step.  Default: ``'AUTO'``.
    acting : {'DIRECT', 'REVERSE'}, optional
        Controller action direction.  Default: ``'REVERSE'``.

    Notes
    -----
    **AUTO mode selection** (evaluated each time step):

    ============  ============  ========
    tauI          tauD          Mode
    ============  ============  ========
    > 0           > 0           PID
    > 0           = 0           PI
    = 0           > 0           PD
    = 0           = 0           P
    ============  ============  ========

    Zero threshold: ``MODE_THRESHOLD = 1e-12``.

    **Derivative filter**: ``tauF = Alpha * tauD``,
    ``Alpha = ALPHA_DERIVATIVE_FILTER = 0.125``.

    Safe keys for per-simulation override via
    ``ct.input_output_response(..., params=)``:
    ``bias``, ``acting``, ``mode``, ``Alpha``, ``M_min``, ``M_max``.

    Examples
    --------
    >>> ctrl = ControllerSystem('TC_100', bias=50.0, acting='REVERSE')
    >>> ctrl.system   # ct.NonlinearIOSystem object
    """

    STATE_NAMES = ["I_state", "D_state"]
    INPUT_NAMES = ["R", "C", "Kc", "tauI", "tauD"]
    OUTPUT_NAMES = ["M"]

    ALPHA_DERIVATIVE_FILTER = 0.125  # tauF = Alpha * tauD
    OUTPUT_MIN = 0.0  # lower output saturation limit [%]
    OUTPUT_MAX = 100.0  # upper output saturation limit [%]
    MODE_THRESHOLD = 1e-12  # tauI/tauD values below this → zero

    VALID_MODES = ("AUTO", "P", "PI", "PD", "PID")
    VALID_ACTIONS = ("DIRECT", "REVERSE")

    def __init__(
        self,
        name: str = "ControllerSystem",
        bias: float | None = None,
        mode: str = "AUTO",
        acting: str = "REVERSE",
    ) -> None:
        if mode not in self.VALID_MODES:
            raise ValueError(f"mode must be one of {self.VALID_MODES}, got '{mode}'.")
        if acting not in self.VALID_ACTIONS:
            raise ValueError(
                f"acting must be one of {self.VALID_ACTIONS}, got '{acting}'."
            )
        bias = 50.0 if bias is None else float(bias)
        if not (self.OUTPUT_MIN <= bias <= self.OUTPUT_MAX):
            raise ValueError(
                f"bias must be in [{self.OUTPUT_MIN}, {self.OUTPUT_MAX}], got {bias}."
            )

        self.params = {
            "name": str(name),
            "bias": bias,
            "mode": str(mode),
            "acting": str(acting),
            "Alpha": float(self.ALPHA_DERIVATIVE_FILTER),
            "M_min": float(self.OUTPUT_MIN),
            "M_max": float(self.OUTPUT_MAX),
        }
        super().__init__()

    # --- Mode resolution -----------------------------------------------------

    @staticmethod
    def _resolve_mode(tauI: float, tauD: float, params: dict) -> str:
        """Determine the active control mode from live tuning parameters.

        If ``params['mode']`` is not ``'AUTO'``, that forced mode is returned
        directly.  Otherwise the mode is inferred from the tauI/tauD values.

        Parameters
        ----------
        tauI, tauD : float
            Current integral and derivative time constants [s].
        params : dict
            Parameter dict (must contain key ``'mode'``).

        Returns
        -------
        str
            One of ``'P'``, ``'PI'``, ``'PD'``, ``'PID'``.
        """
        forced = params["mode"]
        if forced != "AUTO":
            return forced

        has_I = float(tauI) > ControllerSystem.MODE_THRESHOLD
        has_D = float(tauD) > ControllerSystem.MODE_THRESHOLD

        if has_I and has_D:
            return "PID"
        if has_I:
            return "PI"
        if has_D:
            return "PD"
        return "P"

    # --- Individual control laws (pure functions for testability) ------------
    #
    # Each law returns (dI_state, dD_state, M_sat) so that _update and
    # _output can share the same logic without duplicating the math.

    @staticmethod
    def _P_controller(
        R: float,
        C: float,
        Kc: float,
        params: dict,
    ) -> tuple[float, float, float]:
        """Proportional-only (P) control law.

        Returns
        -------
        dI_state, dD_state, M_sat : float
            Both state derivatives are zero; M_sat is the saturated output.
        """
        error = float(R) - float(C)
        Kc_eff = float(Kc) if params["acting"] == "REVERSE" else -float(Kc)
        M_unsat = params["bias"] + Kc_eff * error
        M_sat = float(np.clip(M_unsat, params["M_min"], params["M_max"]))
        return 0.0, 0.0, M_sat

    @staticmethod
    def _PI_controller(
        R: float,
        C: float,
        Kc: float,
        tauI: float,
        I_state: float,
        params: dict,
    ) -> tuple[float, float, float]:
        """Proportional-Integral (PI) control law with back-calculation anti-windup.

        Anti-windup bleed: the integrator is wound back in proportion to the
        output saturation error, preventing integrator wind-up on actuator
        limits.

        Returns
        -------
        dI_state, dD_state, M_sat : float
        """
        error = float(R) - float(C)
        Kc_eff = float(Kc) if params["acting"] == "REVERSE" else -float(Kc)
        M_unsat = Kc_eff * error + float(I_state)
        M_sat = float(np.clip(M_unsat, params["M_min"], params["M_max"]))

        # Back-calculation: dI/dt = Kc/tauI * e + (M_sat - M_unsat) / tauI
        dI_state = (Kc_eff / float(tauI)) * error + (M_sat - M_unsat) / float(tauI)
        return dI_state, 0.0, M_sat

    @staticmethod
    def _PD_controller(
        R: float,
        C: float,
        tauD: float,
        D_state: float,
        params: dict,
    ) -> tuple[float, float, float]:
        """Proportional-Derivative (PD) control law with first-order filter.

        Derivative acts on the measurement C only (derivative-on-PV), which
        avoids the step-change kick on the output when the setpoint R changes.

        Returns
        -------
        dI_state, dD_state, M_sat : float
        """
        diff = float(C) - float(D_state)
        U_d = diff / params["Alpha"]  # filtered derivative term
        dD_state = U_d / float(tauD)  # filter state derivative

        M_unsat = float(C) + U_d
        M_sat = float(np.clip(M_unsat, params["M_min"], params["M_max"]))
        return 0.0, dD_state, M_sat

    @staticmethod
    def _PID_controller(
        R: float,
        C: float,
        Kc: float,
        tauI: float,
        tauD: float,
        I_state: float,
        D_state: float,
        params: dict,
    ) -> tuple[float, float, float]:
        """Full PID control law — derivative-on-measurement with anti-windup.

        Combines proportional, integral (with back-calculation anti-windup),
        and derivative-on-PV (with first-order filter) actions.

        Returns
        -------
        dI_state, dD_state, M_sat : float
        """
        Kc_eff = float(Kc) if params["acting"] == "REVERSE" else -float(Kc)

        # Derivative-on-PV: filter (C - D_state) through 1st-order lag
        diff = float(C) - float(D_state)
        U_d = diff / params["Alpha"]
        dD_state = U_d / float(tauD)

        # Filtered measurement seen by the proportional + integral terms
        C_filt = float(C) + U_d
        error = float(R) - C_filt

        M_unsat = Kc_eff * error + float(I_state)
        M_sat = float(np.clip(M_unsat, params["M_min"], params["M_max"]))

        # Back-calculation anti-windup
        dI_state = (Kc_eff / float(tauI)) * error + (M_sat - M_unsat) / float(tauI)
        return dI_state, dD_state, M_sat

    # --- ControlElement interface (ODE + output) ----------------------------

    def _update(
        self,
        t: float,
        x: np.ndarray,
        u: np.ndarray,
        params: dict,
    ) -> list:
        """State derivatives for the PID controller — ODE right-hand side.

        Parameters
        ----------
        t : float
            Current simulation time [s].
        x : ndarray, shape (2,)
            State [I_state, D_state].
        u : ndarray, shape (5,)
            Input [R, C, Kc, tauI, tauD].
        params : dict
            Parameter dict forwarded by ``ct.nlsys``.  Use *params*, not
            ``self.params``, so per-simulation overrides remain active.

        Returns
        -------
        list of float
            [dI_state/dt, dD_state/dt].
        """
        R, C, Kc, tauI, tauD = u
        I_state, D_state = x

        mode = self._resolve_mode(tauI, tauD, params)

        if mode == "P":
            dI, dD, _ = self._P_controller(R, C, Kc, params)
        elif mode == "PI":
            dI, dD, _ = self._PI_controller(R, C, Kc, tauI, I_state, params)
        elif mode == "PD":
            dI, dD, _ = self._PD_controller(R, C, tauD, D_state, params)
        else:  # PID
            dI, dD, _ = self._PID_controller(
                R, C, Kc, tauI, tauD, I_state, D_state, params
            )

        return [dI, dD]

    def _output(
        self,
        t: float,
        x: np.ndarray,
        u: np.ndarray,
        params: dict,
    ) -> list:
        """Controller output equation.

        Parameters
        ----------
        t : float
            Current simulation time [s].
        x : ndarray, shape (2,)
            State [I_state, D_state].
        u : ndarray, shape (5,)
            Input [R, C, Kc, tauI, tauD].
        params : dict
            Parameter dict forwarded by ``ct.nlsys``.

        Returns
        -------
        list of float
            [M] — controller output / manipulated variable [%].
        """
        R, C, Kc, tauI, tauD = u
        I_state, D_state = x

        mode = self._resolve_mode(tauI, tauD, params)

        if mode == "P":
            _, _, M = self._P_controller(R, C, Kc, params)
        elif mode == "PI":
            _, _, M = self._PI_controller(R, C, Kc, tauI, I_state, params)
        elif mode == "PD":
            _, _, M = self._PD_controller(R, C, tauD, D_state, params)
        else:  # PID
            _, _, M = self._PID_controller(
                R, C, Kc, tauI, tauD, I_state, D_state, params
            )

        return [M]
