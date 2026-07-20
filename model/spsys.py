"""Setpoint normalizer for process control loops.

Provides ``SetPointSystem``, a static (memoryless) gain element that
scales a setpoint from engineering units to a normalized 0–100 % signal
for input to ``ControllerSystem``.

Signal flow
-----------
SP [engineering units]  →  linear scaling  →  R [%]
"""

from model.ctrlbase import ControlElement

# =============================================================================
# Setpoint system
# =============================================================================


class SetPointSystem(ControlElement):
    """Linear scaling of a setpoint from engineering units to 0–100 %.

    This is a purely algebraic (memoryless) element — it has no states and
    no ODE.  It maps the raw setpoint SP to a normalized signal R using the
    configured instrument span [low, hi].

    States  (0): none — static element
    Inputs  (1): SP — setpoint in engineering units
    Outputs (1): R  — normalized setpoint [%]

    Parameters
    ----------
    name : str
        Instrument tag (e.g., ``'LSP_100'``, ``'TSP_100'``).
    hi : float
        Upper span limit → 100 % output (engineering units).
    low : float
        Lower span limit → 0 % output (engineering units).

    Notes
    -----
    Scaling equation::

        R = (SP - low) / (hi - low) * 100

    Safe key for per-simulation override:
    ``hi``, ``low`` (useful for instrument re-ranging studies).

    Examples
    --------
    >>> sp = SetPointSystem('LSP_100', hi=3.0, low=0.0)
    >>> sp.system   # ct.NonlinearIOSystem object
    """

    STATE_NAMES = []  # no dynamics — pure algebraic element
    INPUT_NAMES = ["SP"]
    OUTPUT_NAMES = ["R"]

    def __init__(self, name: str, hi: float, low: float) -> None:
        if hi <= low:
            raise ValueError(f"hi ({hi}) must be greater than low ({low}).")

        self.params = {
            "name": str(name),
            "hi": float(hi),
            "low": float(low),
        }
        super().__init__()

    # --- ControlElement interface --------------------------------------------
    # _update is not overridden: the base class default raises NotImplementedError,
    # but it is never registered with ct.nlsys (STATE_NAMES is empty).

    def _output(
        self,
        t: float,
        x,
        u,
        params: dict,
    ) -> list:
        """Scale setpoint SP to normalized signal R [%].

        Parameters
        ----------
        t : float
            Simulation time [s] (unused).
        x : ndarray
            State vector (empty for this static element).
        u : ndarray, shape (1,)
            Input [SP] — setpoint in engineering units.
        params : dict
            Parameter dict forwarded by ``ct.nlsys``.

        Returns
        -------
        list of float
            [R] — normalized setpoint [%].
        """
        SP = float(u[0])
        R = (SP - params["low"]) / (params["hi"] - params["low"]) * 100.0
        return [R]
