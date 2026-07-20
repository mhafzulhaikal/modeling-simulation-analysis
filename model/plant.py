"""Nonlinear dynamic plant models for process control research.

Provides a structured, python-control compatible base class (``DynamicPlant``)
and a concrete biodiesel transesterification reactor (``BiodieselPlant``).
New plant models are created by subclassing ``DynamicPlant`` and implementing
the ODE right-hand side in ``_update``.

Reproducibility note
--------------------
All physical parameters are stored in ``self.params`` and forwarded verbatim
to ``ct.NonlinearIOSystem``.  To run a sensitivity sweep without rebuilding
the plant object, supply a ``params`` override to the solver::

    resp = ct.input_output_response(
        plant.system, T, U, X0,
        params={'UA': 0.5},   # only overrides UA; all other params unchanged
        solve_ivp_kwargs={'method': 'LSODA'},
    )
"""

from abc import ABC, abstractmethod

import control as ct
import numpy as np

# =============================================================================
# Base class
# =============================================================================


class DynamicPlant(ABC):
    """Abstract base class for nonlinear process plant models.

    Wraps ``ct.NonlinearIOSystem`` with a structured OOP interface that
    supports parameter pre-computation and clean subclassing for different
    unit operations.

    Subclassing contract
    --------------------
    1. Declare ``STATE_NAMES`` and ``INPUT_NAMES`` as class-level lists.
    2. In ``__init__``: build ``self.params``, then call ``super().__init__()``.
    3. Implement ``_update(t, x, u, params)`` — the ODE right-hand side.

    IMPORTANT: Always read parameters from the *params* argument forwarded
    by ``ct.NonlinearIOSystem``, **not** from ``self.params``.  This allows
    per-simulation sensitivity sweeps without rebuilding the plant::

        resp = ct.input_output_response(
            plant.system, T, U, X0,
            params={'UA': 0.5},   # sensitivity: override only UA
            solve_ivp_kwargs={'method': 'LSODA'},
        )

    Example — minimal heat exchanger subclass
    ------------------------------------------
    ::

        class HeatExchangerPlant(DynamicPlant):

            # fmt: off
    STATE_NAMES = [
        "h", "c_TG", "c_MeOH", "c_ME", "c_DG", "c_MG",
        "c_Gly", "c_Cat", "c_Water", "T", "T_coolant"
    ]
    INPUT_NAMES = [
        "c_TG_in", "T_oil", "c_MeOH_in", "T_MeOH",
        "c_Cat_in", "c_Water_in", "T_NaOH", "T_coolant_in",
        "f_oil", "f_MeOH", "f_NaOH", "f_FAME", "f_coolant"
    ]
    # fmt: on

            def __init__(self, UA: float, rho: float,
                         Cp: float, V: float) -> None:
                # fmt: off
        self.params = {
            # --- Feed stream physical properties ---
            "rho_oil":     rho_oil,
            "Cp_oil":      Cp_oil,
            "rho_MeOH":    rho_MeOH,
            "Cp_MeOH":     Cp_MeOH,
            "rho_NaOH":    rho_NaOH,
            "Cp_NaOH":     Cp_NaOH,

            # --- Product mixture properties ---
            "rho":         rho,
            "Cp":          Cp,

            # --- Reactor geometry ---
            "Dr":          Dr,
            "Lr":          Lr,

            # --- Reaction thermodynamics ---
            "R":           R,
            "To":          To,
            "Hrxn1":       Hrxn1,
            "Hrxn2":       Hrxn2,
            "Hrxn3":       Hrxn3,

            # --- Kinetic rate constants ---
            "k1_f":        k1_f,
            "k1_r":        k1_r,
            "k2_f":        k2_f,
            "k2_r":        k2_r,
            "k3_f":        k3_f,
            "k3_r":        k3_r,

            # --- Activation energies ---
            "E1_f":        E1_f,
            "E1_r":        E1_r,
            "E2_f":        E2_f,
            "E2_r":        E2_r,
            "E3_f":        E3_f,
            "E3_r":        E3_r,

            # --- Heat transfer ---
            "UA":          UA,
            "V_coolant":   V_coolant,
            "rho_coolant": rho_coolant,
            "Cp_coolant":  Cp_coolant,

            # --- Pre-computed constants (derived; not independent inputs) ---
            "Ar":               Ar,
            "E1f_R":            E1_f / R,             # Arrhenius pre-factors E/R
            "E1r_R":            E1_r / R,
            "E2f_R":            E2_f / R,
            "E2r_R":            E2_r / R,
            "E3f_R":            E3_f / R,
            "E3r_R":            E3_r / R,
            "rho_oil_Cp_oil":   rho_oil * Cp_oil,     # feed enthalpy coefficients
            "rho_MeOH_Cp_MeOH": rho_MeOH * Cp_MeOH,
            "rho_NaOH_Cp_NaOH": rho_NaOH * Cp_NaOH,
            "inv_coolant_cap":  1.0 / (V_coolant * rho_coolant * Cp_coolant),
            "inv_rho_Ar":       1.0 / (rho * Ar),
        }
        # fmt: on
                super().__init__()

            def _update(self, t, x, u, params):
                T_hot, T_cold = x
                F_hot, F_cold, T_hot_in, T_cold_in = u
                p = params  # use params arg, not self.params
                Q = p['UA'] * (T_hot - T_cold) / (p['rho'] * p['Cp'] * p['V'])
                return [
                    (T_hot_in  - T_hot)  * F_hot  / p['V'] - Q,
                    (T_cold_in - T_cold) * F_cold / p['V'] + Q,
                ]
    """

    STATE_NAMES: list[str] = []
    INPUT_NAMES: list[str] = []
    params: dict  # must be assigned by subclass before super().__init__()

    def __init__(self) -> None:
        if not self.STATE_NAMES:
            raise ValueError(f'{type(self).__name__}.STATE_NAMES must be a non-empty list.')
        if not self.INPUT_NAMES:
            raise ValueError(f'{type(self).__name__}.INPUT_NAMES must be a non-empty list.')
        if not hasattr(self, 'params'):
            raise AttributeError(
                f'{type(self).__name__}.__init__ must assign self.params '
                'before calling super().__init__().'
            )

        self.system = ct.NonlinearIOSystem(
            updfcn=self._update,
            outfcn=self._output,
            inputs=self.INPUT_NAMES,
            outputs=self.STATE_NAMES,  # full-state output (override _output to change)
            states=self.STATE_NAMES,
            name=type(self).__name__.lower(),
            params=self.params,  # forwarded to _update/_output as params arg
        )

    @abstractmethod
    def _update(
        self,
        t: float,
        x: np.ndarray,
        u: np.ndarray,
        params: dict,
    ) -> list:
        """Compute state derivatives (ODE right-hand side).

        Called by the ODE solver at every time step.  Access system
        parameters through *params* — not ``self.params`` — so any
        per-simulation override passed to ``ct.input_output_response``
        remains active.

        Parameters
        ----------
        t : float
            Current simulation time [s].
        x : ndarray, shape (n_states,)
            Current state vector.
        u : ndarray, shape (n_inputs,)
            Current input vector.
        params : dict
            Parameter dictionary forwarded by ``ct.NonlinearIOSystem``.
            Equals ``self.params`` merged with any override supplied via
            ``ct.input_output_response(..., params={...})``.

        Returns
        -------
        list of float
            State derivatives dx/dt, length n_states.
        """
        ...

    def _output(
        self,
        t: float,
        x: np.ndarray,
        u: np.ndarray,
        params: dict,
    ) -> np.ndarray:
        """Return the full state vector as the system output.

        Override in subclasses to expose only a subset of states or to
        apply a sensor transformation.

        Parameters
        ----------
        t : float
            Current simulation time [s].
        x : ndarray, shape (n_states,)
            Current state vector.
        u : ndarray, shape (n_inputs,)
            Current input vector (unused in default implementation).
        params : dict
            Parameter dictionary (unused in default implementation).

        Returns
        -------
        ndarray, shape (n_states,)
            Copy of the current state vector.
        """
        return np.asarray(x)

    def __repr__(self) -> str:
        return (
            f'{type(self).__name__}(states={len(self.STATE_NAMES)}, inputs={len(self.INPUT_NAMES)})'
        )


# =============================================================================
# Biodiesel plant
# =============================================================================


class BiodieselPlant(DynamicPlant):
    """Biodiesel transesterification reactor (CSTR, 3 reversible reactions).

    Models the continuous transesterification of triglyceride oil with
    methanol, catalysed by NaOH, following three sequential reversible
    reactions:

    .. code-block:: none

        (1)  TG  + MeOH  <-->  ME + DG    dH1
        (2)  DG  + MeOH  <-->  ME + MG    dH2
        (3)  MG  + MeOH  <-->  ME + Gly   dH3

    State vector (11 states):
        h, c_TG, c_MeOH, c_ME, c_DG, c_MG, c_Gly, c_Cat, c_Water,
        T, T_coolant

    Input vector (13 inputs):
        Disturbances (8): c_TG_in, T_oil, c_MeOH_in, T_MeOH,
                          c_Cat_in, c_Water_in, T_NaOH, T_coolant_in
        Manipulated  (5): f_oil, f_MeOH, f_NaOH, f_FAME, f_coolant

    Parameters
    ----------
    rho_oil, Cp_oil : float
        Oil feed density [kg/m³] and heat capacity [kcal/(kg·K)].
    rho_MeOH, Cp_MeOH : float
        Methanol feed density [kg/m³] and heat capacity [kcal/(kg·K)].
    rho_NaOH, Cp_NaOH : float
        NaOH feed density [kg/m³] and heat capacity [kcal/(kg·K)].
    rho, Cp : float
        Product mixture density [kg/m³] and heat capacity [kcal/(kg·K)].
    Dr, Lr : float
        Reactor internal diameter and length [m].
    R : float
        Gas constant [kcal/(kmol·K)].
    To : float
        Arrhenius reference temperature [K].
    Hrxn1, Hrxn2, Hrxn3 : float
        Heats of reaction for reactions 1, 2, and 3 [kcal/kmol].
    k1_f, k1_r : float
        Forward and reverse pre-exponential rate constants, reaction 1.
    k2_f, k2_r : float
        Forward and reverse pre-exponential rate constants, reaction 2.
    k3_f, k3_r : float
        Forward and reverse pre-exponential rate constants, reaction 3.
    E1_f, E1_r : float
        Activation energies for reaction 1, forward and reverse [kJ/kmol].
    E2_f, E2_r : float
        Activation energies for reaction 2, forward and reverse [kJ/kmol].
    E3_f, E3_r : float
        Activation energies for reaction 3, forward and reverse [kJ/kmol].
    UA : float
        Overall heat transfer coefficient [kcal/(s·K)].
    V_coolant : float
        Cooling jacket volume [m³].
    rho_coolant, Cp_coolant : float
        Coolant density [kg/m³] and heat capacity [kcal/(kg·K)].

    Notes
    -----
    Twelve composite constants are pre-computed once in ``__init__`` and
    stored in ``self.params`` to avoid redundant arithmetic inside the ODE
    solver (called ~60 000 times per simulation):

    - ``Ar``                = pi * Dr**2 / 4  [m²]
    - ``E1f_R`` .. ``E3r_R``= E / R  (six Arrhenius pre-factors)
    - ``rho_oil_Cp_oil``    = rho_oil  * Cp_oil
    - ``rho_MeOH_Cp_MeOH`` = rho_MeOH * Cp_MeOH
    - ``rho_NaOH_Cp_NaOH`` = rho_NaOH * Cp_NaOH
    - ``inv_coolant_cap``   = 1 / (V_coolant * rho_coolant * Cp_coolant)
    - ``inv_rho_Ar``        = 1 / (rho * Ar)

    Keys safe for per-simulation override via ``params=`` in
    ``ct.input_output_response`` (no pre-computed dependency):
    ``k1_f``, ``k1_r``, ``k2_f``, ``k2_r``, ``k3_f``, ``k3_r``,
    ``UA``, ``Hrxn1``, ``Hrxn2``, ``Hrxn3``, ``To``.

    Examples
    --------
    >>> from model import BiodieselPlant, PROCESS_PARAMS
    >>> import control as ct
    >>> plant = BiodieselPlant(**PROCESS_PARAMS)
    >>> resp = ct.input_output_response(
    ...     plant.system, T, U, X0,
    ...     solve_ivp_kwargs={'method': 'LSODA'},
    ... )

    Sensitivity analysis without creating a new instance:

    >>> resp = ct.input_output_response(
    ...     plant.system, T, U, X0,
    ...     params={'k1_f': 0.046},
    ... )
    """

    # fmt: off
    STATE_NAMES = [
        "h", "c_TG", "c_MeOH", "c_ME", "c_DG", "c_MG",
        "c_Gly", "c_Cat", "c_Water", "T", "T_coolant"
    ]
    INPUT_NAMES = [
        "c_TG_in", "T_oil", "c_MeOH_in", "T_MeOH",
        "c_Cat_in", "c_Water_in", "T_NaOH", "T_coolant_in",
        "f_oil", "f_MeOH", "f_NaOH", "f_FAME", "f_coolant"
    ]
    # fmt: on

    # fmt: off
    def __init__(
        self,
        *,  # keyword-only — prevents silent positional-argument mistakes
        # --- Feed stream physical properties ---
        rho_oil: float,      Cp_oil: float,        # oil feed [kg/m³], [kcal/(kg·K)]
        rho_MeOH: float,     Cp_MeOH: float,       # methanol feed
        rho_NaOH: float,     Cp_NaOH: float,       # NaOH catalyst feed
        # --- Product mixture properties ---
        rho: float,          Cp: float,            # product mixture
        # --- Reactor geometry ---
        Dr: float,           Lr: float,            # reactor diameter, length [m]
        # --- Reaction thermodynamics ---
        R: float,            To: float,            # gas constant, Arrhenius ref. temp [K]
        Hrxn1: float,        Hrxn2: float,         Hrxn3: float,  # heats of rxn [kcal/kmol]
        # --- Kinetic rate constants ---
        k1_f: float,         k1_r: float,          # rate constants, reaction 1
        k2_f: float,         k2_r: float,          # rate constants, reaction 2
        k3_f: float,         k3_r: float,          # rate constants, reaction 3
        # --- Activation energies [kJ/kmol] ---
        E1_f: float,         E1_r: float,          # rxn 1
        E2_f: float,         E2_r: float,          # rxn 2
        E3_f: float,         E3_r: float,          # rxn 3
        # --- Heat transfer ---
        UA: float,           V_coolant: float,     # heat transfer coeff, jacket volume [m³]
        rho_coolant: float,  Cp_coolant: float,    # coolant properties
    ) -> None:
    # fmt: on
        Ar = np.pi * Dr**2 / 4  # reactor cross-sectional area [m²]

        # fmt: off
        self.params = {
            # --- Feed stream physical properties ---
            "rho_oil":     rho_oil,
            "Cp_oil":      Cp_oil,
            "rho_MeOH":    rho_MeOH,
            "Cp_MeOH":     Cp_MeOH,
            "rho_NaOH":    rho_NaOH,
            "Cp_NaOH":     Cp_NaOH,

            # --- Product mixture properties ---
            "rho":         rho,
            "Cp":          Cp,

            # --- Reactor geometry ---
            "Dr":          Dr,
            "Lr":          Lr,

            # --- Reaction thermodynamics ---
            "R":           R,
            "To":          To,
            "Hrxn1":       Hrxn1,
            "Hrxn2":       Hrxn2,
            "Hrxn3":       Hrxn3,

            # --- Kinetic rate constants ---
            "k1_f":        k1_f,
            "k1_r":        k1_r,
            "k2_f":        k2_f,
            "k2_r":        k2_r,
            "k3_f":        k3_f,
            "k3_r":        k3_r,

            # --- Activation energies ---
            "E1_f":        E1_f,
            "E1_r":        E1_r,
            "E2_f":        E2_f,
            "E2_r":        E2_r,
            "E3_f":        E3_f,
            "E3_r":        E3_r,

            # --- Heat transfer ---
            "UA":          UA,
            "V_coolant":   V_coolant,
            "rho_coolant": rho_coolant,
            "Cp_coolant":  Cp_coolant,

            # --- Pre-computed constants (derived; not independent inputs) ---
            "Ar":               Ar,
            "E1f_R":            E1_f / R,             # Arrhenius pre-factors E/R
            "E1r_R":            E1_r / R,
            "E2f_R":            E2_f / R,
            "E2r_R":            E2_r / R,
            "E3f_R":            E3_f / R,
            "E3r_R":            E3_r / R,
            "rho_oil_Cp_oil":   rho_oil * Cp_oil,     # feed enthalpy coefficients
            "rho_MeOH_Cp_MeOH": rho_MeOH * Cp_MeOH,
            "rho_NaOH_Cp_NaOH": rho_NaOH * Cp_NaOH,
            "inv_coolant_cap":  1.0 / (V_coolant * rho_coolant * Cp_coolant),
            "inv_rho_Ar":       1.0 / (rho * Ar),
        }
        # fmt: on
        super().__init__()

    @classmethod
    def from_defaults(cls) -> BiodieselPlant:
        """Instantiate using the default PROCESS_PARAMS from model.config."""
        from model.config import PROCESS_PARAMS

        return cls(**PROCESS_PARAMS)

    @staticmethod
    def _kinetics(
        p: dict,
        c_TG: float,
        c_DG: float,
        c_MG: float,
        c_MeOH: float,
        c_ME: float,
        c_Gly: float,
        c_Cat: float,
        arr: float,
    ) -> tuple:
        """Compute the three reversible transesterification reaction rates.

        Applies Arrhenius temperature dependence and NaOH catalytic effect
        to both forward and reverse rates of each reaction.

        Parameters
        ----------
        p : dict
            Parameter dict forwarded from ``_update``.
        c_TG, c_DG, c_MG : float
            Triglyceride, diglyceride, and monoglyceride concentrations
            [kmol/m³].
        c_MeOH, c_ME, c_Gly : float
            Methanol, methyl ester, and glycerol concentrations [kmol/m³].
        c_Cat : float
            Catalyst (NaOH) concentration [kmol/m³].
        arr : float
            Arrhenius temperature factor, ``(1/To) - (1/T_eff)`` [1/K].

        Returns
        -------
        r1, r2, r3 : float
            Net reaction rates [kmol/(m³·s)] for reactions 1, 2, and 3.
        """
        exp_1f = np.exp(p["E1f_R"] * arr)
        exp_1r = np.exp(p["E1r_R"] * arr)
        exp_2f = np.exp(p["E2f_R"] * arr)
        exp_2r = np.exp(p["E2r_R"] * arr)
        exp_3f = np.exp(p["E3f_R"] * arr)
        exp_3r = np.exp(p["E3r_R"] * arr)

        r1 = (
            p["k1_f"] * exp_1f * c_TG * c_MeOH * c_Cat
            - p["k1_r"] * exp_1r * c_ME * c_DG
        )
        r2 = (
            p["k2_f"] * exp_2f * c_DG * c_MeOH * c_Cat
            - p["k2_r"] * exp_2r * c_ME * c_MG
        )
        r3 = (
            p["k3_f"] * exp_3f * c_MG * c_MeOH * c_Cat
            - p["k3_r"] * exp_3r * c_ME * c_Gly
        )
        return r1, r2, r3

    def _update(
        self,
        t: float,
        x: np.ndarray,
        u: np.ndarray,
        params: dict,
    ) -> list:
        """Compute state derivatives for the biodiesel reactor.

        Implements mole balances for 9 components, a total mass
        balance, and energy balances for the reactor and cooling jacket.
        """
        # --- Unpack states ---
        h, c_TG, c_MeOH, c_ME, c_DG, c_MG, c_Gly, c_Cat, c_Water, T, T_coolant = x

        # --- Unpack inputs ---
        # fmt: off
        (c_TG_in, T_oil, c_MeOH_in, T_MeOH,
         c_Cat_in, c_Water_in, T_NaOH, T_coolant_in,
         f_oil, f_MeOH, f_NaOH, f_FAME, f_coolant) = u
        # fmt: on

        p = params  # params dict forwarded by ct.NonlinearIOSystem

        # --- Numerical safeguards ---
        h_eff = max(h, 1e-8)  # guard against zero reactor volume
        T_eff = max(T, 1.0)  # guard against Arrhenius singularity

        # --- Flow and volume ---
        volume = p["Ar"] * h_eff
        inflow_mass = (
            f_oil * p["rho_oil"] + f_MeOH * p["rho_MeOH"] + f_NaOH * p["rho_NaOH"]
        )
        inflow = inflow_mass / p["rho"]

        # --- Arrhenius temperature factor ---
        arr = (1.0 / p["To"]) - (1.0 / T_eff)

        # --- Reaction rates ---
        r1, r2, r3 = self._kinetics(
            p, c_TG, c_DG, c_MG, c_MeOH, c_ME, c_Gly, c_Cat, arr
        )

        # --- Total mass balance ---
        dh_dt = (inflow_mass - f_FAME * p["rho"]) * p["inv_rho_Ar"]

        # --- Species mole balances ---
        dc_TG_dt = (f_oil * c_TG_in - inflow * c_TG - volume * r1) / volume
        dc_MeOH_dt = (
            f_MeOH * c_MeOH_in - inflow * c_MeOH - volume * (r1 + r2 + r3)
        ) / volume
        dc_ME_dt = (volume * (r1 + r2 + r3) - inflow * c_ME) / volume
        dc_DG_dt = (volume * (r1 - r2) - inflow * c_DG) / volume
        dc_MG_dt = (volume * (r2 - r3) - inflow * c_MG) / volume
        dc_Gly_dt = (volume * r3 - inflow * c_Gly) / volume
        dc_Cat_dt = (f_NaOH * c_Cat_in - inflow * c_Cat) / volume
        dc_Water_dt = (f_NaOH * c_Water_in - inflow * c_Water) / volume

        # --- Reactor energy balance ---
        dT_dt = (
            f_oil * p["rho_oil_Cp_oil"] * T_oil
            + f_MeOH * p["rho_MeOH_Cp_MeOH"] * T_MeOH
            + f_NaOH * p["rho_NaOH_Cp_NaOH"] * T_NaOH
            - p["UA"] * (T_eff - T_coolant)
            - inflow_mass * p["Cp"] * T_eff
            - volume * (r1 * p["Hrxn1"] + r2 * p["Hrxn2"] + r3 * p["Hrxn3"])
        ) / (p["Ar"] * p["rho"] * p["Cp"] * h_eff)

        # --- Jacket energy balance ---
        dT_coolant_dt = (
            f_coolant * p["rho_coolant"] * p["Cp_coolant"] * (T_coolant_in - T_coolant)
            + p["UA"] * (T_eff - T_coolant)
        ) * p["inv_coolant_cap"]

        return [
            dh_dt,
            dc_TG_dt,
            dc_MeOH_dt,
            dc_ME_dt,
            dc_DG_dt,
            dc_MG_dt,
            dc_Gly_dt,
            dc_Cat_dt,
            dc_Water_dt,
            dT_dt,
            dT_coolant_dt,
        ]
