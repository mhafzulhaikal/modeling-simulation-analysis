"""Step response performance metrics for nonlinear closed-loop simulations.

Provides StepMetrics (dataclass) and StepInfo, which compute IAE, ISE, ITAE,
ITSE, rise time, settling time, overshoot, undershoot, decay ratio, and peak
directly from (time, y) arrays produced by ct.input_output_response.
"""

from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass

import numpy as np
from numpy import trapezoid as _trapz
from scipy.signal import find_peaks

# =============================================================================
# Metrics container
# =============================================================================


@dataclass
class StepMetrics:
    """Container for step-response performance metrics.

    All values are ``float``; unavailable metrics are stored as ``np.nan``.
    Overshoot and Undershoot are in percent [%]; all time metrics are in
    seconds [s]; integral indices are in the natural units of the signal.
    """

    RiseTime: float = np.nan
    SettlingTime: float = np.nan
    SettlingMin: float = np.nan
    SettlingMax: float = np.nan
    Overshoot: float = np.nan  # [%]
    Undershoot: float = np.nan  # [%]
    DecayRatio: float = np.nan
    Peak: float = np.nan
    PeakTime: float = np.nan
    IAE: float = np.nan  # Integral Absolute Error
    ISE: float = np.nan  # Integral Squared Error
    ITAE: float = np.nan  # Integral Time-weighted Absolute Error
    ITSE: float = np.nan  # Integral Time-weighted Squared Error
    SteadyStateValue: float = np.nan

    def __post_init__(self) -> None:
        """Cast all fields to float; replace infinities with nan."""
        for field_name in self.__dataclass_fields__:
            val = getattr(self, field_name)
            if isinstance(val, (int, float, np.number)):
                if np.isinf(float(val)):
                    setattr(self, field_name, np.nan)
                else:
                    setattr(self, field_name, float(val))


# =============================================================================
# Main class
# =============================================================================


class StepInfo:
    """Compute step-response performance metrics from nonlinear simulation data.

    Mirrors the behaviour of ``ct.step_info()`` but operates on (time, y) arrays
    produced by a nonlinear closed-loop simulation (e.g. ``ct.input_output_response``).
    An optional evaluation window [step_time, end_time] limits all metrics to a
    specific portion of the response.

    Parameters
    ----------
    time : ndarray
        Full time vector [s].
    y : ndarray
        Process variable (output) array, same length as ``time``.
    u : ndarray or float
        Input / setpoint signal (scalar is broadcast to the full time vector).
    y_final : float, optional
        Known steady-state value.  If ``None``, the last value in the
        evaluation window is used.
    y_initial : float, optional
        Value of PV just before the step.  If ``None``, the value at
        ``time[i0 - 1]`` is used.
    step_time : float, optional
        Time at which the step is applied [s].  Default: 0.
    end_time : float, optional
        End of the evaluation window [s].  If ``None``, the full array is used.
    SettlingTimeThreshold : float, optional
        Relative error band for settling time (e.g. 0.01 → ±1 %).
        Default: 0.01.
    RiseTimeLimits : tuple of float, optional
        (lower, upper) response fractions for rise time.
        Default: (0.1, 0.9).

    Attributes
    ----------
    metrics : StepMetrics
        Computed performance metrics (populated in ``__init__``).

    Examples
    --------
    >>> si = StepInfo(time=t, y=T, u=sp, y_final=340.15,
    ...               y_initial=333.15, step_time=600)
    >>> print(si)
    >>> si['RiseTime']          # dict-style access
    >>> si.to_dict()
    """

    def __init__(
        self,
        time: np.ndarray,
        y: np.ndarray,
        u: np.ndarray | float,
        y_final: float | None = None,
        y_initial: float | None = None,
        step_time: float = 0.0,
        end_time: float | None = None,
        SettlingTimeThreshold: float = 0.01,
        RiseTimeLimits: tuple[float, float] = (0.1, 0.9),
    ) -> None:
        # --- store inputs ---
        self.time = np.asarray(time, dtype=float)
        self.y = np.asarray(y, dtype=float)
        self.step_time = float(step_time)
        self.end_time = float(end_time) if end_time is not None else None
        self.SettlingTimeThreshold = SettlingTimeThreshold
        self.RiseTimeLimits = RiseTimeLimits

        if np.ndim(u) == 0:
            self.u = np.full_like(self.time, float(u), dtype=float)
        else:
            self.u = np.asarray(u, dtype=float)
            if self.u.shape != self.time.shape:
                raise ValueError(
                    f'input shape {self.u.shape} must match time shape {self.time.shape}'
                )

        # --- evaluation window indices ---
        self._i0 = int(np.searchsorted(self.time, self.step_time))
        if self.end_time is not None:
            self._i1 = int(np.searchsorted(self.time, self.end_time, side='right'))
        else:
            self._i1 = len(self.time)

        self._T = self.time[self._i0 : self._i1]
        self._y = self.y[self._i0 : self._i1]

        # --- reference values ---
        if y_final is not None:
            self.yfinal = float(y_final)
        else:
            self.yfinal = float(self._y[-1]) if len(self._y) > 0 else float('nan')

        if y_initial is not None:
            self.y_initial = float(y_initial)
        else:
            self.y_initial = float(self.y[max(0, self._i0 - 1)])

        # --- compute all metrics ---
        self.metrics = self._compute()

    # --- core computation ---

    def _compute(self) -> StepMetrics:
        """Compute all step-response metrics and return a StepMetrics instance.

        Returns
        -------
        StepMetrics
            Populated metrics container.
        """
        m = StepMetrics()

        yout_post = self._y
        T_post = self._T

        if len(T_post) == 0:
            warnings.warn(
                'Evaluation window is empty — check step_time and end_time.',
                stacklevel=3,
            )
            return m

        InfValue = self.yfinal
        InitialValue = self.y_initial
        delta = InfValue - InitialValue

        if np.isnan(InfValue) or np.isinf(InfValue):
            warnings.warn('yfinal is NaN or Inf — returning NaN metrics.', stacklevel=3)
            return m

        t_relative = T_post - self.step_time

        # --- rise time ---
        target_lower = InitialValue + self.RiseTimeLimits[0] * delta
        target_upper = InitialValue + self.RiseTimeLimits[1] * delta

        if delta > 0:
            condition_lower = yout_post >= target_lower
            condition_upper = yout_post >= target_upper
        else:
            condition_lower = yout_post <= target_lower
            condition_upper = yout_post <= target_upper

        has_lower = np.any(condition_lower)
        has_upper = np.any(condition_upper)

        if has_lower:
            tr_lower_index = int(np.argmax(condition_lower))
        else:
            tr_lower_index = 0

        if has_upper:
            tr_upper_index = int(np.argmax(condition_upper))
        else:
            tr_upper_index = 0

        if delta == 0 or not has_lower or not has_upper:
            m.RiseTime = np.nan
        else:
            m.RiseTime = T_post[tr_upper_index] - T_post[tr_lower_index]

        # --- settling time ---
        error_abs = np.abs(yout_post - InfValue)

        if delta == 0:
            outside = np.nonzero(error_abs >= self.SettlingTimeThreshold * np.abs(InfValue))[0]
        else:
            outside = np.nonzero(error_abs >= self.SettlingTimeThreshold * np.abs(delta))[0]

        if outside.size == 0:
            m.SettlingTime = 0.0
        else:
            settled_idx = outside[-1] + 1
            if settled_idx < len(T_post):
                m.SettlingTime = float(t_relative[settled_idx])
            else:
                m.SettlingTime = np.nan

        # --- settling min and max ---
        post_rise = yout_post[tr_upper_index:]
        if len(post_rise) > 0:
            m.SettlingMin = float(np.minimum(post_rise.min(), InfValue))
            m.SettlingMax = float(np.maximum(post_rise.max(), InfValue))
        else:
            m.SettlingMin = InfValue
            m.SettlingMax = InfValue

        # --- overshoot and undershoot ---
        y_max = float(np.max(yout_post))
        y_min = float(np.min(yout_post))

        if delta > 0:
            dy_os = y_max - InfValue
            dy_us = InitialValue - y_min
        elif delta == 0:
            dy_os = y_max - InfValue
            dy_us = InfValue - y_min
        else:
            dy_os = InfValue - y_min
            dy_us = y_max - InitialValue

        if delta == 0:
            m.Overshoot = abs(100.0 * dy_os / InfValue) if (dy_os > 0 and InfValue != 0) else 0.0
            m.Undershoot = abs(100.0 * dy_us / InfValue) if (dy_us > 0 and InfValue != 0) else 0.0
        else:
            m.Overshoot = abs(100.0 * dy_os / delta) if dy_os > 0 else 0.0
            m.Undershoot = abs(100.0 * dy_us / delta) if dy_us > 0 else 0.0

        # --- peak value and peak time ---
        if delta > 0:
            peak_index = int(np.argmax(yout_post))
        elif delta == 0:
            peak_index = int(np.argmax(np.abs(yout_post - InfValue)))
        else:
            peak_index = int(np.argmin(yout_post))

        m.Peak = float(yout_post[peak_index])
        m.PeakTime = float(t_relative[peak_index])

        # --- decay ratio ---
        peak_indices, _ = find_peaks(yout_post) if delta > 0 else find_peaks(-yout_post)

        valid_peaks = []
        for p in peak_indices:
            val = float(yout_post[p])
            if (delta > 0 and val > InfValue) or (delta < 0 and val < InfValue) or (delta == 0):
                valid_peaks.append(val)

        if len(valid_peaks) >= 2:
            amp1 = abs(valid_peaks[0] - InfValue)
            amp2 = abs(valid_peaks[1] - InfValue)
            m.DecayRatio = (amp2 / amp1) if amp1 > 0 else np.nan
        else:
            m.DecayRatio = np.nan

        # --- integral performance indices ---
        m.IAE = float(_trapz(error_abs, t_relative))
        m.ITAE = float(_trapz(t_relative * error_abs, t_relative))
        m.ISE = float(_trapz(error_abs**2, t_relative))
        m.ITSE = float(_trapz(t_relative * error_abs**2, t_relative))

        m.SteadyStateValue = InfValue

        return m

    # --- convenience accessors ---

    def __getitem__(self, key: str) -> float:
        return asdict(self.metrics)[key]

    def __repr__(self) -> str:
        W = 12  # value column width (characters)
        L = 18  # label column width (characters)
        SEP = '=' * 62
        DASH = '-' * 58

        def _na(v: float) -> bool:
            return np.isnan(v) or np.isinf(v)

        def _fmt_time(v: float) -> str:
            return '---'.rjust(W) if _na(v) else f'{v:{W}.2f}'

        def _fmt_pct(v: float) -> str:
            return '---'.rjust(W) if _na(v) else f'{v:{W}.4f}'

        def _fmt_val(v: float) -> str:
            return '---'.rjust(W) if _na(v) else f'{v:{W}.6g}'

        def _fmt_norm(v: float) -> str:
            return '---'.rjust(W) if _na(v) else f'{v:{W}.5f}'

        def _row(label: str, val: str, suffix: str = '') -> str:
            return f'  {label:<{L}}: {val}  {suffix}'.rstrip()

        m = self.metrics

        thresh_str = f'+/- {self.SettlingTimeThreshold * 100:.2f} %'
        lo_pct = int(round(self.RiseTimeLimits[0] * 100))
        hi_pct = int(round(self.RiseTimeLimits[1] * 100))
        rise_str = f'({lo_pct} % - {hi_pct} %)'
        end_part = f'    end_time  =  {self.end_time:.2f} s' if self.end_time else ''
        peak_at = f'at {m.PeakTime:.2f} s' if not _na(m.PeakTime) else ''

        lines = [
            SEP,
            '  StepInfo',
            f'  step_time  =  {self.step_time:.2f} s{end_part}',
            f'  y_initial  =  {self.y_initial:.6g}    ->    y_final  =  {self.yfinal:.6g}',
            SEP,
            '',
            '  TIME-DOMAIN METRICS',
            f'  {DASH}',
            _row('Rise Time', _fmt_time(m.RiseTime), f's    {rise_str}'),
            _row('Settling Time', _fmt_time(m.SettlingTime), f's    ({thresh_str})'),
            _row('Peak', _fmt_val(m.Peak), peak_at),
            _row('Overshoot', _fmt_pct(m.Overshoot), '%'),
            _row('Undershoot', _fmt_pct(m.Undershoot), '%'),
            _row('Decay Ratio', _fmt_pct(m.DecayRatio)),
            _row('Settling Min', _fmt_val(m.SettlingMin)),
            _row('Settling Max', _fmt_val(m.SettlingMax)),
            _row('Steady State', _fmt_val(m.SteadyStateValue)),
            '',
            '  INTEGRAL PERFORMANCE INDICES',
            f'  {DASH}',
            _row('IAE', _fmt_val(m.IAE)),
            _row('ISE', _fmt_val(m.ISE)),
            _row('ITAE', _fmt_val(m.ITAE)),
            _row('ITSE', _fmt_val(m.ITSE)),
        ]

        lines.append(SEP)
        return '\n'.join(lines)

    def to_dict(self) -> dict:
        """Return metrics as a plain dict."""
        return asdict(self.metrics)

    def normalized_metrics(self) -> dict:
        """Return normalized metrics scaled to [0, 1] for cross-loop comparison.

        Returns
        -------
        dict
            Normalized IAE, ISE, ITAE, ITSE, and a composite
            ``'PerformanceIndex'`` in [0, 100] (lower is better).
            Returns an empty dict when the evaluation window or step
            magnitude is degenerate.
        """
        if np.isnan(self.metrics.IAE) or np.isnan(self.metrics.SettlingTime):
            return {}

        time_window = self._T[-1] - self._T[0]
        delta = abs(self.yfinal - self.y_initial)

        is_zero_delta = delta == 0
        ref_value = abs(self.yfinal) if (is_zero_delta and self.yfinal != 0) else delta

        if ref_value == 0 or time_window == 0:
            return {}

        norm = {
            'IAE_normalized': self.metrics.IAE / (time_window * ref_value),
            'ISE_normalized': (
                self.metrics.ISE / (time_window * ref_value**2)
                if not np.isnan(self.metrics.ISE)
                else np.nan
            ),
            'ITAE_normalized': (
                self.metrics.ITAE / (time_window**2 * ref_value)
                if not np.isnan(self.metrics.ITAE)
                else np.nan
            ),
            'ITSE_normalized': (
                self.metrics.ITSE / (time_window**2 * ref_value**2)
                if not np.isnan(self.metrics.ITSE)
                else np.nan
            ),
        }

        # cap each term at 1.0; NaN propagates as maximum penalty
        overshoot_norm = (
            1.0 if np.isnan(self.metrics.Overshoot) else min(self.metrics.Overshoot / 100.0, 1.0)
        )
        settling_norm = (
            1.0
            if np.isnan(self.metrics.SettlingTime)
            else min(self.metrics.SettlingTime / time_window, 1.0)
        )
        iae_raw = norm.get('IAE_normalized', np.nan)
        iae_norm = 1.0 if np.isnan(iae_raw) else min(iae_raw, 1.0)

        perf_cost = (0.4 * overshoot_norm + 0.3 * settling_norm + 0.3 * iae_norm) * 100.0
        norm['PerformanceIndex'] = float(np.clip(perf_cost, 0.0, 100.0))

        return norm

    def to_dataframe(self):
        """Return metrics as a one-row pandas DataFrame.

        Returns
        -------
        pandas.DataFrame
            Single-row DataFrame with one column per metric.

        Raises
        ------
        ImportError
            If pandas is not installed.
        """
        try:
            import pandas as pd

            return pd.DataFrame([asdict(self.metrics)])
        except ImportError:
            raise ImportError(
                'pandas is required for to_dataframe(). Install it with: pip install pandas'
            )

    # --- class-level factories ---

    @classmethod
    def from_loops(
        cls,
        time: np.ndarray,
        loops: dict[str, dict],
    ) -> dict[str, StepInfo | None]:
        """Construct a StepInfo for each loop from a shared time vector.

        Parameters
        ----------
        time : ndarray
            Shared time vector [s].
        loops : dict
            Mapping of loop name → keyword argument dict passed to
            ``StepInfo.__init__`` (excluding ``time``).

        Returns
        -------
        dict
            Mapping of loop name → ``StepInfo`` (or ``None`` on failure).
        """
        results: dict[str, StepInfo | None] = {}
        for name, kwargs in loops.items():
            try:
                results[name] = cls(time=time, **kwargs)
            except Exception as exc:
                warnings.warn(f'[{name}] failed: {exc}', stacklevel=2)
                results[name] = None
        return results

    @classmethod
    def summary_dataframe(
        cls,
        results: dict[str, StepInfo | None],
    ):
        """Build a multi-index summary DataFrame from a dict of StepInfo objects.

        Parameters
        ----------
        results : dict
            Mapping of loop name → ``StepInfo`` (``None`` entries are skipped).

        Returns
        -------
        pandas.DataFrame
            Rows are loop names; columns are metric names.

        Raises
        ------
        ImportError
            If pandas is not installed.
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError('pandas is required. pip install pandas')

        rows = {}
        for name, si in results.items():
            rows[name] = asdict(si.metrics) if si is not None else {}
        return pd.DataFrame(rows).T
