"""FOPDT (First-Order Plus Dead Time) model identification from step response data.

Provides FOPDTModel, a dataclass for FOPDT parameters, and FOPDTIdentifier,
which fits those parameters to measured step response data using either a
classic two-point graphical method or a numerical optimization.
"""

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import differential_evolution, least_squares

# =============================================================================
# FOPDT parameter container
# =============================================================================


@dataclass
class FOPDTModel:
    """FOPDT process model parameters.

    Represents the transfer function:
    ``Gp(s) = K / (tau*s + 1) * exp(-t0*s)``

    Parameters
    ----------
    K : float
        Process gain [output units / input units].
    tau : float
        Process time constant [s].
    t0 : float
        Process dead time [s].
    """

    K: float
    tau: float
    t0: float

    def __str__(self) -> str:
        return (
            f'K={self.K:.4f}, tau={self.tau:.2f} s, t0={self.t0:.2f} s, '
            f't0/tau={self.t0 / self.tau:.3f}'
        )


# =============================================================================
# Identifier
# =============================================================================


class FOPDTIdentifier:
    """Identify FOPDT parameters from measured step response data.

    Automatically detects the largest step change in the input signal,
    estimates steady-state values before and after the step, and supports
    two identification strategies: a classic graphical method and a
    numerical optimization.

    Parameters
    ----------
    time : ndarray
        Time vector [s].
    input_data : ndarray
        Input (manipulated variable) signal.
    output_data : ndarray
        Output (process variable) signal.
    verbose : bool, optional
        Print progress details during identification.  Default: False.
    """

    def __init__(
        self,
        time: np.ndarray,
        input_data: np.ndarray,
        output_data: np.ndarray,
        verbose: bool = False,
    ) -> None:
        self.time = np.asarray(time)
        self.input = np.asarray(input_data)
        self.output = np.asarray(output_data)
        self.verbose = verbose

        self._detect_step()
        self._normalize_signals()

    def _detect_step(self) -> None:
        """Detect step change time, direction, and magnitude from the input."""
        input_diff = np.diff(self.input)
        max_change_idx = int(np.argmax(np.abs(input_diff)))

        self.step_time = self.time[max_change_idx]
        self.step_magnitude = self.input[max_change_idx + 1] - self.input[max_change_idx]
        self.step_direction = 'up' if self.step_magnitude > 0 else 'down'

        if self.verbose:
            print(
                f'Step detected: {self.step_direction} '
                f'by {self.step_magnitude:.2f} at t={self.step_time:.2f} s'
            )

    def _normalize_signals(self) -> None:
        """Compute initial and final steady-state output values."""
        before_mask = self.time < self.step_time
        self.y_initial = float(np.mean(self.output[before_mask]))

        after_mask = self.time >= self.step_time
        if np.any(after_mask):
            last_10_pct = max(int(0.1 * int(np.sum(after_mask))), 1)
            self.y_final = float(np.mean(self.output[after_mask][-last_10_pct:]))
        else:
            self.y_final = float(self.output[-1])

        self.delta_y = self.y_final - self.y_initial

    def _fopdt_response(
        self,
        params: np.ndarray,
        time: np.ndarray,
    ) -> np.ndarray:
        """Simulate FOPDT step response.

        Parameters
        ----------
        params : ndarray, shape (3,)
            Parameter vector [K, tau, t0].
        time : ndarray
            Time vector [s].

        Returns
        -------
        ndarray
            Simulated output, same shape as ``time``.
        """
        K, tau, t0 = params
        response = np.zeros_like(time, dtype=float)

        for i, t in enumerate(time):
            if t < self.step_time + t0:
                response[i] = self.y_initial
            else:
                elapsed = t - self.step_time - t0
                response[i] = self.y_initial + K * self.step_magnitude * (
                    1.0 - np.exp(-elapsed / tau)
                )

        return response

    def _cost_function(self, params: np.ndarray) -> float:
        """Sum of squared errors between model prediction and measured output.

        Parameters
        ----------
        params : ndarray, shape (3,)
            Parameter vector [K, tau, t0].

        Returns
        -------
        float
            SSE value (returns 1e10 for physically invalid parameters).
        """
        K, tau, t0 = params
        if tau <= 0 or K <= 0 or t0 < 0:
            return 1e10

        valid_mask = self.time > self.step_time
        model = self._fopdt_response(params, self.time[valid_mask])
        actual = self.output[valid_mask]

        return float(np.sum((actual - model) ** 2))

    def identify_two_point(self) -> FOPDTModel:
        """Estimate FOPDT parameters using the classic 28.3 % – 63.2 % method.

        Returns
        -------
        FOPDTModel
            Estimated process parameters.
        """
        post_step = self.time > self.step_time
        t_post = self.time[post_step]
        y_post = self.output[post_step]

        y_283 = self.y_initial + 0.283 * self.delta_y
        y_632 = self.y_initial + 0.632 * self.delta_y

        if len(t_post) > 1 and y_post.min() < y_283 < y_post.max():
            if self.delta_y < 0:
                # For downward step, sort y_post to be increasing for np.interp
                sort_idx = np.argsort(y_post)
                t1 = float(np.interp(y_283, y_post[sort_idx], t_post[sort_idx]))
                t2 = float(np.interp(y_632, y_post[sort_idx], t_post[sort_idx]))
            else:
                t1 = float(np.interp(y_283, y_post, t_post))
                t2 = float(np.interp(y_632, y_post, t_post))
        else:
            idx1 = int(np.argmin(np.abs(y_post - y_283)))
            idx2 = int(np.argmin(np.abs(y_post - y_632)))
            t1, t2 = float(t_post[idx1]), float(t_post[idx2])

        tau = (3.0 / 2.0) * (t2 - t1)
        t0 = t2 - tau
        K = self.delta_y / self.step_magnitude

        return FOPDTModel(K=K, tau=max(tau, 0.1), t0=max(t0, 0.0))

    def identify_optimization(self, method: str = 'fast') -> FOPDTModel:
        """Optimize FOPDT parameters by minimizing fit error.

        Parameters
        ----------
        method : {'fast', 'robust'}, optional
            ``'fast'``   — Levenberg-Marquardt via ``scipy.least_squares``
            (default, typically < 1 s).
            ``'robust'`` — Global differential evolution for noisy data
            (typically 5 – 10 s).

        Returns
        -------
        FOPDTModel
            Optimized FOPDT parameters.
        """
        initial = self.identify_two_point()
        x0 = [initial.K, initial.tau, initial.t0]

        if self.verbose:
            print(f'Initial guess (two-point): {initial}')

        if method == 'robust':
            k_bounds = (min(0.1 * x0[0], 10.0 * x0[0]), max(0.1 * x0[0], 10.0 * x0[0]))
            if k_bounds[0] == k_bounds[1]:
                k_bounds = (-1.0, 1.0)

            bounds = [
                k_bounds,
                (0.01 * x0[1], 100.0 * x0[1]),
                (0.0, max(2.0 * x0[2], 1.0)),
            ]
            result = differential_evolution(
                self._cost_function,
                bounds,
                maxiter=300,
                tol=1e-5,
                workers=1,
                updating='deferred',
                polish=True,
            )
        else:
            valid_mask = self.time > self.step_time
            t_data = self.time[valid_mask]
            y_data = self.output[valid_mask]

            def residuals(params: np.ndarray) -> np.ndarray:
                return y_data - self._fopdt_response(params, t_data)

            k_min = min(100.0 * x0[0], 0.01 * x0[0], -0.01)
            k_max = max(100.0 * x0[0], 0.01 * x0[0], 0.01)

            result = least_squares(
                residuals,
                x0,
                bounds=(
                    [k_min, 0.01 * x0[1], 0.0],
                    [k_max, 100.0 * x0[1], max(2.0 * x0[2], 0.1)],
                ),
                method='trf',
                ftol=1e-8,
                xtol=1e-8,
                gtol=1e-8,
                max_nfev=1000,
            )

        K, tau, t0 = result.x

        if self.verbose:
            print(f'Optimization result: K={K:.4f}, tau={tau:.2f} s, t0={t0:.2f} s')
            if hasattr(result.fun, '__len__'):
                cost = float(np.sum(result.fun**2))
            else:
                cost = float(result.fun)
            print(f'Cost: {cost:.6e}')

        return FOPDTModel(K=K, tau=tau, t0=t0)

    def calculate_fit_quality(self, model: FOPDTModel) -> dict:
        """Compute goodness-of-fit metrics for a fitted FOPDT model.

        Parameters
        ----------
        model : FOPDTModel
            FOPDT model to evaluate.

        Returns
        -------
        dict
            Keys: ``'R2'`` (coefficient of determination),
            ``'RMSE'`` (root mean squared error), ``'SSE'`` (sum of squared
            errors).  Metrics are computed on the post-step window only.
        """
        model_response = self._fopdt_response(np.array([model.K, model.tau, model.t0]), self.time)

        valid_mask = self.time > self.step_time
        y_actual = self.output[valid_mask]
        y_model = model_response[valid_mask]

        ss_res = float(np.sum((y_actual - y_model) ** 2))
        ss_tot = float(np.sum((y_actual - np.mean(y_actual)) ** 2))
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 0.0
        rmse = float(np.sqrt(np.mean((y_actual - y_model) ** 2)))

        return {'R2': r_squared, 'RMSE': rmse, 'SSE': ss_res}


# =============================================================================
# Convenience wrapper
# =============================================================================


def identify_fopdt(
    time: np.ndarray,
    input_data: np.ndarray,
    output_data: np.ndarray,
    method: str = 'fast',
    verbose: bool = False,
) -> tuple[FOPDTModel, dict, FOPDTIdentifier]:
    """Identify FOPDT model parameters from a step response experiment.

    Parameters
    ----------
    time : ndarray
        Time vector [s].
    input_data : ndarray
        Input (manipulated variable) signal.
    output_data : ndarray
        Output (process variable) signal.
    method : {'two-point', 'fast', 'robust'}, optional
        Identification strategy:
        ``'two-point'`` — classic 28.3 % – 63.2 % graphical method (~1 ms).
        ``'fast'``      — Levenberg-Marquardt optimization (~500 ms, default).
        ``'robust'``    — Global optimization for noisy data (~5 – 10 s).
    verbose : bool, optional
        Print identification details.  Default: False.

    Returns
    -------
    model : FOPDTModel
        Identified FOPDT parameters.
    metrics : dict
        Fit quality metrics (``'R2'``, ``'RMSE'``, ``'SSE'``).
    identifier : FOPDTIdentifier
        Identifier instance (exposes ``_fopdt_response`` for visualization).

    Examples
    --------
    >>> model, metrics, ident = identify_fopdt(time, u, y, method='fast')
    >>> print(model)
    >>> print(f"R2 = {metrics['R2']:.4f}")
    """
    identifier = FOPDTIdentifier(time, input_data, output_data, verbose=verbose)

    if method == 'two-point':
        model = identifier.identify_two_point()
    else:
        model = identifier.identify_optimization(method=method)

    metrics = identifier.calculate_fit_quality(model)

    if verbose:
        print(f'\nFOPDT Model: {model}')
        print(f'Fit Quality — R2: {metrics["R2"]:.4f}, RMSE: {metrics["RMSE"]:.4f}')

    return model, metrics, identifier


# =============================================================================
# Visualization
# =============================================================================


def plot_fopdt_fit(
    identifier: FOPDTIdentifier,
    model: FOPDTModel,
    title: str = 'FOPDT Model Identification',
):
    """Plot actual vs FOPDT model response with a residuals panel.

    Parameters
    ----------
    identifier : FOPDTIdentifier
        Identifier instance holding measured data and step metadata.
    model : FOPDTModel
        Fitted FOPDT model.
    title : str, optional
        Main plot title.  Default: ``'FOPDT Model Identification'``.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure with two subplots: response comparison (left) and
        fit residuals (right).
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    model_response = identifier._fopdt_response(
        np.array([model.K, model.tau, model.t0]), identifier.time
    )

    # --- Response comparison ---
    ax1.plot(identifier.time, identifier.output, 'b-', linewidth=2, label='Actual')
    ax1.plot(identifier.time, model_response, 'r--', linewidth=2, label='FOPDT Model')
    ax1.axvline(identifier.step_time, color='gray', linestyle=':', alpha=0.5)
    ax1.axvline(
        identifier.step_time + model.t0,
        color='orange',
        linestyle=':',
        alpha=0.5,
        label='Dead Time',
    )
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Output')
    # ax1.set_title(title)  # no title per publication style
    ax1.legend(frameon=False)
    ax1.grid(True, alpha=0.3)

    # --- Residuals ---
    valid_mask = identifier.time > identifier.step_time
    residuals = identifier.output[valid_mask] - model_response[valid_mask]
    ax2.plot(identifier.time[valid_mask], residuals, 'g-', linewidth=1.5)
    ax2.axhline(0, color='k', linestyle='--', alpha=0.3)
    ax2.fill_between(identifier.time[valid_mask], residuals, alpha=0.3, color='green')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Residual (Actual - Model)')
    # ax2.set_title("Fit Residuals")  # no title per publication style
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig
