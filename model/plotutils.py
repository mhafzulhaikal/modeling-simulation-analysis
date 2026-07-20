"""Publication-quality plotting utilities for step response analysis.

Provides functions to create professional plots with rise time and settling time indicators
suitable for academic publications.
"""

import os

import matplotlib.pyplot as plt
import numpy as np


def _save_fig(fig, path, dpi=600):
    """Save a figure to *path*, always adding a visible outer border frame.

    If the path ends with ``.svg``, the figure is saved as a lossless SVG
    vector file.  For all other extensions the figure is saved at the
    requested DPI (default 600).
    A thin rectangular border is drawn around the entire figure area so that
    every exported plot has a consistent outer frame.
    """
    import matplotlib.patches as mpatches

    # Draw a border rectangle just inside the figure edges so it is fully
    # visible even after bbox_inches='tight' trimming.
    border = mpatches.Rectangle(
        xy=(0.005, 0.005),
        width=0.990,
        height=0.990,
        linewidth=1.2,
        edgecolor='#1a1a1a',
        facecolor='none',
        transform=fig.transFigure,
        clip_on=False,
        zorder=10,
    )
    fig.add_artist(border)

    ext = os.path.splitext(str(path))[1].lower()
    if ext == '.svg':
        fig.savefig(path, format='svg', bbox_inches='tight', pad_inches=0.05)
    else:
        fig.savefig(path, dpi=dpi, bbox_inches='tight', pad_inches=0.05)


class _BoxConfig:
    """Configuration for box dimensions and margins (centralized for consistency)."""

    def __init__(self, y_span, extra_margin=0):
        self.y_span = y_span
        self.extra_margin = extra_margin

        # Box dimensions
        self.box_height = y_span * 0.08
        self.box_width_estimate = y_span * 0.15  # Estimated width for collision detection

        # Margins (improved spacing)
        self.margin_from_data = y_span * (0.03 + extra_margin * 0.05)  # 3% + extra
        self.margin_from_axis = y_span * (0.06 + extra_margin * 0.03)  # 6% + extra
        self.margin_between_boxes = y_span * (
            0.12 + extra_margin * 0.08
        )  # 12% + extra (increased for better spacing)

        # Collision detection threshold
        self.min_x_separation = y_span * 0.2


def _boxes_collide(box1, box2, x_threshold=None):
    """Check if two boxes collide in 2D space (x and y).

    Parameters
    ----------
    box1, box2 : dict with keys ('x', 'y_center', 'height', 'width')
        Box positions
    x_threshold : float, optional
        Horizontal distance for collision (if None, no x-separation check)

    Returns
    -------
    bool : True if boxes collide
    """
    # X-axis check
    if x_threshold is not None:
        if abs(box1['x'] - box2['x']) > x_threshold:
            return False

    # Y-axis check (boxes with center y1, y2 and height h)
    box1_top = box1['y_center'] + box1['height'] / 2
    box1_bottom = box1['y_center'] - box1['height'] / 2
    box2_top = box2['y_center'] + box2['height'] / 2
    box2_bottom = box2['y_center'] - box2['height'] / 2

    # Check vertical overlap
    y_overlap = not (box1_top < box2_bottom or box2_top < box1_bottom)
    return y_overlap


def _resolve_box_y(box_y, idx, default):
    """Resolve box y-position from scalar, list/array, or None.

    Parameters
    ----------
    box_y : float, list, or None
        User-supplied value. If scalar, used for all boxes.
        If list/array, index ``idx`` is used (falls back to default if out of range).
        If None, returns ``default``.
    idx : int
        Index of the current box in the iteration.
    default : float
        Auto-calculated fallback value.

    Returns
    -------
    float
    """
    if box_y is None:
        return default
    if isinstance(box_y, (list, tuple, np.ndarray)):  # list / array / sequence
        if idx < len(box_y):
            return box_y[idx]
        return default  # out of range → auto
    return box_y  # scalar → same for all boxes


def _calculate_box_positions(
    tr_x,
    ts_x,
    y_data,
    y_sp_data,
    y_min,
    y_max,
    step_time,
    settling_threshold=0.02,
    min_x_separation=None,
    extra_margin=0,
):
    """Calculate smart positions for tR and tS boxes with improved collision handling.

    Design improvements:
    - Better collision detection using bounding boxes
    - tR positioned below data with margin from axis
    - tS positioned above data with margin from axis
    - Vlines positioned to avoid crossing boxes
    - Handles cases where boxes are too close (x-wise)

    Parameters
    ----------
    tr_x, ts_x : float
        Absolute times (x positions)
    y_data : array
        Process variable
    y_sp_data : array
        Setpoint data
    y_min, y_max : float
        Y-axis limits
    step_time : float
        Step time
    settling_threshold : float
        Settling band threshold
    extra_margin : float
        Additional margin factor (0-1). Default: 0

    Returns
    -------
    dict : Positioning info with improved collision handling
    """
    y_span = y_max - y_min
    config = _BoxConfig(y_span, extra_margin)

    if min_x_separation is None:
        min_x_separation = config.min_x_separation

    # Find actual data bounds
    idx_step = max(0, np.searchsorted(np.arange(len(y_data)), step_time) - 1)
    if idx_step < len(y_data):
        y_final = y_sp_data[-1]
        y_initial = y_data[0]
        delta = y_final - y_initial

        is_zero_delta = abs(delta) < 1e-6
        band_magnitude = abs(y_final) if (is_zero_delta and y_final != 0) else abs(delta)
        y_band_low = y_final - settling_threshold * band_magnitude
        y_band_high = y_final + settling_threshold * band_magnitude

        data_max_after = np.max(y_data[idx_step:]) if idx_step < len(y_data) else np.max(y_data)
        data_min_after = np.min(y_data[idx_step:]) if idx_step < len(y_data) else np.min(y_data)

        data_max = max(data_max_after, y_band_high)
        data_min = min(data_min_after, y_band_low)
    else:
        data_max = np.max(y_data)
        data_min = np.min(y_data)

    # === Position tR box: below all data ===
    tr_y_bottom_min = y_min + config.margin_from_axis
    tr_y_top = data_min - config.margin_from_data
    tr_y_bottom = tr_y_top - config.box_height

    # Constrain to minimum position
    if tr_y_bottom < tr_y_bottom_min:
        tr_y_bottom = tr_y_bottom_min
        tr_y_top = tr_y_bottom + config.box_height

    tr_y_center = (tr_y_bottom + tr_y_top) / 2

    # === Position tS box: above all data ===
    ts_y_top_max = y_max - config.margin_from_axis
    ts_y_bottom_min = data_max + config.margin_from_data

    ts_y_bottom = ts_y_bottom_min
    ts_y_top = ts_y_bottom + config.box_height

    # Constrain to maximum position
    if ts_y_top > ts_y_top_max:
        ts_y_top = ts_y_top_max
        ts_y_bottom = ts_y_top - config.box_height

        if ts_y_bottom < ts_y_bottom_min:
            ts_y_bottom = ts_y_bottom_min
            ts_y_top = ts_y_bottom + config.box_height

    ts_y_center = (ts_y_bottom + ts_y_top) / 2

    # === Improved collision detection ===
    x_distance = abs(ts_x - tr_x)
    will_collide = x_distance < min_x_separation

    # Create box representations for collision check
    tr_box = {
        'x': tr_x,
        'y_center': tr_y_center,
        'height': config.box_height,
        'width': config.box_width_estimate,
    }
    ts_box = {
        'x': ts_x,
        'y_center': ts_y_center,
        'height': config.box_height,
        'width': config.box_width_estimate,
    }

    # If boxes collide horizontally, push tS higher (away from tR)
    if will_collide and _boxes_collide(tr_box, ts_box, x_threshold=None):
        # Push tS up by additional margin
        extra_push = config.margin_between_boxes
        ts_y_center = ts_y_center + extra_push + config.box_height / 2
        ts_y_bottom = ts_y_center - config.box_height / 2
        ts_y_top = ts_y_center + config.box_height / 2

        # Ensure still within bounds
        if ts_y_top > ts_y_top_max:
            ts_y_top = ts_y_top_max
            ts_y_bottom = ts_y_top - config.box_height
            ts_y_center = (ts_y_bottom + ts_y_top) / 2

    # === Vlines always traverse full plot height ===
    # Boxes are drawn ON the vline with white background — the vline "crosses" through the box.
    # This is intentional: the line is visible through the box edge, making the annotation clear.
    tr_vline_ymin = y_min
    tr_vline_ymax = y_max

    ts_vline_ymin = y_min
    ts_vline_ymax = y_max

    return {
        'tr_y': tr_y_center,
        'ts_y': ts_y_center,
        'tr_va': 'center',
        'ts_va': 'center',
        'vline_ymin': y_min,
        'vline_ymax': y_max,
        'tr_box_bottom': tr_y_bottom,
        'tr_box_top': tr_y_top,
        'ts_box_bottom': ts_y_bottom,
        'ts_box_top': ts_y_top,
        'tr_vline_ymin': tr_vline_ymin,
        'tr_vline_ymax': tr_vline_ymax,
        'ts_vline_ymin': ts_vline_ymin,
        'ts_vline_ymax': ts_vline_ymax,
        'adjusted': will_collide,
        'config': config,
    }


def plot_response(
    time,
    y,
    u,
    step_time,
    step_info=None,
    title=None,
    ylabel=None,
    xlabel='Time (s)',
    y_label_unit=None,
    figsize=(14, 7),
    show_settling_band=True,
    show_rise_time=True,
    show_rise_time_box=True,
    show_settling_time_box=True,
    y_initial=None,
    settling_threshold=0.02,
    axhline_color='#1a1a1a',
    curve_color='#0055cc',
    setpoint_color='#d65a00',
    rise_time_color='#008000',
    settling_time_color='#cc0000',
    legend_loc='best',
    grid_alpha=0.3,
    tight_layout=True,
    dpi=None,
    savefig_path=None,
    tuning_label=None,
    box_margin=0,
    tr_box_y=None,
    ts_box_y=None,
    ylim=None,
):
    """Create publication-quality step response plot with performance metrics.

    Parameters
    ----------
    time : array_like
        Time vector [seconds]
    y : array_like
        Process variable (output) response
    setpoint : array_like
        Setpoint reference signal (same length as time)
    step_time : float
        Time at which step input is applied [seconds]
    step_info : StepInfo, optional
        StepInfo object containing calculated metrics (RiseTime, SettlingTime, etc.)
        If None, only basic plot is shown without metric indicators.
    title : str, optional
        Plot title. If None, no title is shown (for publication captions).
    ylabel : str, optional
        Y-axis label
    xlabel : str, optional
        X-axis label. Default: 'Time (s)'
    y_label_unit : str, optional
        Unit for y-axis (added to ylabel if provided)
    figsize : tuple, optional
        Figure size (width, height). Default: (14, 7)
    show_settling_band : bool, optional
        Whether to show settling time band. Default: True
    show_rise_time : bool, optional
        Whether to mark rise time point. Default: True
    show_rise_time_box : bool, optional
        Whether to show rise time (tR) annotation box. Default: True
    show_settling_time_box : bool, optional
        Whether to show settling time (tS) annotation box. Default: True
    y_initial : float, optional
        Initial process variable value. Used to calculate step magnitude (delta).
        If None, uses y[0]. Required for accurate settling band matching step_info.py.
    settling_threshold : float, optional
        Settling time threshold as fraction (default 0.02 → ±2% of step magnitude).
        Must match StepInfo(SettlingTimeThreshold=...) for consistency.
    axhline_color : str, optional
        Color for settling band lines. Default: '#1a1a1a' (dark gray for high contrast)
    curve_color : str, optional
        Color for process variable curve. Default: '#0055cc' (dark blue for high contrast)
    setpoint_color : str, optional
        Color for setpoint reference. Default: '#d65a00' (dark orange for high contrast)
    rise_time_color : str, optional
        Color for rise time marker. Default: '#008000' (dark green for high contrast)
    settling_time_color : str, optional
        Color for settling time marker. Default: '#cc0000' (dark red for high contrast)
    legend_loc : str, optional
        Legend location. Default: 'best' (automatic, avoids plot overlap)
    grid_alpha : float, optional
        Alpha transparency for grid. Default: 0.3
    tight_layout : bool, optional
        Apply tight_layout before showing. Default: True
    dpi : int, optional
        DPI for saved figure (typically 300-600 for publication). Default: None
    savefig_path : str, optional
        Path to save figure (e.g., 'plot.png'). If None, figure is not saved.
    tuning_label : str, optional
        Label for tuning configuration (e.g., 'Tuning 1', 'Aggressive', 'Conservative').
        If provided, will be displayed in tR and tS annotations.
        Example: 'tR = 3246 s (Tuning 1)' instead of 'tR = 3246 s'
    box_margin : float, optional
        Additional margin factor for annotation boxes (0-1). Default: 0 (no extra margin).
        Increase to 0.5 or higher if boxes touch plot lines. Recommended: 0.3-0.5
    tr_box_y : float, optional
        Manual y-position for rise time (tR) annotation box. Default: None (auto-calculated).
    ts_box_y : float, optional
        Manual y-position for settling time (tS) annotation box. Default: None (auto-calculated).

    Returns
    -------
    fig, ax : matplotlib Figure and Axes objects
        For further customization if needed.

    Notes
    -----
    **Settling Band Calculation (matches step_info.py)**:

    The settling band represents the ±2% error tolerance zone:

        delta = yfinal - y_initial  (step magnitude)
        y_band_low = yfinal - settling_threshold * abs(delta)
        y_band_high = yfinal + settling_threshold * abs(delta)

    This matches StepInfo's settling time definition:
    |error| ≤ settling_threshold * abs(delta)

    **Legend Placement**:

    Uses 'best' location by default, which automatically positions the legend
    to avoid overlapping plot data. For manual control, pass:
    - 'upper left', 'upper right', 'lower left', 'lower right'
    - 'upper center', 'lower center', 'center left', 'center right'
    - 'center', 'best', 'outside'

    Examples
    --------
    Basic plot with StepInfo metrics:

        from plot_utils import plot_step_response
        from step_info import StepInfo

        si = StepInfo(time=time, y=T, setpoint=TSP_100_SP,
                      yfinal=340.15, y_initial=333.15, step_time=600)

        fig, ax = plot_step_response(
            time=time, y=T, setpoint=TSP_100_SP, step_time=600,
            step_info=si, y_initial=333.15,
            title='TIC-100 Temperature Control',
            ylabel='Temperature', y_label_unit='K'
        )
        plt.show()

    Plot without metrics (just raw curves):

        fig, ax = plot_step_response(
            time=time, y=T, setpoint=TSP_100_SP, step_time=600,
            step_info=None, title='Temperature Response'
        )
        plt.show()
    """
    # Ensure numpy arrays
    time = np.asarray(time)
    y = np.asarray(y)
    u = np.asarray(u)

    # Handle scalar setpoint
    if u.ndim == 0:
        y_setpoint_final = float(u)
        u = np.full_like(time, float(u))
    else:
        # Extract final setpoint value (after step)
        if step_time is not None:
            idx_step = np.searchsorted(time, step_time)
            if idx_step < len(u):
                y_setpoint_final = float(u[idx_step:].mean())
            else:
                y_setpoint_final = float(u[-1])
        else:
            y_setpoint_final = float(u[-1])

    # Use metrics from step_info if available to ensure exact match
    if step_info is not None and hasattr(step_info, 'yfinal') and hasattr(step_info, 'y_initial'):
        y_setpoint_final = step_info.yfinal
        y_initial = step_info.y_initial
    elif y_initial is None:
        y_initial = float(y[0])
    else:
        y_initial = float(y_initial)

    # Calculate step magnitude (delta) for settling band
    # Matches StepInfo definition: delta = yfinal - y_initial
    delta = y_setpoint_final - y_initial
    is_zero_delta = abs(delta) < 1e-6
    step_magnitude = (
        abs(y_setpoint_final) if (is_zero_delta and y_setpoint_final != 0) else abs(delta)
    )

    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)

    # Plot process variable
    ax.plot(time, y, label='Process Variable (PV)', color=curve_color, linewidth=2.0)

    # Plot setpoint
    ax.plot(time, u, label='Reference (R)', linestyle='--', color=setpoint_color, linewidth=1.5)

    # Add settling time band if requested and step_info available
    # Settling band formula matches step_info.py:
    # y_band = yfinal ± settling_threshold * abs(delta)
    if show_settling_band and step_info is not None:
        y_band_low = y_setpoint_final - settling_threshold * step_magnitude
        y_band_high = y_setpoint_final + settling_threshold * step_magnitude

        ax.axhline(y=y_band_low, color=axhline_color, linestyle=':', linewidth=1.5, alpha=0.7)
        ax.axhline(y=y_band_high, color=axhline_color, linestyle=':', linewidth=1.5, alpha=0.7)

        # Shade the settling band
        ax.fill_between(
            time,
            y_band_low,
            y_band_high,
            alpha=0.1,
            color=axhline_color,
            label=f'Settling Band (±{settling_threshold * 100:.0f}%)',
        )

    # Collect rise time and settling time data BEFORE plotting (for smart y-axis extension)
    rise_times_data = []
    settling_times_data = []
    rise_time_abs = np.nan
    settling_time_abs = np.nan
    safe_step_time = step_time if step_time is not None else 0.0

    if show_rise_time and step_info is not None and show_rise_time_box:
        rise_time = step_info.metrics.RiseTime
        if not np.isnan(rise_time):
            if hasattr(step_info, 'y_initial') and hasattr(step_info, 'yfinal'):
                curve_y_initial = step_info.y_initial
                curve_delta = step_info.yfinal - step_info.y_initial
            else:
                curve_y_initial = y_initial
                curve_delta = delta

            idx_step = np.searchsorted(time, safe_step_time)

            if abs(curve_delta) < 1e-6:
                rise_time_abs = np.nan
            else:
                threshold_90 = curve_y_initial + 0.9 * curve_delta
                y_slice = y[idx_step:]

                if curve_delta > 0:
                    crossing = y_slice >= threshold_90
                else:
                    crossing = y_slice <= threshold_90

                crossing_indices = np.nonzero(crossing)[0]

                if crossing_indices.size > 0:
                    idx_90 = idx_step + crossing_indices[0]
                    rise_time_abs = float(time[idx_90])
                else:
                    rise_time_abs = float(safe_step_time + rise_time)

            if not np.isnan(rise_time_abs):
                rise_times_data.append(
                    {'time': rise_time_abs, 'value': rise_time, 'color': rise_time_color}
                )

    if step_info is not None and show_settling_time_box:
        settling_time = step_info.metrics.SettlingTime
        if not np.isnan(settling_time):
            settling_time_abs = safe_step_time + settling_time
            settling_times_data.append(
                {'time': settling_time_abs, 'value': settling_time, 'color': settling_time_color}
            )

    # === EXTEND Y-AXIS based on stagger levels needed (before plotting boxes) ===
    if ylim is not None:
        ax.set_ylim(ylim)

    y_min, y_max = ax.get_ylim()
    y_span = y_max - y_min

    x_lo, x_hi = ax.get_xlim()
    slot_frac = 0.065 + 0.02  # box_height + margin
    edge_frac = 0.06  # margin_from_axis
    clearance_frac = 0.15  # clearance between box and curves

    # Pre-extend y-axis based on actual stagger levels needed
    n_tr = (
        _needed_levels(rise_times_data, x_lo, x_hi)
        if (rise_times_data and show_rise_time_box)
        else 0
    )
    n_ts = (
        _needed_levels(settling_times_data, x_lo, x_hi)
        if (settling_times_data and show_settling_time_box)
        else 0
    )

    space_bot = (
        (edge_frac + n_tr * slot_frac + 0.065 / 2 + clearance_frac) * y_span if n_tr > 0 else 0
    )
    space_top = (
        (edge_frac + n_ts * slot_frac + 0.065 / 2 + clearance_frac) * y_span if n_ts > 0 else 0
    )

    ax.set_ylim(y_min - space_bot, y_max + space_top)

    # Get final extended y-axis limits
    y_min_extended, y_max_extended = ax.get_ylim()

    # === PLOT RISE TIME BOXES ===
    if rise_times_data and show_rise_time_box:
        positioned = _stack_boxes_smart(
            rise_times_data,
            y_min_extended,
            y_max_extended,
            y_max_extended - y_min_extended,
            position_type='bottom',
            x_min=x_lo,
            x_max=x_hi,
        )

        for i, box in enumerate(positioned):
            if tuning_label:
                tr_text = f'$t_R$ = {box["value"]:.0f} s\n({tuning_label})'
            else:
                tr_text = f'$t_R$ = {box["value"]:.0f} s'

            # Vline extending fully from extended y_min to y_max
            ax.vlines(
                x=box['time'],
                ymin=y_min_extended,
                ymax=y_max_extended,
                colors=box['color'],
                linestyles='--',
                linewidth=1.5,
                alpha=0.7,
                label='Rise Time',
                zorder=2,
            )

            # Box with text
            ax.text(
                x=box['time'],
                y=_resolve_box_y(tr_box_y, i, box['y_center']),
                s=tr_text,
                color=box['color'],
                fontsize=11,
                fontweight='bold',
                ha='center',
                va='center',
                zorder=3,
                bbox={
                    'boxstyle': 'round,pad=0.3',
                    'facecolor': 'white',
                    'edgecolor': box['color'],
                    'alpha': 1.0,
                    'linewidth': 1.5,
                },
            )

    # === PLOT SETTLING TIME BOXES ===
    if settling_times_data and show_settling_time_box:
        positioned = _stack_boxes_smart(
            settling_times_data,
            y_min_extended,
            y_max_extended,
            y_max_extended - y_min_extended,
            position_type='top',
            x_min=x_lo,
            x_max=x_hi,
        )

        for i, box in enumerate(positioned):
            if tuning_label:
                ts_text = f'$t_S$ = {box["value"]:.0f} s\n({tuning_label})'
            else:
                ts_text = f'$t_S$ = {box["value"]:.0f} s'

            # Vline extending fully from extended y_min to y_max
            ax.vlines(
                x=box['time'],
                ymin=y_min_extended,
                ymax=y_max_extended,
                colors=box['color'],
                linestyles='--',
                linewidth=1.5,
                alpha=0.7,
                label='Settling Time',
                zorder=2,
            )

            # Box with text
            ax.text(
                x=box['time'],
                y=_resolve_box_y(ts_box_y, i, box['y_center']),
                s=ts_text,
                color=box['color'],
                fontsize=11,
                fontweight='bold',
                ha='center',
                va='center',
                zorder=3,
                bbox={
                    'boxstyle': 'round,pad=0.3',
                    'facecolor': 'white',
                    'edgecolor': box['color'],
                    'alpha': 1.0,
                    'linewidth': 1.5,
                },
            )

    # Labels and formatting
    if ylabel:
        ax.set_ylabel(
            f'{ylabel} ({y_label_unit})' if y_label_unit else ylabel, fontsize=12, fontweight='bold'
        )

    ax.set_xlabel(xlabel, fontsize=12, fontweight='bold', labelpad=8)

    # if title is not None:  # no title per publication style
    #     ax.set_title(title, fontsize=13, fontweight='bold')

    # Legend with smart placement to avoid overlap
    # 'best' uses an algorithm to minimize overlap with data
    ax.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, -0.09),
        ncol=6,
        fontsize=10,
        frameon=True,
        framealpha=0.95,
        edgecolor='gray',
        fancybox=True,
    )
    ax.grid(True, alpha=grid_alpha)

    # Publication-quality formatting with all spines visible
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)
        spine.set_color('#1a1a1a')
    ax.tick_params(axis='both', which='major', labelsize=10)

    # Freeze y-limits to prevent autoscale margins from being added around vlines
    ax.set_ylim(y_min_extended, y_max_extended)

    if tight_layout:
        fig.tight_layout()

    # Save figure if path provided
    if savefig_path is not None:
        if dpi is None:
            dpi = 600  # publication default
        _save_fig(fig, savefig_path, dpi)
        print(f'[OK] Figure saved to: {savefig_path} (DPI={dpi})')

    return fig, ax


def _stack_boxes_smart(
    boxes_list, y_min, y_max, y_span, position_type='bottom', x_min=None, x_max=None
):
    """Smart stacking algorithm for multiple boxes (rise time or settling time).

    Boxes that are close in x-position are staggered vertically so they never overlap.
    Proximity is judged against the FULL plot x-range (x_min/x_max), ensuring that
    tightly-clustered tR boxes (e.g. all within 2000 s of a 40 000-s plot) are always
    correctly separated regardless of their mutual spread.

    Vlines cross the full plot height; boxes sit on top with an opaque white background.

    Parameters
    ----------
    boxes_list : list of dict
        Each dict: {'time': float, 'label': str, 'value': float, 'color': str}
    y_min, y_max : float
        Current y-axis limits (already extended for annotation space).
    y_span : float
        y_max - y_min
    position_type : str
        'bottom' — boxes grow upward from y_min (rise time)
        'top'    — boxes grow downward from y_max (settling time)
    x_min, x_max : float, optional
        Full plot x-axis limits for computing proximity threshold.

    Returns
    -------
    list of dict with 'y_center', 'vline_ymin', 'vline_ymax', 'va', '_level'
    """
    if not boxes_list:
        return []

    box_height = y_span * 0.065
    margin = y_span * 0.02  # gap between stagger levels
    margin_from_axis = y_span * 0.06  # gap from y_min / y_max edge

    # --- x-proximity threshold ---
    # Use 5% of the full plot x-range. When text boxes are close horizontally
    # they can overlap due to character width. Using 5% of the x-range ensures
    # they stagger correctly on different levels.
    if x_min is not None and x_max is not None and (x_max - x_min) > 0:
        x_close_threshold = (x_max - x_min) * 0.05
    else:
        all_times = [b['time'] for b in boxes_list]
        x_spread = max(all_times) - min(all_times) if len(all_times) > 1 else 1.0
        x_close_threshold = max(x_spread * 0.5, 1.0)

    # Sort left-to-right for deterministic level assignment
    sorted_boxes = sorted(boxes_list, key=lambda b: b['time'])

    # Greedy graph-colouring: assign each box to the lowest available level
    # that doesn't collide with any already-positioned boxes on that level.
    stagger_levels = [0] * len(sorted_boxes)
    for i in range(len(sorted_boxes)):
        occupied_levels = set()
        for j in range(i):
            if abs(sorted_boxes[i]['time'] - sorted_boxes[j]['time']) < x_close_threshold:
                occupied_levels.add(stagger_levels[j])

        # Find the lowest available level (0, 1, 2, ...)
        level = 0
        while level in occupied_levels:
            level += 1
        stagger_levels[i] = level

    positioned_boxes = []
    for idx, box in enumerate(sorted_boxes):
        box_copy = box.copy()
        level = stagger_levels[idx]

        if position_type == 'bottom':
            y_center = y_min + margin_from_axis + level * (box_height + margin) + box_height / 2
        else:
            y_center = y_max - margin_from_axis - level * (box_height + margin) - box_height / 2

        box_copy.update(
            {
                'y_center': y_center,
                'vline_ymin': y_min,
                'vline_ymax': y_max,
                'va': 'center',
                '_level': level,
            }
        )
        positioned_boxes.append(box_copy)

    return positioned_boxes


def _needed_levels(boxes_list, x_min, x_max):
    """Return the number of stagger levels needed (>=1) for a list of boxes."""
    if not boxes_list:
        return 0
    dummy = _stack_boxes_smart(
        boxes_list, 0, 1, 1, position_type='bottom', x_min=x_min, x_max=x_max
    )
    return max(b['_level'] for b in dummy) + 1


def plot_step_response_multiple(
    time,
    y_data_dict,
    u,
    step_time,
    step_info_dict=None,
    title=None,
    ylabel=None,
    xlabel='Time (s)',
    y_label_unit=None,
    figsize=(14, 7),
    show_settling_band=True,
    show_rise_time=True,
    show_rise_time_box=True,
    show_settling_time_box=True,
    y_initial=None,
    settling_threshold=0.01,
    axhline_color='#1a1a1a',
    curve_colors=None,
    setpoint_color='#d65a00',
    rise_time_color='#008000',
    settling_time_color='#cc0000',
    legend_loc='best',
    grid_alpha=0.3,
    tight_layout=True,
    dpi=None,
    savefig_path=None,
    box_margin=0,
    tr_box_y=None,
    ts_box_y=None,
    ylim=None,
):
    """Plot multiple tuning curves on a shared step response axis.

    Parameters
    ----------
    time : array_like
        Time vector [s].
    y_data_dict : dict
        Mapping of label → process variable array.
    u : array_like or scalar
        Setpoint reference signal (scalar is broadcast to full time vector).
    step_time : float
        Time at which the step input is applied [s].
    step_info_dict : dict, optional
        Mapping of label → ``StepInfo`` for annotation boxes.
    title : str, optional
        Plot title.
    ylabel : str, optional
        Y-axis label.
    xlabel : str, optional
        X-axis label.  Default: ``'Time (s)'``.
    y_label_unit : str, optional
        Unit string appended to ylabel in parentheses.
    figsize : tuple, optional
        Figure size.  Default: ``(14, 7)``.
    show_settling_band : bool, optional
        Draw the ±settling_threshold band.  Default: True.
    show_rise_time : bool, optional
        Collect rise time data for box annotation.  Default: True.
    show_rise_time_box : bool, optional
        Draw rise time annotation boxes.  Default: True.
    show_settling_time_box : bool, optional
        Draw settling time annotation boxes.  Default: True.
    y_initial : float, optional
        Initial PV value for step magnitude calculation.
    settling_threshold : float, optional
        Settling band half-width as fraction of step magnitude.
        Default: 0.01.
    axhline_color : str, optional
        Color for settling band lines.  Default: ``'#1a1a1a'``.
    curve_colors : dict or list, optional
        Colors per label.  Auto-assigned when ``None``.
    setpoint_color : str, optional
        Color for the setpoint line.  Default: ``'#d65a00'``.
    rise_time_color : str, optional
        Fallback rise time marker color.  Default: ``'#008000'``.
    settling_time_color : str, optional
        Fallback settling time marker color.  Default: ``'#cc0000'``.
    legend_loc : str, optional
        Legend location keyword.  Default: ``'best'``.
    grid_alpha : float, optional
        Grid transparency.  Default: 0.3.
    tight_layout : bool, optional
        Call ``fig.tight_layout()`` before returning.  Default: True.
    dpi : int, optional
        DPI for saved figure (defaults to 600 when saving).
    savefig_path : str, optional
        File path to save the figure.  If ``None``, figure is not saved.
    box_margin : float, optional
        Extra margin factor for annotation boxes.  Default: 0.
    tr_box_y : float or list, optional
        Manual y-position(s) for rise time boxes.
    ts_box_y : float or list, optional
        Manual y-position(s) for settling time boxes.
    ylim : tuple, optional
        Y-axis limits ``(ymin, ymax)``.

    Returns
    -------
    fig, ax : matplotlib Figure and Axes
    """
    u = np.asarray(u)

    if u.ndim == 0:
        y_setpoint_final = float(u)
        u = np.full_like(time, float(u))
    else:
        if step_time is not None:
            idx_step = np.searchsorted(time, step_time)
            if idx_step < len(u):
                y_setpoint_final = float(u[idx_step:].mean())
            else:
                y_setpoint_final = float(u[-1])
        else:
            y_setpoint_final = float(u[-1])

    # Extract setpoint value (matches step_info.py)
    if step_info_dict and len(step_info_dict) > 0:
        first_info = list(step_info_dict.values())[0]
        if first_info is not None and hasattr(first_info, 'yfinal'):
            y_setpoint_final = first_info.yfinal
            y_initial_global = (
                first_info.y_initial
                if hasattr(first_info, 'y_initial')
                else float(list(y_data_dict.values())[0][0])
            )
        else:
            y_initial_global = (
                y_initial if y_initial is not None else float(list(y_data_dict.values())[0][0])
            )
    else:
        y_initial_global = (
            y_initial if y_initial is not None else float(list(y_data_dict.values())[0][0])
        )

    delta_global = y_setpoint_final - y_initial_global

    # Handle zero-delta case (matches step_info.py logic)
    is_zero_delta = abs(delta_global) < 1e-6
    step_magnitude = (
        abs(y_setpoint_final) if (is_zero_delta and y_setpoint_final != 0) else abs(delta_global)
    )

    default_colors = ['#0055cc', '#d65a00', '#008000', '#cc0000', '#9400d3', '#ff8c00']
    if curve_colors is None:
        colors_dict = {
            label: default_colors[i % len(default_colors)]
            for i, label in enumerate(y_data_dict.keys())
        }
    elif isinstance(curve_colors, dict):
        colors_dict = curve_colors
    else:
        colors_dict = {
            label: curve_colors[i % len(curve_colors)] for i, label in enumerate(y_data_dict.keys())
        }

    fig, ax = plt.subplots(figsize=figsize)

    all_y_data = []
    for label, y in y_data_dict.items():
        y = np.asarray(y)
        all_y_data.append(y)
        color = colors_dict[label]
        ax.plot(time, y, label=f'{label}', color=color, linewidth=2.0)

    ax.plot(time, u, label='Reference', linestyle='--', color=setpoint_color, linewidth=1.5)

    if show_settling_band:
        y_band_low = y_setpoint_final - settling_threshold * step_magnitude
        y_band_high = y_setpoint_final + settling_threshold * step_magnitude

        ax.axhline(y=y_band_low, color=axhline_color, linestyle=':', linewidth=1.5, alpha=0.7)
        ax.axhline(y=y_band_high, color=axhline_color, linestyle=':', linewidth=1.5, alpha=0.7)

        ax.fill_between(
            time,
            y_band_low,
            y_band_high,
            alpha=0.1,
            color=axhline_color,
            label=f'Settling Band (±{settling_threshold * 100:.0f}%)',
        )

    y_min, y_max = ax.get_ylim()
    y_span = y_max - y_min

    rise_times_data = []
    settling_times_data = []

    for label, y in y_data_dict.items():
        y = np.asarray(y)
        step_info = step_info_dict.get(label) if step_info_dict is not None else None

        if step_info is None or not hasattr(step_info, 'metrics'):
            continue

        metrics = step_info.metrics

        if show_rise_time and not np.isnan(metrics.RiseTime):
            rise_time = metrics.RiseTime

            # FIX 1: Use specific properties from step_info of this curve for accurate markers
            curve_y_initial = (
                step_info.y_initial if hasattr(step_info, 'y_initial') else y_initial_global
            )
            curve_y_final = step_info.yfinal if hasattr(step_info, 'yfinal') else y_setpoint_final
            curve_delta = curve_y_final - curve_y_initial

            # Only compute rise_time_abs if there's meaningful delta (not disturbance rejection)
            if abs(curve_delta) > 1e-6:
                safe_step_time = step_time if step_time is not None else 0.0
                idx_step = np.searchsorted(time, safe_step_time)
                threshold_90 = curve_y_initial + 0.9 * curve_delta
                y_slice = y[idx_step:]

                if curve_delta > 0:
                    crossing = y_slice >= threshold_90
                else:
                    crossing = y_slice <= threshold_90

                crossing_indices = np.nonzero(crossing)[0]
                if crossing_indices.size > 0:
                    idx_90 = idx_step + crossing_indices[0]
                    rise_time_abs = time[idx_90]
                else:
                    # Fallback if signal doesn't reach 90%
                    rise_time_abs = safe_step_time + rise_time

                rise_times_data.append(
                    {
                        'label': label,
                        'time': rise_time_abs,
                        'value': rise_time,
                        'color': colors_dict[label],
                    }
                )

        # Gather settling time from this curve's metrics (if present)
        settling_time = getattr(metrics, 'SettlingTime', np.nan)
        if not np.isnan(settling_time):
            safe_step_time = step_time if step_time is not None else 0.0
            settling_time_abs = float(safe_step_time + settling_time)  # Cast to float

            settling_times_data.append(
                {
                    'label': label,
                    'time': settling_time_abs,
                    'value': settling_time,
                    'color': colors_dict[label],
                }
            )

    # === Pre-compute stagger levels to correctly extend y-axis ===
    x_lo, x_hi = ax.get_xlim()
    slot_frac = 0.065 + 0.02  # box_height + margin (as fraction of y_span)
    edge_frac = 0.06  # margin_from_axis
    clearance_frac = 0.15  # clearance between box and curves

    y_min_now, y_max_now = ax.get_ylim()
    y_span_now = y_max_now - y_min_now

    n_tr = (
        _needed_levels(rise_times_data, x_lo, x_hi)
        if (rise_times_data and show_rise_time_box)
        else 0
    )
    n_ts = (
        _needed_levels(settling_times_data, x_lo, x_hi)
        if (settling_times_data and show_settling_time_box)
        else 0
    )

    space_bot = (
        (edge_frac + n_tr * slot_frac + 0.065 / 2 + clearance_frac) * y_span_now if n_tr > 0 else 0
    )
    space_top = (
        (edge_frac + n_ts * slot_frac + 0.065 / 2 + clearance_frac) * y_span_now if n_ts > 0 else 0
    )

    ax.set_ylim(y_min_now - space_bot, y_max_now + space_top)

    # Plot rise time boxes
    if rise_times_data and show_rise_time_box:
        y_min, y_max = ax.get_ylim()
        if ylim is not None:
            ax.set_ylim(ylim)
            y_min, y_max = ylim
        y_span = y_max - y_min
        positioned_boxes = _stack_boxes_smart(
            rise_times_data, y_min, y_max, y_span, position_type='bottom', x_min=x_lo, x_max=x_hi
        )
        for i, box in enumerate(positioned_boxes):
            tr_text = f'$t_R$ = {box["value"]:.0f} s\n({box["label"]})'
            ax.vlines(
                x=box['time'],
                ymin=y_min,
                ymax=y_max,
                colors=box['color'],
                linestyles='--',
                linewidth=1.0,
                alpha=0.5,
                zorder=2,
            )
            ax.text(
                x=box['time'],
                y=_resolve_box_y(tr_box_y, i, box['y_center']),
                s=tr_text,
                color=box['color'],
                fontsize=9,
                fontweight='bold',
                ha='center',
                va='center',
                zorder=3,
                bbox={
                    'boxstyle': 'round,pad=0.4',
                    'facecolor': 'white',
                    'edgecolor': box['color'],
                    'alpha': 1.0,
                    'linewidth': 1.5,
                },
            )

    # Plot settling time boxes
    if settling_times_data and show_settling_time_box:
        y_min, y_max = ax.get_ylim()
        if ylim is not None:
            ax.set_ylim(ylim)
            y_min, y_max = ylim
        y_span = y_max - y_min
        positioned_boxes = _stack_boxes_smart(
            settling_times_data, y_min, y_max, y_span, position_type='top', x_min=x_lo, x_max=x_hi
        )
        for i, box in enumerate(positioned_boxes):
            ts_text = f'$t_S$ = {box["value"]:.0f} s\n({box["label"]})'
            ax.vlines(
                x=box['time'],
                ymin=y_min,
                ymax=y_max,
                colors=box['color'],
                linestyles=':',
                linewidth=1.0,
                alpha=0.5,
                zorder=2,
            )
            ax.text(
                x=box['time'],
                y=_resolve_box_y(ts_box_y, i, box['y_center']),
                s=ts_text,
                color=box['color'],
                fontsize=9,
                fontweight='bold',
                ha='center',
                va=box['va'],
                zorder=3,
                bbox={
                    'boxstyle': 'round,pad=0.4',
                    'facecolor': 'white',
                    'edgecolor': box['color'],
                    'alpha': 1.0,
                    'linewidth': 1.5,
                },
            )

    # Formatting and axis adjustment...
    if ylabel:
        ax.set_ylabel(f'{ylabel} ({y_label_unit})' if y_label_unit else ylabel, fontsize=12)

    ax.set_xlabel(xlabel, fontsize=12, labelpad=8)
    # if title is not None:  # no title per publication style
    #     ax.set_title(title, fontsize=13, fontweight='bold')

    ax.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, -0.09),
        ncol=10,
        borderaxespad=0,
        frameon=True,
        fontsize=9,
        framealpha=0.95,
        edgecolor='gray',
        fancybox=True,
    )

    ax.grid(True, alpha=grid_alpha)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)
        spine.set_color('#1a1a1a')
    ax.tick_params(axis='both', which='major', labelsize=10)

    # Freeze y-limits to prevent autoscale margins from being added around vlines
    ax.set_ylim(y_min, y_max)

    if tight_layout:
        fig.tight_layout()

    if savefig_path is not None:
        if dpi is None:
            dpi = 600
        _save_fig(fig, savefig_path, dpi)
        print(f'[OK] Figure saved to: {savefig_path} (DPI={dpi})')

    return fig, ax


def plot_tuning_comparison(
    tuning_scenarios,
    time,
    step_time,
    title=None,
    ylabel=None,
    y_label_unit=None,
    figsize=(14, 7),
    settling_threshold=0.02,
    show_rise_time_box=True,
    show_settling_time_box=True,
    legend_loc='best',
    grid_alpha=0.3,
    tight_layout=True,
    dpi=None,
    savefig_path=None,
    tr_box_y=None,
    ts_box_y=None,
    ylim=None,
):
    """Plot and compare multiple tuning scenarios on a single step response axis.

    Parameters
    ----------
    tuning_scenarios : dict
        A dictionary where keys are tuning names and values are dicts containing:
        - 'y' : array_like, process variable response.
        - 'setpoint' : array_like, setpoint signal.
        - 'step_info' : StepInfo, performance metrics (optional).
        - 'y_initial' : float, initial value (optional).
    time : array_like
        Time vector [seconds].
    step_time : float
        Time at which step input is applied [seconds].
    title : str, optional
        Plot title.
    ylabel : str, optional
        Y-axis label.
    y_label_unit : str, optional
        Unit for y-axis (added to ylabel if provided).
    figsize : tuple, optional
        Figure size (width, height). Default: (14, 7).
    settling_threshold : float, optional
        Settling band threshold. Default: 0.02.
    show_rise_time_box : bool, optional
        Draw rise time annotation boxes. Default: True.
    show_settling_time_box : bool, optional
        Draw settling time annotation boxes. Default: True.
    legend_loc : str, optional
        Legend location. Default: 'best'.
    grid_alpha : float, optional
        Grid line transparency. Default: 0.3.
    tight_layout : bool, optional
        Apply tight_layout before saving. Default: True.
    dpi : int, optional
        DPI for saved figure. Default: None.
    savefig_path : str, optional
        Path to save the generated figure.
    tr_box_y : float or list, optional
        Custom y-coordinate(s) for rise time boxes.
    ts_box_y : float or list, optional
        Custom y-coordinate(s) for settling time boxes.
    ylim : tuple, optional
        Y-axis limits (ymin, ymax).

    Returns
    -------
    fig : matplotlib.figure.Figure
        The matplotlib figure object.
    ax : matplotlib.axes.Axes
        The matplotlib axes object.
    """
    colors_pv = ['#0055cc', '#d65a00', '#008000']
    colors_sp = ['#1a4d99', '#a64200', '#004d00']
    colors_tr = ['#003d7a', '#994200', '#003300']
    colors_ts = ['#8b0000', '#cc6600', '#660000']

    fig, ax = plt.subplots(figsize=figsize)
    time = np.asarray(time)

    rise_times_data = []
    settling_times_data = []

    for idx, (tuning_name, scenario_data) in enumerate(tuning_scenarios.items()):
        y = np.asarray(scenario_data['y'])
        setpoint = np.asarray(scenario_data['setpoint'])
        step_info = scenario_data.get('step_info')
        y_initial = scenario_data.get('y_initial', float(y[0]))

        idx_step = np.searchsorted(time, step_time)
        y_sp = (
            float(setpoint[idx_step:].mean()) if idx_step < len(setpoint) else float(setpoint[-1])
        )

        color_pv = colors_pv[idx % len(colors_pv)]
        color_sp = colors_sp[idx % len(colors_sp)]
        color_tr = colors_tr[idx % len(colors_tr)]
        color_ts = colors_ts[idx % len(colors_ts)]

        ax.plot(time, y, label=f'{tuning_name} (PV)', color=color_pv, linewidth=2.0)
        ax.plot(time, setpoint, linestyle='--', color=color_sp, linewidth=1.5, alpha=0.7)

        # FIX 2: Collect box data for staggered alignment instead of plotting directly to prevent overlap
        if step_info is not None and hasattr(step_info, 'metrics'):
            metrics = step_info.metrics

            if not np.isnan(metrics.RiseTime):
                # Find the appropriate marker with accurate logic
                curve_delta = (
                    step_info.yfinal - step_info.y_initial
                    if hasattr(step_info, 'y_initial')
                    else y_sp - y_initial
                )

                # Only compute rise time marker if meaningful delta (not disturbance rejection)
                if abs(curve_delta) > 1e-6:
                    threshold_90 = (
                        step_info.y_initial if hasattr(step_info, 'y_initial') else y_initial
                    ) + 0.9 * curve_delta
                    y_slice = y[idx_step:]
                    crossing = (
                        y_slice >= threshold_90 if curve_delta > 0 else y_slice <= threshold_90
                    )

                    crossing_indices = np.nonzero(crossing)[0]
                    rt_abs = (
                        time[idx_step + crossing_indices[0]]
                        if crossing_indices.size > 0
                        else step_time + metrics.RiseTime
                    )

                    rise_times_data.append(
                        {
                            'label': tuning_name,
                            'time': rt_abs,
                            'value': metrics.RiseTime,
                            'color': color_tr,
                        }
                    )

            if not np.isnan(metrics.SettlingTime):
                settling_times_data.append(
                    {
                        'label': tuning_name,
                        'time': step_time + metrics.SettlingTime,
                        'value': metrics.SettlingTime,
                        'color': color_ts,
                    }
                )

    x_lo, x_hi = ax.get_xlim()
    slot_frac = 0.065 + 0.02
    edge_frac = 0.06
    clearance_frac = 0.15

    # Pre-extend y-axis based on actual stagger levels needed
    y_min, y_max = ax.get_ylim()
    y_span = y_max - y_min

    n_tr = (
        _needed_levels(rise_times_data, x_lo, x_hi)
        if (rise_times_data and show_rise_time_box)
        else 0
    )
    n_ts = (
        _needed_levels(settling_times_data, x_lo, x_hi)
        if (settling_times_data and show_settling_time_box)
        else 0
    )

    space_bot = (
        (edge_frac + n_tr * slot_frac + 0.065 / 2 + clearance_frac) * y_span if n_tr > 0 else 0
    )
    space_top = (
        (edge_frac + n_ts * slot_frac + 0.065 / 2 + clearance_frac) * y_span if n_ts > 0 else 0
    )

    ax.set_ylim(y_min - space_bot, y_max + space_top)

    y_min, y_max = ax.get_ylim()
    if ylim is not None:
        ax.set_ylim(ylim)
        y_min, y_max = ylim
    y_span = y_max - y_min

    # Plot rise time boxes
    if rise_times_data and show_rise_time_box:
        positioned_boxes = _stack_boxes_smart(
            rise_times_data, y_min, y_max, y_span, position_type='bottom', x_min=x_lo, x_max=x_hi
        )
        for i, box in enumerate(positioned_boxes):
            ax.vlines(
                x=box['time'],
                ymin=y_min,
                ymax=y_max,
                colors=box['color'],
                linestyles='--',
                linewidth=1.5,
                alpha=0.6,
                zorder=2,
            )
            ax.text(
                x=box['time'],
                y=_resolve_box_y(tr_box_y, i, box['y_center']),
                s=f'$t_R$={box["value"]:.0f}s\n({box["label"]})',
                color=box['color'],
                fontsize=9,
                fontweight='bold',
                ha='center',
                va='center',
                zorder=3,
                bbox={
                    'boxstyle': 'round,pad=0.25',
                    'facecolor': 'white',
                    'edgecolor': box['color'],
                    'alpha': 1.0,
                },
            )

    # Plot settling time boxes
    if settling_times_data and show_settling_time_box:
        positioned_boxes = _stack_boxes_smart(
            settling_times_data, y_min, y_max, y_span, position_type='top', x_min=x_lo, x_max=x_hi
        )
        for i, box in enumerate(positioned_boxes):
            ax.vlines(
                x=box['time'],
                ymin=y_min,
                ymax=y_max,
                colors=box['color'],
                linestyles='--',
                linewidth=1.5,
                alpha=0.6,
                zorder=2,
            )
            ax.text(
                x=box['time'],
                y=_resolve_box_y(ts_box_y, i, box['y_center']),
                s=f'$t_S$={box["value"]:.0f}s\n({box["label"]})',
                color=box['color'],
                fontsize=9,
                fontweight='bold',
                ha='center',
                va=box['va'],
                zorder=3,
                bbox={
                    'boxstyle': 'round,pad=0.25',
                    'facecolor': 'white',
                    'edgecolor': box['color'],
                    'alpha': 1.0,
                },
            )

    if ylabel is not None:
        ax.set_ylabel(f'{ylabel} ({y_label_unit})' if y_label_unit else ylabel, fontsize=12)
    ax.set_xlabel('Time (s)', fontsize=12, labelpad=8)
    # if title is not None:  # no title per publication style
    #     ax.set_title(title, fontsize=13, fontweight='bold')

    ax.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, -0.09),
        ncol=6,
        fontsize=10,
        frameon=True,
        framealpha=0.95,
        edgecolor='gray',
        fancybox=True,
    )
    ax.grid(True, alpha=grid_alpha)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # Freeze y-limits to prevent autoscale margins from being added
    ax.set_ylim(y_min, y_max)

    if tight_layout:
        fig.tight_layout()
    if savefig_path is not None:
        _save_fig(fig, savefig_path, dpi or 600)

    return fig, ax


def compare_step_responses(
    loops_dict,
    time,
    step_time,
    figsize=(14, 7),
    cols=2,
    title_prefix='',
    settling_threshold=0.02,
    show_rise_time_box=True,
    show_settling_time_box=True,
    dpi=None,
    savefig_path=None,
    tr_box_y=None,
    ts_box_y=None,
    ylim=None,
):
    """Create multi-panel comparison of multiple control loops.

    Parameters
    ----------
    loops_dict : dict
        Dictionary mapping loop names to dicts with keys:
        {'y': array, 'setpoint': array, 'step_info': StepInfo, 'y_initial': float}

    time : array_like
        Time vector (same for all loops)

    step_time : float
        Step input time

    figsize : tuple, optional
        Figure size. Default: (16, 10)

    cols : int, optional
        Number of columns in subplot grid. Default: 2

    title_prefix : str, optional
        Prefix for each subplot title. Default: ''

    settling_threshold : float, optional
        Settling time threshold (default 0.02 → ±2% of step magnitude).
        Must match StepInfo(SettlingTimeThreshold=...) for consistency.

    show_rise_time_box : bool, optional
        Whether to show rise time (tR) annotation boxes. Default: True

    show_settling_time_box : bool, optional
        Whether to show settling time (tS) annotation boxes. Default: True

    dpi : int, optional
        DPI for saved figure. Default: None (600 if saving)

    savefig_path : str, optional
        Path to save comparison figure. Default: None

    Returns
    -------
    fig, axes : matplotlib Figure and Axes array

    Examples
    --------
    loops_dict = {
        'TIC-100': {'y': T, 'setpoint': TSP_100_SP, 'step_info': si_tic, 'y_initial': 333.15},
        'LIC-100': {'y': h, 'setpoint': LSP_100_SP, 'step_info': si_lic, 'y_initial': 1.5},
        'FIC-100': {'y': f_oil, 'setpoint': FSP_100_SP, 'step_info': si_fic100, 'y_initial': 3.29e-4},
    }

    fig, axes = compare_step_responses(
        loops_dict, time=time, step_time=600,
        title_prefix='Biodiesel Reactor Control',
        savefig_path='step_response_comparison.png'
    )
    """
    num_loops = len(loops_dict)
    rows = int(np.ceil(num_loops / cols))

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if num_loops == 1:
        axes = np.array([axes])
    else:
        axes = axes.flatten()

    for idx, (_loop_name, loop_data) in enumerate(loops_dict.items()):
        ax = axes[idx]
        y = loop_data.get('y')
        setpoint = loop_data.get('setpoint')
        step_info = loop_data.get('step_info')

        # Plot on this axis
        ax.plot(time, y, label='PV', linewidth=2.0, color='#0055cc')
        ax.plot(time, setpoint, label='SP', linestyle='--', linewidth=1.5, color='#d65a00')

        # Extract setpoint value
        idx_step = np.searchsorted(time, step_time)
        y_sp = (
            float(setpoint[idx_step:].mean()) if idx_step < len(setpoint) else float(setpoint[-1])
        )

        # Calculate step magnitude for settling band
        if (
            step_info is not None
            and hasattr(step_info, 'yfinal')
            and hasattr(step_info, 'y_initial')
        ):
            y_sp = step_info.yfinal
            y_init = step_info.y_initial
        else:
            y_init = loop_data.get('y_initial')
            if y_init is None:
                y_init = float(y[0])
            else:
                y_init = float(y_init)

        delta = y_sp - y_init
        is_zero_delta = abs(delta) < 1e-6
        step_magnitude = abs(y_sp) if (is_zero_delta and y_sp != 0) else abs(delta)

        # Settling band (matches step_info.py formula)
        y_band_low = y_sp - settling_threshold * step_magnitude
        y_band_high = y_sp + settling_threshold * step_magnitude
        ax.axhline(y=y_band_low, color='#1a1a1a', linestyle=':', linewidth=1.0, alpha=0.6)
        ax.axhline(y=y_band_high, color='#1a1a1a', linestyle=':', linewidth=1.0, alpha=0.6)
        ax.fill_between(time, y_band_low, y_band_high, alpha=0.08, color='#1a1a1a')

        # Metrics
        if step_info is not None:
            metrics = step_info.metrics
            if ylim is not None:
                ax.set_ylim(ylim)
            y_min, y_max = ax.get_ylim()

            # Rise time
            if not np.isnan(metrics.RiseTime) and show_rise_time_box:
                tr_abs = step_time + metrics.RiseTime

                # Get smart positioning (use actual settling_threshold parameter)
                if not np.isnan(metrics.SettlingTime):
                    ts_abs = step_time + metrics.SettlingTime
                    positions = _calculate_box_positions(
                        tr_abs,
                        ts_abs,
                        y,
                        setpoint,
                        y_min,
                        y_max,
                        step_time,
                        settling_threshold=settling_threshold,
                    )
                else:
                    positions = _calculate_box_positions(
                        tr_abs,
                        tr_abs + 1000,
                        y,
                        setpoint,
                        y_min,
                        y_max,
                        step_time,
                        settling_threshold=settling_threshold,
                    )

                tr_y = positions['tr_y']
                tr_va = positions['tr_va']

                # Vline crosses full plot height
                _tr_vmin = y_min
                _tr_vmax = y_max
                ax.vlines(
                    x=tr_abs,
                    ymin=_tr_vmin,
                    ymax=_tr_vmax,
                    colors='#008000',
                    linestyles='--',
                    linewidth=1.0,
                    alpha=0.6,
                    zorder=2,
                )

                ax.text(
                    x=tr_abs,
                    y=_resolve_box_y(tr_box_y, 0, tr_y),
                    s=f'$t_R$={metrics.RiseTime:.0f}s',
                    color='#008000',
                    fontsize=9,
                    ha='center',
                    va=tr_va,
                    zorder=3,
                    bbox={
                        'boxstyle': 'round,pad=0.25',
                        'facecolor': 'white',
                        'edgecolor': '#008000',
                        'alpha': 1.0,
                        'linewidth': 1.0,
                    },
                )

            # Settling time
            if not np.isnan(metrics.SettlingTime) and show_settling_time_box:
                ts_abs = step_time + metrics.SettlingTime

                # Get smart positioning
                if not np.isnan(metrics.RiseTime):
                    tr_abs = step_time + metrics.RiseTime
                    positions = _calculate_box_positions(
                        tr_abs,
                        ts_abs,
                        y,
                        setpoint,
                        y_min,
                        y_max,
                        step_time,
                        settling_threshold=0.02,
                    )
                else:
                    positions = _calculate_box_positions(
                        ts_abs - 1000,
                        ts_abs,
                        y,
                        setpoint,
                        y_min,
                        y_max,
                        step_time,
                        settling_threshold=0.02,
                    )

                ts_y = positions['ts_y']
                ts_va = positions['ts_va']

                # Vline crosses full plot height
                _ts_vmin = y_min
                _ts_vmax = y_max
                ax.vlines(
                    x=ts_abs,
                    ymin=_ts_vmin,
                    ymax=_ts_vmax,
                    colors='#cc0000',
                    linestyles='--',
                    linewidth=1.0,
                    alpha=0.6,
                    zorder=2,
                )

                ax.text(
                    x=ts_abs,
                    y=_resolve_box_y(ts_box_y, 0, ts_y),
                    s=f'$t_S$={metrics.SettlingTime:.0f}s',
                    color='#cc0000',
                    fontsize=9,
                    ha='center',
                    va=ts_va,
                    zorder=3,
                    bbox={
                        'boxstyle': 'round,pad=0.25',
                        'facecolor': 'white',
                        'edgecolor': '#cc0000',
                        'alpha': 1.0,
                        'linewidth': 1.0,
                    },
                )

        # Formatting
        # ax.set_title(...)  # no title per publication style
        ax.set_xlabel('Time (s)', fontsize=10)
        ax.set_ylabel('Output', fontsize=10)
        ax.legend(loc='best', fontsize=9, frameon=False, framealpha=0.95)
        ax.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Hide unused subplots
    for idx in range(num_loops, len(axes)):
        axes[idx].set_visible(False)

    fig.tight_layout()

    if savefig_path is not None:
        if dpi is None:
            dpi = 600
        _save_fig(fig, savefig_path, dpi)
        print(f'[OK] Comparison figure saved to: {savefig_path} (DPI={dpi})')

    return fig, axes


def plot_multi_step_response(
    time,
    y_data_dict,
    u_dict,
    steps_config,
    step_info_dict=None,
    title=None,
    ylabel=None,
    figsize=(14, 7),
    settling_threshold=0.02,
    curve_colors=None,
    setpoint_color='#1a1a1a',
    show_boxes=None,
    savefig_path=None,
    dpi=600,
    tr_box_y=None,
    ts_box_y=None,
    ylim=None,
):
    """Plot multiple steps (disturbance + setpoint) in single figure.

    Matches step_info.py logic for handling multiple evaluation windows.

    Parameters
    ----------
    time : array_like
        Time vector (same for all data)
    y_data_dict : dict
        {label: y_array} for each tuning method
    setpoint_dict : dict
        {label: setpoint_array} for each tuning method
    steps_config : dict
        {step_name: {'time': float, 'y_band_low': float, 'y_band_high': float,
                     'label_suffix': str}}
    step_info_dict : dict, optional
        {label: {step_name: StepInfo}} nested dict for metrics
    title : str, optional
        Plot title
    ylabel : str, optional
        Y-axis label
    figsize : tuple, optional
        Figure size. Default: (16, 7)
    settling_threshold : float, optional
        Settling band threshold (e.g., 0.02 for ±2%)
    curve_colors : dict, optional
        {label: color} mapping for curves
    setpoint_color : str, optional
        Color for setpoint line
    show_boxes : dict, optional
        {box_type: bool} to control visibility:
        - 'dist_ts': disturbance settling time
        - 'sp_tr': setpoint rise time
        - 'sp_ts': setpoint settling time
    savefig_path : str, optional
        Path to save figure
    dpi : int, optional
        DPI for saved figure. Default: 600

    Returns
    -------
    fig, ax : matplotlib Figure and Axes
    """
    time = np.asarray(time)

    # Default box visibility
    if show_boxes is None:
        show_boxes = {'dist_ts': True, 'sp_tr': True, 'sp_ts': True}

    # Default colors
    if curve_colors is None:
        curve_colors = {'Syn': '#0055cc', 'IAE': '#d65a00', 'QDR': '#008000'}

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot curves
    for label, y in y_data_dict.items():
        ax.plot(time, y, label=f'{label}', color=curve_colors.get(label, '#000000'), linewidth=2.0)

    # Plot input(s)
    for _label, u in u_dict.items():
        ax.plot(time, u, linestyle='--', color=setpoint_color, linewidth=1.5, alpha=0.7)

    # Add settling bands for each step
    for _step_name, step_info in steps_config.items():
        y_band_low = step_info['y_band_low']
        y_band_high = step_info['y_band_high']
        step_time = step_info['time']
        end_time = step_info.get('end_time', time[-1])

        # Time window for band
        idx_start = np.searchsorted(time, step_time)
        idx_end = np.searchsorted(time, end_time)
        time_window = time[idx_start:idx_end]

        ax.axhline(y=y_band_low, color='#1a1a1a', linestyle=':', linewidth=1.2, alpha=0.6)
        ax.axhline(y=y_band_high, color='#1a1a1a', linestyle=':', linewidth=1.2, alpha=0.6)
        ax.fill_between(
            time_window,
            y_band_low,
            y_band_high,
            alpha=0.08,
            color='#1a1a1a',
            label=f'{step_info["label_suffix"]} Band (±2%)',
        )

    # Add annotation boxes
    if step_info_dict is not None:
        if ylim is not None:
            ax.set_ylim(ylim)
        y_min, y_max = ax.get_ylim()

        # Collect all boxes data
        boxes_data = {'dist_ts': [], 'sp_tr': [], 'sp_ts': []}

        # Disturbance settling times
        if show_boxes.get('dist_ts', True):
            if 'Disturbance' in step_info_dict:
                for idx, (label, si) in enumerate(step_info_dict['Disturbance'].items()):
                    if not np.isnan(si.metrics.SettlingTime):
                        boxes_data['dist_ts'].append(
                            {
                                'label': label,
                                'time': steps_config.get('Disturbance', {}).get('time', 600)
                                + si.metrics.SettlingTime,
                                'value': si.metrics.SettlingTime,
                                'color': curve_colors.get(label, '#000000'),
                                'order': idx,
                            }
                        )

        # Setpoint rise times
        if show_boxes.get('sp_tr', True):
            if 'Setpoint' in step_info_dict:
                for idx, (label, si) in enumerate(step_info_dict['Setpoint'].items()):
                    if not np.isnan(si.metrics.RiseTime):
                        boxes_data['sp_tr'].append(
                            {
                                'label': label,
                                'time': steps_config.get('Setpoint', {}).get('time', 20600)
                                + si.metrics.RiseTime,
                                'value': si.metrics.RiseTime,
                                'color': curve_colors.get(label, '#000000'),
                                'order': idx,
                            }
                        )

        # Setpoint settling times
        if show_boxes.get('sp_ts', True):
            if 'Setpoint' in step_info_dict:
                for idx, (label, si) in enumerate(step_info_dict['Setpoint'].items()):
                    if not np.isnan(si.metrics.SettlingTime):
                        boxes_data['sp_ts'].append(
                            {
                                'label': label,
                                'time': steps_config.get('Setpoint', {}).get('time', 20600)
                                + si.metrics.SettlingTime,
                                'value': si.metrics.SettlingTime,
                                'color': curve_colors.get(label, '#000000'),
                                'order': idx,
                            }
                        )

        # Plot disturbance settling times (bottom, staggered)
        if boxes_data['dist_ts']:
            y_span_curr = y_max - y_min
            positioned = _stack_boxes_smart(
                boxes_data['dist_ts'], y_min, y_max, y_span_curr, position_type='bottom'
            )
            for i, box in enumerate(positioned):
                ax.vlines(
                    x=box['time'],
                    ymin=y_min,
                    ymax=y_max,
                    colors=box['color'],
                    linestyles='--',
                    linewidth=1.0,
                    alpha=0.5,
                    zorder=2,
                )
                ax.text(
                    x=box['time'],
                    y=_resolve_box_y(ts_box_y, i, box['y_center']),
                    s=f'$t_S$ ({box["label"]})\n{box["value"]:.0f}s',
                    color=box['color'],
                    fontsize=9,
                    ha='center',
                    va='center',
                    zorder=3,
                    bbox={
                        'boxstyle': 'round,pad=0.4',
                        'facecolor': 'white',
                        'edgecolor': box['color'],
                        'alpha': 1.0,
                        'linewidth': 1.5,
                    },
                )

        # Plot setpoint rise times (top, staggered)
        if boxes_data['sp_tr']:
            y_span_curr = y_max - y_min
            positioned = _stack_boxes_smart(
                boxes_data['sp_tr'], y_min, y_max, y_span_curr, position_type='top'
            )
            for i, box in enumerate(positioned):
                ax.vlines(
                    x=box['time'],
                    ymin=y_min,
                    ymax=y_max,
                    colors=box['color'],
                    linestyles='--',
                    linewidth=1.0,
                    alpha=0.5,
                    zorder=2,
                )
                ax.text(
                    x=box['time'],
                    y=_resolve_box_y(tr_box_y, i, box['y_center']),
                    s=f'$t_R$ ({box["label"]})\n{box["value"]:.0f}s',
                    color=box['color'],
                    fontsize=9,
                    ha='center',
                    va='center',
                    zorder=3,
                    bbox={
                        'boxstyle': 'round,pad=0.4',
                        'facecolor': 'white',
                        'edgecolor': box['color'],
                        'alpha': 1.0,
                        'linewidth': 1.5,
                    },
                )

        # Plot setpoint settling times (top, staggered below sp_tr)
        if boxes_data['sp_ts']:
            y_span_curr = y_max - y_min
            # Count how many top-level stagger slots are used by sp_tr
            num_tr_levels = (
                max([0] + [b.get('_level', 0) for b in boxes_data['sp_tr']]) + 1
                if boxes_data['sp_tr']
                else 0
            )
            # Position sp_ts below sp_tr by passing a shifted y_max
            offset = num_tr_levels * (y_span_curr * 0.06 + y_span_curr * 0.12)
            adjusted_y_max = y_max - offset
            positioned = _stack_boxes_smart(
                boxes_data['sp_ts'], y_min, adjusted_y_max, y_span_curr, position_type='top'
            )
            for i, box in enumerate(positioned):
                ax.vlines(
                    x=box['time'],
                    ymin=y_min,
                    ymax=y_max,
                    colors=box['color'],
                    linestyles=':',
                    linewidth=1.0,
                    alpha=0.5,
                    zorder=2,
                )
                ax.text(
                    x=box['time'],
                    y=_resolve_box_y(ts_box_y, i, box['y_center']),
                    s=f'$t_S$ ({box["label"]})\n{box["value"]:.0f}s',
                    color=box['color'],
                    fontsize=8,
                    ha='center',
                    va='center',
                    zorder=3,
                    bbox={
                        'boxstyle': 'round,pad=0.3',
                        'facecolor': 'lightyellow',
                        'edgecolor': box['color'],
                        'alpha': 1.0,
                        'linewidth': 1.2,
                    },
                )

    # Formatting
    ax.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    # if title:  # no title per publication style
    #     ax.set_title(title, fontsize=13, fontweight='bold')

    ax.legend(loc='upper left', fontsize=10, frameon=False, framealpha=0.95)
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()

    if savefig_path is not None:
        _save_fig(fig, savefig_path, dpi)
        print(f'[OK] Figure saved to: {savefig_path} (DPI={dpi})')

    return fig, ax


def plot_dual_response(
    time,
    y_data_dict,
    u_array,
    step1_time,
    step1_end_time,
    step2_time,
    step_info_dict=None,
    title=None,
    ylabel=None,
    figsize=(14, 7),
    step1_name='Disturbance',
    step2_name='Setpoint',
    step1_threshold=0.02,
    step2_threshold=0.02,
    curve_colors=None,
    setpoint_color='#1a1a1a',
    show_step1_ts=True,
    show_step2_tr=True,
    show_step2_ts=True,
    savefig_path=None,
    dpi=600,
    tr_box_y=None,
    ts_box_y=None,
    ylim=None,
):
    """Plot dual-step response (e.g., disturbance rejection + setpoint tracking).

    Specifically designed for comparing multiple tuning methods across two separate
    step scenarios in a single plot, with tR and tS boxes for each step.

    Parameters
    ----------
    time : array_like
        Time vector (seconds)
    y_data_dict : dict
        {label: y_array} for each tuning method
    setpoint_array : array_like
        Setpoint array (matches length of time)
    step1_time : float
        Time of step 1 (disturbance) in seconds
    step1_end_time : float
        End evaluation window for step 1
    step2_time : float
        Time of step 2 (setpoint) in seconds
    step_info_dict : dict, optional
        Nested dict: {label: {step_name: StepInfo}}
        Example: {'Syn': {'Disturbance': StepInfo(...), 'Setpoint': StepInfo(...)}, ...}
    title : str, optional
        Plot title
    ylabel : str, optional
        Y-axis label (with units if needed)
    figsize : tuple, optional
        Figure size. Default: (16, 7)
    step1_name, step2_name : str, optional
        Names of step 1 and step 2. Default: 'Disturbance', 'Setpoint'
    step1_threshold, step2_threshold : float, optional
        Settling band thresholds (e.g., 0.02 for ±2%)
    curve_colors : dict, optional
        {label: color} for each tuning method
    setpoint_color : str, optional
        Color for input line
    show_step1_ts : bool, optional
        Show settling time boxes for step 1. Default: True
    show_step2_tr : bool, optional
        Show rise time boxes for step 2. Default: True
    show_step2_ts : bool, optional
        Show settling time boxes for step 2. Default: True
    savefig_path : str, optional
        Path to save figure
    dpi : int, optional
        DPI for saved figure. Default: 300

    Returns
    -------
    fig, ax : matplotlib Figure and Axes

    Examples
    --------
    >>> fig, ax = plot_dual_step_response(
    ...     time=time,
    ...     y_data_dict={'Syn': y_syn, 'IAE': y_iae, 'QDR': y_qdr},
    ...     setpoint_array=setpoint,
    ...     step1_time=600, step1_end_time=20000, step2_time=20600,
    ...     step_info_dict={
    ...         'Syn': {'Disturbance': si_syn_dist, 'Setpoint': si_syn_sp},
    ...         'IAE': {'Disturbance': si_iae_dist, 'Setpoint': si_iae_sp},
    ...         'QDR': {'Disturbance': si_qdr_dist, 'Setpoint': si_qdr_sp}
    ...     },
    ...     title='TIC-100: Dual-Step Response',
    ...     ylabel='Temperature (°C)',
    ...     curve_colors={'Syn': '#0055cc', 'IAE': '#d65a00', 'QDR': '#008000'},
    ...     show_step1_ts=True, show_step2_tr=True, show_step2_ts=True
    ... )
    >>> plt.show()
    """
    time = np.asarray(time)
    u_array = np.asarray(u_array)

    # Default colors
    if curve_colors is None:
        curve_colors = {'Syn': '#0055cc', 'IAE': '#d65a00', 'QDR': '#008000'}

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot all curves
    for label, y in y_data_dict.items():
        y = np.asarray(y)
        color = curve_colors.get(label, '#000000')
        ax.plot(time, y, label=label, color=color, linewidth=2.0)

    # Plot input
    ax.plot(time, u_array, label='Reference', linestyle='--', color=setpoint_color, linewidth=1.5)

    # Get y-axis limits
    y_min, y_max = ax.get_ylim()
    y_span = y_max - y_min

    # === STEP 1: Disturbance Rejection ===
    # Find input value during step 1 (should be constant)
    idx_step1 = np.searchsorted(time, step1_time)
    idx_step1_end = np.searchsorted(time, step1_end_time)
    sp1_value = float(np.mean(u_array[idx_step1:idx_step1_end]))

    # Calculate step 1 reference values (usually no change in input)
    if step_info_dict and len(step_info_dict) > 0:
        first_label = list(step_info_dict.keys())[0]
        if step1_name in step_info_dict[first_label]:
            si_first = step_info_dict[first_label][step1_name]
            y_initial_1 = si_first.y_initial
            y_final_1 = si_first.yfinal
        else:
            y_initial_1 = sp1_value
            y_final_1 = sp1_value
    else:
        y_initial_1 = sp1_value
        y_final_1 = sp1_value

    delta_1 = y_final_1 - y_initial_1
    is_zero_delta_1 = abs(delta_1) < 1e-6
    step_mag_1 = abs(y_final_1) if (is_zero_delta_1 and y_final_1 != 0) else abs(delta_1)

    # Settling band for step 1
    y_band_low_1 = y_final_1 - step1_threshold * step_mag_1
    y_band_high_1 = y_final_1 + step1_threshold * step_mag_1

    time_window_1 = time[idx_step1:idx_step1_end]
    ax.axhline(y=y_band_low_1, color='#1a1a1a', linestyle=':', linewidth=1.2, alpha=0.6)
    ax.axhline(y=y_band_high_1, color='#1a1a1a', linestyle=':', linewidth=1.2, alpha=0.6)
    ax.fill_between(
        time_window_1,
        y_band_low_1,
        y_band_high_1,
        alpha=0.08,
        color='#1a1a1a',
        label=f'Step 1 Band (±{step1_threshold * 100:.0f}%)',
    )

    # === STEP 2: Setpoint Response ===
    idx_step2 = np.searchsorted(time, step2_time)
    time_window_2 = time[idx_step2:]
    sp2_value = float(u_array[idx_step2])

    # Calculate step 2 reference values
    if step_info_dict and len(step_info_dict) > 0:
        first_label = list(step_info_dict.keys())[0]
        if step2_name in step_info_dict[first_label]:
            si_first = step_info_dict[first_label][step2_name]
            y_initial_2 = si_first.y_initial
            y_final_2 = si_first.yfinal
        else:
            y_initial_2 = sp1_value
            y_final_2 = sp2_value
    else:
        y_initial_2 = sp1_value
        y_final_2 = sp2_value

    delta_2 = y_final_2 - y_initial_2
    is_zero_delta_2 = abs(delta_2) < 1e-6
    step_mag_2 = abs(y_final_2) if (is_zero_delta_2 and y_final_2 != 0) else abs(delta_2)

    # Settling band for step 2
    y_band_low_2 = y_final_2 - step2_threshold * step_mag_2
    y_band_high_2 = y_final_2 + step2_threshold * step_mag_2

    ax.axhline(y=y_band_low_2, color='#1a1a1a', linestyle=':', linewidth=1.2, alpha=0.6)
    ax.axhline(y=y_band_high_2, color='#1a1a1a', linestyle=':', linewidth=1.2, alpha=0.6)
    ax.fill_between(
        time_window_2,
        y_band_low_2,
        y_band_high_2,
        alpha=0.08,
        color='#1a1a1a',
        label=f'Step 2 Band (±{step2_threshold * 100:.0f}%)',
    )

    # === ANNOTATION BOXES ===
    # tR boxes at bottom, tS boxes at top; vlines cross full plot height.
    # Boxes drawn with opaque white background so they appear to "interrupt" the vline cleanly.
    margin_from_plot = y_span * 0.15

    # Collect boxes
    boxes_step1_ts = []
    boxes_step2_tr = []
    boxes_step2_ts = []

    # Step 1 Settling Times
    if show_step1_ts and step_info_dict:
        for label, step_dict in step_info_dict.items():
            if step1_name in step_dict:
                si = step_dict[step1_name]
                if not np.isnan(si.metrics.SettlingTime):
                    boxes_step1_ts.append(
                        {
                            'label': label,
                            'time': step1_time + si.metrics.SettlingTime,
                            'value': si.metrics.SettlingTime,
                            'color': curve_colors.get(label, '#000000'),
                        }
                    )

    # Step 2 Rise Times
    if show_step2_tr and step_info_dict:
        for label, step_dict in step_info_dict.items():
            if step2_name in step_dict:
                si = step_dict[step2_name]
                if not np.isnan(si.metrics.RiseTime):
                    # Get specific curve data for this label
                    y_curve = np.asarray(y_data_dict[label])

                    # Get y_initial and y_final from the step_info object (if any), or use default Step 2
                    curve_y_init = si.y_initial if hasattr(si, 'y_initial') else y_initial_2
                    curve_y_fin = si.yfinal if hasattr(si, 'yfinal') else y_final_2
                    curve_delta = curve_y_fin - curve_y_init

                    # Find the absolute time when the curve reaches 90% of delta
                    if abs(curve_delta) > 1e-6:
                        idx_step = np.searchsorted(time, step2_time)
                        threshold_90 = curve_y_init + 0.9 * curve_delta
                        y_slice = y_curve[idx_step:]

                        # Check step direction (up or down)
                        if curve_delta > 0:
                            crossing = y_slice >= threshold_90
                        else:
                            crossing = y_slice <= threshold_90

                        crossing_indices = np.nonzero(crossing)[0]
                        if crossing_indices.size > 0:
                            rt_abs = time[idx_step + crossing_indices[0]]
                        else:
                            # Fallback jika kurva aneh/tidak sampai 90%
                            rt_abs = step2_time + si.metrics.RiseTime
                    else:
                        rt_abs = step2_time + si.metrics.RiseTime

                    # Add to the box list with the x (time) position already accurate at 90%
                    boxes_step2_tr.append(
                        {
                            'label': label,
                            'time': rt_abs,
                            'value': si.metrics.RiseTime,
                            'color': curve_colors.get(label, '#000000'),
                        }
                    )

    # Step 2 Settling Times
    if show_step2_ts and step_info_dict:
        for label, step_dict in step_info_dict.items():
            if step2_name in step_dict:
                si = step_dict[step2_name]
                if not np.isnan(si.metrics.SettlingTime):
                    boxes_step2_ts.append(
                        {
                            'label': label,
                            'time': step2_time + si.metrics.SettlingTime,
                            'value': si.metrics.SettlingTime,
                            'color': curve_colors.get(label, '#000000'),
                        }
                    )

    # === EXTEND Y-AXIS based on actual stagger levels needed ===
    x_lo, x_hi = ax.get_xlim()
    all_ts_boxes_combined = boxes_step1_ts + boxes_step2_ts

    # How many vertical stagger slots are needed?
    n_levels_tr = _needed_levels(boxes_step2_tr, x_lo, x_hi)
    n_levels_ts = _needed_levels(all_ts_boxes_combined, x_lo, x_hi)

    # Each slot: box_height (6.5% y_span) + margin (2%) = 8.5%
    slot_frac = 0.065 + 0.02
    edge_frac = 0.06  # margin_from_axis in _stack_boxes_smart

    if n_levels_tr > 0 or n_levels_ts > 0:
        y_span_data = y_max - y_min
        space_bottom = (
            edge_frac + n_levels_tr * slot_frac + 0.065 / 2
        ) * y_span_data + margin_from_plot
        space_top = (
            edge_frac + n_levels_ts * slot_frac + 0.065 / 2
        ) * y_span_data + margin_from_plot
        ax.set_ylim(y_min - space_bottom, y_max + space_top)

    cur_y_min, cur_y_max = ax.get_ylim()
    if ylim is not None:
        ax.set_ylim(ylim)
        cur_y_min, cur_y_max = ylim
    cur_y_span = cur_y_max - cur_y_min

    # === PLOT tR BOXES at BOTTOM (Step 2 only), staggered ===
    if boxes_step2_tr:
        positioned = _stack_boxes_smart(
            boxes_step2_tr,
            cur_y_min,
            cur_y_max,
            cur_y_span,
            position_type='bottom',
            x_min=x_lo,
            x_max=x_hi,
        )
        for i, box in enumerate(positioned):
            ax.vlines(
                x=box['time'],
                ymin=cur_y_min,
                ymax=cur_y_max,
                colors=box['color'],
                linestyles='--',
                linewidth=1.0,
                alpha=0.5,
                zorder=2,
            )
            ax.text(
                x=box['time'],
                y=_resolve_box_y(tr_box_y, i, box['y_center']),
                s=f'$t_R$ = {box["value"]:.0f} s\n({box["label"]})',
                color=box['color'],
                fontsize=9,
                fontweight='bold',
                ha='center',
                va='center',
                zorder=3,
                bbox={
                    'boxstyle': 'round,pad=0.4',
                    'facecolor': 'white',
                    'edgecolor': box['color'],
                    'alpha': 1.0,
                    'linewidth': 1.5,
                },
            )

    # === PLOT tS BOXES at TOP (Step 1 + Step 2), staggered ===
    if all_ts_boxes_combined:
        positioned = _stack_boxes_smart(
            all_ts_boxes_combined,
            cur_y_min,
            cur_y_max,
            cur_y_span,
            position_type='top',
            x_min=x_lo,
            x_max=x_hi,
        )
        for i, box in enumerate(positioned):
            ax.vlines(
                x=box['time'],
                ymin=cur_y_min,
                ymax=cur_y_max,
                colors=box['color'],
                linestyles=':',
                linewidth=1.0,
                alpha=0.5,
                zorder=2,
            )
            ax.text(
                x=box['time'],
                y=_resolve_box_y(ts_box_y, i, box['y_center']),
                s=f'$t_S$ = {box["value"]:.0f} s\n({box["label"]})',
                color=box['color'],
                fontsize=9,
                fontweight='bold',
                ha='center',
                va='center',
                zorder=3,
                bbox={
                    'boxstyle': 'round,pad=0.4',
                    'facecolor': 'white',
                    'edgecolor': box['color'],
                    'alpha': 1.0,
                    'linewidth': 1.5,
                },
            )

    # Formatting
    ax.set_xlabel('Time (s)', fontsize=12, fontweight='bold', labelpad=8)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    # if title:  # no title per publication style
    #     ax.set_title(title, fontsize=13, fontweight='bold')

    # Legend below the x-axis
    ax.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, -0.09),
        ncol=6,
        fontsize=10,
        frameon=True,
        framealpha=0.95,
        edgecolor='gray',
        fancybox=True,
    )
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)

    # Set y-axis limits (freeze to cur_y_min and cur_y_max to prevent autoscale expansion)
    if ylim is not None:
        ax.set_ylim(ylim)
    else:
        ax.set_ylim(cur_y_min, cur_y_max)

    plt.tight_layout()

    if savefig_path is not None:
        _save_fig(fig, savefig_path, dpi)
        print(f'[OK] Figure saved to: {savefig_path} (DPI={dpi})')

    return fig, ax


def plot_simple_response(
    time,
    y,
    u=None,
    title=None,
    ylabel=None,
    xlabel='Time (s)',
    y_label_unit=None,
    figsize=(14, 7),
    curve_color='#0055cc',
    input_color='#d65a00',
    legend_loc='upper center',
    grid_alpha=0.3,
    tight_layout=True,
    dpi=None,
    savefig_path=None,
    ylim=None,
):
    """Create simple plot of process variable and optional input (no metrics/boxes).

    Uses publication-quality styling matching other plot functions.

    Parameters
    ----------
    time : array_like
        Time vector [seconds]
    y : array_like
        Process variable (output) response
    u : array_like, optional
        Input signal (same length as time). Default: None (not plotted)
    title : str, optional
        Plot title. If None, no title is shown.
    ylabel : str, optional
        Y-axis label
    xlabel : str, optional
        X-axis label. Default: 'Time (s)'
    y_label_unit : str, optional
        Unit for y-axis (added to ylabel if provided)
    figsize : tuple, optional
        Figure size (width, height). Default: (14, 7)
    curve_color : str, optional
        Color for process variable curve. Default: '#0055cc' (dark blue)
    setpoint_color : str, optional
        Color for setpoint reference. Default: '#d65a00' (dark orange)
    legend_loc : str, optional
        Legend location. Default: 'upper center'
    grid_alpha : float, optional
        Alpha transparency for grid. Default: 0.3
    tight_layout : bool, optional
        Apply tight_layout before showing. Default: True
    dpi : int, optional
        DPI for saved figure. Default: None (600 if saving)
    savefig_path : str, optional
        Path to save figure (e.g., 'plot.png'). If None, figure is not saved.
    ylim : tuple, optional
        Y-axis limits (ymin, ymax). If None, auto-calculated.

    Returns
    -------
    fig, ax : matplotlib Figure and Axes objects
    """
    time = np.asarray(time)
    y = np.asarray(y)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot process variable
    ax.plot(time, y, label='Process Variable (PV)', color=curve_color, linewidth=2.0)

    # Plot input if provided
    if u is not None:
        u = np.asarray(u)
        # Handle scalar input
        if u.ndim == 0:
            u = np.full_like(time, float(u))
        ax.plot(time, u, label='Setpoint (SP)', linestyle='--', color=input_color, linewidth=1.5)

    # Labels and formatting
    if ylabel:
        ax.set_ylabel(
            f'{ylabel} ({y_label_unit})' if y_label_unit else ylabel, fontsize=12, fontweight='bold'
        )

    ax.set_xlabel(xlabel, fontsize=12, fontweight='bold', labelpad=8)

    # if title is not None:  # no title per publication style
    #     ax.set_title(title, fontsize=13, fontweight='bold')

    # Legend
    ax.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, -0.09),
        ncol=6,
        fontsize=10,
        frameon=True,
        framealpha=0.95,
        edgecolor='gray',
        fancybox=True,
    )
    ax.grid(True, alpha=grid_alpha)

    # Publication-quality formatting (match plot_step_response_multiple style)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)
        spine.set_color('#1a1a1a')
    ax.tick_params(axis='both', which='major', labelsize=10)

    # Set y-axis limits if specified
    if ylim is not None:
        ax.set_ylim(ylim)

    if tight_layout:
        fig.tight_layout()

    # Save figure if path provided
    if savefig_path is not None:
        if dpi is None:
            dpi = 600
        _save_fig(fig, savefig_path, dpi)
        print(f'[OK] Figure saved to: {savefig_path} (DPI={dpi})')

    return fig, ax


def plot_simple_response_multiple(
    time,
    y_data_dict,
    u=None,
    title=None,
    ylabel=None,
    xlabel='Time (s)',
    y_label_unit=None,
    figsize=(14, 7),
    curve_colors=None,
    input_color='#d65a00',
    legend_loc='upper center',
    grid_alpha=0.3,
    tight_layout=True,
    dpi=None,
    savefig_path=None,
    ylim=None,
):
    """Create simple plot of multiple process variables and optional input (no metrics/boxes).

    Uses publication-quality styling matching other plot functions.

    Parameters
    ----------
    time : array_like
        Time vector [seconds]
    y_data_dict : dict
        Dictionary mapping labels to y arrays: {'Tuning1': y1, 'Tuning2': y2, ...}
    u : array_like or scalar, optional
        Setpoint reference signal (same length as time, or scalar). Default: None (not plotted)
    title : str, optional
        Plot title. If None, no title is shown.
    ylabel : str, optional
        Y-axis label
    xlabel : str, optional
        X-axis label. Default: 'Time (s)'
    y_label_unit : str, optional
        Unit for y-axis (added to ylabel if provided)
    figsize : tuple, optional
        Figure size (width, height). Default: (14, 7)
    curve_colors : dict or list, optional
        Colors for each curve. Can be:
        - dict: {'label1': '#0055cc', 'label2': '#d65a00', ...}
        - list: ['#0055cc', '#d65a00', '#008000', ...]
        Default: automatic colors from palette
    setpoint_color : str, optional
        Color for setpoint reference. Default: '#d65a00' (dark orange)
    legend_loc : str, optional
        Legend location. Default: 'upper center'
    grid_alpha : float, optional
        Alpha transparency for grid. Default: 0.3
    tight_layout : bool, optional
        Apply tight_layout before showing. Default: True
    dpi : int, optional
        DPI for saved figure. Default: None (600 if saving)
    savefig_path : str, optional
        Path to save figure (e.g., 'plot.png'). If None, figure is not saved.
    ylim : tuple, optional
        Y-axis limits (ymin, ymax). If None, auto-calculated.

    Returns
    -------
    fig, ax : matplotlib Figure and Axes objects

    Examples
    --------
    >>> y_dict = {'Tuning1': y1, 'Tuning2': y2, 'Tuning3': y3}
    >>> colors = {'Tuning1': '#0055cc', 'Tuning2': '#d65a00', 'Tuning3': '#008000'}
    >>> fig, ax = plot_simple_response_multiple(
    ...     time=time, y_data_dict=y_dict, u=setpoint,
    ...     title='TIC-100 Comparison', ylabel='Temperature', y_label_unit='K',
    ...     curve_colors=colors, savefig_path='comparison.png'
    ... )
    """
    time = np.asarray(time)

    # Setup colors
    default_colors = ['#0055cc', '#d65a00', '#008000', '#cc0000', '#9400d3', '#ff8c00']
    if curve_colors is None:
        colors_dict = {
            label: default_colors[i % len(default_colors)]
            for i, label in enumerate(y_data_dict.keys())
        }
    elif isinstance(curve_colors, dict):
        colors_dict = curve_colors
    else:
        colors_dict = {
            label: curve_colors[i % len(curve_colors)] for i, label in enumerate(y_data_dict.keys())
        }

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot all curves
    for label, y in y_data_dict.items():
        y = np.asarray(y)
        color = colors_dict[label]
        ax.plot(time, y, label=label, color=color, linewidth=2.0)

    # Plot input if provided
    if u is not None:
        u = np.asarray(u)
        # Handle scalar input
        if u.ndim == 0:
            u = np.full_like(time, float(u))
        ax.plot(time, u, label='Setpoint', linestyle='--', color=input_color, linewidth=1.5)

    # Labels and formatting
    if ylabel:
        ax.set_ylabel(
            f'{ylabel} ({y_label_unit})' if y_label_unit else ylabel, fontsize=12, fontweight='bold'
        )

    ax.set_xlabel(xlabel, fontsize=12, fontweight='bold', labelpad=8)

    # if title is not None:  # no title per publication style
    #     ax.set_title(title, fontsize=13, fontweight='bold')

    # Legend
    ax.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, -0.09),
        ncol=6,
        fontsize=10,
        frameon=True,
        framealpha=0.95,
        edgecolor='gray',
        fancybox=True,
    )
    ax.grid(True, alpha=grid_alpha)

    # Publication-quality formatting (match plot_step_response_multiple style)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)
        spine.set_color('#1a1a1a')
    ax.tick_params(axis='both', which='major', labelsize=10)

    # Set y-axis limits if specified
    if ylim is not None:
        ax.set_ylim(ylim)

    if tight_layout:
        fig.tight_layout()

    # Save figure if path provided
    if savefig_path is not None:
        if dpi is None:
            dpi = 600
        _save_fig(fig, savefig_path, dpi)
        print(f'[OK] Figure saved to: {savefig_path} (DPI={dpi})')

    return fig, ax


def setup_publication_style(font_family='serif', font_size=11, font_serif=None):
    """Configure matplotlib for publication-quality plots.

    Parameters
    ----------
    font_family : str, optional
        Font family ('serif', 'sans-serif', 'monospace'). Default: 'serif'
    font_size : int, optional
        Base font size. Default: 11
    font_serif : list, optional
        List of serif fonts to try (e.g., ['Times New Roman', 'Times']).
        If None, uses matplotlib default.

    Examples
    --------
    setup_publication_style(font_family='serif', font_size=12,
                           font_serif=['Times New Roman', 'Times'])
    """
    plt.rcParams['font.family'] = font_family
    if font_serif is not None:
        plt.rcParams['font.serif'] = font_serif
    plt.rcParams['font.size'] = font_size

    # Line and marker styling
    plt.rcParams['lines.linewidth'] = 1.5
    plt.rcParams['lines.markersize'] = 6

    # Axes styling
    plt.rcParams['axes.linewidth'] = 1.0
    plt.rcParams['axes.labelsize'] = font_size
    plt.rcParams['axes.titlesize'] = font_size + 1

    # Tick styling
    plt.rcParams['xtick.labelsize'] = font_size - 1
    plt.rcParams['ytick.labelsize'] = font_size - 1
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['xtick.major.width'] = 0.8
    plt.rcParams['ytick.major.width'] = 0.8

    # Legend styling
    plt.rcParams['legend.fontsize'] = font_size - 1
    plt.rcParams['legend.frameon'] = False

    # Figure styling
    plt.rcParams['figure.dpi'] = 150
    plt.rcParams['savefig.dpi'] = 600

    print('Publication style configured')
