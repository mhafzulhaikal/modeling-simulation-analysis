"""Generate bar comparison plots for validation metrics.

This script compares the MAE, RMSE, MAPE, and PBIAS of HYSYS vs Python
for the following controllers:
1. FIC-100 (syn)        — Setpoint Tracking
2. LIC-100 (TLC)        — Setpoint-Tight (Tight Level Control)
3. TIC-100 (QDR)        — Setpoint-QDR   (Quarter Decay Ratio)
"""

import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from model.plotutils import setup_publication_style  # noqa: E402

# ── Output directory ────────────────────────────────────────────────────────
OUTPUT_DIR = project_root / 'outputs'
os.makedirs(OUTPUT_DIR / 'plots', exist_ok=True)

# ── Load validation metrics ──────────────────────────────────────────────────
json_path = OUTPUT_DIR / 'reports' / 'validation_metrics.json'

try:
    with open(json_path) as f:
        metrics = json.load(f)

    fic_data = metrics['FIC-100_Setpoint_PV']
    lic_data = metrics['LIC-100_Setpoint-Tight_PV']
    tic_data = metrics['TIC-100_Setpoint-QDR_PV']

    mae_vals = [fic_data['MAE'], lic_data['MAE'], tic_data['MAE']]
    rmse_vals = [fic_data['RMSE'], lic_data['RMSE'], tic_data['RMSE']]
    mape_vals = [fic_data['MAPE'], lic_data['MAPE'], tic_data['MAPE']]
    pbias_vals = [fic_data['PBIAS'], lic_data['PBIAS'], tic_data['PBIAS']]

    print('Loaded metrics successfully from JSON.')
except Exception as e:
    print(f'Warning: Could not read JSON file ({e}). Using fallback values.')
    mae_vals = [0.010069866873214598, 0.013866084705299948, 0.04046838680754328]
    rmse_vals = [0.023137425104036508, 0.03766346402390984, 0.14103085723180295]
    mape_vals = [0.01722279705238138, 0.023760798563472393, 0.0704566987681776]
    pbias_vals = [-0.005808741577535247, 0.023242806848261195, 0.0579770226359586]

# ── Shared settings ──────────────────────────────────────────────────────────
# Colour palette — matched to the project's publication palette in plotutils.py
COLOR_MAE = '#0055cc'  # dark blue  (primary curve colour in plotutils)
COLOR_RMSE = '#d65a00'  # dark orange (setpoint / secondary colour in plotutils)
COLOR_MAPE = '#008000'  # dark green  (rise-time marker colour in plotutils)
COLOR_PBIAS = '#cc0000'  # dark red    (settling-time marker colour in plotutils)

LABELS = ['FIC-100 (syn)', 'LIC-100 (TLC)', 'TIC-100 (QDR)']
x = np.arange(len(LABELS))
WIDTH = 0.35  # bar width

DPI = 600

# Apply publication style — identical call to generate_plots.py
setup_publication_style(
    font_family='serif',
    font_size=11,
    font_serif=['Times New Roman', 'Times'],
)


# ── Helper: annotate bar value ───────────────────────────────────────────────
def _annotate_bars(ax, rects, fmt='{:.4f}', signed=False):
    """Place value labels above (or below for negatives) each bar."""
    for rect in rects:
        height = rect.get_height()
        # For negative bars: xy is at bar bottom (most negative point);
        # va='top' + negative dy pushes label further below the bar.
        if signed and height < 0:
            va, dy = 'top', -4
        else:
            va, dy = 'bottom', 4
        label = fmt.format(height) + ('%' if signed else '')
        ax.annotate(
            label,
            xy=(rect.get_x() + rect.get_width() / 2.0, height),
            xytext=(0, dy),
            textcoords='offset points',
            ha='center',
            va=va,
            fontsize=9,
        )


# ── Helper: apply publication spine / tick / grid style ─────────────────────
def _apply_pub_style(ax, grid_alpha=0.3):
    """Apply the same spine/tick/grid formatting used across plotutils.py."""
    ax.grid(True, alpha=grid_alpha)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)
        spine.set_color('#1a1a1a')
    ax.tick_params(axis='both', which='major', labelsize=10)


# ── Helper: save figure in plotutils style ───────────────────────────────────
def _savefig(fig, path, dpi=DPI):
    """Save figure as SVG vector format with an outer border frame."""
    import matplotlib.patches as mpatches

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
    fig.savefig(path, format='svg', bbox_inches='tight', pad_inches=0.05)
    print(f'[OK] Figure saved to: {path} (SVG vector)')


# ============================================================
# PLOT 1 — MAE & RMSE
# ============================================================
fig, ax = plt.subplots(figsize=(9, 5))

rects1 = ax.bar(
    x - WIDTH / 2, mae_vals, WIDTH, label='MAE', color=COLOR_MAE, edgecolor='#1a1a1a', linewidth=0.8
)
rects2 = ax.bar(
    x + WIDTH / 2,
    rmse_vals,
    WIDTH,
    label='RMSE',
    color=COLOR_RMSE,
    edgecolor='#1a1a1a',
    linewidth=0.8,
)

# Axes labels — matching plotutils.py style (fontsize=12, fontweight='bold', labelpad=8)
ax.set_ylabel('Error Magnitude (%TO)', fontsize=12, fontweight='bold', labelpad=8)
ax.set_xlabel('Controller / Tuning Scenario', fontsize=12, fontweight='bold', labelpad=8)

ax.set_xticks(x)
ax.set_xticklabels(LABELS, fontsize=10)
ax.set_ylim(0, max(max(mae_vals), max(rmse_vals)) * 1.25)

# Value annotations
_annotate_bars(ax, rects1, fmt='{:.4f}')
_annotate_bars(ax, rects2, fmt='{:.4f}')

# Legend — same style as plotutils multi-curve plots (placed below axes)
ax.legend(
    loc='upper center',
    bbox_to_anchor=(0.5, -0.12),
    ncol=2,
    frameon=True,
    fontsize=9,
    framealpha=0.95,
    edgecolor='gray',
    fancybox=True,
)

_apply_pub_style(ax)
fig.tight_layout()

plot1_path = OUTPUT_DIR / 'plots' / 'validation_comparison_mae_rmse.svg'
_savefig(fig, plot1_path)
plt.close(fig)

# ============================================================
# PLOT 2 — MAPE & PBIAS
# ============================================================
fig, ax = plt.subplots(figsize=(9, 5))

rects1 = ax.bar(
    x - WIDTH / 2,
    mape_vals,
    WIDTH,
    label='MAPE (%)',
    color=COLOR_MAPE,
    edgecolor='#1a1a1a',
    linewidth=0.8,
)
rects2 = ax.bar(
    x + WIDTH / 2,
    pbias_vals,
    WIDTH,
    label='PBIAS (%)',
    color=COLOR_PBIAS,
    edgecolor='#1a1a1a',
    linewidth=0.8,
)

# Axes labels
ax.set_ylabel('Error / Bias (%)', fontsize=12, fontweight='bold', labelpad=8)
ax.set_xlabel('Controller / Tuning Scenario', fontsize=12, fontweight='bold', labelpad=8)

ax.set_xticks(x)
ax.set_xticklabels(LABELS, fontsize=10)

# Zero reference line — draw before bars annotations, high zorder so bars don't hide it
ax.axhline(0, color='#1a1a1a', linewidth=0.8, linestyle='-', zorder=3)

# Value annotations (signed: negatives go below bar, positives above)
_annotate_bars(ax, rects1, fmt='{:.4f}', signed=True)
_annotate_bars(ax, rects2, fmt='{:.4f}', signed=True)

# Y-limits — padding is proportional to the full displayed range so that the
# small negative PBIAS bar and its label are clearly visible.
all_vals = mape_vals + pbias_vals
y_min_v, y_max_v = min(all_vals), max(all_vals)
y_range = y_max_v - y_min_v
# Give at least 20 % of full range below the most negative value (room for label)
pad_lo = max(abs(y_min_v) * 0.5, y_range * 0.20) if y_min_v < 0 else 0
pad_hi = y_range * 0.20  # room for positive annotations
ax.set_ylim(y_min_v - pad_lo, y_max_v + pad_hi)

# Legend
ax.legend(
    loc='upper center',
    bbox_to_anchor=(0.5, -0.12),
    ncol=2,
    frameon=True,
    fontsize=9,
    framealpha=0.95,
    edgecolor='gray',
    fancybox=True,
)

_apply_pub_style(ax)
fig.tight_layout()

plot2_path = OUTPUT_DIR / 'plots' / 'validation_comparison_mape_pbias.svg'
_savefig(fig, plot2_path)
plt.close(fig)
