"""Generate plots for step-test response analysis.

Usage:
    python analysis/generate_plots.py
"""

import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np

# Import configurations and the calculation function
from calculate_performance import (
    DATA_DIR,
    FIC_SP_INITIAL,
    FIC_STEP_TIME,
    LIC_SP_INITIAL,
    LIC_SP_STEP_TIME,
    OUTPUT_DIR,
    THRESH_FIC,
    THRESH_LIC,
    THRESH_TIC,
    THRESH_TIC_DIST,
    TIC_DIST_END_TIME,
    TIC_DIST_STEP_TIME,
    TIC_SP_STEP_TIME,
    _arr,
    _load,
    calculate_all_performance,
)

from model import (
    compare_step_responses,
    plot_dual_response,
    plot_response,
    plot_simple_response,
    plot_simple_response_multiple,
    plot_step_response_multiple,
    setup_publication_style,
)
from model.plotutils import _save_fig

# Configuration for plotting
FONT_FAMILY = 'serif'
FONT_SIZE = 12
FONT_SERIF = ['Times New Roman', 'Times']
COLORS = {'Syn': '#0055cc', 'QDR': '#d65a00', 'IAE': '#008000'}
DPI = 600


def _out(loop: str, subfolder: str, filename: str) -> str:
    """Generate output path and ensure the directory exists."""
    if filename.lower().endswith('.png') or filename.lower().endswith('.emf'):
        filename = os.path.splitext(filename)[0] + '.svg'
    d = os.path.join(OUTPUT_DIR, 'plots', loop, subfolder)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, filename)


def _interp(t_dest: np.ndarray, df_src, col_src: str) -> np.ndarray:
    """Interpolate a source DataFrame column onto a destination time array."""
    return np.interp(t_dest, _arr(df_src, 'Time'), _arr(df_src, col_src))


def _generate_fic_plots(results: dict):
    """Generate plots for flow controllers (FIC-100, FIC-101, FIC-102)."""
    for loop_num in (100, 101, 102):
        loop_name = f'FIC-{loop_num}'

        # Load Python files
        py_path = os.path.join(DATA_DIR, loop_name, 'Python', f'FIC_{loop_num}_Setpoint.CSV')
        df_py = _load(py_path)
        t_py = _arr(df_py, 'Time')
        C_py = _arr(df_py, f'FT-{loop_num} - C')
        R_py = _arr(df_py, f'FSP-{loop_num} - R')
        M_py = _arr(df_py, f'FC-{loop_num} - M')

        # Python Setpoint PV plot
        fig, _ = plot_response(
            t_py,
            C_py,
            R_py,
            step_time=FIC_STEP_TIME,
            step_info=results[f'si_fic{loop_num}_py'],
            y_initial=FIC_SP_INITIAL,
            settling_threshold=THRESH_FIC,
            title=f'{loop_name} (Python): Setpoint Tracking',
            ylabel='Transmitter Output (%TO)',
            savefig_path=_out(loop_name, 'Python', f'FIC{loop_num}_Python_Setpoint_PV.png'),
            dpi=DPI,
        )
        plt.close(fig)

        # Python Setpoint CO plot
        fig, _ = plot_simple_response(
            t_py,
            M_py,
            title=f'{loop_name} (Python): Controller Output',
            ylabel='Controller Output (%CO)',
            savefig_path=_out(loop_name, 'Python', f'FIC{loop_num}_Python_Setpoint_CO.png'),
            dpi=DPI,
        )
        plt.close(fig)

        # Python Setpoint PV plot for REPORT
        fig_sp_rep, _ = plot_response(
            t_py,
            C_py,
            R_py,
            step_time=FIC_STEP_TIME,
            step_info=results[f'si_fic{loop_num}_py'],
            y_initial=FIC_SP_INITIAL,
            settling_threshold=THRESH_FIC,
            title=f'{loop_name} (Python): Servo Response',
            ylabel='Transmitter Output (%TO)',
            savefig_path=_out(loop_name, 'report', f'FIC{loop_num}_Servo_Response.png'),
            dpi=DPI,
        )
        plt.close(fig_sp_rep)

        # HYSYS Setpoint PV plot
        hy_path = os.path.join(DATA_DIR, loop_name, 'HYSYS', f'FIC_{loop_num}_setpoint.CSV')
        df_hy = _load(hy_path)
        t_hy = _arr(df_hy, 'Time')
        C_hy_raw = _arr(df_hy, f'{loop_name} - PV')
        R_hy_raw = _arr(df_hy, f'{loop_name} - SP')

        fig, _ = plot_response(
            t_hy,
            C_hy_raw,
            R_hy_raw,
            step_time=FIC_STEP_TIME,
            step_info=results[f'si_fic{loop_num}_hy'],
            y_initial=FIC_SP_INITIAL,
            settling_threshold=THRESH_FIC,
            title=f'{loop_name} (HYSYS): Setpoint Tracking',
            ylabel='Transmitter Output (%TO)',
            savefig_path=_out(loop_name, 'HYSYS', f'FIC{loop_num}_HYSYS_Setpoint_PV.png'),
            dpi=DPI,
        )
        plt.close(fig)

        # Python vs HYSYS comparison plot
        C_hy_interp = _interp(t_py, df_hy, f'{loop_name} - PV')
        fig, _ = plot_step_response_multiple(
            t_py,
            {'Python': C_py, 'HYSYS': C_hy_interp},
            R_py,
            step_time=FIC_STEP_TIME,
            step_info_dict={
                'Python': results[f'si_fic{loop_num}_py'],
                'HYSYS': results[f'si_fic{loop_num}_hy'],
            },
            y_initial=FIC_SP_INITIAL,
            settling_threshold=THRESH_FIC,
            title=f'{loop_name}: Python vs HYSYS',
            ylabel='Transmitter Output (%TO)',
            curve_colors={'Python': '#0055cc', 'HYSYS': '#d65a00'},
            savefig_path=_out(loop_name, 'Comparison', f'FIC{loop_num}_Python_vs_HYSYS.png'),
            dpi=DPI,
        )
        plt.close(fig)

        # Python vs HYSYS Setpoint PV/TO comparison for REPORT
        fig_pv_comp, _ = plot_step_response_multiple(
            t_py,
            {'Python': C_py, 'HYSYS': C_hy_interp},
            R_py,
            step_time=FIC_STEP_TIME,
            step_info_dict={
                'Python': results[f'si_fic{loop_num}_py'],
                'HYSYS': results[f'si_fic{loop_num}_hy'],
            },
            y_initial=FIC_SP_INITIAL,
            settling_threshold=THRESH_FIC,
            title=f'{loop_name}: Servo PV Comparison',
            ylabel='Transmitter Output (%TO)',
            curve_colors={'Python': '#0055cc', 'HYSYS': '#d65a00'},
            savefig_path=_out(loop_name, 'report', f'FIC{loop_num}_Servo_TO_Comparison.png'),
            dpi=DPI,
        )
        plt.close(fig_pv_comp)

        # Python vs HYSYS Setpoint CO comparison for REPORT
        M_hy_interp = _interp(t_py, df_hy, f'{loop_name} - OP')
        fig_co_comp, _ = plot_simple_response_multiple(
            t_py,
            {'Python': M_py, 'HYSYS': M_hy_interp},
            title=f'{loop_name}: Servo CO Comparison',
            ylabel='Controller Output (%CO)',
            curve_colors={'Python': '#0055cc', 'HYSYS': '#d65a00'},
            savefig_path=_out(loop_name, 'report', f'FIC{loop_num}_Servo_CO_Comparison.png'),
            dpi=DPI,
        )
        plt.close(fig_co_comp)

    # FIC All Python Multi-Panel Plot
    df_fic100 = _load(os.path.join(DATA_DIR, 'FIC-100', 'Python', 'FIC_100_Setpoint.CSV'))
    df_fic101 = _load(os.path.join(DATA_DIR, 'FIC-101', 'Python', 'FIC_101_Setpoint.CSV'))
    df_fic102 = _load(os.path.join(DATA_DIR, 'FIC-102', 'Python', 'FIC_102_Setpoint.CSV'))

    fic_t = _arr(df_fic100, 'Time')
    fic_loops = {
        'FIC-100': {
            'y': _arr(df_fic100, 'FT-100 - C'),
            'setpoint': _arr(df_fic100, 'FSP-100 - R'),
            'step_info': results['si_fic100_py'],
            'y_initial': FIC_SP_INITIAL,
        },
        'FIC-101': {
            'y': _arr(df_fic101, 'FT-101 - C'),
            'setpoint': _arr(df_fic101, 'FSP-101 - R'),
            'step_info': results['si_fic101_py'],
            'y_initial': FIC_SP_INITIAL,
        },
        'FIC-102': {
            'y': _arr(df_fic102, 'FT-102 - C'),
            'setpoint': _arr(df_fic102, 'FSP-102 - R'),
            'step_info': results['si_fic102_py'],
            'y_initial': FIC_SP_INITIAL,
        },
    }
    fig, _ = compare_step_responses(
        loops_dict=fic_loops,
        time=fic_t,
        step_time=FIC_STEP_TIME,
        cols=3,
        figsize=(20, 6),
        title_prefix='',
        settling_threshold=THRESH_FIC,
        savefig_path=_out('FIC-102', 'Comparison', 'FIC_All_Python_MultiPanel.png'),
        dpi=DPI,
    )
    plt.close(fig)


def _generate_lic_plots(results: dict):
    """Generate plots for level controller (LIC-100)."""
    # Python Setpoint Tracking & Controller Output
    for label, fname in (
        ('Tight', 'LIC_100_Tight_Setpoint.CSV'),
        ('Averaging', 'LIC_100_Averaging_Setpoint.CSV'),
    ):
        df = _load(os.path.join(DATA_DIR, 'LIC-100', 'Python', fname))
        t = _arr(df, 'Time')
        C = _arr(df, 'LT-100 - C')
        R = _arr(df, 'LSP-100 - R')
        M = _arr(df, 'LC-100 - M')

        # Individual PV plot
        fig, _ = plot_response(
            t,
            C,
            R,
            step_time=LIC_SP_STEP_TIME,
            step_info=results['lic_sp_si'][label],
            y_initial=LIC_SP_INITIAL,
            settling_threshold=THRESH_LIC,
            title=f'LIC-100 (Python, {label}): Setpoint Tracking',
            ylabel='Transmitter Output (%TO)',
            savefig_path=_out('LIC-100', 'Python', f'LIC100_Python_{label}_Setpoint_PV.png'),
            dpi=DPI,
        )
        plt.close(fig)

        # Controller output plot
        fig, _ = plot_simple_response(
            t,
            M,
            title=f'LIC-100 (Python, {label}): Controller Output',
            ylabel='Controller Output (%CO)',
            savefig_path=_out('LIC-100', 'Python', f'LIC100_Python_{label}_Setpoint_CO.png'),
            dpi=DPI,
        )
        plt.close(fig)

    # HYSYS Setpoint PV plots
    for label, fname in (
        ('Tight', 'LIC_100_Tight_setpoint.CSV'),
        ('Averaging', 'LIC_100_Averaging_setpoint.CSV'),
    ):
        df = _load(os.path.join(DATA_DIR, 'LIC-100', 'HYSYS', fname))
        t = _arr(df, 'Time')
        C = _arr(df, 'LIC-100 - PV')
        R = _arr(df, 'LIC-100 - SP')

        fig, _ = plot_response(
            t,
            C,
            R,
            step_time=LIC_SP_STEP_TIME,
            step_info=results['lic_sp_hy_si'][label],
            y_initial=LIC_SP_INITIAL,
            settling_threshold=THRESH_LIC,
            title=f'LIC-100 (HYSYS, {label}): Setpoint Tracking',
            ylabel='Transmitter Output (%TO)',
            savefig_path=_out('LIC-100', 'HYSYS', f'LIC100_HYSYS_{label}_Setpoint_PV.png'),
            dpi=DPI,
        )
        plt.close(fig)

    # Python vs HYSYS, per label (Tight / Averaging)
    for label in ('Tight', 'Averaging'):
        df_py = _load(os.path.join(DATA_DIR, 'LIC-100', 'Python', f'LIC_100_{label}_Setpoint.CSV'))
        df_hy = _load(os.path.join(DATA_DIR, 'LIC-100', 'HYSYS', f'LIC_100_{label}_setpoint.CSV'))
        t_ref = _arr(df_py, 'Time')
        C_hy = _interp(t_ref, df_hy, 'LIC-100 - PV')

        fig, _ = plot_step_response_multiple(
            t_ref,
            {'Python': _arr(df_py, 'LT-100 - C'), 'HYSYS': C_hy},
            _arr(df_py, 'LSP-100 - R'),
            step_time=LIC_SP_STEP_TIME,
            step_info_dict={
                'Python': results['lic_sp_si'][label],
                'HYSYS': results['lic_sp_hy_si'][label],
            },
            y_initial=LIC_SP_INITIAL,
            settling_threshold=THRESH_LIC,
            title=f'LIC-100 ({label}): Python vs HYSYS — Setpoint Tracking',
            ylabel='Transmitter Output (%TO)',
            curve_colors={'Python': '#0055cc', 'HYSYS': '#d65a00'},
            savefig_path=_out(
                'LIC-100', 'Comparison', f'LIC100_{label}_Python_vs_HYSYS_Setpoint.png'
            ),
            dpi=DPI,
        )
        plt.close(fig)

    # Tight vs Averaging Setpoint Tracking comparison (Python)
    df_t = _load(os.path.join(DATA_DIR, 'LIC-100', 'Python', 'LIC_100_Tight_Setpoint.CSV'))
    df_a = _load(os.path.join(DATA_DIR, 'LIC-100', 'Python', 'LIC_100_Averaging_Setpoint.CSV'))
    t_ref = _arr(df_t, 'Time')
    C_avg = _interp(t_ref, df_a, 'LT-100 - C')

    fig, _ = plot_step_response_multiple(
        t_ref,
        {'Tight': _arr(df_t, 'LT-100 - C'), 'Averaging': C_avg},
        _arr(df_t, 'LSP-100 - R'),
        step_time=LIC_SP_STEP_TIME,
        step_info_dict=results['lic_sp_si'],
        y_initial=LIC_SP_INITIAL,
        settling_threshold=THRESH_LIC,
        title='LIC-100 (Python): Tight vs Averaging — Setpoint Tracking',
        ylabel='Transmitter Output (%TO)',
        curve_colors={'Tight': '#0055cc', 'Averaging': '#d65a00'},
        savefig_path=_out('LIC-100', 'Comparison', 'LIC100_Python_Tight_vs_Averaging_SP.png'),
        dpi=DPI,
    )
    plt.close(fig)

    # Tight vs Averaging CO comparison
    M_t = _arr(df_t, 'LC-100 - M')
    M_a = _interp(t_ref, df_a, 'LC-100 - M')

    fig, _ = plot_simple_response_multiple(
        t_ref,
        {'Tight': M_t, 'Averaging': M_a},
        title='LIC-100 (Python): Controller Output — Tight vs Averaging',
        ylabel='Controller Output (%CO)',
        curve_colors={'Tight': '#0055cc', 'Averaging': '#d65a00'},
        savefig_path=_out('LIC-100', 'Comparison', 'LIC100_Python_Tight_vs_Averaging_CO.png'),
        dpi=DPI,
    )
    plt.close(fig)

    # LIC-100 Disturbance Plots (Simple Plots) - Python
    for label, fname in (
        ('Tight', 'LIC_100_Tight_Disturbance_Scenario.CSV'),
        ('Averaging', 'LIC_100_Averaging_Disturbance_Scenario.CSV'),
    ):
        df = _load(os.path.join(DATA_DIR, 'LIC-100', 'Python', fname))
        t = _arr(df, 'Time')
        C = _arr(df, 'LT-100 - C')
        R = _arr(df, 'LSP-100 - R')

        fig, _ = plot_simple_response(
            t,
            C,
            u=R,
            title=f'LIC-100 (Python, {label}): Disturbance Rejection',
            ylabel='Transmitter Output (%TO)',
            savefig_path=_out('LIC-100', 'Python', f'LIC100_Python_{label}_Disturbance_PV.png'),
            dpi=DPI,
        )
        plt.close(fig)

    # LIC-100 Disturbance Plots (Simple Plots) - HYSYS
    for label, fname in (
        ('Tight', 'LIC_100_Tight_disturbance_scenario.CSV'),
        ('Averaging', 'LIC_100_Averaging_disturbance_scenario.CSV'),
    ):
        df = _load(os.path.join(DATA_DIR, 'LIC-100', 'HYSYS', fname))
        t = _arr(df, 'Time')
        C = _arr(df, 'LIC-100 - PV')
        R = _arr(df, 'LIC-100 - SP')

        fig, _ = plot_simple_response(
            t,
            C,
            u=R,
            title=f'LIC-100 (HYSYS, {label}): Disturbance Rejection',
            ylabel='Transmitter Output (%TO)',
            savefig_path=_out('LIC-100', 'HYSYS', f'LIC100_HYSYS_{label}_Disturbance_PV.png'),
            dpi=DPI,
        )
        plt.close(fig)

    # LIC-100 Python vs HYSYS Disturbance comparison
    for label in ('Tight', 'Averaging'):
        df_py = _load(
            os.path.join(DATA_DIR, 'LIC-100', 'Python', f'LIC_100_{label}_Disturbance_Scenario.CSV')
        )
        df_hy = _load(
            os.path.join(DATA_DIR, 'LIC-100', 'HYSYS', f'LIC_100_{label}_disturbance_scenario.CSV')
        )
        t_ref = _arr(df_py, 'Time')
        C_hy = _interp(t_ref, df_hy, 'LIC-100 - PV')

        fig, _ = plot_simple_response_multiple(
            t_ref,
            {'Python': _arr(df_py, 'LT-100 - C'), 'HYSYS': C_hy},
            _arr(df_py, 'LSP-100 - R'),
            title=f'LIC-100 ({label}): Python vs HYSYS — Disturbance Rejection',
            ylabel='Transmitter Output (%TO)',
            curve_colors={'Python': '#0055cc', 'HYSYS': '#d65a00'},
            savefig_path=_out(
                'LIC-100', 'Comparison', f'LIC100_{label}_Python_vs_HYSYS_Disturbance.png'
            ),
            dpi=DPI,
        )
        plt.close(fig)

    # Tight vs Averaging Disturbance comparison (Python)
    df_td = _load(
        os.path.join(DATA_DIR, 'LIC-100', 'Python', 'LIC_100_Tight_Disturbance_Scenario.CSV')
    )
    df_ad = _load(
        os.path.join(DATA_DIR, 'LIC-100', 'Python', 'LIC_100_Averaging_Disturbance_Scenario.CSV')
    )
    t_ref = _arr(df_td, 'Time')
    C_avd = _interp(t_ref, df_ad, 'LT-100 - C')

    fig, _ = plot_simple_response_multiple(
        t_ref,
        {'Tight': _arr(df_td, 'LT-100 - C'), 'Averaging': C_avd},
        _arr(df_td, 'LSP-100 - R'),
        title='LIC-100 (Python): Tight vs Averaging — Disturbance Rejection',
        ylabel='Transmitter Output (%TO)',
        curve_colors={'Tight': '#0055cc', 'Averaging': '#d65a00'},
        savefig_path=_out('LIC-100', 'Comparison', 'LIC100_Python_Tight_vs_Averaging_Dist.png'),
        dpi=DPI,
    )
    plt.close(fig)

    # =========================================================================
    # REPORT PLOTS (Python only - Tight vs Averaging Comparison)
    # =========================================================================
    df_py_t_sp = _load(os.path.join(DATA_DIR, 'LIC-100', 'Python', 'LIC_100_Tight_Setpoint.CSV'))
    t_t_sp = _arr(df_py_t_sp, 'Time')
    df_py_t_dist = _load(
        os.path.join(DATA_DIR, 'LIC-100', 'Python', 'LIC_100_Tight_Disturbance_Scenario.CSV')
    )
    t_t_dist = _arr(df_py_t_dist, 'Time')
    df_py_a_sp = _load(
        os.path.join(DATA_DIR, 'LIC-100', 'Python', 'LIC_100_Averaging_Setpoint.CSV')
    )
    df_py_a_dist = _load(
        os.path.join(DATA_DIR, 'LIC-100', 'Python', 'LIC_100_Averaging_Disturbance_Scenario.CSV')
    )

    # --- OPTION B: Comparison plots (Tight vs Averaging in Python) ---
    # 1. Servo PV Comparison (Tight vs Averaging)
    fig_comp_sp_pv, _ = plot_step_response_multiple(
        t_t_sp,
        {
            'Tight': _arr(df_py_t_sp, 'LT-100 - C'),
            'Averaging': _interp(t_t_sp, df_py_a_sp, 'LT-100 - C'),
        },
        _arr(df_py_t_sp, 'LSP-100 - R'),
        step_time=LIC_SP_STEP_TIME,
        step_info_dict=results['lic_sp_si'],
        y_initial=LIC_SP_INITIAL,
        settling_threshold=THRESH_LIC,
        title='LIC-100 (Python): Servo PV Comparison',
        ylabel='Transmitter Output (%TO)',
        curve_colors={'Tight': '#0055cc', 'Averaging': '#d65a00'},
        savefig_path=_out('LIC-100', 'report', 'LIC100_Servo_PV_Comparison.png'),
        dpi=DPI,
    )
    plt.close(fig_comp_sp_pv)

    # 2. Servo CO Comparison (Tight vs Averaging)
    fig_comp_sp_co, _ = plot_simple_response_multiple(
        t_t_sp,
        {
            'Tight': _arr(df_py_t_sp, 'LC-100 - M'),
            'Averaging': _interp(t_t_sp, df_py_a_sp, 'LC-100 - M'),
        },
        title='LIC-100 (Python): Servo CO Comparison',
        ylabel='Controller Output (%CO)',
        curve_colors={'Tight': '#0055cc', 'Averaging': '#d65a00'},
        savefig_path=_out('LIC-100', 'report', 'LIC100_Servo_CO_Comparison.png'),
        dpi=DPI,
    )
    plt.close(fig_comp_sp_co)

    # 3. Regulatory PV Comparison (Tight vs Averaging)
    fig_comp_dist_pv, _ = plot_simple_response_multiple(
        t_t_dist,
        {
            'Tight': _arr(df_py_t_dist, 'LT-100 - C'),
            'Averaging': _interp(t_t_dist, df_py_a_dist, 'LT-100 - C'),
        },
        _arr(df_py_t_dist, 'LSP-100 - R'),
        title='LIC-100 (Python): Regulatory PV Comparison',
        ylabel='Transmitter Output (%TO)',
        curve_colors={'Tight': '#0055cc', 'Averaging': '#d65a00'},
        savefig_path=_out('LIC-100', 'report', 'LIC100_Regulatory_PV_Comparison.png'),
        dpi=DPI,
    )
    plt.close(fig_comp_dist_pv)

    # 4. Regulatory CO Comparison (Tight vs Averaging)
    fig_comp_dist_co, _ = plot_simple_response_multiple(
        t_t_dist,
        {
            'Tight': _arr(df_py_t_dist, 'LC-100 - M'),
            'Averaging': _interp(t_t_dist, df_py_a_dist, 'LC-100 - M'),
        },
        title='LIC-100 (Python): Regulatory CO Comparison',
        ylabel='Controller Output (%CO)',
        curve_colors={'Tight': '#0055cc', 'Averaging': '#d65a00'},
        savefig_path=_out('LIC-100', 'report', 'LIC100_Regulatory_CO_Comparison.png'),
        dpi=DPI,
    )
    plt.close(fig_comp_dist_co)

    # --- Python vs HYSYS Comparison plots for report ---
    for label in ('Tight', 'Averaging'):
        # Load Setpoint (Servo) Data
        df_py_sp = _load(
            os.path.join(DATA_DIR, 'LIC-100', 'Python', f'LIC_100_{label}_Setpoint.CSV')
        )
        df_hy_sp = _load(
            os.path.join(DATA_DIR, 'LIC-100', 'HYSYS', f'LIC_100_{label}_setpoint.CSV')
        )
        t_sp = _arr(df_py_sp, 'Time')
        C_hy_sp = _interp(t_sp, df_hy_sp, 'LIC-100 - PV')
        M_hy_sp = _interp(t_sp, df_hy_sp, 'LIC-100 - OP')

        # 1. Servo PV/TO Comparison (Python vs HYSYS)
        fig_pv, _ = plot_step_response_multiple(
            t_sp,
            {'Python': _arr(df_py_sp, 'LT-100 - C'), 'HYSYS': C_hy_sp},
            _arr(df_py_sp, 'LSP-100 - R'),
            step_time=LIC_SP_STEP_TIME,
            step_info_dict={
                'Python': results['lic_sp_si'][label],
                'HYSYS': results['lic_sp_hy_si'][label],
            },
            y_initial=LIC_SP_INITIAL,
            settling_threshold=THRESH_LIC,
            title=f'LIC-100 ({label}): Servo PV Comparison',
            ylabel='Transmitter Output (%TO)',
            curve_colors={'Python': '#0055cc', 'HYSYS': '#d65a00'},
            savefig_path=_out('LIC-100', 'report', f'LIC100_{label}_Servo_TO_Comparison.png'),
            dpi=DPI,
        )
        plt.close(fig_pv)

        # 2. Servo CO Comparison (Python vs HYSYS)
        fig_co, _ = plot_simple_response_multiple(
            t_sp,
            {'Python': _arr(df_py_sp, 'LC-100 - M'), 'HYSYS': M_hy_sp},
            title=f'LIC-100 ({label}): Servo CO Comparison',
            ylabel='Controller Output (%CO)',
            curve_colors={'Python': '#0055cc', 'HYSYS': '#d65a00'},
            savefig_path=_out('LIC-100', 'report', f'LIC100_{label}_Servo_CO_Comparison.png'),
            dpi=DPI,
        )
        plt.close(fig_co)

        # Load Disturbance (Regulatory) Data
        df_py_dist = _load(
            os.path.join(DATA_DIR, 'LIC-100', 'Python', f'LIC_100_{label}_Disturbance_Scenario.CSV')
        )
        df_hy_dist = _load(
            os.path.join(DATA_DIR, 'LIC-100', 'HYSYS', f'LIC_100_{label}_disturbance_scenario.CSV')
        )
        t_dist = _arr(df_py_dist, 'Time')
        C_hy_dist = _interp(t_dist, df_hy_dist, 'LIC-100 - PV')
        M_hy_dist = _interp(t_dist, df_hy_dist, 'LIC-100 - OP')

        # 3. Regulatory PV/TO Comparison (Python vs HYSYS)
        fig_pv_dist, _ = plot_simple_response_multiple(
            t_dist,
            {'Python': _arr(df_py_dist, 'LT-100 - C'), 'HYSYS': C_hy_dist},
            _arr(df_py_dist, 'LSP-100 - R'),
            title=f'LIC-100 ({label}): Regulatory PV Comparison',
            ylabel='Transmitter Output (%TO)',
            curve_colors={'Python': '#0055cc', 'HYSYS': '#d65a00'},
            savefig_path=_out('LIC-100', 'report', f'LIC100_{label}_Regulatory_TO_Comparison.png'),
            dpi=DPI,
        )
        plt.close(fig_pv_dist)

        # 4. Regulatory CO Comparison (Python vs HYSYS)
        fig_co_dist, _ = plot_simple_response_multiple(
            t_dist,
            {'Python': _arr(df_py_dist, 'LC-100 - M'), 'HYSYS': M_hy_dist},
            title=f'LIC-100 ({label}): Regulatory CO Comparison',
            ylabel='Controller Output (%CO)',
            curve_colors={'Python': '#0055cc', 'HYSYS': '#d65a00'},
            savefig_path=_out('LIC-100', 'report', f'LIC100_{label}_Regulatory_CO_Comparison.png'),
            dpi=DPI,
        )
        plt.close(fig_co_dist)


def _generate_tic_plots(results: dict):
    """Generate plots for temperature controller (TIC-100)."""
    # HYSYS Setpoint PV setup placeholder
    y_hy_sp = {}
    t_hy_ref = None
    u_hy_ref = None

    # Python Setpoint tuning (Syn / QDR / IAE)
    for tuning in ('Syn', 'QDR', 'IAE'):
        df = _load(
            os.path.join(DATA_DIR, 'TIC-100', 'Python', 'setpoint-tuning', f'TIC_100_{tuning}.CSV')
        )
        t, C, R = (_arr(df, c) for c in ('Time', 'TT-100 - C', 'TSP-100 - R'))

        si_sp = results['tic_sp_py_si'][tuning]
        si_dist_dual = results['si_nested'][tuning]['Disturbance']

        fig, _ = plot_dual_response(
            t,
            {tuning: C},
            R,
            step1_time=TIC_DIST_STEP_TIME,
            step1_end_time=TIC_DIST_END_TIME,
            step2_time=TIC_SP_STEP_TIME,
            step_info_dict={tuning: {'Disturbance': si_dist_dual, 'Setpoint': si_sp}},
            step1_threshold=THRESH_TIC_DIST,
            step2_threshold=THRESH_TIC,
            title=f'TIC-100 (Python, {tuning}): Disturbance Rejection & Setpoint Tracking',
            ylabel='Transmitter Output (%TO)',
            curve_colors={tuning: COLORS[tuning]},
            savefig_path=_out('TIC-100', 'Python', f'TIC100_Python_{tuning}_DualStep.png'),
            dpi=DPI,
        )
        plt.close(fig)

    # Controller output comparison — Setpoint Step
    t_ref = _arr(
        _load(os.path.join(DATA_DIR, 'TIC-100', 'Python', 'setpoint-tuning', 'TIC_100_QDR.CSV')),
        'Time',
    )
    M_dict_sp = {
        k: _arr(
            _load(
                os.path.join(DATA_DIR, 'TIC-100', 'Python', 'setpoint-tuning', f'TIC_100_{k}.CSV')
            ),
            'TC-100 - M',
        )
        for k in ('Syn', 'QDR', 'IAE')
    }
    fig, _ = plot_simple_response_multiple(
        t_ref,
        M_dict_sp,
        title='TIC-100 (Python): Controller Output — Setpoint Step',
        ylabel='Controller Output (%CO)',
        curve_colors=COLORS,
        savefig_path=_out('TIC-100', 'Python', 'TIC100_Python_SP_CO_Comparison.png'),
        dpi=DPI,
    )
    plt.close(fig)

    # Controller output comparison — Disturbance Step
    t_dist_ref = _arr(
        _load(os.path.join(DATA_DIR, 'TIC-100', 'Python', 'disturbance-tuning', 'TIC_100_QDR.CSV')),
        'Time',
    )
    M_dict_dist = {
        k: _arr(
            _load(
                os.path.join(
                    DATA_DIR, 'TIC-100', 'Python', 'disturbance-tuning', f'TIC_100_{k}.CSV'
                )
            ),
            'TC-100 - M',
        )
        for k in ('Syn', 'QDR', 'IAE')
    }
    fig, _ = plot_simple_response_multiple(
        t_dist_ref,
        M_dict_dist,
        title='TIC-100 (Python): Controller Output — Disturbance Step',
        ylabel='Controller Output (%CO)',
        curve_colors=COLORS,
        savefig_path=_out('TIC-100', 'Python', 'TIC100_Python_Dist_CO_Comparison.png'),
        dpi=DPI,
    )
    plt.close(fig)

    # Python Dual-Step comparison (all tunings)
    df_ref = _load(
        os.path.join(DATA_DIR, 'TIC-100', 'Python', 'setpoint-tuning', 'TIC_100_QDR.CSV')
    )
    t_dual = _arr(df_ref, 'Time')
    u_dual = _arr(df_ref, 'TSP-100 - R')
    y_dual = {
        k: _arr(
            _load(
                os.path.join(DATA_DIR, 'TIC-100', 'Python', 'setpoint-tuning', f'TIC_100_{k}.CSV')
            ),
            'TT-100 - C',
        )
        for k in ('Syn', 'QDR', 'IAE')
    }
    fig, _ = plot_dual_response(
        t_dual,
        y_dual,
        u_dual,
        step1_time=TIC_DIST_STEP_TIME,
        step1_end_time=TIC_DIST_END_TIME,
        step2_time=TIC_SP_STEP_TIME,
        step_info_dict=results['si_nested'],
        step1_threshold=THRESH_TIC_DIST,
        step2_threshold=THRESH_TIC,
        title='TIC-100 (Python): Dual-Step — Disturbance & Setpoint',
        ylabel='Transmitter Output (%TO)',
        curve_colors=COLORS,
        savefig_path=_out('TIC-100', 'Comparison', 'TIC100_Python_DualStep_Comparison.png'),
        dpi=DPI,
    )
    plt.close(fig)

    # HYSYS Setpoint tuning (Syn / QDR / IAE)
    for tuning in ('Syn', 'QDR', 'IAE'):
        df = _load(
            os.path.join(DATA_DIR, 'TIC-100', 'HYSYS', 'setpoint-tuning', f'TIC_100_{tuning}.CSV')
        )
        t, C, R = (_arr(df, c) for c in ('Time', 'TIC-100 - PV', 'TIC-100 - SP'))

        si_sp = results['tic_sp_hy_si'][tuning]
        si_dist_hy_dual = results['si_nested_hy'][tuning]['Disturbance']
        y_hy_sp[tuning] = C
        if t_hy_ref is None:
            t_hy_ref = t
            u_hy_ref = R

        fig, _ = plot_dual_response(
            t,
            {tuning: C},
            R,
            step1_time=TIC_DIST_STEP_TIME,
            step1_end_time=TIC_DIST_END_TIME,
            step2_time=TIC_SP_STEP_TIME,
            step_info_dict={tuning: {'Disturbance': si_dist_hy_dual, 'Setpoint': si_sp}},
            step1_threshold=THRESH_TIC_DIST,
            step2_threshold=THRESH_TIC,
            title=f'TIC-100 (HYSYS, {tuning}): Disturbance Rejection & Setpoint Tracking',
            ylabel='Transmitter Output (%TO)',
            curve_colors={tuning: COLORS[tuning]},
            savefig_path=_out('TIC-100', 'HYSYS', f'TIC100_HYSYS_{tuning}_DualStep.png'),
            dpi=DPI,
        )
        plt.close(fig)

    # HYSYS Dual-Step comparison (all tunings)
    fig, _ = plot_dual_response(
        t_hy_ref,
        y_hy_sp,
        u_hy_ref,
        step1_time=TIC_DIST_STEP_TIME,
        step1_end_time=TIC_DIST_END_TIME,
        step2_time=TIC_SP_STEP_TIME,
        step_info_dict=results['si_nested_hy'],
        step1_threshold=THRESH_TIC_DIST,
        step2_threshold=THRESH_TIC,
        title='TIC-100 (HYSYS): Dual-Step — Disturbance & Setpoint',
        ylabel='Transmitter Output (%TO)',
        curve_colors=COLORS,
        savefig_path=_out('TIC-100', 'Comparison', 'TIC100_HYSYS_DualStep_Comparison.png'),
        dpi=DPI,
    )
    plt.close(fig)

    # Python vs HYSYS (QDR), both steps
    df_py_q = _load(
        os.path.join(DATA_DIR, 'TIC-100', 'Python', 'setpoint-tuning', 'TIC_100_QDR.CSV')
    )
    df_hy_q = _load(
        os.path.join(DATA_DIR, 'TIC-100', 'HYSYS', 'setpoint-tuning', 'TIC_100_QDR.CSV')
    )
    t_ref = _arr(df_py_q, 'Time')
    C_hy = _interp(t_ref, df_hy_q, 'TIC-100 - PV')

    fig, _ = plot_dual_response(
        t_ref,
        {'Python': _arr(df_py_q, 'TT-100 - C'), 'HYSYS': C_hy},
        _arr(df_py_q, 'TSP-100 - R'),
        step1_time=TIC_DIST_STEP_TIME,
        step1_end_time=TIC_DIST_END_TIME,
        step2_time=TIC_SP_STEP_TIME,
        step_info_dict={
            'Python': results['si_nested']['QDR'],
            'HYSYS': results['si_nested_hy']['QDR'],
        },
        step1_threshold=THRESH_TIC_DIST,
        step2_threshold=THRESH_TIC,
        title='TIC-100 QDR: Python vs HYSYS — Disturbance & Setpoint',
        ylabel='Transmitter Output (%TO)',
        curve_colors={'Python': '#0055cc', 'HYSYS': '#d65a00'},
        savefig_path=_out('TIC-100', 'Comparison', 'TIC100_QDR_Python_vs_HYSYS.png'),
        dpi=DPI,
    )
    plt.close(fig)

    # Python vs HYSYS (Syn and IAE)
    for tuning in ('Syn', 'IAE'):
        df_py_t = _load(
            os.path.join(DATA_DIR, 'TIC-100', 'Python', 'setpoint-tuning', f'TIC_100_{tuning}.CSV')
        )
        df_hy_t = _load(
            os.path.join(DATA_DIR, 'TIC-100', 'HYSYS', 'setpoint-tuning', f'TIC_100_{tuning}.CSV')
        )
        t_ref = _arr(df_py_t, 'Time')
        C_hy = _interp(t_ref, df_hy_t, 'TIC-100 - PV')

        fig, _ = plot_dual_response(
            t_ref,
            {'Python': _arr(df_py_t, 'TT-100 - C'), 'HYSYS': C_hy},
            _arr(df_py_t, 'TSP-100 - R'),
            step1_time=TIC_DIST_STEP_TIME,
            step1_end_time=TIC_DIST_END_TIME,
            step2_time=TIC_SP_STEP_TIME,
            step_info_dict={
                'Python': results['si_nested'][tuning],
                'HYSYS': results['si_nested_hy'][tuning],
            },
            step1_threshold=THRESH_TIC_DIST,
            step2_threshold=THRESH_TIC,
            title=f'TIC-100 {tuning}: Python vs HYSYS — Disturbance & Setpoint',
            ylabel='Transmitter Output (%TO)',
            curve_colors={'Python': '#0055cc', 'HYSYS': '#d65a00'},
            savefig_path=_out('TIC-100', 'Comparison', f'TIC100_{tuning}_Python_vs_HYSYS.png'),
            dpi=DPI,
        )
        plt.close(fig)

    # Python vs HYSYS, disturbance-tuning (Syn / QDR / IAE)
    for tuning in ('Syn', 'QDR', 'IAE'):
        df_py_d = _load(
            os.path.join(
                DATA_DIR, 'TIC-100', 'Python', 'disturbance-tuning', f'TIC_100_{tuning}.CSV'
            )
        )
        df_hy_d = _load(
            os.path.join(
                DATA_DIR, 'TIC-100', 'HYSYS', 'disturbance-tuning', f'TIC_100_{tuning}.CSV'
            )
        )
        t_ref = _arr(df_py_d, 'Time')
        C_hy = _interp(t_ref, df_hy_d, 'TIC-100 - PV')

        si_nested_dist = {
            'Python': {
                'Disturbance': results['tic_dist_py_si'][tuning],
                'Setpoint': results['tic_dist_py_sp_si'][tuning],
            },
            'HYSYS': {
                'Disturbance': results['tic_dist_hy_si'][tuning],
                'Setpoint': results['tic_dist_hy_sp_si'][tuning],
            },
        }
        fig, _ = plot_dual_response(
            t_ref,
            {'Python': _arr(df_py_d, 'TT-100 - C'), 'HYSYS': C_hy},
            _arr(df_py_d, 'TSP-100 - R'),
            step1_time=TIC_DIST_STEP_TIME,
            step1_end_time=TIC_DIST_END_TIME,
            step2_time=TIC_SP_STEP_TIME,
            step_info_dict=si_nested_dist,
            step1_threshold=THRESH_TIC_DIST,
            step2_threshold=THRESH_TIC,
            title=f'TIC-100 ({tuning}): Python vs HYSYS — Disturbance & Setpoint',
            ylabel='Transmitter Output (%TO)',
            curve_colors={'Python': '#0055cc', 'HYSYS': '#d65a00'},
            savefig_path=_out(
                'TIC-100', 'Comparison', f'TIC100_{tuning}_Python_vs_HYSYS_Disturbance.png'
            ),
            dpi=DPI,
        )
        plt.close(fig)

    # =========================================================================
    # REPORT PLOTS (Disturbance-tuning / Formula Input Disturbance only)
    # =========================================================================
    # 1. Transmitter Output (PV) Comparison — Disturbance Tuning
    df_ref_d = _load(
        os.path.join(DATA_DIR, 'TIC-100', 'Python', 'disturbance-tuning', 'TIC_100_QDR.CSV')
    )
    t_dual_d = _arr(df_ref_d, 'Time')
    u_dual_d = _arr(df_ref_d, 'TSP-100 - R')
    y_dual_d = {
        k: _arr(
            _load(
                os.path.join(
                    DATA_DIR, 'TIC-100', 'Python', 'disturbance-tuning', f'TIC_100_{k}.CSV'
                )
            ),
            'TT-100 - C',
        )
        for k in ('Syn', 'QDR', 'IAE')
    }
    si_nested_dist_py = {
        k: {
            'Disturbance': results['tic_dist_py_si'][k],
            'Setpoint': results['tic_dist_py_sp_si'][k],
        }
        for k in ('Syn', 'QDR', 'IAE')
    }
    fig_pv_rep, _ = plot_dual_response(
        t_dual_d,
        y_dual_d,
        u_dual_d,
        step1_time=TIC_DIST_STEP_TIME,
        step1_end_time=TIC_DIST_END_TIME,
        step2_time=TIC_SP_STEP_TIME,
        step_info_dict=si_nested_dist_py,
        step1_threshold=THRESH_TIC_DIST,
        step2_threshold=THRESH_TIC,
        title='TIC-100 (Python): PV Comparison (Disturbance Tuning)',
        ylabel='Transmitter Output (%TO)',
        curve_colors=COLORS,
        savefig_path=_out('TIC-100', 'report', 'TIC100_Python_PV_Comparison.png'),
        dpi=DPI,
    )
    plt.close(fig_pv_rep)

    # 2. Controller Output (CO) Comparison — Disturbance Tuning
    M_dict_dist_d = {
        k: _arr(
            _load(
                os.path.join(
                    DATA_DIR, 'TIC-100', 'Python', 'disturbance-tuning', f'TIC_100_{k}.CSV'
                )
            ),
            'TC-100 - M',
        )
        for k in ('Syn', 'QDR', 'IAE')
    }
    fig_co_rep, _ = plot_simple_response_multiple(
        t_dual_d,
        M_dict_dist_d,
        title='TIC-100 (Python): CO Comparison (Disturbance Tuning)',
        ylabel='Controller Output (%CO)',
        curve_colors=COLORS,
        savefig_path=_out('TIC-100', 'report', 'TIC100_Python_CO_Comparison.png'),
        dpi=DPI,
    )
    plt.close(fig_co_rep)

    # 3. Transmitter Output (PV / TO) Comparison: Python vs HYSYS — Disturbance Tuning (QDR Only)
    df_py = _load(
        os.path.join(DATA_DIR, 'TIC-100', 'Python', 'disturbance-tuning', 'TIC_100_QDR.CSV')
    )
    df_hy = _load(
        os.path.join(DATA_DIR, 'TIC-100', 'HYSYS', 'disturbance-tuning', 'TIC_100_QDR.CSV')
    )
    t_py = _arr(df_py, 'Time')
    C_hy = _interp(t_py, df_hy, 'TIC-100 - PV')

    si_nested_dist_qdr = {
        'Python': {
            'Disturbance': results['tic_dist_py_si']['QDR'],
            'Setpoint': results['tic_dist_py_sp_si']['QDR'],
        },
        'HYSYS': {
            'Disturbance': results['tic_dist_hy_si']['QDR'],
            'Setpoint': results['tic_dist_hy_sp_si']['QDR'],
        },
    }

    fig_pv_comp, _ = plot_dual_response(
        t_py,
        {'Python': _arr(df_py, 'TT-100 - C'), 'HYSYS': C_hy},
        _arr(df_py, 'TSP-100 - R'),
        step1_time=TIC_DIST_STEP_TIME,
        step1_end_time=TIC_DIST_END_TIME,
        step2_time=TIC_SP_STEP_TIME,
        step_info_dict=si_nested_dist_qdr,
        step1_threshold=THRESH_TIC_DIST,
        step2_threshold=THRESH_TIC,
        title='TIC-100 QDR: Python vs HYSYS — Disturbance & Setpoint',
        ylabel='Transmitter Output (%TO)',
        curve_colors={'Python': '#0055cc', 'HYSYS': '#d65a00'},
        savefig_path=_out('TIC-100', 'report', 'TIC100_TO_Comparison_Python_vs_HYSYS.png'),
        dpi=DPI,
    )
    plt.close(fig_pv_comp)

    # 4. Controller Output (CO) Comparison: Python vs HYSYS — Disturbance Tuning (QDR Only)
    fig_co_comp, ax = plt.subplots(figsize=(14, 7))
    df_py = _load(
        os.path.join(DATA_DIR, 'TIC-100', 'Python', 'disturbance-tuning', 'TIC_100_QDR.CSV')
    )
    df_hy = _load(
        os.path.join(DATA_DIR, 'TIC-100', 'HYSYS', 'disturbance-tuning', 'TIC_100_QDR.CSV')
    )
    t_py = _arr(df_py, 'Time')
    op_py = _arr(df_py, 'TC-100 - M')
    op_hy = 100.0 - _interp(
        t_py, df_hy, 'TIC-100 - OP'
    )  # Invert HYSYS to match Python's 80% convention

    ax.plot(t_py, op_py, label='Python QDR', color='#0055cc', linestyle='-', linewidth=2.0)
    ax.plot(t_py, op_hy, label='HYSYS QDR', color='#d65a00', linestyle='--', linewidth=1.5)

    ax.set_xlabel('Time (s)', fontsize=12, fontweight='bold', labelpad=8)
    ax.set_ylabel('Controller Output (%CO)', fontsize=12, fontweight='bold')
    ax.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, -0.09),
        ncol=2,
        fontsize=10,
        frameon=True,
        framealpha=0.95,
        edgecolor='gray',
        fancybox=True,
    )
    ax.grid(True, alpha=0.3)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)
        spine.set_color('#1a1a1a')
    ax.tick_params(axis='both', which='major', labelsize=10)
    fig_co_comp.tight_layout()
    _save_fig(
        fig_co_comp, _out('TIC-100', 'report', 'TIC100_CO_Comparison_Python_vs_HYSYS.png'), DPI
    )
    plt.close(fig_co_comp)


def generate_all_plots(results: dict):
    """Main plotting orchestrator."""
    # Setup publication style
    setup_publication_style(
        font_family=FONT_FAMILY,
        font_size=FONT_SIZE,
        font_serif=FONT_SERIF,
    )

    # Generate Loop Specific Plots
    _generate_fic_plots(results)
    _generate_lic_plots(results)
    _generate_tic_plots(results)


if __name__ == '__main__':
    # First, run the calculations to get the metrics
    results = calculate_all_performance()

    print('\n' + '=' * 62)
    print('  Generating Plots...')
    print('=' * 62)

    # Generate all plots
    generate_all_plots(results)
    print(f'Plots saved    ->  {OUTPUT_DIR}/')
