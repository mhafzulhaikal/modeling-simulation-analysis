"""
Equilibrium operating point finder and verification for the Biodiesel CSTR.

This module utilizes the general `OperatingPoint` class from `model`
to locate and verify the steady-state operating point (x*, u*) such
that dx/dt = 0.
"""

import os
import sys
from pathlib import Path

# Add workspace directory to path for imports
sys.path.append(str(Path(__file__).resolve().parent.parent))

import control as ct
import numpy as np

from model import BiodieselReactorSystem
from model.config import PROCESS_PARAMS
from model.find_eqpt import OperatingPoint


class BiodieselOperatingPoint(OperatingPoint):
    """
    Concrete equilibrium operating point finder for the Biodiesel CSTR.

    Parameters
    ----------
    plant : control.NonlinearIOSystem
        The nonlinear input/output system representing the biodiesel plant.
    """

    def __init__(self, plant: ct.NonlinearIOSystem) -> None:
        super().__init__(plant)

    def get_initial_guess(self) -> np.ndarray:
        """
        Get the initial guess for the system states.

        Returns
        -------
        numpy.ndarray
            A 1D array of initial guesses for the plant states.
        """
        return np.array(
            [
                1.5,  # h          [m]
                0.0142,  # c_TG       [kmol/m³]
                2.3760,  # c_MeOH     [kmol/m³]
                2.0250,  # c_ME       [kmol/m³]
                0.0055,  # c_DG       [kmol/m³]
                0.0292,  # c_MG       [kmol/m³]
                0.6537,  # c_Gly      [kmol/m³]
                0.1481,  # c_Cat      [kmol/m³]
                1.3156,  # c_Water    [kmol/m³]
                333.15,  # T          [K]
                323.15,  # T_coolant  [K]
            ]
        )

    def get_nominal_inputs(self) -> np.ndarray:
        """
        Get the nominal steady-state inputs for the system.

        Returns
        -------
        numpy.ndarray
            A 1D array of nominal input values (disturbances and flows).
        """
        # --- Feed temperatures [K] ---
        T_oil = 333.15
        T_MeOH = 298.15
        T_NaOH = 298.15
        T_coolant_in = 298.15

        # --- Feed concentrations [kmol/m³] ---
        c_TG_in = 0.9992
        c_MeOH_in = 24.7462
        c_Cat_in = 5.2060
        c_Water_in = 46.2325

        # --- Feed stream mass flow rates [kg/s] ---
        m_oil = 0.291666666655013  # Oil feed mass flow rate        [kg/s]
        m_MeOH = 6.61111111082406e-002  # Methanol feed mass flow rate   [kg/s]
        m_NaOH = 1.38888888883481e-002  # NaOH feed mass flow rate       [kg/s]

        # --- Flow rates [m³/s] ---
        f_oil = m_oil / PROCESS_PARAMS['rho_oil']
        f_MeOH = m_MeOH / PROCESS_PARAMS['rho_MeOH']
        f_NaOH = m_NaOH / PROCESS_PARAMS['rho_NaOH']
        f_FAME = 4.62092118065036e-004  # initial guess for FAME outflow from HYSYS
        f_coolant = 0.1787813786090115  # initial guess for coolant flow rate from HYSYS

        return np.array(
            [
                c_TG_in,
                T_oil,
                c_MeOH_in,
                T_MeOH,
                c_Cat_in,
                c_Water_in,
                T_NaOH,
                T_coolant_in,
                f_oil,
                f_MeOH,
                f_NaOH,
                f_FAME,
                f_coolant,
            ]
        )

    def get_target_outputs(self) -> np.ndarray:
        """
        Get the target steady-state outputs for the system.

        Returns
        -------
        numpy.ndarray
            A 1D array of target output values, where constrained values are
            specified and others are treated as free variables.
        """
        h_steady = 1.5
        T_steady = 333.15
        return np.array(
            [
                h_steady,  # h          — CONSTRAINED
                0.0142,  # c_TG       — free
                2.3760,  # c_MeOH     — free
                2.0250,  # c_ME       — free
                0.0055,  # c_DG       — free
                0.0292,  # c_MG       — free
                0.6537,  # c_Gly      — free
                0.1481,  # c_Cat      — free
                1.3156,  # c_Water    — free
                T_steady,  # T          — CONSTRAINED
                323.15,  # T_coolant  — free
            ]
        )

    def get_input_indices(self) -> list[int]:
        """
        Get the indices of the fixed inputs in the input vector.

        Returns
        -------
        list of int
            A list of integers representing indices of inputs that are fixed
            during the equilibrium search.
        """
        # Fixed inputs: indices 0-10 (8 disturbances + f_oil, f_MeOH, f_NaOH)
        return list(range(11))

    def get_output_indices(self) -> list[int]:
        """
        Get the indices of the constrained outputs in the output vector.

        Returns
        -------
        list of int
            A list of integers representing indices of outputs that are
            constrained during the equilibrium search.
        """
        # Constrained outputs: h (idx 0) and T (idx 9)
        return [0, 9]


if __name__ == '__main__':
    print('Running equilibrium finder for Biodiesel Reactor CSTR...')
    plant = BiodieselReactorSystem(**PROCESS_PARAMS)
    finder = BiodieselOperatingPoint(plant.system)

    result = finder.find()

    output_path = os.path.join(
        os.path.dirname(__file__), '..', 'outputs', 'reports', 'find_eqpt_output.txt'
    )

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if result is not None:
        x_eqpt, u_eqpt = result
        x_guess = finder.get_initial_guess()
        u_steady = finder.get_nominal_inputs()

        state_labels = plant.system.state_labels
        input_labels = plant.system.input_labels

        lines = []
        lines.append('=' * 80)
        lines.append('  EQUILIBRIUM STATES (x*)')
        lines.append('=' * 80)
        for name, val, guess in zip(state_labels, x_eqpt, x_guess, strict=False):
            delta = val - guess
            lines.append(
                f'  {name:>12s} = {val:>14.6f}   (guess: {guess:>12.6f}, delta = {delta:>+12.6f})'
            )

        lines.append('')
        lines.append('=' * 80)
        lines.append('  EQUILIBRIUM INPUTS (u*)')
        lines.append('=' * 80)
        for name, val, orig in zip(input_labels, u_eqpt, u_steady, strict=False):
            marker = ' *FREE*' if name in ('f_FAME', 'f_coolant') else ''
            lines.append(f'  {name:>14s} = {val:>14.6e}   (initial: {orig:>14.6e}){marker}')

        # Mass balance verification
        p = plant.params
        Ar = p['Ar']
        rho = p['rho']
        rho_oil = p['rho_oil']
        rho_MeOH = p['rho_MeOH']
        rho_NaOH = p['rho_NaOH']

        inflow_mass = u_eqpt[8] * rho_oil + u_eqpt[9] * rho_MeOH + u_eqpt[10] * rho_NaOH
        outflow_mass = u_eqpt[11] * rho
        dh_dt = (inflow_mass - outflow_mass) / (rho * Ar)

        lines.append('')
        lines.append('=' * 80)
        lines.append('  MASS BALANCE VERIFICATION')
        lines.append('=' * 80)
        lines.append(f'  Inflow mass rate  = {inflow_mass:.6e} kg/s')
        lines.append(f'  Outflow mass rate = {outflow_mass:.6e} kg/s')
        lines.append(f'  dh/dt             = {dh_dt:.6e} m/s   (should be ~0)')

        # Evaluate derivatives at equilibrium
        dx = plant.system.updfcn(0, x_eqpt, u_eqpt, plant.params)
        dx = np.asarray(dx)
        lines.append('')
        lines.append('=' * 80)
        lines.append('  STATE DERIVATIVES AT EQUILIBRIUM (dx/dt)')
        lines.append('=' * 80)
        for name, d in zip(state_labels, dx, strict=False):
            status = 'OK' if abs(d) < 1e-6 else '!!'
            deriv_label = f'd({name})/dt'
            lines.append(f'  {status} {deriv_label:<18} = {d:>+14.6e}')
        lines.append(f'\n  Max |dx/dt| = {np.max(np.abs(dx)):.2e}')
        lines.append(f'  Min |dx/dt| = {np.min(np.abs(dx)):.2e}')

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines) + '\n')
        print(f'Results saved to: {os.path.abspath(output_path)}')
    else:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('FAILED to find equilibrium point.\n')
        print(f'FAILED. Details saved to: {os.path.abspath(output_path)}')
