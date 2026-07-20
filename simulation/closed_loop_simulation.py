"""
Closed-loop simulation of the biodiesel reactor control system.

This script runs a closed-loop simulation of the biodiesel reactor with
five PID feedback loops.

Notes
-----
Five PID feedback loops:
- LIC-100 : Level control (h [m] via LCV-100, f_FAME)
- TIC-100 : Temperature control (T [K] via TCV-100, f_coolant)
- FIC-100 : Flow control (f_oil [m³/s] via FCV-100)
- FIC-101 : Flow control (f_MeOH via FCV-101)
- FIC-102 : Flow control (f_NaOH via FCV-102)

FIC loops use ``pv_source='actuator'``: the flow transmitter measures the
actuator output F directly (flow meter on control valve outlet), not a
plant state.

Usage
-----
1. Edit the ``# --- Configuration ---`` section:
   - Set TIME_END, TIME_STEP, STEP_TIME.
   - Adjust PID tuning constants (Kc, tauI, tauD) per loop.
   - Adjust setpoint step profiles.
2. Run: ``python simulation/closed_loop_simulation.py``
3. Access results via ``result['signal_name']``.

Signal layout
-------------
Inputs: 8 disturbances + 4 per loop × 5 loops = 28 total
- Disturbances: c_TG_in, T_oil, c_MeOH_in, T_MeOH, c_Cat_in,
                c_Water_in, T_NaOH, T_coolant_in
- Per loop (keyed by controller tag, e.g. LC_100):
    SP_LC_100, Kc_LC_100, tauI_LC_100, tauD_LC_100
    SP_TC_100, Kc_TC_100, ...  etc.

Inspect at runtime: ``print(sim.input_names)``
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter

from model import (
    ActuatorSystem,
    BiodieselPlant,
    ControllerSystem,
    SensorTransmitterSystem,
    SetPointSystem,
)
from model.config import (
    ACTUATOR_PARAMS,
    CONTROLLER_PARAMS,
    PROCESS_PARAMS,
    SENSOR_TRANSMITTER_PARAMS,
    SETPOINT_PARAMS,
    TUNING_PARAMS,
    U_NOMINAL,
    X0_CLOSED_LOOP,
    X0_NOMINAL,
)
from model.simsys import ClosedLoopSimulation, ControlLoop

# =============================================================================
# Configuration
# =============================================================================

TIME_END = 10000  # simulation horizon [s]
TIME_STEP = 0.5  # output time step   [s]
STEP_TIME = 600  # time of setpoint step [s]

# --- Setpoint nominal values (before step) ---
NOM_LSP_100 = X0_NOMINAL['h']
NOM_TSP_100 = X0_NOMINAL['T']
NOM_FSP_100 = U_NOMINAL['f_oil']
NOM_FSP_101 = U_NOMINAL['f_MeOH']
NOM_FSP_102 = U_NOMINAL['f_NaOH']

# --- Setpoint values after step ---
SP_LSP_100 = X0_NOMINAL['h']
SP_TSP_100 = X0_NOMINAL['T']
SP_FSP_100 = U_NOMINAL['f_oil']
SP_FSP_101 = U_NOMINAL['f_MeOH']
SP_FSP_102 = U_NOMINAL['f_NaOH']


# =============================================================================
# Build elements
# =============================================================================

plant = BiodieselPlant(**PROCESS_PARAMS)

# Controllers
LC_100 = ControllerSystem(**CONTROLLER_PARAMS['LIC-100'])
TC_100 = ControllerSystem(**CONTROLLER_PARAMS['TIC-100'])
FC_100 = ControllerSystem(**CONTROLLER_PARAMS['FIC-100'])
FC_101 = ControllerSystem(**CONTROLLER_PARAMS['FIC-101'])
FC_102 = ControllerSystem(**CONTROLLER_PARAMS['FIC-102'])

# Setpoint stations
LSP_100 = SetPointSystem(**SETPOINT_PARAMS['LIC-100'])
TSP_100 = SetPointSystem(**SETPOINT_PARAMS['TIC-100'])
FSP_100 = SetPointSystem(**SETPOINT_PARAMS['FIC-100'])
FSP_101 = SetPointSystem(**SETPOINT_PARAMS['FIC-101'])
FSP_102 = SetPointSystem(**SETPOINT_PARAMS['FIC-102'])

# Actuators (control valves)
LCV_100 = ActuatorSystem(**ACTUATOR_PARAMS['LIC-100'])  # f_FAME
TCV_100 = ActuatorSystem(**ACTUATOR_PARAMS['TIC-100'])  # f_coolant
FCV_100 = ActuatorSystem(**ACTUATOR_PARAMS['FIC-100'])  # f_oil
FCV_101 = ActuatorSystem(**ACTUATOR_PARAMS['FIC-101'])  # f_MeOH
FCV_102 = ActuatorSystem(**ACTUATOR_PARAMS['FIC-102'])  # f_NaOH

# Sensor-transmitters
LT_100 = SensorTransmitterSystem(**SENSOR_TRANSMITTER_PARAMS['LIC-100'])
TT_100 = SensorTransmitterSystem(**SENSOR_TRANSMITTER_PARAMS['TIC-100'])
FT_100 = SensorTransmitterSystem(**SENSOR_TRANSMITTER_PARAMS['FIC-100'])
FT_101 = SensorTransmitterSystem(**SENSOR_TRANSMITTER_PARAMS['FIC-101'])
FT_102 = SensorTransmitterSystem(**SENSOR_TRANSMITTER_PARAMS['FIC-102'])


# =============================================================================
# Define control loops
# =============================================================================
#
# LIC-100 and TIC-100: sensor measures a plant state → pv_source='plant'
# FIC-100/101/102: sensor measures valve output flow → pv_source='actuator'

loops = [
    # LIC-100 — Level control: h [m] via f_FAME (FAME outlet valve)
    ControlLoop(
        mv='f_FAME',
        pv='h',
        pv_source='plant',
        actuator=LCV_100,
        controller=LC_100,
        setpoint=LSP_100,
        sensor=LT_100,
    ),
    # TIC-100 — Temperature control: T [K] via f_coolant (cooling valve, FO)
    ControlLoop(
        mv='f_coolant',
        pv='T',
        pv_source='plant',
        actuator=TCV_100,
        controller=TC_100,
        setpoint=TSP_100,
        sensor=TT_100,
    ),
    # FIC-100 — Oil flow control: FCV-100 outlet flow via FT-100
    ControlLoop(
        mv='f_oil',
        pv='F',
        pv_source='actuator',
        actuator=FCV_100,
        controller=FC_100,
        setpoint=FSP_100,
        sensor=FT_100,
    ),
    # FIC-101 — MeOH flow control: FCV-101 outlet flow via FT-101
    ControlLoop(
        mv='f_MeOH',
        pv='F',
        pv_source='actuator',
        actuator=FCV_101,
        controller=FC_101,
        setpoint=FSP_101,
        sensor=FT_101,
    ),
    # FIC-102 — NaOH flow control: FCV-102 outlet flow via FT-102
    ControlLoop(
        mv='f_NaOH',
        pv='F',
        pv_source='actuator',
        actuator=FCV_102,
        controller=FC_102,
        setpoint=FSP_102,
        sensor=FT_102,
    ),
]

sim = ClosedLoopSimulation(plant, loops)

# Inspect signal layout before building U and X0
print(sim)
print('Input names :', sim.input_names)
print('State names :', sim.state_names)
print('Output names:', sim.output_names)


# =============================================================================
# Time vector
# =============================================================================

time = np.arange(0, TIME_END + TIME_STEP, TIME_STEP)


# =============================================================================
# Input profiles
# =============================================================================
#
# Use sim.make_U(time, **named_signals).
#
# Setpoint step pattern:
#   SP = np.full_like(time, NOM); SP[time >= STEP_TIME] = SP_AFTER
#
# Multiple steps:
#   SP[time >= t1] = val1; SP[time >= t2] = val2
#
# Disturbance step (uncomment lines below):
#   T_oil[time >= STEP_TIME] = 338.15   # +5 K feed-oil temperature step
#
# PID tuning profiles are constant by default; use time-varying arrays to
# implement gain scheduling during the simulation.

# --- Setpoint profiles ---
SP_profiles = {
    'SP_LC_100': np.full_like(time, NOM_LSP_100),
    'SP_TC_100': np.full_like(time, NOM_TSP_100),
    'SP_FC_100': np.full_like(time, NOM_FSP_100),
    'SP_FC_101': np.full_like(time, NOM_FSP_101),
    'SP_FC_102': np.full_like(time, NOM_FSP_102),
}
SP_profiles['SP_LC_100'][time >= STEP_TIME] = SP_LSP_100
SP_profiles['SP_TC_100'][time >= STEP_TIME] = SP_TSP_100
SP_profiles['SP_FC_100'][time >= STEP_TIME] = SP_FSP_100
SP_profiles['SP_FC_101'][time >= STEP_TIME] = SP_FSP_101
SP_profiles['SP_FC_102'][time >= STEP_TIME] = SP_FSP_102

# --- Disturbance profiles ---
U_disturbances = {k: np.full_like(time, v) for k, v in U_NOMINAL.items() if not k.startswith('f_')}

# --- PID tuning profiles (constant → fixed gains) ---
tuning_profiles = {}
for loop_id, strategy in [
    ('LIC-100', 'Tight'),
    ('TIC-100', 'QDR'),
    ('FIC-100', None),
    ('FIC-101', None),
    ('FIC-102', None),
]:
    p = TUNING_PARAMS[loop_id][strategy] if strategy else TUNING_PARAMS[loop_id]
    tag = loop_id.replace('I', '').replace('-', '_')  # e.g., 'LC_100'
    for key in ['Kc', 'tauI', 'tauD']:
        tuning_profiles[f'{key}_{tag}'] = np.full_like(time, p[key])

# Build U — named inputs map to sim.input_names automatically
# Input keys for tuning/SP follow the pattern SP_{ctrl_tag}, Kc_{ctrl_tag}, ...
U = sim.make_U(time, **U_disturbances, **SP_profiles, **tuning_profiles)


# =============================================================================
# Initial conditions
# =============================================================================
#
# State order (see sim.state_names):
#   Plant (11): h, c_TG, c_MeOH, c_ME, c_DG, c_MG, c_Gly, c_Cat, c_Water, T, T_coolant
#   LIC-100 loop: LC_100.I_state, LC_100.D_state, LCV_100.vp
#   TIC-100 loop: TT_100.PVm, TC_100.I_state, TC_100.D_state, TCV_100.vp
#   FIC-100 loop: FC_100.I_state, FC_100.D_state, FCV_100.vp
#   FIC-101 loop: FC_101.I_state, FC_101.D_state, FCV_101.vp
#   FIC-102 loop: FC_102.I_state, FC_102.D_state, FCV_102.vp
#
# Controller I_state pre-loaded for bumpless transfer (= initial output M bias).
# Actuator vp_ss = initial steady-state valve position [%].

X0 = sim.make_X0(**X0_CLOSED_LOOP)


# =============================================================================
# Run simulation
# =============================================================================

result = sim.run(time, U, X0)

print(result)


# =============================================================================
# Extract signals for analysis / plotting
# =============================================================================

# Removed explicit signal extraction to reduce verbosity.
# Signals can be plotted directly from the `result` dictionary.


# =============================================================================
# Optional: export loop data to CSV
# Uncomment and adjust column names / path.
# =============================================================================

# import pandas as pd
# pd.DataFrame({
#     'Time':        time,
#     'SP_T':        SP_TSP,
#     'R_TSP_100':   R_TSP_100,
#     'C_TT_100':    C_TT_100,
#     'M_TC_100':    M_TC_100,
#     'vp_TCV_100':  vp_TCV_100,
# }).to_csv('TIC_100_closed_loop_step.csv', index=False)


# =============================================================================
# Quick plot — TIC-100 closed-loop step response
# =============================================================================

fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# Top: measurement vs setpoint
axes[0].plot(time, result['TT_100.C'], label='TT-100 (%TO)')
axes[0].plot(time, result['TSP_100.R'], label='TSP-100 SP (%TO)', linestyle='--')
axes[0].axvline(STEP_TIME, color='gray', linestyle=':', alpha=0.5)
axes[0].set_ylabel('Signal (%TO)')
# axes[0].set_title("TIC-100 Closed-Loop Step Response")  # no title per publication style
axes[0].legend(frameon=False)
axes[0].grid(True, alpha=0.3)

# Bottom: controller output
axes[1].plot(time, result['TC_100.M'], label='TC-100 M (%CO)', color='tab:orange')
axes[1].axvline(STEP_TIME, color='gray', linestyle=':', alpha=0.5)
axes[1].set_xlabel('Time (s)')
axes[1].set_ylabel('Controller Output (%CO)')
axes[1].legend(frameon=False)
axes[1].grid(True, alpha=0.3)

for ax in axes:
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_formatter(FormatStrFormatter('%.2f'))

plt.tight_layout()
plt.show()
