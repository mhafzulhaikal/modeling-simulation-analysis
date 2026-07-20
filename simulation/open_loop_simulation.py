"""
Open-loop simulation of the biodiesel reactor control system.

This script performs an open-loop test of the plant with five actuators
(LCV-100, TCV-100, FCV-100/101/102) wired to the plant. No controllers
or setpoint stations are used; each valve is driven directly by a
controller-output (M) signal supplied externally.

Usage
-----
1. Edit the ``# --- Configuration ---`` section to adjust the time horizon,
   step time, disturbance profiles, and M (valve opening) profiles.
2. Run: ``python simulation/open_loop_simulation.py``
3. Access results via ``result['signal_name']``.
   e.g., ``result['T']``, ``result['LCV_100.vp']``.

Signal layout
-------------
Inputs  : 8 disturbances + 5 M signals (see ``sim.input_names``)
States  : 11 plant states + 5 actuator valve positions
Outputs : 11 plant states
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np

from model import ActuatorSystem, BiodieselPlant, SensorTransmitterSystem
from model.config import (
    ACTUATOR_PARAMS,
    PROCESS_PARAMS,
    SENSOR_TRANSMITTER_PARAMS,
    X0_OPEN_LOOP,
    U_NOMINAL,
)
from model.simsys import OpenLoopSimulation

# =============================================================================
# Configuration
# =============================================================================

TIME_END  = 30_000   # simulation horizon [s]
TIME_STEP = 0.5      # output time step   [s]
FIRST_STEP = 600     # time of first step change [s]
SECOND_STEP = 1_800  # time of second step change [s]


# =============================================================================
# Build elements
# =============================================================================

plant = BiodieselPlant(**PROCESS_PARAMS)

LCV_100 = ActuatorSystem(**ACTUATOR_PARAMS["LIC-100"])  # level    → f_FAME
TCV_100 = ActuatorSystem(**ACTUATOR_PARAMS["TIC-100"])  # temp     → f_coolant
FCV_100 = ActuatorSystem(**ACTUATOR_PARAMS["FIC-100"])  # oil flow → f_oil
FCV_101 = ActuatorSystem(**ACTUATOR_PARAMS["FIC-101"])  # MeOH     → f_MeOH
FCV_102 = ActuatorSystem(**ACTUATOR_PARAMS["FIC-102"])  # NaOH     → f_NaOH

# Plant MV name → ActuatorSystem
actuators = {
    "f_FAME": LCV_100,
    "f_coolant": TCV_100,
    "f_oil": FCV_100,
    "f_MeOH": FCV_101,
    "f_NaOH": FCV_102,
}

# Plant PV name → SensorTransmitterSystem
LT_100 = SensorTransmitterSystem(**SENSOR_TRANSMITTER_PARAMS["LIC-100"])
TT_100 = SensorTransmitterSystem(**SENSOR_TRANSMITTER_PARAMS["TIC-100"])
FT_100 = SensorTransmitterSystem(**SENSOR_TRANSMITTER_PARAMS["FIC-100"])
FT_101 = SensorTransmitterSystem(**SENSOR_TRANSMITTER_PARAMS["FIC-101"])
FT_102 = SensorTransmitterSystem(**SENSOR_TRANSMITTER_PARAMS["FIC-102"])

sensors = {
    "h": LT_100,
    "T": TT_100,
    "f_oil": FT_100,
    "f_MeOH": FT_101,
    "f_NaOH": FT_102,
}

sim = OpenLoopSimulation(plant, actuators, sensors)

# Inspect signal layout before running
print(sim)
print("Input names :", sim.input_names)
print("State names :", sim.state_names)
print("Output names:", sim.output_names)


# =============================================================================
# Time vector
# =============================================================================

time = np.arange(0, TIME_END + TIME_STEP, TIME_STEP)


# =============================================================================
# Input profiles
# =============================================================================
#
# Build U with sim.make_U(time, **named_signals).
# Any unspecified input defaults to zero.
#
# Disturbance step pattern:
#   signal = np.where(time < STEP_TIME, before, after)
#
# Multiple steps:
#   signal = np.full_like(time, val0)
#   signal[time >= t1] = val1
#   signal[time >= t2] = val2

# Disturbances — from nominal
U_disturbances = {k: np.full_like(time, v) for k, v in U_NOMINAL.items() if not k.startswith("f_")}

# Controller output (M) profiles [%CO]
# Keep these fixed for an open-loop test; change one to observe step response.
M_profiles = {
    "M_f_FAME":    np.full_like(time, 50.0),
    "M_f_coolant": np.full_like(time, 80.0),
    "M_f_oil":     np.full_like(time, 50.0),
    "M_f_MeOH":    np.full_like(time, 50.0),
    "M_f_NaOH":    np.full_like(time, 50.0),
}

M_profiles["M_f_coolant"][time >= FIRST_STEP] = 90.0  # step change in temp controller output

U = sim.make_U(time, **U_disturbances, **M_profiles)


# =============================================================================
# Initial conditions
# =============================================================================
#
# Use sim.make_X0(**state_name=value).
# Unspecified states default to 0.
#
# State names: plant states use short names; actuator states use 'InstrTag.vp'.

X0 = sim.make_X0(**X0_OPEN_LOOP)


# =============================================================================
# Run simulation
# =============================================================================

result = sim.run(time, U, X0)

print(result)
print("h  : %.4f -> %.4f m" % (result["h"][0], result["h"][-1]))
print("T  : %.2f -> %.2f K" % (result["T"][0], result["T"][-1]))


# Removed explicit signal extraction to reduce verbosity.
# Signals can be plotted directly from the `result` dictionary.


# =============================================================================
# Optional: export to CSV
# Uncomment and adjust as needed.
# =============================================================================

# import pandas as pd
# pd.DataFrame({
#     'Time':      time,
#     'M_temp':    M_temp,
#     'T':         T,
#     'T_coolant': T_coolant,
# }).to_csv('TIC_100_open_loop_step.csv', index=False)


# =============================================================================
# Quick plot — TIC-100 response (temperature)
# =============================================================================

plt.figure(figsize=(14, 5))
plt.plot(time, result["TT_100.C"], label="C_TT_100 — transmitter output [%TO]")
plt.axvline(
    FIRST_STEP,
    color="gray",
    linestyle=":",
    alpha=0.5,
    label=f"Step at t = {FIRST_STEP} s",
)
plt.xlabel("Time (s)")
plt.ylabel("Temperature [%TO]")
# plt.title("TIC-100 Open-Loop Step Response")  # no title per publication style
plt.legend(frameon=False)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
