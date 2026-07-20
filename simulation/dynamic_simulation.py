"""
Dynamic simulation of the biodiesel reactor control system.

This script performs a dynamic simulation (plant-only) of the biodiesel reactor.
All inputs (manipulated variables and disturbances) are provided directly.

Usage
-----
1. Edit the ``# --- Configuration ---`` section to adjust the time horizon,
   step time, and input profiles.
2. Run: ``python simulation/dynamic_simulation.py``
3. Access results via ``result['signal_name']``.
   e.g., ``result['T']``, ``result['h']``.

Signal layout
-------------
Inputs  : 13 inputs (8 disturbances + 5 manipulated variables)
States  : 11 plant states
Outputs : 11 plant states
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from model import BiodieselPlant
from model.config import (
    PROCESS_PARAMS,
    X0_NOMINAL,
    U_NOMINAL,
)
from model.simsys import DynamicSimulation

# =============================================================================
# Configuration
# =============================================================================

TIME_END   = 20_000   # simulation horizon [s]
TIME_STEP  = 0.5      # output time step   [s]
FIRST_STEP = 600      # time of first step change [s]


# =============================================================================
# Build elements
# =============================================================================

plant = BiodieselPlant(**PROCESS_PARAMS)

sim = DynamicSimulation(plant)

# Inspect signal layout before running
# print(sim)
# print("Input names :", sim.input_names)
# print("State names :", sim.state_names)
# print("Output names:", sim.output_names)


# =============================================================================
# Time vector
# =============================================================================

time = np.arange(0, TIME_END + TIME_STEP, TIME_STEP)


# =============================================================================
# Input profiles
# =============================================================================
#
# Build U with sim.make_U(time, **named_signals).
# We generate steady-state arrays for all nominal inputs.

U_profiles = {name: np.full_like(time, val) for name, val in U_NOMINAL.items()}

# Example step change in one of the inputs
# U_profiles["f_coolant"][time >= FIRST_STEP] *= 1.1

U = sim.make_U(time, **U_profiles)


# =============================================================================
# Initial conditions
# =============================================================================
#
# Use sim.make_X0(**state_name=value).
# Unspecified states default to 0.

X0 = sim.make_X0(**X0_NOMINAL)


# =============================================================================
# Run simulation
# =============================================================================

result = sim.run(time, U, X0)


# =============================================================================
# Quick plot — Temperature and Level
# =============================================================================

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

ax1.plot(time, result["T"], label="T — reactor temperature [K]", color="red")
ax1.axvline(FIRST_STEP, color="gray", linestyle=":", alpha=0.5)
ax1.set_ylabel("Temperature [K]")
ax1.legend(frameon=False)
ax1.grid(True, alpha=0.3)
ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

ax2.plot(time, result["h"], label="h — liquid level [m]", color="blue")
ax2.axvline(FIRST_STEP, color="gray", linestyle=":", alpha=0.5)
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Level [m]")
ax2.legend(frameon=False)
ax2.grid(True, alpha=0.3)
ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

plt.tight_layout()
plt.show()
