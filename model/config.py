"""
Default parameter sets for the biodiesel reactor control system.

This module provides the default constants and configuration
dictionaries for the biodiesel plant and its control systems. Each
constant is a dictionary (or nested dictionary) that can be unpacked
directly into the corresponding class constructor.

Notes
-----
Five control loops are defined:
- LIC-100 : Level control (reactor liquid level [m])
- FIC-100 : Flow control (oil feed flow [m³/s])
- FIC-101 : Flow control (methanol feed flow [m³/s])
- FIC-102 : Flow control (NaOH catalyst feed flow [m³/s])
- TIC-100 : Temperature control (reactor temperature [K])
"""

# =============================================================================
# Reactor model parameters (BiodieselPlant)
# =============================================================================
# fmt: off
PROCESS_PARAMS = {
    # --- Feed stream physical properties ---
    "rho_oil":     884.72,                # oil density                    [kg/m³]
    "Cp_oil":      0.4454,                # oil heat capacity              [kcal/(kg·K)]
    "rho_MeOH":    792.92,                # methanol density               [kg/m³]
    "Cp_MeOH":     0.6734,                # methanol heat capacity         [kcal/(kg·K)]
    "rho_NaOH":    1041.12,               # NaOH solution density          [kg/m³]
    "Cp_NaOH":     0.8511,                # NaOH heat capacity             [kcal/(kg·K)]

    # --- Product mixture properties ---
    "rho":         792.77,                # mixture density                [kg/m³]
    "Cp":          0.5234,                # mixture heat capacity          [kcal/(kg·K)]

    # --- Reactor geometry ---
    "Dr":          1.219,                 # internal diameter              [m]
    "Lr":          3.0,                   # length                         [m]

    # --- Reaction thermodynamics ---
    "R":           1.987,                 # gas constant                   [kcal/(kmol·K)]
    "To":          323.15,                # Arrhenius ref. temp.           [K]
    "Hrxn1":      -14379.67807351148,     # heat of reaction 1             [kcal/kmol]
    "Hrxn2":      -237.28589845450915,    # heat of reaction 2             [kcal/kmol]
    "Hrxn3":      -12351.943870983516,    # heat of reaction 3             [kcal/kmol]

    # --- Kinetic rate constants ---
    "k1_f":        0.02311,               # reaction 1 forward
    "k1_r":        0.001867,              # reaction 1 reverse
    "k2_f":        0.10659,               # reaction 2 forward
    "k2_r":        0.002217,              # reaction 2 reverse
    "k3_f":        0.05754,               # reaction 3 forward
    "k3_r":        0.000267,              # reaction 3 reverse

    # --- Activation energies [kJ/kmol] ---
    "E1_f":        13500.0,               # reaction 1 forward
    "E1_r":        10300.0,               # reaction 1 reverse
    "E2_f":        17400.0,               # reaction 2 forward
    "E2_r":        16200.0,               # reaction 2 reverse
    "E3_f":        6200.0,                # reaction 3 forward
    "E3_r":        11900.0,               # reaction 3 reverse

    # --- Heat transfer ---
    "UA":          0.3770945,             # overall heat transfer coeff    [kcal/(s·K)]
    "V_coolant":   0.3607,                # cooling jacket volume          [m³]

    # --- Cooling fluid properties ---
    "rho_coolant": 998.0,                 # water density                  [kg/m³]
    "Cp_coolant":  1.0,                   # water heat capacity            [kcal/(kg·K)]
}
# fmt: on

# =============================================================================
# Nominal Mass Flow Rates [kg/s]
# =============================================================================
M_NOMINAL = {
    'm_oil': 0.291666666655013,
    'm_MeOH': 6.61111111082406e-002,
    'm_NaOH': 1.38888888883481e-002,
    'm_FAME': 4.688202967463472219e-04 * PROCESS_PARAMS['rho'],
    'm_coolant': 1.511403610545926292e-04 * PROCESS_PARAMS['rho_coolant'],
}

# =============================================================================
# Nominal Input operating points (u*)
# =============================================================================
# fmt: off
U_NOMINAL = {
    # Disturbances
    "c_TG_in":      0.9992,               # inlet TG concentration         [kmol/m³]
    "T_oil":        333.15,               # inlet oil temperature          [K]
    "c_MeOH_in":    24.7462,              # inlet MeOH concentration       [kmol/m³]
    "T_MeOH":       298.15,               # inlet MeOH temperature         [K]
    "c_Cat_in":     5.2060,               # inlet Catalyst concentration   [kmol/m³]
    "c_Water_in":   46.2325,              # inlet Water concentration      [kmol/m³]
    "T_NaOH":       298.15,               # inlet NaOH temperature         [K]
    "T_coolant_in": 298.15,               # inlet coolant temperature      [K]

    # Manipulated Variables (Flows)
    "f_oil":        M_NOMINAL["m_oil"] / PROCESS_PARAMS["rho_oil"],            # nominal oil flow            [m³/s]
    "f_MeOH":       M_NOMINAL["m_MeOH"] / PROCESS_PARAMS["rho_MeOH"],          # nominal MeOH flow           [m³/s]
    "f_NaOH":       M_NOMINAL["m_NaOH"] / PROCESS_PARAMS["rho_NaOH"],          # nominal NaOH flow           [m³/s]
    "f_FAME":       M_NOMINAL["m_FAME"] / PROCESS_PARAMS["rho"],               # nominal FAME flow           [m³/s]
    "f_coolant":    M_NOMINAL["m_coolant"] / PROCESS_PARAMS["rho_coolant"],    # nominal coolant flow        [m³/s]
}
# fmt: on

# =============================================================================
# Nominal Plant Initial States (x*)
# =============================================================================
# fmt: off
X0_NOMINAL = {
    "h":            1.500000000000000000e+00,  # liquid level                [m]
    "c_TG":         1.424673186536476975e-02,  # TG concentration            [kmol/m³]
    "c_MeOH":       2.375994113139722863e+00,  # MeOH concentration          [kmol/m³]
    "c_ME":         2.024963615183559273e+00,  # ME concentration            [kmol/m³]
    "c_DG":         5.470101247941898566e-03,  # DG concentration            [kmol/m³]
    "c_MG":         2.924736691761852317e-02,  # MG concentration            [kmol/m³]
    "c_Gly":        6.536662600334592899e-01,  # Gly concentration           [kmol/m³]
    "c_Cat":        1.481373162754769934e-01,  # Cat concentration           [kmol/m³]
    "c_Water":      1.315550993988857220e+00,  # Water concentration         [kmol/m³]
    "T":            3.331499999999999773e+02,  # reactor temperature         [K]
    "T_coolant":    3.231499995331167270e+02,  # coolant temperature         [K]
}
# fmt: on

# =============================================================================
# Open-Loop Initial State Vector (X0_OPEN_LOOP)
# =============================================================================
# fmt: off
X0_OPEN_LOOP = {
    # Plant states
    **X0_NOMINAL,
    # Actuator valve positions [%vp]
    "LCV_100.vp":   50.0,
    "TCV_100.vp":   20.0,
    "FCV_100.vp":   50.0,
    "FCV_101.vp":   50.0,
    "FCV_102.vp":   50.0,
    "TT_100.PVm":   X0_NOMINAL["T"],
}
# fmt: on

# =============================================================================
# Closed-Loop Initial State Vector (X0_CLOSED_LOOP)
# =============================================================================
# fmt: off
X0_CLOSED_LOOP = {
    # Plant states
    **X0_NOMINAL,
    # Controller integrator/derivative states
    "LC_100.I_state": 0.0,
    "LC_100.D_state": 0.0,
    "TC_100.I_state": 80.0,
    "TC_100.D_state": 50.0,
    "TT_100.PVm":     X0_NOMINAL["T"],    # TT-100 nominal measurement [K]
    "FC_100.I_state": 50.0,
    "FC_100.D_state": 0.0,
    "FC_101.I_state": 50.0,
    "FC_101.D_state": 0.0,
    "FC_102.I_state": 50.0,
    "FC_102.D_state": 0.0,
    # Actuator valve positions [%vp]
    "LCV_100.vp":     50.0,
    "TCV_100.vp":     20.0,
    "FCV_100.vp":     50.0,
    "FCV_101.vp":     50.0,
    "FCV_102.vp":     50.0,
}
# fmt: on

# =============================================================================
# Sensor / transmitter parameters (SensorTransmitterSystem)
# =============================================================================
# fmt: off
SENSOR_TRANSMITTER_PARAMS = {
    "LIC-100": {
        "name":  "LT_100",                              # name
        "hi":    3.0,                                   # high range value [m] | 100% level
        "low":   0.0,                                   # low range value [m]
        "tauT":  0.0                                    # time constant [s]
    },
    "FIC-100": {
        "name":  "FT_100",
        "hi":    U_NOMINAL["f_oil"] * 2.0,             # 100% overcapacity oil flow rate [m³/s]
        "low":   0.0,
        "tauT":  0.0
    },
    "FIC-101": {
        "name":  "FT_101",
        "hi":    U_NOMINAL["f_MeOH"] * 2.0,            # 100% overcapacity MeOH flow rate [m³/s]
        "low":   0.0,
        "tauT":  0.0
    },
    "FIC-102": {
        "name":  "FT_102",
        "hi":    U_NOMINAL["f_NaOH"] * 2.0,            # 100% overcapacity NaOH flow rate [m³/s]
        "low":   0.0,
        "tauT":  0.0
    },
    "TIC-100": {
        "name":  "TT_100",
        "hi":    368.15,                                # high range value [K]
        "low":   298.15,                                # low range value [K]
        "tauT":  45.0                                   # time constant [s]
    },
}
# fmt: on

# =============================================================================
# Control valve parameters (ActuatorSystem)
# =============================================================================
# fmt: off
ACTUATOR_PARAMS = {
    "LIC-100": {
        "name":         "LCV_100",
        "tauV":         12.0,                           # time constant        [s]
        "f_max":        U_NOMINAL["f_FAME"] * 2.0,      # max flow rate        [m³/s]
        "vp_min":       0.0,                            # min valve position   [%]
        "vp_max":       100.0,                          # max valve position   [%]
        "valve_type":   "linear",
        "valve_action": "FC",                           # fail-closed
    },
    "FIC-100": {
        "name":         "FCV_100",
        "tauV":         12.0,                           # time constant        [s]
        "f_max":        U_NOMINAL["f_oil"] * 2.0,       # max flow rate        [m³/s]
        "vp_min":       0.0,                            # min valve position   [%]
        "vp_max":       100.0,                          # max valve position   [%]
        "valve_type":   "linear",
        "valve_action": "FC",                           # fail-closed
    },
    "FIC-101": {
        "name":         "FCV_101",
        "tauV":         12.0,                           # time constant        [s]
        "f_max":        U_NOMINAL["f_MeOH"] * 2.0,      # max flow rate        [m³/s]
        "vp_min":       0.0,                            # min valve position   [%]
        "vp_max":       100.0,                          # max valve position   [%]
        "valve_type":   "linear",
        "valve_action": "FC",                           # fail-closed
    },
    "FIC-102": {
        "name":         "FCV_102",
        "tauV":         12.0,                           # time constant        [s]
        "f_max":        U_NOMINAL["f_NaOH"] * 2.0,      # max flow rate        [m³/s]
        "vp_min":       0.0,                            # min valve position   [%]
        "vp_max":       100.0,                          # max valve position   [%]
        "valve_type":   "linear",
        "valve_action": "FC",                           # fail-closed
    },
    "TIC-100": {
        "name":         "TCV_100",
        "tauV":         12.0,                           # time constant        [s]
        "f_max":        U_NOMINAL["f_coolant"] * 5.0,   # max flow rate        [m³/s]
        "vp_min":       0.0,                            # min valve position   [%]
        "vp_max":       100.0,                          # max valve position   [%]
        "valve_type":   "linear",
        "valve_action": "FO",                           # fail-open (cooling valve)
    },
}
# fmt: on

# =============================================================================
# PID controller parameters (ControllerSystem)
# =============================================================================
# fmt: off
CONTROLLER_PARAMS = {
    # tag         name              bias (%)      mode            acting
    "LIC-100": { "name": "LC_100", "bias": 50.0, "mode": "AUTO", "acting": "DIRECT"  },
    "TIC-100": { "name": "TC_100", "bias": 80.0, "mode": "AUTO", "acting": "REVERSE" },
    "FIC-100": { "name": "FC_100", "bias": 50.0, "mode": "AUTO", "acting": "REVERSE" },
    "FIC-101": { "name": "FC_101", "bias": 50.0, "mode": "AUTO", "acting": "REVERSE" },
    "FIC-102": { "name": "FC_102", "bias": 50.0, "mode": "AUTO", "acting": "REVERSE" },
}
# fmt: on

# =============================================================================
# Setpoint parameters (SetPointSystem)
# =============================================================================
# fmt: off
SETPOINT_PARAMS = {
    # tag         name               hi (eng. unit)                   low (eng. unit)
    "LIC-100": { "name": "LSP_100", "hi": 3.0,                       "low": 0.0 },
    "FIC-100": { "name": "FSP_100", "hi": U_NOMINAL["f_oil"]  * 2.0, "low": 0.0 },
    "FIC-101": { "name": "FSP_101", "hi": U_NOMINAL["f_MeOH"] * 2.0, "low": 0.0 },
    "FIC-102": { "name": "FSP_102", "hi": U_NOMINAL["f_NaOH"] * 2.0, "low": 0.0 },
    "TIC-100": { "name": "TSP_100", "hi": 368.15,                    "low": 298.15 },
}
# fmt: on

# =============================================================================
# PID controller tuning parameters (Kc, tauI, tauD)
# =============================================================================
# fmt: off
TUNING_PARAMS = {
    "LIC-100": {
        "Tight" :     { "Kc": 1.0, "tauI": 0.0, "tauD": 0.0 },  # Tight Level (P-only)
        "Averaging" : { "Kc": 77.8, "tauI": 0.0, "tauD": 0.0 }, # Averaging Level (P-only)
    },
    "TIC-100": {
        "QDR":              { "Kc": 10.34,  "tauI": 1070.07, "tauD": 267.52 },  # Temperature (PID)
        "Syn-Disturbance":  { "Kc": 8.62,  "tauI": 4117.77, "tauD": 267.52 },   # Temperature (PID)
        "Syn-Setpoint":     { "Kc": 7.18,  "tauI": 4117.77, "tauD": 267.52 },   # Temperature (PID)
        "IAE-Disturbance":  { "Kc": 7.80,  "tauI": 754.06, "tauD": 262.99 },    # Temperature (PID)
        "IAE-Setpoint":     { "Kc": 6.87,  "tauI": 5463.23, "tauD": 231.31 },   # Temperature (PID)
    },
    "FIC-100": { "Kc": 0.33, "tauI": 12.0, "tauD": 0.0 }, # Oil flow (PI)
    "FIC-101": { "Kc": 0.33, "tauI": 12.0, "tauD": 0.0 }, # MeOH flow (PI)
    "FIC-102": { "Kc": 0.33, "tauI": 12.0, "tauD": 0.0 }, # NaOH flow (PI)
}
# fmt: on
