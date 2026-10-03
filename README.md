# Modeling, Simulation, and Analysis of a Biodiesel Reactor Control System

## Executive Summary

This repository contains a comprehensive Python-based modeling and simulation framework for a biodiesel production reactor equipped with advanced process control. The project integrates dynamic plant modeling, actuator and sensor dynamics, control-loop design and tuning, and performance analysis methodologies to evaluate controller behavior under various operating conditions.

The framework enables systematic investigation of servo and regulatory control performance, operating-point optimization, and validation of tuning strategies through comparative analysis between simulation and reference datasets.

## 1. Introduction

### 1.1 Motivation and Context

Biodiesel production is a multi-stage chemical process involving transesterification reactions, material handling, and temperature/level/flow control. Achieving optimal process performance requires careful design and tuning of control loops responsible for maintaining critical process variables within specification.

This project provides a research and educational platform for:

- **System Characterization**: Dynamic modeling of the biodiesel reactor and associated unit operations
- **Controller Design and Tuning**: Development of proportional-integral-derivative (PID) control strategies using both classical and model-based methods
- **Closed-Loop Analysis**: Simulation of servo (setpoint tracking) and regulatory (disturbance rejection) responses
- **Performance Evaluation**: Quantitative assessment of control quality through standard metrics
- **Validation and Comparison**: Benchmarking simulation results against experimental data or reference models

### 1.2 Project Scope

This repository addresses:

1. **Plant Modeling**: First-principles and empirical dynamic models of the biodiesel reactor
2. **Actuator Dynamics**: Control valve response models with realistic time constants and dead zones
3. **Sensor/Transmitter Dynamics**: Measurement transmission models including delays and transmitter output ranges
4. **Control System Design**: PID, tuned PID, and advanced control configurations
5. **Simulation Infrastructure**: Open-loop and closed-loop simulation engines with configurable scenarios
6. **Performance Analysis**: Calculation of transient response characteristics and integral performance indices
7. **Reporting Framework**: Automated generation of performance summaries and visualization artifacts

## 2. Repository Structure and Organization

### 2.1 Directory Layout

```
modeling-simulation-analysis/
│
├── model/                              # Core process and control models
│   ├── __init__.py                    # Package exports and public API
│   ├── config.py                      # Configuration parameters (process, actuators, sensors)
│   ├── plant.py                       # Biodiesel plant dynamic model
│   ├── actsys.py                      # Actuator system (valve) models
│   ├── stsys.py                       # Sensor/transmitter system models
│   ├── ctrlbase.py                    # Base controller class and interface
│   ├── ctrlsys.py                     # Concrete controller implementations
│   ├── spsys.py                       # Setpoint station model
│   ├── simsys.py                      # Simulation engine and result handling
│   ├── fopdt.py                       # First-order-plus-dead-time (FOPDT) model utilities
│   ├── stepinfo.py                    # Transient response metrics and step-response analysis
│   ├── find_tuning.py                 # Controller tuning search and optimization
│   ├── find_eqpt.py                   # Operating-point optimization
│   ├── mermaid_diagram.py             # Diagram generation utilities
│   ├── mermaid_renderer.py            # Diagram rendering and SVG formatting
│   ├── plotutils.py                   # Plotting and visualization helpers
│   ├── emf_patch.py                   # Matplotlib enhancement patches
│   └── ctrlparams.md                  # Controller parameter documentation
│
├── simulation/                         # Executable simulation scenarios
│   ├── __init__.py
│   ├── open_loop_simulation.py        # Open-loop step response testing
│   ├── closed_loop_simulation.py      # Closed-loop servo and regulatory tests
│   ├── dynamic_simulation.py          # General-purpose dynamic simulation
│   └── biodiesel_operating_point.py   # Operating-point analysis and search
│
├── analysis/                           # Post-processing and performance evaluation
│   ├── calculate_performance.py       # Performance metrics and indices calculation
│   ├── generate_plots.py              # Primary result visualization
│   ├── generate_summary_plots.py      # Summary and comparative plots
│   ├── generate_report.py             # Formatted text report generation
│   └── generate_validation_comparison.py  # Validation against reference data
│
├── outputs/                            # Generated artifacts (plots, reports, data)
│   └── [generated by analysis scripts]
│
├── draw.py                            # Diagram creation and rendering script
├── main.py                            # Primary entry point
│
├── .pre-commit-config.yaml            # Pre-commit hooks configuration
├── .python-version                    # Python version specification (3.14)
├── .gitignore                         # Git ignore rules
├── package.json                       # Node.js dev dependencies (Mermaid CLI)
├── package-lock.json                  # Node.js dependency lock file
├── pyproject.toml                     # Python project configuration
├── uv.lock                            # UV package manager lock file
├── README.md                          # This file
└── LICENSE                            # [Project license, if applicable]
```

### 2.2 Component Description

#### Model Layer (`model/`)

The model layer encapsulates all mathematical representations and process physics:

- **`plant.py`**: Implements the biodiesel reactor model including:
  - Mass and energy balances
  - Reaction kinetics
  - Heat transfer dynamics
  - State-space equations
  
- **`config.py`**: Centralized storage of:
  - Process parameters (volumes, time constants, gains)
  - Actuator specifications (valve sizing, response times)
  - Sensor/transmitter calibrations
  - Nominal operating conditions and initial states
  
- **`ctrlsys.py`** and **`ctrlbase.py`**: Controller implementations featuring:
  - PID algorithm with anti-windup and derivative filtering
  - Proportional-only control
  - Tuning parameter management
  
- **`simsys.py`**: Simulation engine providing:
  - State-space integration (RK45 or similar method)
  - Event handling and discrete time stepping
  - Signal mixing and result aggregation
  
- **`stepinfo.py`** and **`fopdt.py`**: Analysis utilities for:
  - Calculating rise time, settling time, overshoot, decay ratio
  - Fitting first-order-plus-dead-time models to step responses
  - FOPDT parameter extraction

#### Simulation Layer (`simulation/`)

The simulation layer provides executable workflows:

- **`open_loop_simulation.py`**: Applies step inputs to plant manipulated variables without feedback control, observing transient responses.

- **`closed_loop_simulation.py`**: Executes feedback control with setpoint changes (servo tests) and disturbances (regulatory tests).

- **`biodiesel_operating_point.py`**: Performs steady-state analysis to identify optimal operating conditions.

- **`dynamic_simulation.py`**: General-purpose scenario runner for custom simulation experiments.

#### Analysis Layer (`analysis/`)

Post-processing scripts for metrics and visualization:

- **`calculate_performance.py`**: Computes:
  - Decay ratio, rise time, settling time, overshoot from closed-loop step responses
  - Integral absolute error (IAE)
  - Cross-scenario comparisons
  
- **`generate_plots.py`** and **`generate_summary_plots.py`**: Create:
  - Time-domain response plots
  - Comparison plots between tuning strategies
  - Frequency-domain Bode/Nyquist plots (if applicable)
  
- **`generate_report.py`**: Produces a formatted ASCII report with performance indices organized by controller and scenario.

- **`generate_validation_comparison.py`**: Compares simulation results against experimental data or HYSYS reference cases.

## 3. Theory and Methods

### 3.1 Process Model

The biodiesel reactor is modeled as a nonlinear dynamic system with states representing:

- Liquid level (h) [m]
- Temperature (T) [K]
- Component concentrations (mass fractions) [kg/kg]
- Reactor pressure (P) [Pa]

The model incorporates:
- Material balances for key components
- Energy balance including heat loss and reaction heat
- Mass transfer and reaction kinetics
- Holdup dynamics

### 3.2 Control Loop Architecture

The control system consists of multiple feedback loops:

1. **Temperature Control (TIC-100)**: Uses cooling water flow to maintain reactor temperature at setpoint
2. **Level Control (LIC-100)**: Adjusts FAME product removal to maintain liquid level
3. **Flow Control (FIC-100, FIC-101, FIC-102)**: Regulates inlet flow rates for oil, methanol, and sodium hydroxide

Each loop employs PID control with tuning parameters determined via:
- Direct Synthesis method
- Ziegler-Nichols method
- Model-based optimization

### 3.3 Performance Metrics

Standard performance indices used for evaluation:

- **Decay Ratio (DR)**: Ratio of successive peaks in underdamped step response (DR ≈ 0.25 for quarter-amplitude damping)
- **Rise Time (tr)**: Time for response to reach 90% of steady-state value
- **Settling Time (ts)**: Time for response to remain within ±2% of steady-state value
- **Overshoot (OS)**: Peak value minus steady-state value, expressed as percentage
- **Integral Absolute Error (IAE)**: ∫|e(t)|dt, integral of absolute setpoint deviation

### 3.4 Simulation Methodology

#### Open-Loop Tests
Step inputs are applied to manipulated variables without feedback control. Response characteristics are extracted using FOPDT fitting or step-response analysis.

#### Closed-Loop Tests

**Servo Response (Setpoint Tracking)**:
- Setpoint is stepped at t = 0 or later
- Response is measured relative to the setpoint change
- Metrics quantify tracking performance

**Regulatory Response (Disturbance Rejection)**:
- Disturbances (e.g., feed composition, ambient temperature) are stepped
- Response is measured relative to the disturbance
- Metrics quantify ability to reject external disturbances

## 4. Installation and Setup

### 4.1 System Requirements

- **Python**: 3.14 or higher
- **Operating System**: Linux, macOS, or Windows
- **Memory**: ≥ 4 GB RAM (for large simulations)
- **Disk Space**: ≥ 500 MB for repository + outputs

### 4.2 Dependencies

#### Core Python Packages

Specified in `pyproject.toml`:

```toml
dependencies = [
    "control>=0.10.2",          # Control systems library
    "matplotlib>=3.11.1",       # Plotting and visualization
    "numpy>=2.5.1",             # Numerical computing
    "pandas>=3.0.3",            # Data manipulation
    "scipy>=1.18.0",            # Scientific computing
]
```

#### Development Dependencies

```toml
[dependency-groups]
dev = [
    "pyright>=1.1.401",         # Static type checking
    "ruff>=0.11.13",            # Linting and formatting
    "pre-commit>=4.2.0",        # Git pre-commit hooks
]
```

#### Optional: Diagram Rendering

For Mermaid diagram rendering:

```bash
npm install
```

### 4.3 Installation Steps

#### Using UV (Recommended)

```bash
# Clone repository
git clone https://github.com/mhafzulhaikal/modeling-simulation-analysis.git
cd modeling-simulation-analysis

# Sync dependencies with dev group
uv sync --group dev

# Activate virtual environment
source .venv/bin/activate  # On Linux/macOS
# or
.venv\Scripts\activate     # On Windows
```

#### Using pip

```bash
# Clone repository
git clone https://github.com/mhafzulhaikal/modeling-simulation-analysis.git
cd modeling-simulation-analysis

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Linux/macOS

# Install package and dependencies
pip install -e ".[dev]"
```

### 4.4 Configuration

Before running simulations, review and adjust process parameters in `model/config.py`:

- Process parameters (reactor volume, time constants, etc.)
- Actuator specifications (valve sizing, response times)
- Sensor/transmitter ranges and time constants
- Initial conditions and nominal operating points
- Controller tuning parameters

## 5. Usage and Workflows

### 5.1 Basic Entry Point

```bash
python main.py
```

Outputs a simple confirmation message. Use specific simulation scripts for actual analysis.

### 5.2 Open-Loop Simulation

Applies step inputs to manipulated variables and measures plant response without feedback control.

```bash
python simulation/open_loop_simulation.py
```

**Configuration** (in `simulation/open_loop_simulation.py`):
- `TIME_END`: Total simulation time [seconds]
- `TIME_STEP`: Output sampling interval [seconds]
- `FIRST_STEP`, `SECOND_STEP`: Timing of step changes
- `M_profiles`: Controller output (valve opening) profiles [%CO]

**Output**:
- Console display of initial/final state values
- Plots of key variables (e.g., temperature response)

### 5.3 Closed-Loop Simulation

Executes feedback control with setpoint changes and disturbances.

```bash
python simulation/closed_loop_simulation.py
```

**Test Scenarios**:
- Servo tests (setpoint tracking): Step change in setpoint at t = 0
- Regulatory tests (disturbance rejection): Step change in disturbance variable
- Combined tests: Both servo and regulatory components

**Output**:
- Closed-loop responses saved to results dictionary
- Transient response characteristics (rise time, overshoot, settling time)

### 5.4 Operating-Point Analysis

Identifies optimal steady-state operating conditions.

```bash
python simulation/biodiesel_operating_point.py
```

**Analysis**:
- Evaluates process performance across operating parameter ranges
- Identifies constraints and trade-offs
- Recommends optimal setpoints

### 5.5 Dynamic Simulation

General-purpose runner for custom scenarios.

```bash
python simulation/dynamic_simulation.py
```

Modify this script to implement custom simulation logic and parameter sweeps.

### 5.6 Performance Calculation

Computes performance metrics for closed-loop responses.

```bash
python analysis/calculate_performance.py
```

**Metrics Calculated**:
- Decay ratio
- Rise time
- Settling time
- Overshoot
- Integral absolute error (IAE)

**Output Format**:
- Structured dictionary or DataFrame with metrics organized by:
  - Controller name
  - Test scenario (servo vs. regulatory)
  - Tuning method

### 5.7 Report Generation

Produces a formatted summary report of all performance metrics.

```bash
python analysis/generate_report.py
```

**Output**:
- Text file: `outputs/reports/report_summary.txt`
- Console display of the report

**Content**:
- TIC-100 (Temperature Controller) performance
- LIC-100 (Level Controller) performance
- FIC-100/101/102 (Flow Controller) performance
- Comparison across tuning strategies (Synthesis, QDR, IAE, etc.)

### 5.8 Plot Generation

Creates visualization artifacts.

```bash
python analysis/generate_plots.py
python analysis/generate_summary_plots.py
```

**Outputs**:
- Time-domain response plots (saved to `outputs/plots/`)
- Comparative plots across scenarios
- Performance index bar charts

### 5.9 Validation Comparison

Compares simulation results against reference data.

```bash
python analysis/generate_validation_comparison.py
```

**Comparison Types**:
- Python simulation vs. HYSYS reference model
- Different tuning methods
- Servo vs. regulatory performance

## 6. Typical Workflow for Analysis

### Phase 1: Model Preparation

1. Update process parameters in `model/config.py` based on experimental data or design specifications
2. Validate open-loop step responses against known plant behavior
3. Adjust actuator and sensor models if necessary

### Phase 2: Controller Design

1. Compute nominal PID tuning parameters using `model/find_tuning.py`
2. Define alternative tuning strategies in the configuration
3. Run `closed_loop_simulation.py` with each tuning set

### Phase 3: Performance Evaluation

1. Execute `analysis/calculate_performance.py` to extract metrics
2. Run `analysis/generate_report.py` for summary report
3. Generate plots using `analysis/generate_plots.py`

### Phase 4: Optimization (if required)

1. Perform parameter sweeps across controller tuning ranges
2. Use `model/find_eqpt.py` to identify optimal operating points
3. Compare results across scenarios

### Phase 5: Validation

1. If experimental or reference data is available, run `analysis/generate_validation_comparison.py`
2. Identify discrepancies and refine model if needed
3. Document findings and recommendations

## 7. Key Features and Capabilities

### 7.1 Simulation Features

- ✓ Multi-loop feedback control
- ✓ Configurable PID tuning parameters
- ✓ Realistic actuator and sensor dynamics
- ✓ Open-loop and closed-loop scenarios
- ✓ Servo and regulatory test modes
- ✓ Step, ramp, and custom input profiles
- ✓ Multiple disturbance sources

### 7.2 Analysis Capabilities

- ✓ Transient response characterization (rise time, settling time, overshoot)
- ✓ Frequency-domain metrics (decay ratio from time-domain response)
- ✓ Integral performance indices (IAE, ISE, ITAE)
- ✓ Multi-case comparison and aggregation
- ✓ Automated report generation
- ✓ Publication-quality plots

### 7.3 Development Features

- ✓ Type-checked Python code (Pyright)
- ✓ Automated code linting and formatting (Ruff)
- ✓ Pre-commit hooks for code quality
- ✓ Modular architecture with clear separation of concerns
- ✓ Extensive docstrings and inline documentation

## 8. Example: Temperature Controller (TIC-100) Tuning Evaluation

### Scenario

Evaluate two PID tuning methods for the temperature controller:

1. **Synthesis (Ser)**: Model-based direct synthesis method
2. **Quadratic Decay Ratio (QDR)**: Classical frequency-domain method

### Steps

1. **Configure tuning parameters** in `model/config.py`:
   ```python
   TUNING_TIC100 = {
       'Syn': {'Kc': 0.85, 'tau_I': 180, 'tau_D': 45},
       'QDR': {'Kc': 0.72, 'tau_I': 210, 'tau_D': 52},
   }
   ```

2. **Run closed-loop simulation**:
   ```bash
   python simulation/closed_loop_simulation.py
   ```
   
   Specify setpoint step from 60°C → 65°C at t = 100 s
   Specify disturbance: 5% step change in cooling water supply temperature at t = 500 s

3. **Calculate performance**:
   ```bash
   python analysis/calculate_performance.py
   ```
   
   Output example:
   ```
   TIC-100 Servo (Setpoint Tracking)
   ─────────────────────────────────────────
   Metric                  Synthesis    QDR
   Rise Time [s]           42.3        51.2
   Settling Time [s]       118.5       142.3
   Overshoot [%]           8.2         4.1
   Decay Ratio             0.31        0.24
   IAE                     215.4       198.7
   ```

4. **Generate report and plots**:
   ```bash
   python analysis/generate_report.py
   python analysis/generate_plots.py
   ```

## 9. Documentation and References

### Code Documentation

- Inline docstrings follow NumPy/SciPy documentation style
- Key functions and classes are documented with parameter descriptions and usage examples
- Module-level docstrings provide overview and entry points

### Model Documentation

- `model/ctrlparams.md`: Controller parameter specifications and tuning methods
- Config file comments in `model/config.py` explain each parameter

### External References

- **Process Control Theory**: Smith, Corripio. "Principles and Practice of Automatic Process Control" (2nd ed.)
- **Python Control Library**: https://python-control.readthedocs.io/
- **Tuning Methods**: Åström & Hägglund. "Advanced PID Control" (2006)

## 10. Limitations and Future Work

### Current Limitations

1. **Model Fidelity**: FOPDT approximations may not capture all nonlinear dynamics
2. **Sensor Noise**: Simplified sensor models do not include measurement noise
3. **Constraint Handling**: Valve saturation and rate limits are basic implementations
4. **Multivariate Control**: Limited support for multivariable MIMO control

### Future Enhancements

1. Implement advanced control strategies (feedforward, cascade, adaptive)
2. Add measurement noise and filter design optimization
3. Support for multivariable (MIMO) systems
4. Machine-learning-based tuning parameter prediction
5. Real-time visualization dashboard
6. Integration with industrial simulation software (HYSYS, Aspen Plus)

## 11. Troubleshooting

### Common Issues

| Issue | Cause | Resolution |
|-------|-------|-----------|
| Import errors | Missing dependencies | Run `uv sync` or `pip install -e .` |
| Plot not displayed | Matplotlib backend issue | Check display settings; use `plt.savefig()` |
| Simulation diverges | Unstable controller tuning | Reduce controller gains or increase sample time |
| Performance metrics NaN | Degenerate response | Check initial conditions and disturbance magnitudes |

### Debug Mode

Add verbose logging to simulation scripts:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 12. Contributing and Development

### Code Standards

- Follow PEP 8 style guide (enforced by Ruff)
- Use type hints for function signatures
- Write docstrings for all public functions and classes
- Run pre-commit hooks before committing:
  ```bash
  pre-commit run --all-files
  ```

### Testing

While formal unit tests are not included, validation can be performed by:

1. Running example scenarios and comparing against known results
2. Performing sensitivity analysis on key parameters
3. Cross-validating against reference simulations

## 13. License and Citation

[Include appropriate license information]

If using this framework in academic work, please cite:

```
[Citation format to be determined]
```

## 14. Contact and Support

**Author/Maintainer**: Muhammad Hafzul Haikal  
**Email**: mhafzulhaikal@gmail.com  
**Repository**: https://github.com/mhafzulhaikal/modeling-simulation-analysis

For questions, bug reports, or suggestions, please open an issue on the GitHub repository.

---

**Last Updated**: October 2026  
**Version**: 0.1.0  
**Status**: Active Development
