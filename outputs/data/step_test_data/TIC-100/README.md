### **TIC-100**

##### **INPUT/DISTURBANCE**

Disturbance input (Oil stream temperature)
0s -> 600s = 333.15 K
600s -> 40000 = 343.15 K

Set-point input (TIC-100 - SP)
0s -> 20600 = 333.15 K
20600 -> 40000 = 340.15 K

\----------------------------------------------------------------------------

##### **SYSTEM IDENTIFICATION (FOPDT)**

Step detected: up by 10.00 at t=599.50s
Initial guess (two-point): K=0.8908, τ=4051.98s, t0=1194.60s, t0/τ=0.295
Optimization result: K=0.8932, τ=4117.77s, t0=535.03s
Cost: 9.072281e+01

FOPDT Model: K=0.8932, τ=4117.77s, t0=535.03s, t0/τ=0.130
Fit Quality - R²: 0.9997, RMSE: 0.0393

K=0.8932, τ=4117.77s, t0=535.03s
R²=0.9997, RMSE=0.0393

\----------------------------------------------------------------------------

##### **CONTROLLER PARAMETERS**

Kc Syn - Setpoint   : 7.18
tauI Syn - Setpoint : 4117.77 seconds
tauD Syn - Setpoint : 267.52 seconds

Kc Syn - Disturbance : 8.62
tauI Syn - Disturbance : 4117.77 seconds
tauD Syn - Disturbance : 267.52 seconds

Kc QDR : 10.34
tauI QDR : 1070.07 seconds
tauD QDR : 267.52 seconds

Kc IAE - Setpoint (Ideal) : 7.16
tauI IAE - Setpoint (Ideal) : 5694.54 seconds
tauD IAE - Setpoint (Ideal) : 221.91 seconds

Kc IAE - Disturbance (Ideal) : 10.52
tauI IAE - Disturbance (Ideal) : 1017.05 seconds
tauD IAE - Disturbance (Ideal) : 194.99 seconds

Kc IAE - Setpoint : 6.87
tauI IAE - Setpoint : 5463.23 seconds
tauD IAE - Setpoint : 231.31 seconds

Kc IAE - Disturbance : 7.80
tauI IAE - Disturbance : 754.06 seconds
tauD IAE - Disturbance : 262.99 seconds
