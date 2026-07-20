"""PID controller tuning reference formulas.

Summary of classical tuning correlations from Corripio & Smith (2006),
Chapter 7.  All formulas use FOPDT model parameters:

    K   = process gain
    tau = process time constant  [s]
    t0  = process dead time      [s]

Nomenclature
------------
    Kc    Proportional gain
    tauI  Integral time   [s]
    tauD  Derivative time [s]
    tauC  Closed-loop time constant (Dahlin synthesis) [s]


1  QUARTER DECAY RATIO (Table 7-2.1)
================================================================================

    Type     Kc                          tauI          tauD
    ----     --------------------------  ------------  ----------
    P        1/K * (t0/tau)**(-1)        -             -
    PI       0.9/K * (t0/tau)**(-1)      3.33 * t0     -
    PID(a)   1.2/K * (t0/tau)**(-1)      2.0 * t0      t0 / 2.0

    (a) PID formulas are for the actual (series) controller, Eq. 5-3.19.
        To convert to the ideal (parallel) PID, Eq. 5-3.17:

            Kc   = Kc' * (1 + tauD'/tauI')
            tauI = tauI' + tauD'
            tauD = tauD' * tauI' / (tauI' + tauD')


2  MINIMUM IAE — DISTURBANCE INPUTS (Table 7-2.2)
================================================================================

    Type   Kc                             tauI                               tauD
    ----   ------------------------------  ---------------------------------  --------------------------------
    P      0.902/K * (t0/tau)**(-0.985)    -                                  -
    PI     0.994/K * (t0/tau)**(-0.986)    tau/0.608 * (t0/tau)**0.707        -
    PID    1.435/K * (t0/tau)**(-0.921)    tau/0.878 * (t0/tau)**0.749        0.482*tau * (t0/tau)**1.137


3  MINIMUM IAE — SET POINT CHANGES (Table 7-2.3)
================================================================================

    Type   Kc                             tauI                                tauD
    ----   ------------------------------  ----------------------------------  --------------------------------
    PI     0.758/K * (t0/tau)**(-0.861)    tau / (1.02 - 0.323*(t0/tau))      -
    PID    1.086/K * (t0/tau)**(-0.869)    tau / (0.740 - 0.130*(t0/tau))     0.348*tau * (t0/tau)**0.914


4  DAHLIN SYNTHESIS
================================================================================

    Process model                              Type    Kc                     tauI    tauD
    -----------------------------------------  ------  ---------------------  ------  ------
    G(s) = K                                   I       1 / (K * tauC)         -       -
    G(s) = K / (tau*s + 1)                     PI      tau / (K * tauC)       tau     -
    G(s) = K / [(tau1*s+1)(tau2*s+1)]          PID     tau1 / (K * tauC)      tau1    tau2
    G(s) = K * exp(-t0*s) / (tau*s + 1)        PID(a)  tau / [K*(t0+tauC)]    tau     t0/2
    G(s) = K / s                               P       1 / (K * tauC)         -       -

    Minimum IAE closed-loop time constant (tauC):
        Disturbance inputs (approx. minimum IAE):
            tauC = 0    for t0/tau in [0.1, 0.5] (PI) or [0.1, 1.5] (PID)
        Set point changes (approx. minimum IAE, t0/tau in [0.1, 1.5]):
            PI  controller (tauD = 0) :  tauC = 2/3 * t0
            PID controller            :  tauC = 1/5 * t0


5  5 % OVERSHOOT CRITERION
================================================================================

    Kc = 0.5 / K * (t0 / tau)**(-1)


References
----------
1. Corripio, A. & Smith, C. A. (2006). Principles and Practice of
   Automatic Process Control (Third Edition). John Wiley & Sons, Inc.
"""
