# PID Controller Tuning Reference

Classical tuning correlations from **Corripio & Smith (2006), Chapter 7**.
All formulas are based on the **First-Order Plus Dead Time (FOPDT)** model:

$$
G(s) = \frac{K \, e^{-t_0 s}}{\tau s + 1}
$$

| Symbol | Description | Unit |
|--------|-------------|------|
| $K$ | Process gain | — |
| $\tau$ | Process time constant | s |
| $t_0$ | Process dead time | s |
| $K_c$ | Controller proportional gain | — |
| $\tau_I$ | Integral time | s |
| $\tau_D$ | Derivative time | s |
| $\tau_C$ | Closed-loop time constant (Dahlin) | s |

---

## 1. Quarter Decay Ratio

*Table 7-2.1*

| Type | $K_c$ | $\tau_I$ | $\tau_D$ |
|------|--------|----------|----------|
| P | $\dfrac{1.000}{K}\left(\dfrac{t_0}{\tau}\right)^{-1}$ | — | — |
| PI | $\dfrac{0.900}{K}\left(\dfrac{t_0}{\tau}\right)^{-1}$ | $3.33\, t_0$ | — |
| PID $^{(a)}$ | $\dfrac{1.200}{K}\left(\dfrac{t_0}{\tau}\right)^{-1}$ | $2.00\, t_0$ | $\dfrac{t_0}{2}$ |

> **$^{(a)}$** PID formulas apply to the **series (actual)** form, Eq. 5-3.19.
> To convert to the **parallel (ideal)** form, Eq. 5-3.17:
>
> $$K_c = K_c'\!\left(1 + \frac{\tau_D'}{\tau_I'}\right), \qquad \tau_I = \tau_I' + \tau_D', \qquad \tau_D = \frac{\tau_D'\,\tau_I'}{\tau_I' + \tau_D'}$$

---

## 2. Minimum IAE — Disturbance Inputs

*Table 7-2.2*

| Type | $K_c$ | $\tau_I$ | $\tau_D$ |
|------|--------|----------|----------|
| P | $\dfrac{0.902}{K}\left(\dfrac{t_0}{\tau}\right)^{-0.985}$ | — | — |
| PI | $\dfrac{0.994}{K}\left(\dfrac{t_0}{\tau}\right)^{-0.986}$ | $\dfrac{\tau}{0.608}\left(\dfrac{t_0}{\tau}\right)^{0.707}$ | — |
| PID | $\dfrac{1.435}{K}\left(\dfrac{t_0}{\tau}\right)^{-0.921}$ | $\dfrac{\tau}{0.878}\left(\dfrac{t_0}{\tau}\right)^{0.749}$ | $0.482\,\tau\left(\dfrac{t_0}{\tau}\right)^{1.137}$ |

---

## 3. Minimum IAE — Set Point Changes

*Table 7-2.3*

| Type | $K_c$ | $\tau_I$ | $\tau_D$ |
|------|--------|----------|----------|
| PI | $\dfrac{0.758}{K}\left(\dfrac{t_0}{\tau}\right)^{-0.861}$ | $\dfrac{\tau}{1.020 - 0.323\,(t_0/\tau)}$ | — |
| PID | $\dfrac{1.086}{K}\left(\dfrac{t_0}{\tau}\right)^{-0.869}$ | $\dfrac{\tau}{0.740 - 0.130\,(t_0/\tau)}$ | $0.348\,\tau\left(\dfrac{t_0}{\tau}\right)^{0.914}$ |

---

## 4. Dahlin Synthesis

| Process model | Type | $K_c$ | $\tau_I$ | $\tau_D$ |
|---------------|------|--------|----------|----------|
| $G(s) = K$ | I | $\dfrac{1}{K\,\tau_C}$ | — | — |
| $G(s) = \dfrac{K}{\tau s+1}$ | PI | $\dfrac{\tau}{K\,\tau_C}$ | $\tau$ | — |
| $G(s) = \dfrac{K}{(\tau_1 s+1)(\tau_2 s+1)}$ | PID | $\dfrac{\tau_1}{K\,\tau_C}$ | $\tau_1$ | $\tau_2$ |
| $G(s) = \dfrac{K\,e^{-t_0 s}}{\tau s+1}$ | PID $^{(a)}$ | $\dfrac{\tau}{K\,(t_0+\tau_C)}$ | $\tau$ | $\dfrac{t_0}{2}$ |
| $G(s) = \dfrac{K}{s}$ | P | $\dfrac{1}{K\,\tau_C}$ | — | — |

### Recommended $\tau_C$ for minimum IAE

**Disturbance inputs** — use $\tau_C \approx 0$:

| Controller | Valid range of $t_0/\tau$ |
|------------|--------------------------|
| PI | $[0.1,\ 0.5]$ |
| PID | $[0.1,\ 1.5]$ |

**Set point changes** — valid for $t_0/\tau \in [0.1,\ 1.5]$:

$$\tau_C^{\text{PI}} = \tfrac{2}{3}\,t_0 \qquad \tau_C^{\text{PID}} = \tfrac{1}{5}\,t_0$$

---

## 5. Five-Percent Overshoot Criterion

P controller only:

$$K_c = \frac{0.5}{K}\left(\frac{t_0}{\tau}\right)^{-1}$$

---

## References

1. Corripio, A. & Smith, C. A. (2006). *Principles and Practice of Automatic Process Control* (3rd ed.). John Wiley & Sons.
