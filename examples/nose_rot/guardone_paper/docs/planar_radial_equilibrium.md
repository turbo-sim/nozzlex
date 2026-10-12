# Reproducing Gajoni & Guardone (JFM 2023)

This folder reproduces selected results from Gajoni & Guardone, "Ideal
and non-ideal planar compressible fluid flows in radial equilibrium"
(JFM 2023) — the theory that motivated the JFM extension work in
`nose_rot`. Purpose: build confidence in the governing equations and the
numerical machinery (root-finding + ODE integration with a real
equation of state) before extending them to a real duct geometry.

## The physical setup

Two-dimensional flow confined to a single flat plane, purely tangential
(no axial component): fluid travels in circles around a central axis,
with all streamlines sharing the same total enthalpy $h_t$ and entropy
$s$ (uniform upstream conditions, adiabatic, isentropic). The radial
direction $r$ is transverse to the flow, from an inner wall $r_i$ to an
outer wall $r_e$. Because the flow curves, it needs a radial pressure
gradient purely from that curvature (no swirl or blades involved):

$$
\frac{1}{\rho}\frac{dP}{dr} = \frac{u_\theta^2}{r}
$$

## Core equations

With $\tilde r = r/r_i$ and $M$ the local Mach number, combining the
momentum balance above with the energy and entropy constraints and a
general equation of state gives (Eq. 2.9 in the paper):

$$
\frac{d\rho}{d\tilde r} = \frac{\rho\, M(\rho)^2}{\tilde r}
$$

where, at fixed $h_t$ and $s$ (Eq. 2.10):

$$
M(\rho) = \frac{\sqrt{2\left(h_t - h(\rho,s)\right)}}{c(\rho,s)}
$$

Density (not Mach number) is the natural integration variable because
$d\rho/d\tilde r > 0$ always — density increases monotonically outward
regardless of the fluid — while $M(\tilde r)$ can be non-monotone for
non-ideal fluids.

**Fundamental derivative of gas dynamics** $\Gamma$ controls the
qualitative behavior:

$$
\Gamma = 1 + \frac{\rho}{c}\left(\frac{\partial c}{\partial \rho}\right)_s
$$

and the sign of

$$
J = 1 - \Gamma - \frac{1}{M^2}
$$

decides whether $M(\tilde r)$ can double back on itself: $J>0$ is
required for non-monotone behavior. Since $1/M^2 > 1$ whenever $M<1$,
$J>0$ subsonically requires $\Gamma<0$ (a BZT fluid); supersonically
($M>1$), $0<\Gamma<1$ (an HMC fluid) is enough.

**Ideal-gas special case:** for a calorically-ideal gas ($\Gamma =
(\gamma+1)/2 = $ const), Eq. (2.9) integrates in closed form (Eq. 2.8):

$$
M(\tilde r) = \frac{M_i}{\sqrt{\left(1+\tfrac{\gamma-1}{2}M_i^2\right)\tilde r^2 - \tfrac{\gamma-1}{2}M_i^2}}
$$

For a real fluid (multiparameter Helmholtz-energy equation of state),
$M(\rho)$ has no closed form — evaluating it requires numerically
inverting the entropy equation for temperature at each $\rho$ — so
Eq. (2.9) is integrated numerically (RK45) instead.

## What each script reproduces

| Script | Fluid | Case | Paper figure |
|---|---|---|---|
| `figure2_ideal_gas.py` | N2, CO2, MM (ideal-gas) | Eq. (2.8), $M_i=0.5,1,1.5,2$ | Fig. 2 |
| `figure4_co2_nonideal.py` | CO2 | $P_t/P_c=0.5$ and $2$, $M_i=0.5$ | Fig. 4 |
| `figure6a_mm_ideal_conditions.py` | MM | $P_t/P_c=0.5$, $M_i=1.75$ (J<0 throughout — monotone) | Fig. 6a |
| `figure7a_mm_nonideal.py` | MM | $P_t/P_c=2$, $M_i=1.75$ (J>0 for low $T_t/T_c$ — non-monotone) | Fig. 7a |
| `figure8a_d6_bzt.py` | D6 (BZT) | $P_t/P_c=1.1171$, $M_i=1.2$ (subsonic non-monotone, $\Gamma<0$) | Fig. 8a |

Shared numerical machinery (root-finding for $\rho_i$ given $M_i$, RK45
integration of Eq. 2.9, the ideal-gas comparison model) lives in
`common.py` and is imported by all five scripts.
