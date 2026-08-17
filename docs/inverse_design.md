# 1D Inverse Nozzle Design

This describes the inverse-design workflow implemented in `nozzlex/inverse_design/`
and driven by `examples/inverse_design/run_design.py`.

## Problem statement

Given a target axial pressure distribution $p(x)$ (normalized $x/L \in [0,1]$)
and inlet conditions, recover the flow-passage area $A(x)$ that produces it in a
steady, quasi-1D, adiabatic, frictionless duct with real-gas thermodynamics.
This is the inverse of the usual nozzle problem: instead of solving for $p(x)$
given $A(x)$, the area is the unknown.

## 1. Pressure profile generation (NURBS)

For each case, the imposed pressure profile is built as a NURBS curve through
five control points in $(x/L, p)$ space:

$$
P_0 = (0,\ p_\text{in}), \quad
P_1 = (0.5,\ p_\text{in}), \quad
P_2 = (0.5,\ p_\text{mid}), \quad
P_3 = (0.5,\ p_\text{out}), \quad
P_4 = (1,\ p_\text{out})
$$

with

$$
p_\text{mid} = \tfrac{1}{2}\left(p_{0,\text{total}} + p_\text{out}\right).
$$

Stacking three control points at $x/L = 0.5$ concentrates the curvature (and
therefore the expansion) around the mid-length station. The curve uses an
open-uniform knot vector, degree 4, unit weights, evaluated via the
Cox–de Boor recursion for the B-spline basis functions $N_{i,p}(u)$:

$$
N_{i,0}(u) =
\begin{cases}
1 & \text{if } u_i \le u < u_{i+1} \\
0 & \text{otherwise}
\end{cases}
$$

$$
N_{i,p}(u) = \frac{u - u_i}{u_{i+p} - u_i} N_{i,p-1}(u)
           + \frac{u_{i+p+1} - u}{u_{i+p+1} - u_{i+1}} N_{i+1,p-1}(u)
$$

and the NURBS curve

$$
C(u) = \frac{\sum_i N_{i,p}(u)\, w_i\, P_i}{\sum_i N_{i,p}(u)\, w_i}.
$$

The curve is resampled onto a uniform axial grid via linear interpolation to
obtain $p(x)$.

## 2. Area recovery (real-gas ODE marching)

The area is recovered by integrating the quasi-1D governing equations
(continuity, momentum, energy) as a first-order ODE system in the state
vector $\mathbf{y} = [u,\, h,\, A]$, with the imposed $p(x)$ and $dp/dx$
(both cubic-interpolated) as forcing terms, for a duct with wall friction.

Continuity is unaffected by friction. Momentum picks up a wall-shear source
term, and since the duct is still adiabatic, stagnation enthalpy is still
conserved ($dh/dx = -u\,du/dx$ holds regardless of friction — the same
reasoning behind Fanno flow), which carries the same source term into the
energy equation with the opposite sign:

$$
\frac{d(\rho u A)}{dx} = 0, \qquad
\rho u \frac{du}{dx} = -\frac{dp}{dx} - \frac{\Pi}{A}\,\tau_w, \qquad
\frac{dh}{dx} = -u\,\frac{du}{dx}.
$$

Here $\Pi$ is the wetted perimeter and $\tau_w$ the wall shear stress,
computed from the Darcy friction factor $f$,

$$
\tau_w = \frac{1}{8} f \rho u^2,
$$

with $f$ obtained from the Haaland correlation as an explicit approximation
to Colebrook,

$$
\frac{1}{\sqrt{f}} = -1.8 \log_{10}\!\left(\frac{6.9}{Re} +
\left(\frac{\varepsilon}{3.7\, D_h}\right)^{1.11}\right), \qquad
Re = \frac{\rho u D_h}{\mu},
$$

where $\varepsilon$ is the wall roughness, $\mu$ the local viscosity, and
$D_h = 4A/\Pi$ the hydraulic diameter. Setting $\tau_w = 0$ (equivalently
$f = 0$) recovers the frictionless limit.

Expanding $d\rho$ in terms of the real-gas EOS derivatives at fixed $p$/$h$,

$$
\left.\frac{\partial \rho}{\partial p}\right|_h = \frac{1+G}{a^2}, \qquad
\left.\frac{\partial \rho}{\partial h}\right|_p = -\frac{\rho G}{a^2},
$$

where $a$ is the local speed of sound and $G$ the Grüneisen parameter, the
system can be written as a linear system for $\big[du/dx,\ dh/dx,\ dA/dx\big]$:

$$
\begin{bmatrix}
A\rho & u A \left.\dfrac{\partial \rho}{\partial h}\right|_p & \rho u \\[4pt]
\rho u & 0 & 0 \\[4pt]
0 & \rho & 0
\end{bmatrix}
\begin{bmatrix}
du/dx \\ dh/dx \\ dA/dx
\end{bmatrix}
=
\begin{bmatrix}
-u A \left.\dfrac{\partial \rho}{\partial p}\right|_h \dfrac{dp}{dx} \\[6pt]
-\dfrac{dp}{dx} - \dfrac{\Pi}{A}\,\tau_w \\[4pt]
\dfrac{dp}{dx} + \dfrac{\Pi}{A}\,\tau_w
\end{bmatrix}.
$$

Only the momentum- and energy-row right-hand sides carry the friction term;
the coefficient matrix and the continuity row are unchanged from the
frictionless case.

This system is solved at every station of a Runge–Kutta 4(5) integration
(`solve_ivp`, RK45), starting from the inlet state $(u_\text{in}, h_\text{in},
A_\text{in})$.

Note: $D_h$ and $\Pi$ require a duct cross-section shape, which the rest of
this formulation deliberately avoids (area is reported only as the
dimensionless $A/A^*$, with no assumed aspect ratio). Using the
friction term in practice needs either a local shape closure just for
$\Pi(x)$ (e.g. an equivalent circular or rectangular section from $A(x)$), or
an externally supplied $D_h(x)$/$\Pi(x)$.

## 2b. Frictionless shortcut: closed-form isentropic area recovery

In the frictionless limit ($\tau_w = 0$), the flow is isentropic, and $A(x)$
can be recovered directly from $p(x)$ point-by-point — no ODE integration
needed. With $s(x) = s_\text{in}$ fixed, the local state follows from an
equation of state evaluated at $(p(x), s_\text{in})$:

$$
h(x),\ \rho(x),\ a(x) = \mathrm{EOS}\big(p(x),\, s_\text{in}\big).
$$

Velocity follows from conservation of stagnation enthalpy (energy),

$$
u(x) = \sqrt{2\big(h_0 - h(x)\big)}, \qquad h_0 = h_\text{in} + \tfrac{1}{2} u_\text{in}^2,
$$

and the area from conservation of mass,

$$
A(x) = \frac{\dot m}{\rho(x)\, u(x)}, \qquad \dot m = \rho_\text{in} A_\text{in} u_\text{in}.
$$

This is exactly the frictionless limit of §2's ODE system, solved in closed
form instead of by marching — useful as a fast, independent cross-check
against `march_from_p_ode` whenever friction is switched off.

### Why this is algebraic and not an ODE

The classic quasi-1D area-velocity relation, combined with the frictionless
momentum equation $\rho u\, du = -dp$, gives $A(x)$ in differential form too:

$$
\frac{dA}{dx} = A\,\frac{1-M^2}{\rho u^2}\,\frac{dp}{dx}.
$$

This is mathematically equivalent to the algebraic form above, and could be
integrated with an ODE solver — but it would be redundant work. The
right-hand side is $A$ times a factor that only depends on $M(x)$, $\rho(x)$,
$u(x)$, and in the isentropic case *all three of those are already known
algebraically from $p(x)$ alone* (via $s = s_\text{in}$ fixed $\rightarrow$
EOS $\rightarrow$ $h,\rho,a$ $\rightarrow$ energy equation $\rightarrow$ $u$
$\rightarrow$ $M = u/a$), independent of $A$ itself. The ODE is therefore
separable,

$$
\frac{dA}{A} = \frac{1-M(x)^2}{\rho(x)\, u(x)^2}\,dp,
$$

and integrates in closed form to

$$
A(x) = A_\text{in}\,
\exp\!\left(\int_{x_\text{in}}^{x} \frac{1-M^2}{\rho u^2}\,\frac{dp}{dx'}\,dx'\right),
$$

which is exactly the statement of mass conservation, $A = \dot m/(\rho u)$,
used directly above. Integrating the differential form numerically would
just reconstruct that same antiderivative with discretization error, for no
benefit.

The real dividing line is not "ODE vs. algebraic" — it is whether $\rho(x)$
and $u(x)$ can be evaluated *without knowing $A$*. In the frictionless case
they can (isentropic EOS + energy equation alone decouple them from area).
Once friction is present, entropy generation makes $\rho$, $u$, $M$
path-dependent unknowns coupled to $A$ itself, and the algebraic shortcut no
longer applies — at that point $u$, $h$, $A$ (and implicitly $s$) must be
marched together as a genuinely coupled system, which is exactly what §2's
`march_from_p_ode` does.

## 2c. Alternative: discrete marching via root-finding

A third way to recover $A(x)$, avoiding both the continuous ODE of §2 and
the isentropic-only shortcut of §2b, is to step station-to-station and solve
the *discrete* momentum balance directly for the unknown velocity at each
new station.

Between stations $i$ and $i+1$ (spacing need not be small — this is not a
finite-difference approximation of a derivative, but an exact discrete
statement of momentum conservation over the interval), the momentum balance
is

$$
\dot m \big(u_{i+1} - u_i\big) + \bar A \,\big(p_{i+1} - p_i\big) = 0,
\qquad \bar A = \tfrac{1}{2}\big(A_i + A_{i+1}\big),
$$

where $A_{i+1}$ itself depends on $u_{i+1}$ through mass conservation and
the local energy equation:

$$
h_{i+1} = h_0 - \tfrac{1}{2} u_{i+1}^2, \qquad
\rho_{i+1} = \mathrm{EOS}\big(h_{i+1},\, p_{i+1}\big), \qquad
A_{i+1} = \frac{\dot m}{\rho_{i+1}\, u_{i+1}}.
$$

Substituting these into the momentum balance leaves a single nonlinear
equation in the one unknown $u_{i+1}$,

$$
R(u_{i+1}) = \dot m \big(u_{i+1} - u_i\big) +
\tfrac{1}{2}\Big(A_i + \frac{\dot m}{\rho_{i+1}(u_{i+1})\, u_{i+1}}\Big)
\big(p_{i+1} - p_i\big) = 0,
$$

solved at each step with a secant iteration ($R(u_{i+1}) \to 0$), after
which $h_{i+1}$, $\rho_{i+1}$, $A_{i+1}$ follow directly. Marching station by
station from the inlet state reconstructs the full $A(x)$ profile.

This is mathematically equivalent to §2's `march_from_p_ode` (same
conservation laws, no isentropic assumption, friction could be added the
same way as in §2) — the difference is purely numerical: a sequence of
scalar root-finds over discrete steps, instead of integrating a 3-variable
Jacobian ODE.

## 3. Case setup and configuration

Each run is driven by a YAML settings file with a top-level fluid name and
CoolProp backend, and a list of cases. Each case specifies the inlet
stagnation pressure $p_{0,\text{total}}$, inlet vapor quality $Q_0$, outlet
static pressure $p_\text{out}$, the inlet passage opening and width (whose
product gives $A_\text{in}$), and the inlet velocity $u_\text{in}$ (settable
per case, or inherited from a shared default).

The `backend` field matters for fluids whose default backend fails to
compute the critical point — an alternative backend is used as a fallback in
that case.

For each case, the inlet static state follows from $(p_{0,\text{total}},
Q_0)$, de-accelerated to $u_\text{in}$ to obtain the inlet static enthalpy

$$
h_\text{in} = h_{0} - \tfrac{1}{2} u_\text{in}^2,
$$

from which $A_\text{in} = \text{width} \times \text{inlet\_opening}$ closes
the inlet mass-flow state.
