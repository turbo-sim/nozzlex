# Notes: extending inverse design to 2D

This is a conceptual scoping note, not an implemented method. It sketches
what it would take to generalize the 1D inverse-design workflow in
`inverse_design.md` to a genuinely two-dimensional passage, and why that
generalization changes the character of the problem rather than just its
dimensionality.

## Why 1D is special

In the 1D formulation, the cross-sectional area $A(x)$ is a single scalar
unknown at each station, and the governing equations (continuity, momentum,
energy) are ODEs in $x$. That is what makes a direct recovery possible: given
$p(x)$, either march a 3-variable ODE system (§2 of `inverse_design.md`) or,
in the frictionless/isentropic case, evaluate $A(x)$ pointwise from an
algebraic relation (§2b). Either way, the unknown geometry is *one function
of one variable*, and it is recovered in a single forward pass with no
iteration on the geometry itself.

## What changes in 2D

In a genuinely 2D passage, the unknown is no longer a scalar $A(x)$ but a
full wall shape (or, equivalently, a streamline) $y_\text{wall}(x)$, and the
flow field $\big(u(x,y), v(x,y), p(x,y)\big)$ is governed by a PDE — the
steady 2D Euler equations (or their potential-flow subset for irrotational
flow):

$$
\nabla\!\cdot(\rho \mathbf{V}) = 0, \qquad
\rho(\mathbf{V}\!\cdot\!\nabla)\mathbf{V} = -\nabla p, \qquad
\mathbf{V}\!\cdot\!\nabla h_0 = 0.
$$

Prescribing a target pressure (now typically along the wall,
$p_\text{wall}(x)$, rather than everywhere) no longer pins down the geometry
by direct substitution the way it did in 1D, because the wall shape and the
2D flow field are coupled through the domain, not just through a local
algebraic/ODE relation. This is the fundamental change: **the problem
becomes PDE-constrained**, not just "one more spatial dimension" grafted onto
the same marching idea.

## Design-variable parametrization

Rather than treating $A(x)$ (or the wall coordinate $y_\text{wall}(x)$)
itself as the controlled unknown, it can be more natural to parametrize its
*slope*, $dA/dx$, as a function on the same normalized unit mesh $x/L \in
[0,1]$ already used for the pressure profile in §1 of `inverse_design.md`
(e.g. via the same NURBS control-point machinery). $A(x)$ then follows by
integrating the slope from the inlet value. This keeps the parametrization
consistent with the rest of the codebase, and — as in the pressure-profile
case — makes it easy to guarantee smoothness (no abrupt area-ratio
discontinuities) directly through the control points, rather than through
post-hoc smoothing of a recovered $A(x)$.

## Imposing both wall and centerline pressure

A single target $p_\text{wall}(x)$ under-constrains a 2D passage: it says
nothing directly about the transverse (normal-to-streamline) pressure
gradient set up by streamline curvature,

$$
\frac{\partial p}{\partial n} = \frac{\rho V^2}{R_c},
$$

which is exactly the extra piece of information that separates a 2D design
from a 1D one. Imposing **both** $p_\text{wall}(x)$ and a centerline target
$p_\text{center}(x)$ gives two matching conditions instead of one, and
directly constrains that curvature-driven gradient rather than leaving it as
an emergent, under-determined byproduct of whatever geometry the iteration
converges to. This pairs naturally with streamline-curvature-type forward
solvers, which already carry $\partial p/\partial n$ as a state quantity.

## Two ways the field has historically handled this

### 1. Direct methods (restrictive but non-iterative)

Under strong enough assumptions — incompressible or specific compressible
potential flow, irrotational, isentropic — the flow can be mapped into a
different coordinate system (the hodograph plane, using flow angle and speed
or a related pair as independent variables) in which the *geometry* becomes
the dependent variable and the equations linearize or otherwise simplify
enough to solve directly for the wall shape from the prescribed surface
velocity/pressure. This preserves the 1D method's "no iteration on geometry"
character, but only within its restrictive assumptions.

**Why this does not extend to two-phase flows.** The hodograph method's
linearity is not a generic feature of potential flow — it comes from a very
specific algebraic collapse that only holds for a single-field, barotropic,
irrotational flow:

- *Barotropic closure.* The transformation relies on density being a
  single-valued function of speed alone, $\rho = \rho(q)$ (via the isentropic
  relation), so that the whole state is pinned down by one hodograph
  variable. Two-phase mixtures are not barotropic in general: density depends
  on pressure *and* on quality/void fraction, which are themselves separate
  transported (or relaxing) unknowns, not a function of local speed.
- *Single velocity field.* The transformation assumes one potential
  function for the whole flow. A two-fluid (6-equation) model carries
  independent phasic velocities with slip ($u_g \neq u_l$); there is no
  single scalar potential describing two fields with relative motion,
  so irrotationality itself loses its meaning at the mixture level.
- *No source terms.* The linear hodograph PDE comes from combining mass
  conservation and irrotationality with nothing else in the way. Interfacial
  mass/momentum/energy exchange and phase-change relaxation (the source
  terms that define 5- and 6-equation models) are exactly the kind of
  extra, path-dependent physics that break the assumption that local state
  depends only on local speed — the same reason friction broke the
  closed-form isentropic shortcut in the 1D case (§2b of
  `inverse_design.md`).

So it is not that the hodograph method is merely "less convenient" for
two-phase or non-equilibrium flow — the transformation's validity is
contingent on exactly the assumptions (barotropic, single-field,
source-free) that those models are built to relax.

### 2. Iterative methods (general, but not a single forward pass)

The general-purpose approach used in turbomachinery blade and nozzle-contour
inverse design (streamline curvature methods, panel-method-based inverse
design, adjoint/optimization-based shape design) instead treats the problem
as a fixed-point iteration:

1. Guess an initial wall shape $y_\text{wall}^{(0)}(x)$ (its slope $dA/dx$
   parametrized on the unit mesh, as above).
2. Solve the forward 2D flow field for that geometry (Euler/potential
   solver, streamline-curvature method, or a 5-/6-equation two-phase
   forward model).
3. Compare both the resulting wall pressure $p_\text{wall}^{(k)}(x)$ and
   centerline pressure $p_\text{center}^{(k)}(x)$ to their targets
   $p_\text{wall}^\star(x)$, $p_\text{center}^\star(x)$.
4. Perturb the geometry to reduce the combined mismatch — e.g. via a
   transpiration velocity proportional to the pressure error
   (streamline-curvature/surface-transpiration methods), or via a full
   shape-optimization update (gradient- or adjoint-based).
5. Repeat until both $p_\text{wall}^{(k)} \to p_\text{wall}^\star$ and
   $p_\text{center}^{(k)} \to p_\text{center}^\star$.

This handles arbitrary flow physics (real gas, rotational flow, two-phase),
at the cost of no longer being a single closed-form or single-marching-pass
recovery — each iteration requires a full forward flow solve.

## Implication for this codebase

A 2D extension consistent with the rest of `nozzlex` would have to follow
the iterative route (§ above), not the hodograph/direct route, for the
reasons given above. Beyond that, the formulation should stay general rather
than committing to a specific flow model: `nozzlex` already spans
single-phase, homogeneous-relaxation (5-equation), and two-fluid
(6-equation) forward models elsewhere in the package, and whichever of those
best fits a given case should be pluggable as the "forward solve" step in
the iteration, without changing the outer geometry-update loop or the
wall/centerline pressure matching. Concretely, that means:

- A forward 2D solver — of whichever fidelity (Euler, streamline-curvature,
  5-eq, 6-eq) suits the case — would need to exist first; `march_from_p_ode`
  and friends do not generalize to this, a genuinely different forward
  model is needed regardless of which one.
- The "inverse" part becomes a geometry-update loop wrapped around
  whichever forward solver is in use, not a new closed-form or
  ODE-marching routine, and not tied to a single flow-physics assumption.
- Convergence, uniqueness, and well-posedness (does a target
  $\{p_\text{wall}(x), p_\text{center}(x)\}$ pair even correspond to a
  physically realizable 2D passage, for the chosen flow model?) become real
  concerns in a way they are not in the 1D case, where the algebraic/ODE
  recovery is unconditionally well-posed given a physically consistent
  inlet state.

This is a substantially larger undertaking than the 1D workflow — worth
scoping as a separate forward-solver-plus-optimization-loop project rather
than an extension of `nozzlex/inverse_design/`.

## Worked example

`examples/quasi2d_design/run_quasi2d_design.py` implements the minimal,
concrete version of the "impose both wall and centerline pressure" idea
above — deliberately restricted to two streamlines (centerline and wall)
so it stays a self-contained illustration rather than a real forward-solver
project. It does not modify anything in `nozzlex`; it only imports and
reuses the existing 1D `march_from_p_isentropic` as the per-streamline
building block, exactly as proposed above.

A first, non-iterative version simply marched both streamlines from their
independently-imposed pressure targets and assembled the passage half-height
$h(x)$ from global mass conservation (a 2-point trapezoidal quadrature of
mass flux across the span). Its own curvature-consistency diagnostic showed
what the theory here predicts: the resulting shape's curvature does **not**
imply the same $p_\text{wall}(x) - p_\text{center}(x)$ that was imposed —
closing the problem with mass conservation alone is well-posed in the
degrees-of-freedom sense (one unknown function $h(x)$, one equation), but
it is not automatically *physically* self-consistent with an independently
chosen wall-pressure target.

The script now adds the iterative correction this implies: $p_\text{center}
(x)$ is kept fixed as the true design target, while $p_\text{wall}(x)$ is
corrected every iteration toward whatever the *current* shape's own
curvature implies via radial equilibrium, until the imposed and
curvature-implied $\Delta p$ agree. A handful of interior streamlines are
then reconstructed for visualization, assuming a linear mass-flux profile
across the span and finding the transverse position that carries each equal
fraction of the total mass flow.

**Concrete result.** A stress-test case (centerline exit pressure 0.90 bar,
*initial* wall-exit target 0.60 bar — a deliberately large 0.30 bar spread)
converged in 27 iterations, but the wall-exit pressure drifted from its
0.60 bar target to 0.895 bar, i.e. it ended up almost equal to the
centerline value rather than 0.30 bar below it. This is the degrees-of-
freedom argument from "Imposing both wall and centerline pressure" above,
demonstrated numerically rather than just counted: a smooth nozzle contour
(the kind a NURBS-parametrized shape produces) simply does not have a tight
enough radius of curvature to sustain a large wall-vs-centerline pressure
difference, so the self-consistent solution collapses toward the smallest
$\Delta p$ that the achievable curvature allows — confirming that, with
only two streamlines closed this way, wall pressure is not an independently
free design choice.
