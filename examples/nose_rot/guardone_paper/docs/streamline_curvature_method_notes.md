# The Streamline Curvature Method (SCM) for the circular-bend Figure 10

Notes on Gajoni & Guardone's Section 4.1 subsonic validation (Fig. 10)
and its faithful reproduction in `figure10_slc.py`. Supersedes an
earlier version of this file written before a working implementation
existed; kept as a record of the method and of the bug that made the
first attempts collapse to the wrong (trivial) solution.

## What the SCM is

A specialized, iterative method from the turbomachinery throughflow
literature (Wu 1952; Novak 1967; Korpela, *Principles of
Turbomachinery*, App. A), built around a mesh that is aligned with the
flow rather than fixed:

- The computational grid consists of fixed cross-stream cuts, called
  **quasi-orthogonals** (radial lines here), together with a set of
  **streamlines** that are free to move ("float") within that grid.
- Streamlines are **guessed at the start** (equispaced, matching the
  uniform inlet profile) and updated by iterating the whole domain —
  a **global relaxation**, not a downstream march.
- At each quasi-orthogonal, a reduced cross-stream momentum balance —
  the **Normal Momentum Equation (NME)** — is integrated radially for
  the streamwise velocity profile $u_\ell(r)$ (Gajoni thesis Eq. 3.15,
  matching Korpela App. A specialized to zero sweep / zero swirl):

$$
\frac{\partial u_\ell}{\partial r} + \frac{\cos\phi}{R_c}\,u_\ell = \sin\phi\,\frac{\partial u_\ell}{\partial \ell}
$$

  with $R_c$ the local streamline radius of curvature and $\phi$ the
  angle between the streamline and the local flow (quasi-orthogonal
  normal) direction. The streamwise-gradient term
  $\partial u_\ell/\partial \ell$ is evaluated from the **previous
  global iteration's** solution — lagged coupling, which is what makes
  the method cheap (no simultaneous 2D solve).
- Given the velocity profile, streamline positions are updated to
  satisfy mass conservation in each streamtube, and the whole thing is
  iterated (with relaxation) until self-consistent (three nested
  loops: mass-flow matching at each station, velocity-field
  convergence, streamline-position convergence).

## The bug that made early attempts collapse to a trivial solution

The tempting simplification for this specific geometry (constant wall
curvature) is to compute $\phi$ and $R_c$ from the streamline shape
using **separate, one-sided finite-difference stencils** for the
straight section and the bend — i.e. never differentiating across the
junction at $\theta=0$. With the naive equispaced initial guess (zero
$\theta$-variation everywhere), this makes $\phi=0$ and $R_c=r$ exactly
at *every* bend station from the very first iteration — a state that
turns out to be an exact, **stable fixed point** of the discretized
iteration (confirmed empirically: neither relaxation factor nor
artificial seeding of the initial guess changed the converged answer).
The result is an unphysical instant jump to the equilibrium profile
right at $\theta=0$, with zero dependence on the channel width
$r_e/r_i$ — contradicted directly by the paper's own stated result
that wider channels take a larger $\theta$ to reach equilibrium.

**The fix**: compute $\phi$ and $R_c$ from a *single, unified*
parametric-curve curvature calculation on the global $(x,y)$
embedding (straight section + bend together, parametrized by arc
length), instead of two decoupled per-region stencils. This lets the
straight section's incoming flow direction genuinely carry across the
junction into the first few bend stations, breaking the degenerate
fixed point and producing genuine, $r_e/r_i$-dependent gradual
development — matching the shape and approximate transition angle of
the published Figure 10.

## Remaining known limitation

The per-station radial integration is a sequential march (an
integrating-factor product over ~N streamline steps). Error in the
curvature estimate compounds multiplicatively across that march, and
empirically **grows with N** for this discretization rather than
converging — the reliable, accurate regime found by direct scanning is
N≈15-20 streamlines, not the paper's stated N=100. Beyond that,
results degrade and can become unstable even with hard clamps on the
per-step growth factor and a bisection (not multiplicative) wall-
velocity update. A genuinely more accurate scheme (adaptive step size,
or an implicit / non-sequential radial solve) would be needed to
safely match N=100; this is flagged as an open item, not fixed.
