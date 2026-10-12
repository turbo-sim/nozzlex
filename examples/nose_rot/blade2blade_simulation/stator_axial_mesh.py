"""
Build the passage geometry and initial (straight-line) streamline mesh for
the AXIAL stator blade-to-blade SLC solve, before any flow physics is run.

Simpler than the radial-outflow case: the neighbor blade is a pure
translation by the (constant) pitch spacing, not a rotation about the
machine axis -- so there is no radius-dependent pitch growth and no
blade-span/height bookkeeping needed, the 2D unit-depth passage area is
directly valid.

Steps:
  1. Load the single stator blade contour (closed loop, LE->surfaceA->TE->surfaceB->LE).
  2. Split it into the two surfaces at the trailing-edge (max-x) point.
  3. Build the neighboring blade by translating the same contour by the
     blade pitch (spacing) in the tangential (y) direction.
  4. Select the wall pair: blade1's "B" surface vs the neighbor's "A"
     surface (translated by +spacing) -- verified by checking that the
     OTHER (non-wall) surface of each blade does NOT fall inside the
     passage polygon (tip regions excluded from both the polygon and the
     test points, since the two walls meet almost exactly there by
     construction and any mismatch reads as a false "inside").
  5. Extend both walls with straight inlet/outlet sections.
  6. Resample the full (inlet + passage + outlet) walls onto common
     x-stations (the natural choice for a linear cascade -- both surfaces
     are single-valued functions of axial position), giving the
     quasi-orthogonal cuts, and seed N_STREAMLINES initial streamlines as
     straight-line interpolation across each cut.
"""

import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import yaml

BLADE_FILE = "stator_blade_axial.csv"
N_STATIONS = 30
N_STREAMLINES = 15
N_EXT = 60
EXT_FRAC = 2.0  # inlet/outlet extension length, in units of one pitch (spacing)
TIP_MARGIN = 5  # stations trimmed from each end before the containment check


def load_blade(path):
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    return data[:, 0], data[:, 1]


def _trim_near_vertical_tail(x, y, frac=0.2):
    """Drop the tail where the curve goes nearly vertical (dx collapses
    toward 0 while y keeps changing) as it wraps around the rounded LE/TE
    cap. This isn't a resolution or smoothing issue: resample_by_x()
    fundamentally treats the curve as y=f(x), and once dx==0 (verified:
    happens exactly, not just approximately, right at the tip)
    np.unique() silently drops the duplicate-x point, compressing/losing
    that part of the geometry -- which is what actually produced the
    wiggle several other fixes (smoothing, more extension points) didn't
    touch. Trims from the tip inward until dx recovers above `frac` times
    the surface's typical (median) spacing."""
    dx = np.diff(x)
    typical = np.median(np.abs(dx))
    ok = np.abs(dx) > frac * typical
    # walk in from the tip (end of array) while dx stays too small
    i = len(ok)
    while i > 1 and not ok[i - 2]:
        i -= 1
    return x[:i], y[:i]


def split_surfaces(x, y):
    i_te = int(np.argmax(x))
    xA, yA = x[: i_te + 1], y[: i_te + 1]
    xB, yB = x[i_te:][::-1], y[i_te:][::-1]
    xA, yA = _trim_near_vertical_tail(xA, yA)
    xB, yB = _trim_near_vertical_tail(xB, yB)
    return {"A": (xA, yA), "B": (xB, yB)}


def resample_by_x(x, y, x_query):
    """Linear interpolation. (Tried cubic spline to remove a small
    mid-chord curvature bump in the mean camberline -- made the
    trailing-edge region far worse via spline ringing, and left the
    mid-chord bump completely unchanged. That result is actually
    informative: since the bump persists identically under both linear
    and cubic interpolation, it isn't an interpolation artifact -- see
    the note in build_curvilinear_mesh() about the passage mean
    camberline vs. the blade's own camberline.)"""
    order = np.argsort(x)
    xs, ys = x[order], y[order]
    xs_u, idx = np.unique(xs, return_index=True)
    return x_query, np.interp(x_query, xs_u, ys[idx])


def translate(x, y, dy):
    return x, y + dy


def estimate_tangent(x, y, n_fit=20, skip=15, at_start=True):
    """Robust local tangent direction from a least-squares line fit over
    a window set back from the tip -- NOT the 2-point tangent, and not
    even a naive multi-point fit from the very first/last points either:
    the rounded LE/TE nose cap is a small feature but dense in points, so
    even ~15 points from the literal tip can sit entirely within the
    cap's wraparound and give a near-meaningless direction (observed:
    pointing almost opposite the actual flow direction). `skip` points
    are dropped nearest the tip before fitting the next `n_fit` points,
    landing the fit window on the actual blade surface past the cap.
    Not the nominal design metal angle either: the parameterized curve's
    real tangent at its tip is not exactly equal to the design angle, so
    extending along the design angle leaves a slope discontinuity right
    where the extension meets the real surface, which blows up the
    finite-difference curvature there."""
    if at_start:
        xs = x[skip:skip + n_fit]
        ys = y[skip:skip + n_fit]
    else:
        xs = x[-(skip + n_fit):-skip]
        ys = y[-(skip + n_fit):-skip]
    # fit y = a*x + b in the local frame least-squares sense, but guard
    # against a near-vertical segment by fitting whichever of (x,y) has
    # more spread as the independent variable.
    if np.ptp(xs) >= np.ptp(ys):
        a, b = np.polyfit(xs, ys, 1)
        d = np.array([1.0, a])
    else:
        a, b = np.polyfit(ys, xs, 1)
        d = np.array([a, 1.0])
    d /= np.linalg.norm(d)
    # extend_straight() uses dir_in/dir_out both pointing in the
    # "downstream" (+x-ish) sense -- it subtracts dir_in for the inlet
    # extension and adds dir_out for the outlet one -- so both must have
    # a positive x-component regardless of which end they were fit from.
    if d[0] < 0:
        d = -d
    return d


def _tip_tangent(x, y, at_start, n_pts=4):
    """True local tangent at the tip, from a small quadratic fit through
    the first/last n_pts points (steadier than a bare 2-point difference,
    but still local -- unlike estimate_tangent() this is deliberately NOT
    set back from the tip, because here we WANT the actual tip direction,
    however "sideways" it looks, as the exact-match end of a Hermite
    blend -- not a good-looking straight-line direction."""
    xs = x[:n_pts] if at_start else x[-n_pts:][::-1]
    ys = y[:n_pts] if at_start else y[-n_pts:][::-1]
    if np.ptp(xs) >= np.ptp(ys):
        c = np.polyfit(xs - xs[0], ys, 2)
        slope0 = c[1]  # dy/dx at xs[0]
        d = np.array([1.0, slope0])
    else:
        c = np.polyfit(ys - ys[0], xs, 2)
        slope0 = c[1]  # dx/dy at ys[0]
        d = np.array([slope0, 1.0])
    d /= np.linalg.norm(d)
    if at_start and d[0] > 0:
        d = -d  # tip tangent should point away from the passage (upstream)
    if not at_start and d[0] < 0:
        d = -d  # ... or downstream, at the trailing edge
    return d


BLEND_FRAC = 0.15  # fraction of the extension length used for the tip-to-design-angle blend


def _hermite(p0, t0_dir, p1, t1_dir, n, t0_scale, t1_scale):
    t = np.linspace(0.0, 1.0, n)
    h00 = 2 * t**3 - 3 * t**2 + 1
    h10 = t**3 - 2 * t**2 + t
    h01 = -2 * t**3 + 3 * t**2
    h11 = t**3 - t**2
    T0, T1 = t0_dir * t0_scale, t1_dir * t1_scale
    pts = (h00[:, None] * p0 + h10[:, None] * T0
           + h01[:, None] * p1 + h11[:, None] * T1)
    return pts[:, 0], pts[:, 1]


def extend_straight(x, y, length, n_ext, dir_in, dir_out):
    """Extend in two pieces: a SHORT cubic-Hermite blend (length
    BLEND_FRAC*length) from the blade's own true tip tangent to the
    design flow direction, followed by a plain straight run the rest of
    the way in that design direction.

    A pure straight extension in the design direction leaves a visible
    kink where it meets the blade curve, because the parameterized
    curve's actual tangent at its tip is not exactly the nominal design
    angle. A full-length Hermite blend fixes that but overshoots into a
    loop when the tip/design tangents differ a lot and the blend is
    stretched across the whole (multi-pitch) extension -- tried that,
    made it worse. Confining the blend to a short local region (a small
    fraction of the extension) keeps the position/tangent mismatch being
    absorbed over a short distance without overshoot, and the straight
    segment picks up exactly at the blend's end position with exactly the
    design tangent already, so the join between the two pieces is C1 by
    construction. dir_in/dir_out must be the SAME for both walls of a
    passage so the extensions stay parallel (periodic) far from the blade."""
    n_blend = max(3, int(round(n_ext * BLEND_FRAC)))
    n_straight = n_ext - n_blend
    blend_len = BLEND_FRAC * length
    straight_len = length - blend_len

    # inlet (upstream): blend near the blade, then straight further out
    p0_in = np.array([x[0], y[0]])
    tip_in = _tip_tangent(x, y, at_start=True)
    p_blend_in = p0_in - blend_len * dir_in
    xb_in, yb_in = _hermite(p0_in, tip_in, p_blend_in, -dir_in, n_blend + 1,
                             blend_len, blend_len)
    xb_in, yb_in = xb_in[:0:-1], yb_in[:0:-1]  # drop blade-end dup, reverse to far->near
    s_far = np.linspace(straight_len, 0.0, n_straight, endpoint=False)
    x_far_in = p_blend_in[0] - dir_in[0] * s_far
    y_far_in = p_blend_in[1] - dir_in[1] * s_far
    x_in = np.concatenate([x_far_in, xb_in])
    y_in = np.concatenate([y_far_in, yb_in])

    # outlet (downstream): blend near the blade, then straight further out
    p0_out = np.array([x[-1], y[-1]])
    tip_out = _tip_tangent(x, y, at_start=False)
    p_blend_out = p0_out + blend_len * dir_out
    xb_out, yb_out = _hermite(p0_out, tip_out, p_blend_out, dir_out, n_blend + 1,
                               blend_len, blend_len)
    xb_out, yb_out = xb_out[1:], yb_out[1:]
    s_far = np.linspace(0.0, straight_len, n_straight + 1)[1:]
    x_far_out = p_blend_out[0] + dir_out[0] * s_far
    y_far_out = p_blend_out[1] + dir_out[1] * s_far
    x_out = np.concatenate([xb_out, x_far_out])
    y_out = np.concatenate([yb_out, y_far_out])

    return np.concatenate([x_in, x, x_out]), np.concatenate([y_in, y, y_out])


def check_combo(surfaces, spacing, label1, label2, sign, x_common):
    other1 = "B" if label1 == "A" else "A"
    other2 = "B" if label2 == "A" else "A"

    x1, y1 = surfaces[label1]
    xs1, ys1 = resample_by_x(x1, y1, x_common)
    x2, y2 = surfaces[label2]
    x2t, y2t = translate(x2, y2, sign * spacing)
    xs2, ys2 = resample_by_x(x2t, y2t, x_common)

    poly_x = np.concatenate([xs1[TIP_MARGIN:-TIP_MARGIN], xs2[TIP_MARGIN:-TIP_MARGIN][::-1]])
    poly_y = np.concatenate([ys1[TIP_MARGIN:-TIP_MARGIN], ys2[TIP_MARGIN:-TIP_MARGIN][::-1]])
    path = mpath.Path(np.column_stack([poly_x, poly_y]))

    xo1, yo1 = surfaces[other1]
    xo2, yo2 = surfaces[other2]
    xo2t, yo2t = translate(xo2, yo2, sign * spacing)
    m1 = int(len(xo1) * 0.15)
    m2 = int(len(xo2t) * 0.15)
    test_x = np.concatenate([xo1[m1:-m1], xo2t[m2:-m2]])
    test_y = np.concatenate([yo1[m1:-m1], yo2t[m2:-m2]])
    frac_inside = path.contains_points(np.column_stack([test_x, test_y])).mean()

    width = np.hypot(xs2 - xs1, ys2 - ys1)
    return dict(label1=label1, label2=label2, sign=sign,
                frac_inside=frac_inside, mean_width=np.mean(width))


if __name__ == "__main__":
    case_file = sorted(glob.glob("meanline_stage_axial_*.yaml"))[-1]
    with open(case_file) as f:
        cfg = yaml.safe_load(f)
    spacing = cfg["geometry"]["stator"]["spacing"]

    x, y = load_blade(BLADE_FILE)
    surfaces = split_surfaces(x, y)
    x_common = np.linspace(x.min() + 1e-5, x.max() - 1e-5, N_STATIONS)

    candidates = [
        check_combo(surfaces, spacing, l1, l2, sign, x_common)
        for l1 in ("A", "B") for l2 in ("A", "B") if l2 != l1
        for sign in (+1.0, -1.0)
    ]
    for c in candidates:
        print(c)
    valid = [c for c in candidates if c["frac_inside"] < 1e-6]
    best = min(valid, key=lambda c: abs(c["mean_width"] - spacing))
    print(f"\nSelected: blade1[{best['label1']}] vs neighbor(sign={best['sign']:+.0f})"
          f"[{best['label2']}]  mean_width={best['mean_width']:.5f} (spacing={spacing:.5f})")

    x1, y1 = surfaces[best["label1"]]
    x2, y2 = surfaces[best["label2"]]

    # Use the nominal design metal angle for the extension direction.
    # (Tried matching the blade's own local tangent instead -- doesn't
    # work here: this blade is cambered continuously from LE to TE with
    # no flat run to fit a tangent from, so any nearby window is still
    # curving. A small kink at the extension/blade junction is
    # unavoidable either way; it's handled downstream by bounding the
    # curvature term to a physical value, not by chasing exact C1
    # continuity in the geometry.)
    geom = cfg["geometry"]["stator"]
    angle_in = np.deg2rad(geom["metal_angle_in"])
    angle_out = np.deg2rad(geom["metal_angle_out"])
    dir_in = np.array([np.cos(angle_in), np.sin(angle_in)])
    dir_out = np.array([np.cos(angle_out), np.sin(angle_out)])

    ext_len = EXT_FRAC * spacing
    x1e, y1e = extend_straight(x1, y1, ext_len, N_EXT, dir_in, dir_out)
    x2e, y2e = extend_straight(x2, y2, ext_len, N_EXT, dir_in, dir_out)
    x2e, y2e = translate(x2e, y2e, best["sign"] * spacing)

    n_full = N_STATIONS + 2 * N_EXT
    x_full_common = np.linspace(x1e.min() + 1e-6, x1e.max() - 1e-6, n_full)
    _, yW1 = resample_by_x(x1e, y1e, x_full_common)
    _, yW2 = resample_by_x(x2e, y2e, x_full_common)
    xW1 = xW2 = x_full_common

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(x, y, "k-", lw=1.0, alpha=0.4, label="Blade 1 (solid contour)")
    xt, yt = translate(x, y, best["sign"] * spacing)
    ax.plot(xt, yt, "k-", lw=1.0, alpha=0.4, label="Blade 2 (solid contour)")

    ax.plot(xW1, yW1, color="tab:blue", lw=2.0, label="Passage wall (blade 1 side)")
    ax.plot(xW2, yW2, color="tab:orange", lw=2.0, label="Passage wall (neighbor side)")

    for j in range(n_full):
        ax.plot([xW1[j], xW2[j]], [yW1[j], yW2[j]], color="gray", lw=0.6)
    frac = np.linspace(0.0, 1.0, N_STREAMLINES)
    for f in frac:
        ys = yW1 + f * (yW2 - yW1)
        ax.plot(xW1, ys, color="tab:green", lw=0.8, alpha=0.8)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("axial direction [m]")
    ax.set_ylabel("tangential direction [m]")
    ax.set_title("Axial stator passage: initial (straight-line) streamline mesh")
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout(pad=1)
    fig.savefig("stator_axial_initial_mesh.png", dpi=150)
    print("saved stator_axial_initial_mesh.png")
