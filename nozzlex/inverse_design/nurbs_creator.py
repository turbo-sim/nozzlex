"""Simple NURBS creator and plotter.

Creates a cubic NURBS curve using the start and end points plus three
intermediate control points (A, B, C). Uses an open uniform knot
vector for degree 4 and lets you set weights for the control points.

Usage example:
    from nurbs_creator import create_and_plot

    start = (0.0, 10.0)
    A = (0.15, 9.8)
    B = (0.35, 5.5)
    C = (0.40, 1.0)
    end = (1.0, 1.0)
    weights = [1.0, 1.2, 1.0, 0.9, 1.0]
    create_and_plot(start, end, A, B, C, weights=weights)

This file has no external dependencies besides numpy and matplotlib.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from functools import lru_cache
from typing import Sequence, Iterable, Tuple, List


def make_open_uniform_knot(n_ctrl_pts: int, degree: int) -> np.ndarray:
    """Return open-uniform knot vector for given control points and degree.

    n_ctrl_pts: number of control points (n+1)
    degree: polynomial degree p
    """
    n = n_ctrl_pts - 1
    m = n + degree + 1  # highest knot index
    # open uniform: first p+1 zeros, last p+1 ones, internal uniform
    kv = np.zeros(m + 1)
    kv[-(degree + 1):] = 1.0
    internal_count = m + 1 - 2 * (degree + 1)
    if internal_count > 0:
        internal = np.linspace(0, 1, internal_count + 2)[1:-1]
        kv[degree + 1:degree + 1 + internal_count] = internal
    return kv


@lru_cache(maxsize=1024)
def bspline_basis(i: int, p: int, u: float, knot_tuple: Tuple[float, ...]) -> float:
    """Cox-de Boor recursive B-spline basis function.

    knot_tuple is hashed for caching; small recursion depth (p<=3) so OK.
    """
    knots = knot_tuple
    if p == 0:
        if knots[i] <= u < knots[i + 1] or (u == knots[-1] and knots[i + 1] == knots[-1] and knots[i] < knots[i + 1]):
            return 1.0
        return 0.0
    denom1 = knots[i + p] - knots[i]
    denom2 = knots[i + p + 1] - knots[i + 1]
    term1 = 0.0
    term2 = 0.0
    if denom1 != 0:
        term1 = (u - knots[i]) / denom1 * bspline_basis(i, p - 1, u, knot_tuple)
    if denom2 != 0:
        term2 = (knots[i + p + 1] - u) / denom2 * bspline_basis(i + 1, p - 1, u, knot_tuple)
    return term1 + term2


def nurbs_curve_points(ctrl_pts: Iterable[Tuple[float, float]],
                       weights: Sequence[float],
                       degree: int = 4,
                       num: int = 200) -> np.ndarray:
    """Evaluate a NURBS curve defined by control points and weights.

    Returns an array shape (num, 2).
    """
    ctrl = np.asarray(list(ctrl_pts), dtype=float)
    n_ctrl = len(ctrl)
    if len(weights) != n_ctrl:
        raise ValueError("weights length must match control points length")
    kv = make_open_uniform_knot(n_ctrl, degree)
    knot_tuple = tuple(float(x) for x in kv)
    u_vals = np.linspace(0.0, 1.0, num)
    curve = np.zeros((num, 2), dtype=float)
    w = np.asarray(weights, dtype=float)
    for iu, u in enumerate(u_vals):
        num_x = 0.0
        num_y = 0.0
        denom = 0.0
        for i in range(n_ctrl):
            N = bspline_basis(i, degree, u, knot_tuple)
            Niw = N * w[i]
            num_x += Niw * ctrl[i, 0]
            num_y += Niw * ctrl[i, 1]
            denom += Niw
        if denom == 0:
            curve[iu, :] = ctrl[-1]
        else:
            curve[iu, 0] = num_x / denom
            curve[iu, 1] = num_y / denom
    return curve


def create_and_plot(start: Tuple[float, float],
                    end: Tuple[float, float],
                    A: Tuple[float, float],
                    B: Tuple[float, float],
                    C: Tuple[float, float],
                    weights: List[float] | None = None,
                    degree: int = 4,
                    num: int = 400,
                    show_control_polygon: bool = True,
                    save: str | None = None,
                    xlim: Tuple[float, float] | None = None,
                    ylim: Tuple[float, float] | None = None,
                    margin: float = 0.05) -> np.ndarray:
    """Create the NURBS curve from provided points and plot it.

    Control polygon order: [start, A, B, C, end]
    weights: list of 5 weights (default all ones).
    """
    ctrl = [start, A, B, C, end]
    if weights is None:
        weights = [1.0] * len(ctrl)
    curve = nurbs_curve_points(ctrl, weights, degree=degree, num=num)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(curve[:, 0], curve[:, 1], '-r', label='NURBS')
    if show_control_polygon:
        cp = np.array(ctrl)
        ax.plot(cp[:, 0], cp[:, 1], '--o', color='gray', label='control polygon')
        for i, lbl in enumerate(['P0(start)', 'A', 'B', 'C', 'P4(end)']):
            ax.text(cp[i, 0], cp[i, 1], '  ' + lbl, verticalalignment='bottom')
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    ax.grid(True, linestyle=':', linewidth=0.5)
    # set equal aspect without changing axis limits, then apply limits
    # ax.set_aspect('equal', adjustable='box')
    # determine axis limits from control polygon and curve when not provided
    all_x = np.hstack([curve[:, 0], cp[:, 0]])
    all_y = np.hstack([curve[:, 1], cp[:, 1]])
    if xlim is None:
        xmin, xmax = float(all_x.min()), float(all_x.max())
        xrange = xmax - xmin
        padx = margin * (xrange if xrange > 0 else 1.0)
        ax.set_xlim(xmin - padx, xmax + padx)
    else:
        ax.set_xlim(*xlim)
    if ylim is None:
        ymin, ymax = float(all_y.min()), float(all_y.max())
        yrange = ymax - ymin
        pady = margin * (yrange if yrange > 0 else 1.0)
        ax.set_ylim(ymin - pady, ymax + pady)
    else:
        ax.set_ylim(*ylim)
    ax.legend()
    plt.tight_layout()
    plt.show()

    if save:
        fig.savefig(save, dpi=200)
    else:
        plt.show()
    return curve


if __name__ == '__main__':
    # Example that mirrors the style in the user's image (pressure vs x/L)
    start = (0.0, 10.0)
    A = (0.5, 10.0)
    B = (0.5, 5.5)
    C = (0.5, 1.0)
    end = (1.0, 1.0)
    weights = [1.0, 1.0, 1.0, 1.0, 1.0]
    create_and_plot(start, end, A, B, C, weights=weights, save=None)
