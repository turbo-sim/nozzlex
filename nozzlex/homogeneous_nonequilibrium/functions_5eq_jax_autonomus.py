
# %%
# import scipy.linalg
import scipy.integrate
import numpy as np
import matplotlib as mpl
# import matplotlib.pyplot as plt
# import perfect_gas_prop.perfect_gas_prop as perfect_gas_prop
# import real_gas_prop.real_gas_prop as rg
from cycler import cycler
import numpy as np
from scipy.linalg import det
# import CoolProp.CoolProp as CP
import jaxprop as jxp
import pandas as pd
import os
import jax.numpy as jnp

# %%

# TODO: Implement the Elliot area
# TODO: Check the heat exchange coefficient
# TODO: Find the mixture speed of sound
# TODO: Determinant is always negative. What to do?
# TODO: Implement bicubic interpolator to get faster convergence

zero = 1e-3
backend = "HEOS"



COLORS_PYTHON = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]

COLORS_MATLAB = [
    "#0072BD",
    "#D95319",
    "#EDB120",
    "#7E2F8E",
    "#77AC30",
    "#4DBEEE",
    "#A2142F",
]


try:
    import IPython.core.ultratb
except ImportError:
    # No IPython. Use default exception printing.
    pass
else:
    import sys

    sys.excepthook = IPython.core.ultratb.ColorTB()


def print_dict(data, indent=0):
    """
    Recursively prints nested dictionaries with indentation.

    Parameters
    ----------
    data : dict
        The dictionary to print. It can contain nested dictionaries as values.
    indent : int, optional
        The initial level of indentation for the keys of the dictionary, by default 0.
        It controls the number of spaces before each key.

    Returns
    -------
    None

    Examples
    --------
    >>> data = {"a": 1, "b": {"c": 2, "d": {"e": 3}}}
    >>> print_dict(data)
    a: 1
    b:
        c: 2
        d:
            e: 3
    """

    for key, value in data.items():
        print("    " * indent + str(key) + ":", end=" ")
        if isinstance(value, dict):
            print("")
            print_dict(value, indent + 1)
        else:
            print(value)


def set_plot_options(
    fontsize=13,
    grid=True,
    major_ticks=True,
    minor_ticks=True,
    margin=0.05,
    color_order="matlab",
):
    """Set plot options for publication-quality figures"""

    if isinstance(color_order, str):
        if color_order.lower() == "default":
            color_order = COLORS_PYTHON

        elif color_order.lower() == "matlab":
            color_order = COLORS_MATLAB

    # Define dictionary of custom settings
    rcParams = {
        "text.usetex": False,
        "font.size": fontsize,
        "font.style": "normal",
        "font.family": "serif",  # 'serif', 'sans-serif', 'cursive', 'fantasy', 'monospace'
        "font.serif": ["Times New Roman"],  # ['times new roman', 'cmr10']
        "mathtext.fontset": "stix",  # ["stix", 'cm']
        "axes.edgecolor": "black",
        "axes.linewidth": 1.25,
        "axes.titlesize": fontsize,
        "axes.titleweight": "normal",
        "axes.titlepad": fontsize * 1.4,
        "axes.labelsize": fontsize,
        "axes.labelweight": "normal",
        "axes.labelpad": fontsize,
        "axes.xmargin": margin,
        "axes.ymargin": margin,
        "axes.zmargin": margin,
        "axes.grid": grid,
        "axes.grid.axis": "both",
        "axes.grid.which": "major",
        "axes.prop_cycle": cycler(color=color_order),
        "grid.alpha": 0.5,
        "grid.color": "black",
        "grid.linestyle": "-",
        "grid.linewidth": 0.5,
        "legend.borderaxespad": 1,
        "legend.borderpad": 0.6,
        "legend.edgecolor": "black",
        "legend.facecolor": "white",
        "legend.labelcolor": "black",
        "legend.labelspacing": 0.3,
        "legend.fancybox": True,
        "legend.fontsize": fontsize - 2,
        "legend.framealpha": 1.00,
        "legend.handleheight": 0.7,
        "legend.handlelength": 1.25,
        "legend.handletextpad": 0.8,
        "legend.markerscale": 1.0,
        "legend.numpoints": 1,
        "lines.linewidth": 1.25,
        "lines.markersize": 5,
        "lines.markeredgewidth": 1.25,
        "lines.markerfacecolor": "white",
        "xtick.direction": "in",
        "xtick.labelsize": fontsize - 1,
        "xtick.bottom": major_ticks,
        "xtick.top": major_ticks,
        "xtick.major.size": 6,
        "xtick.major.width": 1.25,
        "xtick.minor.size": 3,
        "xtick.minor.width": 0.75,
        "xtick.minor.visible": minor_ticks,
        "ytick.direction": "in",
        "ytick.labelsize": fontsize - 1,
        "ytick.left": major_ticks,
        "ytick.right": major_ticks,
        "ytick.major.size": 6,
        "ytick.major.width": 1.25,
        "ytick.minor.size": 3,
        "ytick.minor.width": 0.75,
        "ytick.minor.visible": minor_ticks,
        "savefig.dpi": 600,
    }

    # Update the internal Matplotlib settings dictionary
    mpl.rcParams.update(rcParams)


# # Statically add phase indices to the module (IDE autocomplete)
# iphase_critical_point = CP.iphase_critical_point
# iphase_gas = CP.iphase_gas
# iphase_liquid = CP.iphase_liquid
# iphase_not_imposed = CP.iphase_not_imposed
# iphase_supercritical = CP.iphase_supercritical
# iphase_supercritical_gas = CP.iphase_supercritical_gas
# iphase_supercritical_liquid = CP.iphase_supercritical_liquid
# iphase_twophase = CP.iphase_twophase
# iphase_unknown = CP.iphase_unknown

# # Statically add INPUT fields to the module (IDE autocomplete)
# QT_INPUTS = CP.QT_INPUTS
# PQ_INPUTS = CP.PQ_INPUTS
# QSmolar_INPUTS = CP.QSmolar_INPUTS
# QSmass_INPUTS = CP.QSmass_INPUTS
# HmolarQ_INPUTS = CP.HmolarQ_INPUTS
# HmassQ_INPUTS = CP.HmassQ_INPUTS
# DmolarQ_INPUTS = CP.DmolarQ_INPUTS
# DmassQ_INPUTS = CP.DmassQ_INPUTS
# PT_INPUTS = CP.PT_INPUTS
# DmassT_INPUTS = CP.DmassT_INPUTS
# DmolarT_INPUTS = CP.DmolarT_INPUTS
# HmolarT_INPUTS = CP.HmolarT_INPUTS
# HmassT_INPUTS = CP.HmassT_INPUTS
# SmolarT_INPUTS = CP.SmolarT_INPUTS
# SmassT_INPUTS = CP.SmassT_INPUTS
# TUmolar_INPUTS = CP.TUmolar_INPUTS
# TUmass_INPUTS = CP.TUmass_INPUTS
# DmassP_INPUTS = CP.DmassP_INPUTS
# DmolarP_INPUTS = CP.DmolarP_INPUTS
# HmassP_INPUTS = CP.HmassP_INPUTS
# HmolarP_INPUTS = CP.HmolarP_INPUTS
# PSmass_INPUTS = CP.PSmass_INPUTS
# PSmolar_INPUTS = CP.PSmolar_INPUTS
# PUmass_INPUTS = CP.PUmass_INPUTS
# PUmolar_INPUTS = CP.PUmolar_INPUTS
# HmassSmass_INPUTS = CP.HmassSmass_INPUTS
# HmolarSmolar_INPUTS = CP.HmolarSmolar_INPUTS
# SmassUmass_INPUTS = CP.SmassUmass_INPUTS
# SmolarUmolar_INPUTS = CP.SmolarUmolar_INPUTS
# DmassHmass_INPUTS = CP.DmassHmass_INPUTS
# DmolarHmolar_INPUTS = CP.DmolarHmolar_INPUTS
# DmassSmass_INPUTS = CP.DmassSmass_INPUTS
# DmolarSmolar_INPUTS = CP.DmolarSmolar_INPUTS
# DmassUmass_INPUTS = CP.DmassUmass_INPUTS
# DmolarUmolar_INPUTS = CP.DmolarUmolar_INPUTS

# # Define dictionary with dynamically generated fields
# PHASE_INDEX = {attr: getattr(CP, attr) for attr in dir(CP) if attr.startswith("iphase")}
# INPUT_PAIRS = {attr: getattr(CP, attr) for attr in dir(CP) if attr.endswith("_INPUTS")}
# INPUT_PAIRS = sorted(INPUT_PAIRS.items(), key=lambda x: x[1])


def postprocess_ode(t, y, ode_handle):
    """
    Post-processes the output of an ordinary differential equation (ODE) solver.

    This function takes the time points and corresponding ODE solution matrix,
    and for each time point, it calls a user-defined ODE handling function to
    process the state of the ODE system. It collects the results into a
    dictionary where each key corresponds to a state variable and the values
    are numpy arrays of that state variable at each integration step.

    Parameters
    ----------
    t : array_like
        Integration points at which the ODE was solved, as a 1D numpy array.
    y : array_like
        The solution of the ODE system, as a 2D numpy array with shape (n,m) where
        n is the number of points and m is the number of state variables.
    ode_handle : callable
        A function that takes in a integration point and state vector and returns a tuple,
        where the first element is ignored, the second is a dictionary representing the processed state,
        and the third is a flag (0 or 1) indicating singularity.

    Returns
    -------
    ode_out : dict
        A dictionary where each key corresponds to a state variable and each value
        is a numpy array containing the values of that state variable at each integration step.
        Also includes "singularity_detected" key with a boolean flag.
    """
    # Initialize ode_out as a dictionary
    ode_out = {}

    for t_i, y_i in zip(t, y.T):
        _, out = ode_handle(t_i, y_i)

        for key, value in out.items():
            # Initialize with an empty list
            if key not in ode_out:
                ode_out[key] = []
            # Append the value to list of current key
            ode_out[key].append(value)

    # Convert lists to numpy arrays
    for key in ode_out:
        ode_out[key] = np.array(ode_out[key])

    return ode_out


# Post-process when an autonomous system is solved:
# the output of the solver is only y and not t,y
def postprocess_ode_autonomous(t, y, ode_handle):
    """
    """
    # Initialize ode_out as a dictionary
    ode_out = {}
 
    for t_i, y_i in zip(t, y.T):
        _, out = ode_handle(y_i)
 
        for key, value in out.items():
            # Initialize with an empty list
            if key not in ode_out:
                ode_out[key] = []
            # Append the value to list of current key
            ode_out[key].append(value)
 
    # Convert lists to numpy arrays
    for key in ode_out:
        ode_out[key] = np.array(ode_out[key])
 
    return ode_out
 


def get_linear_convergent_divergent(
    length,
    convergent_length,
    divergent_length,
    radius_in,
    radius_throat,
    radius_out,
    width,
    type,
):

    if type == "Planar":
        area_in = 2 * radius_in * width
        area_throat = 2 * radius_throat * width
        area_out = 2 * radius_out * width

        if length <= convergent_length:
            area_slope = (area_throat - area_in) / convergent_length
            area = area_in + area_slope * length

        elif length > convergent_length:
            area_slope = (area_out - area_throat) / divergent_length
            area = area_throat + area_slope * (length - convergent_length)

        radius = area / (2 * width)
        perimeter = 2 * width + 2 * 2 * radius

    return area, area_slope, perimeter, radius


def get_wall_friction(velocity, density, viscosity, roughness, diameter):
    """
    Computes the frictional stress at the wall of a pipe due to viscous effects.

    The function first calculates the Reynolds number to characterize the flow.
    It then uses the Haaland equation to find the Darcy-Weisbach friction factor.
    Finally, it calculates the wall shear stress using the Darcy-Weisbach equation.

    Parameters
    ----------
    velocity : float
        The flow velocity of the fluid in the pipe (m/s).
    density : float
        The density of the fluid (kg/m^3).
    viscosity : float
        The dynamic viscosity of the fluid (Pa·s or N·s/m^2).
    roughness : float
        The absolute roughness of the pipe's internal surface (m).
    diameter : float
        The inner diameter of the pipe (m).

    Returns
    -------
    stress_wall : float
        The shear stress at the wall due to friction (Pa or N/m^2).
    friction_factor : float
        The Darcy-Weisbach friction factor, dimensionless.
    reynolds : float
        The Reynolds number, dimensionless, indicating the flow regime.
    """
    reynolds = velocity * density * diameter / viscosity
    friction_factor = get_friction_factor_haaland(reynolds, roughness, diameter)
    stress_wall = (1 / 8) * friction_factor * density * velocity**2
    return stress_wall, friction_factor, reynolds


def get_friction_factor_haaland(reynolds, roughness, diameter):
    """
    Computes the Darcy-Weisbach friction factor using the Haaland equation.

    The Haaland equation provides an explicit formulation for the friction factor
    that is simpler to use than the Colebrook equation, with an acceptable level
    of accuracy for most engineering applications.
    This function implements the Haaland equation as it is presented in many fluid
    mechanics textbooks, such as "Fluid Mechanics Fundamentals and Applications"
    by Cengel and Cimbala (equation 12-93).

    Parameters
    ----------
    reynolds : float
        The Reynolds number, dimensionless, indicating the flow regime.
    roughness : float
        The absolute roughness of the pipe's internal surface (m).
    diameter : float
        The inner diameter of the pipe (m).

    Returns
    -------
    f : float
        The computed friction factor, dimensionless.
    """
    # f = 6
    f = (-1.8 * np.log10(6.9 / abs(reynolds) + (roughness / diameter / 3.7) ** 1.11)) ** -2
    return f


# The method pipeline_steady_state_1D computes the critical mass flow rate with the Possible-Impossible flow (PIF) algorithm (if specified, else one can directly impose the mass flow/velocity at the inlet)
# and integrate in space to find the adapted supersonic solution. The PIF algorithm is the simplest among the ones available in the
# iterature and the ODE is a conventional ODE system and not an autonomous system of equations.

def mixture_prop(prop1, prop2, alpha1, alpha2):
    return prop1 * alpha1 + prop2 * alpha2



def mixture_speed_of_sound(alpha_g, alpha_l, rho_g, rho_l, a_g, a_l, u_g, u_l, k=0.01):
    """
    Calculate the speed of sound in a two-phase mixture using Eqs. (5.37)-(5.38).
    
    Parameters:
        alpha_g : float  # Gas volume fraction
        alpha_l : float  # Liquid volume fraction (should satisfy alpha_g + alpha_l = 1)
        rho_g   : float  # Gas density
        rho_l   : float  # Liquid density
        a_g     : float  # Speed of sound in gas
        a_l     : float  # Speed of sound in liquid
        u_g     : float  # Gas velocity
        u_l     : float  # Liquid velocity
        k       : float  # Model parameter
        
    Returns:
        a : float  # Mixture speed of sound
    """
    # Mixture density
    rho = alpha_g * rho_g + alpha_l * rho_l

    # Compute tilde(a)^2
    first_term_numerator = alpha_g * rho_l + alpha_l * rho_g
    first_term_denominator = (alpha_g * rho_l / (a_g**2)) + (alpha_l * rho_g / (a_l**2))
    first_term = first_term_numerator / first_term_denominator

    second_term_numerator = 1 + k * (alpha_g * rho_g + alpha_l * rho_l) / (alpha_g * rho_l + alpha_l * rho_g)
    second_term_denominator = 1 + k * (rho**2) / (rho_g * rho_l)
    second_term = second_term_numerator / second_term_denominator

    a_tilde_sq = first_term * second_term

    # Compute Δa^2
    delta_a_sq = (alpha_g * alpha_l * rho_g * rho_l * (u_g - u_l)**2 *
                  ((rho_l + k * rho) * (rho_g + k * rho)) /
                  ((rho_g * rho_l + k * rho**2)**2))

    # Final mixture speed of sound
    a_sq = a_tilde_sq - delta_a_sq
    return jnp.sqrt(max(a_sq, 0.0))  


# TODO: solver is not putting equispaced points, check why
# TODO: insert sourcve terms

def pipeline_steady_state_1D_two_component(
    fluid_name_1,
    fluid_name_2,
    pressure_in_1,
    pressure_in_2,
    temperature_in_1,
    temperature_in_2,
    properties_in_1,
    properties_in_2,
    area_1,
    area_2,
    m_dot_in_1,
    m_dot_in_2,
    mixture_ratio, # m_dot_liquid / m_dot_gas
    convergent_length,
    nozzle_length,
    number_of_points,
    source_terms,
    divergent_length = None,
    roughness = None,
    radius_in = None,
    radius_throat = None,
    radius_out = None,
    nozzle_type = None,
    width = None,
    mass_flow=None,
    mach_in=None,
    critical_flow=False,
    include_friction=True,
    include_heat_transfer=False,
    temperature_external=None,
):
    # # Check for correct inputs
    # if (mass_flow is None and mach_in is None and critical_flow is False) or (
    #     mass_flow is not None and mach_in is not None
    # ):
    #     raise ValueError("Check input settins for the velocity.")

    t_evaluation = 0

    # fluid = rg.Fluid(fluid_name, backend="HEOS", exceptions=True)
    fluid1 = jxp.FluidJAX(fluid_name_1, backend="HEOS")
    fluid2 = jxp.FluidJAX(fluid_name_2, backend="HEOS")

    # Define inlet area and length of the nozzle
    # total_length = convergent_length + divergent_length

    if nozzle_type == "Planar":
        area_in = 2 * radius_in * width
    elif nozzle_type == "Axisymmetric":
        area_in = np.pi * radius_in**2

    # Calculate inlet density
    q_in = m_dot_in_2 / (m_dot_in_1 + m_dot_in_2) 
    prop_in_1 = fluid1.get_state(jxp.PT_INPUTS, pressure_in_1, temperature_in_1)
    prop_in_2 = fluid2.get_state(jxp.PT_INPUTS, pressure_in_2, temperature_in_2)
    
    rho1_in = prop_in_1.rho
    rho2_in = prop_in_2.rho
    h1_in = prop_in_1.h
    h2_in = prop_in_2.h

    alpha2_in = 1 / (1 + ((1 - q_in) / q_in) * (rho2_in / rho1_in))
    alpha1_in = 1 - alpha2_in
    print(f"Void fraction. Nitrogen:{alpha2_in}, Water:{alpha1_in}")

    # Calculate velocity intlet from the mass flows
    A_1_in = 61 * jnp.pi * (4.57 * 1e-3) ** 2 / 4
    u_in_1 = m_dot_in_1 / (rho1_in * A_1_in)

    A_inlet_external = jnp.pi * (50.96 * 1e-3)**2 / 4
    A_inlet_internal = 61 * jnp.pi * (6.35 * 1e-3) ** 2 / 4
    A_2_in = A_inlet_external - A_inlet_internal

    u_in_2 = m_dot_in_2 / (rho2_in * A_2_in)
    u_in = u_in_1 * alpha1_in + u_in_2 * alpha2_in

    # Initialize out_list to store all the state outputs
    out_list = []



    # Define the ODE system
    def odefun(y):
        x, alpha1, alpha2, rho1, rho2, u, p, h1, h2 = (
            y  # void fraction, density, velocity, pressure, enthalpy
        )
        # print(x)
    


        try:

            # Calculate area and geometry properties for convergent-divergent nozzles
            area, area_slope, perimeter, radius = get_linear_convergent_divergent(
                length=x,
                type="elliot",
            )
            diameter = radius * 2           
            # print(f"r={radius}, x={x}, alpha1={alpha1}, alpha2={alpha2}, rho1={rho1}, rho2={rho2}, u={u}, p={p}, h1={h1}, h2={h2}")
 

            k = 0.5
            c = k
            
            # Fluid 1
            state1 = fluid1.get_state(jxp.HmassP_INPUTS, h1, p)
            viscosity_1 = state1["mu"]
            # drho_dP_1 = state1["drho_dP"]
            # drho_dh_1 = state1["drho_dh"]
            G1 = state1["gruneisen"]
            c1 = state1["speed_of_sound"]
            T1 = state1["T"]



            # Fluid 2
            state2 = fluid2.get_state(jxp.HmassP_INPUTS, h2, p)
            viscosity_2 = state2["mu"]
            # drho_dP_2 = state2["drho_dP"]
            # drho_dh_2 = state2["drho_dh"]
            G2 = state2["gruneisen"]
            c2 = state2["speed_of_sound"]
            T2 = state2["T"]

            rho_mix = rho1 * alpha1 + rho2 * alpha2
            viscosity_mix = viscosity_1 * alpha1 + viscosity_2 * alpha2



            # Wall friction calculations
            stress_wall, friction_factor, reynolds = get_wall_friction(
                velocity=u,
                density=rho_mix,
                viscosity=viscosity_mix,
                roughness=roughness,
                diameter=diameter,
            )

            if not include_friction:
                stress_wall = 0.0
                friction_factor = 0.0

            # Heat transfer (if applicable)
            if include_heat_transfer:
                U = 10  # To include correlations
            else:
                U = 0.0
                heat_in = 0

            M = np.asarray(
                [
                    [1.0,               1.0,               -0.0,               -0.0,               -0.0,                    -0.0,                    0.0,                    0.0],
                    [rho1 * u,          0.0,               alpha1 * u,        0.0,               alpha1 * rho1,          0.0,                    0.0,                    0.0],
                    [0.0,               rho2 * u,          0.0,               alpha2 * u,        alpha2 * rho2,          0.0,                    0.0,                    0.0],
                    [0.0,               0.0,               0.0,               0.0,               rho_mix * u,            1.0,                    0.0,                    0.0],
                    [0.0,               0.0,               0.0,               0.0,               alpha1 * rho1 * u**2,   0.0,                    alpha1 * rho1 * u,      0.0],
                    [0.0,               0.0,               0.0,               0.0,               alpha2 * rho2 * u**2,   0.0,                    0.0,                    alpha2 * rho2 * u],
                    [0.0,               0.0,              -1.0,               0.0,               0.0,                    (1 + G1) / c1**2,       -(rho1 * G1) / c1**2,    0.0],
                    [0.0,               0.0,               0.0,              -1.0,               0.0,                    (1 + G2) / c2**2,       0.0,                    -(rho2 * G2) / c2**2],
                ]
            )

            determinant = det(M)
            # print(f"Determinant is: {determinant}")

            # If singularity detected
            flag = 1

            # print("Determinant is:", determinant)
            # print("x is:", u2)

            if determinant < (1e-14):
                flag = 0
            
            # print(flag)

            gas = fluid1.get_state(jxp.HmassP_INPUTS, h1, p)
            a1 = gas.a
            liquid = fluid2.get_state(jxp.HmassP_INPUTS, h2, p)
            a2 = liquid.a

 
            a_mix = 1 ##
            # print("Warning: mixture speed of sound not calculated!")
            mach_mix = u/a_mix

            # print("Mixture mach is:", mach_mix)


            # b = np.asarray(
            #     [
            #         0.0,
            #         0.0,
            #         0.0,
            #         source_terms,
            #         -source_terms,
            #         source_terms,
            #         -source_terms,
            #         0.0,
            #         0.0
            #     ]
            # )

            area_term = (1/area) * area_slope

            Ai, d_2 = interfacial_area(alpha2)

            # print("Warning: Heat transfer coefficient not true!!!")
            ht_1 = 6 # water
            Nu_2 = 12
            k_2 = state2["k"]
            ht_2 = (k_2 * Nu_2) / d_2 # nitrogen
            ht_21 = (1 / ((1/ht_1) + (1/ht_2))) * 100



            b = np.asarray(
                [
                    -0.0,
                    -area_term * alpha1 * rho1 *  u,
                    -area_term * alpha2 * rho2 *  u,
                    -(stress_wall * perimeter) / area,
                    ht_21 * Ai * (T2 - T1),
                    ht_21 * Ai * (T1 - T2),
                    0.0,
                    0.0
                ]
            )

            M1 = M.copy()
            M1[:, 0] = b
            M2 = M.copy()
            M2[:, 1] = b
            M3 = M.copy()
            M3[:, 2] = b
            M4 = M.copy()
            M4[:, 3] = b
            M5 = M.copy()
            M5[:, 4] = b
            M6 = M.copy()
            M6[:, 5] = b
            M7 = M.copy()
            M7[:, 6] = b
            M8 = M.copy()
            M8[:, 7] = b
            
            
            # Compute determinants
            det_D = det(M)
            det_N1 = det(M1)
            det_N2 = det(M2)
            det_N3 = det(M3)
            det_N4 = det(M4)
            det_N5 = det(M5)
            det_N6 = det(M6)
            det_N7 = det(M7)
            det_N8 = det(M8)

            # print("det 1", det_D, "position", x-141.63e-3)
            # print("det 2", det_N1)
            # print("det 3", det_N2)
            # print("det 4", det_N3)
            # print("det 5", det_N4)
            # print(" ")

            dy = [det_D, det_N1, det_N2, det_N3, det_N4, det_N5, det_N6, det_N7, det_N8]


            # Save the output at each step in the dictionary
            out = {
                "distance": x,
                "alpha_1": alpha1,
                "alpha_2": alpha2,
                "density_1": rho1,
                "density_2": rho2,
                "density_mix": rho_mix,
                "velocity": u,
                "pressure": p,
                "temperature_1": T1,
                "temperature_2": T2,
                "speed_of_sound_1": np.nan if state1 is None else state1["a"],
                "speed_of_sound_2": state2["a"],
                "viscosity_1": np.nan if state1 is None else state1["mu"],
                "viscosity_2": state2["mu"],
                "enthalpy_1": h1,
                "enthalpy_2": h2,
                "entropy_1": np.nan if state1 is None else state1["s"],
                "entropy_2": state2["s"],
                "total_enthalpy_1": h1 + 0.5 * u**2,
                "total_enthalpy_2": h2 + 0.5 * u**2,
                "mach_number_1": u / a1,
                "mach_number_2": u / a2,
                "mach_mix": mach_mix,
                "mass_flow_1": u * rho1 * area * alpha1,
                "mass_flow_2": u * rho2 * area * alpha2,
                "area": area,
                "area_slope": area_slope,
                "perimeter": perimeter,
                "diameter": diameter,
                "stress_wall": stress_wall,
                "friction_factor": friction_factor,
                "reynolds": reynolds,
                "source_1": b[0],
                "source_2": b[1],
                "source_3": b[2],
                "source_4": b[3],
                "source_5": b[4],
                "source_6": b[5],
                "source_7": b[6],
                "source_8": b[7],
                "determinant": det_D,
                "flag": flag,
            }


            # Append the output dictionary to out_list
            out_list.append(out)

            return dy, out

        except Exception as e:
            # print(f"[ODEFUN ERROR @ x={y[0]:.4f}] {e}")
            return [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan], 0  # forces integrator to stop

    def stop_at_singularity(t, y):
        # print(odefun(t, y))
        _, out = odefun(y)
        # print(out['flag'])
        flag = out["flag"]
        return flag

    stop_at_singularity.terminal = True
    stop_at_singularity.direction = 1

    def stop_at_length(t, y):
        z = y[0]               
        return z - nozzle_length     
    stop_at_length.terminal = True     
    stop_at_length.direction = 0

    def stop_at_inlet(t, y):
        x = y[0]
        return x  # Will be zero or negative when at or past the inlet
    stop_at_inlet.terminal = True  # Stop the solver when event is triggered
    stop_at_inlet.direction = -1

    def stop_at_zero_det(t, y):
        _, out = odefun(y)
        det_M = out["determinant"]
        # print(f"t={t:.5f}, det={det_M:.5e}")
        return det_M
    stop_at_zero_det.terminal = True  
    stop_at_zero_det.direction = 0   

    # raw_solution = scipy.integrate.solve_ivp(
    #     lambda t, y: odefun(y)[0],
    #     t_span = [0.0, 1.0],
    #     y0 = [0.00, alpha1_in, alpha2_in, rho1_in, rho2_in, u_in_1, u_in_2, pressure_in_1, h1_in, h2_in],
    #     # t_eval=np.linspace(0, nozzle_length, number_of_points),
    #     method="RK45", #"BDF"
    #     rtol=1e-6,
    #     atol=1e-6,
    #     # max_step=nozzle_length/(number_of_points*10),
    #     # events=[stop_at_inlet, stop_at_length],
    # )
    # solution = postprocess_ode_autonomous(raw_solution.t, raw_solution.y, odefun)

    pif_iterations = 0
    print("Possible-Impossible Flow (PIF) algorithm starts...")
    u_possible = 2.88242932
    u_impossible = 2.88242932 # 2.8771  # 2.878

    u_guess = (u_impossible + u_possible) / 2
    error = abs(u_impossible-u_possible)/abs(u_possible)
    

    tol = 1e-9
    while error > tol:
        pif_iterations += 1  
        raw_solution = scipy.integrate.solve_ivp(
            lambda t, y: odefun(y)[0],
            [0, 1],
            y0 = [1e-9, alpha1_in, alpha2_in, rho1_in, rho2_in, u_guess, pressure_in_1, h1_in, h2_in],
            # t_eval=np.linspace(0, convergent_length, number_of_points) if number_of_points else None,
            method="LSODA",
            rtol=1e-9,
            atol=1e-9,
            # first_step=1e-9, 
            max_step=1e-9,
            events=[stop_at_zero_det, stop_at_length, stop_at_inlet]
        )
        solution = postprocess_ode_autonomous(raw_solution.t, raw_solution.y, odefun)

        if raw_solution.t_events[0].size > 0:
            u_impossible = u_guess
        else:
            u_possible = u_guess

        u_guess = (u_impossible + u_possible) / 2
        error = abs(u_impossible-u_possible)/u_possible
        print(u_impossible)
        print(u_possible)
        print(pif_iterations)

    # flow_rate = u_guess*area_in*density_in
    print("Final solution of the subsonic part...")
    print(f"Velocity is:    {u_possible}")
    # Calculate the solution with the last possible flow rate calculated
    possible_solution = scipy.integrate.solve_ivp(
        lambda t, y: odefun(y)[0],
        [0, 1],
        y0 = [1e-9, alpha1_in, alpha2_in, rho1_in, rho2_in, u_possible, pressure_in_1, h1_in, h2_in],
        # t_eval=np.linspace(0, convergent_length, number_of_points) if number_of_points else None,
        method="LSODA",
        rtol=1e-9,
        atol=1e-9,
        events=[stop_at_zero_det, stop_at_length, stop_at_inlet]
    )
    solution = postprocess_ode_autonomous(possible_solution.t, possible_solution.y, odefun)
    

    # supersonic_solution = scipy.integrate.solve_ivp(
    #         lambda t, y: odefun(y)[0],
    #         [0, 1], # Inverting dummy variable limits so you so not go backward at the singularity (all the determinants<0)
    #         [x,
    #          velocity, 
    #          solution["density"][-1], 
    #          solution["pressure"][-1]],
    #         t_eval=np.linspace(0, convergent_length, number_of_points) if number_of_points else None,
    #         method="RK45",
    #         rtol=1e-9,
    #         atol=1e-9,
    #         events=[stop_at_length, stop_at_zero]
    #     ) 
    print("Solving the supersonic part...")

    x = solution["distance"][-1]*1.001
    throat_position = (42.23 + 99.40)*1e-3
    if solution["distance"][-1] < throat_position:
        location = "convergent"
    else:
        location = "divergent"

    # Calculate area
    # from mass flow rate and area, you calculate the velocity
    # x = 142e-3
    area, _, _, _ = get_linear_convergent_divergent(x)
    v_mix = (solution["mass_flow_1"][-1] + solution["mass_flow_2"][-1]) / (area * solution["density_mix"][-1])

    print(f"Final x is in the {location}")

    # Calculate the supersonic branch
    supersonic_solution = scipy.integrate.solve_ivp(
        lambda t, y: odefun(y)[0],
        [1, 0],
        [x,
        solution["alpha_1"][-1],
        solution["alpha_2"][-1],
        solution["density_1"][-1],
        solution["density_2"][-1],
        v_mix,
        solution["pressure"][-1],
        solution["enthalpy_1"][-1],
        solution["enthalpy_2"][-1]],
        method="LSODA",
        rtol=1e-9,
        atol=1e-9,
        # first_step=1e-9,
        # max_step=1e-6,
        events=[stop_at_length]
    )
    supersonic_solution = postprocess_ode_autonomous(supersonic_solution.t, supersonic_solution.y, odefun)

    return (
        supersonic_solution,
        solution,
        # raw_solution,
        out_list,
        x
    )



def save_selected_to_csv(out_dict, keys_to_save, filename):
    """
    Save selected entries from a simulation output dictionary to output/filename.

    Parameters:
    - out_dict: dict containing simulation data
    - keys_to_save: list of keys to include in the CSV
    - filename: name of the CSV file only (e.g., "run1.csv")
    """
    # Ensure 'output/' directory exists
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    # Full path to the output file
    full_path = os.path.join(output_dir, filename)

    # Expand scalars to match the length of the first non-scalar entry
    first_key = next(key for key in keys_to_save if not isinstance(out_dict[key], (int, float)))
    length = len(out_dict[first_key])

    filtered = {}
    for key in keys_to_save:
        val = out_dict[key]
        filtered[key] = [val] * length if isinstance(val, (int, float)) else val

    # Save to CSV
    df = pd.DataFrame(filtered)
    df.to_csv(full_path, index=False)

def save_all_to_csv(out_dict, filename):
    """
    Save all entries from a simulation output dictionary to output/filename.

    Parameters:
    - out_dict: dict containing simulation data
    - filename: name of the CSV file only (e.g., "run1.csv")
    """
    # Ensure 'output/' directory exists
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    # Full path to the output file
    full_path = os.path.join(output_dir, filename)

    # Find first non-scalar entry to determine length
    first_key = next(key for key in out_dict if not isinstance(out_dict[key], (int, float)))
    length = len(out_dict[first_key])

    # Expand scalars to match the length
    filtered = {}
    for key, val in out_dict.items():
        filtered[key] = [val] * length if isinstance(val, (int, float)) else val

    # Save to CSV
    df = pd.DataFrame(filtered)
    df.to_csv(full_path, index=False)

def get_linear_convergent_divergent_old(
    length,
    total_length=150.93*1e-3,
    convergent_length=99.40*1e-3,
    divergent_length=97.18*1e-3,
    radius_in=50.96/2*1e-3,
    radius_throat=13.12/2*1e-3,
    radius_out=22.21*1e-3,
    width=None,
    type="elliot",
):

    if type == "Planar":
        area_in = 2 * radius_in * width
        area_throat = 2 * radius_throat * width
        area_out = 2 * radius_out * width

        if length <= convergent_length:
            area_slope = (area_throat - area_in) / convergent_length
            area = area_in + area_slope * length

        elif length > convergent_length:
            area_slope = (area_out - area_throat) / divergent_length
            area = area_throat + area_slope * (length - convergent_length)

        radius = area / (2 * width)
        perimeter = 2 * width + 2 * 2 * radius

    elif type == "elliot":
        # Dimensions comes for Amit's 3D scketch for turbo expo 2024
        throat_second_lenght = 11.51 * 1e-3
        throat_first_lenght = 42.23 * 1e-3
        radius_throat_in = 15.90/2 * 1e-3
        radius_throat_out = 13.72/2 * 1e-3
        area_in = jnp.pi * radius_in ** 2
        area_throat_in = jnp.pi * radius_throat_in ** 2
        area_throat_out = jnp.pi * radius_throat_out ** 2
        area_throat = jnp.pi * radius_throat ** 2
        area_out = jnp.pi * radius_out ** 2


        if length <= convergent_length:
            area_slope = (area_throat_in - area_in) / convergent_length
            area = area_in + area_slope * length

        # Throat section convergent
        elif convergent_length < length <= convergent_length + throat_first_lenght:
            area_slope = (area_throat - area_throat_in) / throat_first_lenght
            area = area_throat_in + area_slope * (length - convergent_length)

        # Throat section divergent
        elif convergent_length + throat_first_lenght < length <= convergent_length + throat_first_lenght + throat_second_lenght:
            area_slope = (area_throat_out - area_throat) / throat_second_lenght
            area = area_throat + area_slope * (length - convergent_length - throat_first_lenght)

                # Throat section divergent
        elif length > convergent_length + throat_first_lenght + throat_second_lenght:
            area_slope = (area_out - area_throat_out) / divergent_length
            area = area_throat_out + area_slope * (length - convergent_length - throat_first_lenght - throat_second_lenght)

        radius = jnp.sqrt(area / jnp.pi)
        perimeter = 2 * jnp.pi * radius

    return area, area_slope, perimeter, radius

def get_linear_convergent_divergent(
    length,
    total_length=150.93*1e-3,
    convergent_length=99.40*1e-3,
    divergent_length=97.18*1e-3,
    radius_in=50.96/2*1e-3,
    radius_throat=13.12/2*1e-3,
    radius_out=22.21*1e-3,
    width=None,
    type="elliot",
):

    if type == "Planar":
        area_in = 2 * radius_in * width
        area_throat = 2 * radius_throat * width
        area_out = 2 * radius_out * width

        if length <= convergent_length:
            area_slope = (area_throat - area_in) / convergent_length
            area = area_in + area_slope * length

        elif length > convergent_length:
            area_slope = (area_out - area_throat) / divergent_length
            area = area_throat + area_slope * (length - convergent_length)

        radius = area / (2 * width)
        perimeter = 2 * width + 2 * 2 * radius

    elif type == "elliot":
        # Dimensions comes for Amit's 3D scketch for turbo expo 2024
        throat_second_lenght = 11.51 * 1e-3
        throat_first_length = 42.23 * 1e-3
        radius_throat_in = 15.90/2 * 1e-3
        radius_throat_out = 13.72/2 * 1e-3
        area_in = jnp.pi * radius_in ** 2
        area_throat_in = jnp.pi * radius_throat_in ** 2
        area_throat_out = jnp.pi * radius_throat_out ** 2
        area_throat = jnp.pi * radius_throat ** 2
        area_out = jnp.pi * radius_out ** 2


        if length <= convergent_length:
            # area_slope = 2*np.pi*((radius_in+((radius_throat_in-radius_in)/convergent_length)*length)*((radius_throat_in-radius_in)/convergent_length))
            area = np.pi*((radius_in+((radius_throat_in-radius_in)/convergent_length)*length)**2)
            radius = jnp.sqrt(area / jnp.pi)
            perimeter = 2 * jnp.pi * radius
            area_slope = perimeter * (radius_throat_in-radius_in)/convergent_length

        # Throat section convergent
        elif convergent_length < length <= convergent_length + throat_first_length:
            length = length-convergent_length
            area = np.pi*((radius_throat_in+((radius_throat-radius_throat_in)/throat_first_length)*length)**2)
            radius = jnp.sqrt(area / jnp.pi)
            perimeter = 2 * jnp.pi * radius
            area_slope = perimeter * (radius_throat-radius_throat_in)/throat_first_length

        # Throat section divergent
        elif convergent_length + throat_first_length < length <= convergent_length + throat_first_length + throat_second_lenght:
            length = length - convergent_length - throat_first_length
            # area_slope = 2*np.pi*((radius_throat+((radius_throat_out-radius_throat)/throat_second_lenght))*length*((radius_throat_out-radius_throat)/throat_second_lenght))
            area = np.pi*((radius_throat+((radius_throat_out-radius_throat)/throat_second_lenght)*length)**2)
            radius = jnp.sqrt(area / jnp.pi)
            perimeter = 2 * jnp.pi * radius
            area_slope = perimeter * ((radius_throat_out-radius_throat)/throat_second_lenght)

        # Throat section divergent
        # elif length > convergent_length + throat_first_length + throat_second_lenght:
        else:
            length = length - convergent_length - throat_first_length - throat_second_lenght
            # area_slope = 2*np.pi*((radius_throat_out+((radius_out-radius_throat_out)/divergent_length))*length)*((radius_out-radius_throat_out)/divergent_length)
            area = np.pi*((radius_throat_out+((radius_out-radius_throat_out)/divergent_length)*length)**2)
            radius = jnp.sqrt(area / jnp.pi)
            perimeter = 2 * jnp.pi * radius
            area_slope = perimeter * ((radius_out-radius_throat_out)/divergent_length)           

        # radius = jnp.sqrt(area / jnp.pi)
        # perimeter = 2 * jnp.pi * radius

    return area, area_slope, perimeter, radius

def interfacial_area(alpha, alpha_b=0.3, alpha_d=0.7, Nb=1e10, Nd=1e8):
    """
    Compute Ai and equivalent spherical diameter based on alpha.
    Returns (Ai, d), where d is the equivalent spherical diameter.
    """

    # --- Bubbly regime ---
    if alpha <= alpha_b:
        A_ib = (6 * alpha)**(2/3) * (np.pi * Nb)**(1/3)
        Ai = A_ib
        N = Nb

    # --- Droplet regime ---
    elif alpha >= alpha_d:
        A_id = (6 * (1 - alpha))**(2/3) * (np.pi * Nd)**(1/3)
        Ai = A_id
        N = Nd

    # --- Transition regime (linear interpolation) ---
    else:
        A_ib_b = (6 * alpha_b)**(2/3) * (np.pi * Nb)**(1/3)
        A_id_d = (6 * (1 - alpha_d))**(2/3) * (np.pi * Nd)**(1/3)
        Ai = A_ib_b + (A_id_d - A_ib_b) * (alpha - alpha_b) / (alpha_d - alpha_b)
        
        # Smooth transition in N as well (optional)
        N = Nb + (Nd - Nb) * (alpha - alpha_b) / (alpha_d - alpha_b)

    # --- Equivalent spherical diameter ---
    d = np.sqrt(Ai / (N * np.pi))
    
    return Ai, d
