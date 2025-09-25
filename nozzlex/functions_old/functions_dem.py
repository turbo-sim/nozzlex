import scipy.linalg
import scipy.integrate
import numpy as np
import matplotlib as mpl
from cycler import cycler
import numpy as np
from scipy.linalg import det
import CoolProp.CoolProp as CP
import jaxprop as jxp
import pandas as pd
import os
import barotropy as bpy



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

# Statically add phase indices to the module (IDE autocomplete)
iphase_critical_point = CP.iphase_critical_point
iphase_gas = CP.iphase_gas
iphase_liquid = CP.iphase_liquid
iphase_not_imposed = CP.iphase_not_imposed
iphase_supercritical = CP.iphase_supercritical
iphase_supercritical_gas = CP.iphase_supercritical_gas
iphase_supercritical_liquid = CP.iphase_supercritical_liquid
iphase_twophase = CP.iphase_twophase
iphase_unknown = CP.iphase_unknown

# Statically add INPUT fields to the module (IDE autocomplete)
QT_INPUTS = CP.QT_INPUTS
PQ_INPUTS = CP.PQ_INPUTS
QSmolar_INPUTS = CP.QSmolar_INPUTS
QSmass_INPUTS = CP.QSmass_INPUTS
HmolarQ_INPUTS = CP.HmolarQ_INPUTS
HmassQ_INPUTS = CP.HmassQ_INPUTS
DmolarQ_INPUTS = CP.DmolarQ_INPUTS
DmassQ_INPUTS = CP.DmassQ_INPUTS
PT_INPUTS = CP.PT_INPUTS
DmassT_INPUTS = CP.DmassT_INPUTS
DmolarT_INPUTS = CP.DmolarT_INPUTS
HmolarT_INPUTS = CP.HmolarT_INPUTS
HmassT_INPUTS = CP.HmassT_INPUTS
SmolarT_INPUTS = CP.SmolarT_INPUTS
SmassT_INPUTS = CP.SmassT_INPUTS
TUmolar_INPUTS = CP.TUmolar_INPUTS
TUmass_INPUTS = CP.TUmass_INPUTS
DmassP_INPUTS = CP.DmassP_INPUTS
DmolarP_INPUTS = CP.DmolarP_INPUTS
HmassP_INPUTS = CP.HmassP_INPUTS
HmolarP_INPUTS = CP.HmolarP_INPUTS
PSmass_INPUTS = CP.PSmass_INPUTS
PSmolar_INPUTS = CP.PSmolar_INPUTS
PUmass_INPUTS = CP.PUmass_INPUTS
PUmolar_INPUTS = CP.PUmolar_INPUTS
HmassSmass_INPUTS = CP.HmassSmass_INPUTS
HmolarSmolar_INPUTS = CP.HmolarSmolar_INPUTS
SmassUmass_INPUTS = CP.SmassUmass_INPUTS
SmolarUmolar_INPUTS = CP.SmolarUmolar_INPUTS
DmassHmass_INPUTS = CP.DmassHmass_INPUTS
DmolarHmolar_INPUTS = CP.DmolarHmolar_INPUTS
DmassSmass_INPUTS = CP.DmassSmass_INPUTS
DmolarSmolar_INPUTS = CP.DmolarSmolar_INPUTS
DmassUmass_INPUTS = CP.DmassUmass_INPUTS
DmolarUmolar_INPUTS = CP.DmolarUmolar_INPUTS

# Define dictionary with dynamically generated fields
PHASE_INDEX = {attr: getattr(CP, attr) for attr in dir(CP) if attr.startswith("iphase")}
INPUT_PAIRS = {attr: getattr(CP, attr) for attr in dir(CP) if attr.endswith("_INPUTS")}
INPUT_PAIRS = sorted(INPUT_PAIRS.items(), key=lambda x: x[1])

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
    singularity_detected = False

    for t_i, y_i in zip(t, y.T):
        _, out, flag = ode_handle(t_i, y_i)

        if flag == 1:
            singularity_detected = True
        
        for key, value in out.items():
            # Initialize with an empty list
            if key not in ode_out:
                ode_out[key] = []
            # Append the value to list of current key
            ode_out[key].append(value)

    # Convert lists to numpy arrays
    for key in ode_out:
        ode_out[key] = np.array(ode_out[key])

    # Add singularity info
    ode_out["singularity_detected"] = singularity_detected

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


def get_linear_convergent_divergent(z_coordinate, convergent_length, divergent_length, radius_in, radius_throat, radius_out, width, type):

    if type == "Planar":
        area_in = 2*radius_in*width
        area_throat = 2*radius_throat*width
        area_out = 2*radius_out*width

        dAdz_div = (area_out - area_throat)/divergent_length
        dAdz_conv = (area_throat - area_in)/convergent_length

        if z_coordinate <= convergent_length:
            area_slope = (area_throat-area_in)/convergent_length
            area = area_in + area_slope*z_coordinate
        else:
            area_slope = (area_out-area_throat)/divergent_length
            area = area_throat + area_slope*(z_coordinate-convergent_length)

        radius = area/(2*width)
        perimeter = 2 * (width + 2 * radius)

    return area, area_slope, perimeter, radius, dAdz_div, dAdz_conv


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
    f = (-1.8 * np.log10(6.9 / reynolds + (roughness / diameter / 3.7) ** 1.11)) ** -2
    return f


# The method pipeline_steady_state_1D computes the critical mass flow rate with the Possible-Impossible flow (PIF) algorithm (if specified, else one can directly impose the mass flow/velocity at the inlet)
# and integrate in space to find the adapted supersonic solution. The PIF algorithm is the simplest among the ones available in the 
# iterature and the ODE is a conventional ODE system and not an autonomous system of equations.

def dem_term_angielczyk_1(perimeter, p, y, p_sat_T_LM, p_cr, area):

    ## Angielczyk tuned the correlation for water with this coefficients for CO2
    C_1 = 5.17
    C_2 = 0.87
    C_3 = 0.25

    ## Accoding to Angielczyk, correlation originally developed for water!!
    # C_1 = 0.008390
    # C_2 = 0.633691
    # C_3 = 0.228127

    base = (p_sat_T_LM - p) / (p_cr - p_sat_T_LM)

    f = (C_1 * perimeter / area + C_2) * (1 - y) * base ** C_3

    return f

def dem_term_angielczyk_2(p, y, p_sat_T_LM, p_cr, dAdz_div, dAdz_conv):

    ## New correlation developed by Angielczyk for CO2
    C_1 = 38
    C_2 = 1.291e-31
    C_3 = 75.28
    C_4 = -0.22

    dAdz_ref = 0.0801424e-4

    f = (C_1 + C_2*np.exp(C_3*(dAdz_div - dAdz_conv)/(dAdz_ref - dAdz_conv)))*(1-y)*((p_sat_T_LM - p) / (p_cr - p_sat_T_LM))**C_4 
    
    return f

def pipeline_steady_state_1D(
    fluid_name,
    pressure_in,
    temperature_in,
    properties_in,
    convergent_length,
    divergent_length,
    roughness,
    radius_in,
    radius_throat,
    radius_out,
    nozzle_type,
    width,
    mass_flow=None,
    mach_in=None,
    critical_flow = False,
    include_friction=True,
    include_heat_transfer=False,
    temperature_external=None,
    number_of_points=None,
):
    # Check for correct inputs
    if (
        (mass_flow is None and mach_in is None and critical_flow is False) or 
        (mass_flow is not None and mach_in is not None) 
    ):
        raise ValueError(
            "Check input settins for the velocity."
        )
    
    fluid = jxp.Fluid(fluid_name, backend="HEOS")
    state_in = fluid.get_state(jxp.PT_INPUTS, pressure_in, temperature_in)
    p_cr = fluid.critical_point.p
    s_meta = state_in.s  # The metastable phase is assumed to follow an isentropi expansion
    rhoT_guess = [state_in["rho"], state_in["T"]]

    
    # Define inlet area and length of the nozzle
    total_length = convergent_length+divergent_length
    if nozzle_type == "Planar":
        area_in = 2*radius_in*width
    elif nozzle_type == "Axisymmetric":
        area_in = np.pi*radius_in**2

    # Calculate inlet density
    density_in = properties_in.rho

    # Calculate velocity based on specified parameter
    if mass_flow is not None:
        velocity_in = mass_flow / (area_in * density_in)
    elif mach_in is not None:
        velocity_in = mach_in * properties_in["a"]
    elif critical_flow is True:
        # mach_impossible = 0.1
        # mach_possible = 0.00001
        # u_impossible = mach_impossible*properties_in.a
        # u_possible = mach_possible*properties_in.a
        # m_impossible = density_in*u_impossible*area_in
        # m_possible = density_in*u_possible*area_in
        m_impossible = 0.1
        m_possible = 0.01
        m_guess = (m_impossible+m_possible) / 2
        u_guess = m_guess / (density_in * area_in)

    # Initialize out_list to store all the state outputs
    out_list = []

    # Define the ODE system
    def odefun_HEM(t, y):
        z = t  # distance
        v, rho, p = y  # velocity, density, pressure
        
        try:

            # Thermodynamic state from perfect gas properties
            state = fluid.get_state(DmassP_INPUTS, rho, p)
            quality = state.Q

            # # Calculate area and geometry properties for convergent nozzles only
            # area, area_slope, perimeter, diameter = get_geometry(
            #     length=x, total_length=length, area_in=area_in, area_ratio=area_ratio
            # )
            
            # Calculate area and geometry properties for convergent-divergent nozzles
            area, area_slope, perimeter, radius, _, _ = get_linear_convergent_divergent(
                z_coordinate=z, convergent_length=convergent_length, divergent_length=divergent_length, radius_in=radius_in, radius_throat=radius_throat,
                radius_out=radius_out, width=width, type=nozzle_type)
            diameter = radius*2

            # Wall friction calculations
            stress_wall, friction_factor, reynolds = get_wall_friction(
                velocity=v,
                density=rho,
                viscosity=state["mu"],
                roughness=roughness,
                diameter=diameter,
            )
            if not include_friction:
                stress_wall = 0.0
                friction_factor = 0.0

            # Heat transfer (if applicable)
            if include_heat_transfer:
                U = 10 # To include correlations 
            else:
                U = 0.0
                heat_in = 0

            # Coefficient matrix M for ODE system
            M = np.asarray(
                [
                    [rho, v, 0.0],
                    [rho * v, 0.0, 1.0],
                    [0.0, -state["a"]**2, 1.0],
                ]
            )

            determinant = det(M)

            # If singularity detected
            flag = 0
            if abs(determinant) < 2:
                flag = 1

            # Right-hand side of the system
            # G = state.isobaric_expansion_coefficient * state.a**2 / state.cp
            G = 0
            b = np.asarray(
                [
                    -rho * v / area * area_slope,
                    -perimeter / area * stress_wall,
                    perimeter / area * G / v * (stress_wall * v + heat_in),
                ]
            )

            # Solve the system to get the change in state
            dy = scipy.linalg.solve(M, b)

            # Save the output at each step in the dictionary
            out = {
                "distance": z,
                "velocity": v,
                "density": rho,
                "pressure": p,
                "temperature": state["T"],
                "speed_of_sound": state["a"],
                "viscosity": state["mu"],
                "enthalpy": state["h"],
                "entropy": state["s"],
                "total_enthalpy": state["h"] + 0.5 * v**2,
                "mach_number": v / state["a"],
                "mass_flow": v * rho * area,
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
                "determinant": determinant,
                "quality": quality,
            }

            return dy, out, flag
        
        except Exception as e:
            # print(f"[ODEFUN ERROR @ x={x:.4f}] {e}")
            return [np.nan, np.nan, np.nan, np.nan]  # forces integrator to stop
        
    
    # Define the ODE system
    def odefun_DEM(t, Y_ode):
        z = t  # distance
        x, y, p, v = Y_ode 

        quality = x

        try:
            
            # # Calculate area and geometry properties for convergent nozzles only
            # area, area_slope, perimeter, diameter = get_geometry(
            #     length=x, total_length=length, area_in=area_in, area_ratio=area_ratio
            # )

            # Calculate area and geometry properties for convergent-divergent nozzles
            area, area_slope, perimeter, radius, dAdz_div, dAdz_conv = get_linear_convergent_divergent(
                z_coordinate=z, convergent_length=convergent_length, divergent_length=divergent_length, radius_in=radius_in, radius_throat=radius_throat,
                radius_out=radius_out, width=width, type=nozzle_type)
            diameter = radius*2

            # # Wall friction calculations
            # stress_wall, friction_factor, reynolds = get_wall_friction(
            #     velocity=v,
            #     density=rho,
            #     viscosity=state["mu"],
            #     roughness=roughness,
            #     diameter=diameter,
            # )

            if not include_friction:
                stress_wall = 0.0
                friction_factor = 0.0

            # Heat transfer (if applicable)
            if include_heat_transfer:
                U = 10 # To include correlations 
            else:
                U = 0.0
                heat_in = 0

            # Saturated liquid properties
            state_L = fluid.get_state(bpy.PQ_INPUTS, p, 0.00)
            rho_L = state_L.d
            h_L = state_L.h
            temp_L = CP.AbstractState("HEOS", fluid_name)
            temp_L.update(CP.PQ_INPUTS, p, 0.00)
            drhodp_L = temp_L.first_saturation_deriv(CP.iDmass, CP.iP)
            dvdp_L = -1.0 / (rho_L**2) * drhodp_L
            dhdp_L = temp_L.first_saturation_deriv(CP.iHmass, CP.iP)

            # Saturated vapor properties(bpy.fluid_properties.PT_INPUTS, p_stagnation, T_stagnation)
            state_V = fluid.get_state(bpy.PQ_INPUTS, p, 1.00)
            rho_V = state_V.d
            h_V = state_V.h
            T = state_V.T
            temp_V = CP.AbstractState("HEOS", fluid_name)
            temp_V.update(CP.PQ_INPUTS, p, 1.00)
            drhodp_V = temp_V.first_saturation_deriv(CP.iDmass, CP.iP)
            dvdp_V = -1.0 / (rho_V**2) * drhodp_V
            dhdp_V = temp_V.first_saturation_deriv(CP.iHmass, CP.iP)

            # Metastable properties
            meta_state = fluid.get_state_metastable(
                prop_1="s",
                prop_1_value=s_meta,
                prop_2="p",
                prop_2_value=p,
                rhoT_guess=rhoT_guess,
                print_convergence=False
            )

            rho_meta = meta_state["rho"]
            T_meta = meta_state["T"]
            h_meta = meta_state["h"]
            # pressure_meta = meta_state["p"]

            state_meta = fluid.get_state(bpy.QT_INPUTS, 0.00, T_meta)
            p_sat_T_LM = state_meta.p

            spec_vol_mix = x/rho_V + (y-x)/rho_L + (1-y)/rho_meta
            h_mix = x*h_V + (y-x)*h_L + (1-y)*h_meta

            rho_m = 1/spec_vol_mix

            state = fluid.get_state(bpy.DmassP_INPUTS, rho_m, p)
            
            # Coefficient matrix M for ODE system for DEM 
            M = np.asarray(
                [
                    [(1/rho_V-1/rho_L), (1/rho_L-1/rho_meta), (x*dvdp_V+(y-x)*dvdp_L), -(spec_vol_mix/v)],
                    [0.0, 0.0, 1.0, (v/spec_vol_mix)],
                    [(h_V-h_L), (h_L-h_meta), (x*dhdp_V+(y-x)*dhdp_L+(1-y)*(1/rho_meta)), v],
                    [0.0, 1.0, 0.0, 0.0],
                ]
            )

            determinant = det(M)
            # print(determinant)

            # If singularity detected
            flag = 0
            if abs(determinant) < 0.5:
                flag = 1
            
            # Right-hand side of the system for DEM
            b = np.asarray(
                [
                    spec_vol_mix / area * area_slope,
                    -perimeter / area * stress_wall,
                    0.0, # To modify in case of heat transfer to be included
                    dem_term_angielczyk_2(p=p, y=y, p_sat_T_LM=p_sat_T_LM, p_cr=p_cr, dAdz_div=dAdz_div, dAdz_conv=dAdz_conv), # dem_term(perimeter=perimeter, p=p, y=y, p_sat_T_LM=p_sat_T_LM, p_cr=p_cr, area=area, dAdz_div=dAdz_div, dAdz_conv=dAdz_conv),
                ]
            )
            
            # Solve the system to get the change in state
            dy = scipy.linalg.solve(M, b)
    
            # Save the output at each step in the dictionary
            out = {
                "distance": z,
                "velocity": v,
                "density": rho_m,
                "pressure": p,
                # "temperature": state["T"],
                "speed_of_sound": state["a"],
                # "viscosity": state["mu"],
                # "enthalpy": state["h"],
                "entropy": s_meta,
                # "total_enthalpy": state["h"] + 0.5 * v**2,
                "mach_number": v / state["a"],
                "mass_flow": v * rho_m * area,
                "area": area,
                "area_slope": area_slope,
                "perimeter": perimeter,
                "diameter": diameter,
                "stress_wall": stress_wall,
                "friction_factor": friction_factor,
                # "reynolds": reynolds,
                "source_1": b[0],
                "source_2": b[1],
                "determinant": determinant,
                "quality": quality,
                "stable_fraction": y,
            }

            # Append the output dictionary to out_list
            out_list.append(out)

            return dy, out, flag
    
        except Exception as e:
            print(f"[ODEFUN ERROR @ x={x:.4f}] {e}")
            exit()
            return [np.nan, np.nan, np.nan, np.nan] 
        
    def stop_at_two_phase(t, y):
        density = y[1]
        pressure = y[2]
        
        # Update state
        state = fluid.get_state(DmassP_INPUTS, density, pressure)
        
        try:
            quality = state.Q  # Vapor quality
        except ValueError:
            return 1.0
        
        if 0.0 <= quality <= 1.0:
            return 0.0  # Stop integration
        else:
            return 1.0  # Keep integrating
    stop_at_two_phase.terminal = True  
    stop_at_two_phase.direction = 0    

    # Possible-impossible flow (PIF) algorithm to find critical flow rate
    if critical_flow is True: 
        pif_iterations = 0
        print("Possible-Impossible Flow (PIF) algorithm starts...")
        error = abs(m_impossible-m_possible)/m_possible
        tol = 1e-2
        while error > tol:
            pif_iterations += 1  

            raw_solution_1 = scipy.integrate.solve_ivp(
                lambda t, y: odefun_HEM(t, y)[0],
                [0.0, total_length],
                [u_guess, density_in, pressure_in],
                t_eval=np.linspace(0, convergent_length, number_of_points) if number_of_points else None,
                method="RK45",
                rtol=1e-9,
                atol=1e-9,
                events = stop_at_two_phase
            )
            solution_1 = postprocess_ode(raw_solution_1.t, raw_solution_1.y, odefun_HEM)
            solution = solution_1

            solution_2 = None
            if raw_solution_1.t_events[0].size > 0:
                
                raw_solution_2 = scipy.integrate.solve_ivp(
                    lambda t, y: odefun_DEM(t, y)[0],
                    [solution_1["distance"][-1], total_length],
                    [solution_1["quality"][-1], solution_1["quality"][-1], solution_1["pressure"][-1], solution_1["velocity"][-1]],
                    t_eval=np.linspace(0, convergent_length, number_of_points) if number_of_points else None,
                    method="RK45",
                    rtol=1e-9,
                    atol=1e-9,
                )
                solution_2 = postprocess_ode(raw_solution_2.t, raw_solution_2.y, odefun_DEM)

                all_keys = set(solution_1.keys()) | set(solution_2.keys())
                for key in all_keys:
                    data_1 = solution_1.get(key, [])

                    data_2 = solution_2.get(key, [])
                    if isinstance(data_2, (list, np.ndarray)):
                        data_2 = data_2[1:]  # Skip first point
                    else:
                        data_2 = []

                    solution[key] = np.concatenate([
                        np.atleast_1d(data_1),
                        np.atleast_1d(data_2)
                    ])

            if solution_2["distance"][-1] < convergent_length:
                m_impossible = m_guess
            else:
                m_possible = m_guess

            print(m_impossible)
            print(m_possible)

            m_guess = (m_impossible+m_possible) / 2
            u_guess = m_guess / (density_in * area_in)  
            error = abs(m_impossible-m_possible)/m_possible

        flow_rate = u_guess*area_in*density_in

        # Calculate the solution with u_guess
        print("STARTING CRITICAL")
        u_avg = (m_possible+m_impossible)/(2*density_in * area_in)
        raw_solution_1 = scipy.integrate.solve_ivp(
            lambda t, y: odefun_HEM(t, y)[0],
            [0.0, (total_length)],
            [u_avg, density_in, pressure_in],
            t_eval=np.linspace(0, convergent_length, number_of_points) if number_of_points else None,
            method="RK45",
            rtol=1e-9,
            atol=1e-9,
            events = stop_at_two_phase
        ) 
        solution_1 = postprocess_ode(raw_solution_1.t, raw_solution_1.y, odefun_HEM)
        solution = solution_1

        solution_2 = None
        if raw_solution_1.t_events[0].size > 0:
            raw_solution_2 = scipy.integrate.solve_ivp(
                lambda t, y: odefun_DEM(t, y)[0],
                [solution_1["distance"][-1], total_length],
                [solution_1["quality"][-1], solution_1["quality"][-1], solution_1["pressure"][-1], solution_1["velocity"][-1]],
                t_eval=np.linspace(0, convergent_length, number_of_points) if number_of_points else None,
                method="RK45",
                rtol=1e-9,
                atol=1e-9,
            )
            solution_2 = postprocess_ode(raw_solution_2.t, raw_solution_2.y, odefun_DEM)

            all_keys = set(solution_1.keys()) | set(solution_2.keys())
            for key in all_keys:
                data_1 = solution_1.get(key, [])
                data_2 = solution_2.get(key, [])

                solution[key] = np.concatenate([
                    np.atleast_1d(data_1),
                    np.atleast_1d(data_2)
                ])


        # mass_flow = solution_HEM["mass_flow"][0]
        # density = solution_HEM["density"][-1]
        z = convergent_length*1.00001
        # area, area_slope, perimeter, radius = get_linear_convergent_divergent(z, 
        #                                                                       convergent_length=convergent_length,
        #                                                                       divergent_length=divergent_length,
        #                                                                       radius_in=radius_in,
        #                                                                       radius_out=radius_out,
        #                                                                       radius_throat=radius_throat,
        #                                                                       width=width,
        #                                                                       type=nozzle_type)
        # velocity = mass_flow/(density*area)

        print("STARTING SUPERSONIC")
        supersonic_solution = scipy.integrate.solve_ivp(
            lambda t, y: odefun_DEM(t, y)[0],
            [solution["distance"][-1]*1.1, convergent_length+divergent_length],
            [solution["quality"][-1],
             solution["stable_fraction"][-1], 
             solution["pressure"][-1], 
             solution["velocity"][-1]],
            t_eval=np.linspace(0, convergent_length, number_of_points) if number_of_points else None,
            method="RK45",
            rtol=1e-9,
            atol=1e-9,
        ) 
        supersonic_solution = postprocess_ode(supersonic_solution.t, supersonic_solution.y, odefun_DEM) 
 
        # To fix:
        supersonic_solution_HEM = 0
        supersonic_solution_DEM = 0

    else:
        pif_iterations = None
        solution = scipy.integrate.solve_ivp(
        lambda t, y: odefun_HEM(t, y)[0],
        [0.0, total_length],
        [velocity_in, density_in, pressure_in],
        t_eval=np.linspace(0, total_length, number_of_points) if number_of_points else None,
        method="Radau",
        rtol=1e-9,
        atol=1e-9,
        )
        solution = postprocess_ode(solution.t, solution.y, odefun_HEM)

        flow_rate = velocity_in*density_in*area_in

    return solution, supersonic_solution, pif_iterations


# Autonomous solver 
def pipeline_steady_state_1D_autonomous(
    fluid_name,
    pressure_in,
    temperature_in,
    properties_in,
    convergent_length,
    divergent_length,
    roughness,
    radius_in,
    radius_throat,
    radius_out,
    nozzle_type,
    width,
    mass_flow=None,
    mach_in=None,
    critical_flow = False,
    include_friction=True,
    include_heat_transfer=False,
    temperature_external=None,
    number_of_points=None,
):
    # Check for correct inputs
    if (
        (mass_flow is None and mach_in is None and critical_flow is False) or 
        (mass_flow is not None and mach_in is not None) 
    ):
        raise ValueError(
            "Check input settins for the velocity."
        )
    
    fluid = bpy.Fluid(fluid_name, backend="HEOS")
    state_in = fluid.get_state(bpy.PT_INPUTS, pressure_in, temperature_in)
    p_cr = fluid.critical_point.p
    s_meta = state_in.s  # The metastable phase is assumed to follow an isentropi expansion
    rhoT_guess = [state_in["rho"], state_in["T"]]

        
    # Define inlet area and length of the nozzle
    total_length = convergent_length+divergent_length
    if nozzle_type == "Planar":
        area_in = 2*radius_in*width
    elif nozzle_type == "Axisymmetric":
        area_in = np.pi*radius_in**2

    # Calculate inlet density
    density_in = properties_in.rho

    # Calculate velocity based on specified parameter
    if mass_flow is not None:
        velocity_in = mass_flow / (area_in * density_in)
    elif mach_in is not None:
        velocity_in = mach_in * properties_in["a"]
    elif critical_flow is True:
        mach_impossible = 0.01
        mach_possible = 0.000001
        u_impossible = mach_impossible*properties_in.a
        u_possible = mach_possible*properties_in.a
        m_impossible = density_in*u_impossible*area_in
        m_possible = density_in*u_possible*area_in
        # m_impossible = 0.031
        # m_possible = 0.031

        m_guess = (m_impossible+m_possible) / 2
        u_guess = m_guess / (density_in * area_in)

    # Initialize out_list to store all the state outputs
    out_list = []

    # Define the ODE system
    def odefun_DEM(Y):

        z, x, y, p, v = Y
 
        try:
            # Thermodynamic state from perfect gas properties

        
            # Calculate area and geometry properties for convergent-divergent nozzles
            area, area_slope, perimeter, radius, dAdz_div, dAdz_conv = get_linear_convergent_divergent(
                z_coordinate=z, convergent_length=convergent_length, divergent_length=divergent_length, radius_in=radius_in, radius_throat=radius_throat,
                radius_out=radius_out, width=width, type=nozzle_type)
            diameter = radius*2

            # Heat transfer (if applicable)
            if include_heat_transfer:
                U = 10 # To include correlations 
            else:
                U = 0.0
                heat_in = 0

            # Saturated liquid properties
            state_L = fluid.get_state(bpy.fluid_properties.PQ_INPUTS, p, 0.00)
            rho_L = state_L.d
            h_L = state_L.h
            temp_L = CP.AbstractState("HEOS", fluid_name)
            temp_L.update(CP.PQ_INPUTS, p, 0.00)
            drhodp_L = temp_L.first_saturation_deriv(CP.iDmass, CP.iP)
            dvdp_L = -1.0 / (rho_L**2) * drhodp_L
            dhdp_L = temp_L.first_saturation_deriv(CP.iHmass, CP.iP)
            
            # Saturated vapor properties(bpy.fluid_properties.PT_INPUTS, p_stagnation, T_stagnation)
            state_V = fluid.get_state(bpy.fluid_properties.PQ_INPUTS, p, 1.00)
            rho_V = state_V.d
            h_V = state_V.h
            T = state_L.T
            temp_V = CP.AbstractState("HEOS", fluid_name)
            temp_V.update(CP.PQ_INPUTS, p, 1.00)
            drhodp_V = temp_V.first_saturation_deriv(CP.iDmass, CP.iP)
            dvdp_V = -1.0 / (rho_V**2) * drhodp_V
            dhdp_V = temp_V.first_saturation_deriv(CP.iHmass, CP.iP)

            # Metastable properties
            meta_state = fluid.get_state_metastable(
                prop_1="s",
                prop_1_value=s_meta,
                prop_2="p",
                prop_2_value=p,
                rhoT_guess=rhoT_guess,
                print_convergence=False
            )

            rho_meta = meta_state["rho"]
            T_meta = meta_state["T"]
            h_meta = meta_state["h"]
            # pressure_meta = meta_state["p"]

            state_meta = fluid.get_state(bpy.fluid_properties.QT_INPUTS, 0.00, T_meta)
            p_sat_T_LM = state_meta.p

            spec_vol_mix = x/rho_V + (y-x)/rho_L + (1-y)/rho_meta
            h_mix = x*h_V + (y-x)*h_L + (1-y)*h_meta

            rho_m = 1/spec_vol_mix

            state = fluid.get_state(bpy.fluid_properties.DmassP_INPUTS, rho_m, p)
            
            # Coefficient matrix M for ODE system for DEM 
            M = np.asarray(
                [
                    [(1/rho_V-1/rho_L), (1/rho_L-1/rho_meta), (x*dvdp_V+(y-x)*dvdp_L), -(spec_vol_mix/v)],
                    [0.0, 0.0, 1.0, (v/spec_vol_mix)],
                    [(h_V-h_L), (h_L-h_meta), (x*dhdp_V+(y-x)*dhdp_L+(1-y)*(1/rho_meta)), v],
                    [0.0, 1.0, 0.0, 0.0],
                ]
            )

            # # Wall friction calculations
            stress_wall, friction_factor, reynolds = get_wall_friction(
                velocity=v,
                density=rho_m,
                viscosity=state["mu"],
                roughness=roughness,
                diameter=diameter,
            )

            if not include_friction:
                stress_wall = 0.0
                friction_factor = 0.0
            
            # Right-hand side of the system for DEM
            b = np.asarray(
                [
                    spec_vol_mix / area * area_slope,
                    -perimeter / area * stress_wall,
                    0.0, # To modify in case of heat transfer to be included
                    # dem_term_angielczyk_2(p=p, y=y, p_sat_T_LM=p_sat_T_LM, p_cr=p_cr, dAdz_div=dAdz_div, dAdz_conv=dAdz_conv), 
                    dem_term_angielczyk_1(perimeter=perimeter, p=p, y=y, p_sat_T_LM=p_sat_T_LM, p_cr=p_cr, area=area),
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
            
            # Compute determinants
            det_M = det(M)
            det_M1 = det(M1)
            det_M2 = det(M2)
            det_M3 = det(M3)
            det_M4 = det(M4)
            # print("dem det 1", det_M)
            # print("dem det 2", det_M1)
            # print("dem det 3", det_M2)
            # print("dem det 4", det_M3)
            # print("dem det 5", det_M4)
            # print(" ")
            
            dy = [det_M, det_M1, det_M2, det_M3, det_M4]

            quality = x

            # Save the output at each step in the dictionary
            out = {
                "distance": z,
                "velocity": v,
                "density": rho_m,
                "pressure": p,
                # "temperature": state["T"],
                "speed_of_sound": state["a"],
                # "viscosity": state["mu"],
                # "enthalpy": state["h"],
                "entropy": s_meta,
                # "total_enthalpy": state["h"] + 0.5 * v**2,
                "mach_number": v / state["a"],
                "mass_flow": v * rho_m * area,
                "area": area,
                "area_slope": area_slope,
                "perimeter": perimeter,
                "diameter": diameter,
                "stress_wall": stress_wall,
                "friction_factor": friction_factor,
                # "reynolds": reynolds,
                "source_1": b[0],
                "source_2": b[1],
                "determinant": det_M,
                "quality": quality,
                "stable_fraction": y,
            }

            # Append the output dictionary to out_list
            out_list.append(out)

            return dy, out
        
        except Exception as e:
            # Log the error for debugging if helpful
            # print(f"odefun error at y={y}: {e}")
            
            dy = [np.nan, np.nan, np.nan, np.nan, np.nan]
            out = {"determinant": np.nan}  # or 0.0, depending on what you want
            return dy, out  
        
    # Define the ODE system
    def odefun_HEM(y):

        z, v, rho, p = y

        try:

            # Thermodynamic state from perfect gas properties
            state = fluid.get_state(DmassP_INPUTS, rho, p)

            # Calculate area and geometry properties for convergent-divergent nozzles
            area, area_slope, perimeter, radius, _, _ = get_linear_convergent_divergent(
                z_coordinate=z, convergent_length=convergent_length, divergent_length=divergent_length, radius_in=radius_in, radius_throat=radius_throat,
                radius_out=radius_out, width=width, type=nozzle_type)
            diameter = radius*2

            # Wall friction calculations
            stress_wall, friction_factor, reynolds = get_wall_friction(
                velocity=v,
                density=rho,
                viscosity=state["mu"],
                roughness=roughness,
                diameter=diameter,
            )

            if not include_friction:
                stress_wall = 0.0
                friction_factor = 0.0

            # Heat transfer (if applicable)
            if include_heat_transfer:
                U = 10 # To include correlations 
            else:
                U = 0.0
                heat_in = 0

            # Coefficient matrix M for single phase and HEM
            M = np.asarray(
                [
                    [rho, v, 0.0],
                    [rho * v, 0.0, 1.0],
                    [0.0, -state["a"]**2, 1.0],
                ]
            )

            # Right-hand side of the system
            # G = state.isobaric_expansion_coefficient * state.a**2 / state.cp
            G = 0
            b = np.asarray(
                [
                    -rho * v / area * area_slope,
                    -perimeter / area * stress_wall,
                    perimeter / area * G / v * (stress_wall * v + heat_in),
                ]
            )

            M1 = M.copy()
            M1[:, 0] = b
            M2 = M.copy()
            M2[:, 1] = b
            M3 = M.copy()
            M3[:, 2] = b

            # Compute determinants
            det_M = det(M)
            det_M1 = det(M1)
            det_M2 = det(M2)
            det_M3 = det(M3)
            # print("hem det", det_M)
        
            dy = [det_M, det_M1, det_M2, det_M3]

            two_phase_flag = False
            quality = state.Q
            if quality > 0:
                two_phase_flag = True

            # Save the output at each step in the dictionary
            out = {
                "distance": z,
                "velocity": v,
                "density": rho,
                "pressure": p,
                "temperature": state["T"],
                "speed_of_sound": state["a"],
                "viscosity": state["mu"],
                "enthalpy": state["h"],
                "entropy": state["s"],
                "total_enthalpy": state["h"] + 0.5 * v**2,
                "mach_number": v / state["a"],
                "mass_flow": v * rho * area,
                "area": area,
                "area_slope": area_slope,
                "perimeter": perimeter,
                "diameter": diameter,
                "stress_wall": stress_wall,
                "friction_factor": friction_factor,
                "reynolds": reynolds,
                "source_1": b[0],
                "source_2": b[1],
                "determinant": det_M,
                "two_phase_flag": two_phase_flag,
                "quality": quality,
                "stable_fraction": 0,
            }

            # Append the output dictionary to out_list
            out_list.append(out)

            return dy, out
        
        except Exception as e:

            # print(f"odefun error at y={y}: {e}")
            dy = [np.nan, np.nan, np.nan, np.nan]
            out = {"determinant": np.nan}  # or 0.0, depending on what you want
            return dy, out  # always return a tuple

    def stop_at_zero_det_HEM(t, y):
        _, out = odefun_HEM(y)
        det_M = out["determinant"]
        return det_M
    stop_at_zero_det_HEM.terminal = True
    stop_at_zero_det_HEM.direction = 0

    def stop_at_zero_det_DEM(t, y):
        _, out = odefun_DEM(y)
        det_M = out["determinant"]
        return det_M
    stop_at_zero_det_DEM.terminal = True
    stop_at_zero_det_DEM.direction = 0

    def stop_at_length(t, y):
        x = y[0]               
        return x - total_length      
    stop_at_length.terminal = True     
    stop_at_length.direction = 1 

    def stop_at_inlet(t, y):
        x = y[0]
        return x  # Will be zero or negative when at or past the inlet
    stop_at_inlet.terminal = True  # Stop the solver when event is triggered
    stop_at_inlet.direction = -1 

    def stop_at_two_phase(t, y):
        _, out = odefun_HEM(y)
        state = fluid.get_state(DmassP_INPUTS, out["density"], out["pressure"])
        Q = state.Q

        # If Q is undefined (e.g., in compressed liquid or superheated vapor), skip event
        if Q is None:
            return 1.0

        # Trigger when entering two-phase: 0 < Q < 1
        if 0.0 < Q < 1.0:
            return 0.0  # Event triggers
        else:
            return 1.0  # No event

    stop_at_two_phase.terminal = True
    stop_at_two_phase.direction = -1  # Only trigger when entering from single-phase

    # Possible-impossible flow (PIF) algorithm to find critical flow rate
    if critical_flow is True: 
        pif_iterations = 0
        print("Possible-Impossible Flow (PIF) algorithm starts...")
        error = abs(m_impossible-m_possible)/m_possible
        tol = 1e-2

        while error > tol:
            pif_iterations += 1  
            
            raw_solution_1 = scipy.integrate.solve_ivp(
                lambda t, y: odefun_HEM(y)[0],
                [0, 1],
                [0, u_guess, density_in, pressure_in],
                t_eval=np.linspace(0, convergent_length, number_of_points) if number_of_points else None,
                method="RK45",
                rtol=1e-9,
                atol=1e-9,
                events=[stop_at_two_phase]
            )
            solution_1 = postprocess_ode_autonomous(raw_solution_1.t, raw_solution_1.y, odefun_HEM)
            solution = solution_1
            raw_solution_2 = None  # Ensure it's defined for later logic

            if raw_solution_1.t_events[0].size > 0:

                raw_solution_2 = scipy.integrate.solve_ivp(
                    lambda t, y: odefun_DEM(y)[0],
                    [0, 1],
                    [solution_1["distance"][-1], solution_1["quality"][-1], solution_1["quality"][-1], solution_1["pressure"][-1], solution_1["velocity"][-1]],
                    t_eval=np.linspace(0, convergent_length, number_of_points) if number_of_points else None,
                    method="RK45",
                    rtol=1e-9,
                    atol=1e-9,
                    events=[stop_at_length, stop_at_zero_det_DEM]
                )
                solution_2 = postprocess_ode_autonomous(raw_solution_2.t, raw_solution_2.y, odefun_DEM)
                
                all_keys = set(solution_1.keys()) | set(solution_2.keys())
                for key in all_keys:
                    data_1 = solution_1.get(key, [])
                    data_2 = solution_2.get(key, [])[1:]  # skip the first point to avoid duplication

                    solution[key] = np.concatenate([
                        np.atleast_1d(data_1),
                        np.atleast_1d(data_2)
                    ])

                if raw_solution_2 is not None and raw_solution_2.t_events[1].size > 0:
                    m_impossible = m_guess
                else:
                    m_possible = m_guess

            m_guess = (m_impossible + m_possible) / 2
            u_guess = m_guess / (density_in * area_in)
            error = abs(m_impossible - m_possible) / m_possible

            print(m_impossible)
            print(m_possible)

        flow_rate = u_guess*area_in*density_in

        print("STARTING POSSIBLE")

        # # Calculate the solution with the last possible flow rate calculated
        # u_possible = m_possible/(density_in * area_in)
        # raw_possible_solution_1 = scipy.integrate.solve_ivp(
        #     lambda t, y: odefun_HEM(y)[0],
        #     [0, 1],
        #     [0, u_possible, density_in, pressure_in],
        #     t_eval=np.linspace(0, convergent_length, number_of_points) if number_of_points else None,
        #     method="Radau",
        #     rtol=1e-9,
        #     atol=1e-9,
        #     events=[stop_at_length, stop_at_zero_det_HEM, stop_at_two_phase]
        # )

        # possible_solution_1 = postprocess_ode_autonomous(raw_possible_solution_1.t, raw_possible_solution_1.y, odefun_HEM)
        # possible_solution = possible_solution_1
        # raw_possible_solution_2 = None  # Ensure it's defined for later logic

        # if raw_possible_solution_1.t_events[2].size > 0:
        #     raw_possible_solution_2 = scipy.integrate.solve_ivp(
        #         lambda t, y: odefun_DEM(y)[0],
        #         [0, 1],
        #         [possible_solution_1["distance"][-1], 1e-1, 1e-1, possible_solution_1["pressure"][-1], possible_solution_1["velocity"][-1]],
        #         t_eval=np.linspace(0, convergent_length, number_of_points) if number_of_points else None,
        #         method="Radau",
        #         rtol=1e-9,
        #         atol=1e-9,
        #         events=[stop_at_length, stop_at_zero_det_DEM]
        #     )
        #     possible_solution_2 = postprocess_ode_autonomous(raw_possible_solution_2.t, raw_possible_solution_2.y, odefun_DEM)

        #     all_keys = set(possible_solution_1.keys()) | set(possible_solution_2.keys())
        #     for key in all_keys:
        #         data_1 = possible_solution_1.get(key, [])
        #         data_2 = possible_solution_2.get(key, [])[1:]  # skip the first point to avoid duplication

        #         possible_solution[key] = np.concatenate([
        #             np.atleast_1d(data_1),
        #             np.atleast_1d(data_2)
        #         ])

        print("STARTING IMPOSSIBLE")

        # # Calculate the solution with the last impossible flow rate calculated
        # u_impossible = m_impossible/(density_in * area_in)
        # raw_impossible_solution_1 = scipy.integrate.solve_ivp(
        #     lambda t, y: odefun_HEM(y)[0],
        #     [0, 1],
        #     [0, u_impossible, density_in, pressure_in],
        #     t_eval=np.linspace(0, convergent_length, number_of_points) if number_of_points else None,
        #     method="Radau",
        #     rtol=1e-9,
        #     atol=1e-9,
        #     events=[stop_at_length, stop_at_zero_det_HEM, stop_at_two_phase]
        # )

        # impossible_solution_1 = postprocess_ode_autonomous(raw_impossible_solution_1.t, raw_impossible_solution_1.y, odefun_HEM)
        # impossible_solution = impossible_solution_1
        # raw__impossible_solution_2 = None  # Ensure it's defined for later logic

        # if raw_impossible_solution_1.t_events[2].size > 0:
        #     raw__impossible_solution_2 = scipy.integrate.solve_ivp(
        #         lambda t, y: odefun_DEM(y)[0],
        #         [0, 1],
        #         [impossible_solution_1["distance"][-1], 1e-1, 1e-1, impossible_solution_1["pressure"][-1], impossible_solution_1["velocity"][-1]],
        #         t_eval=np.linspace(0, convergent_length, number_of_points) if number_of_points else None,
        #         method="Radau",
        #         rtol=1e-9,
        #         atol=1e-9,
        #         events=[stop_at_length, stop_at_zero_det_DEM]
        #     )
        #     impossible_solution_2 = postprocess_ode_autonomous(raw__impossible_solution_2.t, raw__impossible_solution_2.y, odefun_DEM)

        #     all_keys = set(impossible_solution_1.keys()) | set(impossible_solution_2.keys())
        #     for key in all_keys:
        #         data_1 = impossible_solution_1.get(key, [])
        #         data_2 = impossible_solution_2.get(key, [])[1:]  # skip the first point to avoid duplication

        #         impossible_solution[key] = np.concatenate([
        #             np.atleast_1d(data_1),
        #             np.atleast_1d(data_2)
        #         ])

        print("STARTING CRITICAL")

        # Calculate the solution with u_guess
        u_avg = (m_possible+m_impossible)/(2*density_in * area_in)
        raw_solution_1 = scipy.integrate.solve_ivp(
            lambda t, y: odefun_HEM(y)[0],
            [0, 1],
            [0, u_avg, density_in, pressure_in],
            t_eval=np.linspace(0, convergent_length, number_of_points) if number_of_points else None,
            method="RK45",
            rtol=1e-9,
            atol=1e-9,
            events=[stop_at_length, stop_at_two_phase, stop_at_inlet]
        )

        solution_1 = postprocess_ode_autonomous(raw_solution_1.t, raw_solution_1.y, odefun_HEM)
        solution = solution_1

        if raw_solution_1.t_events[1].size > 0:
            raw_solution_2 = scipy.integrate.solve_ivp(
                lambda t, y: odefun_DEM(y)[0],
                [0, 1],
                [solution_1["distance"][-1], solution_1["quality"][-1], solution_1["quality"][-1], solution_1["pressure"][-1], solution_1["velocity"][-1]],
                t_eval=np.linspace(0, convergent_length, number_of_points) if number_of_points else None,
                method="RK45",
                rtol=1e-9,
                atol=1e-9,
                events=[stop_at_length, stop_at_inlet, stop_at_zero_det_DEM]
            )
            solution_2 = postprocess_ode_autonomous(raw_solution_2.t, raw_solution_2.y, odefun_DEM)

            all_keys = set(solution_1.keys()) | set(solution_2.keys())
            for key in all_keys:
                data_1 = solution_1.get(key, [])
                data_2 = solution_2.get(key, [])[1:]  # skip the first point to avoid duplication

                solution[key] = np.concatenate([
                    np.atleast_1d(data_1),
                    np.atleast_1d(data_2)
                ])

        # mass_flow = impossible_solution["mass_flow"][0]
        # density = impossible_solution["density"][-1]
        # z = convergent_length*1.001
        # area, area_slope, perimeter, radius, _, _ = get_linear_convergent_divergent(z, 
        #                                                                       convergent_length=convergent_length,
        #                                                                       divergent_length=divergent_length,
        #                                                                       radius_in=radius_in,
        #                                                                       radius_out=radius_out,
        #                                                                       radius_throat=radius_throat,
        #                                                                       width=width,
        #                                                                       type=nozzle_type)
        # velocity = mass_flow/(density*area)

        print("STARTING SUPERSONIC")

        arr = solution["determinant"]
        first_negative_index = int(np.flatnonzero(arr < 0)[0]) if np.any(arr < 0) else -1

        supersonic_solution_1 = scipy.integrate.solve_ivp(
            lambda t, y: odefun_DEM(y)[0],
            [1, 0],
            [solution["distance"][first_negative_index], solution["quality"][first_negative_index], solution["stable_fraction"][first_negative_index], solution["pressure"][first_negative_index], solution["velocity"][first_negative_index]*1.1],
            t_eval=np.linspace(0, convergent_length, number_of_points) if number_of_points else None,
            method="RK45",
            rtol=1e-9,
            atol=1e-9,
            events=[stop_at_length, stop_at_inlet]
        )
        supersonic_solution_1 = postprocess_ode_autonomous(supersonic_solution_1.t, supersonic_solution_1.y, odefun_DEM)
        supersonic_solution = supersonic_solution_1
 
        supersonic_solution_2 = scipy.integrate.solve_ivp(
            lambda t, y: odefun_HEM(y)[0],
            [1, 0],
            [supersonic_solution_1["distance"][-1], supersonic_solution_1["velocity"][-1], supersonic_solution_1["density"][-1], supersonic_solution_1["pressure"][-1]],
            t_eval=np.linspace(0, convergent_length, number_of_points) if number_of_points else None,
            method="RK45",
            rtol=1e-9,
            atol=1e-9,
            events=[stop_at_length, stop_at_inlet]
        )
        supersonic_solution_2 = postprocess_ode_autonomous(supersonic_solution_2.t, supersonic_solution_2.y, odefun_HEM)

        all_keys = set(supersonic_solution_1.keys()) | set(supersonic_solution_2.keys())
        for key in all_keys:
            data_1 = supersonic_solution_1.get(key, [])
            data_2 = supersonic_solution_2.get(key, [])[1:]  # skip the first point to avoid duplication

            supersonic_solution[key] = np.concatenate([
                np.atleast_1d(data_1),
                np.atleast_1d(data_2)
            ])

        all_keys = set(solution.keys()) | set(supersonic_solution.keys())
        for key in all_keys:
            data_1 = solution.get(key, [])
            data_2 = supersonic_solution.get(key, [])[1:]  # skip the first point to avoid duplication

            solution[key] = np.concatenate([
                np.atleast_1d(data_1),
                np.atleast_1d(data_2)
            ])

    # To fix:
        possible_solution = 0
        impossible_solution=0

    else:
        pif_iterations = None
        solution = scipy.integrate.solve_ivp(
        lambda t, y: odefun_HEM(t, y)[0],
        [0.0, total_length],
        [velocity_in, density_in, pressure_in],
        t_eval=np.linspace(0, total_length, number_of_points) if number_of_points else None,
        method="Radau",
        rtol=1e-9,
        atol=1e-9,
        )
        solution = postprocess_ode(solution.t, solution.y, odefun_HEM)

        flow_rate = velocity_in*density_in*area_in
 

    return possible_solution, impossible_solution, solution, flow_rate, pif_iterations

