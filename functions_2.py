import scipy.linalg
import scipy.integrate
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import perfect_gas_prop.perfect_gas_prop as perfect_gas_prop 
import real_gas_prop.real_gas_prop as rg
from cycler import cycler
import numpy as np
from scipy.linalg import det
import CoolProp.CoolProp as CP


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
        _, out, flag, flag_2 = ode_handle(t_i, y_i)

        if flag == 1:
            singularity_detected = True
        
        if flag_2 == 1:
            subsonic_detected = True

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
    ode_out["subsonic_detected"] = subsonic_detected

    return ode_out


def get_geometry(length, total_length, area_in, area_ratio):
    """
    Calculates the cross-sectional area, area slope, perimeter, and diameter
    of a pipe or nozzle at a given length along its axis.

    This function is useful for analyzing variable area pipes or nozzles, where the
    area changes linearly from the inlet to the outlet. The area slope is calculated
    based on the total change in area over the total length, assuming a linear variation.

    Parameters
    ----------
    length : float
        The position along the pipe or nozzle from the inlet (m).
    total_length : float
        The total length of the pipe or nozzle (m).
    area_in : float
        The cross-sectional area at the inlet of the pipe or nozzle (m^2).
    area_ratio : float
        The ratio of the area at the outlet to the area at the inlet.

    Returns
    -------
    area : float
        The cross-sectional area at the specified length (m^2).
    area_slope : float
        The rate of change of the area with respect to the pipe or nozzle's length (m^2/m).
    perimeter : float
        The perimeter of the cross-section at the specified length (m).
    diameter : float
        The diameter of the cross-section at the specified length (m).
    """
    area_slope = (area_ratio - 1.0) * area_in / total_length
    area = area_in + area_slope * length
    radius = np.sqrt(area / np.pi)
    diameter = 2 * radius
    perimeter = np.pi * diameter
    return area, area_slope, perimeter, diameter


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

## Andrea's code, only for perfect gases

def pipeline_steady_state_1D(
    fluid_name,
    pressure_in,
    temperature_in,
    diameter_in,
    properties_in,
    length,
    roughness,
    area_ratio=1.00,
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
    
    # Define geometry
    radius_in = 0.5 * diameter_in
    area_in = np.pi * radius_in**2

    # # Calculate inlet density
    # density_in = properties_in["d"]
    # speed_sound_in = properties_in["a"]
    # R = properties_in["R"]

    fluid = rg.Fluid(fluid_name, backend="HEOS", exceptions=True)

    # Calculate inlet density
    state_in = fluid.set_state(PT_INPUTS, pressure_in, temperature_in)
    density_in = state_in.rho

    # Calculate velocity based on specified parameter
    if mass_flow is not None:
        velocity_in = mass_flow / (area_in * density_in)
    elif mach_in is not None:
        velocity_in = mach_in * properties_in["a"]
    elif critical_flow is True:
        mach_impossible = 0.7
        mach_possible = 0.1
        u_impossible = mach_impossible*state_in.a
        u_possible = mach_possible*state_in.a
        m_impossible = density_in*u_impossible*area_in
        m_possible = density_in*u_possible*area_in

    # Initialize out_list to store all the state outputs
    out_list = []
    
    # Define the ODE system
    def odefun(t, y):
        x = t  # distance
        v, rho, p = y  # velocity, density, pressure
        
        # Thermodynamic state from perfect gas properties
        # T = p / (rho * R)
        # state = perfect_gas_prop.perfect_gas_props("PT_INPUTS", p, T)
        state = fluid.set_state(DmassP_INPUTS, rho, p)

        # Calculate area and geometry properties
        area, area_slope, perimeter, diameter = get_geometry(
            length=x, total_length=length, area_in=area_in, area_ratio=area_ratio
        )

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
        flag_2 = 1
        if abs(determinant) < 1:
            flag = 1

        # Right-hand side of the system
        # G = 1 / T  # For perfect gases
        G = state.isobaric_expansion_coefficient * state.a**2 / state.cp

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
            "distance": x,
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
        }

        # Append the output dictionary to out_list
        out_list.append(out)

        return dy, out, flag, flag_2
    
    # Possible-impossible flow (PIF) algorithm to find critical flow rate
    if critical_flow is True: 
        pif_iterations = 0
        u_guess = (m_impossible+m_possible) / (2*density_in * area_in)
        solution = scipy.integrate.solve_ivp(
            lambda t, y: odefun(t, y)[0],
            [0.0, length],
            [u_guess, density_in, pressure_in],
            t_eval=np.linspace(0, length, number_of_points) if number_of_points else None,
            method="RK45",
            rtol=1e-9,
            atol=1e-9,
        )
        solution = postprocess_ode(solution.t, solution.y, odefun)

        while solution["singularity_detected"] == True:
            m_impossible = m_impossible - m_impossible * 0.005
            m_possible = m_possible  
            u_guess = (m_impossible+m_possible) / (2*density_in * area_in)
            solution = scipy.integrate.solve_ivp(
                lambda t, y: odefun(t, y)[0],
                [0.0, length],
                [u_guess, density_in, pressure_in],
                t_eval=np.linspace(0, length, number_of_points) if number_of_points else None,
                method="RK45",
                rtol=1e-9,
                atol=1e-9,
            )
            solution = postprocess_ode(solution.t, solution.y, odefun)

        while (abs((solution["mach_number"][-1] - 1)) > 0.001):
            pif_iterations += 1
            m_possible = m_possible + m_possible * 0.01
            m_impossible = m_impossible  
            u_guess = (m_impossible+m_possible) / (2*density_in * area_in)
            solution = scipy.integrate.solve_ivp(
                lambda t, y: odefun(t, y)[0],
                [0.0, length],
                [u_guess, density_in, pressure_in],
                t_eval=np.linspace(0, length, number_of_points) if number_of_points else None,
                method="RK45",
                rtol=1e-9,
                atol=1e-9,
            )
            solution = postprocess_ode(solution.t, solution.y, odefun)
        
        flow_rate = u_guess*area_in*density_in
    else:
        pif_iterations = None
        solution = scipy.integrate.solve_ivp(
        lambda t, y: odefun(t, y)[0],
        [0.0, length],
        [velocity_in, density_in, pressure_in],
        t_eval=np.linspace(0, length, number_of_points) if number_of_points else None,
        method="RK45",
        rtol=1e-9,
        atol=1e-9,
        )
        # Optionally process the solution further with postprocess_ode
        solution = postprocess_ode(solution.t, solution.y, odefun)

        flow_rate = velocity_in*density_in*area_in

    return solution, out_list, flow_rate, pif_iterations
