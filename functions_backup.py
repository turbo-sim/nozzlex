import scipy.linalg
import scipy.integrate
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import CoolProp.CoolProp as CP
import perfect_gas_prop.perfect_gas_prop as perfect_gas_prop 
from cycler import cycler
import numpy as np
from scipy.linalg import det

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


# Define property aliases
PROPERTY_ALIAS = {
    "P": "p",
    "rho": "rhomass",
    "d": "rhomass",
    "u": "umass",
    "h": "hmass",
    "s": "smass",
    "cv": "cvmass",
    "cp": "cpmass",
    "a": "speed_sound",
    "Z": "compressibility_factor",
    "mu": "viscosity",
    "k": "conductivity",
}


def postprocess_ode(t, y, ode_handle):
    """
    Post-processes the output of an ordinary differential equation (ODE) solver.

    This function takes the time points and corresponding ODE solution matrix,
    and for each time point, it calls a user-defined ODE handling function to
    process the state of the ODE system. It collects the results into a
    dictionary where each key corresponds to a state variable and the values
    are numpy arrays of that state variable at each integration step

    Parameters
    ----------
    t : array_like
        Integration points at which the ODE was solved, as a 1D numpy array.
    y : array_like
        The solution of the ODE system, as a 2D numpy array with shape (n,m) where
        n is the number of points and m is the number of state variables.
    ode_handle : callable
        A function that takes in a integration point and state vector and returns a tuple,
        where the first element is ignored (can be None) and the second element
        is a dictionary representing the processed state of the system.

    Returns
    -------
    ode_out : dict
        A dictionary where each key corresponds to a state variable and each value
        is a numpy array containing the values of that state variable at each integration step.
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

    # Calculate inlet density
    density_in = properties_in["d"]
    speed_sound_in = properties_in["a"]
    R = properties_in["R"]

    # Calculate velocity based on specified parameter
    if mass_flow is not None:
        velocity_in = mass_flow / (area_in * density_in)
    elif mach_in is not None:
        velocity_in = mach_in * properties_in["a"]
    elif critical_flow is True:
        mach_impossible = 0.99
        mach_possible = 0.1
        u_impossible = mach_impossible*speed_sound_in
        u_possible = mach_possible*speed_sound_in
        m_impossible = density_in*u_impossible*area_in
        m_possible = density_in*u_possible*area_in

    # Initialize out_list to store all the state outputs
    out_list = []
    
    # Define the ODE system
    def odefun(t, y):
        x = t  # distance
        v, rho, p = y  # velocity, density, pressure
        
        # Thermodynamic state from perfect gas properties
        T = p / (rho * R)
        state = perfect_gas_prop.perfect_gas_props("PT_INPUTS", p, T)

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
        
        singularity = False
        if determinant < 1e-6:
            singularity = True

        # Right-hand side of the system
        G = 1 / T  # For perfect gases
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

        return dy, out
    
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

        flag = 1
        while flag == 1:
            pif_iterations += 1
            flag = 0
            if (abs((solution["mach_number"][-1] - 1)) > 0.001): # The solution is in the possible region (subsonic)
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
                flag = 1

            if (abs(solution["distance"][-1] - length) > 0.001): # The solution is in the impossible region
                m_impossible = m_impossible - m_impossible * 0.01
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
                flag = 1

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
