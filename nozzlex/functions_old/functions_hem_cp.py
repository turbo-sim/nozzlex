import scipy.linalg
import scipy.integrate
import numpy as np
import barotropy as bpy
import matplotlib as mpl
import matplotlib.pyplot as plt
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


def get_linear_convergent_divergent(length, convergent_length, divergent_length, radius_in, radius_throat, radius_out, width, type):
    
    # if type == "Axisymmetric":
    #     area_in = np.pi*radius_in**2
    #     area_throat = np.pi*radius_throat**2
    #     area_out = np.pi*radius_out**2
        
    #     if length <= convergent_length:
    #         radius_slope = (radius_throat-radius_in)/convergent_length
    #         radius = radius_in + radius_slope*length
    #         area = np.pi*radius**2
    #     elif length > convergent_length:
    #         radius_slope = (radius_out-radius_in)/divergent_length
    #         radius = radius_in + radius_slope*(length-convergent_length)
    #         area = np.pi*radius**2

    #     radius = np.sqrt(area/np.pi)
    #     perimeter = np.pi*radius*2

    if type == "Planar":
        area_in = 2*radius_in*width
        area_throat = 2*radius_throat*width
        area_out = 2*radius_out*width

        if length <= convergent_length:
            area_slope = (area_throat-area_in)/convergent_length
            area = area_in + area_slope*length

        elif length > convergent_length:
            area_slope = (area_out-area_throat)/divergent_length
            area = area_throat + area_slope*(length-convergent_length)

        radius = area/(2*width)
        perimeter = 2*width+2*2*radius

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
    f = (-1.8 * np.log10(6.9 / reynolds + (roughness / diameter / 3.7) ** 1.11)) ** -2
    return f

def richardson(x, rho_L, rho_v):
 
    # gamma = 1/(1+((1-x)/x)*(rho_v/rho_L))
    gamma = rho_L * x / (rho_L * x + rho_v * (1 - x))
    phi_lm_squared = (1-gamma)**(-1.75)
   
    return phi_lm_squared

# The method pipeline_steady_state_1D computes the critical mass flow rate with the Possible-Impossible flow (PIF) algorithm (if specified, else one can directly impose the mass flow/velocity at the inlet)
# and integrate in space to find the adapted supersonic solution. The PIF algorithm is the simplest among the ones available in the 
# iterature and the ODE is a conventional ODE system and not an autonomous system of equations.

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
    
    fluid = bpy.Fluid(fluid_name, backend="HEOS", exceptions=True)
    
    # Define inlet area and length of the nozzle
    total_length = convergent_length+divergent_length
    if nozzle_type == "Planar":
        area_in = 2*radius_in*width
    elif nozzle_type == "Axisymmetric":
        area_in = np.pi*radius_in**2

    # Calculate inlet density
    enthalpy_in = properties_in.h
    density_in = properties_in.rho

    # Calculate velocity based on specified parameter
    if mass_flow is not None:
        velocity_in = mass_flow / (area_in * density_in)
    elif mach_in is not None:
        velocity_in = mach_in * properties_in["a"]
    elif critical_flow is True:
        mach_impossible = 0.2
        mach_possible = 0.0000000001
        u_impossible = mach_impossible*properties_in.a
        u_possible = mach_possible*properties_in.a
        m_impossible = density_in*u_impossible*area_in
        m_possible = density_in*u_possible*area_in
        # m_possible = 19
        # m_impossible = 19
        m_guess = (m_impossible+m_possible) / 2
        u_guess = m_guess / (density_in * area_in)

    # Initialize out_list to store all the state outputs
    out_list = []

    # Define the ODE system
    def odefun(t, y):
        x = t  # distance
        p, v, h = y  # velocity, density, pressure
        
        try:

            # Thermodynamic state from perfect gas properties
            state = fluid.get_state(HmassP_INPUTS, h, p)
            rho = state["rho"]
            viscosity = state["mu"]
            drho_dP = state["drho_dP"]
            drho_dh = state["drho_dh"]

            # Calculate area and geometry properties for convergent-divergent nozzles
            area, dA_dz, p_wall, height = get_linear_convergent_divergent(
                length=x, convergent_length=convergent_length, divergent_length=divergent_length, radius_in=radius_in, radius_throat=radius_throat,
                radius_out=radius_out, width=width, type=nozzle_type)
            diameter = height*2

            # Wall friction calculations
            stress_wall, friction_factor, reynolds = get_wall_friction(
                velocity=v,
                density=rho,
                viscosity=viscosity,
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
                    [v * drho_dP,    rho,          v * drho_dh],
                    [1,              rho * v,      0.00],
                    [-v,              0.00,          rho * v],
                ]
            )

            determinant = det(M)

            # If singularity detected
            flag = 0

            # if determinant > 0:
            #     flag = 1

            if (1-1e-3) < v/state["a"] :
                flag = 1

            # Right-hand side of the system
            b = np.asarray(
                [
                    -rho * v / area * dA_dz,
                    -p_wall / area * stress_wall,
                    v * stress_wall * p_wall / area
                ]
            )

            # Solve the system to get the change in state
            dy = scipy.linalg.solve(M, b)

            # Save the output at each step in the dictionary
            out = {
                "distance": x,
                "velocity": v,
                "density": rho,
                "quality": state["Q"],
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
                "area_slope": dA_dz,
                "perimeter": p_wall,
                "diameter": height,
                "stress_wall": stress_wall,
                "friction_factor": friction_factor,
                # "reynolds": reynolds,
                "source_1": b[0],
                "source_2": b[1],
                "source_3": b[2],
                "determinant": determinant,
                "flag": flag,
            }

            # Append the output dictionary to out_list
            out_list.append(out)

            return dy, out
        
        except Exception as e:
            print(f"[ODEFUN ERROR @ x={x:.4f}] {e}")
            return [np.nan, np.nan, np.nan, np.nan]  # forces integrator to stop
        
    def stop_at_singularity(t, y):
        _, out = odefun(t, y)
        flag = out["flag"]
        return flag-1
    stop_at_singularity.terminal = True  
    stop_at_singularity.direction = 1  
    
    # Possible-impossible flow (PIF) algorithm to find critical flow rate
    if critical_flow is True: 
        pif_iterations = 0
        print("Possible-Impossible Flow (PIF) algorithm starts...")
        error = abs(m_impossible-m_possible)/m_possible
        tol = 1e-3
        while error > tol:
            pif_iterations += 1  
            raw_solution = scipy.integrate.solve_ivp(
                lambda t, y: odefun(t, y)[0],
                [0.0, total_length],
                [pressure_in, u_guess, enthalpy_in],
                # t_eval=np.linspace(0, convergent_length, number_of_points) if number_of_points else None,
                method="RK45",
                rtol=1e-9,
                atol=1e-9,
                events=[stop_at_singularity]
            )
            solution = postprocess_ode(raw_solution.t, raw_solution.y, odefun)

            if raw_solution.t_events[0].size > 0:
                m_impossible = m_guess
            else:
                m_possible = m_guess
            print(m_impossible)
            print(m_possible)

            m_guess = (m_impossible+m_possible) / 2
            u_guess = m_guess / (density_in * area_in)  
            error = abs(m_impossible-m_possible)/m_possible

        flow_rate = u_guess*area_in*density_in

        # Calculate the solution with the last possible flow rate calculated
        u_possible = m_possible/(density_in * area_in)
        possible_solution = scipy.integrate.solve_ivp(
            lambda t, y: odefun(t, y)[0],
            [0.0, total_length],
            [pressure_in, u_possible, enthalpy_in],
            # t_eval=np.linspace(0, convergent_length, number_of_points) if number_of_points else None,
            method="RK45",
            rtol=1e-6,
            atol=1e-6,
        )     
        possible_solution = postprocess_ode(possible_solution.t, possible_solution.y, odefun)

        # Calculate the solution with the last impossible flow rate calculated
        u_impossible = m_impossible/(density_in * area_in)
        impossible_solution = scipy.integrate.solve_ivp(
            lambda t, y: odefun(t, y)[0],
            [0.0, (total_length)],
            [pressure_in, u_impossible, enthalpy_in],
            t_eval=np.linspace(0, convergent_length, number_of_points) if number_of_points else None,
            method="RK45",
            rtol=1e-9,
            atol=1e-9,
        ) 
        impossible_solution = postprocess_ode(impossible_solution.t, impossible_solution.y, odefun) 

        # Calculate the solution with u_guess
        u_avg = (m_possible+m_impossible)/(2*density_in * area_in)
        solution = scipy.integrate.solve_ivp(
            lambda t, y: odefun(t, y)[0],
            [0.0, (total_length)],
            [pressure_in, u_avg, enthalpy_in],
            t_eval=np.linspace(0, convergent_length, number_of_points) if number_of_points else None,
            method="RK45",
            rtol=1e-9,
            atol=1e-9,
        ) 
        solution = postprocess_ode(solution.t, solution.y, odefun) 
        mass_flow = impossible_solution["mass_flow"][0]
        density = impossible_solution["density"][-1]
        x = convergent_length*1.00001
        area, _, _, _ = get_linear_convergent_divergent(x, 
                                                        convergent_length=convergent_length,
                                                        divergent_length=divergent_length,
                                                        radius_in=radius_in,
                                                        radius_out=radius_out,
                                                        radius_throat=radius_throat,
                                                        width=width,
                                                        type=nozzle_type)
        velocity = mass_flow/(density*area)

        # Calculate the supersonic branch
        supersonic_solution = scipy.integrate.solve_ivp(
            lambda t, y: odefun(t, y)[0],
            [x, (total_length)],
            [impossible_solution["pressure"][-1],
             velocity, 
             impossible_solution["enthalpy"][-1]],
            # t_eval=np.linspace(0, convergent_length, number_of_points) if number_of_points else None,
            method="RK45",
            rtol=1e-9,
            atol=1e-9,
        ) 
        supersonic_solution = postprocess_ode(supersonic_solution.t, supersonic_solution.y, odefun) 

    else:
        pif_iterations = None
        solution = scipy.integrate.solve_ivp(
        lambda t, y: odefun(t, y)[0],
        [0.0, total_length],
        [pressure_in, velocity_in, enthalpy_in],
        t_eval=np.linspace(0, total_length, number_of_points) if number_of_points else None,
        method="RK45",
        rtol=1e-9,
        atol=1e-9,
        )
        solution = postprocess_ode(solution.t, solution.y, odefun)

        flow_rate = velocity_in*density_in*area_in
 

    return supersonic_solution, possible_solution, impossible_solution, solution, flow_rate, pif_iterations

def pipeline_steady_state_1D_old_matrix(
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
    
    fluid = bpy.Fluid(fluid_name, backend="HEOS", exceptions=True)
    
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
        mach_impossible = 0.2
        mach_possible = 0.0001
        u_impossible = mach_impossible*properties_in.a
        u_possible = mach_possible*properties_in.a
        m_impossible = density_in*u_impossible*area_in
        m_possible = density_in*u_possible*area_in
        m_possible = 0.025
        m_impossible = 0.025
        m_guess = (m_impossible+m_possible) / 2
        u_guess = m_guess / (density_in * area_in)

    # Initialize out_list to store all the state outputs
    out_list = []
    
    # Define the ODE system
    def odefun(t, y):
        x = t  # distance
        v, rho, p = y  # velocity, density, pressure    

        try:

            # Thermodynamic state from perfect gas properties
            state = fluid.set_state(DmassP_INPUTS, rho, p)

            # Calculate area and geometry properties for convergent-divergent nozzles
            area, area_slope, perimeter, radius = get_linear_convergent_divergent(
                length=x, convergent_length=convergent_length, divergent_length=divergent_length, radius_in=radius_in, radius_throat=radius_throat,
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

            # If singularity detected¨
            flag = 0
            # if (1-1e-6) < v/state["a"] < (1+1e-6):
            #     flag = 1
            # if determinant < 1e-6:
            #     flag = 1

            # Right-hand side of the system
            G = state.isobaric_expansion_coefficient * state.a**2 / state.cp
            # G = 0
            b = np.asarray(
                [
                    -rho * v / area * area_slope,
                    -perimeter / area * stress_wall,
                    perimeter / area * G / v * (stress_wall * v),
                ]
            )

            # Solve the system to get the change in state
            dy = scipy.linalg.solve(M, b)

            # Save the output at each step in the dictionary
            out = {
                "distance": x,
                "velocity": v,
                "density": rho,
                "quality": state["Q"],
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
                "flag": flag
            }

            # Append the output dictionary to out_list
            out_list.append(out)

            return dy, out
            
        except Exception as e:
            print(f"[ODEFUN ERROR @ x={x:.4f}] {e}")
            return [np.nan, np.nan, np.nan, np.nan]  # forces integrator to stop
        
    def stop_at_singularity(t, y):
        _, out = odefun(t, y)
        flag = out["flag"]
        return flag-1

    stop_at_singularity.terminal = True  
    stop_at_singularity.direction = 1  
    
    # Possible-impossible flow (PIF) algorithm to find critical flow rate
    if critical_flow is True: 
        pif_iterations = 0
        print("Possible-Impossible Flow (PIF) algorithm starts...")
        error = abs(m_impossible-m_possible)/m_possible
        tol = 1e-2
        while error > tol:
            pif_iterations += 1  
            raw_solution = scipy.integrate.solve_ivp(
                lambda t, y: odefun(t, y)[0],
                [0.0, 0.463],
                [u_guess, density_in, pressure_in],
                t_eval=np.linspace(0, convergent_length, number_of_points) if number_of_points else None,
                method="RK45",
                rtol=1e-9,
                atol=1e-9,
                events=[stop_at_singularity]
            )
            solution = postprocess_ode(raw_solution.t, raw_solution.y, odefun)

            if raw_solution.t_events[0].size > 0:
                m_impossible = m_guess
            else:
                m_possible = m_guess

            m_guess = (m_impossible+m_possible) / 2
            u_guess = m_guess / (density_in * area_in)  
            error = abs(m_impossible-m_possible)/m_possible

        flow_rate = u_guess*area_in*density_in

        # Calculate the solution with the last possible flow rate calculated
        u_possible = m_possible/(density_in * area_in)
        possible_solution = scipy.integrate.solve_ivp(
            lambda t, y: odefun(t, y)[0],
            [0.0, 0.463],
            [u_possible, density_in, pressure_in],
            # t_eval=np.linspace(0, convergent_length, 500),
            method="RK45",
            rtol=1e-5,
            atol=1e-5,
        )     
        possible_solution = postprocess_ode(possible_solution.t, possible_solution.y, odefun)

        # # Calculate the solution with the last impossible flow rate calculated
        # u_impossible = m_impossible/(density_in * area_in)
        # impossible_solution = scipy.integrate.solve_ivp(
        #     lambda t, y: odefun(t, y)[0],
        #     [0.0, 0.463],
        #     [u_impossible, density_in, pressure_in],
        #     t_eval=np.linspace(0, convergent_length, number_of_points) if number_of_points else None,
        #     method="RK45",
        #     rtol=1e-9,
        #     atol=1e-9,
        # ) 
        # impossible_solution = postprocess_ode(impossible_solution.t, impossible_solution.y, odefun) 

        # # Calculate the solution with u_guess
        # u_avg = (m_possible+m_impossible)/(2*density_in * area_in)
        # solution = scipy.integrate.solve_ivp(
        #     lambda t, y: odefun(t, y)[0],
        #     [0.0, 0.46],
        #     [u_avg, density_in, pressure_in],
        #     t_eval=np.linspace(0, convergent_length, number_of_points) if number_of_points else None,
        #     method="RK45",
        #     rtol=1e-5,
        #     atol=1e-5,
        # ) 
        # solution = postprocess_ode(solution.t, solution.y, odefun) 
        # mass_flow = solution["mass_flow"][0]
        # density = solution["density"][-1]
        # x = convergent_length*1.00001
        # area, area_slope, perimeter, radius = get_linear_convergent_divergent(x, 
        #                                                                       convergent_length=convergent_length,
        #                                                                       divergent_length=divergent_length,
        #                                                                       radius_in=radius_in,
        #                                                                       radius_out=radius_out,
        #                                                                       radius_throat=radius_throat,
        #                                                                       width=width,
        #                                                                       type=nozzle_type)
        # velocity = mass_flow/(density*area)

        # # Calculate the supersonic branch
        # supersonic_solution = scipy.integrate.solve_ivp(
        #     lambda t, y: odefun(t, y)[0],
        #     [x, 0.463],
        #     [impossible_solution["velocity"][-1]*1.02, 
        #      impossible_solution["density"][-1], 
        #      impossible_solution["pressure"][-1]],
        #     t_eval=np.linspace(0, convergent_length, number_of_points) if number_of_points else None,
        #     method="RK45",
        #     rtol=1e-9,
        #     atol=1e-9,
        # ) 
        # supersonic_solution = postprocess_ode(supersonic_solution.t, supersonic_solution.y, odefun) 
        supersonic_solution = 0
        impossible_solution = 0
        solution = 0
        flow_rate = 0
        pif_iterations = 0


    else:
        pif_iterations = None
        solution = scipy.integrate.solve_ivp(
        lambda t, y: odefun(t, y)[0],
        [0.0, total_length],
        [velocity_in, density_in, pressure_in],
        t_eval=np.linspace(0, total_length, number_of_points) if number_of_points else None,
        method="RK45",
        rtol=1e-9,
        atol=1e-9,
        )
        solution = postprocess_ode(solution.t, solution.y, odefun)

        flow_rate = velocity_in*density_in*area_in
 

    return supersonic_solution, possible_solution, impossible_solution, solution, flow_rate, pif_iterations


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
    
    fluid = bpy.Fluid(fluid_name, backend="HEOS", exceptions=True)
    
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
        mach_impossible = 0.25
        mach_possible = 0.0000001
        u_impossible = mach_impossible*properties_in.a
        u_possible = mach_possible*properties_in.a
        m_impossible = density_in*u_impossible*area_in
        m_possible = density_in*u_possible*area_in
        # m_possible = 0.02535
        # m_impossible = 0.02535
        m_guess = (m_impossible+m_possible) / 2
        u_guess = m_guess / (density_in * area_in)

    # Initialize out_list to store all the state outputs
    out_list = []
    
    # Define the ODE system
    def odefun(y):
        x, v, rho, p = y  # position, velocity, density, pressure

        try:
            # Thermodynamic state from perfect gas properties
            state = fluid.get_state(DmassP_INPUTS, rho, p)

            # Calculate area and geometry properties for convergent-divergent nozzles
            area, area_slope, perimeter, radius = get_linear_convergent_divergent(
                length=x, convergent_length=convergent_length, divergent_length=divergent_length, radius_in=radius_in, radius_throat=radius_throat,
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
            det_D = det(M)
            det_N1 = det(M1)
            det_N2 = det(M2)
            det_N3 = det(M3)
            # print("det 1", det_D)
            # print("det 2", det_N1)
            # print("det 3", det_N2)
            # print("det 4", det_N3)
            # print(" ")
        
            dy = [det_D, det_N1, det_N2, det_N3]

            # Save the output at each step in the dictionary
            out = {
                "distance": x,
                "velocity": v,
                "density": rho,
                "pressure": p,
                "temperature": state["T"],
                "quality": state["Q"],
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
                "determinant_D": det_D,
                "determinant_N1": det_N1,
                "determinant_N2": det_N2,
                "determinant_N3": det_N3,
            }

            # Append the output dictionary to out_list
            out_list.append(out)

            return dy, out
        
        except Exception as e:
            print(f"[ODEFUN ERROR @ x={x:.4f}] {e}")
            return [np.nan, np.nan, np.nan, np.nan]  # forces integrator to stop

    # This function avoid to keep solving for the Impossible Flow when the matrix
    # is singular (backward flow)    
    def stop_at_zero_det(t, y):
        _, out = odefun(y)
        det_M = out["determinant_D"]
        return det_M

    stop_at_zero_det.terminal = True  # Stop when det(M) = 0
    stop_at_zero_det.direction = 0   # Trigger event when det(M) == 0, no direction preference
    
    # As in this method we are not integrating in the space variable x but for the dummy
    # variable t, it is necessary to stop when the length of the nozzle is achieved
    def stop_at_length(t, y):
        x = y[0]               
        return x - total_length      
    stop_at_length.terminal = True     
    stop_at_length.direction = 1  

    def stop_at_zero(t, y):
        x = y[0]  # assuming y[0] is the nozzle length variable (x)
        return x  

    stop_at_zero.terminal = True
    stop_at_zero.direction = -1 

    # Possible-impossible flow (PIF) algorithm to find critical flow rate
    if critical_flow is True: 
        pif_iterations = 0
        print("Possible-Impossible Flow (PIF) algorithm starts...")
        error = abs(m_impossible-m_possible)/m_possible
        tol = 1e-2
        while error > tol:
            pif_iterations += 1  
            raw_solution = scipy.integrate.solve_ivp(
                lambda t, y: odefun(y)[0],
                [0, 1],
                [0, u_guess, density_in, pressure_in],
                t_eval=np.linspace(0, convergent_length, number_of_points) if number_of_points else None,
                method="RK45",
                rtol=1e-9,
                atol=1e-9,
                events=[stop_at_length, stop_at_zero_det]
            )
            solution = postprocess_ode_autonomous(raw_solution.t, raw_solution.y, odefun)
            
            if raw_solution.t_events[1].size > 0:
                m_impossible = m_guess
            else:
                m_possible = m_guess

            m_guess = (m_impossible+m_possible) / 2
            u_guess = m_guess / (density_in * area_in)  
            error = abs(m_impossible-m_possible)/m_possible
            print(m_impossible)
            print(m_possible)

        flow_rate = u_guess*area_in*density_in


        # Calculate the solution with the last possible flow rate calculated
        u_possible = m_possible/(density_in * area_in)
        possible_solution = scipy.integrate.solve_ivp(
            lambda t, y: odefun(y)[0],
            [0, 1],
            [0, u_possible, density_in, pressure_in],
            # t_eval=np.linspace(0, convergent_length, number_of_points) if number_of_points else None,
            method="Radau",
            rtol=1e-9,
            atol=1e-9,
            events=[stop_at_length, stop_at_zero_det]
        )     
        possible_solution = postprocess_ode_autonomous(possible_solution.t, possible_solution.y, odefun)
        
        # Calculate the solution with the last impossible flow rate calculated
        u_impossible = m_impossible/(density_in * area_in)
        impossible_solution = scipy.integrate.solve_ivp(
            lambda t, y: odefun(y)[0],
            [0, 1],
            [0, u_impossible, density_in, pressure_in],
            t_eval=np.linspace(0, convergent_length, number_of_points) if number_of_points else None,
            method="RK45",
            rtol=1e-9,
            atol=1e-9,
            events=[stop_at_length, stop_at_zero_det]
        ) 
        impossible_solution = postprocess_ode_autonomous(impossible_solution.t, impossible_solution.y, odefun) 

        # Calculate the solution with u_guess
        u_avg = (m_possible+m_impossible)/(2*density_in * area_in)
        solution = scipy.integrate.solve_ivp(
            lambda t, y: odefun(y)[0],
            [0, 1],
            [0, u_avg, density_in, pressure_in],
            t_eval=np.linspace(0, convergent_length, number_of_points) if number_of_points else None,
            method="Radau",
            rtol=1e-9,
            atol=1e-9,
            events=[stop_at_length, stop_at_zero_det]
        ) 
        solution = postprocess_ode_autonomous(solution.t, solution.y, odefun) 

        mass_flow = solution["mass_flow"][0]
        density = solution["density"][-1]
        x = convergent_length*1.00001
        area, _, _, _ = get_linear_convergent_divergent(x, 
                                                        convergent_length=convergent_length,
                                                        divergent_length=divergent_length,
                                                        radius_in=radius_in,
                                                        radius_out=radius_out,
                                                        radius_throat=radius_throat,
                                                        width=width,
                                                        type=nozzle_type)
        velocity = mass_flow/(density*area)

        supersonic_solution = scipy.integrate.solve_ivp(
            lambda t, y: odefun(y)[0],
            [1, 0], # Inverting dummy variable limits so you so not go backward at the singularity (all the determinants<0)
            [x,
             velocity, 
             solution["density"][-1], 
             solution["pressure"][-1]],
            t_eval=np.linspace(0, convergent_length, number_of_points) if number_of_points else None,
            method="RK45",
            rtol=1e-9,
            atol=1e-9,
            events=[stop_at_length, stop_at_zero]
        ) 
        supersonic_solution = postprocess_ode_autonomous(supersonic_solution.t, supersonic_solution.y, odefun) 
        print("rate out", supersonic_solution["mass_flow"][-1])
    # To fix:
    else:
        pif_iterations = None
        solution = scipy.integrate.solve_ivp(
        lambda t, y: odefun(t, y)[0],
        [0.0, total_length],
        [velocity_in, density_in, pressure_in],
        t_eval=np.linspace(0, total_length, number_of_points) if number_of_points else None,
        method="Radau",
        rtol=1e-9,
        atol=1e-9,
        )
        solution = postprocess_ode(solution.t, solution.y, odefun)

        flow_rate = velocity_in*density_in*area_in
 

    return supersonic_solution, possible_solution, impossible_solution, solution, flow_rate, pif_iterations

