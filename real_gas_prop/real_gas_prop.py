import matplotlib as mpl
import CoolProp.CoolProp as CP
import numpy as np

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

# Dynamically add INPUTS fields to the module
# for attr in dir(CP):
#     if attr.endswith('_INPUTS'):
#         globals()[attr] = getattr(CP, attr)

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


def states_to_dict(states):
    """
    Convert a list of state objects into a dictionary.
    Each key is a field name of the state objects, and each value is a NumPy array of all the values for that field.
    """
    state_dict = {}
    for field in states[0].keys():
        state_dict[field] = np.array([getattr(state, field) for state in states])
    return state_dict


class FluidState:
    """
    A class representing the thermodynamic state of a fluid.

    This class is used to store and access the properties of a fluid state.
    Properties can be accessed directly as attributes (e.g., `fluid_state.p` for pressure)
    or through dictionary-like access (e.g., `fluid_state['T']` for temperature).

    Methods
    -------
    to_dict():
        Convert the FluidState properties to a dictionary.
    keys():
        Return the keys of the FluidState properties.
    items():
        Return the items (key-value pairs) of the FluidState properties.

    """

    def __init__(self, properties):
        for key, value in properties.items():
            setattr(self, key, value)

    def __getitem__(self, key):
        return getattr(self, key, None)

    def __str__(self):
        properties_str = "\n   ".join(
            [f"{key}: {getattr(self, key)}" for key in self.__dict__]
        )
        return f"FluidState:\n   {properties_str}"

    # def get(self, key, default=None):
    #     return getattr(self, key, default)

    def to_dict(self):
        return {key: getattr(self, key) for key in self.__dict__}

    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()


class Fluid:
    """
    Represents a fluid with various thermodynamic properties computed via CoolProp.

    This class provides a convenient interface to CoolProp for various fluid property calculations.

    Properties can be accessed directly as attributes (e.g., `fluid.properties["p"]` for pressure)
    or through dictionary-like access (e.g., `fluid.T` for temperature).

    Critical and triple point properties are computed upon initialization and stored internally for convenience.

    Attributes
    ----------
    name : str
        Name of the fluid.
    backend : str
        Backend used for CoolProp, default is 'HEOS'.
    exceptions : bool
        Determines if exceptions should be raised during state calculations. Default is True.
    converged_flag : bool
        Flag indicating whether properties calculations converged.
    properties : dict
        Dictionary of various fluid properties. Accessible directly as attributes (e.g., `fluid.p` for pressure).
    critical_point : FluidState
        Properties at the fluid's critical point.
    triple_point_liquid : FluidState
        Properties at the fluid's triple point in the liquid state.
    triple_point_vapor : FluidState
        Properties at the fluid's triple point in the vapor state.

    Methods
    -------
    set_state(input_type, prop_1, prop_2):
        Set the thermodynamic state of the fluid using specified property inputs.

    Examples
    --------
    Accessing properties:

        - fluid.T - Retrieves temperature directly as an attribute.
        - fluid.properties['p'] - Retrieves pressure through dictionary-like access.

    Accessing critical point properties:

        - fluid.critical_point.p - Retrieves critical pressure.
        - fluid.critical_point['T'] - Retrieves critical temperature.

    Accessing triple point properties:

        - fluid.triple_point_liquid.h - Retrieves liquid enthalpy at the triple point.
        - fluid.triple_point_vapor.s - Retrieves vapor entropy at the triple point.
    """

    def __init__(
        self,
        name,
        backend="HEOS",
        exceptions=True,
        initialize_critical=True,
        initialize_triple=True,
    ):
        self.name = name
        self.backend = backend
        self._AS = CP.AbstractState(backend, name)
        self.exceptions = exceptions
        self.converged_flag = False
        self.properties = {}

        # Initialize variables
        self.sat_liq = None
        self.sat_vap = None
        self.spinodal_liq = None
        self.spinodal_vap = None
        self.pseudo_critical_line = None
        self.Q_quality = None

        # Assign critical point properties
        if initialize_critical:
            self.critical_point = self._compute_critical_point()

        # Assign triple point properties
        if initialize_triple:
            self.triple_point_liquid = self._compute_triple_point_liquid()
            self.triple_point_vapor = self._compute_triple_point_vapor()

    def __getattr__(self, name):
        if name in self.properties:
            return self.properties[name]
        raise AttributeError(f"'Fluid' object has no attribute '{name}'")

    def _compute_critical_point(self):
        """Calculate the properties at the critical point"""
        rho_crit, T_crit = self._AS.rhomass_critical(), self._AS.T_critical()
        self.set_state(DmassT_INPUTS, rho_crit, T_crit)
        return FluidState(self.properties)

    def _compute_triple_point_liquid(self):
        """Calculate the properties at the triple point (liquid state)"""
        self.set_state(QT_INPUTS, 0.00, self._AS.Ttriple())
        return FluidState(self.properties)

    def _compute_triple_point_vapor(self):
        """Calculate the properties at the triple point (vapor state)"""
        self.set_state(QT_INPUTS, 1.00, self._AS.Ttriple())
        return FluidState(self.properties)

    def set_state(self, input_type, prop_1, prop_2):
        """
        Set the thermodynamic state of the fluid based on input properties.

        This method updates the thermodynamic state of the fluid in the CoolProp ``abstractstate`` object
        using the given input properties. It then calculates either single-phase or two-phase
        properties based on the current phase of the fluid.

        If the calculation of properties fails, `converged_flag` is set to False, indicating an issue with
        the property calculation. Otherwise, it's set to True.

        Aliases of the properties are also added to the ``Fluid.properties`` dictionary for convenience.

        Parameters
        ----------
        input_type : str or int
            The variable pair used to define the thermodynamic state. This should be one of the
            predefined input pairs in CoolProp, such as ``PT_INPUTS`` for pressure and temperature.
            For all available input pairs, refer to :ref:`this list <module-input-pairs-table>`.
        prop_1 : float
            The first property value corresponding to the input type (e.g., pressure in Pa if the input
            type is CP.PT_INPUTS).
        prop_2 : float
            The second property value corresponding to the input type (e.g., temperature in K if the input
            type is CP.PT_INPUTS).

        Returns
        -------
        dict
            A dictionary of computed properties for the current state of the fluid. This includes both the
            raw properties from CoolProp and any additional alias properties.

        Raises
        ------
        Exception
            If `throw_exceptions` attribute is set to True and an error occurs during property calculation,
            the original exception is re-raised.


        """
        try:
            # Update Coolprop thermodynamic state
            self._AS.update(input_type, prop_1, prop_2)

            # Retrieve single-phase properties
            if self._AS.phase() != CP.iphase_twophase:
                self.properties = self.compute_properties_1phase()
            else:
                self.properties = self.compute_properties_2phase()

            # Add properties as aliases
            for key, value in PROPERTY_ALIAS.items():
                self.properties[key] = self.properties[value]

            # No errors computing the properies
            self.converged_flag = True

        # Something went wrong while computing the properties
        except Exception as e:
            self.converged_flag = False
            if self.exceptions:
                raise e

        return FluidState(self.properties)
    

    def compute_properties_1phase(self):
        """Get single-phase properties from CoolProp low level interface"""

        props = {}
        props["T"] = self._AS.T()
        props["p"] = self._AS.p()
        props["rhomass"] = self._AS.rhomass()
        props["umass"] = self._AS.umass()
        props["hmass"] = self._AS.hmass()
        props["smass"] = self._AS.smass()
        props["gibbsmass"] = self._AS.gibbsmass()
        props["cvmass"] = self._AS.cvmass()
        props["cpmass"] = self._AS.cpmass()
        props["gamma"] = props["cpmass"] / props["cvmass"]
        props["compressibility_factor"] = self._AS.compressibility_factor()
        props["speed_sound"] = self._AS.speed_sound()
        props["isentropic_bulk_modulus"] = props["rhomass"] * props["speed_sound"] ** 2
        props["isentropic_compressibility"] = 1 / props["isentropic_bulk_modulus"]
        props["isothermal_bulk_modulus"] = 1 / self._AS.isothermal_compressibility()
        props["isothermal_compressibility"] = self._AS.isothermal_compressibility()
        isobaric_expansion_coefficient = self._AS.isobaric_expansion_coefficient()
        props["isobaric_expansion_coefficient"] = isobaric_expansion_coefficient
        props["viscosity"] = self._AS.viscosity()
        props["conductivity"] = self._AS.conductivity()
        props["Q"] = np.nan
        props["quality_mass"] = np.nan
        props["quality_volume"] = np.nan
        props["drho_dP"] = self._AS.first_partial_deriv(CP.iDmass, CP.iP, CP.iHmass)
        props["drho_dh"] = self._AS.first_partial_deriv(CP.iDmass, CP.iHmass, CP.iP)

        return props

    def compute_properties_2phase(self):
        """Get two-phase properties from mixing rules and single-phase CoolProp properties"""

        # Basic properties of the two-phase mixture
        T_mix = self._AS.T()
        p_mix = self._AS.p()
        rho_mix = self._AS.rhomass()
        u_mix = self._AS.umass()
        h_mix = self._AS.hmass()
        s_mix = self._AS.smass()
        gibbs_mix = self._AS.gibbsmass()

        # Instantiate new fluid object to compute saturation properties without changing the state of the class
        temp = CP.AbstractState(self.backend, self.name)

        # Saturated liquid properties
        temp.update(CP.QT_INPUTS, 0.00, T_mix)
        rho_L = temp.rhomass()
        h_L = temp.hmass()
        cp_L = temp.cpmass()
        cv_L = temp.cvmass()
        k_L = temp.conductivity()
        mu_L = temp.viscosity()
        speed_sound_L = temp.speed_sound()
        dsdp_L = temp.first_saturation_deriv(CP.iSmass, CP.iP)

        # Saturated vapor properties
        temp.update(CP.QT_INPUTS, 1.00, T_mix)
        rho_V = temp.rhomass()
        h_V = temp.hmass()
        cp_V = temp.cpmass()
        cv_V = temp.cvmass()
        k_V = temp.conductivity()
        mu_V = temp.viscosity()
        speed_sound_V = temp.speed_sound()
        dsdp_V = temp.first_saturation_deriv(CP.iSmass, CP.iP)

        # Mass fractions of vapor and liquid
        mass_frac_V = (h_mix - h_L) / (h_V - h_L)
        mass_frac_L = 1.0 - mass_frac_V

        # Volume fractions of vapor and liquid
        v_mix = 1.0 / rho_mix
        vol_frac_V = mass_frac_V * (1 / rho_V) / v_mix
        vol_frac_L = mass_frac_L * (1 / rho_L) / v_mix

        # Heat capacities of the two-phase mixture
        cp_mix = mass_frac_L * cp_L + mass_frac_V * cp_V
        cv_mix = mass_frac_L * cv_L + mass_frac_V * cv_V

        # Transport properties of the two-phase mixture
        k_mix = vol_frac_L * k_L + vol_frac_V * k_V
        mu_mix = vol_frac_L * mu_L + vol_frac_V * mu_V

        # Compressibility factor of the two-phase mixture
        M = self._AS.molar_mass()
        R = self._AS.gas_constant()
        Z_mix = p_mix / (rho_mix * (R / M) * T_mix)

        # Speed of sound of the two-phase mixture
        mechanical_equilibrium = vol_frac_L / (
            rho_L * speed_sound_L**2
        ) + vol_frac_V / (rho_V * speed_sound_V**2)
        thermal_equilibrium = T_mix * (
            vol_frac_L * rho_L / cp_L * dsdp_L**2
            + vol_frac_V * rho_V / cp_V * dsdp_V**2
        )
        compressibility_HEM = mechanical_equilibrium + thermal_equilibrium
        if mass_frac_V < 1e-6:  # Avoid discontinuity when Q_v=0
            a_HEM = speed_sound_L
        elif mass_frac_V > 1.0 - 1e-6:  # Avoid discontinuity when Q_v=1
            a_HEM = speed_sound_V
        else:
            a_HEM = (1 / rho_mix / compressibility_HEM) ** 0.5

        # Define small perturbations
        delta_p = 1  # Pa
        delta_h = 1  # J/kg

        # Perturb pressure keeping h fixed (forward difference)
        AS_p_base = CP.AbstractState(self.backend, self.name)
        AS_p_base.update(CP.HmassP_INPUTS, h_mix, p_mix)
        rho_base_p = AS_p_base.rhomass()

        AS_p_plus = CP.AbstractState(self.backend, self.name)
        AS_p_plus.update(CP.HmassP_INPUTS, h_mix, p_mix + delta_p)
        rho_p_plus = AS_p_plus.rhomass()

        drho_dP = (rho_p_plus - rho_base_p) / delta_p
        # print("drho_dP FD (forward)", drho_dP)

        # Perturb enthalpy keeping p fixed (forward difference)
        AS_h_base = CP.AbstractState(self.backend, self.name)
        AS_h_base.update(CP.HmassP_INPUTS, h_mix, p_mix)
        rho_base_h = AS_h_base.rhomass()

        AS_h_plus = CP.AbstractState(self.backend, self.name)
        AS_h_plus.update(CP.HmassP_INPUTS, h_mix + delta_h, p_mix)
        rho_h_plus = AS_h_plus.rhomass()

        drho_dh = (rho_h_plus - rho_base_h) / delta_h
        # print("drho_dh FD (forward)", drho_dh)

        # For reference, CoolProp's internal two-phase derivatives
        try:
            drho_dP = self._AS.first_two_phase_deriv(CP.iDmass, CP.iP, CP.iHmass)
            # print("drho_dP (CoolProp)", drho_dP_exact)

            drho_dh = self._AS.first_two_phase_deriv(CP.iDmass, CP.iHmass, CP.iP)
            # print("drho_dh (CoolProp)", drho_dh_exact)
        except ValueError as e:
            print("CoolProp derivative error:", e)


        # Store properties in dictionary
        properties = {}
        properties["T"] = T_mix
        properties["p"] = p_mix
        properties["rhomass"] = rho_mix
        properties["umass"] = u_mix
        properties["hmass"] = h_mix
        properties["smass"] = s_mix
        properties["gibbsmass"] = gibbs_mix
        properties["cvmass"] = cv_mix
        properties["cpmass"] = cp_mix
        properties["gamma"] = properties["cpmass"] / properties["cvmass"]
        properties["compressibility_factor"] = Z_mix
        properties["speed_sound"] = a_HEM
        properties["isentropic_bulk_modulus"] = rho_mix * a_HEM**2
        properties["isentropic_compressibility"] = (rho_mix * a_HEM**2) ** -1
        properties["isothermal_bulk_modulus"] = np.nan
        properties["isothermal_compressibility"] = np.nan
        properties["isobaric_expansion_coefficient"] = np.nan
        properties["viscosity"] = mu_mix
        properties["conductivity"] = k_mix
        properties["Q"] = mass_frac_V
        properties["quality_mass"] = mass_frac_V
        properties["quality_volume"] = vol_frac_V
        properties["drho_dP"] = drho_dP
        properties["drho_dh"] = drho_dh

        return properties

