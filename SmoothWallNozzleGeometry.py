import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

class SmoothWallNozzleGeometry:
    def __init__(self, file_path=None, shape="circular", nozzle_width=None):
        """
        Initialize the smooth wall nozzle geometry. Geometry can be circular or rectangular.

        Args:
            file_path (str): Path to the CSV file containing z and r (or height) data.
            shape (str): "circular" for circular cross-sections, "rectangular" for rectangular ones.
            nozzle_width (float): Width of the nozzle for rectangular shapes (ignored for circular).
        """
        if shape not in ["circular", "rectangular"]:
            raise ValueError("Shape must be either 'circular' or 'rectangular'.")
        self.shape = shape
        self.nozzle_width = nozzle_width if shape == "rectangular" else None

        if file_path:
            self.load_geometry_from_csv(file_path)

        # Discretized geometry
        self.sectional_positions = []  # Axial positions
        self.sectional_heights = []  # Radii (or heights for rectangular sections)
        self.sectional_areas = []  # Cross-sectional areas
        self.inclination_angles = []  # Inclination angles between sections
        self.pressure_gradient_areas = []  # Pressure gradient areas
        self.shear_stress_areas = []  # Shear stress areas

        self.convergent_d = None
        self.straight_d = None
        self.divergent_d = None
        self.CVs_convergent = None
        self.CVs_straight = None
        self.CVs_divergent = None
        # Section lengths
        self.convergent_length = None
        self.straight_length = None
        self.divergent_length = None


    def load_geometry_from_csv(self, file_path):
        """
        Load smooth wall geometry data from a CSV file.

        Args:
            file_path (str): Path to the CSV file.

        Raises:
            ValueError: If required columns are missing.
        """
        try:
            # Read the CSV file
            data = pd.read_csv(file_path)

            # Ensure required columns are present
            if 'z(m)' not in data.columns or 'y(m)' not in data.columns:
                raise ValueError("The CSV file must contain 'z(m)' and 'y(m)' columns.")

            # Assign z (axial positions) and r (radial positions)
            self.z = np.array(data['z(m)'].values)
            self.r = np.array(data['y(m)'].values)

            # Sort z and r for consistency
            sorted_indices = np.argsort(self.z)
            self.z = self.z[sorted_indices]
            self.r = self.r[sorted_indices]

        except FileNotFoundError:
            raise ValueError(f"File not found: {file_path}")
    def discretize_geometry(self, CVs_convergent=10, CVs_straight=10, CVs_divergent=10,
                            convergent_d=0.85, straight_d=1.0, divergent_d=1.1, convergent_length=None, straight_length=None, divergent_length=None):
        """
        Discretize the smooth wall nozzle geometry using geometric progression for each section.
        Radii/height interpolation is handled using a cubic spline for smooth transitions.

        Args:
            CVs_convergent (int): Number of control volumes for the convergent section.
            CVs_straight (int): Number of control volumes for the straight section.
            CVs_divergent (int): Number of control volumes for the divergent section.
            convergent_d (float): Geometric progression factor for the convergent section.
            straight_d (float): Geometric progression factor for the straight section.
            divergent_d (float): Geometric progression factor for the divergent section.
        """
        if not hasattr(self, "z") or not hasattr(self, "r"):
            raise ValueError("Geometry data is missing. Load geometry using `load_geometry_from_csv` first.")

        # Discretization factors and control volumes
        self.convergent_d = convergent_d
        self.straight_d = straight_d
        self.divergent_d = divergent_d
        self.CVs_convergent = CVs_convergent
        self.CVs_straight = CVs_straight
        self.CVs_divergent = CVs_divergent
        # Section lengths
        self.convergent_length = convergent_length
        self.straight_length = straight_length
        self.divergent_length = divergent_length


        # Sort axial positions (z) and radii (r) if not sorted
        sorted_indices = np.argsort(self.z)
        z_sorted = self.z[sorted_indices]
        r_sorted = self.r[sorted_indices]

        # Identify throat location (minimum r)
        throat_index = np.argmin(r_sorted)

        # Separate geometry into sections
        z_convergent = z_sorted[:throat_index + 1]
        r_convergent = r_sorted[:throat_index + 1]
        z_straight = z_sorted[throat_index:throat_index + 1]  # Single-point straight section
        r_straight = r_sorted[throat_index:throat_index + 1]
        z_divergent = z_sorted[throat_index:]
        r_divergent = r_sorted[throat_index:]

        # Helper function for geometric progression spacing
        def _geometric_spacing(z_start, z_end, CVs, d):
            """
            Generate axial positions with geometric progression between z_start and z_end.

            Args:
                z_start (float): Starting axial position.
                z_end (float): Ending axial position.
                CVs (int): Number of control volumes (points).
                d (float): Geometric progression factor.

            Returns:
                np.ndarray: Discretized axial positions (CVs + 1 points including start and end).
            """
            length = z_end - z_start

            if np.isclose(length, 0.0, atol=1e-12):  # Handle zero-length sections directly
                return np.array([z_start] * (CVs + 1))

            # Generate the geometric progression for the intervals
            dz_list = [(length * (1 - d) / (1 - d ** CVs)) * (d ** (i - 1)) for i in range(1, CVs + 1)]
            z_smooth = np.cumsum([z_start] + dz_list)

            # Ensure that the last point exactly matches z_end due to floating-point precision
            z_smooth[-1] = z_end

            return z_smooth

        # Helper function for smooth radii interpolation using cubic spline
        def _interpolate_radii(z_section, r_section, z_smooth):
            """
            Interpolate radii/heights over discretized `z_smooth` positions using a cubic spline.

            Args:
                z_section (np.ndarray): Original axial positions in the section.
                r_section (np.ndarray): Original radii/height values in the section.
                z_smooth (np.ndarray): Discretized axial positions.

            Returns:
                np.ndarray: Smoothly interpolated radii/heights at `z_smooth` positions.
            """
            spline_fit = CubicSpline(z_section, r_section, bc_type='natural')
            return spline_fit(z_smooth)

        # Discretized outputs
        self.sectional_positions = []
        self.sectional_heights = []

        # Convergent section
        if len(z_convergent) > 1:
            z_convergent_smooth = _geometric_spacing(z_convergent[0], z_convergent[-1], CVs_convergent, convergent_d)
            r_convergent_smooth = _interpolate_radii(z_convergent, r_convergent, z_convergent_smooth)
            self.sectional_positions.extend(z_convergent_smooth)
            self.sectional_heights.extend(r_convergent_smooth)

        # Straight section (handle only if a valid straight section exists)
        if len(z_straight) > 1:
            z_straight_smooth = _geometric_spacing(z_straight[0], z_straight[-1], CVs_straight, straight_d)
            r_straight_smooth = _interpolate_radii(z_straight, r_straight, z_straight_smooth)
            self.sectional_positions.extend(z_straight_smooth[1:])  # Avoid duplicating the throat point
            self.sectional_heights.extend(r_straight_smooth[1:])

        # Divergent section
        if len(z_divergent) > 1:
            z_divergent_smooth = _geometric_spacing(z_divergent[0], z_divergent[-1], CVs_divergent, divergent_d)
            r_divergent_smooth = _interpolate_radii(z_divergent, r_divergent, z_divergent_smooth)
            self.sectional_positions.extend(z_divergent_smooth[1:])  # Avoid duplicating the throat point
            self.sectional_heights.extend(r_divergent_smooth[1:])

        # Compute cross-sectional areas based on current shape
        if self.shape == "circular":
            self.sectional_areas = (np.pi * np.array(self.sectional_heights) ** 2).tolist()
        elif self.shape == "rectangular":
            if self.nozzle_width is None:
                raise ValueError("nozzle_width must be provided for rectangular cross-sections.")
            self.sectional_areas = (self.nozzle_width * np.array(self.sectional_heights)).tolist()

    def calculate_inclination_angles(self):
        """Calculate angles between consecutive sections."""
        if not self.sectional_positions or not self.sectional_heights:
            raise ValueError("Discretized geometry data required.")
        self.inclination_angles = []
        for i in range(1, len(self.sectional_positions)):
            z1, z2 = self.sectional_positions[i - 1], self.sectional_positions[i]
            r1, r2 = self.sectional_heights[i - 1], self.sectional_heights[i]
            theta = np.arctan(np.abs((r2 - r1)) / (z2 - z1))
            self.inclination_angles.append(np.degrees(theta))


    def visualize_geometry(self):
        """
        Plot the geometry of the smooth wall nozzle based on its discretized positions, heights, and sectional areas.

        Args:
            self (SmoothWallNozzleGeometry): An instance of the SmoothWallNozzleGeometry class with pre-discretized geometry.
        """
        if not self.sectional_positions:
            print("The geometry has not been discretized yet. Please call `discretize_geometry` first.")
            return

        # Update plot style for better aesthetics
        plt.rcParams.update({
            "text.usetex": False,  # Disable LaTeX rendering for compatibility
            "font.family": "sans-serif",
            "axes.labelsize": 12,
            "font.size": 12,
            "legend.fontsize": 12,
            "axes.titlesize": 14,
        })

        # Extract data for plotting
        positions = self.sectional_positions
        heights = self.sectional_heights
        areas = self.sectional_areas

        # Create subplots for geometry profile and cross-sectional area profile
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        # 1. Plot heights (geometry profile)
        upper_bound = heights
        lower_bound = [-h for h in heights]  # Mirror along x-axis for lower wall
        ax1.plot(positions, upper_bound, marker='o', linestyle='-', color='b', label="Upper Wall")
        ax1.plot(positions, lower_bound, marker='o', linestyle='-', color='r', label="Lower Wall")
        ax1.fill_between(positions, lower_bound, upper_bound, color='blue', alpha=0.1, label="Nozzle Shape")
        ax1.set_xlabel("Axial Position (Z) [m]")
        ax1.set_ylabel("Height/Radius (H) [m]")
        ax1.set_title("Nozzle Geometry Profile")
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.legend(loc="best")
        ax1.set_xlim(left=min(positions), right=max(positions))  # Limit x-axis to the nozzle range
        ax1.set_ylim(bottom=min(lower_bound) * 1.1,
                     top=max(upper_bound) * 1.1)  # Expand y-axis for better visualization

        # 2. Plot cross-sectional areas
        ax2.plot(positions, areas, marker='o', linestyle='-', color='g', label="Area Profile")
        ax2.fill_between(positions, 0, areas, color='green', alpha=0.2)
        ax2.set_xlabel("Axial Position (Z) [m]")
        ax2.set_ylabel("Cross-Sectional Area (A) [m²]")
        ax2.set_title("Nozzle Cross-Sectional Area Profile")
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.legend(loc="best")
        ax2.set_xlim(left=min(positions), right=max(positions))  # Limit x-axis to the nozzle range
        ax2.set_ylim(bottom=0, top=max(areas) * 1.1)  # Expand y-axis to enhance area profile visibility

        # Adjust layout and show plot
        plt.tight_layout()
        plt.show()
    def compute_circumferential_areas(self):
        """Compute pressure gradient and shear stress areas."""
        if not self.sectional_positions or not self.sectional_heights:
            raise ValueError("Discretized geometry data required.")
        self.pressure_gradient_areas = []
        self.shear_stress_areas = []

        for i in range(1, len(self.sectional_positions)):
            z1, z2 = self.sectional_positions[i - 1], self.sectional_positions[i]
            r1, r2 = self.sectional_heights[i - 1], self.sectional_heights[i]

            # Slant length between sections
            slant_length = np.sqrt((z2 - z1)**2 + (r2 - r1)**2)

            # Pressure gradient area (LU walls)
            if self.shape == "circular":
                pressure_area = np.pi * (r1 + r2) * (z2 - z1)
            elif self.shape == "rectangular":
                pressure_area = 2 * self.nozzle_width * slant_length

            self.pressure_gradient_areas.append(pressure_area)

            # Shear stress area (All Walls)
            if self.shape == "circular":
                shear_area = np.pi * (r1 + r2) * (z2 - z1)
            elif self.shape == "rectangular": # Side Walls  + Top Wall + Lower Wall
                shear_area = 2 * self.nozzle_width * slant_length + (r1 + r2) * (z2 - z1)
            self.shear_stress_areas.append(shear_area)

    def get_sectional_positions(self):
        return self.sectional_positions

    def get_sectional_heights(self):
        return self.sectional_heights

    def get_sectional_areas(self):
        return self.sectional_areas

    def get_inclination_angles(self):
        return self.inclination_angles

    def get_pressure_gradient_areas(self):
        return self.pressure_gradient_areas

    def get_shear_stress_areas(self):
        return self.shear_stress_areas

if __name__ == "__main__":

    SuperMobiDick = SmoothWallNozzleGeometry(file_path=r"C:\Users\ancio\OneDrive - Danmarks Tekniske Universitet\Documents\python_scripts\space_marching\Super_Moby_Dick_Water_Nozzle.csv")
    SuperMobiDick.discretize_geometry()
    SuperMobiDick.visualize_geometry()
    SuperMobiDick.calculate_inclination_angles()
    SuperMobiDick.compute_circumferential_areas()
    sectional_positions = SuperMobiDick.get_sectional_positions()
    sectional_radii = SuperMobiDick.get_sectional_heights()
    pressure_gradient_areas = SuperMobiDick.get_pressure_gradient_areas()
    shear_stress_areas = SuperMobiDick.get_shear_stress_areas() 
    cross_sectional_areas = SuperMobiDick.get_sectional_areas()
    z = 0.01

    # Interpolate geometry-dependent parameters
    A = np.interp(
        z, SuperMobiDick.get_sectional_positions(),
        SuperMobiDick.get_sectional_areas()
    )  # Cross-sectional area
    dA_dz = (np.interp(
        z + 1e-3, SuperMobiDick.get_sectional_positions(),
        SuperMobiDick.get_sectional_areas()
    ) - A) / 1e-3  # Area derivative

    height = np.interp(
        z, SuperMobiDick.get_sectional_positions(),
        SuperMobiDick.get_sectional_heights()
    )  # Sectional height
    p_wall = 2 * np.pi * height  # Wetted perimeter for circular cross-section
    

    