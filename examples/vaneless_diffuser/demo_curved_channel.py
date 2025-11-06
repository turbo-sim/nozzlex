import os
import yaml
import jaxprop as jxp
import matplotlib.pyplot as plt

from nozzlex import vaneless_channel as vcm

jxp.set_plot_options(grid=False)

OUTDIR = "output"
os.makedirs(OUTDIR, exist_ok=True)

if __name__ == "__main__":

    # Load configuration file
    config_file = "case_radial_axial_bend.yaml"

    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    # Define the working fluid
    fluid = jxp.FluidPerfectGas("air", T_ref=300, p_ref=101325)

    # Create the vaneless channel component
    diffuser = vcm.VanelessChannel.from_dict(config, fluid)

    # Solve the model
    out = diffuser.solve()

    # Plot the solution
    diffuser.plot_geometry(plot_control_points=True)
    diffuser.plot_streamlines(solution=out)
    diffuser.plot_solution_contour(solution=out, var_name="Ma")
    diffuser.plot_efficiency_breakdown(solution=out)
    diffuser.plot_skin_friction_distribution(solution=out)

    # # Save figures
    # jxp.savefig_in_formats(fig_1, os.path.join(OUTDIR, "case_bend_mach"))
    # jxp.savefig_in_formats(fig_2, os.path.join(OUTDIR, "case_bend_skin_friction"))
    # jxp.savefig_in_formats(fig_3, os.path.join(OUTDIR, "case_bend_efficiency"))
    # jxp.savefig_in_formats(fig_4, os.path.join(OUTDIR, "case_bend_streamlines"))

    # Show figures
    plt.show()
