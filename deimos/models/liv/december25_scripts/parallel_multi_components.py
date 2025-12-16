'''
Comparison of oscillation probability skymaps for ARCA in RA,DEC.
- Atmospheric neutrinos
- Sidereal SME parameters and matter effects can be activated
- Earth layer boundaries are shown
- 3x3 grid: SME c-parameter components
'''

import numpy as np
import time as time_module
from deimos.wrapper.osc_calculator import OscCalculator
from deimos.utils.plotting import plt, get_intermediate_points
from deimos.utils.constants import *
from deimos.models.liv.sme import get_sme_state_matrix
from deimos.models.liv.paper_plots.paper_def import *

def plot_colormap(ax, x, y, z, zlabel=None, **kw):
    x = get_intermediate_points(x, bounding_points=True)
    y = get_intermediate_points(y, bounding_points=True)
    cmesh = ax.pcolormesh(x, y, z.T, **kw)
    if not any([x in kw for x in ["edgecolor", "edgecolors"]]):
        cmesh.set_edgecolor("face")
    fig = ax.get_figure()
    return cmesh

if __name__ == "__main__":
    print("Script started at ", time_module.asctime())
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--solver", type=str, required=False, default="nusquids", help="Solver name")
    parser.add_argument("-n", "--num-points", type=int, required=False, default=25, help="Num scan points")
    args = parser.parse_args()

    initial_flavor = 1
    final_flavor = 1
    nubar = False
    E_GeV = REF_E_GeV

    kw = {}
    if args.solver == "nusquids":
        kw["energy_nodes_GeV"] = E_GeV
        kw["nusquids_variant"] = "sme"

    matter = "earth"
    detector = "ARCA"
    sme_basis = REF_SME_BASIS

    ra_values_deg = np.linspace(0.0, 360.0, num=args.num_points)
    dec_values_deg = np.linspace(-90.0, 90.0, num=args.num_points+1)
    ra_values_rad = np.deg2rad(ra_values_deg)
    dec_values_rad = np.deg2rad(dec_values_deg)
    ra_grid_rad, dec_grid_rad = np.meshgrid(ra_values_rad, dec_values_rad, indexing="ij")
    grid_shape = ra_grid_rad.shape
    ra_grid_flat_rad, dec_grid_flat_rad = ra_grid_rad.flatten(), dec_grid_rad.flatten()
    time = REF_TIME

    # 3x3 grid of c-parameters
    c_components = [
        ["c_tx", "c_xy", "c_xx"],
        ["c_ty", "c_xz", "c_yy"],
        ["c_tz", "c_yz", "c_zz"]
    ]
    c_magnitude = REF_SME_c_MAGNITUDE

    plt.rcParams.update({'font.size': 16})
    fig, ax = plt.subplots(3, 3, sharex=True, sharey=True, figsize=(15, 12))
    fig.subplots_adjust(right=0.88, wspace=0.07, hspace=0.07)
    fig.suptitle(fr"ARCA: $E$ = {E_GeV*1e-3:.3g} TeV // Time: {time} // Matter: {matter.title()}")

    data_dict = {}

    def calc_panel(args):
        i, j, c_key, initial_flavor, nubar, E_GeV, sme_basis, c_magnitude, ra_grid_flat_rad, dec_grid_flat_rad, time, solver, kw = args
        # Only c parameter, a = 0
        sme_params = {"basis": sme_basis}
        # Set only one c component nonzero
        if c_key.startswith("c_t"):
            # Temporal-spatial components
            direction = c_key[-1]
            c_matrix = get_sme_state_matrix(**{f"p33": 0.0})  # zero matrix
            sme_params[f"c_t{direction}"] = get_sme_state_matrix(p33=c_magnitude)
        else:
            # Spatial-spatial components
            idx_map = {"x": 0, "y": 1, "z": 2}
            i1, i2 = idx_map[c_key[-2]], idx_map[c_key[-1]]
            c_matrix = np.zeros((3, 3))
            c_matrix[i1, i2] = c_magnitude
            sme_params[f"c_{c_key[-2]}{c_key[-1]}"] = c_matrix
        calculator = OscCalculator(solver=solver, atmospheric=True, **kw)
        calculator.set_matter(matter)
        calculator.set_detector(detector)
        calc_kw = {
            "initial_flavor": initial_flavor,
            "nubar": nubar,
            "energy_GeV": E_GeV,
            "ra_rad": ra_grid_flat_rad,
            "dec_rad": dec_grid_flat_rad,
            "time": time,
            "sme_params": sme_params,
        }
        print(f'Setup done: {c_key}')
        P_detector, _, _ = calculator.calc_osc_prob_sme_directional_atmospheric(**calc_kw)
        return (i, j, c_key, P_detector)

    import concurrent.futures

    args_list = [
        (i, j, c_key, initial_flavor, nubar, E_GeV, sme_basis, c_magnitude, ra_grid_flat_rad, dec_grid_flat_rad, time, args.solver, kw)
        for i, row in enumerate(c_components)
        for j, c_key in enumerate(row)
    ]

    with concurrent.futures.ProcessPoolExecutor() as executor:
        for result in executor.map(calc_panel, args_list):
            i, j, c_key, P_detector = result
            data_dict[c_key] = P_detector

    # --- Plotting ---
    for i, row in enumerate(c_components):
        for j, c_key in enumerate(row):
            P_detector = data_dict[c_key]
            c_label = r"$%s$ = %.1e" % (c_key.replace("c_", "c_{").replace("t", "t}"), c_magnitude)
            zlabel = r"$%s$" % OscCalculator(solver=args.solver, atmospheric=True, **kw).get_transition_prob_tex(initial_flavor, final_flavor, nubar)
            cmesh = plot_colormap(ax=ax[i, j], x=ra_values_deg, y=dec_values_deg, z=P_detector[..., final_flavor].reshape(grid_shape), zlabel=zlabel, cmap="RdPu", vmin=0., vmax=1.)
            ra_ticks = [0, 90, 180, 270, 360]
            dec_ticks = [-90, -45, 0, 45, 90]
            ax[i, j].set_xticks([0+4, 90, 180, 270, 360-8])
            ax[i, j].set_yticks([-90+4, -45, 0, 45, 90-4])
            if j == 0:
                ax[i, j].set_ylabel("Declination [deg]")
                ax[i, j].set_yticklabels([ "%i"%t for t in dec_ticks ])
            # if i == 0:
                # ax[i, j].set_title(c_label, fontsize=14)
            if i == 2:
                ax[i, j].set_xlabel("RA [deg]")
                ax[i, j].set_xticklabels([ "%i"%t for t in ra_ticks ])
            ax[i, j].set_xlim(ra_values_deg[0], ra_values_deg[-1])
            ax[i, j].set_ylim(dec_values_deg[0], dec_values_deg[-1])

             # Add small textbox in top left corner
            # Format: c^{xx}_{33}, c^{tx}_{33}, etc.
            if c_key.startswith("c_t"):
                box_label = r"$c^{t%s}_{33}$" % c_key[-1]
            else:
                box_label = r"$c^{%s}_{33}$" % c_key[-2:]
            ax[i, j].text(
                0.035, 0.965, box_label,
                transform=ax[i, j].transAxes,
                fontsize=18, color="white",
                verticalalignment='top', horizontalalignment='left',
                bbox=dict(boxstyle='round', facecolor='black', edgecolor='gray', alpha=0.7)
            )

    # Add colorbar
    cbar = fig.colorbar(cmesh, ax=ax, shrink=1, pad=0.01, aspect=30)
    cbar.set_label(zlabel)

    plt.savefig(__file__.replace(".py", "_" + args.solver + "_arca_sme_cgrid.png"))
    print("Figure saved to " + __file__.replace(".py", "_" + args.solver + "_arca_sme_cgrid.png"))
    print("Script finished at ", time_module.asctime())