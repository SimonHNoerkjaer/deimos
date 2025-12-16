'''
Comparison of oscillation probability skymaps for IceCube and ARCA in RA,DEC. 
- Atmospheric neutrinos
- Sidereal SME parameters and matter effects can be activated
- Earth layer boundaries are shown
Edited by Johann Ioannou-Nikolaides based on a script by Simon Hilding-Nørkjær
'''

import numpy as np
import time as time_module
from deimos.wrapper.osc_calculator import OscCalculator
from deimos.utils.oscillations import get_coszen_from_path_length
from deimos.utils.plotting import plt, dump_figures_to_pdf, get_intermediate_points, get_number_tex
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
    parser.add_argument("-n", "--num-points", type=int, required=False, default=125, help="Num scan points")
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

    detectors = ["ICECUBE", "ARCA", "P_ONE", "TRIDENT", "HUNT", "GVD"]

    sme_basis = REF_SME_BASIS
    # a_magnitude_eV = REF_SME_a_MAGNITUDE_eV
    a_magnitude_eV = 0.6e-13
    a_mu_eV = get_sme_state_matrix(p33=a_magnitude_eV)
    c_magnitude = 0.
    c_t_nu = get_sme_state_matrix(p33=c_magnitude)

    ra_values_deg = np.linspace(0.0, 360.0, num=args.num_points)
    dec_values_deg = np.linspace(-90.0, 90.0, num=args.num_points+1)
    ra_values_rad = np.deg2rad(ra_values_deg)
    dec_values_rad = np.deg2rad(dec_values_deg)
    ra_grid_rad, dec_grid_rad = np.meshgrid(ra_values_rad, dec_values_rad, indexing="ij")
    grid_shape = ra_grid_rad.shape
    ra_grid_flat_rad, dec_grid_flat_rad = ra_grid_rad.flatten(), dec_grid_rad.flatten()
    time = REF_TIME

    field_directions = ["x", "y", "z"]

    plt.rcParams.update({'font.size': 20})
    fig, ax = plt.subplots(len(field_directions), len(detectors), sharex=True, sharey=True, figsize=(7*len(detectors), 5*len(field_directions)))
    fig.subplots_adjust(right=0.88, wspace=0.07, hspace=0.07)
    fig.suptitle(fr"$E$ = {E_GeV*1e-3:.3g} TeV // Time: {time} // Matter: {matter.title()}")

    data_dict = {}

    # --- Worker function: create calculator inside ---
    def calc_panel(args):
        time_init = time_module.time()
        i, field_direction, j, detector, initial_flavor, nubar, E_GeV, sme_basis, a_mu_eV, c_t_nu, ra_grid_flat_rad, dec_grid_flat_rad, time, solver, kw = args
        sme_params = { "basis":sme_basis, ("a_%s_eV"%field_direction):a_mu_eV, ("c_t%s"%field_direction):c_t_nu }
        calculator = OscCalculator(solver=solver, atmospheric=True, **kw)
        calculator.set_matter("earth")
        # calculator.set_matter('vacuum')
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
        print(f'Setup done: {field_direction}, {detector}')
        P_detector, _, _ = calculator.calc_osc_prob_sme_directional_atmospheric(**calc_kw)
        elapsed = time_module.time() - time_init
        minutes, seconds = divmod(elapsed, 60)
        print(f"Calculation done: {field_direction}, {detector} in {int(minutes)}m {int(seconds)}s")
        return (i, field_direction, j, detector, P_detector)

    import concurrent.futures

    args_list = [
        (i, field_direction, j, detector, initial_flavor, nubar, E_GeV, sme_basis, a_mu_eV, c_t_nu, ra_grid_flat_rad, dec_grid_flat_rad, time, args.solver, kw)
        for i, field_direction in enumerate(field_directions)
        for j, detector in enumerate(detectors)
    ]

    with concurrent.futures.ProcessPoolExecutor() as executor:
        for result in executor.map(calc_panel, args_list):
            i, field_direction, j, detector, P_detector = result
            data_dict[(field_direction, detector)] = P_detector

    # --- Plotting ---
    for i, field_direction in enumerate(field_directions):
        for j, detector in enumerate(detectors):
            P_detector = data_dict[(field_direction, detector)]
            a_label = r"$a^{%s}_{33}$" % (field_direction)
            c_label = r"$c^{t%s}_{33}$" % (field_direction)
            sme_label = []
            if a_magnitude_eV > 0.: sme_label.append(a_label)
            if c_magnitude > 0.: sme_label.append(c_label)
            sme_label = r"\n ".join(sme_label)
            zlabel = r"$%s$" % OscCalculator(solver=args.solver, atmospheric=True, **kw).get_transition_prob_tex(initial_flavor, final_flavor, nubar)
            cmesh = plot_colormap(ax=ax[i, j], x=ra_values_deg, y=dec_values_deg, z=P_detector[..., final_flavor].reshape(grid_shape), zlabel=zlabel, cmap="RdPu", vmin=0., vmax=1.)
            ra_ticks = [0, 90, 180, 270, 360]
            dec_ticks = [-90, -45, 0, 45, 90]
            ax[i, j].set_xticks([0+4, 90, 180, 270, 360-8])
            ax[i, j].set_yticks([-90+4, -45, 0, 45, 90-4])
            if j == 0:
                ax[i, j].set_ylabel("Declination [deg]")
                ax[i, j].text(-0.4, 0.5, sme_label, transform=ax[i, j].transAxes, fontsize=24, color="black", fontweight="bold", verticalalignment='center', bbox=dict(boxstyle='round', facecolor='white', edgecolor='black'))
                ax[i, j].set_yticklabels([ "%i"%t for t in dec_ticks ])
            if i == 0:
                if detector == "P_ONE": detector = "P-ONE"
                ax[i, j].text(0.5, 1.0, detector, transform=ax[i, j].transAxes, color="black",fontweight="bold", va='bottom', ha="center")#, bbox=dict(boxstyle='round', facecolor='white', edgecolor='black'))
            if i == len(field_directions) - 1:
                ax[i, j].set_xlabel("RA [deg]")
                ax[i, j].set_xticklabels([ "%i"%t for t in ra_ticks ])
            ax[i, j].set_xlim(ra_values_deg[0], ra_values_deg[-1])
            ax[i, j].set_ylim(dec_values_deg[0], dec_values_deg[-1])

    # Save data dictionary
    data_dict_str_keys = {str(k): v for k, v in data_dict.items()}
    np.savez_compressed("sme_sidereal_direction_dependence_data_matter_a06e-13.npz", **data_dict_str_keys)
    print("\nData saved to sme_sidereal_direction_dependence_data_matter_a06e-13.npz")
    
    # Add colorbar
    cbar = fig.colorbar(cmesh, ax=ax, shrink=1, pad=0.01, aspect=30)
    cbar.set_label(zlabel)


    plt.savefig(__file__.replace(".py", "_" + args.solver + "_matter_a06e-13.png"))
    print("Figure saved to " + __file__.replace(".py", "_" + args.solver + "_matter_a06e-13.png"))
    print("Script finished at ", time_module.asctime())

    # print("")
    # dump_figures_to_pdf(__file__.replace(".py", "_" + args.solver + ".png"))