'''
Comparison of oscillation probability skymaps for ARCA in RA,DEC.
- Atmospheric neutrinos
- Sidereal SME parameters and matter effects can be activated
- Earth layer boundaries are shown
- Compares neutrino (nubar=False) and antineutrino (nubar=True) in 1x2 subplot
Edited by Johann Ioannou-Nikolaides based on a script by Simon Hilding-Nørkjær
'''

import numpy as np
import time as time_module
from deimos.wrapper.osc_calculator import OscCalculator
from deimos.utils.oscillations import get_coszen_from_path_length
from deimos.utils.plotting import plt, plot_colormap, get_number_tex
from deimos.utils.constants import *
from deimos.models.liv.sme import get_sme_state_matrix
from deimos.models.liv.paper_plots.paper_def import *
import argparse
import concurrent.futures

if __name__ == "__main__":

    print("Script started at ", time_module.asctime())
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--solver", type=str, required=False, default="nusquids", help="Solver name")
    parser.add_argument("-n", "--num-points", type=int, required=False, default=25, help="Num scan points")
    parser.add_argument("-m", "--matter", type=str, required=False, default="earth", help="Matter effects: vacuum or earth")
    args = parser.parse_args()

    initial_flavor = 1
    E_GeV = REF_E_GeV

    kw = {}
    if args.solver == "nusquids":
        kw["energy_nodes_GeV"] = E_GeV
        kw["nusquids_variant"] = "sme"

    matter = args.matter

    sme_basis = REF_SME_BASIS
    a_magnitude_eV = 0
    a_mu_eV = get_sme_state_matrix(p33=a_magnitude_eV)
    c_magnitude = REF_SME_c_MAGNITUDE*2
    c_t_nu = get_sme_state_matrix(p33=c_magnitude)
    direction = "y"
    sme_params = { "basis":sme_basis, ("a_%s_eV"%direction):a_mu_eV, ("c_t%s"%direction):c_t_nu }
    a_label = r"$a^{%s}_{33}$ = %s eV" % (direction, get_number_tex(a_magnitude_eV))
    c_label = r"$c^{t%s}_{33}$ = %s" % (direction, get_number_tex(c_magnitude))

    ra_values_deg = np.linspace(0.0, 360.0, num=args.num_points)
    dec_values_deg = np.linspace(-90.0, 90.0, num=args.num_points+1)
    ra_values_rad = np.deg2rad(ra_values_deg)
    dec_values_rad = np.deg2rad(dec_values_deg)
    ra_grid_rad, dec_grid_rad = np.meshgrid(ra_values_rad, dec_values_rad, indexing="ij")
    grid_shape = ra_grid_rad.shape
    ra_grid_flat_rad, dec_grid_flat_rad = ra_grid_rad.flatten(), dec_grid_rad.flatten()
    time = REF_TIME

    azimuth = np.deg2rad(np.linspace(0, 360, 1000))

    # --- Worker function ---
    def calc_detector(args):
        ini_time = time_module.time()
        detector_name, initial_flavor, nubar, E_GeV, sme_params, ra_grid_flat_rad, dec_grid_flat_rad, time, solver, kw, matter = args
        print(f"Starting calculation for detector {detector_name} nubar={nubar}")
        calculator = OscCalculator(solver=solver, atmospheric=True, **kw)
        calculator.set_matter(matter)
        calculator.set_detector(detector_name)
        calc_kw = {
            "initial_flavor": initial_flavor,
            "nubar": nubar,
            "energy_GeV": E_GeV,
            "ra_rad": ra_grid_flat_rad,
            "dec_rad": dec_grid_flat_rad,
            "time": time,
            "sme_params": sme_params,
        }
        P, _, _ = calculator.calc_osc_prob_sme_directional_atmospheric(**calc_kw)
        # Calculate horizon and boundaries
        RA_horizon, DEC_horizon = calculator.detector_coords.get_right_ascension_and_declination(0, azimuth, time)
        pathlength_inner_core = 2 * np.sqrt(EARTH_RADIUS_km**2 - EARTH_INNER_CORE_RADIUS_km**2)
        cosz_inner_core = get_coszen_from_path_length(pathlength_inner_core)
        RA_core, DEC_core = calculator.detector_coords.get_right_ascension_and_declination(cosz_inner_core, azimuth, time)
        pathlength_outer_core = 2 * np.sqrt(EARTH_RADIUS_km**2 - EARTH_OUTER_CORE_RADIUS_km**2)
        cosz_outer_core = get_coszen_from_path_length(pathlength_outer_core)
        RA_outer_core, DEC_outer_core = calculator.detector_coords.get_right_ascension_and_declination(cosz_outer_core, azimuth, time)
        pathlength_mantle = 2 * np.sqrt(EARTH_RADIUS_km**2 - EARTH_MANTLE_RADIUS_km**2)
        cosz_mantle = get_coszen_from_path_length(pathlength_mantle)
        RA_mantle, DEC_mantle = calculator.detector_coords.get_right_ascension_and_declination(cosz_mantle, azimuth, time)
        elapsed = time_module.time() - ini_time
        minutes, seconds = divmod(elapsed, 60)
        print(f"Calculation done for detector {detector_name}, nubar={nubar} in {int(minutes)}m {int(seconds)}s")
        return (P, RA_horizon, DEC_horizon, RA_core, DEC_core, RA_outer_core, DEC_outer_core, RA_mantle, DEC_mantle)

    # Run for nubar=False and nubar=True
    args_list = [
        ("icecube", initial_flavor, False, E_GeV, sme_params, ra_grid_flat_rad, dec_grid_flat_rad, time, args.solver, kw, matter),
        ("icecube", initial_flavor, True, E_GeV, sme_params, ra_grid_flat_rad, dec_grid_flat_rad, time, args.solver, kw, matter)
    ]

    t_init = time_module.time()
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(executor.map(calc_detector, args_list))

    print(f"Total calculation time: {(time_module.time() - t_init) / 60.0:.2f} minutes")

    # --- Plotting ---
    linewidth = 2
    alpha = 1
    fig, ax = plt.subplots(1, 2, figsize=(7, 3), sharex=True, sharey=True)
    # fig.suptitle(fr"ARCA: $E$ = {E_GeV*1e-3:.3g} TeV // Time: {time} // SME: {a_label},  {c_label} // Matter: {matter.title()}", fontsize=14)

    nubar_labels = [r"$\nu_e$", r"$\bar{\nu}_e$"]
    for i, (P, RA_horizon, DEC_horizon, RA_core, DEC_core, RA_outer_core, DEC_outer_core, RA_mantle, DEC_mantle) in enumerate(results):
        zlabel = r"$%s$" % OscCalculator(solver=args.solver, atmospheric=True, **kw).get_transition_prob_tex(initial_flavor, 2, bool(i))
        plot_colormap(ax=ax[i], x=ra_values_deg, y=dec_values_deg, z=P[...,0].reshape(grid_shape), zlabel=zlabel, cbar=False, cmap="RdPu", vmin=0., vmax=1.)
        ax[i].plot(np.rad2deg(RA_horizon), np.rad2deg(DEC_horizon), color="lime", alpha=alpha, lw=linewidth)
        ax[i].plot(np.rad2deg(RA_core), np.rad2deg(DEC_core), color="red", alpha=alpha, lw=linewidth)
        ax[i].plot(np.rad2deg(RA_outer_core), np.rad2deg(DEC_outer_core), color="orange", alpha=alpha, lw=linewidth)
        ax[i].plot(np.rad2deg(RA_mantle), np.rad2deg(DEC_mantle), color="yellow", alpha=alpha, lw=linewidth)
        ax[i].set_xticks([90, 180, 270])
        ax[i].set_xticklabels([ "%i"%t for t in [90, 180, 270] ], fontsize=14)
        ax[i].set_yticks([-90, -45, 0, 45, 90])
        ax[i].set_yticklabels([ "%i"%t for t in [-90, -45, 0, 45, 90] ], fontsize=14)
        ax[i].set_xlim(ra_values_deg[0], ra_values_deg[-1])
        ax[i].set_ylim(dec_values_deg[0], dec_values_deg[-1])
        ax[i].set_xlabel("RA [deg]", fontsize=14)
        # ax[i].text(0.04, 0.15, nubar_labels[i], transform=ax[i].transAxes, fontsize=15, color="black", verticalalignment='top')#, bbox=dict(boxstyle='round', facecolor='black', alpha=0.9))
        if i == 0:
            ax[i].set_ylabel("Declination [deg]", fontsize=14)

    # ax[0].text(0.04, 0.15, 'Neutrino: '+nubar_labels[0], transform=ax[0].transAxes, fontsize=15, color="black", verticalalignment='top')#, bbox=dict(boxstyle='round', facecolor='black', alpha=0.9))
    # ax[1].text(0.04, 0.15, 'Anti-neutrino: '+nubar_labels[1], transform=ax[1].transAxes, fontsize=15, color="black", verticalalignment='top')#, bbox=dict(boxstyle='round', facecolor='black', alpha=0.9))
    ax[0].set_title('Neutrino: '+nubar_labels[0], fontsize=14)
    ax[1].set_title('Anti-neutrino: '+nubar_labels[1], fontsize=14)

    pos = ax[0].get_position()
    cbar_ax = fig.add_axes([0.89, pos.y0, 0.02, pos.height])
    cbar = fig.colorbar(ax[0].collections[0], cax=cbar_ax, orientation="vertical", shrink=1.0, pad=0.02)
    cbar.set_label("Probability", fontsize=14)
    cbar.ax.tick_params(labelsize=14)

    linewidth = 4
    manuallist = [
        plt.Line2D([0], [0], color='lime', marker='None', markersize=10, markeredgewidth=0.0, linewidth=linewidth),
        plt.Line2D([0], [0], color='yellow', marker='None', linestyle='-', markersize=10, linewidth=linewidth),
        plt.Line2D([0], [0], color='orange', marker='None', markersize=10, markeredgewidth=0.0, linewidth=linewidth),
        plt.Line2D([0], [0], color='red', marker='None', linestyle='-', markersize=10, linewidth=linewidth)
    ]
    legend_handeles = manuallist
    legend_labels = ["Horizon", "Mantle", "Outer Core", "Inner Core"]

    fig.legend(
        legend_handeles, legend_labels,
        fontsize=13, loc='upper center', ncol=4, handlelength=0.6, fancybox=True,
        bbox_to_anchor=(0.5, 1.14)
    )

    fig.subplots_adjust(top=0.88, right=0.88, wspace=0.07)

    plt.savefig(__file__.replace(".py", "_" + args.solver + "_nubar.png"), bbox_inches='tight')
    print("Figure saved to " + __file__.replace(".py", "_" + args.solver + "_nubar.png"))
    print("Script finished at ", time_module.asctime())