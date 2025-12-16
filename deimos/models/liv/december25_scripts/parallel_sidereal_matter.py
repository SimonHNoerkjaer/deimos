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
from deimos.utils.plotting import plt, dump_figures_to_pdf, plot_colormap, get_number_tex
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
    nubar = False
    E_GeV = REF_E_GeV

    kw = {}
    if args.solver == "nusquids":
        kw["energy_nodes_GeV"] = E_GeV
        kw["nusquids_variant"] = "sme"

    matter = args.matter

    sme_basis = REF_SME_BASIS
    a_magnitude_eV = 0# 1e-13
    a_mu_eV = get_sme_state_matrix(p33=a_magnitude_eV)
    c_magnitude =  REF_SME_c_MAGNITUDE/2
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
        print(f"Starting calculation for detector {detector_name} {initial_flavor}")
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
        print(f"Calculation done for detector {detector_name}, {initial_flavor} in {int(minutes)}m {int(seconds)}s")
        return (detector_name, P, RA_horizon, DEC_horizon, RA_core, DEC_core, RA_outer_core, DEC_outer_core, RA_mantle, DEC_mantle)

    detectors = ["icecube", "arca"]
    args_list = [
        (detector, initial_flavor, nubar, E_GeV, sme_params, ra_grid_flat_rad, dec_grid_flat_rad, time, args.solver, kw, matter)
        for detector in detectors
    ]

    t_init = time_module.time()
    results = {}
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for result in executor.map(calc_detector, args_list):
            detector_name, P, RA_horizon, DEC_horizon, RA_core, DEC_core, RA_outer_core, DEC_outer_core, RA_mantle, DEC_mantle = result

            # --- SORT horizon and mantle arrays by RA ---
            horizon_indices = np.argsort(RA_horizon)
            RA_horizon = RA_horizon[horizon_indices]
            DEC_horizon = DEC_horizon[horizon_indices]

            mantle_indices = np.argsort(RA_mantle)
            RA_mantle = RA_mantle[mantle_indices]
            DEC_mantle = DEC_mantle[mantle_indices]

            results[detector_name] = {
                "P": P,
                "RA_horizon": RA_horizon,
                "DEC_horizon": DEC_horizon,
                "RA_core": RA_core,
                "DEC_core": DEC_core,
                "RA_outer_core": RA_outer_core,
                "DEC_outer_core": DEC_outer_core,
                "RA_mantle": RA_mantle,
                "DEC_mantle": DEC_mantle,
            }
    print(f"Total calculation time: {(time_module.time() - t_init) / 60.0:.2f} minutes")

    # --- Plotting ---
    linewidth = 2
    alpha = 1
    fig, ax = plt.subplots(3, 2, figsize=(9, 10), sharex=True, sharey=True)
    ax = ax.flatten()
    fig.suptitle(fr"$E$ = {E_GeV*1e-3:.3g} TeV // Time: {time} // SME: {a_label},  {c_label} // Matter: {matter.title()}", fontsize=12)

    probabilities = [results["icecube"]["P"], results["arca"]["P"]]
    horizons_RA = [results["icecube"]["RA_horizon"], results["arca"]["RA_horizon"], results["icecube"]["RA_core"], results["arca"]["RA_core"], results["icecube"]["RA_outer_core"], results["arca"]["RA_outer_core"], results["icecube"]["RA_mantle"], results["arca"]["RA_mantle"]]
    horizons_DEC = [results["icecube"]["DEC_horizon"], results["arca"]["DEC_horizon"], results["icecube"]["DEC_core"], results["arca"]["DEC_core"], results["icecube"]["DEC_outer_core"], results["arca"]["DEC_outer_core"], results["icecube"]["DEC_mantle"], results["arca"]["DEC_mantle"]]
    detectors = ["IceCube", "ARCA"]

    for i in range(probabilities[1].shape[-1]):
        for j in range(len(detectors)):
            idx = 2 * i + j
            zlabel = r"$%s$" % OscCalculator(solver=args.solver, atmospheric=True, **kw).get_transition_prob_tex(initial_flavor, i, nubar)
            plot_colormap(ax=ax[idx], x=ra_values_deg, y=dec_values_deg, z=probabilities[j][...,i].reshape(grid_shape), zlabel=zlabel, cmap="RdPu", vmin=0., vmax=1.)

    if matter == "earth":
        for i in range(probabilities[1].shape[-1]):
            for j in range(len(detectors)):
                idx = 2 * i + j
                ax[idx].plot(np.rad2deg(horizons_RA)[j], np.rad2deg(horizons_DEC)[j], color="lime", alpha=alpha, lw=linewidth)
                ax[idx].plot(np.rad2deg(horizons_RA)[j + 2], np.rad2deg(horizons_DEC)[j + 2], color="red", alpha=alpha, lw=linewidth)
                ax[idx].plot(np.rad2deg(horizons_RA)[j + 4], np.rad2deg(horizons_DEC)[j + 4], color="orange", alpha=alpha, lw=linewidth)
                ax[idx].plot(np.rad2deg(horizons_RA)[j + 6], np.rad2deg(horizons_DEC)[j + 6], color="yellow", alpha=alpha, lw=linewidth)

    ra_ticks = [0, 90, 180, 270, 360]
    dec_ticks = [-90, -45, 0, 45, 90]
    for i in range(probabilities[1].shape[-1]):
        for j in range(len(detectors)):
            idx = 2 * i + j
            ax[idx].set_xticks(ra_ticks)
            ax[idx].set_yticks(dec_ticks)
            ax[idx].set_yticklabels([ "%i"%t for t in dec_ticks ])
            ax[idx].tick_params(labelsize=14)
            if j == 0:
                ax[idx].text(0.04, 0.12, "IceCube", transform=ax[idx].transAxes, fontsize=14, color="white", verticalalignment='top', bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
                ax[idx].set_ylabel("Declination [deg]", fontsize=14)
            else:
                ax[idx].text(0.04, 0.95, "ARCA", transform=ax[idx].transAxes, fontsize=14, color="white", verticalalignment='top', bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
            ax[idx].set_xlim(ra_values_deg[0], ra_values_deg[-1])
            ax[idx].set_ylim(dec_values_deg[0], dec_values_deg[-1])
    ax[4].set_xlabel("RA [deg]", fontsize=14)
    ax[5].set_xlabel("RA [deg]", fontsize=14)
    ax[4].set_xticklabels([ "%i"%t for t in ra_ticks ])   
    ax[5].set_xticklabels([ "%i"%t for t in ra_ticks ])

    if matter == "earth":
        ax[0].plot([], [], color="lime", alpha=alpha, lw=linewidth, label="Horizon")
        ax[0].plot([], [], color="yellow", alpha=alpha, lw=linewidth, label="Mantle")
        ax[0].plot([], [], color="orange", alpha=alpha, lw=linewidth, linestyle=None, label="Outer core")
        ax[0].plot([], [], color="red", alpha=alpha, lw=linewidth, linestyle=None, label="Inner core")
        fig.legend(loc="upper center", fontsize=12, ncol=5, bbox_to_anchor=(0.5, 0.93))

    plt.savefig(__file__.replace(".py", "_" + args.solver + ".png"))
    print("Figure saved to " + __file__.replace(".py", "_" + args.solver + ".png"))
    print("Script finished at ", time_module.asctime())