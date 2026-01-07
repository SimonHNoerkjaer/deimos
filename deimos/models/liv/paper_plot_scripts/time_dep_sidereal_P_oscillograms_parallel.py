'''
Calculate and plot time-dependent sidereal LIV oscillograms including matter effects in parallel using the SME nuSQuIDS atmospheric neutrino oscillation probabilities.

Script by Simon Hilding-Nørkjær
'''

import time as time_module
import numpy as np
from astropy.time import Time
from deimos.wrapper.osc_calculator import OscCalculator
from deimos.utils.plotting import plt
from deimos.utils.oscillations import get_coszen_from_path_length
from deimos.utils.coordinates import *
from deimos.utils.constants import *
from deimos.models.liv.sme import get_sme_state_matrix
import argparse
import concurrent.futures

if __name__ == "__main__":

    # --- Steering ---
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--solver", type=str, required=False, default="nusquids", help="Solver name")
    parser.add_argument("-n", "--num-points", type=int, required=False, default=25, help="Num scan point")
    args = parser.parse_args()

    # --- System ---
    off_axis_detector = "ARCA"
    initial_flavor = 1
    nubar = False
    E_GeV = 10000.
    atmospheric = True
    sme_basis = "mass"
    a_magnitude_eV = 0
    c_magnitude = 1e-26

    solver = args.solver
    kw = {}
    if solver == "nusquids":
        kw["energy_nodes_GeV"] = E_GeV
        kw["nusquids_variant"] = "sme"

    # SME parameters
    a_y_eV = get_sme_state_matrix(p33=a_magnitude_eV)
    c_t_nu = get_sme_state_matrix(p33=c_magnitude)
    sme_params = { "basis":sme_basis, "a_y_eV":a_y_eV, "c_ty":c_t_nu }
    direction_structure = np.array([0., 1., 0.])
    direction_dec = np.array([0,90,0]) @ direction_structure

    # --- Earth Layer Boundaries ---
    ra_values_deg = np.linspace(0., 360., num=args.num_points)
    dec_values_deg = np.linspace(-90., 90., num=args.num_points)
    ra_values_rad = np.deg2rad(ra_values_deg)
    dec_values_rad = np.deg2rad(dec_values_deg)
    time = "July 16, 1999, 22:30"
    daytimes = ["July 16, 1999, 04:00", "July 16, 1999, 10:00", "July 16, 1999, 16:00", "July 16, 1999, 22:00"]

    azimuth = np.deg2rad(np.linspace(0, 360, 1000))

    # --- Output arrays ---
    P_shape = (3, len(dec_values_deg), len(ra_values_deg), len(daytimes))
    P_Off_axis = np.zeros(P_shape)
    cosz_Off_axis = np.zeros((len(dec_values_deg), len(ra_values_deg), len(daytimes)))

    Off_axis_DEC_horizon = np.zeros((len(azimuth), len(daytimes)))
    Off_axis_RA_horizon = np.zeros((len(azimuth), len(daytimes)))
    Off_axis_DEC_core = np.zeros((len(azimuth), len(daytimes)))
    Off_axis_RA_core = np.zeros((len(azimuth), len(daytimes)))
    Off_axis_DEC_outer_core = np.zeros((len(azimuth), len(daytimes)))
    Off_axis_RA_outer_core = np.zeros((len(azimuth), len(daytimes)))
    Off_axis_DEC_mantle = np.zeros((len(azimuth), len(daytimes)))
    Off_axis_RA_mantle = np.zeros((len(azimuth), len(daytimes)))

    # --- Worker function: create calculator inside ---
    def calc_point(args):
        i, j, dec_rad, ra_rad, time, initial_flavor, nubar, E_GeV, sme_params, solver, atmospheric, detector, kw = args
        calculator = OscCalculator(solver=solver, atmospheric=atmospheric, **kw)
        calculator.set_matter("vacuum")
        calculator.set_detector(detector)
        calc_kw = {
            "initial_flavor": initial_flavor,
            "nubar": nubar,
            "energy_GeV": E_GeV,
            "ra_rad": ra_rad,
            "dec_rad": dec_rad,
            "time": time,
            "sme_params": sme_params,
        }
        P_Off_axis_results, coszen_values_Off_axis, _ = calculator.calc_osc_prob_sme_directional_atmospheric(**calc_kw)
        return (i, j, P_Off_axis_results, coszen_values_Off_axis)

    t_init = time_module.time()

    for time_index, time in enumerate(daytimes):
        # --- Detector horizon and boundaries ---
        calculator = OscCalculator(solver=solver, atmospheric=atmospheric, **kw)
        calculator.set_matter("vacuum")
        calculator.set_detector(off_axis_detector)

        Off_axis_RA_horizon[:, time_index], Off_axis_DEC_horizon[:, time_index] = calculator.detector_coords.get_right_ascension_and_declination(0, azimuth, time)
        Off_axis_indices = np.argsort(Off_axis_RA_horizon[:, time_index])
        Off_axis_RA_horizon[:, time_index] = Off_axis_RA_horizon[Off_axis_indices, time_index]
        Off_axis_DEC_horizon[:, time_index] = Off_axis_DEC_horizon[Off_axis_indices, time_index]

        pathlength_inner_core = 2 * np.sqrt(EARTH_RADIUS_km**2 - EARTH_INNER_CORE_RADIUS_km**2)
        cosz_inner_core = get_coszen_from_path_length(pathlength_inner_core)
        Off_axis_RA_core[:, time_index], Off_axis_DEC_core[:, time_index] = calculator.detector_coords.get_right_ascension_and_declination(cosz_inner_core, azimuth, time)

        pathlength_outer_core = 2 * np.sqrt(EARTH_RADIUS_km**2 - EARTH_OUTER_CORE_RADIUS_km**2)
        cosz_outer_core = get_coszen_from_path_length(pathlength_outer_core)
        Off_axis_RA_outer_core[:, time_index], Off_axis_DEC_outer_core[:, time_index] = calculator.detector_coords.get_right_ascension_and_declination(cosz_outer_core, azimuth, time)

        pathlength_mantle = 2 * np.sqrt(EARTH_RADIUS_km**2 - EARTH_MANTLE_RADIUS_km**2)
        cosz_mantle = get_coszen_from_path_length(pathlength_mantle)
        Off_axis_RA_mantle[:, time_index], Off_axis_DEC_mantle[:, time_index] = calculator.detector_coords.get_right_ascension_and_declination(cosz_mantle, azimuth, time)
        Off_axis_indices = np.argsort(Off_axis_RA_mantle[:, time_index])
        Off_axis_RA_mantle[:, time_index] = Off_axis_RA_mantle[Off_axis_indices, time_index]
        Off_axis_DEC_mantle[:, time_index] = Off_axis_DEC_mantle[Off_axis_indices, time_index]

        # --- Prepare args for parallel calculation ---
        args_list = [
            (i, j, dec_rad, ra_rad, time, initial_flavor, nubar, E_GeV, sme_params, solver, atmospheric, off_axis_detector, kw)
            for i, dec_rad in enumerate(dec_values_rad)
            for j, ra_rad in enumerate(ra_values_rad)
        ]

        # --- Parallel calculation ---
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for result in executor.map(calc_point, args_list):
                i, j, P_Off_axis_results, coszen_values_Off_axis = result
                P_Off_axis[0, i, j, time_index] = P_Off_axis_results[0]
                P_Off_axis[1, i, j, time_index] = P_Off_axis_results[1]
                P_Off_axis[2, i, j, time_index] = P_Off_axis_results[2]
                cosz_Off_axis[i, j, time_index] = coszen_values_Off_axis
        print(f"Completed time index {time_index+1} / {len(daytimes)}")

    elapsed = time_module.time() - t_init
    minutes, seconds = divmod(elapsed, 60)
    print(f"Total calculation time: {int(minutes)} min {int(seconds)} sec")

    # --- Convert RA/DEC to degrees ---
    Off_axis_RA_horizon = np.rad2deg(Off_axis_RA_horizon)
    Off_axis_DEC_horizon = np.rad2deg(Off_axis_DEC_horizon)
    Off_axis_RA_core = np.rad2deg(Off_axis_RA_core)
    Off_axis_DEC_core = np.rad2deg(Off_axis_DEC_core)
    Off_axis_RA_outer_core = np.rad2deg(Off_axis_RA_outer_core)
    Off_axis_DEC_outer_core = np.rad2deg(Off_axis_DEC_outer_core)
    Off_axis_RA_mantle = np.rad2deg(Off_axis_RA_mantle)
    Off_axis_DEC_mantle = np.rad2deg(Off_axis_DEC_mantle)

    # --- Plotting ---
    linewidth = 1.2
    marker = "o"
    markersize = 1
    alpha = 1

    fig, ax = plt.subplots(2, 2, figsize=(9, 7), sharex=True, sharey=True)
    ax = ax.flatten()
    fig.suptitle(r"$E$ = %0.3g GeV // SME: a_eV_y=%0.3g // c=%0.3g" % (E_GeV, a_magnitude_eV, c_magnitude), fontsize=12)

    for time_index, time in enumerate(daytimes):
        ax[time_index].imshow(P_Off_axis[1, :, :, time_index], origin="lower",
                             extent=[ra_values_deg[0], ra_values_deg[-1], dec_values_deg[0], dec_values_deg[-1]],
                             aspect="auto", cmap="RdPu", vmin=0., vmax=1.)
        ax[time_index].plot(Off_axis_RA_horizon[:, time_index], Off_axis_DEC_horizon[:, time_index], color="lime", alpha=alpha, marker=marker, ms=markersize, linestyle="None")
        ax[time_index].plot(Off_axis_RA_core[:, time_index], Off_axis_DEC_core[:, time_index], color="red", alpha=alpha, marker=marker, ms=markersize, linestyle="None")
        ax[time_index].plot(Off_axis_RA_outer_core[:, time_index], Off_axis_DEC_outer_core[:, time_index], color="orange", alpha=alpha, marker=marker, ms=markersize, linestyle="None")
        ax[time_index].plot(Off_axis_RA_mantle[:, time_index], Off_axis_DEC_mantle[:, time_index], color="yellow", alpha=alpha, marker=marker, ms=markersize, linestyle="None")
        ax[time_index].text(0.05, 0.95, time.replace("July 16, 1999, ", "") + " UTC", transform=ax[time_index].transAxes, fontsize=14, color="white", verticalalignment='top', bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))
        ax[time_index].tick_params(axis='both', which='major', labelsize=14)

    ax[3].imshow(P_Off_axis[1, :, :, 3], origin="lower", extent=[ra_values_deg[0], ra_values_deg[-1], dec_values_deg[0], dec_values_deg[-1]], aspect="auto", cmap="RdPu", vmin=0., vmax=1.)
    ax[3].plot(Off_axis_RA_horizon[:, 3], Off_axis_DEC_horizon[:, 3], color="lime", alpha=alpha, marker=marker, ms=markersize, linestyle="None", label="Horizon")
    ax[3].plot(Off_axis_RA_core[:, 3], Off_axis_DEC_core[:, 3], color="red", alpha=alpha, marker=marker, ms=markersize, linestyle="None", label="Earth inner core")
    ax[3].plot(Off_axis_RA_outer_core[:, 3], Off_axis_DEC_outer_core[:, 3], color="orange", alpha=alpha, marker=marker, ms=markersize, linestyle="None", label="Earth outer core")
    ax[3].plot(Off_axis_RA_mantle[:, 3], Off_axis_DEC_mantle[:, 3], color="yellow", alpha=alpha, marker=marker, ms=markersize, linestyle="None", label="Earth mantle")
    ax[3].tick_params(axis='both', which='major', labelsize=14)
    ax[2].set_xticks([0, 90, 180, 270, 360])
    ax[3].set_xticks([0, 90, 180, 270, 360])
    ax[2].set_xticklabels([" 0", 90, 180, 270, "360   "])
    ax[3].set_xticklabels([" 0", 90, 180, 270, "360   "])
    ax[0].set_yticks([-90, -45, 0, 45, 90])
    ax[1].set_yticks([-90, -45, 0, 45, 90])
    ax[0].set_yticklabels(["-90", "-45", " 0", "45", "\n90"])
    ax[2].set_xlabel("RA[deg]", fontsize=15)
    ax[3].set_xlabel("RA[deg]", fontsize=15)
    ax[0].set_ylabel("DEC[deg]", fontsize=15)
    ax[2].set_ylabel("DEC[deg]", fontsize=15)
    ax[3].text(0.05, 0.95, time.replace("July 16, 1999, ", "") + " UTC", transform=ax[3].transAxes, fontsize=14, color="white", verticalalignment='top', bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))

    # --- Colorbar and legend ---
    fig.subplots_adjust(right=0.88, wspace=0.07, hspace=0.07)
    cbar_ax = fig.add_axes([0.90, 0.107, 0.025, 0.775])
    cbar = fig.colorbar(ax[1].images[0], cax=cbar_ax, orientation="vertical", fraction=0.05, pad=0.05)
    cbar.set_label(r"$P(\nu_{\mu} \to \nu_{\mu})$", fontsize=15)
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
    lgnd = ax[3].legend(legend_handeles, legend_labels, fontsize=13, loc=(-0.85, 2.10), ncol=4, handlelength=0.6, fancybox=True)

    plt.savefig(__file__.replace(".py", "_STD10gev_" + solver + ".png"), bbox_extra_artists=[cbar_ax, lgnd])
    print("")