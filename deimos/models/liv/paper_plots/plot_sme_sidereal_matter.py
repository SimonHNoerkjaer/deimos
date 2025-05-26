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

#
# Main 
#
if __name__ == "__main__":

    #
    # Steering
    #
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--solver", type=str, required=False, default="deimos", help="Solver name")
    parser.add_argument("-n", "--num-points", type=int, required=False, default=25, help="Num scan points")
    parser.add_argument("-m", "--matter", type=str, required=False, default="vacuum", help="Matter effects: vacuum or earth")
    args = parser.parse_args()


    #
    # Define basic system parameters
    #
    initial_flavor = 1  # 1 corresponds to numu
    nubar = False  # False for neutrino, True for antineutrino
    E_GeV = REF_E_GeV

    #
    # Solver settings
    #
    kw = {}
    if args.solver == "nusquids":
        kw["energy_nodes_GeV"] = E_GeV
        kw["nusquids_variant"] = "sme"

    # Initialize oscillation calculators for IceCube and off-axis detectors
    IC_calculator = OscCalculator(solver=args.solver, atmospheric=True, **kw)
    ARCA_calculator = OscCalculator(solver=args.solver, atmospheric=True, **kw)

    # Set matter effects and detectors
    matter = args.matter # "earth"(only nuSQuIDS) or "vacuum"
    IC_calculator.set_matter(matter)
    IC_calculator.set_detector("icecube")
    ARCA_calculator.set_matter(matter)
    ARCA_calculator.set_detector("arca")

    #
    # Set SME parameters
    #

    # Choose basis SME operators are defined in
    sme_basis = REF_SME_BASIS

    # Define "a" operator (magnitude and state texture)
    # a_magnitude_eV = REF_SME_a_MAGNITUDE_eV
    a_magnitude_eV = 0
    a_mu_eV = get_sme_state_matrix(p33=a_magnitude_eV) # Choosing 33 element as only non-zero element in germs of flavor

    # Define "c" operator (magnitude and state texture)
    c_magnitude = REF_SME_c_MAGNITUDE
    # c_magnitude = 0
    c_t_nu = get_sme_state_matrix(p33=c_magnitude) # Choosing 33 element as only non-zero element in germs of flavor

    # Choose direction (sticking to axis directions for simplicity here)
    direction = "y" #  x y z
    sme_params = { "basis":sme_basis, ("a_%s_eV"%direction):a_mu_eV, ("c_t%s"%direction):c_t_nu}
    a_label = r"$a^{%s}_{33}$ = %s eV" % (direction, get_number_tex(a_magnitude_eV))
    c_label = r"$c^{t%s}_{33}$ = %s" % (direction, get_number_tex(c_magnitude))


    #
    # Define sky scan parameter space
    #

    ra_values_deg = np.linspace(0.0, 360.0, num=args.num_points)
    dec_values_deg = np.linspace(-90.0, 90.0, num=args.num_points+1)
    ra_values_rad = np.deg2rad(ra_values_deg)
    dec_values_rad = np.deg2rad(dec_values_deg)
    ra_grid_rad, dec_grid_rad = np.meshgrid(ra_values_rad, dec_values_rad, indexing="ij")
    grid_shape = ra_grid_rad.shape
    ra_grid_flat_rad, dec_grid_flat_rad = ra_grid_rad.flatten(), dec_grid_rad.flatten()
    time = REF_TIME


    #
    # Definition of Earth layer boundaries
    #

    azimuth = np.deg2rad(np.linspace(0, 360, 1000))
    IC_RA_horizon, IC_DEC_horizon = IC_calculator.detector_coords.get_right_ascension_and_declination(0, azimuth, time)
    ARCA_RA_horizon, ARCA_DEC_horizon = ARCA_calculator.detector_coords.get_right_ascension_and_declination(0, azimuth, time)

    # Order off-axis horizon arrays by RA
    ARCA_indices = np.argsort(ARCA_RA_horizon)
    ARCA_RA_horizon = ARCA_RA_horizon[ARCA_indices]
    ARCA_DEC_horizon = ARCA_DEC_horizon[ARCA_indices]

    # Earth core boundaries
    pathlength_inner_core = 2 * np.sqrt(EARTH_RADIUS_km**2 - EARTH_INNER_CORE_RADIUS_km**2)
    cosz_inner_core = get_coszen_from_path_length(pathlength_inner_core)
    IC_RA_core, IC_DEC_core = IC_calculator.detector_coords.get_right_ascension_and_declination(cosz_inner_core, azimuth, time)
    ARCA_RA_core, ARCA_DEC_core = ARCA_calculator.detector_coords.get_right_ascension_and_declination(cosz_inner_core, azimuth, time)

    pathlength_outer_core = 2 * np.sqrt(EARTH_RADIUS_km**2 - EARTH_OUTER_CORE_RADIUS_km**2)
    cosz_outer_core = get_coszen_from_path_length(pathlength_outer_core)
    IC_RA_outer_core, IC_DEC_outer_core = IC_calculator.detector_coords.get_right_ascension_and_declination(cosz_outer_core, azimuth, time)
    ARCA_RA_outer_core, ARCA_DEC_outer_core = ARCA_calculator.detector_coords.get_right_ascension_and_declination(cosz_outer_core, azimuth, time)

    pathlength_mantle = 2 * np.sqrt(EARTH_RADIUS_km**2 - EARTH_MANTLE_RADIUS_km**2)
    cosz_mantle = get_coszen_from_path_length(pathlength_mantle)
    IC_RA_mantle, IC_DEC_mantle = IC_calculator.detector_coords.get_right_ascension_and_declination(cosz_mantle, azimuth, time)
    ARCA_RA_mantle, ARCA_DEC_mantle = ARCA_calculator.detector_coords.get_right_ascension_and_declination(cosz_mantle, azimuth, time)

    ARCA_indices = np.argsort(ARCA_RA_mantle)
    ARCA_RA_mantle = ARCA_RA_mantle[ARCA_indices]
    ARCA_DEC_mantle = ARCA_DEC_mantle[ARCA_indices]


    #
    # Calculate oscillation probabilities
    #

    t_init = time_module.time()

    # Calc osc probs for both detectors
    calc_kw = {
        "initial_flavor": initial_flavor,
        "nubar": nubar,
        "energy_GeV": E_GeV,
        "ra_rad": ra_grid_flat_rad,
        "dec_rad": dec_grid_flat_rad,
        "time": time,
        "sme_params": sme_params,
    }
    P_IC, _, _ = IC_calculator.calc_osc_prob_sme_directional_atmospheric(**calc_kw)
    P_ARCA, _, _ = ARCA_calculator.calc_osc_prob_sme_directional_atmospheric(**calc_kw)

    print(f"Total calculation time: {(time_module.time() - t_init) / 60.0:.2f} minutes")

    #
    # Plotting
    #
    linewidth = 2
    alpha = 1

    # Create the figure and axis objects
    fig, ax = plt.subplots(3, 2, figsize=(9, 10), sharex=True, sharey=True)
    # fig.subplots_adjust(right=0.88, wspace=0.045, hspace=0.05)

    # Flatten the axis array for easy iteration
    ax = ax.flatten()

    # Title for the Plots
    fig.suptitle(fr"$E$ = {E_GeV*1e-3:.3g} TeV // Time: {time} // SME: {a_label},  {c_label} // Matter: {matter.title()}", fontsize=12)

    # Plot probability matrices for IceCube and Off-axis detectors
    probabilities = [P_IC, P_ARCA]
    detectors = ["IceCube", "ARCA"]
    for i in range(ARCA_calculator.num_neutrinos):
        for j in range(len(detectors)):
            idx = 2 * i + j
            # ax[idx].imshow(probabilities[j][...,i].reshape(grid_shape).T, origin="lower", extent=[ra_values_deg[0], ra_values_deg[-1], dec_values_deg[0], dec_values_deg[-1]], aspect="auto", cmap=colors[idx], vmin=0., vmax=1.)
            zlabel = r"$%s$" % ARCA_calculator.get_transition_prob_tex(initial_flavor, i, nubar)
            plot_colormap(ax=ax[idx], x=ra_values_deg, y=dec_values_deg, z=probabilities[j][...,i].reshape(grid_shape), zlabel=zlabel, cmap="RdPu", vmin=0., vmax=1.)

    # Plot overlays for horizon and core boundaries
    # Only if matter included, otherwise not relevent
    if matter == "earth" :
        horizons_RA = [IC_RA_horizon, ARCA_RA_horizon, IC_RA_core, ARCA_RA_core, IC_RA_outer_core, ARCA_RA_outer_core, IC_RA_mantle, ARCA_RA_mantle]
        horizons_DEC = [IC_DEC_horizon, ARCA_DEC_horizon, IC_DEC_core, ARCA_DEC_core, IC_DEC_outer_core, ARCA_DEC_outer_core, IC_DEC_mantle, ARCA_DEC_mantle]
        for i in range(ARCA_calculator.num_neutrinos):
            for j in range(len(detectors)):
                idx = 2 * i + j
                ax[idx].plot(np.rad2deg(horizons_RA)[j], np.rad2deg(horizons_DEC)[j], color="lime", alpha=alpha, lw=linewidth)
                ax[idx].plot(np.rad2deg(horizons_RA)[j + 2], np.rad2deg(horizons_DEC)[j + 2], color="red", alpha=alpha, lw=linewidth)
                ax[idx].plot(np.rad2deg(horizons_RA)[j + 4], np.rad2deg(horizons_DEC)[j + 4], color="orange", alpha=alpha, lw=linewidth)
                ax[idx].plot(np.rad2deg(horizons_RA)[j + 6], np.rad2deg(horizons_DEC)[j + 6], color="yellow", alpha=alpha, lw=linewidth)

    # Mark LIV field direction   #TODO dynamic
    # ax[idx].plot(90, 0, markerfacecolor="gold", markeredgecolor="black", marker="D", markersize=7, linestyle="None")

    # Axis labels and ticks
    ra_ticks = [0, 90, 180, 270, 360]
    dec_ticks = [-90, -45, 0, 45, 90]
    for i in range(ARCA_calculator.num_neutrinos):
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

    # Legend
    if matter == "earth" :
        ax[0].plot([], [], color="lime", alpha=alpha, lw=linewidth, label="Horizon")
        ax[0].plot([], [], color="yellow", alpha=alpha, lw=linewidth, label="Mantle")
        ax[0].plot([], [], color="orange", alpha=alpha, lw=linewidth, linestyle=None, label="Outer core")
        ax[0].plot([], [], color="red", alpha=alpha, lw=linewidth, linestyle=None, label="Inner core")
        fig.legend(loc="upper center", fontsize=12, ncol=5, bbox_to_anchor=(0.5, 0.93))
 
    # Save the figure

    print("")
    dump_figures_to_pdf( __file__.replace(".py","_" + args.solver + "_" + args.matter + "_c" +".pdf"))

    # Done