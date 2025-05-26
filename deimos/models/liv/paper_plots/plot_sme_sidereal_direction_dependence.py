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
    args = parser.parse_args()


    #
    # Define basic system parameters
    #
    initial_flavor = 1  # 1 corresponds to numu
    final_flavor = 1
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
    matter = "vacuum" # "earth" or "vacuum"
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
    a_magnitude_eV = REF_SME_a_MAGNITUDE_eV
    a_mu_eV = get_sme_state_matrix(p33=a_magnitude_eV) # Choosing 33 element as only non-zero element in germs of flavor

    # Define "c" operator (magnitude and state texture)
    c_magnitude = 0.
    c_t_nu = get_sme_state_matrix(p33=c_magnitude) # Choosing 33 element as only non-zero element in germs of flavor


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
    # Loop over field direction
    #

    # Choose field directions
    field_directions = ["x", "y", "z"]

    # Create the figure and axis objects
    fig, ax = plt.subplots(len(field_directions), 2, figsize=(9, 10), sharex=True, sharey=True)
    # fig.subplots_adjust(right=0.88, wspace=0.045, hspace=0.05)

    # Titles
    ax[0,0].text(0.5, 1.1, "IceCube", transform=ax[0,0].transAxes, fontsize=14, color="white", va='bottom', ha="center", bbox=dict(boxstyle='round', facecolor='black'))
    ax[0,1].text(0.5, 1.1, "ARCA", transform=ax[0,1].transAxes, fontsize=14, color="white", va='bottom', ha="center", bbox=dict(boxstyle='round', facecolor='black'))

    # Title for the Plots
    fig.suptitle(fr"$E$ = {E_GeV*1e-3:.3g} TeV // Time: {time} // Matter: {matter.title()}", fontsize=12)

    # Loop over directions
    for i, field_direction in enumerate(field_directions) :

        # Configure SME params
        sme_params = { "basis":sme_basis, ("a_%s_eV"%field_direction):a_mu_eV, ("c_t%s"%field_direction):c_t_nu}
        a_label = r"$a^{%s}_{33}$ = %s eV" % (field_direction, get_number_tex(a_magnitude_eV))
        c_label = r"$c^{t%s}_{33}$ = %s" % (field_direction, get_number_tex(c_magnitude))

        # Labels
        sme_label = []
        if a_magnitude_eV > 0. :
            sme_label.append(a_label)
        if c_magnitude > 0. :
            sme_label.append(c_label)
        sme_label = r", ".join(sme_label)
        ax[i,0].text(0.04, 0.12, sme_label, transform=ax[i,0].transAxes, fontsize=10, color="white", verticalalignment='top', bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
        ax[i,1].text(0.04, 0.95, sme_label, transform=ax[i,1].transAxes, fontsize=10, color="white", verticalalignment='top', bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
        

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

        # Plot probability matrices for IceCube and Off-axis detectors
        probabilities = [P_IC, P_ARCA]
        detectors = ["IceCube", "ARCA"]
        for j in range(len(detectors)):
            zlabel = r"$%s$" % ARCA_calculator.get_transition_prob_tex(initial_flavor, final_flavor, nubar)
            plot_colormap(ax=ax[i,j], x=ra_values_deg, y=dec_values_deg, z=probabilities[j][...,final_flavor].reshape(grid_shape), zlabel=zlabel, cmap="RdPu", vmin=0., vmax=1.)

        # Plot overlays for horizon and core boundaries
        # Only if matter included, otherwise not relevent
        if matter == "earth" :
            ax[i,0].plot(np.rad2deg(IC_RA_horizon), np.rad2deg(IC_DEC_horizon), color="lime", alpha=alpha, lw=linewidth)
            ax[i,1].plot(np.rad2deg(ARCA_RA_horizon), np.rad2deg(ARCA_DEC_horizon), color="lime", alpha=alpha, lw=linewidth)
            ax[i,0].plot(np.rad2deg(IC_RA_core), np.rad2deg(IC_DEC_core), color="red", alpha=alpha, lw=linewidth)
            ax[i,1].plot(np.rad2deg(ARCA_RA_core), np.rad2deg(ARCA_DEC_core), color="red", alpha=alpha, lw=linewidth)
            ax[i,0].plot(np.rad2deg(IC_RA_outer_core), np.rad2deg(IC_DEC_outer_core), color="orange", alpha=alpha, lw=linewidth)
            ax[i,1].plot(np.rad2deg(ARCA_RA_outer_core), np.rad2deg(ARCA_DEC_outer_core), color="orange", alpha=alpha, lw=linewidth)
            ax[i,0].plot(np.rad2deg(IC_RA_mantle), np.rad2deg(IC_DEC_mantle), color="yellow", alpha=alpha, lw=linewidth)
            ax[i,1].plot(np.rad2deg(ARCA_RA_mantle), np.rad2deg(ARCA_DEC_mantle), color="yellow", alpha=alpha, lw=linewidth)

    # Mark LIV field direction   #TODO dynamic
    # ax[idx].plot(90, 0, markerfacecolor="gold", markeredgecolor="black", marker="D", markersize=7, linestyle="None")

    # Axis labels and ticks
    ra_ticks = [0, 90, 180, 270, 360]
    dec_ticks = [-90, -45, 0, 45, 90]
    for i in range(len(field_directions)):
        for j in range(len(detectors)):
            ax[i,j].set_xticks(ra_ticks)
            ax[i,j].set_yticks(dec_ticks)
            ax[i,j].set_yticklabels([ "%i"%t for t in dec_ticks ])
            ax[i,j].tick_params(labelsize=14)
            if j == 0:
                ax[i,j].set_ylabel("Declination [deg]", fontsize=14)
            ax[i,j].set_xlim(ra_values_deg[0], ra_values_deg[-1])
            ax[i,j].set_ylim(dec_values_deg[0], dec_values_deg[-1])
    ax[2,0].set_xlabel("RA [deg]", fontsize=14)
    ax[2,1].set_xlabel("RA [deg]", fontsize=14)
    ax[2,0].set_xticklabels([ "%i"%t for t in ra_ticks ])   
    ax[2,1].set_xticklabels([ "%i"%t for t in ra_ticks ])

    # Legend
    if matter == "earth" :
        ax[0,0].plot([], [], color="lime", alpha=alpha, lw=linewidth, label="Horizon")
        ax[0,0].plot([], [], color="yellow", alpha=alpha, lw=linewidth, label="Mantle")
        ax[0,0].plot([], [], color="orange", alpha=alpha, lw=linewidth, linestyle=None, label="Outer core")
        ax[0,0].plot([], [], color="red", alpha=alpha, lw=linewidth, linestyle=None, label="Inner core")
        fig.legend(loc="upper center", fontsize=12, ncol=5, bbox_to_anchor=(0.5, 0.93))


   # Save the figure
    print("")
    dump_figures_to_pdf( __file__.replace(".py","_" + args.solver + ".pdf") )

    # Done