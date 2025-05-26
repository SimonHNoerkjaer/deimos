'''
Plot sideral SME energy-dependence
'''


import numpy as np
from deimos.wrapper.osc_calculator import OscCalculator
from deimos.utils.oscillations import get_coszen_from_path_length
from deimos.utils.plotting import plt, dump_figures_to_pdf, plot_colormap, get_number_tex
from deimos.utils.constants import *
from deimos.models.liv.sme import get_sme_state_matrix
from deimos.models.liv.paper_plots.paper_def import *
import collections

#
# Main 
#
if __name__ == "__main__":

    #
    # Steering
    #
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--solver", type=str, required=False, default="nusquids", help="Solver name")
    parser.add_argument("-n", "--num-points", type=int, required=False, default=100, help="Num scan points")
    args = parser.parse_args()


    #
    # Define basic system parameters
    #

    initial_flavor = 1  # 1 corresponds to numu
    final_flavors = (0,1,2)
    nubar = False  # False for neutrino, True for antineutrino

    E_values_GeV = np.geomspace(1e2, 1e7, num=args.num_points)

    detector = "IceCube"
    ra_deg = 30.
    dec_deg = +75. # Upgoing for IceCube

    time = REF_TIME

    matter = "earth" # "earth" or "vacuum"


    #
    # Set SME parameters
    #

    # Choose basis SME operators are defined in
    sme_basis = REF_SME_BASIS

    # Define "a" operator (magnitude and state texture)
    a_magnitude_eV = REF_SME_a_MAGNITUDE_eV
    a_mu_eV = get_sme_state_matrix(p33=a_magnitude_eV) # Choosing 33 element as only non-zero element in germs of flavor

    # Define "c" operator (magnitude and state texture)
    c_magnitude = 0.#REF_SME_c_MAGNITUDE
    c_t_nu = get_sme_state_matrix(p33=c_magnitude) # Choosing 33 element as only non-zero element in germs of flavor

    # Choose direction (sticking to axis directions for simplicity here)
    liv_direction = "y" #  x y z


    #
    # Create solver
    #

    kw = {}
    if args.solver == "nusquids":
        kw["energy_nodes_GeV"] = E_values_GeV
        kw["nusquids_variant"] = "sme"

    # Initialize oscillation calculators for IceCube and off-axis detectors
    calculator = OscCalculator(solver=args.solver, atmospheric=True, **kw)

    # Set matter effects and detectors
    calculator.set_matter(matter)
    calculator.set_detector(detector)



    # Create the figure and axis objects
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.suptitle(fr"{detector} // Matter: {matter.title()} // $\alpha,\delta$ = {int(ra_deg)},{int(dec_deg)} deg // {time}", fontsize=10)

    linestyles = ["-", "--", "-.", ":"]
    
    for i, final_flavor in enumerate(final_flavors):
        if final_flavor == 0: label = r"$\nu_e$"
        elif final_flavor == 1: label = r"$\nu_\mu$"
        elif final_flavor == 2: label = r"$\nu_\tau$"

        # Calculate oscillation probabilities
        sme_kw = {"sme_params":
                {"a_%s_eV"%liv_direction : a_mu_eV,
                    "c_t%s"%liv_direction : c_t_nu,
                    "basis":sme_basis}}
        
        osc_probs_sme, coszen_values, azimuth_values = calculator.calc_osc_prob_sme_directional_atmospheric(
            initial_flavor=initial_flavor,
            nubar=nubar,
            energy_GeV=E_values_GeV,
            ra_rad=np.deg2rad(ra_deg),
            dec_rad=np.deg2rad(dec_deg),
            time=time,
            **sme_kw
        )

        # STD oscillation probabilities
        std_kw = {"sme_params":
                {"a_%s_eV"%liv_direction : np.zeros_like(a_mu_eV),
                    "c_t%s"%liv_direction : np.zeros_like(c_t_nu),
                    "basis":sme_basis}}
        
        osc_probs_std, coszen_values, azimuth_values_std = calculator.calc_osc_prob_sme_directional_atmospheric(
            initial_flavor=initial_flavor,
            nubar=nubar,
            energy_GeV=E_values_GeV,
            ra_rad=np.deg2rad(ra_deg),
            dec_rad=np.deg2rad(dec_deg),
            time=time,
            **std_kw
        )
        
        flux_ratios = osc_probs_sme[:,final_flavor] / osc_probs_std[:,final_flavor]

        ax.plot(E_values_GeV, flux_ratios, linestyle=linestyles[i], label=label)
    # ax.plot(E_values_GeV, flux_ratios[:,1], linestyle="-", color="dodgerblue", label=r"$\nu_\mu$")
    # ax.plot(E_values_GeV, flux_ratios[:,2], linestyle="-", color="green", label=r"$\nu_\tau$")


    # Format plot
    ax.set_title(rf"Initial flavor: ${calculator.get_nu_flavor_tex(initial_flavor, nubar=nubar)}$", fontsize=12)
    ax.set_ylim(0., 100)
    ax.set_xlabel(ENERGY_LABEL, fontsize=14)
    ax.set_xscale("log")
    ax.tick_params(labelsize=12)
    ax.grid(True)
    ax.legend(fontsize=12, loc="lower right")
    fig.tight_layout()

    # Save the figure
    print("")
    dump_figures_to_pdf( __file__.replace(".py","_" + args.solver + ".pdf") )

    # Done