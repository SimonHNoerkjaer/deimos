'''
Plot sideral SME energy-dependence

Script by Simon Hilding-Nørkjær
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
    final_flavor = 1
    nubar = False  # False for neutrino, True for antineutrino

    E_values_GeV = np.geomspace(1., 1e6, num=args.num_points)

    detector = "IceCube"
    ra_deg = 30.
    dec_deg = +80. # Upgoing for IceCube

    time = REF_TIME

    matter = "vacuum" # "earth" or "vacuum"


    #
    # Set SME parameters
    #

    # Choose basis SME operators are defined in
    sme_basis = REF_SME_BASIS

    # Define "a" operator (magnitude and state texture)
    a_magnitude_eV = REF_SME_a_MAGNITUDE_eV/2
    a_mu_eV = get_sme_state_matrix(p33=a_magnitude_eV) # Choosing 33 element as only non-zero element in germs of flavor

    # Define "c" operator (magnitude and state texture)
    c_magnitude = REF_SME_c_MAGNITUDE
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


    #
    # Loop over cases
    #

    # Define cases
    cases = collections.OrderedDict()
    cases[r"$a^{%s}_{33}$ = %s eV" % (liv_direction, get_number_tex(a_magnitude_eV))] = { "sme_params":{ "basis":sme_basis, ("a_%s_eV"%liv_direction):a_mu_eV}, "color":"orange" , 'ls':"-", 'lw':2}
    cases[r"$c^{%s}_{33}$ = %s" % (liv_direction, get_number_tex(c_magnitude))] = { "sme_params":{ "basis":sme_basis, ("c_t%s"%liv_direction):c_t_nu}, "color":"dodgerblue", 'ls':"-", 'lw':2}
    cases["No SME"] = { "sme_params":None, "color":"black" , 'ls':"--", 'lw':1}

    # Create the figure and axis objects
    fig, ax = plt.subplots(figsize=(6, 4))
    fig.suptitle(fr"{detector} // Matter: {matter.title()} // $\alpha,\delta$ = {int(ra_deg)},{int(dec_deg)} deg // {time}", fontsize=10)

    # Loop over cases
    for case_label, case in cases.items() :

        # Calculate oscillation probabilities
        sme_kw = {"std_osc":True} if case["sme_params"] is None else {"std_osc":False, "sme_params":case["sme_params"]}
        osc_probs, coszen_values, azimuth_values = calculator.calc_osc_prob_sme_directional_atmospheric(
            initial_flavor=initial_flavor,
            nubar=nubar,
            energy_GeV=E_values_GeV,
            ra_rad=np.deg2rad(ra_deg),
            dec_rad=np.deg2rad(dec_deg),
            time=time,
            **sme_kw
        )

        # Plot osc probs
        ax.plot(E_values_GeV, osc_probs[:,final_flavor], color=case["color"], label=case_label, lw=case['lw'], ls=case['ls'])

    # Format plot
    ax.set_xlabel(ENERGY_LABEL, fontsize=14)
    ax.set_xscale("log")
    ax.set_xlim(E_values_GeV[0], E_values_GeV[-1])
    ax.set_ylabel(r"$%s$"%calculator.get_transition_prob_tex(initial_flavor, final_flavor, nubar), fontsize=14)
    ax.set_ylim(-0.03, 1.03)
    ax.tick_params(labelsize=12)
    ax.grid(True)
    ax.legend(fontsize=12, loc="lower right")
    fig.tight_layout()

    # Save the figure
    print("")
    dump_figures_to_pdf( __file__.replace(".py","_" + args.solver + "a1c1.pdf") )

    # Done