'''
Script for producing 1D plots of sidereal time dependence of parameters in the SME model in RA/Dec coordinates,
for the case of atmospheric neutrinos

Edited by Johann Ioannou-Nikolaides based on a script by Simon Hilding-Nørkjær
'''

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from deimos.utils.plotting import *
from deimos.wrapper.osc_calculator import OscCalculator
from deimos.utils.oscillations import calc_path_length_from_coszen
from deimos.models.liv.sme import get_sme_state_matrix

def main():

    #
    # Steering
    #
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--solver", type=str, required=False, default="deimos", help="Solver name")
    parser.add_argument("-n", "--num-points", type=int, required=False, default=25, help="Num scan point")
    args = parser.parse_args()

    #
    # Define basic system parameters
    #
    initial_flavor, final_flavor, nubar = 1, 1, False    # numu survival
    E_GeV = 1e3

    #
    # Set SME parameters
    #

    # Choose basis SME operators are defined in
    sme_basis = "mass" # "mass" or "flavor"

    # Define a operator
    a_magnitude_eV = 5e-14
    a_y_eV = get_sme_state_matrix(p33=a_magnitude_eV) # Choosing 33 element as only non-zero element in germs of flavor, and +y direction 

    # Define c operator
    c_magnitude = 0
    c_ty = get_sme_state_matrix(p33=c_magnitude) # Choosing 33 element as only non-zero element in germs of flavor, and ty tensor component 

    # Group as an arg
    sme_params = { "basis":sme_basis, "a_y_eV":a_y_eV, "c_ty":c_ty}

    # Labels for label
    a_label = r"$a^y_{33}$ = %0.3g eV" % a_magnitude_eV
    c_label = r"$c^{yt}_{33}$ = %0.3g" % c_magnitude


    #
    # Create oscillation calculators
    #

    kw = {"energy_nodes_GeV": E_GeV, "nusquids_variant": "sme"} if args.solver == "nusquids" else {}
    calculatorIC = OscCalculator(solver=args.solver, atmospheric=True, **kw)
    calculatorARCA = OscCalculator(solver=args.solver, atmospheric=True, **kw)

    # Set detector and matter type
    matter = "vacuum" # vacuum or earth
    calculatorIC.set_matter(matter)
    calculatorIC.set_detector("icecube")
    calculatorARCA.set_matter(matter)
    calculatorARCA.set_detector("arca")


    #
    # Plot time-dependence
    #

    # Define time array
    hour_array = np.arange(1, 25, 2)
    time_array = [f"July 16, 1999, {time}:00" for time in hour_array]
    time_array = np.array(time_array)

    # Define coordinates
    dec_deg = np.linspace(-90, 90, num=args.num_points)
    ra_deg = np.full(dec_deg.size, 90)
    dec_rad = np.deg2rad(dec_deg)
    ra_rad = np.deg2rad(ra_deg)
    
    # Calc osc probs vs time, dec, etc
    calc_kw = {
        "initial_flavor": initial_flavor,
        "nubar": nubar,
        "energy_GeV": E_GeV,
        "ra_rad": ra_rad,
        "dec_rad": dec_rad,
        "time": time_array,
        "sme_params" : sme_params,
    }
    P_ARCA, cosz_ARCA, _ = calculatorARCA.calc_osc_prob_sme_directional_atmospheric(**calc_kw)
    P_IC, cosz_IC, _ = calculatorIC.calc_osc_prob_sme_directional_atmospheric(**calc_kw)

    # Choose final flavor
    P_ARCA = P_ARCA[...,final_flavor]
    P_IC = P_IC[...,final_flavor]

    # Convert cosz to baseline
    baseline_ARCA = calc_path_length_from_coszen(cosz_ARCA)
    baseline_IC = calc_path_length_from_coszen(cosz_IC)

    # Plot results
    colors = plt.cm.jet(hour_array / 24)
    plt.rcParams.update({'font.size': 14})

    fig, ax = plt.subplots(3, 1, figsize=(6, 14))

    # For ARCA, plot one "something vs declination" curve per hour
    for time_index, time in enumerate(time_array):
        label = "ARCA" if time_index == int(time_array.size/2) else None
        ax[0].plot(dec_deg, cosz_ARCA[:, time_index], color=colors[time_index], label=label)
        ax[1].plot(dec_deg, baseline_ARCA[:, time_index], color=colors[time_index], label=label)
        ax[2].plot(dec_deg, P_ARCA[:, time_index], color=colors[time_index], label=label)

    # For IceCube there is no time-dependence since it sits on the Earth's rotation axis, so only one curve required
    # First verify there is no time dependence, then plot the single curve...
    for time_index in range(1, len(time_array)):
        assert np.all( cosz_IC[:, time_index] - cosz_IC[:, 0] < 1e-3 ) # Adding some numerical tolerance since not exactly on rotation axis
        assert np.all( P_IC[:, time_index] - P_IC[:, 0] < 1e-3 )
    ax[0].plot(dec_deg, cosz_IC[:,0], "--k", lw=2, label="IceCube")
    ax[1].plot(dec_deg, baseline_IC[:,0], "--k", lw=2, label="IceCube")
    ax[2].plot(dec_deg, P_IC[:,0], "--k", lw=2, label="IceCube")

    # Customize plots
    ax[2].set(ylim=(-0.05, 1.05))
    for axis in ax:
        axis.legend()
        axis.set_xlabel("Declination [deg]", fontsize=16)
        axis.set_xticks([-90, -45, 0, 45, 90])
        axis.grid(alpha=0.2)
    ax[0].set_ylabel(r"$\cos(\theta_{Zenith})$", fontsize=16)
    ax[1].set_ylabel("Baseline [km]", fontsize=16)
    ax[2].set_ylabel(r"$P(\nu_\mu \to \nu_\mu)$", fontsize=16)

    # Add colorbars for time
    norm = mpl.colors.Normalize(vmin=0, vmax=24)
    cbar_axes = [fig.add_axes([0.92, 0.655 - 0.273 * i, 0.02, 0.226]) for i in range(3)]
    for cax in cbar_axes:
        cb = mpl.colorbar.ColorbarBase(cax, cmap=plt.cm.jet, norm=norm, orientation='vertical')
        cb.set_label('Time [h]')
        cb.set_ticks(np.arange(0, 25, 4))
        cb.set_ticklabels(np.arange(0, 25, 4))

    fig.suptitle(fr"SME: {a_label}, {c_label} // E = {E_GeV} GeV // Matter: {matter.title()}", fontsize=16)

    # Save the figure
    # plt.savefig(__file__.replace(".py", ".pdf"), bbox_inches='tight')
    # print("Dumped file to", __file__.replace(".py", ".pdf"))


    print("")
    dump_figures_to_pdf( __file__.replace(".py","_" + args.solver + ".pdf") )

if __name__ == "__main__":
    main()