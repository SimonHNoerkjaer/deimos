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
    parser.add_argument("-n", "--num-points", type=int, required=False, default=1000, help="Num scan points")
    args = parser.parse_args()


    #
    # Define basic system parameters
    #

    initial_flavor = 2  # 1 corresponds to numu
    final_flavor = 2
    nubar = False  # False for neutrino, True for antineutrino

    E_values_GeV = np.geomspace(1e2, 1e7, num=args.num_points)

    detector = "IceCube"
    ra_deg = 30.            ; ra_rad = np.deg2rad(ra_deg)
    dec_deg = +80.          ; dec_rad = np.deg2rad(dec_deg)
    coszen_values = np.array([np.cos(np.deg2rad(dec_deg))]) # IceCube is at the South Pole, so zenith angle is 0 for upgoing events
    print("coszen dimensions", coszen_values)

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



    sme_kw = {"sme_params":
              {"a_%s_eV"%liv_direction : a_mu_eV,
                "c_t%s"%liv_direction : c_t_nu,
                "basis":sme_basis}}
    
    sme_params = sme_kw["sme_params"]

    

    # Initialize oscillation calculators for IceCube and off-axis detectors
    calculator = OscCalculator(solver=args.solver, atmospheric=True, **kw)
    calculator.set_matter(matter)
    calculator.set_detector(detector)

    print("check1")

    get_neutrino_flux_kw = dict(grid=True, source='atmospheric', overwrite_cache=False)

    # For nuSQuIDS, must define the initital state vector as the flux at the E and coszen nodes of the nuSQuIDSAtm instance
    initial_state = calculator.get_neutrino_flux(energy_GeV=E_values_GeV, coszen=calculator.coszen_nodes, **get_neutrino_flux_kw)
    initial_state = initial_state[:, :, :, 0]   # Output flux shape is: [E, cz, flavor, nu/nubar]
    

    print("check2")
    # print("initial state", initial_state[0:1])
    # print("initial state shape", initial_state[:,1:2,:].shape)

    # initial_state = initial_state[0:1, 0:1, :, 0] # Take first energy and coszen node

    # # Propagate, (ab)using the calc_osc_prob function
    # final_flux = calculator.calc_osc_prob(
    #     energy_GeV=E_values_GeV,
    #     initial_state=initial_state,
    #     coszen=coszen_values,
    #     nubar=nubar,
    # )



    print("check3")

    # Reset oscillation calculators for IceCube and off-axis detectors
    calculator = OscCalculator(solver=args.solver, atmospheric=True, **kw)
    calculator.set_matter(matter)
    calculator.set_detector(detector)



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



    # flux_ratios = osc_probs_sme / osc_probs_std


    # plot osc_probs_std

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.suptitle(fr"{detector} // Matter: {matter.title()} // $\alpha,\delta$ = {int(ra_deg)},{int(dec_deg)} deg // {time}", fontsize=10)

    ax.plot(E_values_GeV, osc_probs_std[:,0], linestyle="-", color="orange", label=r"$\nu_e$")
    ax.plot(E_values_GeV, osc_probs_std[:,1], linestyle="--", color="dodgerblue", label=r"$\nu_\mu$")
    ax.plot(E_values_GeV, osc_probs_std[:,2], linestyle="-.", color="green", label=r"$\nu_\tau$")



#    # Create the figure and axis objects
#     fig, ax = plt.subplots(figsize=(6, 4))
#     fig.suptitle(fr"{detector} // Matter: {matter.title()} // $\alpha,\delta$ = {int(ra_deg)},{int(dec_deg)} deg // {time}", fontsize=10)
    
    


#     ax.plot(E_values_GeV, flux_ratios[:,0], linestyle="-", color="orange", label=r"$\nu_e$")
#     ax.plot(E_values_GeV, flux_ratios[:,1], linestyle="-", color="dodgerblue", label=r"$\nu_\mu$")
#     ax.plot(E_values_GeV, flux_ratios[:,2], linestyle="-", color="green", label=r"$\nu_\tau$")


#     # Format plot
#     ax.set_ylim(0., 10)
#     ax.set_xlabel(ENERGY_LABEL, fontsize=14)
#     ax.set_xscale("log")
#     ax.tick_params(labelsize=12)
#     ax.grid(True)
#     ax.legend(fontsize=12, loc="lower right")
#     fig.tight_layout()

    # Save the figure
    print("")
    dump_figures_to_pdf( __file__.replace(".py","_" + args.solver + ".pdf") )

    # Done