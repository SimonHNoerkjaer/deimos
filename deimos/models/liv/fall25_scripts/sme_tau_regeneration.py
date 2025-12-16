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



    nubar = False  # False for neutrino, True for antineutrino

    # E_values_GeV = np.geomspace(1e2, 1e7, num=args.num_points)

    detector = "IceCube"
    ra_deg = 0.
    dec_deg = +90. # Upgoing for IceCube

    time = REF_TIME

    matter = "earth" # "earth" or "vacuum"


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
    # Load flux data from a file (assuming it's in a .npz format)
    flux_data = np.load('mceq_fluxes.npz')

    # Extract the energy grid and flux values
    E_grid = flux_data['energy_grid']
    flux_nu_e = flux_data['nue']
    flux_nu_mu = flux_data['numu']
    flux_nu_tau = flux_data['nutau']

    energy_mask = (E_grid >= 100.) & (E_grid <= 1000000.)  # 100 GeV to 1 PeV
    E_grid = E_grid[energy_mask]
    flux_nu_e = flux_nu_e[energy_mask]
    flux_nu_mu = flux_nu_mu[energy_mask]
    flux_nu_tau = flux_nu_tau[energy_mask]

    # E_grid = E_grid[0]
    # flux_nu_e = flux_nu_e[0]
    # flux_nu_mu = flux_nu_mu[0]
    # flux_nu_tau = flux_nu_tau[0]

    # Print the loaded flux data for verification
    print("Energy Grid:", E_grid)
    print("Flux Nu_e:", flux_nu_e)
    print("Flux Nu_mu:", flux_nu_mu)
    print("Flux Nu_tau:", flux_nu_tau)



    states = np.array([0,1,2])

    # loop over E grid
    # for E_i , E in enumerate(E_grid):
    #     print("E:", E)

    #  for each energy propagate the pure states (1,0,0), (0,1,0), (0,0,1)

    # initialize osc calculator outside loop
    state_osc_probs_std = np.zeros((len(E_grid), len(states), 3)) #E, initial state, final state
    state_osc_probs_sme = np.zeros((len(E_grid), len(states), 3)) #E, initial state, final state
    # state_osc_probs_std = np.zeros((len(states), 3)) #E, initial state, final state
    # state_osc_probs_sme = np.zeros((len(states), 3)) #E, initial state, final state


    for i, state in enumerate(states):
        print("state:", state)


        initial_flavor = state  # 1 corresponds to numu


        #
        # Set SME parameters
        #

        # Choose basis SME operators are defined in
        sme_basis = REF_SME_BASIS

        # Define "a" operator (magnitude and state texture)
        a_magnitude_eV = REF_SME_a_MAGNITUDE_eV
        a_mu_eV = get_sme_state_matrix(p33=a_magnitude_eV*0) # Choosing 33 element as only non-zero element in germs of flavor

        # Define "c" operator (magnitude and state texture)
        c_magnitude = REF_SME_c_MAGNITUDE
        c_t_nu = get_sme_state_matrix(p33=c_magnitude) # Choosing 33 element as only non-zero element in germs of flavor

        # Choose direction (sticking to axis directions for simplicity here)
        liv_direction = "z" #  x y z


        #
        # Create solver
        #

        kw = {}
        if args.solver == "nusquids":
            kw["energy_nodes_GeV"] = E_grid
            kw["nusquids_variant"] = "sme"
            kw["interactions"] = True

        # Initialize oscillation calculators for IceCube and off-axis detectors
        calculator = OscCalculator(solver=args.solver, atmospheric=True, **kw)

        # Set matter effects and detectors
        calculator.set_matter(matter)
        calculator.set_detector(detector)


        # Calculate oscillation probabilities
        sme_kw = {"sme_params":
                {"a_%s_eV"%liv_direction : a_mu_eV,
                    "c_t%s"%liv_direction : c_t_nu,
                    "basis":sme_basis}}

        state_osc_probs_sme[:,i,:], coszen_values, azimuth_values = calculator.calc_osc_prob_sme_directional_atmospheric(
            initial_flavor=initial_flavor,
            nubar=nubar,
            energy_GeV=E_grid,
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

        state_osc_probs_std[:, i,:], coszen_values, azimuth_values_std = calculator.calc_osc_prob_sme_directional_atmospheric(
            initial_flavor=initial_flavor,
            nubar=nubar,
            energy_GeV=E_grid,
            ra_rad=np.deg2rad(ra_deg),
            dec_rad=np.deg2rad(dec_deg),
            time=time,
            **std_kw
        )



# save the results
np.savez('tau_regeneration_results.npz', E_grid=E_grid,
        state_osc_probs_std=state_osc_probs_std,
        state_osc_probs_sme=state_osc_probs_sme)

print("check5")


        # flux_ratios = osc_probs_sme / osc_probs_std

        # ax.plot(E_values_GeV, flux_ratios[:,0], linestyle="-", color="orange", label=r"$\nu_e$")
        # ax.plot(E_values_GeV, flux_ratios[:,1], linestyle="-", color="dodgerblue", label=r"$\nu_\mu$")
        # ax.plot(E_values_GeV, flux_ratios[:,2], linestyle="-", color="green", label=r"$\nu_\tau$")


        # # Format plot
        # ax.set_ylim(0., 10)
        # ax.set_xlabel(ENERGY_LABEL, fontsize=14)
        # ax.set_xscale("log")
        # ax.tick_params(labelsize=12)
        # ax.grid(True)
        # ax.legend(fontsize=12, loc="lower right")
        # fig.tight_layout()

        # # Save the figure
        # print("")
        # dump_figures_to_pdf( __file__.replace(".py","_" + args.solver + ".pdf") )

        # Done