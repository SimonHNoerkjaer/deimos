'''
Calculate tau regeneration effects including LIV using interpolated atmospheric neutrino fluxes from MCEq.

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

import time

#
# Main 
#
if __name__ == "__main__":

    t_start = time.time()

    nubar = False  # False for neutrino, True for antineutrino

    detector = "IceCube"
    ra_deg = 0.
    dec_deg = +90. # Upgoing for IceCube

    ref_time = REF_TIME

    matter = "earth" # "earth" or "vacuum"


    #
    # Steering
    #
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--solver", type=str, required=False, default="nusquids", help="Solver name")
    parser.add_argument("-n", "--num-points", type=int, required=False, default=500, help="Num scan points")
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

    energy_mask = (E_grid >= 10.) & (E_grid <= 1000000.)  # 30 GeV to 1 PeV
    E_grid = E_grid[energy_mask]
    flux_nu_e = flux_nu_e[energy_mask]
    flux_nu_mu = flux_nu_mu[energy_mask]
    flux_nu_tau = flux_nu_tau[energy_mask]

    #INTERPOLATE FLUXES TO A FINER GRID
    import scipy.interpolate

    # Define a finer energy grid (logarithmic)
    E_grid_fine = np.geomspace(E_grid[0], E_grid[-1], args.num_points)


    # Interpolate fluxes onto the fine grid using log-log interpolation
    log_E_grid = np.log10(E_grid)
    log_E_grid_fine = np.log10(E_grid_fine)

    flux_nu_e_fine = 10**np.interp(log_E_grid_fine, log_E_grid, np.log10(flux_nu_e))
    flux_nu_mu_fine = 10**np.interp(log_E_grid_fine, log_E_grid, np.log10(flux_nu_mu))
    flux_nu_tau_fine = 10**np.interp(log_E_grid_fine, log_E_grid, np.log10(flux_nu_tau))


    # Use the fine grid for all further calculations
    E_grid = E_grid_fine
    flux_nu_e = flux_nu_e_fine
    flux_nu_mu = flux_nu_mu_fine
    flux_nu_tau = flux_nu_tau_fine


    # Print the loaded flux data for verification
    print("Energy Grid:", E_grid)
    print("Flux Nu_e:", flux_nu_e)
    print("Flux Nu_mu:", flux_nu_mu)
    print("Flux Nu_tau:", flux_nu_tau)



    states = np.array([0,1,2])

    # initialize osc calculator outside loop
    state_osc_probs_std = np.zeros((len(E_grid), len(states), 3)) #E, initial state, final state
    state_osc_probs_sme = np.zeros((len(E_grid), len(states), 3)) #E, initial state, final state

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
            time=ref_time,
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
            time=ref_time,
            **std_kw
        )


        # Save the results, including interpolated fluxes
    np.savez(
        'tau_regeneration_interp_results_10GeV_500.npz',
        E_grid=E_grid,
        flux_nu_e=flux_nu_e,
        flux_nu_mu=flux_nu_mu,
        flux_nu_tau=flux_nu_tau,
        state_osc_probs_std=state_osc_probs_std,
        state_osc_probs_sme=state_osc_probs_sme
    )

    print('Saved results to tau_regeneration_interp_results_10GeV_500.npz')
    t_end = time.time()
    min,sec  = np.divmod(t_end - t_start,60)
    print(f"Total elapsed time: {int(min)} minutes and {sec:.2f} seconds")
