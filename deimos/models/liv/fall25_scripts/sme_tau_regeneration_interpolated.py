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

import time

#
# Main 
#
if __name__ == "__main__":

    t_start = time.time()

    nubar = False  # False for neutrino, True for antineutrino

    # E_values_GeV = np.geomspace(1e2, 1e7, num=args.num_points)

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

# --- Plotting: Final Fluxes and Ratios (using interpolated grid and results) ---

# import matplotlib.pyplot as plt

# # Use the same E_grid and fluxes as above (already interpolated)
# # state_osc_probs_std and state_osc_probs_sme are already in memory

# pure_e_std = state_osc_probs_std[:,0,:]
# pure_mu_std = state_osc_probs_std[:,1,:]
# pure_tau_std = state_osc_probs_std[:,2,:]

# pure_e_sme = state_osc_probs_sme[:,0,:]
# pure_mu_sme = state_osc_probs_sme[:,1,:]
# pure_tau_sme = state_osc_probs_sme[:,2,:]

# # Initial fluxes for each flavor
# initial_fluxes = np.stack([flux_nu_e, flux_nu_mu, flux_nu_tau], axis=1)  # Shape: (E, 3)

# # Calculate final fluxes by applying oscillation probabilities
# final_flux_std = np.einsum('ei,eif->ef', initial_fluxes, state_osc_probs_std)
# final_flux_sme = np.einsum('ei,eif->ef', initial_fluxes, state_osc_probs_sme)

# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
# color_e = "dodgerblue"
# color_mu = "red"
# color_tau = "green"
# alpha = 0.8

# # Plot 1: Final flux comparison (Standard vs SME)
# ax1.plot(E_grid, E_grid**3 * final_flux_std[:, 0], linestyle="-", color=color_e, label=r"$\nu_e$ (Standard)", linewidth=2, alpha=alpha)
# ax1.plot(E_grid, E_grid**3 * final_flux_std[:, 1], linestyle="-", color=color_mu, label=r"$\nu_\mu$ (Standard)", linewidth=2, alpha=alpha)
# ax1.plot(E_grid, E_grid**3 * final_flux_std[:, 2], linestyle="-", color=color_tau, label=r"$\nu_\tau$ (Standard)", linewidth=2, alpha=alpha)
# ax1.plot(E_grid, E_grid**3 * final_flux_sme[:, 0], linestyle="--", color=color_e, label=r"$\nu_e$ (SME)", linewidth=2, alpha=alpha)
# ax1.plot(E_grid, E_grid**3 * final_flux_sme[:, 1], linestyle="--", color=color_mu, label=r"$\nu_\mu$ (SME)", linewidth=2, alpha=alpha)
# ax1.plot(E_grid, E_grid**3 * final_flux_sme[:, 2], linestyle="--", color=color_tau, label=r"$\nu_\tau$ (SME)", linewidth=2, alpha=alpha)
# ax1.set_ylabel(r"$E^3 \times \phi$  [GeV$^2$ cm$^{-2}$ s$^{-1}$ sr$^{-1}$]", fontsize=12)
# ax1.set_xscale("log")
# ax1.set_yscale("log")
# ax1.set_title(f"Final Flux Comparison: Standard vs SME", fontsize=14)
# ax1.legend(fontsize=11)
# ax1.grid(True, alpha=0.3)
# ax1.text(0.05, 1.2, f"(Det: {detector}, Dec: {dec_deg}°, Matter: {matter}, a_mag: {REF_SME_a_MAGNITUDE_eV*0:.1e} eV, c_mag: {REF_SME_c_MAGNITUDE:.1e})", transform=ax1.transAxes, fontsize=12, verticalalignment='top')

# # Plot 2: Flux ratio (SME / Standard)
# ratio_e = final_flux_sme[:, 0] / final_flux_std[:, 0]
# ratio_mu = final_flux_sme[:, 1] / final_flux_std[:, 1]
# ratio_tau = final_flux_sme[:, 2] / final_flux_std[:, 2]
# ratio_total = np.sum(final_flux_sme, axis=1) / np.sum(final_flux_std, axis=1)

# ax2.plot(E_grid, ratio_e, linestyle="-", color=color_e, label=r"$\nu_e$", linewidth=2, alpha=alpha)
# ax2.plot(E_grid, ratio_mu, linestyle="-", color=color_mu, label=r"$\nu_\mu$", linewidth=2, alpha=alpha)
# ax2.plot(E_grid, ratio_tau, linestyle="-", color=color_tau, label=r"$\nu_\tau$", linewidth=2, alpha=alpha)
# ax2.plot(E_grid, ratio_total, linestyle="-", color="black", label=r"Total", linewidth=2, alpha=alpha)

# # Zoom-in inset
# ax2_zoom = ax2.inset_axes([0.66, 0.25, 0.32, 0.32])
# ax2_zoom.plot(E_grid, ratio_e, linestyle="-", color=color_e, linewidth=2, alpha=alpha)
# ax2_zoom.plot(E_grid, ratio_mu, linestyle="-", color=color_mu, linewidth=2, alpha=alpha)
# ax2_zoom.plot(E_grid, ratio_tau, linestyle="-", color=color_tau, linewidth=2, alpha=alpha)
# ax2_zoom.plot(E_grid, ratio_total, linestyle="-", color="black", linewidth=2, alpha=alpha)
# ax2_zoom.set(ylim = (0, 5), xscale="log", ylabel=r'$\phi_{\text{SME}}$ / $\phi_{\text{Standard}}$', title="Zoom-In")
# ax2_zoom.set_xlabel("E (GeV)", labelpad=1)
# ax2_zoom.grid(True, alpha=0.3)

# ax2.axhline(1.0, color='black', linestyle='--', alpha=0.5, linewidth=1)
# ax2.set_ylabel(r'$\phi_{\text{SME}}$ / $\phi_{\text{Standard}}$', fontsize=12)
# ax2.set_xlabel("E (GeV)", fontsize=12)
# ax2.set_xscale("log")
# ax2.legend(fontsize=11)
# ax2.grid(True, alpha=0.3)

# fig.tight_layout()

# plt.savefig(f"tau_regeneration_flux_comparison_interp_{REF_SME_a_MAGNITUDE_eV*0:.1e}_{REF_SME_c_MAGNITUDE:.1e}.pdf", dpi=150)
#         # flux_ratios = osc_probs_sme / osc_probs_std

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