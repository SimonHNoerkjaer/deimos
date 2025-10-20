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
from MCEq.core import MCEqRun
import crflux.models as pm

#
# Main 
#
if __name__ == "__main__":

    #
    # Steering
    # #
    # import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument("-s", "--solver", type=str, required=False, default="nusquids", help="Solver name")
    # parser.add_argument("-n", "--num-points", type=int, required=False, default=1000, help="Num scan points")
    # args = parser.parse_args()

        # #
    # Define basic system parameters
    #
    ENERGY_BINS = np.logspace(1, 5, 200)  # GeV - bin centers

    # For MCEq, we need bin edges (N+1 values for N bins)
    # Create bin edges from bin centers
    log_bin_centers = np.log10(ENERGY_BINS)
    bin_width = log_bin_centers[1] - log_bin_centers[0]
    log_bin_edges = np.concatenate([[log_bin_centers[0] - bin_width/2], 
                                     log_bin_centers + bin_width/2])
    ENERGY_BIN_EDGES = 10**log_bin_edges

    # Choose a model setup
    mceq = MCEqRun(
        interaction_model='SIBYLL23C',
        primary_model=(pm.HillasGaisser2012, 'H3a'),
        theta_deg=0.0,  # vertical flux
        e_bins=ENERGY_BIN_EDGES
    )
    # Compute for a range of energies

    mceq.solve()
    
    # Use the energy grid that MCEq actually uses (bin centers)
    E_grid = mceq.e_grid
    
    # Get neutrino fluxes
    flux_nu_mu = mceq.get_solution('numu')
    flux_nu_mu_bar = mceq.get_solution('antinumu')
    flux_nu_e = mceq.get_solution('nue')
    flux_nu_e_bar = mceq.get_solution('antinue')
    flux_nu_tau = mceq.get_solution('nutau')
    flux_nu_tau_bar = mceq.get_solution('antinutau')  # Fixed typo


    print("Flux nu_mu:", flux_nu_mu)
    print("Flux nu_mu shape:", flux_nu_mu.shape)
    print("E_grid:", E_grid)
    print("E_grid shape:", E_grid.shape)
    
    # Plot fluxes for neutrinos and antineutrinos
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Neutrinos - use E_grid from MCEq
    ax1.plot(E_grid, E_grid**3*flux_nu_e, label=r'$\nu_e$', color='orange')
    ax1.plot(E_grid, E_grid**3*flux_nu_mu, label=r'$\nu_\mu$', color='dodgerblue')
    ax1.plot(E_grid, E_grid**3*flux_nu_tau, label=r'$\nu_\tau$', color='green')
    ax1.set_xlabel('Energy [GeV]')
    ax1.set_ylabel(r'Flux [GeV$^{-1}$ cm$^{-2}$ s$^{-1}$ sr$^{-1}$]')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_title('Neutrinos')
    ax1.legend()
    ax1.grid(True)

    # Antineutrinos - use E_grid from MCEq
    ax2.plot(E_grid, E_grid**3*flux_nu_e_bar, label=r'$\bar{\nu}_e$', color='orange', linestyle='--')
    ax2.plot(E_grid, E_grid**3*flux_nu_mu_bar, label=r'$\bar{\nu}_\mu$', color='dodgerblue', linestyle='--')
    ax2.plot(E_grid, E_grid**3*flux_nu_tau_bar, label=r'$\bar{\nu}_\tau$', color='green', linestyle='--')
    ax2.set_xlabel('Energy [GeV]')
    ax2.set_ylabel(r'Flux [GeV$^{-1}$ cm$^{-2}$ s$^{-1}$ sr$^{-1}$]')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_title('Antineutrinos')
    ax2.legend()
    ax2.grid(True)

    fig.tight_layout()
    plt.show()

    plt.savefig("mceq_fluxes.png")


    # Save all fluxes in a single numpy file
    fluxes = {
        'energy_grid': E_grid,
        'nue': flux_nu_e,
        'numu': flux_nu_mu,
        'nutau': flux_nu_tau,
        'nuebar': flux_nu_e_bar,
        'numubar': flux_nu_mu_bar,
        'nutaubar': flux_nu_tau_bar
    }
    np.savez("mceq_fluxes.npz", **fluxes)
