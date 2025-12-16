'''
Plot SME energy-dependence with flat initial spectrum, combining pure flavor results weighted by fractions at E_mono.
'''

import numpy as np
from deimos.wrapper.osc_calculator import OscCalculator
from deimos.utils.plotting import plt
from deimos.utils.constants import *
from deimos.models.liv.paper_plots.paper_def import *
from deimos.models.liv.sme import get_sme_state_matrix
import argparse

if __name__ == "__main__":

    # --- Config ---
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--solver", type=str, default="nusquids")
    parser.add_argument("-n", "--num-points", type=int, default=30)
    args = parser.parse_args()

    detector = "IceCube"
    ra_deg = 0.
    dec_deg = 90.
    ref_time = REF_TIME
    matter = "earth"
    nubar = False

    # --- Load fluxes ---
    flux_data = np.load('mceq_fluxes.npz')
    E_grid = flux_data['energy_grid']
    flux_nu_e = flux_data['nue']
    flux_nu_mu = flux_data['numu']
    flux_nu_tau = flux_data['nutau']

    energy_mask = (E_grid >= 10.) & (E_grid <= 1e6)
    E_grid = E_grid[energy_mask]
    flux_nu_e = flux_nu_e[energy_mask]
    flux_nu_mu = flux_nu_mu[energy_mask]
    flux_nu_tau = flux_nu_tau[energy_mask]

    # Interpolate to finer grid
    E_grid_fine = np.geomspace(E_grid[0], E_grid[-1], args.num_points)
    log_E_grid = np.log10(E_grid)
    log_E_grid_fine = np.log10(E_grid_fine)
    flux_nu_e_fine = 10**np.interp(log_E_grid_fine, log_E_grid, np.log10(flux_nu_e))
    flux_nu_mu_fine = 10**np.interp(log_E_grid_fine, log_E_grid, np.log10(flux_nu_mu))
    flux_nu_tau_fine = 10**np.interp(log_E_grid_fine, log_E_grid, np.log10(flux_nu_tau))

    E_grid = E_grid_fine
    flux_nu_e = flux_nu_e_fine
    flux_nu_mu = flux_nu_mu_fine
    flux_nu_tau = flux_nu_tau_fine

    # --- Initial flavor fractions at E_mono ---
    E_mono = 1e4  # 100 TeV
    mono_idx = np.abs(E_grid - E_mono).argmin()
    total_mono_flux = flux_nu_e[mono_idx] + flux_nu_mu[mono_idx] + flux_nu_tau[mono_idx]
    flavor_frac = np.array([flux_nu_e[mono_idx], flux_nu_mu[mono_idx], flux_nu_tau[mono_idx]]) / total_mono_flux

    # --- SME parameters ---
    sme_basis = REF_SME_BASIS
    a_magnitude_eV = REF_SME_a_MAGNITUDE_eV
    a_mu_eV = get_sme_state_matrix(p33=a_magnitude_eV*0)
    c_magnitude = REF_SME_c_MAGNITUDE
    c_t_nu = get_sme_state_matrix(p33=c_magnitude)
    liv_direction = "z"

    kw = {}
    if args.solver == "nusquids":
        kw["energy_nodes_GeV"] = E_grid
        kw["nusquids_variant"] = "sme"
        kw["interactions"] = True

    calculator = OscCalculator(solver=args.solver, atmospheric=True, **kw)
    calculator.set_matter(matter)
    calculator.set_detector(detector)

    sme_kw = {"sme_params": {
        f"a_{liv_direction}_eV": a_mu_eV,
        f"c_t{liv_direction}": c_t_nu,
        "basis": sme_basis
    }}

    std_kw = {"sme_params": {
        f"a_{liv_direction}_eV": np.zeros_like(a_mu_eV),
        f"c_t{liv_direction}": np.zeros_like(c_t_nu),
        "basis": sme_basis
    }}

    # --- Propagate each pure flavor, weight by initial fractions ---
    state_osc_probs_sme = np.zeros((len(E_grid), 3))
    state_osc_probs_std = np.zeros((len(E_grid), 3))
    for i_flavor in range(3):
        # SME
        osc_probs_sme, _, _ = calculator.calc_osc_prob_sme_directional_atmospheric(
            initial_flavor=i_flavor,
            nubar=nubar,
            energy_GeV=E_grid,
            ra_rad=np.deg2rad(ra_deg),
            dec_rad=np.deg2rad(dec_deg),
            time=ref_time,
            **sme_kw
        )
        state_osc_probs_sme += osc_probs_sme * flavor_frac[i_flavor]
        # Standard
        osc_probs_std, _, _ = calculator.calc_osc_prob_sme_directional_atmospheric(
            initial_flavor=i_flavor,
            nubar=nubar,
            energy_GeV=E_grid,
            ra_rad=np.deg2rad(ra_deg),
            dec_rad=np.deg2rad(dec_deg),
            time=ref_time,
            **std_kw
        )
        state_osc_probs_std += osc_probs_std * flavor_frac[i_flavor]

        #save data
    np.savez_compressed(f'monoBeamApprox_results_{E_mono:.0e}.npz',
        E_grid=E_grid,
        flavor_frac=flavor_frac,
        state_osc_probs_std=state_osc_probs_std,
        state_osc_probs_sme=state_osc_probs_sme
    )
    
    # --- Flat initial spectrum ---
    initial_fluxes = np.ones((len(E_grid), 3))  # Flat spectrum

    # --- Final fluxes ---
    # final_flux_std = np.einsum('ei,if->ef', initial_fluxes, state_osc_probs_std)
    # final_flux_sme = np.einsum('ei,if->ef', initial_fluxes, state_osc_probs_sme)

    final_flux_std = initial_fluxes * state_osc_probs_std
    final_flux_sme = initial_fluxes * state_osc_probs_sme   

    #subtrackt initial fluxes to get only regenerated component
    final_flux_std -= initial_fluxes
    final_flux_sme -= initial_fluxes

    # --- Plot ---
    fig, ax1 = plt.subplots(1, 1, figsize=(8, 6))
    color_e = "dodgerblue"
    color_mu = "red"
    color_tau = "green"
    alpha = 0.8

    ax1.plot(E_grid, E_grid**3 * final_flux_std[:, 0], "-", color=color_e, label=r"$\nu_e$ (Standard)", linewidth=2, alpha=alpha)
    ax1.plot(E_grid, E_grid**3 * final_flux_std[:, 1], "-", color=color_mu, label=r"$\nu_\mu$ (Standard)", linewidth=2, alpha=alpha)
    ax1.plot(E_grid, E_grid**3 * final_flux_std[:, 2], "-", color=color_tau, label=r"$\nu_\tau$ (Standard)", linewidth=2, alpha=alpha)
    ax1.plot(E_grid, E_grid**3 * final_flux_sme[:, 0], "--", color=color_e, label=r"$\nu_e$ (SME)", linewidth=2, alpha=alpha)
    ax1.plot(E_grid, E_grid**3 * final_flux_sme[:, 1], "--", color=color_mu, label=r"$\nu_\mu$ (SME)", linewidth=2, alpha=alpha)
    ax1.plot(E_grid, E_grid**3 * final_flux_sme[:, 2], "--", color=color_tau, label=r"$\nu_\tau$ (SME)", linewidth=2, alpha=alpha)
    ax1.set_ylabel(r"$E^3 \times \phi$  [GeV$^2$ cm$^{-2}$ s$^{-1}$ sr$^{-1}$]", fontsize=12)
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_title(f"Final Flux Comparison: Standard vs SME", fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.text(0.05, 1.2, f"(Det: {detector}, Dec: {dec_deg}°, Matter: {matter}, a_mag: {REF_SME_a_MAGNITUDE_eV*0:.1e} eV, c_mag: {REF_SME_c_MAGNITUDE:.1e})", transform=ax1.transAxes, fontsize=12, verticalalignment='top')
    ax1.set_xlabel("E (GeV)", fontsize=12)

    fig.tight_layout()
    plt.savefig(f"monoBeam_weighted_init_{E_mono:.0e}.pdf", dpi=150)