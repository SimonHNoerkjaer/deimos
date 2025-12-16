'''
Compare standard oscillations with and without tau regeneration (no SME)
'''

import numpy as np
from deimos.wrapper.osc_calculator import OscCalculator
from deimos.utils.plotting import plt
from deimos.utils.constants import *
from deimos.models.liv.paper_plots.paper_def import *
import argparse

if __name__ == "__main__":

    import time as time_module
    ini_time = time_module.time()
    nubar = False
    detector = "IceCube"
    ra_deg = 0.
    dec_deg = +90.
    time = REF_TIME
    matter = "earth"

    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--solver", type=str, required=False, default="nusquids", help="Solver name")
    parser.add_argument("-n", "--num-points", type=int, required=False, default=1000, help="Num scan points")
    args = parser.parse_args()

    # Load flux data
    flux_data = np.load('mceq_fluxes.npz')
    E_grid = flux_data['energy_grid']
    flux_nu_e = flux_data['nue']
    flux_nu_mu = flux_data['numu']
    flux_nu_tau = flux_data['nutau']

    energy_mask = (E_grid >= 100.) & (E_grid <= 1000000.)
    E_grid = E_grid[energy_mask]
    flux_nu_e = flux_nu_e[energy_mask]
    flux_nu_mu = flux_nu_mu[energy_mask]
    flux_nu_tau = flux_nu_tau[energy_mask]

    states = np.array([0, 1, 2])
    state_osc_probs_regen = np.zeros((len(E_grid), len(states), 3))  # with tau regen
    state_osc_probs_no_regen = np.zeros((len(E_grid), len(states), 3))  # without tau regen

    for i, state in enumerate(states):
        initial_flavor = state
        sme_basis = REF_SME_BASIS
        a_mu_eV = np.zeros((3, 3), dtype=float)
        c_t_nu = np.zeros((3, 3), dtype=float)
        liv_direction = "z"

        # With tau regeneration (interactions=True)
        kw_regen = {}
        if args.solver == "nusquids":
            kw_regen["energy_nodes_GeV"] = E_grid
            kw_regen["nusquids_variant"] = "sme"
            kw_regen["interactions"] = True
        calculator_regen = OscCalculator(solver=args.solver, atmospheric=True, **kw_regen)
        calculator_regen.set_matter(matter)
        calculator_regen.set_detector(detector)
        std_kw = {"sme_params": {"a_%s_eV" % liv_direction: a_mu_eV, "c_t%s" % liv_direction: c_t_nu, "basis": sme_basis}}
        state_osc_probs_regen[:, i, :], _, _ = calculator_regen.calc_osc_prob_sme_directional_atmospheric(
            initial_flavor=initial_flavor,
            nubar=nubar,
            energy_GeV=E_grid,
            ra_rad=np.deg2rad(ra_deg),
            dec_rad=np.deg2rad(dec_deg),
            time=time,
            **std_kw
        )

        # Without tau regeneration (interactions=False)
        kw_no_regen = {}
        if args.solver == "nusquids":
            kw_no_regen["energy_nodes_GeV"] = E_grid
            kw_no_regen["nusquids_variant"] = "sme"
            kw_no_regen["interactions"] = False
        calculator_no_regen = OscCalculator(solver=args.solver, atmospheric=True, **kw_no_regen)
        calculator_no_regen.set_matter(matter)
        calculator_no_regen.set_detector(detector)
        state_osc_probs_no_regen[:, i, :], _, _ = calculator_no_regen.calc_osc_prob_sme_directional_atmospheric(
            initial_flavor=initial_flavor,
            nubar=nubar,
            energy_GeV=E_grid,
            ra_rad=np.deg2rad(ra_deg),
            dec_rad=np.deg2rad(dec_deg),
            time=time,
            **std_kw
        )

    # Calculate final fluxes
    initial_fluxes = np.stack([flux_nu_e, flux_nu_mu, flux_nu_tau], axis=1)
    final_flux_regen = np.einsum('ei,eif->ef', initial_fluxes, state_osc_probs_regen)
    final_flux_no_regen = np.einsum('ei,eif->ef', initial_fluxes, state_osc_probs_no_regen)

    # Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    color_e = "dodgerblue"
    color_mu = "red"
    color_tau = "green"
    alpha = 0.8

    # Plot 1: Final flux comparison (With vs Without tau regeneration)
    ax1.plot(E_grid, E_grid**3 * final_flux_no_regen[:, 0], linestyle="-", color=color_e, label=r"$\nu_e$ (No Regen)", linewidth=2, alpha=alpha)
    ax1.plot(E_grid, E_grid**3 * final_flux_no_regen[:, 1], linestyle="-", color=color_mu, label=r"$\nu_\mu$ (No Regen)", linewidth=2, alpha=alpha)
    ax1.plot(E_grid, E_grid**3 * final_flux_no_regen[:, 2], linestyle="-", color=color_tau, label=r"$\nu_\tau$ (No Regen)", linewidth=2, alpha=alpha)
    ax1.plot(E_grid, E_grid**3 * final_flux_regen[:, 0], linestyle="--", color=color_e, label=r"$\nu_e$ (Tau Regen)", linewidth=2, alpha=alpha)
    ax1.plot(E_grid, E_grid**3 * final_flux_regen[:, 1], linestyle="--", color=color_mu, label=r"$\nu_\mu$ (Tau Regen)", linewidth=2, alpha=alpha)
    ax1.plot(E_grid, E_grid**3 * final_flux_regen[:, 2], linestyle="--", color=color_tau, label=r"$\nu_\tau$ (Tau Regen)", linewidth=2, alpha=alpha)
    ax1.set_ylabel(r"$E^3 \times \phi$  [GeV$^2$ cm$^{-2}$ s$^{-1}$ sr$^{-1}$]", fontsize=12)
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_title(f"Final Flux Comparison: With vs Without Tau Regeneration", fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.text(0.05, 1.2, f"(Det: {detector}, Dec: {dec_deg}°, Matter: {matter})", transform=ax1.transAxes, fontsize=12, verticalalignment='top')

    # Plot 2: Flux ratio (Tau Regen / No Regen)
    ratio_e = final_flux_regen[:, 0] / final_flux_no_regen[:, 0]
    ratio_mu = final_flux_regen[:, 1] / final_flux_no_regen[:, 1]
    ratio_tau = final_flux_regen[:, 2] / final_flux_no_regen[:, 2]
    ratio_total = np.sum(final_flux_regen, axis=1) / np.sum(final_flux_no_regen, axis=1)

    ax2.plot(E_grid, ratio_e, linestyle="-", color=color_e, label=r"$\nu_e$", linewidth=2, alpha=alpha)
    ax2.plot(E_grid, ratio_mu, linestyle="-", color=color_mu, label=r"$\nu_\mu$", linewidth=2, alpha=alpha)
    ax2.plot(E_grid, ratio_tau, linestyle="-", color=color_tau, label=r"$\nu_\tau$", linewidth=2, alpha=alpha)
    ax2.plot(E_grid, ratio_total, linestyle="-", color="black", label=r"Total", linewidth=2, alpha=alpha)

    # # Zoom-in inset
    # ax2_zoom = ax2.inset_axes([0.66, 0.25, 0.32, 0.32])
    # ax2_zoom.plot(E_grid, ratio_e, linestyle="-", color=color_e, linewidth=2, alpha=alpha)
    # ax2_zoom.plot(E_grid, ratio_mu, linestyle="-", color=color_mu, linewidth=2, alpha=alpha)
    # ax2_zoom.plot(E_grid, ratio_tau, linestyle="-", color=color_tau, linewidth=2, alpha=alpha)
    # ax2_zoom.plot(E_grid, ratio_total, linestyle="-", color="black", linewidth=2, alpha=alpha)
    # ax2_zoom.set(ylim=(0, 2), xscale="log", ylabel=r'$\phi_{\text{regen}}$ / $\phi_{\text{no regen}}$', title="Zoom-In")
    # ax2_zoom.set_xlabel("E (GeV)", labelpad=1)
    # ax2_zoom.grid(True, alpha=0.3)

    ax2.axhline(1.0, color='black', linestyle='--', alpha=0.5, linewidth=1)
    ax2.set_ylabel(r'$\phi_{\text{regen}}$ / $\phi_{\text{no regen}}$', fontsize=12)
    ax2.set_xlabel("E (GeV)", fontsize=12)
    ax2.set_xscale("log")
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    plt.savefig(f"STDonly_tau_regeneration_comparison.pdf", dpi=150)

    total_time = time_module.time() - ini_time
    minutes, seconds = divmod(total_time, 60)
    print(f"Total runtime: {int(minutes)}m {int(seconds)}s")