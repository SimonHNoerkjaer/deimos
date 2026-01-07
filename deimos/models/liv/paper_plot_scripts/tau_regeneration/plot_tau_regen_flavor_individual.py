'''
Plot individuallly each neutrino flavor for tau regeneration effects including LIV using interpolated atmospheric neutrino fluxes from MCEq.

Script by Simon Hilding-Nørkjær
'''

import numpy as np
import matplotlib.pyplot as plt

# Load all scenarios
data_vac = np.load('tau_regeneration_interp_results_10GeV_vacuum.npz')
data_earth_no_tau = np.load('tau_regeneration_interp_results_10GeV_earth_noTauRegen.npz') 
data_earth_tau = np.load('tau_regeneration_interp_results_10GeV.npz') # Earth with tau regeneration

E_grid = data_vac['E_grid']

# Calculate final fluxes for each scenario and oscillation type
def get_final_flux(data):
    flux_nu_e = data['flux_nu_e']
    flux_nu_mu = data['flux_nu_mu']
    flux_nu_tau = data['flux_nu_tau']
    initial_fluxes = np.stack([flux_nu_e, flux_nu_mu, flux_nu_tau], axis=1)  # (E, 3)
    final_flux_std = np.einsum('ei,eif->ef', initial_fluxes, data['state_osc_probs_std'])
    final_flux_sme = np.einsum('ei,eif->ef', initial_fluxes, data['state_osc_probs_sme'])
    return final_flux_std, final_flux_sme

final_flux_std_vac, final_flux_sme_vac = get_final_flux(data_vac)
final_flux_std_earth_no_tau, final_flux_sme_earth_no_tau = get_final_flux(data_earth_no_tau)
final_flux_std_earth_tau, final_flux_sme_earth_tau = get_final_flux(data_earth_tau)

# Plot settings
flavor_labels = [r"$\nu_e$", r"$\nu_\mu$", r"$\nu_\tau$"]

scenario_colors = ["#00a6ff",  "#003fd1", "#d90000",  "#ffa200", "#8ada00",  "#008962"]
panel_colors = [scenario_colors, scenario_colors, scenario_colors]

styles = ["-", '-', (0,(4.5,4)), (0,(2.5,1.5)), (0, (1, 1.5)), (0, (1, 1.5))]

scenarios = [
    "STD (Vacuum)",
    "SME (Vacuum)",
    "STD (Earth)",
    "SME (Earth)",
    "STD (Tau regen)",
    "SME (Tau regen)",
]

fluxes = [
    final_flux_std_vac,
    final_flux_sme_vac,
    final_flux_std_earth_no_tau,
    final_flux_sme_earth_no_tau,
    final_flux_std_earth_tau,
    final_flux_sme_earth_tau,
]


fig, axes = plt.subplots(3, 1, figsize=(6, 10), sharex=True)

for i, ax in enumerate(axes):
    for j, (label, flux, style) in enumerate(zip(scenarios, fluxes, styles)):
        if j<3:
            linewidth=2
        else:
            linewidth=2.3
        if j!=5 and j!=4:
            ax.plot(E_grid, E_grid**3 * flux[:, i], linestyle=style, alpha=1, color=panel_colors[i][j], label=label, linewidth=linewidth, zorder=1)
        if j==5 or j==4:
            ax.plot(E_grid, E_grid**3 * flux[:, i], linestyle=style, alpha=1, color=panel_colors[i][j], label=label, linewidth=linewidth+1, zorder=0)
    ax.set_ylabel(r"$E^3 \times \phi$  [GeV$^3$ cm$^{-2}$ s$^{-1}$ sr$^{-1}$]", fontsize=12)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.text(0.94, 0.96, f"{flavor_labels[i]}", transform=ax.transAxes, fontsize=14, verticalalignment='top',bbox=dict(boxstyle="round,pad=0.3",alpha=1, fc='w', ec='k', lw=1))
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.grid(True, alpha=0.3)
    
axes[0].legend(handletextpad=0.5, columnspacing=0.5, ncol=3, fontsize=11, loc='upper center', bbox_to_anchor=(0.5, 1.22))
axes[-1].set_xlabel("Energy [GeV]", fontsize=12)

fig.tight_layout()
plt.savefig("tau_regeneration_flux_comparison_flavor_panels_2.pdf", dpi=150)