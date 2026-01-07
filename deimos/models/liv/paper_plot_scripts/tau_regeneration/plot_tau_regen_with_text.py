'''
Plot tau regeneration effects including LIV using interpolated atmospheric neutrino fluxes from MCEq.

Script by Simon Hilding-Nørkjær
'''

from deimos.models.liv.paper_plots.paper_def import *
import collections
import numpy as np
import matplotlib.pyplot as plt

# Define system parameters
nubar = False  # False for neutrino, True for antineutrino
detector = "IceCube"
ra_deg = 0.
dec_deg = +90. # Upgoing for IceCube
time = REF_TIME
a_magnitude = REF_SME_a_MAGNITUDE_eV*0.
c_magnitude = REF_SME_c_MAGNITUDE
matter = "earth" # "earth" or "vacuum"


# ####################### Tau Regen (nuSQuIDS/deimos data) #########################

data = np.load('tau_regeneration_interp_results_10GeV_500.npz')
E_grid = data['E_grid']
flux_nu_e = data['flux_nu_e']
flux_nu_mu = data['flux_nu_mu']
flux_nu_tau = data['flux_nu_tau']
state_osc_probs_std = data['state_osc_probs_std']
state_osc_probs_sme = data['state_osc_probs_sme']

pure_e_std = state_osc_probs_std[:,0,:]
pure_mu_std = state_osc_probs_std[:,1,:]
pure_tau_std = state_osc_probs_std[:,2,:]

pure_e_sme = state_osc_probs_sme[:,0,:]
pure_mu_sme = state_osc_probs_sme[:,1,:]
pure_tau_sme = state_osc_probs_sme[:,2,:]



################################## Calculate final fluxes ##########################

# Initial fluxes for each flavor
initial_fluxes = np.stack([flux_nu_e, flux_nu_mu, flux_nu_tau], axis=1)  # Shape: (E, 3)

# Calculate final fluxes by applying oscillation probabilities : final_flux[E, flavor_f] = sum_i (initial_flux[E, flavor_i] * P(flavor_i -> flavor_f))
final_flux_std = np.einsum('ei,eif->ef', initial_fluxes, state_osc_probs_std)
final_flux_sme = np.einsum('ei,eif->ef', initial_fluxes, state_osc_probs_sme)


############################## Plotting ###########################################

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.5, 8), sharex=True)
# Update color scheme
color_e = "dodgerblue"
color_mu = "red"
color_tau = "green"
alpha = 0.8
E_power = 2

# Plot 1: Final flux comparison (Standard vs SME)
ax1.plot(E_grid, E_grid**E_power * final_flux_std[:, 0], linestyle="-", color=color_e, label=r"$\nu_e$ (Standard)", linewidth=2, alpha=alpha)
ax1.plot(E_grid, E_grid**E_power * final_flux_std[:, 1], linestyle="-", color=color_mu, label=r"$\nu_\mu$ (Standard)", linewidth=2, alpha=alpha)
ax1.plot(E_grid, E_grid**E_power * final_flux_std[:, 2], linestyle="-", color=color_tau, label=r"$\nu_\tau$ (Standard)", linewidth=2, alpha=alpha)
#total flux
# ax1.plot(E_grid, E_grid**3 * np.sum(final_flux_std, axis=1), linestyle="-", color="black", label=r"Total (Standard)", linewidth=2, alpha=alpha)

ax1.plot(E_grid, E_grid**E_power * final_flux_sme[:, 0], linestyle="--", color=color_e, label=r"$\nu_e$ (SME)", linewidth=2, alpha=alpha)
ax1.plot(E_grid, E_grid**E_power * final_flux_sme[:, 1], linestyle="--", color=color_mu, label=r"$\nu_\mu$ (SME)", linewidth=2, alpha=alpha)
ax1.plot(E_grid, E_grid**E_power * final_flux_sme[:, 2], linestyle="--", color=color_tau, label=r"$\nu_\tau$ (SME)", linewidth=2, alpha=alpha)
#total flux
# ax1.plot(E_grid, E_grid**3 * np.sum(final_flux_sme, axis=1), linestyle="--", color="black", label=r"Total (SME)", linewidth=2, alpha=alpha)

ax1.set_ylabel(rf"$E^{E_power} \times \phi$  [GeV$^2$ cm$^{-2}$ s$^{-1}$ sr$^{-1}$]", fontsize=13)
ax1.set_xscale("log")
ax1.set_yscale("log")
ax1.set_title(f"Final Flux Comparison: Standard vs SME", fontsize=14)
ax1.legend(fontsize=12, loc="lower left")
ax1.grid(True, alpha=0.3)
ax1.text(0.05, 1.2, f"(Det: {detector}, Dec: {dec_deg}°, Matter: {matter}, a_mag: {a_magnitude:.1e} eV, c_mag: {c_magnitude:.1e})", transform=ax1.transAxes, fontsize=12, verticalalignment='top')

# Plot 2: Flux ratio (SME / Standard)
ratio_e = final_flux_sme[:, 0] / final_flux_std[:, 0]
ratio_mu = final_flux_sme[:, 1] / final_flux_std[:, 1]
ratio_tau = final_flux_sme[:, 2] / final_flux_std[:, 2]
ratio_total = np.sum(final_flux_sme, axis=1) / np.sum(final_flux_std, axis=1)

ax2.plot(E_grid, ratio_e, linestyle="-", color=color_e, label=r"$\nu_e$", linewidth=2, alpha=alpha)
ax2.plot(E_grid, ratio_mu, linestyle="-", color=color_mu, label=r"$\nu_\mu$", linewidth=2, alpha=alpha)

ax2.plot(E_grid, ratio_tau, linestyle="-", color=color_tau, label=r"$\nu_\tau$", linewidth=2, alpha=alpha)
ax2.plot(E_grid, ratio_total, linestyle="-", color="black", label=r"Total", linewidth=2, alpha=alpha)

#add zoom-in ax for y=0 to 10
ax2_zoom = ax2.inset_axes([0.57, 0.34, 0.4, 0.4])  # x , y , x_size, y_size
ax2_zoom.plot(E_grid, ratio_e, linestyle="-", color=color_e, linewidth=2, alpha=alpha)
ax2_zoom.plot(E_grid, ratio_mu, linestyle="-", color=color_mu, linewidth=2, alpha=alpha)
ax2_zoom.plot(E_grid, ratio_tau, linestyle="-", color=color_tau, linewidth=2, alpha=alpha)
ax2_zoom.plot(E_grid, ratio_total, linestyle="-", color="black", linewidth=2, alpha=alpha)
ax2_zoom.set(ylim = (0, 20), xscale="log", ylabel=r'$\phi_{\text{SME}}$ / $\phi_{\text{Standard}}$')
ax2_zoom.set_title("Zoom-in",horizontalalignment="center", loc="left")
ax2_zoom.set_xlabel("E (GeV)", labelpad=1)
ax2_zoom.grid(True, alpha=0.3)

ax2.axhline(1.0, color='black', linestyle='--', alpha=0.5, linewidth=1)
ax2.set_ylabel(r'$\phi_{\text{SME}}$ / $\phi_{\text{Standard}}$', fontsize=14)
ax2.set_xlabel("E (GeV)", fontsize=14)
ax2.set_xscale("log")
ax2.legend(fontsize=12, loc="upper left")
ax2.grid(True, alpha=0.3)

# ticks fontsize
for ax in [ax1, ax2]:
    ax.tick_params(axis='both', which='major', labelsize=12)

ax2.text(0.37,0.6, "Tau \nregen",  transform=ax2.transAxes,  horizontalalignment="center", color="green", fontsize=13)
ax2_zoom.text(0.38, 0.65, "MSW", transform=ax2_zoom.transAxes, color="dodgerblue", fontsize=12)
ax2.text(0.84,0.755, "High Energy\nLIV Oscillations", horizontalalignment="center" , transform=ax2.transAxes, color="green", fontsize=12)

fig.tight_layout()

# Save the figure
plt.savefig(f"tau_regeneration_flux_comparison_interp_10GeV_text_{a_magnitude:.1e}_{c_magnitude:.1e}.pdf", dpi=150)
