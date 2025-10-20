from deimos.models.liv.paper_plots.paper_def import *
import collections
import numpy as np
import matplotlib.pyplot as plt




nubar = False  # False for neutrino, True for antineutrino
detector = "IceCube"
ra_deg = 0.
dec_deg = +90. # Upgoing for IceCube
time = REF_TIME
a_magnitude = REF_SME_a_MAGNITUDE_eV
c_magnitude = REF_SME_c_MAGNITUDE
matter = "earth" # "earth" or "vacuum"



######################### MCEq DATA  #########################
# # Load flux data from a file (assuming it's in a .npz format)
flux_data = np.load('mceq_fluxes.npz')

E_grid = flux_data['energy_grid']
flux_nu_e = flux_data['nue']
flux_nu_mu = flux_data['numu']
flux_nu_tau = flux_data['nutau']

energy_mask = (E_grid >= 100.) & (E_grid <= 1000000.)  # 100 GeV to 1 PeV
E_grid = E_grid[energy_mask]
flux_nu_e = flux_nu_e[energy_mask]
flux_nu_mu = flux_nu_mu[energy_mask]
flux_nu_tau = flux_nu_tau[energy_mask]



####################### Tau Regen (nuSQuIDS/deimos data) #########################
data = np.load('tau_regeneration_results.npz')

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

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
# Update color scheme
color_e = "dodgerblue"
color_mu = "red"
color_tau = "green"
# Plot 1: Final flux comparison (Standard vs SME)
ax1.plot(E_grid, E_grid**3 * final_flux_std[:, 0], linestyle="-", color=color_e, label=r"$\nu_e$ (Standard)", linewidth=2)
ax1.plot(E_grid, E_grid**3 * final_flux_std[:, 1], linestyle="-", color=color_mu, label=r"$\nu_\mu$ (Standard)", linewidth=2)
ax1.plot(E_grid, E_grid**3 * final_flux_std[:, 2], linestyle="-", color=color_tau, label=r"$\nu_\tau$ (Standard)", linewidth=2)

ax1.plot(E_grid, E_grid**3 * final_flux_sme[:, 0], linestyle="--", color=color_e, label=r"$\nu_e$ (SME)", linewidth=2)
ax1.plot(E_grid, E_grid**3 * final_flux_sme[:, 1], linestyle="--", color=color_mu, label=r"$\nu_\mu$ (SME)", linewidth=2)
ax1.plot(E_grid, E_grid**3 * final_flux_sme[:, 2], linestyle="--", color=color_tau, label=r"$\nu_\tau$ (SME)", linewidth=2)

ax1.set_ylabel(r"$E^3 \times \phi$  [GeV$^2$ cm$^{-2}$ s$^{-1}$ sr$^{-1}$]", fontsize=12)
ax1.set_xscale("log")
ax1.set_yscale("log")
ax1.set_title(f"Final Flux Comparison: Standard vs SME", fontsize=14)
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.text(0.05, 1.2, f"(Det: {detector}, Dec: {dec_deg}°, Matter: {matter}, a_mag: {a_magnitude:.1e} eV, c_mag: {c_magnitude:.1e})", transform=ax1.transAxes, fontsize=12, verticalalignment='top')

# Plot 2: Flux ratio (SME / Standard)
ratio_e = final_flux_sme[:, 0] / final_flux_std[:, 0]
ratio_mu = final_flux_sme[:, 1] / final_flux_std[:, 1]
ratio_tau = final_flux_sme[:, 2] / final_flux_std[:, 2]

ax2.plot(E_grid, ratio_e, linestyle="-", color=color_e, label=r"$\nu_e$", linewidth=2)
ax2.plot(E_grid, ratio_mu, linestyle="-", color=color_mu, label=r"$\nu_\mu$", linewidth=2)
ax2.plot(E_grid, ratio_tau, linestyle="-", color=color_tau, label=r"$\nu_\tau$", linewidth=2)
ax2.axhline(1.0, color='black', linestyle='--', alpha=0.5, linewidth=1)
ax2.set_ylabel(r'$\phi_{\text{SME}}$ / $\phi_{\text{Standard}}$', fontsize=12)
ax2.set_xlabel("E (GeV)", fontsize=12)
ax2.set_xscale("log")
# ax2.set_title("SME Effect on Final Flux", fontsize=14)
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)

fig.tight_layout()
# Save the figure
plt.savefig("tau_regeneration_flux_comparison.pdf", dpi=150)
