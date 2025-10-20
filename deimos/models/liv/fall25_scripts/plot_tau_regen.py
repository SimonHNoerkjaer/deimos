from deimos.models.liv.paper_plots.paper_def import *
import collections



nubar = False  # False for neutrino, True for antineutrino

# E_values_GeV = np.geomspace(1e2, 1e7, num=args.num_points)

detector = "IceCube"
ra_deg = 0.
dec_deg = +90. # Upgoing for IceCube

time = REF_TIME

matter = "earth" # "earth" or "vacuum"





import numpy as np

import matplotlib.pyplot as plt

# Load the data
data = np.load('tau_regeneration_results.npz')

E_grid = data['E_grid']
state_osc_probs_std = data['state_osc_probs_std']
state_osc_probs_sme = data['state_osc_probs_sme']

pure_e_std = state_osc_probs_std[:,0,:]
pure_mu_std = state_osc_probs_std[:,1,:]
pure_tau_std = state_osc_probs_std[:,2,:]

pure_e_sme = state_osc_probs_sme[:,0,:]
pure_mu_sme = state_osc_probs_sme[:,1,:]
pure_tau_sme = state_osc_probs_sme[:,2,:]





# # Load flux data from a file (assuming it's in a .npz format)
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




#
# Calculate final fluxes
#

# Initial fluxes for each flavor
initial_fluxes = np.stack([flux_nu_e, flux_nu_mu, flux_nu_tau], axis=1)  # Shape: (E, 3)

# Calculate final fluxes by applying oscillation probabilities
# final_flux[E, flavor_f] = sum_i (initial_flux[E, flavor_i] * P(flavor_i -> flavor_f))
final_flux_std = np.einsum('ei,eif->ef', initial_fluxes, state_osc_probs_std)
final_flux_sme = np.einsum('ei,eif->ef', initial_fluxes, state_osc_probs_sme)

#
# Plot the results
#

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Initial atmospheric flux
ax = axes[0, 0]
ax.plot(E_grid, E_grid**3 * flux_nu_e, linestyle="-", color="orange", label=r"$\nu_e$", linewidth=2)
ax.plot(E_grid, E_grid**3 * flux_nu_mu, linestyle="-", color="dodgerblue", label=r"$\nu_\mu$", linewidth=2)
ax.plot(E_grid, E_grid**3 * flux_nu_tau, linestyle="-", color="green", label=r"$\nu_\tau$", linewidth=2)
ax.set_ylabel(r"$E^3 \times$ Flux [GeV$^2$ cm$^{-2}$ s$^{-1}$ sr$^{-1}$]", fontsize=12)
ax.set_xlabel("E (GeV)", fontsize=12)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_title("Initial Atmospheric Flux (MCEq)", fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

# Plot 2: Final flux with standard oscillations
ax = axes[0, 1]
ax.plot(E_grid, E_grid**3 * final_flux_std[:, 0], linestyle="-", color="orange", label=r"$\nu_e$", linewidth=2)
ax.plot(E_grid, E_grid**3 * final_flux_std[:, 1], linestyle="-", color="dodgerblue", label=r"$\nu_\mu$", linewidth=2)
ax.plot(E_grid, E_grid**3 * final_flux_std[:, 2], linestyle="-", color="green", label=r"$\nu_\tau$", linewidth=2)
ax.set_ylabel(r"$E^3 \times$ Flux [GeV$^2$ cm$^{-2}$ s$^{-1}$ sr$^{-1}$]", fontsize=12)
ax.set_xlabel("E (GeV)", fontsize=12)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_title(f"Final Flux (Standard Oscillations)\nIceCube, Matter: Earth", fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

# Plot 3: Final flux with SME
ax = axes[1, 0]
ax.plot(E_grid, E_grid**3 * final_flux_sme[:, 0], linestyle="-", color="orange", label=r"$\nu_e$", linewidth=2)
ax.plot(E_grid, E_grid**3 * final_flux_sme[:, 1], linestyle="-", color="dodgerblue", label=r"$\nu_\mu$", linewidth=2)
ax.plot(E_grid, E_grid**3 * final_flux_sme[:, 2], linestyle="-", color="green", label=r"$\nu_\tau$", linewidth=2)
ax.set_ylabel(r"$E^3 \times$ Flux [GeV$^2$ cm$^{-2}$ s$^{-1}$ sr$^{-1}$]", fontsize=12)
ax.set_xlabel("E (GeV)", fontsize=12)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_title(f"Final Flux (SME)\n IceCube, Matter: Earth, $a_{{33}}^z$ = 2e-13 eV", fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

# Plot 4: Flux ratio (SME / Standard)
ax = axes[1, 1]
ratio_e = final_flux_sme[:, 0] / final_flux_std[:, 0]
ratio_mu = final_flux_sme[:, 1] / final_flux_std[:, 1]
ratio_tau = final_flux_sme[:, 2] / final_flux_std[:, 2]

ax.plot(E_grid, ratio_e, linestyle="-", color="orange", label=r"$\nu_e$", linewidth=2)
ax.plot(E_grid, ratio_mu, linestyle="-", color="dodgerblue", label=r"$\nu_\mu$", linewidth=2)
ax.plot(E_grid, ratio_tau, linestyle="-", color="green", label=r"$\nu_\tau$", linewidth=2)
ax.axhline(1.0, color='black', linestyle='--', alpha=0.5, linewidth=1)
ax.set_ylabel("Flux Ratio (SME / Standard)", fontsize=12)
ax.set_xlabel("E (GeV)", fontsize=12)
ax.set_xscale("log")
ax.set_title("SME Effect on Final Flux", fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

fig.tight_layout()

# Save the figure
# output_file = __file__.replace(".py", f"_{args.solver}_{matter}_flux.pdf")
plt.savefig("tau_regeneration_flux_comparison_OLD.pdf", dpi=150)


# Also save as PNG
# output_png = output_file.replace(".pdf", ".png")
# plt.savefig(output_png, dpi=150)
# print(f"Saved plot to: {output_png}")

# plt.show()
