'''
Comparison of oscillation probability skymaps for IceCube and ARCA in RA,DEC. 
- Atmospheric neutrinos
- Sidereal SME parameters and matter effects can be activated
- Earth layer boundaries are shown
Edited by Johann Ioannou-Nikolaides based on a script by Simon Hilding-Nørkjær
'''


import numpy as np
import time as time_module
from deimos.wrapper.osc_calculator import OscCalculator
from deimos.utils.oscillations import get_coszen_from_path_length
from deimos.utils.plotting import plt, dump_figures_to_pdf, get_intermediate_points, get_number_tex
from deimos.utils.constants import *
from deimos.models.liv.sme import get_sme_state_matrix
from deimos.models.liv.paper_plots.paper_def import *



def plot_colormap(ax, x, y, z, zlabel=None, **kw) :
    '''
    Plot a 2D colormap
    '''

    # Checks
    #TODO

    # Get corners of mesh
    x = get_intermediate_points(x, bounding_points=True)
    y = get_intermediate_points(y, bounding_points=True)
    # assert x.shape[0] == x.size - 2

    # Plot the colormesh
    cmesh = ax.pcolormesh(x, y, z.T, **kw)

    # The mesh is drawn as rectangles. If not drawing edges explicitly, make the edges the same color
    # as the face to avoid an annoying white grid.
    if not any([x in kw for x in ["edgecolor", "edgecolors"]]):
        cmesh.set_edgecolor("face")

    # Add colorbar
    fig = ax.get_figure()
    # cbar = fig.colorbar(cmesh, ax=ax, label=zlabel)
    return cmesh


#
# Main 
#
if __name__ == "__main__":

    #
    # Steering
    #
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--solver", type=str, required=False, default="deimos", help="Solver name")
    parser.add_argument("-n", "--num-points", type=int, required=False, default=25, help="Num scan points")
    args = parser.parse_args()


    #
    # Define basic system parameters
    #
    initial_flavor = 1  # 1 corresponds to numu
    final_flavor = 1
    nubar = False  # False for neutrino, True for antineutrino
    E_GeV = REF_E_GeV

    #
    # Solver settings
    #
    kw = {}
    if args.solver == "nusquids":
        kw["energy_nodes_GeV"] = E_GeV
        kw["nusquids_variant"] = "sme"

    matter = "vacuum" # "earth" or "vacuum"

    def setup_detector(detector_name):
        calculator = OscCalculator(solver=args.solver, atmospheric=True, **kw)
        calculator.set_matter("vaccum")
        calculator.set_detector(detector_name)
        return calculator
    

    # Initialize oscillation calculators for IceCube and off-axis detectors
    detectors = ["ICECUBE", "ARCA", "P_ONE", "TRIDENT", "HUNT", "GVD"]
    calculators = [setup_detector(detector) for detector in detectors]



    
    # Set SME parameters

    # Choose basis SME operators are defined in
    sme_basis = REF_SME_BASIS

    # Define "a" operator (magnitude and state texture)
    a_magnitude_eV = REF_SME_a_MAGNITUDE_eV
    a_mu_eV = get_sme_state_matrix(p33=a_magnitude_eV) # Choosing 33 element as only non-zero element in germs of flavor

    # Define "c" operator (magnitude and state texture)
    c_magnitude = 0.
    c_t_nu = get_sme_state_matrix(p33=c_magnitude) # Choosing 33 element as only non-zero element in germs of flavor


    
    # Define sky scan parameter space
    ra_values_deg = np.linspace(0.0, 360.0, num=args.num_points)
    dec_values_deg = np.linspace(-90.0, 90.0, num=args.num_points+1)
    ra_values_rad = np.deg2rad(ra_values_deg)
    dec_values_rad = np.deg2rad(dec_values_deg)
    ra_grid_rad, dec_grid_rad = np.meshgrid(ra_values_rad, dec_values_rad, indexing="ij")
    grid_shape = ra_grid_rad.shape
    ra_grid_flat_rad, dec_grid_flat_rad = ra_grid_rad.flatten(), dec_grid_rad.flatten()
    time = REF_TIME




    # Choose field directions
    field_directions = ["x", "y", "z"]

    # Create the figure and axis objects
    plt.rcParams.update({'font.size': 20})
    fig, ax = plt.subplots(len(field_directions), len(detectors),sharex=True, sharey=True, figsize=(7*len(detectors), 5*len(field_directions)))
    fig.subplots_adjust(right=0.88, wspace=0.07, hspace=0.07)

    # Title for the Plots
    fig.suptitle(fr"$E$ = {E_GeV*1e-3:.3g} TeV // Time: {time} // Matter: {matter.title()}")

    # Loop over directions
    for i, field_direction in enumerate(field_directions) :
        for j, detector in enumerate(detectors):

            progress = (i * len(detectors) + j + 1) / (len(field_directions) * len(detectors))
            print(f"\rProgress: {progress:.2%}", end=" ")


            # Configure SME params
            sme_params = { "basis":sme_basis, ("a_%s_eV"%field_direction):a_mu_eV, ("c_t%s"%field_direction):c_t_nu}
            a_label = r"$a^{%s}_{33}$" % (field_direction)
            c_label = r"$c^{t%s}_{33}$" % (field_direction)


            #
            # Calculate oscillation probabilities
            #

            t_init = time_module.time()

            # Calc osc probs for both detectors
            calc_kw = {
                "initial_flavor": initial_flavor,
                "nubar": nubar,
                "energy_GeV": E_GeV,
                "ra_rad": ra_grid_flat_rad,
                "dec_rad": dec_grid_flat_rad,
                "time": time,
                "sme_params": sme_params,
            }
            P_detector, _, _ = calculators[j].calc_osc_prob_sme_directional_atmospheric(**calc_kw)

            print(f"Panel calculation time: {(time_module.time() - t_init) / 60.0:.2f} minutes")

            #
            # Plotting
            #


            # Plot probability matrices

            zlabel = r"$%s$" % calculators[0].get_transition_prob_tex(initial_flavor, final_flavor, nubar)
            cmesh = plot_colormap(ax=ax[i,j], x=ra_values_deg, y=dec_values_deg, z=P_detector[...,final_flavor].reshape(grid_shape), zlabel=zlabel, cmap="RdPu", vmin=0., vmax=1.)


            # Labels
            sme_label = []
            if a_magnitude_eV > 0. : sme_label.append(a_label)
            if c_magnitude > 0. :    sme_label.append(c_label)
            sme_label = r"\n ".join(sme_label)


            # Setup subplots 
            ra_ticks = [0, 90, 180, 270, 360]
            dec_ticks = [-90, -45, 0, 45, 90]
            ax[i,j].set_xticks([0+4, 90, 180, 270, 360-8])
            ax[i,j].set_yticks([-90+4, -45, 0, 45, 90-4])

            # first column (ylabel and SME label)
            if j == 0:
                ax[i,j].set_ylabel("Declination [deg]")
                ax[i,j].text(-0.4, 0.5, sme_label, transform=ax[i,j].transAxes, color="white", verticalalignment='center', bbox=dict(boxstyle='round', facecolor='black'))
                ax[i,j].set_yticklabels([ "%i"%t for t in dec_ticks ])

            # first row (detector name)
            if i == 0:
                if detector == "P_ONE": detector = "P-ONE"
                ax[i,j].text(0.5, 1.1, detector, transform=ax[i,j].transAxes, color="white", va='bottom', ha="center", bbox=dict(boxstyle='round', facecolor='black'))

            # last row (xlabel)
            if i == len(field_directions) - 1:
                ax[i,j].set_xlabel("RA [deg]")
                ax[i,j].set_xticklabels([ "%i"%t for t in ra_ticks ])
                
            ax[i,j].set_xlim(ra_values_deg[0], ra_values_deg[-1])
            ax[i,j].set_ylim(dec_values_deg[0], dec_values_deg[-1])



    # Mark LIV field direction   #TODO dynamic
    # ax[idx].plot(90, 0, markerfacecolor="gold", markeredgecolor="black", marker="D", markersize=7, linestyle="None")


    # Add colorbar 
    cbar = fig.colorbar(cmesh, ax=ax, shrink=1, pad=0.01, aspect=30)
    cbar.set_label(zlabel)

   # Save the figure
    print("")
    dump_figures_to_pdf( __file__.replace(".py","_" + args.solver + ".pdf") )

    # Done