'''
Script for producing 1-dimensional tests of neutrino matter effects with the SME

Script by Simon Hilding-Nørkjær
'''



import sys, os, collections, datetime
from astropy.time import Time
import time as time_module
import numpy as np

from deimos.wrapper.osc_calculator import *
from deimos.utils.plotting import *
from deimos.utils.oscillations import * #calc_path_length_from_coszen, get_coszen_from_path_length
from deimos.utils.coordinates import * #get_right_ascension_and_declination
from deimos.utils.constants import * 

#
# Main
#

if __name__ == "__main__" :


    #
    # Define basic system
    #
    detector = "arca"     # "arca" or "dune"

    initial_flavor = 1          # numu survival
    nubar = False             # neutrino or antineutrino

    E_array_type = True
    E_GeV = np.array([1000.,10000.])
    E_node = 0

    baseline = np.linspace(0,EARTH_DIAMETER_km, num=1000)

    directional = True
    atmospheric = False
   
    a_magnitude_eV = 4e-13 # Overall strength of a component
    c_magnitude = 0#2e-26 # Overall strength of c component


    flavor_structure =    np.array([0., 0., 1.])         # numu->nutau
    field_direction_structure = np.array([0., 1., 0.])        # Orientation of field

    neutrino_offset_from_field_direction_RA_deg = 5
    neutrino_offset_from_field_direction_DEC_deg = 0

    # Choose solver (nusquids or deimos)
    solver = "nusquids"



    layer_matter_model = "layers"
    layer_matter_kwargs = {
                        "layer_endpoint_km":np.array([0.33*EARTH_DIAMETER_km, (0.66)*EARTH_DIAMETER_km, EARTH_DIAMETER_km]),
                        "matter_density_g_per_cm3":np.array([0.0, 10.0, 0.0]), 
                        "electron_fraction":np.array([0.0, 0.5, 0.0])
                        }


    const_matter_model = "constant"
    const_matter_kwargs = {"matter_density_g_per_cm3":10.0, "electron_fraction":0.5}

    vac_matter_model = "vacuum"
    vac_matter_kwargs = {}
    

    # Create calculators
    # For nuSQuIDS case, need to specify energy nodes covering full space
    kw = {}
    if solver == "nusquids" :
        kw["energy_nodes_GeV"] = E_GeV
        kw["nusquids_variant"] = "sme"


    vac_calculator =   OscCalculator(solver=solver,atmospheric=atmospheric,**kw)
    const_calculator = OscCalculator(solver=solver,atmospheric=atmospheric,**kw)
    layer_calculator = OscCalculator(solver=solver,atmospheric=atmospheric,**kw)


    if field_direction_structure[0]   != 0: field_direction_coords = (0,0)
    elif field_direction_structure[1] != 0: field_direction_coords = (90,0)
    elif field_direction_structure[2] != 0: field_direction_coords = (0,90)
    else: raise Exception("Direction structure must be a unit vector")
    direction_string = np.array(["x","y","z"])[field_direction_structure.astype(bool)]

    a_eV = np.array([ a_magnitude_eV*n*np.diag(flavor_structure) for n in field_direction_structure ])
    ct = np.array([ c_magnitude*n*np.diag(flavor_structure) for n in field_direction_structure ])

    time = "July 16, 1999, 10:30"

    print(ct,a_eV)


    #
    # MAIN LOOP
    #

    
    # Neutrino direction
    ra_deg = field_direction_coords[0] + neutrino_offset_from_field_direction_RA_deg
    dec_deg = field_direction_coords[1] + neutrino_offset_from_field_direction_DEC_deg
    ra_rad = np.deg2rad(ra_deg)
    dec_rad = np.deg2rad(dec_deg)
    neutrino_coords = (ra_deg, dec_deg)


    #
    # Calculate oscillation probabilities:
    #


    # Define args to osc prob calc
    calc_kw = {
        "initial_flavor":initial_flavor,
        "nubar" : nubar,
        "energy_GeV":E_GeV,
    }


    vac_osc_prob = np.zeros((3, len(baseline)))
    layer_osc_prob = np.zeros((3, len(baseline)))
    const_osc_prob = np.zeros((3, len(baseline)))

        # Choose basis SME operators are defined in
    sme_basis = "mass"

    from deimos.models.liv.sme import get_sme_state_matrix
    # Define "a" operator (magnitude and state texture)

    a_mu_eV = get_sme_state_matrix(p33=a_magnitude_eV)  # Only 33 element non-zero

    # Define "c" operator (magnitude and state texture)
    c_t_nu = get_sme_state_matrix(p33=c_magnitude)  # Only 33 element non-zero

    print(c_t_nu,a_mu_eV)

    # Choose direction (x, y, z)
    liv_direction = "y"

    # SME keyword arguments for set_sme_directional
    sme_kw = dict(
        basis=sme_basis,
        a_t_eV=a_mu_eV if liv_direction == "t" else None,
        a_x_eV=a_mu_eV if liv_direction == "x" else None,
        a_y_eV=a_mu_eV if liv_direction == "y" else None,
        a_z_eV=a_mu_eV if liv_direction == "z" else None,
        c_tt=c_t_nu if liv_direction == "t" else None,
        c_tx=c_t_nu if liv_direction == "x" else None,
        c_ty=c_t_nu if liv_direction == "y" else None,
        c_tz=c_t_nu if liv_direction == "z" else None,
        ra_rad=ra_rad,
        dec_rad=dec_rad,
    )

    # Remove None values (only keep the selected direction)
    sme_kw = {k: v for k, v in sme_kw.items() if v is not None}



    # Layer Earth case
    layer_calculator.set_sme_directional(**sme_kw)
    layer_calculator.set_matter(layer_matter_model, **layer_matter_kwargs)
    osc_prob_loop = layer_calculator.calc_osc_prob(distance_km=baseline, **calc_kw)
    layer_osc_prob[:,:] = osc_prob_loop[E_node].T

    # Constant Earth case  
    const_calculator.set_sme_directional(**sme_kw)
    const_calculator.set_matter(const_matter_model, **const_matter_kwargs)
    osc_prob_loop = const_calculator.calc_osc_prob(distance_km=baseline, **calc_kw)
    const_osc_prob[:,:] = osc_prob_loop[E_node].T

    # Vacuum case
    vac_calculator.set_sme_directional(**sme_kw)
    vac_calculator.set_matter(vac_matter_model, **vac_matter_kwargs)
    osc_prob_loop = vac_calculator.calc_osc_prob(distance_km=baseline, **calc_kw)
    vac_osc_prob[:,:] = osc_prob_loop[E_node].T

    # Check that probabilities sum to 1 for each data point
    assert np.isclose( np.sum(layer_osc_prob[:,:]), len(layer_osc_prob[0,:]), atol=1e-10)
    assert np.isclose( np.sum(const_osc_prob[:,:]), len(const_osc_prob[0,:]), atol=1e-10)
    assert np.isclose( np.sum(vac_osc_prob[:,:]), len(vac_osc_prob[0,:]), atol=1e-10)
    


    #
    # plot oscillation probabilities
    #

    labes = [r"$\nu_\mu \rightarrow \nu_e$", r"$\nu_\mu \rightarrow \nu_\mu$", r"$\nu_\mu \rightarrow \nu_\tau$"]

    fig, ax= plt.subplots(3,1,figsize=(4,4.5), sharex=True, sharey=True)

    ax = ax.flatten()

    ax[0].axvspan(0,EARTH_DIAMETER_km, color="cyan", alpha=0.2, label="vacuum")

    ax[0].plot(baseline, vac_osc_prob[1,:],c='k', ls="--", lw=1., alpha=0.4)#, label=f"P("+labes[1]+")")
    ax[0].plot(baseline, vac_osc_prob[2,:],c='k', ls=":", lw=1., alpha=0.4)#, label=f"P("+labes[2]+")")
    ax[0].plot(baseline, vac_osc_prob[0,:],c='k', ls="-", lw=2.5, alpha=1, label=f"P("+labes[0]+")")

    ax[1].axvspan(0,0, color="cyan", alpha=0.2, label="vacuum")
    ax[1].axvspan(0,EARTH_DIAMETER_km, color="orangered", alpha=0.4, label="Matter")
    ax[1].plot(baseline, const_osc_prob[1,:],c='k', ls="--", lw=1., alpha=0.4)#, label=f"P("+labes[1]+")")
    ax[1].plot(baseline, const_osc_prob[2,:],c='k', ls=":", lw=1., alpha=0.4)#, label=f"P("+labes[2]+")")
    ax[1].plot(baseline, const_osc_prob[0,:],c='k', ls="-", lw=2.5, alpha=1, label=f"P("+labes[0]+")")

    for i in range(3):
        ax[i].set(xlim=(baseline[0],baseline[-1]),ylim=(-0.03,1.03))
        ax[i].set_ylabel("Probability", fontsize=10)
        ax[i].tick_params(axis='both', labelsize=10)
        # ax[i].set_yticks([0,0.25,0.5,0.75,1])
        ax[i].set_yticks([0.,0.5,1.0])

        #xticks should be boundaries at 
        ax[i].set_xticks([0,0.33*EARTH_DIAMETER_km, (0.6666666)*EARTH_DIAMETER_km, EARTH_DIAMETER_km])
        ax[i].set_xticklabels([0, "1/3", "2/3", "1"])

    ax[2].axvspan(1/3*EARTH_DIAMETER_km,2/3*EARTH_DIAMETER_km, color="orangered", alpha=0.4, label="Matter")
    ax[2].axvspan(0,1/3*EARTH_DIAMETER_km, color="cyan", alpha=0.2, label="vacuum")
    ax[2].axvspan(2/3*EARTH_DIAMETER_km,EARTH_DIAMETER_km, color="cyan", alpha=0.2)
    ax[2].plot(baseline, layer_osc_prob[1,:],c='k', ls="--", lw=1., alpha=0.4, label=f"P("+labes[1]+")")
    ax[2].plot(baseline, layer_osc_prob[2,:],c='k', ls=":", lw=1., alpha=0.4, label=f"P("+labes[2]+")")
    ax[2].plot(baseline, layer_osc_prob[0,:],c='k', ls="-", lw=2.5, alpha=1, label=f"P("+labes[0]+")")
    # ax[2].legend(fontsize=9, ncol=3, loc=(-0.008,3.35))
    ax[2].set_xlabel(r"Baseline [D$_{\text{Earth}}$]", fontsize=10)

    import matplotlib.patches as mpatches
    matter_patch = mpatches.Patch(color="orangered", alpha=0.4, label="Matter")
    vacuum_patch = mpatches.Patch(color="cyan", alpha=0.2, label="vacuum")

    line1, = ax[0].plot([], [], 'k--', label=r"P($\nu_\mu \rightarrow \nu_\mu$)")
    line2, = ax[0].plot([], [], 'k:', label=r"P($\nu_\mu \rightarrow \nu_\tau$)")
    line3, = ax[0].plot([], [], 'k-', lw=2.5, label=r"P($\nu_\mu \rightarrow \nu_e$)")

    # First legend: patches (top row)
    legend1 = ax[0].legend(handles=[matter_patch, vacuum_patch], loc='upper left', bbox_to_anchor=(0.02, 1.63), ncol=2, frameon=False)

    # Second legend: lines (bottom row)
    legend2 = ax[0].legend(handles=[line1, line2, line3], loc='upper left', bbox_to_anchor=(-0.23, 1.45), ncol=3, frameon=False)

    # Add both legends to the axes
    ax[0].add_artist(legend1)

    layer1_end, layer2_end, layer3_end = layer_matter_kwargs["layer_endpoint_km"]

    fig.subplots_adjust(hspace=0.1)

    # fig.tight_layout()


    import matplotlib.patches as mpatches

    # After creating legend1 and legend2 and adding them to the axes
    fig.canvas.draw()  # Needed to get correct legend positions

    # Get bounding boxes in figure coordinates
    bbox1 = legend1.get_window_extent(fig.canvas.get_renderer())
    bbox2 = legend2.get_window_extent(fig.canvas.get_renderer())

    # Transform to figure coordinates
    bbox1_fig = bbox1.transformed(fig.transFigure.inverted())
    bbox2_fig = bbox2.transformed(fig.transFigure.inverted())

    # Combine bounding boxes
    x0 = min(bbox1_fig.x0, bbox2_fig.x0)
    y0 = min(bbox1_fig.y0, bbox2_fig.y0)
    x1 = max(bbox1_fig.x1, bbox2_fig.x1)
    y1 = max(bbox1_fig.y1, bbox2_fig.y1)

    # Draw a box around the combined area
    box = mpatches.FancyBboxPatch(
        (x0, y0), x1-x0, y1-y0,
        boxstyle="round,pad=0.0",
        edgecolor=legend1.get_frame().get_edgecolor(),
        facecolor="none",
        linewidth=legend1.get_frame().get_linewidth(),
        zorder=10,
        transform=fig.transFigure
    )

    fig.patches.append(box)

    plt.savefig(
        __file__.replace(".py",".pdf"),
        bbox_inches='tight',
        bbox_extra_artists=[legend1, legend2]
    )
