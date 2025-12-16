'''
Script for producing 1-dimensional tests of neutrino matter effects with the SME

Simon Hilding-Nørkjær
'''



import sys, os, collections, datetime
from astropy.time import Time
import time as time_module
import numpy as np

from deimos.wrapper.osc_calculator_old import *
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
    E_GeV = np.array([10000.,20000.])
    E_node = 0

    # cosz_deg = np.linspace(-1,0, num=149)
    # baseline = calc_path_length_from_coszen(cosz_deg)
    baseline = np.linspace(0,EARTH_DIAMETER_km, num=1000)

    directional = True
    atmospheric = False
   
    a_magnitude_eV = 4e-13 # Overall strength of a component
    c_magnitude = 0#2e-26 # Overall strength of c component

    flavor_structure =    np.array([0., 0., 1.])         # numu->nutau
    field_direction_structure = np.array([0., 1., 0.])        # Orientation of field

    neutrino_offset_from_field_direction_RA_deg = 180
    neutrino_offset_from_field_direction_DEC_deg = 0

    # Choose solver (nusquids or deimos)
    solver = "nusquids"
    sme_basis = "mass"



    layer_matter_model = "layers"
    layer_matter_kwargs = {
                        "layer_endpoint_km":np.array([0.5*EARTH_DIAMETER_km, (0.5+1/4)*EARTH_DIAMETER_km, EARTH_DIAMETER_km]),
                        "matter_density_g_per_cm3":np.array([0.0, 10.0, 0.0]), 
                        "electron_fraction":np.array([0.0, 0.5, 0.0])
                        }


    const_matter_model = "constant"
    const_matter_kwargs = {"matter_density_g_per_cm3":5.0, "electron_fraction":0.5}

    vac_matter_model = "vacuum"
    vac_matter_kwargs = {}
    

    





    # Create calculators
    # Create calculators
    # For nuSQuIDS case, need to specify energy nodes covering full space
    kw = {}
    if solver == "nusquids" :
        kw["energy_nodes_GeV"] = E_GeV
        kw["nusquids_variant"] = "sme"


    vac_calculator = OscCalculator(tool=solver,atmospheric=atmospheric,**kw)
    const_calculator = OscCalculator(tool=solver,atmospheric=atmospheric,**kw)
    layer_calculator = OscCalculator(tool=solver,atmospheric=atmospheric,**kw)



    


    if field_direction_structure[0]   != 0: field_direction_coords = (0,0)
    elif field_direction_structure[1] != 0: field_direction_coords = (90,0)
    elif field_direction_structure[2] != 0: field_direction_coords = (0,90)
    else: raise Exception("Direction structure must be a unit vector")
    direction_string = np.array(["x","y","z"])[field_direction_structure.astype(bool)]

    a_eV = np.array([ a_magnitude_eV*n*np.diag(flavor_structure) for n in field_direction_structure ])
    ct = np.array([ c_magnitude*n*np.diag(flavor_structure) for n in field_direction_structure ])

    time = "July 16, 1999, 10:30"




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
        "ra_rad":ra_rad,
        "dec_rad":dec_rad,
        # "time":time,
    }


    vac_osc_prob = np.zeros((3, len(baseline)))
    layer_osc_prob = np.zeros((3, len(baseline)))
    const_osc_prob = np.zeros((3, len(baseline)))

    # layer Earth case
    layer_calculator.set_sme(directional=directional, basis=sme_basis, a_eV=a_eV, c=ct, ra_rad=ra_rad, dec_rad=dec_rad)
    layer_calculator.set_matter(layer_matter_model, **layer_matter_kwargs)
    osc_prob_loop = layer_calculator.calc_osc_prob(distance_km = baseline, **calc_kw)
    layer_osc_prob[:,:] = osc_prob_loop[E_node].T       # Transpose and select energy node

    # Constant Earth case  
    const_calculator.set_sme(directional=directional, basis=sme_basis, a_eV=a_eV, c=ct, ra_rad=ra_rad, dec_rad=dec_rad)
    const_calculator.set_matter(const_matter_model, **const_matter_kwargs)
    osc_prob_loop = const_calculator.calc_osc_prob(distance_km = baseline, **calc_kw)
    const_osc_prob[:,:] = osc_prob_loop[E_node].T       # Transpose and select energy node

    # Vacuum case
    vac_calculator.set_sme(directional=directional, basis=sme_basis, a_eV=a_eV, c=ct, ra_rad=ra_rad, dec_rad=dec_rad)
    vac_calculator.set_matter(vac_matter_model, **vac_matter_kwargs)
    osc_prob_loop = vac_calculator.calc_osc_prob(distance_km = baseline, **calc_kw)
    vac_osc_prob[:,:] = osc_prob_loop[E_node].T       # Transpose and select energy node





    # Check that probabilities sum to 1 for each data point
    assert np.isclose( np.sum(layer_osc_prob[:,:]), len(layer_osc_prob[0,:]), atol=1e-10)
    assert np.isclose( np.sum(const_osc_prob[:,:]), len(const_osc_prob[0,:]), atol=1e-10)
    assert np.isclose( np.sum(vac_osc_prob[:,:]), len(vac_osc_prob[0,:]), atol=1e-10)
    


    #
    # plot oscillation probabilities
    #

    labes = [r"$\nu_\mu \rightarrow \nu_e$", r"$\nu_\mu \rightarrow \nu_\mu$", r"$\nu_\mu \rightarrow \nu_\tau$"]





    fig, ax= plt.subplots(1,1,figsize=(4,1.5), sharex=True, sharey=True)

    # ax = ax.flatten()

    # ax[0].axvspan(0,EARTH_DIAMETER_km, color="cyan", alpha=0.2, label="vacuum")

    # ax[0].plot(baseline, vac_osc_prob[1,:],c='k', ls="--", lw=1., alpha=0.4)#, label=f"P("+labes[1]+")")
    # ax[0].plot(baseline, vac_osc_prob[2,:],c='k', ls="-", lw=1., alpha=0.4)#, label=f"P("+labes[2]+")")
    # ax[0].plot(baseline, vac_osc_prob[0,:],c='k', ls="-", lw=2.5, alpha=1, label=f"P("+labes[0]+")")

    # ax[1].axvspan(0,0, color="cyan", alpha=0.2, label="vacuum")
    # ax[1].axvspan(0,EARTH_DIAMETER_km, color="orangered", alpha=0.4, label="Matter")
    # ax[1].plot(baseline, const_osc_prob[1,:],c='k', ls="--", lw=1., alpha=0.4)#, label=f"P("+labes[1]+")")
    # ax[1].plot(baseline, const_osc_prob[2,:],c='k', ls="-", lw=1., alpha=0.4)#, label=f"P("+labes[2]+")")
    # ax[1].plot(baseline, const_osc_prob[0,:],c='k', ls="-", lw=2.5, alpha=1, label=f"P("+labes[0]+")")

    # ax[1].legend(fontsize=9, ncol=5, loc=(-0.008,2.15))
    # ax[1].set_xlabel("Baseline [km]", fontsize=10)



    # for i in range(2):
    #     ax[i].set(xlim=(baseline[0],baseline[-1]),ylim=(-0.03,1.03))
    #     ax[i].set_ylabel("Probability", fontsize=10)
    #     ax[i].tick_params(axis='both', labelsize=9)
    #     ax[i].set_yticks([0,0.25,0.5,0.75,1])

    # ax[2].axvspan(1/3*EARTH_DIAMETER_km,2/3*EARTH_DIAMETER_km, color="orangered", alpha=0.4, label="Matter")
    # ax[2].axvspan(0,1/3*EARTH_DIAMETER_km, color="cyan", alpha=0.2, label="vacuum")
    # ax[2].axvspan(2/3*EARTH_DIAMETER_km,EARTH_DIAMETER_km, color="cyan", alpha=0.2)
    # ax[2].plot(baseline, layer_osc_prob[1,:],c='k', ls="--", lw=1., alpha=0.4)#, label=f"P("+labes[1]+")")
    # ax[2].plot(baseline, layer_osc_prob[2,:],c='k', ls="-", lw=1., alpha=0.4)#, label=f"P("+labes[2]+")")
    # ax[2].plot(baseline, layer_osc_prob[0,:],c='k', ls="-", lw=2.5, alpha=1, label=f"P("+labes[0]+")")
    # ax[2].legend(fontsize=9, ncol=5, loc=(-0.008,3.25))
    # ax[2].set_xlabel("Baseline [km]", fontsize=10)

    layer1_end, layer2_end, layer3_end = layer_matter_kwargs["layer_endpoint_km"]

    ax.axvspan(0,layer1_end, color="cyan", alpha=0.2, label="vacuum")
    ax.axvspan(layer1_end,layer2_end, color="orangered", alpha=0.4, label="Matter")
    ax.axvspan(layer2_end,layer3_end, color="cyan", alpha=0.2)
    ax.plot(baseline, layer_osc_prob[1,:],c='k', ls="--", lw=1., alpha=0.4)#, label=f"P("+labes[1]+")")
    ax.plot(baseline, layer_osc_prob[2,:],c='k', ls="-", lw=1., alpha=0.4)#, label=f"P("+labes[2]+")")
    ax.plot(baseline, layer_osc_prob[0,:],c='k', ls="-", lw=2.5, alpha=1, label=f"P("+labes[0]+")")
    ax.legend(fontsize=9, ncol=5, loc=(-0.008,1.05))
    ax.set_xlabel("Baseline [km]", fontsize=10)
    ax.set(xlim=(baseline[0],baseline[-1]),ylim=(-0.03,1.03))
    ax.set_ylabel("Probability", fontsize=10)
    ax.tick_params(axis='both', labelsize=9)
    ax.set_yticks([0,0.25,0.5,0.75,1])

    # fig.suptitle("SME: a = {} eV, c = {} // LIV-offset:{}deg // E={}GeV".format(a_magnitude_eV, c_magnitude, neutrino_offset_from_field_direction_RA_deg, E_GeV[E_node] ), fontsize=8)

    fig.subplots_adjust(hspace=0.1)

    # fig.tight_layout()


    

    # fig , axs = plt.subplots(3, 1, figsize=(12,12))

    # for i , ax in enumerate(axs):
    #     ax.axvspan(0,EARTH_DIAMETER_km, color="yellow", alpha=0.2, label="Mantle")
    #     ax.axvspan(earth_radius_km-outer_core_radius,earth_radius_km+outer_core_radius, color="orange", alpha=0.2, label="Outer core")
    #     ax.axvspan(earth_radius_km-inner_core_radius,earth_radius_km+inner_core_radius, color="red", alpha=0.2, label="Inner core")
    #     ax.plot(baseline, layer_osc_prob[i,:],c='k', lw=2, label=f"P("+labes[i]+")")
    #     ax.set(xlim=(baseline[0],baseline[-1]), ylim=(-0.03,1.03), ylabel="Oscillation Probability", xlabel="Baseline [km]")
    #     ax.set_title("Simple Earth")
    #     ax.axvline(x=earth_radius_km, color="black", alpha=0.4, label="Center")
    #     ax.legend(fontsize=8)

    




    plt.savefig(__file__.replace(".py",".pdf"), bbox_inches='tight')




  



    #
    # Done
    #

    print("")
    # dump_figures_to_pdf( __file__.replace(".py",".pdf") )
