'''
SME neutrino oscillations plot of coszen vs energy

Simon Hilding-Nørkjær
'''

import sys, os, collections
import matplotlib.pyplot as plt
import numpy as np

from deimos.wrapper.osc_calculator import *
from deimos.utils.plotting import *
from deimos.utils.constants import *
from deimos.utils.oscillations import calc_path_length_from_coszen, get_coszen_from_path_length


#
# Main
#

if __name__ == "__main__" :

    #
    # Create model
    #

    # Choose solver
    solver = "nusquids"
    # solver = "deimos"

    # Initial definitions
    energy_range_GeV = np.geomspace(1, 1e5, num=151)
    coszen_range = np.linspace(-1,1, num=149)
    initial_flavor = 1      #mu
    nubar = False

    sme_on = True
    directional=True     #TODO Reimplement directional=False
    atmospheric=False
    interactions=False
    matter_model="vacuum" # "earth", "constant", "vacuum"

    ra_rad = 90.
    dec_rad = 0.

    basis="mass"

    # Define LIV coefficients as matrices 

    a_magnitude_eV = 2e-13 # Overall strength of a component
    c_magnitude = 1e-26 # Overall strength of c component


    flavor_structure = np.array([0.0, 0.0,1.]) # not Equally shared between flavors (diagonal)

    if directional :
        direction_structure = np.array([0., 1., 0.]) # Orient field in x direction
        a_eV = np.array([ a_magnitude_eV*n*np.diag(flavor_structure) for n in direction_structure ])
        ct = np.array([ c_magnitude*n*np.diag(flavor_structure) for n in direction_structure ])
        null_operator = np.zeros( (3, 3, 3) )

    else :
        a_eV = a_magnitude_eV * np.diag(flavor_structure)
        ct = c_magnitude * np.diag(flavor_structure)
        null_operator = np.zeros( (3,3) )



    #
    # Create calculators
    #


    kw =  {}
    if solver == "nusquids":
        kw = {"energy_nodes_GeV":energy_range_GeV,
              "interactions":interactions,}
        kw["nusquids_variant"] = "sme"


    a_calculator = OscCalculator(tool=solver,atmospheric=atmospheric,**kw)
    c_calculator = OscCalculator(tool=solver,atmospheric=atmospheric,**kw)
    ac_calculator = OscCalculator(tool=solver,atmospheric=atmospheric,**kw)
    
    a_calculator.set_matter(matter_model)
    c_calculator.set_matter(matter_model)
    ac_calculator.set_matter(matter_model)

   
    baseline = calc_path_length_from_coszen(coszen_range)

 
    # Define args to osc prob calc
    calc_kw = {
        "initial_flavor":initial_flavor,
        "nubar" : nubar,
        "energy_GeV":energy_range_GeV,
        "ra_rad":ra_rad,
        "dec_rad":dec_rad,
        # "time":time,
    }


    a_osc_prob = np.zeros((3, len(coszen_range),len(energy_range_GeV)))
    c_osc_prob = np.zeros((3, len(coszen_range),len(energy_range_GeV)))
    ac_osc_prob = np.zeros((3, len(coszen_range),len(energy_range_GeV)))

    # Simple Earth case
    a_calculator.set_sme(directional=directional,basis=basis, a_eV=a_eV, c=null_operator, ra_rad=ra_rad, dec_rad=dec_rad)
    a_osc_prob_loop = a_calculator.calc_osc_prob(distance_km = baseline, **calc_kw,)
    a_osc_prob[:,:] = a_osc_prob_loop.T       # Transpose and select energy node

    c_calculator.set_sme(directional=directional,basis=basis, a_eV=null_operator, c=ct, ra_rad=ra_rad, dec_rad=dec_rad)
    c_osc_prob_loop = c_calculator.calc_osc_prob(distance_km = baseline, **calc_kw,)
    c_osc_prob[:,:] = c_osc_prob_loop.T       # Transpose and select energy node

    ac_calculator.set_sme(directional=directional,basis=basis, a_eV=a_eV, c=ct, ra_rad=ra_rad, dec_rad=dec_rad)
    ac_osc_prob_loop = ac_calculator.calc_osc_prob(distance_km = baseline, **calc_kw,)
    ac_osc_prob[:,:] = ac_osc_prob_loop.T       # Transpose and select energy node








    #
    # Plot oscillation vs energy
    #
    P_nu_osc_title_dict = {
        0 : r"$P(\nu_{\mu} \to \nu_e)$",
        1 : r"$P(\nu_{\mu} \to \nu_{\mu})$",
        2 : r"$P(\nu_{\mu} \to \nu_{\tau})$",
    }

    fig, ax = plt.subplots(1,3, figsize=(14, 4.5), sharey=True)
    
    
    ax[0].imshow(a_osc_prob[1], origin="lower", extent=[energy_range_GeV[0], energy_range_GeV[-1], coszen_range[0], coszen_range[-1]], aspect="auto", cmap="Reds", vmin=0, vmax=1)
    ax[1].imshow(c_osc_prob[1], origin="lower", extent=[energy_range_GeV[0], energy_range_GeV[-1], coszen_range[0], coszen_range[-1]], aspect="auto", cmap="Reds", vmin=0, vmax=1)
    ax[2].imshow(ac_osc_prob[1], origin="lower", extent=[energy_range_GeV[0], energy_range_GeV[-1], coszen_range[0], coszen_range[-1]], aspect="auto", cmap="Reds", vmin=0, vmax=1)

    # ax[0].set_title(r"$|a_L^{\alpha}|=2\times10^{-13}$ eV", fontsize=16)
    # ax[1].set_title(r"$|c_L^{\alpha\beta}|=10^{-26}$", fontsize=16)
    # ax[2].set_title(r"$|a_L^{\alpha}|=2\times10^{-13}$ eV, $|c_L^{\alpha\beta}|=10^{-26}$", fontsize=16)
    ax[0].set_title(r"Only  $a_L^{\alpha}$ ", fontsize=18)
    ax[1].set_title(r"Only  $c_L^{\alpha\beta}$", fontsize=18)
    ax[2].set_title(r"Both $a_L^{\alpha}$  and  $c_L^{\alpha\beta}$", fontsize=18)
    ax[0].set_ylabel(r"$cos(\theta_{zenith})$", fontsize=18)

    for j in range(3):
        ax[j].set_xlabel("Energy [GeV]", fontsize=18)
        
        #fix the tick labels
        latex_labels = [r"$10^{}$".format(range(0,6)[i]) for i in range(6)]
        latex_labels = [r"   $1$",r"$10^{1}$", r"$10^{2}$",r"$10^{3}$",r"$10^{4}$",r"$10^{5}$  "]
        ax[j].set_xticks(ticks=[0,2e4,4e4,6e4,8e4,1e5],labels=latex_labels)
        ax[j].axvline(x=8e4, color="deepskyblue", linestyle="--", lw=2.5)
        ax[j].tick_params(axis='y', which='major', labelsize=15, rotation=0)
        ax[j].tick_params(axis='x', which='major', labelsize=16, rotation=0)


    # cbar = fig.colorbar(ax[2].images[0], ax=ax[2], orientation="vertical", pad=0.1)
    cbar_ax = fig.add_axes([0.912, 0.107, 0.025, 0.775])
    cbar = fig.colorbar(ax[1].images[0], cax=cbar_ax, orientation="vertical", fraction=0.05, pad=0.05)
    cbar.set_label(r"$P(\nu_{\mu} \to \nu_{\mu})$", fontsize=18)
    # cbar.set_label("Oscillation probability", fontsize=16)
    cbar.ax.tick_params(labelsize=16)

    fig.subplots_adjust(wspace=0.05)

    # fig.tight_layout()

    # plt.savefig("sme_comparison_coszen_vs_energy.pdf",bbox_inches='tight')




    #
    # Done
    # #

    print("")
    dump_figures_to_pdf( __file__.replace(".py",".pdf") )
