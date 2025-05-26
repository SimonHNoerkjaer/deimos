'''
SME neutrino oscillations plot of coszen vs energy

'''

import matplotlib.pyplot as plt
import numpy as np
import argparse

from deimos.wrapper.osc_calculator import OscCalculator
from deimos.utils.plotting import *
from deimos.utils.constants import *
from deimos.utils.oscillations import calc_path_length_from_coszen
from deimos.models.liv.sme import get_sme_state_matrix

#
# Functions
#

# Can be removed
def matter_models(earth_model="prem"):
    # Load EARTH_MODEL_PREM.dat file and extract layer endpoints, matter densities and electron fractions
    # scale layer endpoints to km and mirror the arrays to get the full earth model as a function of baseline

    layer_endpoint_array_as_fraction_of_earth_radius = np.loadtxt("EARTH_MODEL_PREM.dat", usecols=0)
    matter_density_g_per_cm3_array = np.loadtxt("EARTH_MODEL_PREM.dat", usecols=1)
    electron_fraction_array = np.loadtxt("EARTH_MODEL_PREM.dat", usecols=2)

    layer_endpoint_km_array = layer_endpoint_array_as_fraction_of_earth_radius*EARTH_RADIUS_km
    layer_endpoint_km_array = np.append(layer_endpoint_km_array, EARTH_RADIUS_km+layer_endpoint_km_array)

    matter_density_g_per_cm3_array = np.flip(matter_density_g_per_cm3_array)
    matter_density_g_per_cm3_array = np.append(matter_density_g_per_cm3_array, np.flip(matter_density_g_per_cm3_array))

    electron_fraction_array = np.flip(electron_fraction_array)
    electron_fraction_array = np.append(electron_fraction_array, np.flip(electron_fraction_array))

    #
    # Definition of the simple earth model (3 layers)
    #

    if earth_model == "simple":
            
        inner_core_thickness_km = 1221.5*2          #x2 because 1221 is the radius
        outer_core_thicknes_km = 3480.0-1221.5      #3480 is the outer core radius
        mantle_thickness_km = 5701.0-3480.0         #5701 is the mantle radius
        transition_and_crust_thickness_km = 6371.0-5701.0 #6371 is the earth radius

        #for simplicity is the transition and crust layer treated as part of the mantle (quite thin layers) #TODO maybe implement layers for both transition and crust, as densities vary a lot
        mantle_thickness_km = mantle_thickness_km + transition_and_crust_thickness_km

        # define the matter densities (g/cm3) and electron fractions for each of the earths layers

        mask_mantle = layer_endpoint_km_array < mantle_thickness_km
        mask_outer_core = (layer_endpoint_km_array > mantle_thickness_km) & (layer_endpoint_km_array < mantle_thickness_km+outer_core_thicknes_km)
        mask_inner_core = (layer_endpoint_km_array > mantle_thickness_km+outer_core_thicknes_km) & (layer_endpoint_km_array < mantle_thickness_km+outer_core_thicknes_km+inner_core_thickness_km)
        
        matter_density_mantle = np.average(matter_density_g_per_cm3_array[mask_mantle])
        matter_density_outer_core = np.average(matter_density_g_per_cm3_array[mask_outer_core])
        matter_density_inner_core = np.average(matter_density_g_per_cm3_array[mask_inner_core])



        layer_endpoint_km_array = np.array([ mantle_thickness_km, 
                                            mantle_thickness_km+outer_core_thicknes_km, 
                                            mantle_thickness_km+outer_core_thicknes_km+inner_core_thickness_km, 
                                            mantle_thickness_km+outer_core_thicknes_km+inner_core_thickness_km+outer_core_thicknes_km, 
                                            mantle_thickness_km+outer_core_thicknes_km+inner_core_thickness_km+outer_core_thicknes_km+mantle_thickness_km])
        matter_density_g_per_cm3_array = np.array([matter_density_mantle, matter_density_outer_core, matter_density_inner_core, matter_density_outer_core, matter_density_mantle])
        electron_fraction_array = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
        

    #
    # Define layer matter model
    #
        
    layer_matter_model = "layers"
    layer_matter_kwargs = {
                        "layer_endpoint_km":layer_endpoint_km_array,
                        "matter_density_g_per_cm3":matter_density_g_per_cm3_array, 
                        "electron_fraction":electron_fraction_array
                        }
    
    return layer_matter_model, layer_matter_kwargs

def set_sme(a_magnitude_eV = 2e-13, c_magnitude = 0, flavor_structure = np.array([0., 0., 1.]), field_direction_structure = np.array([0., 1., 0.])):
   
    """
    Generate SME matrices for a given magnitude and direction
    """

    if not isinstance(a_magnitude_eV, (float, int)) or not isinstance(c_magnitude, (float, int)):
        raise Exception("a_magnitude_eV and c_magnitude should be float or int values.")

    if len(flavor_structure) != 3 or len(field_direction_structure) != 3:
        raise Exception("flavor_structure (e, mu, tau) and field_direction_structure (x, y, z) should be (1,3) arrays.")

    if field_direction_structure[0]   != 0: field_direction_coords = (0,0)
    elif field_direction_structure[1] != 0: field_direction_coords = (90,0)
    elif field_direction_structure[2] != 0: field_direction_coords = (0,90)
    else: raise Exception("Direction structure must be a unit vector")
    direction_string = np.array(["x","y","z"])[field_direction_structure.astype(bool)]

    a_eV = np.array([ a_magnitude_eV*n*np.diag(flavor_structure) for n in field_direction_structure ])
    ct = np.array([ c_magnitude*n*np.diag(flavor_structure) for n in field_direction_structure ])

    return a_eV, ct, field_direction_coords

#
# Steering
#


solver = "nusquids" # "nusquids" or "deimos
parser = argparse.ArgumentParser()
parser.add_argument("-s", "--solver", type=str, required=False, default=solver, help="Solver name")
args = parser.parse_args()

#
# Basic system
#

# Neutrino or antineutrino
initial_flavor, nubar = 1, False
normal_ordering = True
atmospheric = True

# neutrino direction
ra_rad = np.pi/2 #np.pi/4
dec_rad = 0.

if normal_ordering:
    MASS_SPLITTINGS_eV = np.array([ 7.39e-5, 2.528e-3 ])
else:
    MASS_SPLITTINGS_eV = np.array([ 7.39e-5, -2.512e-3 ])


# Earth model
earth_model = "prem" # "prem" or "simple"
# layer_matter_model, layer_matter_kwargs = matter_models(earth_model=earth_model)

# Solver settings
sme_on = True
directional=True     #TODO Reimplement directional=False
interactions=False
sme_basis="mass"
matter_model="earth" # "earth" only an option in atmospheric, "constant", "vacuum"


# Initial definitions neutrinos
energy_range_GeV = np.geomspace(1, 1e5, num=250)
coszen_range = np.linspace(-1,1, num=250)


baseline = calc_path_length_from_coszen(coszen_range) # for not atmospheric





# Define LIV coefficients as matrices 
a_magnitude_eV = 0 # Overall strength of a component
a_y_eV = get_sme_state_matrix(p33=a_magnitude_eV) # Choosing 33 element as only non-zero element in germs of flavor, and +y direction

c_magnitude = 1e-26 # Overall strength of c component
c_ty = get_sme_state_matrix(p33=c_magnitude) # Choosing 33 element as only non-zero element in germs of flavor, and ty tensor component

# Group as an arg
sme_params = { "basis":sme_basis, "a_y_eV":a_y_eV, "c_ty":c_ty}

# Labels for label
a_label = r"$a^y_{33}$ = %0.3g eV" % a_magnitude_eV
c_label = r"$c^{yt}_{33}$ = %0.3g" % c_magnitude


flavor_structure = np.array([0.0, 0.0, 1.]) # not Equally shared between flavors (diagonal)
direction_structure = np.array([0., 1., 0.]) # Orient field in y-direction



# Plotting considerations
if nubar:
    P_nu_osc_title_dict = {
        0 : r"$P(\bar{\nu}_{\mu} \to \bar{\nu}_e)$",
        1 : r"$P(\bar{\nu}_{\mu} \to \bar{\nu}_{\mu})$",
        2 : r"$P(\bar{\nu}_{\mu} \to \bar{\nu}_{\tau})$",
    }

else:    
    P_nu_osc_title_dict = {
        0 : r"$P(\nu_{\mu} \to \nu_e)$",
        1 : r"$P(\nu_{\mu} \to \nu_{\mu})$",
        2 : r"$P(\nu_{\mu} \to \nu_{\tau})$",
    }


#
# Main
#

if __name__ == "__main__" :

    #
    # Create calculators
    #

    kw =  {}
    if solver == "nusquids":
        kw = {"energy_nodes_GeV":energy_range_GeV,
            "interactions":interactions,}
        kw["nusquids_variant"] = "sme"


    calculator = OscCalculator(
        solver=args.solver,
        atmospheric=True,
        mass_splittings_eV2 = MASS_SPLITTINGS_eV,
        **kw
        )

    calc_kw = {
        "initial_flavor":initial_flavor,
        "nubar" : nubar,
        "energy_GeV":energy_range_GeV,
        #"ra_rad":ra_rad,
        #"dec_rad":dec_rad,
    }

    osc_prob = np.zeros((3, len(coszen_range),len(energy_range_GeV)))

    # Simple Earth case
    calculator.set_sme_directional(basis=sme_basis, a_y_eV=a_y_eV, c_ty=c_ty, ra_rad=ra_rad, dec_rad=dec_rad)
    #calculator.set_matter(matter_model)
    calculator.set_matter("vacuum")

    if atmospheric:
        osc_prob_loop = calculator.calc_osc_prob(coszen = coszen_range, **calc_kw,)
    else:
        osc_prob_loop = calculator.calc_osc_prob(distance_km = baseline, **calc_kw,)
    osc_prob[:,:] = osc_prob_loop.T       # Transpose and select energy node



    #
    # Plot oscillation vs energy
    #

    fig, ax = plt.subplots(1,3, figsize=(16, 4.5), sharey=False)
    if normal_ordering:
        fig.suptitle("Normal ordering", fontsize=18)
    else:
        fig.suptitle("Inverted ordering", fontsize=18)
    for i in range(3):
        ax[i].imshow(osc_prob[i], origin="lower", extent=[energy_range_GeV[0], energy_range_GeV[-1], coszen_range[0], coszen_range[-1]], aspect="auto", cmap="Reds", vmin=0, vmax=1)
        ax[i].set_xlabel("Energy [GeV]", fontsize=16)
        ax[i].set_ylabel(r"$cos(\theta_{zenith})$", fontsize=16)
        ax[i].set_title(P_nu_osc_title_dict[i], fontsize=16)
        #fix the tick labels
        latex_labels = [r"$10^{}$".format(range(0,6)[i]) for i in range(6)]
        ax[i].set_xticks(ticks=[0,2e4,4e4,6e4,8e4,1e5],labels=latex_labels)
        #ax[i].axvline(x=8e4, color="dodgerblue", linestyle="--", lw=2)


        # Add colorbar
        cbar = fig.colorbar(ax[i].images[0], ax=ax[i], orientation="vertical", pad=0.1)
        # cbar.set_label("Oscillation probability", fontsize=16)
        cbar.ax.tick_params(labelsize=14)


    fig.tight_layout()






    #
    # Done
    #

    print("")
    dump_figures_to_pdf( __file__.replace(".py","_" + solver + ".pdf") )
