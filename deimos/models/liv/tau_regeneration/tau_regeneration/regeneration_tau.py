import sys, os, collections
import time as time_module

from matplotlib import colors
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


from deimos.wrapper.osc_calculator import *
from deimos.utils.plotting import *
from deimos.utils.constants import *
from deimos.utils.oscillations import calc_effective_osc_params_in_matter_2flav
from deimos.utils.coordinates import *

#
# Functions
#

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

    return a_eV, ct, field_direction_coords, direction_string


#
# Define parameter space
#

# Mixing angles
MIXING_ANGLES_rad = np.deg2rad( np.array([ 33.82, 8.60, 48.6 ]) ) 

# Mass splittings
MASS_SPLITTINGS_eV2 = np.array([ 7.39e-5, 2.528e-3 ])

# Energy values
num_scan_points = 40
E_values_GeV = np.geomspace(1e2, 1e7, num=num_scan_points) # Staying above the standard oscillations for simplicity here

# Neutrino or Antineutrino
nubar = False  

# Define detector position
detector = DetectorCoords(name = "IceCube")
time = "July 16, 1999, 10:30"

# Vary neutrino direction
ra_deg_values = np.linspace(0., 360., num=1)
dec_deg_values = np.linspace(90., -90., num=10)

#
# Set SME strength and direction
#

a_magnitude_eV = 2e-15
c_magnitude = 0    
km_to_eV = 5.06773093741e9 # [km] -> [1/eV]
flavour_structure = np.array([0., 0., 1.])
#print("MASS_SPLITTINGS_eV2/2E*L at an energy of", int(E), " GeV is", MASS_SPLITTINGS_eV2 / (2 * E*1e9)*12000*km_to_eV, ", c*E*L = ", c_magnitude*E*1e9*km_to_eV, " and a*L = ",  a_magnitude_eV*12000*km_to_eV)
a_eV, ct, field_direction_coords, direction = set_sme(a_magnitude_eV = a_magnitude_eV, 
                                           c_magnitude = c_magnitude, 
                                           flavor_structure=flavour_structure, 
                                           field_direction_structure=np.array([0., 0., 1.]))


if __name__ == "__main__" :
    
    #
    # Define cases
    #

    cases = collections.OrderedDict()
    cases["No interactions"] = {"interactions":False}
    cases["Include interactions/regeneration"] = {"interactions":True}

    # Loop over cases
    for i_case, (case_label, case_kw) in enumerate(cases.items()) :


        #
        # Set Calculator 
        #

        calculator = OscCalculator(
            tool="nusquids",
            atmospheric=True,
            energy_nodes_GeV=E_values_GeV,
            mixing_angles_rad=MIXING_ANGLES_rad,
            mass_splittings_eV2=MASS_SPLITTINGS_eV2,
            **case_kw # This passes the interaction information to the model
            )

        # Set Matter
        calculator.set_matter("earth")


        t_init = time_module.time()


        #
        # Calculation of standard flux
        #

        standard_flux = np.zeros((len(E_values_GeV), calculator.num_neutrinos, len(ra_deg_values), len(dec_deg_values)))

        for i, ra_val in enumerate(ra_deg_values):
            for j, dec_val in enumerate(dec_deg_values):
                # Calculation timing, estimation and progress
                if i==0 and j==0:
                    start_time = time_module.time()
                    t = 0
                t += 1

                print("Progress: %0.2f%%" % (100.*(i*len(ra_deg_values)+j)/(2*len(dec_deg_values)*len(ra_deg_values))), end="\r")

                # Get coszen value corresponding to neutrino direction RA/DEC 
                coszen_value = np.array([detector.get_coszen_altitude_and_azimuth(ra_val, dec_val, time, deg=True)[0]])        
                            
                #Propagate the atmospheric flux w/o SME
                initial_flux, final_flux_var = calculator.calc_final_flux(
                    source="atmospheric",
                    energy_GeV=E_values_GeV,
                    coszen=coszen_value,
                    nubar=nubar,
                    )
                
                standard_flux[:, :, i, j] = final_flux_var.reshape(len(E_values_GeV), calculator.num_neutrinos)

                # Timing
                if t== 5:
                    delta_time = (time_module.time() - start_time)/5
                    print("Calculation time for one iteration: %0.4f seconds" % delta_time)
                    print("Total calculation time estimate: %0.2f minutes" % (2*delta_time*len(ra_deg_values)*len(dec_deg_values)/60.))


        #
        # Calculation of SME flux
        #
                    
        sme_flux = np.zeros((len(E_values_GeV), calculator.num_neutrinos, len(ra_deg_values), len(dec_deg_values)))

        for i, ra_val in enumerate(ra_deg_values):
            for j, dec_val in enumerate(dec_deg_values):   
                if i==0 and j==0:
                    start_time = time_module.time()
                    t = 0
                t += 1

                print("Progress: %0.2f%%" % (50 + 100.*(i*len(ra_deg_values)+j)/(2*len(dec_deg_values)*len(ra_deg_values))), end="\r")
                
                #
                # Set Calculator 
                #

                calculator = OscCalculator(
                    tool="nusquids",
                    atmospheric=True,
                    nusquids_variant="sme",
                    energy_nodes_GeV=E_values_GeV,
                    **case_kw # This passes the interaction information to the model
                    )

                # Set Matter
                calculator.set_matter("earth")

                # Set SME
                calculator.set_sme(directional = True, basis = "mass", a_eV=a_eV, c=ct, ra_rad=ra_val, dec_rad=dec_val)
        
                # Get coszen value corresponding to neutrino direction RA/DEC 
                coszen_value = np.array([detector.get_coszen_altitude_and_azimuth(ra_val, dec_val, time, deg=True)[0]])        

                # Propagate the atmospheric neutrino flux w/ SME
                _, final_flux_SME_var = calculator.calc_final_flux(
                    source="atmospheric",
                    energy_GeV=E_values_GeV,
                    coszen=coszen_value,
                    nubar=nubar,
                )

                sme_flux[:, :, i, j] = final_flux_SME_var.reshape(len(E_values_GeV), calculator.num_neutrinos)


        print("Total calculation time: %0.2f minutes" % ((time_module.time()-t_init)/60.) )

        #
        # Plotting
        #

        # # Get ratio of sme flux to standard flux
        # sme_flux_over_standard_flux = (sme_flux / standard_flux -1 ) * 1e2

        # # plot RA vs dec oscillogram of ratio of sme/standard flux for different neutrino flavors
        # linewidth = 2
        # alpha = 1

        # # Create figure with a row for each energy and a column for each neutrino flavor
        # fig, ax = plt.subplots(len(E_values_GeV), calculator.num_neutrinos, figsize=(10, 3*len(E_values_GeV)), sharex='col', sharey='row')

        # images = []
        # cbar_ax = []
        # # Loop over energies and flavors
        # for E in range(len(E_values_GeV)):
        #     for i_f in range(calculator.num_neutrinos):
        #         images.append(ax[E, i_f].imshow(sme_flux_over_standard_flux[E, i_f, :, :].T, extent=[ra_deg_values[0], ra_deg_values[-1], dec_deg_values[0], dec_deg_values[-1]], aspect="auto", cmap="RdYlBu", vmin=min(sme_flux_over_standard_flux[E, i_f, :, :].flatten()), vmax=max(sme_flux_over_standard_flux[E, i_f, :, :].flatten())))
        #         ax[E, i_f].set_title(" E={} GeV, ".format(round(E_values_GeV[E])) + r"$%s$" % calculator.get_nu_flavor_tex(i_f, nubar=nubar))
        #         # create an Axes on the right side of ax. The width of cax will be 5% of ax and the padding between cax and ax will be fixed at 0.05 inch.
        #         divider = make_axes_locatable(ax[E, i_f])
        #         cax = divider.append_axes("right", size="5%", pad=0.05)
        #         cbar_ax.append(cax)

        # vmin = min(sme_flux_over_standard_flux[:, :, :, :].flatten())
        # vmax = max(sme_flux_over_standard_flux[:, :, :, :].flatten())
        # print("Minimum and maximum values of sme_flux_over_standard_flux: ", vmin, vmax)

        # norm = colors.Normalize(vmin=vmin, vmax=vmax)
        # for im in images:
        #     im.set_norm(norm)
        #     plt.colorbar(im, cax=cbar_ax[images.index(im)], format='%.2f')#, label=r"$(\phi_{SME}/\phi_{Standard}-1)*10^4$")
            

        # # Set labels and ticks for all subplots
        # ax = ax.flatten()
        # for i in range(len(ax)):
        #     ax[i].set_xticks([0, 90, 180, 270, 360])
        #     ax[i].set_xticklabels([" 0", 90, 180, 270, "360  "])
        #     ax[i].set_yticks([-90, -45, 0, 45, 90])
        #     ax[i].tick_params(axis='both', which='major', labelsize=12)
        #     ax[i].set_xlabel("RA [deg]", fontsize=14)
        #     ax[i].set_ylabel("DEC [deg]", fontsize=14)

        # fig.tight_layout()


        
        # Export sme_flux_over_standard_flux
        np.save("sme_flux_{}.npy".format(case_label.replace("/", "").replace(" ", "_")), sme_flux)
        np.save("standard_flux_{}.npy".format(case_label.replace("/", "").replace(" ", "_")), standard_flux)
        print(case_label, " done.")

        print("")
        #dump_figures_to_pdf("regeneration_tau_{}.pdf".format(case_label.replace("/", "").replace(" ", "_")))
        #plt.close(fig)