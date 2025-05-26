import sys, os, collections

from matplotlib import cm

from deimos.wrapper.osc_calculator import *
from deimos.utils.plotting import *
from deimos.utils.constants import *
from deimos.utils.oscillations import calc_effective_osc_params_in_matter_2flav
#from deimos.models.osc.neutrino_mass_eigenstates_visual_representation import make_visual_plot_mass_eigenstates_composition


#
# Globals
#

COLORS = [ "red", "blue", "green", "orange", "purple", "magenta" ]

#
# Testing functions
#

def generate_neutrino_hamiltonian(mass_squared_diff, mixing_angles, E_eV):
    """
    Generate a 3x3 neutrino flavor Hamiltonian matrix based on mass squared differences and mixing angles.

    Parameters:
    - mass_squared_diff (tuple): Mass squared differences in eV^2.
    - mixing_angles (triple): Vacuum mixing angles in radians.

    Returns:
    - numpy.ndarray: 3x3 neutrino flavor Hamiltonian matrix.
    """
    # Check if input lists/tuples have correct length
    if len(mass_squared_diff) != 2 or len(mixing_angles) != 3:
        raise ValueError("Incorrect input dimensions. Mass squared differences and mixing angles should have length 2.")

    # Check if mixing angles are in the first quadrant
    if not np.all((0 <= np.array(mixing_angles)) & (np.array(mixing_angles) <= np.pi/2)):
        raise ValueError("Mixing angles should be in the first quadrant.")
    
    # Unpack mass squared differences and mixing angles
    delta_m21_sq, delta_m31_sq = mass_squared_diff
    #theta12, theta13, theta23 = mixing_angles

    # Construct the neutrino flavor Hamiltonian matrix
    H = np.zeros((3, 3))

    # Diagonal elements (normal mass hierarchy)
    H[1, 1] = delta_m21_sq
    H[2, 2] = delta_m31_sq
    
    U_PMNS = get_pmns_matrix(mixing_angles, dcp=0.)
    #print("U_PMNS: ", U_PMNS)

    #flavor state matrix
    H_f = 1/(2*E_eV) * np.dot(U_PMNS, np.dot(H, U_PMNS.conjugate().T))

    return H_f


def get_pmns_matrix(theta, dcp=0.) :
    """ Get the PMNS matrix (rotation from mass to flavor basis)"""

    if len(theta) == 1 :
        assert (dcp is None) or np.isclose(dcp, 0.)
        # This is just the standard unitary rotation matrix in 2D
        pmns = np.array( [ [np.cos(theta[0]),np.sin(theta[0])], [-np.sin(theta[0]),np.cos(theta[0])] ], dtype=np.complex128 )
    elif len(theta) == 3 :
        # Check if mixing angles are in the first quadrant
        #if not np.all((0 <= np.array(theta)) & (np.array(theta) <= np.pi/2)):
        #    raise ValueError("Mixing angles should be in the first quadrant.")
        
        # Using definition from https://en.wikipedia.org/wiki/Neutrino_oscillation
        pmns = np.array( [
            [   np.cos(theta[0])*np.cos(theta[1]),                                                                        np.sin(theta[0])*np.cos(theta[1]),                                                                        np.sin(theta[1])*np.exp(-1.j*dcp)  ], 
            [  -np.sin(theta[0])*np.cos(theta[2]) -np.cos(theta[0])*np.sin(theta[2])*np.sin(theta[1])*np.exp(1.j*dcp),    np.cos(theta[0])*np.cos(theta[2]) -np.sin(theta[0])*np.sin(theta[2])*np.sin(theta[1])*np.exp(1.j*dcp),    np.sin(theta[2])*np.cos(theta[1])  ], 
            [  np.sin(theta[0])*np.sin(theta[2]) -np.cos(theta[0])*np.cos(theta[2])*np.sin(theta[1])*np.exp(1.j*dcp),    -np.cos(theta[0])*np.sin(theta[2]) -np.sin(theta[0])*np.cos(theta[2])*np.sin(theta[1])*np.exp(1.j*dcp),    np.cos(theta[2])*np.cos(theta[1])  ],
        ], dtype=np.complex128 )
    else :
        raise Exception("Only 2x2 or 3x3 PMNS matrix supported")

    return pmns


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

    #
    # For testing of tau regeneration
    #

    #a_eV = np.zeros((3,3,3))
    #ct[2, :, :] = generate_neutrino_hamiltonian(mass_squared_diff=np.array([1e-4/(12000*5.06773093741e9), 1e-4/(12000*5.06773093741e9)]) ,mixing_angles=np.array([ 0, 0, 0 ]),E_eV=1)
    #a_eV[2, :, :] = generate_neutrino_hamiltonian(mass_squared_diff=np.array([1e1/(12000*5.06773093741e9), 1e1/(12000*5.06773093741e9)]) ,mixing_angles=np.array([ 0, 0, 0 ]),E_eV=1)

    return a_eV, ct, field_direction_coords


#
# Define parameter space
#

# Mixing angles
MIXING_ANGLES_rad = np.deg2rad( np.array([ 33.82, 8.60, 48.6 ]) ) 
#MIXING_ANGLES_rad = np.array([ np.pi/2-0.2, np.pi/2-0.2, np.pi/2-0.2 ])

# Mass splittings
MASS_SPLITTINGS_eV2 = np.array([ 7.39e-5, 2.528e-3 ])

# Energy values
num_scan_points = 100
E_values_GeV = np.geomspace(1e0, 1e4, num=num_scan_points) # Staying above the standard oscillations for simplicity here

# Neutrino or Antineutrino
nubar = False

#SME strength and direction
a_magnitude_eV = 2e-15
#a_magnitude_eV = 0

c_magnitude = 2e-24 
c_magnitude = 0
print("a_magnitude_eV", a_magnitude_eV)
for i in range(0, len(E_values_GeV), 10):
    E = E_values_GeV[i]
    km_to_eV = 5.06773093741e9 # [km] -> [1/eV]
    print("MASS_SPLITTINGS_eV2/2E*L at an energy of", int(E), " GeV is", MASS_SPLITTINGS_eV2 / (2 * E*1e9)*12000*km_to_eV, " and  c*E*L = ", a_magnitude_eV*12000*km_to_eV)  #c_magnitude*E*1e9*

a_eV, ct, field_direction_coords = set_sme(a_magnitude_eV = a_magnitude_eV, 
                                           c_magnitude = c_magnitude, 
                                           flavor_structure=np.array([.45, .5, 0.]),#flavor_structure=np.array([0, 0, 1.]), 
                                           field_direction_structure=np.array([0., 0., 1.])
                                           )


#
# Neutrino direction
#

# Option1:  Neutrino direction as an offset from the field direction 

neutrino_offset_from_field_direction_RA_deg = 30.
neutrino_offset_from_field_direction_DEC_deg = -89.

ra_rad = np.deg2rad(field_direction_coords[0] + neutrino_offset_from_field_direction_RA_deg)
dec_rad = np.deg2rad(field_direction_coords[1] + neutrino_offset_from_field_direction_DEC_deg)

# Option2:  Pass (RA, DEC) in rad and calculate coszen based on position

# Define detector position
# detector_lat=lat_deg, 
#             detector_long=long_deg, 
#             detector_height_m=height_m,
detector = DetectorCoords(detector_lat="-89.99 degree",#icecube
                detector_long="-63.45 degree",
                detector_height_m=-1400.)

time = "July 16, 1999, 10:30"

ra_rad = np.deg2rad(0.)
dec_rad = np.deg2rad(+89.)
# Get coszen value corresponding to neutrino direction RA/DEC 
coszen_values = np.array([detector.get_coszen_altitude_and_azimuth(ra_rad, dec_rad, time, deg=False)[0]])
#print("coszen_values", coszen_values)


#
# Define cases
#

cases = collections.OrderedDict()
cases["No interactions"] = {"interactions":False}
cases["Include interactions/regeneration"] = {"interactions":True}


# 
# Plot
#

# Loop over cases
for i_case, (case_label, case_kw) in enumerate(cases.items()) :


    #
    # Create model
    #

    # Create calculator
    calculator = OscCalculator(
        tool="nusquids",
        atmospheric=True,
        energy_nodes_GeV=E_values_GeV,
        mixing_angles_rad=MIXING_ANGLES_rad,
        mass_splittings_eV2=MASS_SPLITTINGS_eV2,
        **case_kw # This passes the interaction information to the model
    )

    # Enable Earth matter
    calculator.set_matter("earth")


    #
    # Propagate an atmospheric flux
    #

    # Note that the flux has an impact of tau/NC regeneration, since there are HE->LE transitions and so the relative rates of HE to LE neutrinos matters

    # Propagate the atmospheric flux
    initial_flux, final_flux = calculator.calc_final_flux(
        source="atmo",
        energy_GeV=E_values_GeV,
        coszen=coszen_values,
        nubar=nubar,
    )


    #
    # Reset calculator for SME
    #

    calculator = OscCalculator(
    tool="nusquids",
    atmospheric=True,
    energy_nodes_GeV=E_values_GeV,
    mixing_angles_rad=MIXING_ANGLES_rad,
    mass_splittings_eV2=MASS_SPLITTINGS_eV2,
    nusquids_variant="sme",
    **case_kw # This passes the interaction information to the model
    )

    # Enable Earth matter
    calculator.set_matter("earth")

    # Set SME
    calculator.set_sme(directional = True, 
                       basis = "mass", 
                       a_eV=a_eV, 
                       c=ct, 
                       # Neutrino direction
                       ra_rad=ra_rad, dec_rad=dec_rad)

    # Propagate the atmospheric flux with SME
    initial_flux_SME, final_flux_SME = calculator.calc_final_flux(
            source="atmo",
            energy_GeV=E_values_GeV,
            coszen=coszen_values, # Should be neglected by calculator, but make sure it matches (RA, DEC)
            nubar=nubar,
        )

    # Plot initial flux and final flux separately
    fig, axs = plt.subplots(3, 1, figsize=(7, 15), gridspec_kw={'height_ratios': [5, 5, 5]})
    fig.suptitle(case_label + " for coszen={}".format(coszen_values[0]) + "\nSME: a = {} GeV, c = {} // Neutrino beam direction (RA, DEC): ({},  {}) deg".format(a_magnitude_eV*1e-9,c_magnitude, np.rad2deg(ra_rad), np.rad2deg(dec_rad)))

    # Plot steering
    linestyles = ["-","--", ":"]

    # Loop over flavors
    for i_f in range(calculator.num_neutrinos) :
        axs[0].plot(E_values_GeV, E_values_GeV**3*initial_flux[:,0,i_f], color=NU_COLORS[i_f], linestyle=linestyles[i_f], lw=2, label=r"$%s$"%calculator.get_nu_flavor_tex(i_f, nubar=nubar))
        axs[1].plot(E_values_GeV, E_values_GeV**3*final_flux[:,0,i_f], color=NU_COLORS[i_f], linestyle=linestyles[i_f], lw=2, label=r"$%s$"%calculator.get_nu_flavor_tex(i_f, nubar=nubar))
        axs[2].plot(E_values_GeV, E_values_GeV**3*final_flux_SME[:,0,i_f], color=NU_COLORS[i_f], linestyle=linestyles[i_f], lw=2, label=r"$%s$"%calculator.get_nu_flavor_tex(i_f, nubar=nubar))
    
    # Format axes
    for ax in axs:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True)
        ax.legend(fontsize=12)
        ax.set_xlabel(r"$E$ [GeV]")
        ax.set_xlim(E_values_GeV[0], E_values_GeV[-1])
    
    # Specify labels and titles
    axs[0].set_ylabel(r"$E^3\phi_i$")    
    axs[0].set_title("Initial flux")

    axs[1].set_ylabel(r"$E^3\phi_f$")
    axs[1].set_title("Final flux w/o SME")
    
    axs[2].set_ylabel(r"$E^3\phi_f$")
    axs[2].set_title("Final flux w/ SME")

    fig.tight_layout()


    # Make figure plotting the ratio of final to initial flux
    fig, axs = plt.subplots(2, 1, figsize=(7, 10), sharex=True, gridspec_kw={'height_ratios': [5, 5]})
    
    # Standard tau regeneration
    fig.suptitle(case_label + " for coszen={}".format(coszen_values[0]) + "\nSME: a = {} GeV, c = {} //  Neutrino beam direction (RA, DEC): ({},  {}) deg".format(a_magnitude_eV*1e-9,c_magnitude, np.rad2deg(ra_rad), np.rad2deg(dec_rad)))

    # Plot steering
    linestyles = ["-","--", ":"]

    # Loop over flavors
    for i_f in range(calculator.num_neutrinos) :

        # Get the flux for this flavor. Only a single coszen value.
        assert coszen_values.size == 1

        # Subplot Standard Model
        flavor_initial_flux = initial_flux[:,0,i_f]
        flavor_final_flux = final_flux[:,0,i_f]
        ratio = flavor_final_flux / flavor_initial_flux

        # Subplot SME
        flavor_initial_flux_SME = initial_flux_SME[:,0,i_f]
        flavor_final_flux_SME = final_flux_SME[:,0,i_f]
        ratio_SME = flavor_final_flux_SME / flavor_initial_flux_SME
        assert ratio.ndim == 1

        # Plot ratios vs energy
        axs[0].plot(E_values_GeV, ratio, color=NU_COLORS[i_f], linestyle=linestyles[i_f], lw=2, label=r"$%s$"%calculator.get_nu_flavor_tex(i_f, nubar=nubar))
        axs[1].plot(E_values_GeV, ratio_SME, color=NU_COLORS[i_f], linestyle=linestyles[i_f], lw=2, label=r"$%s$"%calculator.get_nu_flavor_tex(i_f, nubar=nubar))
        
    # Format
    for ax in axs:
        ax.set_xscale("log")
        # ax.set_xlabel(r"$E$ [GeV]")
        ax.set_ylabel(r"$\phi_f / \phi_i$")
        ax.set_xlim(E_values_GeV[0], E_values_GeV[-1])
        ax.set_ylim(0., 7.1)
        ax.grid(True)
        ax.legend(fontsize=12)
    
    #Set title
    axs[0].set_title("Standard Model")
    axs[1].set_title("SME")
    
    fig.tight_layout()


    #
    # Plot relative fluxes SME/ no SME
    #

    #if case_label == "Include interactions/regeneration":
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    for i_f in range(calculator.num_neutrinos) :
        ratio_SME = final_flux_SME[:,0,i_f] / final_flux[:,0,i_f]
        ax.plot(E_values_GeV, ratio_SME, color=NU_COLORS[i_f], linestyle=linestyles[i_f], lw=2, label=r"$%s$"%calculator.get_nu_flavor_tex(i_f, nubar=nubar))
    ax.set_xscale("log")
    ax.set_xlabel(r"$E$ [GeV]")
    ax.set_ylabel(r"$\phi_{\rm SME} / \phi_{\rm SM}$")
    ax.set_xlim(E_values_GeV[0], E_values_GeV[-1])
    ax.set_ylim(0., 7.1)
    ax.grid(True)
    ax.legend(fontsize=12)
    ax.set_title("SME / SM")
    fig.tight_layout()

    #
    # Percentage of flux flavors
    #
        
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    sum_fluxes = np.sum(final_flux[:,0,:], axis=1)
    sum_fluxes_SME = np.sum(final_flux_SME[:,0,:], axis=1)
    sum_ratio = np.zeros_like(sum_fluxes)
    sum_ratio_SME = np.zeros_like(sum_fluxes_SME)
    for i_f in range(calculator.num_neutrinos):
        ratio = final_flux[:, 0, i_f] / sum_fluxes
        ratio_SME = final_flux_SME[:, 0, i_f] / sum_fluxes_SME
        sum_ratio += ratio
        sum_ratio_SME += ratio_SME
        if i_f == 0:
            ax.fill_between(E_values_GeV, 0, 1, color='blue')
            ax.fill_between(E_values_GeV, sum_ratio, 1, color='red')
        elif i_f == 2:
            ax.fill_between(E_values_GeV, sum_ratio, 1, color='green')
        else:
            ax.fill_between(E_values_GeV, sum_ratio, 1, color='green')
            ax.plot(E_values_GeV, sum_ratio_SME, color='k', linestyle=linestyles[i_f], lw=2, label=r"$%s$ (SME)"%calculator.get_nu_flavor_tex(i_f, nubar=nubar))
        ax.set_xscale("log")
        ax.set_xlabel(r"$E$ [GeV]")
        ax.set_ylabel("Final flux composition")
        ax.set_xlim(E_values_GeV[0], E_values_GeV[-1])
        ax.set_ylim(0., 1.)
        ax.grid(True)
        ax.legend(fontsize=12)
        ax.set_title("SME / SM")
        fig.tight_layout()

    #
    # Change in flux composition
    #
    if case_label == "Include interactions/regeneration":
        fluxes_int = final_flux[:,0,:]
        fluxes_SME_int = final_flux_SME[:,0,:]
        sum_ratio_int = np.zeros_like(sum_fluxes)
        sum_ratio_SME_int = np.zeros_like(sum_fluxes_SME)
    
    if case_label == "No interactions":
        fluxes_no_int = final_flux[:,0,:]
        fluxes_SME_no_int = final_flux_SME[:,0,:]
        sum_ratio_no_int = np.zeros_like(sum_fluxes)
        sum_ratio_SME_no_int = np.zeros_like(sum_fluxes_SME)

    if case_label == "Include interactions/regeneration":
        
        fig, ax = plt.subplots(2, 1, figsize=(7, 10))
        for i_f in range(calculator.num_neutrinos) :
            # Get ratios
            #print("Difference fluxes (absolute): ", fluxes_no_int[21, i_f]-fluxes_int[21, i_f])
            # print("Difference fluxes (relative): ", (fluxes_no_int[21, i_f]- fluxes_int[21, i_f])/( np.sum(fluxes_int[21,:], axis=0)))
            # print("Difference SME fluxes (absolute): ",fluxes_SME_no_int[21, i_f]-fluxes_SME_int[21, i_f])
            # print("Difference SME fluxes (relative): ",(fluxes_SME_no_int[21, i_f]-fluxes_SME_int[21, i_f])/np.sum(fluxes_SME_int[21,:], axis=0))
            ratio = fluxes_no_int[:, i_f] / np.sum(fluxes_no_int, axis=1)
            ratio_SME = fluxes_SME_no_int[:, i_f] / np.sum(fluxes_SME_no_int, axis=1)

            ratio_int = fluxes_int[:, i_f] / np.sum(fluxes_int, axis=1)
            ratio_SME_int = fluxes_SME_int[:, i_f] / np.sum(fluxes_SME_int, axis=1)
            
            # Stack ratios
            sum_ratio_no_int += ratio
            sum_ratio_SME_no_int += ratio_SME

            sum_ratio_int += ratio
            sum_ratio_SME_int += ratio_SME
            
            colors = np.array([[131, 50, 172], [255, 196, 61], [125, 116, 97]])
            # Plotting
            ax[0].plot(E_values_GeV, sum_ratio_no_int, color=NU_COLORS[i_f], linestyle='-', lw=2, label=r"$%s$ (no int)"%calculator.get_nu_flavor_tex(i_f, nubar=nubar), alpha=0.5)
            ax[0].plot(E_values_GeV, sum_ratio_int, color=colors[i_f]/255, linestyle='--', lw=2, label=r"$%s$"%calculator.get_nu_flavor_tex(i_f, nubar=nubar), alpha=0.5)
            ax[1].plot(E_values_GeV, sum_ratio_SME_no_int, color=NU_COLORS[-i_f], linestyle='-', lw=2, label=r"$%s$ (no int)"%calculator.get_nu_flavor_tex(i_f, nubar=nubar), alpha=0.5)
            ax[1].plot(E_values_GeV, sum_ratio_SME_int, color=colors[i_f]/255, linestyle='--', lw=2, label=r"$%s$"%calculator.get_nu_flavor_tex(i_f, nubar=nubar), alpha=0.5)
            
        for i in range(2):
            ax[i].set_xscale("log")
            ax[i].set_xlabel(r"$E$ [GeV]")
            ax[i].set_ylabel("Flux composition")
            ax[i].set_xlim(E_values_GeV[0], E_values_GeV[-1])
            ax[i].set_ylim(-0.1, 1.1)
            ax[i].grid(True)
            ax[i].legend(fontsize=12)

            fig.tight_layout()


    # Make figure with mass state composition
    #make_visual_plot_mass_eigenstates_composition(MASS_SPLITTINGS_eV2, MIXING_ANGLES_rad, DELTACP_rad , title="Standard Model"+case_label)
    #make_visual_plot_mass_eigenstates_composition(MASS_SPLITTINGS_eV2+[0,a_magnitude_eV], MIXING_ANGLES_rad, DELTACP_rad , title="Standard Model Extension"+case_label)


print("")
dump_figures_to_pdf( __file__.replace(".py",".pdf") )

# Done