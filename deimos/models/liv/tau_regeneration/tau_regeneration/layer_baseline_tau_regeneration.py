'''
Script for producing 1-dimensional tests of neutrino matter effects with the SME

Simon Hilding-Nørkjær
'''

import sys, os, collections, datetime
from astropy.time import Time
import time as time_module
import numpy as np
import itertools
from scipy.spatial.transform import Rotation

from deimos.wrapper.osc_calculator import *
from deimos.utils.plotting import *
from deimos.utils.oscillations import * #calc_path_length_from_coszen, get_coszen_from_path_length
from deimos.utils.coordinates import * #get_right_ascension_and_declination
from deimos.utils.constants import * 


#
# Globals
#

# g/cm^3 to eV^4 = 4.310129777833244e+18

labels = [r"$\nu_\mu \rightarrow \nu_e$", r"$\nu_\mu \rightarrow \nu_\mu$", r"$\nu_\mu \rightarrow \nu_\tau$"]

# For plotting 
# layer radii  from the simple earth model: Originally from PREM paper.
inner_core_radius = 1221.5
outer_core_radius = 3480.0
mantle_radius = 5701.0
earth_radius_km = 6371.0

# Thicknesses of the layers
inner_core_thickness_km = inner_core_radius*2                               #x2 because 1221 is the radius
outer_core_thicknes_km = outer_core_radius - inner_core_radius              #3480 is the outer core radius
mantle_thickness_km = mantle_radius - outer_core_radius                     #5701 is the mantle radius
transition_and_crust_thickness_km = earth_radius_km - mantle_radius         #6371 is the earth radius

# currently transition and crust layer is treated as part of the mantle
mantle_thickness_km = mantle_thickness_km + transition_and_crust_thickness_km


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


def get_mixing_matrix(matrix, E_eV=1):
    """
    Calculates the mixing matrix and eigenvalues of a given Hamiltonian matrix.
    """
    # Rescale matrix to account for energy dependence
    matrix = matrix*2*E_eV

    # Find eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    # Sort eigenvectors based on eigenvalues
    sorted_indices = np.argsort(eigenvalues)
    sorted_eigenvectors = eigenvectors[:, sorted_indices]
    sorted_eigenvalues = eigenvalues[sorted_indices]

    return sorted_eigenvectors, sorted_eigenvalues


def diagonalize_HE(HE, deg=True):
    """
    Returns all mixing angles that diagonalize the matrix H*2*E.
    H = \Delta m^2 / 2 E
    """
    # Calculate the mixing matrix
    mixing_matrix, eigenvalues = get_mixing_matrix(HE)

    # Initialize a list to store the diagonalizing angles
    diagonalizing_angles = []
    diagonalized_matrices = []

    # Try all mixing orders
    n = mixing_matrix.shape[0]
    signs = [-1, 1]
    column_orders = list(itertools.permutations(range(n)))

    for order in column_orders:
        for sign_combination in itertools.product(signs, repeat=n):
            modified_matrix = mixing_matrix[:, order] * np.array(sign_combination)
            angles = pmns_angles_from_PMNS(matrix=modified_matrix, deg=False)

            if angles is not None:
                theta12, theta13, theta23 = angles
                
                if 0-0.01 <= theta12 <= np.pi/2+0.01 and 0-0.01 <= theta13 <= np.pi/2+0.01 and 0-0.01 <= theta23 <= np.pi/2+0.01:
                    result_array = np.dot(modified_matrix.conjugate().T, np.dot(HE, modified_matrix))
                    # Set values close to zero to zero based on the tolerance
                    tolerance = 1e-9
                    result_array[np.isclose(result_array, 0, atol=tolerance)] = 0

                    # Assert that result_array is diagonal
                    if np.allclose(result_array, np.diag(np.diagonal(result_array))):
                        diagonalizing_angles.append((theta12, theta13, theta23))
                        diagonalized_matrices.append(result_array)

        
        if len(diagonalizing_angles) > 1:
            # Sort diagonalizing angles based on negative angles
            diagonalizing_angles.sort(key=lambda angles: any(angle < 0 for angle in angles))
            # Sort diagonalizing angles based on non-zero angles
            diagonalizing_angles.sort(key=lambda angles: sum(angle != 0 for angle in angles), reverse=True)

        if deg:
            return [np.rad2deg(angles) for angles in diagonalizing_angles], diagonalized_matrices, eigenvalues
        else:
            return diagonalizing_angles, diagonalized_matrices, eigenvalues


def pmns_angles_from_PMNS(matrix, deg=True):
    """
    Returns the mixing angles.
    """
    # Convert the matrix to a rotation object
    rotation = Rotation.from_matrix(matrix)
    
    # Use canonical rotation order
    order = 'zyx'
    angles = rotation.as_euler(order, degrees=True)

    # Correct for wrong sign of theta23 and theta12 
    '''
    Code to check signs of angles
    r_z = R.from_matrix([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    r_y = R.from_matrix([[0, 0, -1], [0, 1, 0], [1, 0, 0]])
    r_x = R.from_matrix([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    r.as_euler('xyz', degrees=True)
    '''
    angles[0] = -angles[0]
    angles[2] = -angles[2]
        
    # Extract the angles of rotation around z, y, and x axes
    theta12, theta13, theta23 = angles 

    # Return mixing angles
    if deg:
        return theta12, theta13, theta23
    else:
        return np.deg2rad(theta12), np.deg2rad(theta13), np.deg2rad(theta23)
    

#
# Define parameter space
#

# Initial flux
initial_flavor = 1          # numu survival
initial_state = np.array([[[1.22867580e+06, 1.83641636e+07, 1.21435408e+03],
                                    [1.22867580e+06, 1.83641636e+07, 1.21435408e+03]],
                                  [[1.22867580e+06, 1.83641636e+07, 1.21435408e+03],
                                    [1.22867580e+06, 1.83641636e+07, 1.21435408e+03]]]
                                ) # Change calc_kw accordingly


nubar = False             # neutrino or antineutrino

# Earth model
earth_model = "simple" # "prem" or "simple"
layer_matter_model, layer_matter_kwargs = matter_models(earth_model=earth_model)
print("Matter density: ", layer_matter_kwargs["matter_density_g_per_cm3"])

# Energy values
E_array_type = True
E_GeV = np.array([10000.,20000.])
E_GeV = np.array([100.,1000.])
E_node = 0

#SME strength and direction
a_magnitude_eV = 2e-15 # Overall strength of a component
c_magnitude = 0#2e-26 # Overall strength of c component
a_eV, ct, field_direction_coords, direction_string = set_sme(a_magnitude_eV = a_magnitude_eV, 
                                           c_magnitude = c_magnitude, 
                                           flavor_structure=np.array([0., 0., 1.]), 
                                           field_direction_structure=np.array([0., 0., 1.])
                                           )

# Mixing angles
MIXING_ANGLES_rad = np.deg2rad( np.array([ 33.82, 8.60, 48.6 ]) ) 

# Mass splittings
MASS_SPLITTINGS_eV2 = np.array([ 7.39e-5, 2.528e-3 ])

#
# Neutrino direction
#

# Option1:  Neutrino direction as an offset from the field direction 

# neutrino_offset_from_field_direction_RA_deg = 0
# neutrino_offset_from_field_direction_DEC_deg = -180

# ra_rad = np.deg2rad(field_direction_coords[0] + neutrino_offset_from_field_direction_RA_deg)
# dec_rad = np.deg2rad(field_direction_coords[1] + neutrino_offset_from_field_direction_DEC_deg)

# Option2:  Pass (RA, DEC) in rad and calculate coszen based on position

# Define detector position
detector = DetectorCoords(name = "Arca")
time = "July 16, 1999, 10:30"

ra_rad = np.deg2rad(0.)
dec_rad = np.deg2rad(+89.)
# Get coszen value corresponding to neutrino direction RA/DEC 
coszen_values = np.array([detector.get_coszen_altitude_and_azimuth(ra_rad, dec_rad, time, deg=False)[0]])        




#
# Calculator settings
#

# Choose solver (nusquids or deimos)
solver = "nusquids"
sme_basis = "mass"
directional = True
atmospheric = False


#
# Main
#

if __name__ == "__main__" :

    print("Running", __file__)
    
    #
    # Define basic system
    #
    

    # cosz_deg = np.linspace(-1,0, num=149)
    # baseline = calc_path_length_from_coszen(cosz_deg)
    baseline = np.linspace(0,EARTH_DIAMETER_km, num=1000)
    

    #
    # Create calculators
    # 
    
    # For nuSQuIDS case, need to specify energy nodes covering full space
    kw = {}
    if solver == "nusquids" :
        kw["energy_nodes_GeV"] = E_GeV
        #kw["nusquids_variant"] = "sme"

    # Create calculator
    calculator = OscCalculator(tool=solver,
                               atmospheric=atmospheric,
                               mixing_angles_rad=MIXING_ANGLES_rad,
                                mass_splittings_eV2=MASS_SPLITTINGS_eV2,
                                **kw)


    #
    # MAIN LOOP
    #

    #
    # Calculate oscillation probabilities:
    #

    # Define args to osc prob calc
    calc_kw = {
        #"initial_flavor":initial_flavor,
        "initial_state":initial_state,
        "nubar" : nubar,
        "energy_GeV":E_GeV,
        #"ra_rad":ra_rad,
        #"dec_rad":dec_rad,
        # "time":time,
    }

    # Oscillation probabilities
    layer_osc_prob = np.zeros((3, len(baseline)))
    layer_osc_prob_vac = np.zeros((3, len(baseline)))

    # Set matter
    calculator.set_matter(layer_matter_model, **layer_matter_kwargs)

    # Calculate oscillation probabilities in vacuum
    osc_prob_loop_vac = calculator.calc_osc_prob(distance_km = baseline, **calc_kw)
    layer_osc_prob_vac[:,:] = osc_prob_loop_vac[E_node].T       # Transpose and select energy node
    
    kw = {}
    if solver == "nusquids" :
        kw["energy_nodes_GeV"] = E_GeV
        kw["nusquids_variant"] = "sme"

    # Create calculator
    calculator = OscCalculator(tool=solver,
                               atmospheric=atmospheric,
                               mixing_angles_rad=MIXING_ANGLES_rad,
                                mass_splittings_eV2=MASS_SPLITTINGS_eV2,
                                **kw)
    # Set SME
    calculator.set_sme(directional=directional, basis=sme_basis, a_eV=a_eV, c=ct, ra_rad=ra_rad, dec_rad=dec_rad)

    # Set matter
    calculator.set_matter(layer_matter_model, **layer_matter_kwargs)

    # Calculate oscillation probabilities with SME
    osc_prob_loop = calculator.calc_osc_prob(distance_km = baseline, **calc_kw)
    layer_osc_prob[:,:] = osc_prob_loop[E_node].T       # Transpose and select energy node
    
    # Divide propagated flux by initial flux
    #for i in range(len(baseline)):
    #    layer_osc_prob[:,i] = layer_osc_prob[:,i]/initial_state[0,0,:]
    # Check that probabilities sum to 1 for each data point
    #assert np.isclose( np.sum(layer_osc_prob[:,:]), len(layer_osc_prob[0,:]), atol=1e-10)


    #
    # Calculate mixing angles and effective mass squared differences
    #

    # Set SME again to get the Hamiltonian
    
    # Mixing angles 
    angles_over_baseline = np.zeros((3, len(layer_matter_kwargs["layer_endpoint_km"])))
    distances = np.zeros(len(layer_matter_kwargs["layer_endpoint_km"]))
    counter = 0

    # Effective eigenvalues
    eff_mass_squared_diff = np.zeros((2,len(layer_matter_kwargs["layer_endpoint_km"])))

    # Get Hamiltonian vs time (e.g. distance)
    for x in layer_matter_kwargs["layer_endpoint_km"]:

        # Always set some matter (to make sure there is a time-dependent component of the Hamiltonian)
        calculator.set_matter(matter="constant", matter_density_g_per_cm3=layer_matter_kwargs["matter_density_g_per_cm3"][counter], electron_fraction=layer_matter_kwargs["electron_fraction"][counter])

        # Set SME
        calculator.set_sme(directional=directional, basis=sme_basis, a_eV=a_eV, c=ct, ra_rad=ra_rad, dec_rad=dec_rad)

        # Get Hamiltonian at distance x
        H = calculator.nusquids.GetHamiltonianAtTime(x, 0, 0, True) # args: [x, E node, rho (nu vs nubar), flavor (True) or mass (False) basis]
        H = np.array(H[:,:,0] + H[:,:,1]*1j, dtype=np.complex128) # This properly builds the real and imaginary components
        # print("\n x = %0.2f km \n%s" % (x, H*2*E_GeV[0]*1e9))

        # Calculate all mixing angles
        all_angles, mixing_matrices, eigenvalues = diagonalize_HE(H*2*E_GeV[0]*1e9, deg=True)#
        distances[counter] = x

        # Store mixing angles
        angles_over_baseline[0,counter] = all_angles[0][0]
        angles_over_baseline[1,counter] = all_angles[0][1]
        angles_over_baseline[2,counter] = all_angles[0][2]


        # Calculate effective mass squared differences
        eff_mass_squared_diff[1,counter] = np.abs(eigenvalues[0] - eigenvalues[1])
        eff_mass_squared_diff[0,counter] = np.abs(eigenvalues[1] - eigenvalues[2])

        if len(all_angles) == 0:
            print("No angles found at distance x = %0.2f km" % x)
            continue
        
        counter += 1


    #
    # Plot oscillation probabilities
    #

    fig, axs = plt.subplots(3, 1, figsize=(7, 10), sharex=True, gridspec_kw={'height_ratios': [3, 3, 3]})
    
    #
    # Oscillation probabilities subplot
    #

    # Plot background colors of the earth layers
    axs[0].axvspan(0, EARTH_DIAMETER_km, color="yellow", alpha=0.5, label="Mantle")
    axs[0].axvspan(earth_radius_km - outer_core_radius, earth_radius_km + outer_core_radius, color="orange", alpha=0.5, label="Outer core")
    axs[0].axvspan(earth_radius_km - inner_core_radius, earth_radius_km + inner_core_radius, color="red", alpha=0.3, label="Inner core")
    axs[0].axvspan(0, EARTH_DIAMETER_km, color="white", alpha=0.5)
    for i in range(2):
        axs[i+1].axvspan(0, EARTH_DIAMETER_km, color="yellow", alpha=0.5)
        axs[i+1].axvspan(earth_radius_km - outer_core_radius, earth_radius_km + outer_core_radius, color="orange", alpha=0.5)
        axs[i+1].axvspan(earth_radius_km - inner_core_radius, earth_radius_km + inner_core_radius, color="red", alpha=0.3)
        axs[i+1].axvspan(0, EARTH_DIAMETER_km, color="white", alpha=0.5)
    
    # Plot SME flux/ normal flux
    axs[0].plot(baseline, layer_osc_prob[0, :]/layer_osc_prob_vac[0, :], c='k', label=f"P(" + labels[0] + ")", zorder=10)
    axs[0].plot(baseline, layer_osc_prob[1, :]/layer_osc_prob_vac[1, :], c='darkorchid', ls="--", lw=2, alpha=1, label=f"P(" + labels[1] + ")")
    axs[0].plot(baseline, layer_osc_prob[2, :]/layer_osc_prob_vac[2, :], c='dodgerblue', ls="-.", lw=2, alpha=1, label=f"P(" + labels[2] + ")")
    
    # Formatting
    axs[0].set(xlim=(baseline[0], baseline[-1]))#, ylim=(-0.03, 3.03))
    axs[0].set_ylabel(r"$\phi_{SME}/\phi_{vac}$", fontsize=20)
    #axs[0].set_yscale("log")
    #ax.set_xlabel("Baseline [km]", fontsize=20)
    
    if earth_model == "prem":
        axs[0].set_title("200 layer PREM Earth Model", fontsize=20)
    elif earth_model == "simple":
        axs[0].set_title("3 layer Simple Earth Model", fontsize=20)

    axs[0].tick_params(axis='both', labelsize=18, which='both', direction='in', top=True, right=True)
    axs[0].axvline(x=earth_radius_km, color="black", alpha=0.2)  # label="Center")
    axs[0].legend(fontsize=13, ncol=2, loc="best")
    

    #
    # Sin^2 of mixing angles subplot
    #
    
    distances = np.insert(distances, 0, 0)
    angles_over_baseline = np.insert(angles_over_baseline, 0, np.array([angles_over_baseline[0][0], angles_over_baseline[1][0], angles_over_baseline[2][0]]), axis=1)
    
    # Plot sin^2 of mixing angles
    ax = axs[1] 
    ax.plot(distances, (angles_over_baseline[0]), c='green', lw=2, label=r"$\theta_{12}$", ds= 'steps-pre')
    ax.plot(distances, (angles_over_baseline[1]), c='blue', lw=2, label=r"$\theta_{13}$", linestyle = "--", ds= 'steps-pre')
    ax.plot(distances, (angles_over_baseline[2]), c='red', lw=2, label=r"$\theta_{23}$", linestyle = "-.", ds= 'steps-pre')
    
    # Formatting
    ax.set(xlim=(baseline[0], baseline[-1]), ylim=(-1., 91.))
    ax.set_ylabel(r"$\theta_{ij}$", fontsize=20)
    #ax.set_xlabel("Baseline [km]", fontsize=20)

    ax.tick_params(axis='both', labelsize=18, which='both', direction='in', top=True, right=True)
    ax.axvline(x=earth_radius_km, color="black", alpha=0.2)  # label="Center")
    ax.legend(fontsize=12, loc="upper right")


    #
    # Effective mass squared differences subplot
    #
    
    eff_mass_squared_diff = np.insert(eff_mass_squared_diff, 0, np.array([eff_mass_squared_diff[0][0], eff_mass_squared_diff[1][0]]), axis=1)
    print("eff_mass_squared_diff", eff_mass_squared_diff)
    # Plot effective mass squared differences
    ax = axs[2]
    ax.plot(distances, eff_mass_squared_diff[0], c='green', lw=2, label=r"$\Delta m^2_{21}$", ds= 'steps-pre')
    ax.plot(distances, eff_mass_squared_diff[1], c='blue', lw=2, label=r"$\Delta m^2_{31}$", linestyle = "--", ds= 'steps-pre')
    
    # Formatting
    ax.set_ylabel(r"$\Delta m^2_{ij}$ [eV$^2$]", fontsize=20)
    ax.set_yscale("log")
    ax.set(xlim=(baseline[0], baseline[-1]), ylim=(1e-3, 3e-1))
    ax.set_xlabel("Baseline [km]", fontsize=20)
    ax.tick_params(axis='both', labelsize=18, which='both', direction='in', top=True, right=True)
    ax.axvline(x=earth_radius_km, color="black", alpha=0.2)  # label="Center")
    ax.legend(fontsize=12, loc="upper right")
    
    
    
    fig.suptitle("SME: a = {} eV, c = {}, Direction: -{}\nRA, DEC: ({}, {}) deg // E={}GeV".format(a_magnitude_eV, c_magnitude, direction_string, np.rad2deg(ra_rad), np.rad2deg(dec_rad), E_GeV[E_node]))

    fig.tight_layout()      


    #
    # Done
    #

    print("")
    dump_figures_to_pdf( __file__.replace(".py",".pdf") )
