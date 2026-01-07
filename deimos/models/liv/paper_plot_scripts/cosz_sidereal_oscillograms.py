'''
Compare IceCube with an off-axis detector in neutrino RA,DEC oscillograms for probabilities with sidereal SME parameters and matter effects activated

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
    # Steering
    #
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--solver", type=str, required=False, default="deimos", help="Solver name")
    parser.add_argument("-n", "--num-points", type=int, required=False, default=25, help="Num scan point")
    args = parser.parse_args()


    #
    # Define basic system
    #
    off_axis_detector = "Arca"     # "arca" or "dune" or "equator"

    initial_flavor = 1          # numu survival
    nubar = False             # neutrino or antineutrino

    E_array_type =True
    E_GeV = 1e3

    atmospheric = True
    sme_basis = "mass"


    # Create calculators
    # For nuSQuIDS case, need to specify energy nodes covering full space
    kw = {}
    if args.solver == "nusquids" :
        kw["energy_nodes_GeV"] = E_GeV
        kw["nusquids_variant"] = "sme"

    # Initialize oscillation calculators for IceCube and off-axis detectors
    IC_calculator = OscCalculator(solver=args.solver, atmospheric=True, **kw)
    Off_axis_calculator = OscCalculator(solver=args.solver, atmospheric=True, **kw)


    # IC_calculator.set_matter(matter_model, matter_kwargs)
    IC_calculator.set_matter("vacuum")
    IC_calculator.set_detector("icecube")

    Off_axis_calculator.set_matter("vacuum")
    Off_axis_calculator.set_detector(off_axis_detector)






############################# EARTH LAYER BOUNDARIES #########################################

    # Define RA/dec grid (0.0587s per point)  208*202*0.0587s = 41min 7s
    ra_values_deg = np.linspace(0., 360., num=args.num_points)
    dec_values_deg = np.linspace(-90., 90., num=args.num_points)
    ra_values_rad = np.deg2rad(ra_values_deg)
    dec_values_rad = np.deg2rad(dec_values_deg)
    time = "July 16, 1999, 10:00"

    # horizontal lines
    azimuth = np.linspace(0,360,1000)
    IC_RA_horizon, IC_DEC_horizon = IC_calculator.detector_coords.get_right_ascension_and_declination(0,azimuth,time)
    Off_axis_RA_horizon, Off_axis_DEC_horizon = Off_axis_calculator.detector_coords.get_right_ascension_and_declination(0,azimuth,time)

    # Order detector horizon arrays as RA-ascending for off-axis detector
    Off_axis_indices=np.argsort(Off_axis_RA_horizon)
    Off_axis_RA_horizon = Off_axis_RA_horizon[Off_axis_indices]
    Off_axis_DEC_horizon = Off_axis_DEC_horizon[Off_axis_indices]

    # layer radii and thicknesses from the simple earth model: Originally from PREM paper.
    inner_core_radius = 1221.5
    outer_core_radius = 3480.0
    mantle_radius = 5701.0
    earth_radius_km = 6371.0

    inner_core_thickness_km = inner_core_radius*2                               #x2 because 1221 is the radius
    outer_core_thicknes_km = outer_core_radius - inner_core_radius              #3480 is the outer core radius
    mantle_thickness_km = mantle_radius - outer_core_radius                     #5701 is the mantle radius
    transition_and_crust_thickness_km = earth_radius_km - mantle_radius         #6371 is the earth radius

    # currently transition and crust layer is treated as part of the mantle
    mantle_thickness_km = mantle_thickness_km + transition_and_crust_thickness_km


    # Earth core boundary lines (inner core radius 1216km. https://www.ucl.ac.uk/seismin/explore/Earth.html) 
    pathlength_inner_core = 2 * np.sqrt(EARTH_RADIUS_km**2 - EARTH_INNER_CORE_RADIUS_km**2)
    cosz_inner_core = get_coszen_from_path_length(pathlength_inner_core)
    IC_RA_core, IC_DEC_core = IC_calculator.detector_coords.get_right_ascension_and_declination(cosz_inner_core,azimuth,time) #Wrong sign in coord transformation
    Off_axis_RA_core, Off_axis_DEC_core = Off_axis_calculator.detector_coords.get_right_ascension_and_declination(cosz_inner_core,azimuth,time)

    # Earth Outer core boundary lines (outer core radius 3486km)
    pathlength_outer_core = 2 * np.sqrt(EARTH_RADIUS_km**2 - EARTH_OUTER_CORE_RADIUS_km**2)
    cosz_outer_core = get_coszen_from_path_length(pathlength_outer_core)
    IC_RA_outer_core, IC_DEC_outer_core = IC_calculator.detector_coords.get_right_ascension_and_declination(cosz_outer_core,azimuth,time)
    Off_axis_RA_outer_core, Off_axis_DEC_outer_core = Off_axis_calculator.detector_coords.get_right_ascension_and_declination(cosz_outer_core,azimuth,time)

    # Earth mantle boundary lines (mantle radius 5701km)
    pathlength_mantle = 2 * np.sqrt(EARTH_RADIUS_km**2 - EARTH_MANTLE_RADIUS_km**2)
    cosz_mantle = get_coszen_from_path_length(pathlength_mantle)
    IC_RA_mantle, IC_DEC_mantle = IC_calculator.detector_coords.get_right_ascension_and_declination(cosz_mantle,azimuth,time)
    Off_axis_RA_mantle, Off_axis_DEC_mantle = Off_axis_calculator.detector_coords.get_right_ascension_and_declination(cosz_mantle,azimuth,time)

    # Order detector mantle arrays as RA-ascending for off-axis detector
    Off_axis_indices=np.argsort(Off_axis_RA_mantle)
    Off_axis_RA_mantle = Off_axis_RA_mantle[Off_axis_indices]
    Off_axis_DEC_mantle = Off_axis_DEC_mantle[Off_axis_indices]

    # Convert to degrees
    IC_RA_core, IC_DEC_core = np.rad2deg(IC_RA_core), np.rad2deg(IC_DEC_core)
    IC_RA_outer_core, IC_DEC_outer_core = np.rad2deg(IC_RA_outer_core), np.rad2deg(IC_DEC_outer_core)
    IC_RA_mantle, IC_DEC_mantle = np.rad2deg(IC_RA_mantle), np.rad2deg(IC_DEC_mantle)
    IC_RA_horizon, IC_DEC_horizon = np.rad2deg(IC_RA_horizon), np.rad2deg(IC_DEC_horizon)

    Off_axis_RA_core, Off_axis_DEC_core = np.rad2deg(Off_axis_RA_core), np.rad2deg(Off_axis_DEC_core)
    Off_axis_RA_outer_core, Off_axis_DEC_outer_core = np.rad2deg(Off_axis_RA_outer_core), np.rad2deg(Off_axis_DEC_outer_core)
    Off_axis_RA_mantle, Off_axis_DEC_mantle = np.rad2deg(Off_axis_RA_mantle), np.rad2deg(Off_axis_DEC_mantle)
    Off_axis_RA_horizon, Off_axis_DEC_horizon = np.rad2deg(Off_axis_RA_horizon), np.rad2deg(Off_axis_DEC_horizon)





    #
    # MAIN LOOP
    #

    ra_dec_shape = (len(dec_values_deg), len(ra_values_deg))
    cosz_IC, cosz_Off_axis = np.zeros(ra_dec_shape), np.zeros(ra_dec_shape)

    t_init = time_module.time()

    # Loop over dec
    for i , dec_rad in enumerate(dec_values_rad) : 

        # Loop over RA
        for j, ra_rad in enumerate(ra_values_rad) :

            # Calculation timing, estimation and progress
            if i==0 and j==0:
                start_time = time_module.time()
                t = 0
            t += 1

            print("Progress: %0.2f%%" % (100.*(i*len(ra_values_rad)+j)/(len(dec_values_rad)*len(ra_values_rad))), end="\r")



            #
            # Calculate oscillation probabilities:
            #


            # Get coszen/azimuth from RA/dec
            IC_cosz_value, _, IC_azimuth_deg = IC_calculator.detector_coords.get_coszen_altitude_and_azimuth(ra_rad=ra_rad, dec_rad=dec_rad, time=time)
            Off_axis_cosz_value, _, Off_axis_azimuth_deg = Off_axis_calculator.detector_coords.get_coszen_altitude_and_azimuth(ra_rad=ra_rad, dec_rad=dec_rad, time=time)
            
            # # Save coszen values
            cosz_IC[i,j] = IC_cosz_value
            cosz_Off_axis[i,j] = Off_axis_cosz_value

            # Timing
            if t== 100:
                delta_time = (time_module.time() - start_time)/100.
                print("Calculation time for one iteration: %0.4f seconds" % delta_time)
                print("Total calculation time estimate: %0.2f minutes" % (delta_time*len(dec_values_deg)*len(ra_values_deg)/60.))

    print("Total calculation time: %0.2f minutes" % ((time_module.time()-t_init)/60.) )



    
    # # plot RA vs dec oscillogram
    linewidth = 2
    alpha = 1

    
    # plot RA vs dec oscillogram

    fig, ax = plt.subplots(1,2, figsize=(6,3), sharex=True, sharey=True)
    ax = ax.flatten()

    # fig.suptitle( r"Sidereal dependence of $cos(\theta_{zenith})$ in RA,DEC-oscillogram"+f", Time: {time} UTC", fontsize=12 )
    ax[0].imshow(cosz_IC[:,:], origin="lower", extent=[ra_values_deg[0], ra_values_deg[-1], dec_values_deg[0], dec_values_deg[-1]], aspect="auto", cmap="RdYlBu", vmin=-1., vmax=1.)
    ax[1].imshow(cosz_Off_axis[:,:], origin="lower", extent=[ra_values_deg[0], ra_values_deg[-1], dec_values_deg[0], dec_values_deg[-1]], aspect="auto", cmap="RdYlBu", vmin=-1., vmax=1.)

    #plot horizontal lines 
    ax[0].plot(IC_RA_horizon,IC_DEC_horizon,color="lime",lw=linewidth, label="Horizon")
    ax[1].plot(Off_axis_RA_horizon,Off_axis_DEC_horizon,color="lime",lw=linewidth, label="Horizon")
    #plot earth core boundary
    ax[0].plot(IC_RA_core,IC_DEC_core,color="red",lw=linewidth, label="Inner core")
    ax[1].plot(Off_axis_RA_core,Off_axis_DEC_core,color="red",lw=linewidth, label="Inner core")
    #plot earth outer core boundary
    ax[0].plot(IC_RA_outer_core,IC_DEC_outer_core,color="orange",lw=linewidth, label="Outer core")
    ax[1].plot(Off_axis_RA_outer_core,Off_axis_DEC_outer_core,color="orange",lw=linewidth, label="Outer core")
    #plot earth mantle boundary
    ax[0].plot(IC_RA_mantle,IC_DEC_mantle,color="yellow",lw=linewidth, label="Mantle")
    ax[1].plot(Off_axis_RA_mantle,Off_axis_DEC_mantle,color="yellow",lw=linewidth, label="Mantle")



    for i in range(len(ax)):
        #plot ra,dec=0,0
        # ax[i].plot(direction_dec,0, markerfacecolor="gold", markeredgecolor="black", marker="D", markersize=6, linestyle="None",label="LIV-field direction")
        ax[0].legend(fontsize=11,ncol=4, loc="upper center",bbox_to_anchor=(1.06, 1.2))
        ax[i].set_xticks([0,90,180,270,360])
        ax[i].set_xticklabels([" 0",90,180,270,"360  "])
        ax[i].set_yticks([-90,-45,0,45,90])
        ax[i].tick_params(axis='both', which='major', labelsize=12)
        ax[i].set_xlabel("RA [deg]",fontsize=14)
        ax[0].set_ylabel("DEC [deg]",fontsize=14)

    ax[0].text(0.05, 0.13, "IceCube", transform=ax[0].transAxes, fontsize=14, color="black", verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    ax[1].text(0.05, 0.95, off_axis_detector, transform=ax[1].transAxes, fontsize=14, color="black", verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

        


    cbar_ms = 4


    fig.subplots_adjust(right=0.88, wspace=0.1)
    cbar_ax = fig.add_axes([0.90, 0.107, 0.025, 0.775])
    cbar = fig.colorbar(ax[1].images[0], cax=cbar_ax, orientation="vertical", fraction=0.05, pad=0.05)
    cbar.set_label(r"Cos($\theta_z)$", fontsize=14)
    cbar.ax.tick_params(labelsize=12)
    cbar.ax.set_yticks([-1,-0.5,0,0.5,1])

    plt.savefig(__file__.replace(".py",".pdf"),  bbox_inches='tight')
