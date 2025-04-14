'''
Compare IceCube with an off-axis detector in neutrino RA,DEC oscillograms for probabilities with sidereal SME parameters and matter effects activated

Simon Hilding-Nørkjær
'''


import time as time_module
from astropy.time import Time
import numpy as np
from deimos.wrapper.osc_calculator import *
from deimos.utils.plotting import *
from deimos.utils.oscillations import * #calc_path_length_from_coszen, get_coszen_from_path_length
from deimos.utils.coordinates import * #get_right_ascension_and_declination
from deimos.utils.constants import * 
from deimos.models.liv.sme import get_sme_state_matrix

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
    off_axis_detector = "ARCA"     # "arca" or "dune"

    initial_flavor = 1          # numu survival
    nubar = False             # neutrino or antineutrino

    E_GeV = 10000.


    atmospheric = True
    sme_basis = "mass"

    a_magnitude_eV = 2e-13 # Overall strength of a component
    c_magnitude = 0#e-26 # Overall strength of c component
    

    # Choose solver (nusquids or deimos)
    solver = args.solver

    # Create calculators
    # For nuSQuIDS case, need to specify energy nodes covering full space
    kw = {}
    if solver == "nusquids" :
        kw["energy_nodes_GeV"] = E_GeV
        kw["nusquids_variant"] = "sme"

    Off_axis_calculator = OscCalculator(solver=args.solver ,atmospheric=atmospheric,**kw)

    Off_axis_calculator.set_matter("vacuum") #
    Off_axis_calculator.set_detector(off_axis_detector)


   

############################ SME PARAMETERS #########################################

    a_y_eV = get_sme_state_matrix(p33=a_magnitude_eV)
    sme_params = { "basis":sme_basis, "a_y_eV":a_y_eV}
    direction_structure = np.array([0., 1., 0.]) # Orientation of field (x, y, z)
    direction_dec = np.array([0,90,0]) @ direction_structure



############################# EARTH LAYER BOUNDARIES #########################################

    # Define RA/dec grid (0.0587s per point)  208*202*0.0587s = 41min 7s
    ra_values_deg = np.linspace(0., 360., num=args.num_points)
    dec_values_deg = np.linspace(-90., 90., num=args.num_points)
    ra_values_rad = np.deg2rad(ra_values_deg)
    dec_values_rad = np.deg2rad(dec_values_deg)
    time = "July 16, 1999, 22:30"
    daytimes = ["July 16, 1999, 04:00", "July 16, 1999, 10:00", "July 16, 1999, 16:00", "July 16, 1999, 22:00"]
 

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


    #
    # Definition of Earth layer boundaries
    #

    azimuth = np.deg2rad(np.linspace(0, 360, 1000))
    ARCA_RA_horizon, ARCA_DEC_horizon = Off_axis_calculator.detector_coords.get_right_ascension_and_declination(0, azimuth, time)

    # Order off-axis horizon arrays by RA
    ARCA_indices = np.argsort(ARCA_RA_horizon)
    ARCA_RA_horizon = ARCA_RA_horizon[ARCA_indices]
    ARCA_DEC_horizon = ARCA_DEC_horizon[ARCA_indices]

    # Earth core boundaries
    pathlength_inner_core = 2 * np.sqrt(EARTH_RADIUS_km**2 - EARTH_INNER_CORE_RADIUS_km**2)
    cosz_inner_core = get_coszen_from_path_length(pathlength_inner_core)
    ARCA_RA_core, ARCA_DEC_core = Off_axis_calculator.detector_coords.get_right_ascension_and_declination(cosz_inner_core, azimuth, time)

    pathlength_outer_core = 2 * np.sqrt(EARTH_RADIUS_km**2 - EARTH_OUTER_CORE_RADIUS_km**2)
    cosz_outer_core = get_coszen_from_path_length(pathlength_outer_core)
    ARCA_RA_outer_core, ARCA_DEC_outer_core = Off_axis_calculator.detector_coords.get_right_ascension_and_declination(cosz_outer_core, azimuth, time)

    pathlength_mantle = 2 * np.sqrt(EARTH_RADIUS_km**2 - EARTH_MANTLE_RADIUS_km**2)
    cosz_mantle = get_coszen_from_path_length(pathlength_mantle)
    ARCA_RA_mantle, ARCA_DEC_mantle = Off_axis_calculator.detector_coords.get_right_ascension_and_declination(cosz_mantle, azimuth, time)

    ARCA_indices = np.argsort(ARCA_RA_mantle)
    ARCA_RA_mantle = ARCA_RA_mantle[ARCA_indices]
    ARCA_DEC_mantle = ARCA_DEC_mantle[ARCA_indices]
    





    #
    # MAIN LOOP
    #

    # print(dec_values_deg)
    P_shape = (3,len(dec_values_deg), len(ra_values_deg), len(daytimes))
    P_Off_axis = np.zeros(P_shape)

    ra_dec_shape = (len(dec_values_deg), len(ra_values_deg), len(daytimes))
    cosz_Off_axis = np.zeros(ra_dec_shape)



    Off_axis_DEC_horizon = np.zeros((len(azimuth),len(daytimes)))
    Off_axis_RA_horizon = np.zeros((len(azimuth),len(daytimes)))   
    Off_axis_DEC_core = np.zeros((len(azimuth),len(daytimes)))
    Off_axis_RA_core = np.zeros((len(azimuth),len(daytimes)))
    Off_axis_DEC_outer_core = np.zeros((len(azimuth),len(daytimes)))
    Off_axis_RA_outer_core = np.zeros((len(azimuth),len(daytimes)))
    Off_axis_DEC_mantle = np.zeros((len(azimuth),len(daytimes)))
    Off_axis_RA_mantle = np.zeros((len(azimuth),len(daytimes)))



    t_init = time_module.time()

    for time_index , time in enumerate(daytimes):

        Off_axis_RA_horizon[:,time_index], Off_axis_DEC_horizon[:,time_index] = Off_axis_calculator.detector_coords.get_right_ascension_and_declination(0, azimuth,time)

        # Order detector horizon arrays as RA-ascending for off-axis detector
        Off_axis_indices=np.argsort(Off_axis_RA_horizon[:,time_index])
        Off_axis_RA_horizon[:,time_index] = Off_axis_RA_horizon[Off_axis_indices,time_index]
        Off_axis_DEC_horizon[:,time_index] = Off_axis_DEC_horizon[Off_axis_indices,time_index]
        
        # Earth core boundary lines (inner core radius 1216km. https://www.ucl.ac.uk/seismin/explore/Earth.html) 
        pathlength_inner_core = 2 * np.sqrt(EARTH_RADIUS_km**2 - EARTH_INNER_CORE_RADIUS_km**2)
        cosz_inner_core = get_coszen_from_path_length(pathlength_inner_core)
        Off_axis_RA_core[:,time_index], Off_axis_DEC_core[:,time_index] = Off_axis_calculator.detector_coords.get_right_ascension_and_declination(cosz_inner_core, azimuth, time)

        # Earth Outer core boundary lines (outer core radius 3486km)
        pathlength_outer_core = 2 * np.sqrt(EARTH_RADIUS_km**2 - EARTH_OUTER_CORE_RADIUS_km**2)
        cosz_outer_core = get_coszen_from_path_length(pathlength_outer_core)
        Off_axis_RA_outer_core[:,time_index], Off_axis_DEC_outer_core[:,time_index] = Off_axis_calculator.detector_coords.get_right_ascension_and_declination(cosz_outer_core, azimuth, time)

        # Earth mantle boundary lines (mantle radius 5701km)
        pathlength_mantle = 2 * np.sqrt(EARTH_RADIUS_km**2 - EARTH_MANTLE_RADIUS_km**2)
        cosz_mantle = get_coszen_from_path_length(pathlength_mantle)
        Off_axis_RA_mantle[:,time_index], Off_axis_DEC_mantle[:,time_index] = Off_axis_calculator.detector_coords.get_right_ascension_and_declination(cosz_mantle, azimuth, time)

        # Order detector mantle arrays as RA-ascending for off-axis detector
        Off_axis_indices=np.argsort(Off_axis_RA_mantle[:,time_index])
        Off_axis_RA_mantle[:,time_index] = Off_axis_RA_mantle[Off_axis_indices,time_index]
        Off_axis_DEC_mantle[:,time_index] = Off_axis_DEC_mantle[Off_axis_indices,time_index]


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



                # Define args to osc prob calc
                calc_kw = {
                    "initial_flavor":initial_flavor,
                    "nubar" : nubar,
                    "energy_GeV":E_GeV,
                    "ra_rad":ra_rad,
                    "dec_rad":dec_rad,
                    "time":time,
                    "sme_params":sme_params,
                }

                P_Off_axis_results, coszen_values_Off_axis, azimuth_values_Off_axis = Off_axis_calculator.calc_osc_prob_sme_directional_atmospheric(**calc_kw)


                # Save probabilities

                P_Off_axis[0,i,j,time_index] = P_Off_axis_results[0]
                P_Off_axis[1,i,j,time_index] = P_Off_axis_results[1]
                P_Off_axis[2,i,j,time_index] = P_Off_axis_results[2]
        

                # Check that probabilities sum to 1
                assert np.isclose( np.sum(P_Off_axis[:,i,j,time_index]), 1.0, atol=1e-10)
                
                # Save coszen values
                cosz_Off_axis[i,j,time_index] = coszen_values_Off_axis

                # Timing
                if t== 100:
                    delta_time = (time_module.time() - start_time)/100.
                    print("Calculation time for one iteration: %0.4f seconds" % delta_time)
                    print("Total calculation time estimate: %0.2f minutes" % (delta_time*len(dec_values_deg)*len(ra_values_deg)/60.))

    print("Total calculation time: %0.2f minutes" % ((time_module.time()-t_init)/60.) )


    # Convert RA/DEC to degrees
    Off_axis_RA_horizon = np.rad2deg(Off_axis_RA_horizon)
    Off_axis_DEC_horizon = np.rad2deg(Off_axis_DEC_horizon)
    Off_axis_RA_core = np.rad2deg(Off_axis_RA_core)
    Off_axis_DEC_core = np.rad2deg(Off_axis_DEC_core)
    Off_axis_RA_outer_core = np.rad2deg(Off_axis_RA_outer_core)
    Off_axis_DEC_outer_core = np.rad2deg(Off_axis_DEC_outer_core)
    Off_axis_RA_mantle = np.rad2deg(Off_axis_RA_mantle)
    Off_axis_DEC_mantle = np.rad2deg(Off_axis_DEC_mantle)


    # Numpy-ify
    P_Off_axis = np.array(P_Off_axis)
    assert P_shape == P_Off_axis.shape



    
    # plot RA vs dec oscillogram for mu-survival for the off-axis detector at four different times
    linewidth = 1.2
    marker = "o"
    markersize = 1
    alpha = 1

    fig, ax = plt.subplots(2,2, figsize=(9,7), sharex=True, sharey=True)
    ax = ax.flatten()

    fig.suptitle( r"$E$ = %0.3g GeV // SME: a_eV_y=%0.3g // c=%0.3g" % (E_GeV, a_magnitude_eV,c_magnitude), fontsize=12 )

    for time_index, time in enumerate(daytimes):

        ax[time_index].imshow(P_Off_axis[1,:,:,time_index], origin="lower", extent=[ra_values_deg[0], ra_values_deg[-1], dec_values_deg[0], dec_values_deg[-1]], aspect="auto", cmap="RdPu", vmin=0., vmax=1.)

        ax[time_index].plot(Off_axis_RA_horizon[:,time_index],Off_axis_DEC_horizon[:,time_index],color="lime",alpha=alpha,marker=marker,ms=markersize,linestyle="None")#, label="Horizon")
        ax[time_index].plot(Off_axis_RA_core[:,time_index],Off_axis_DEC_core[:,time_index],color="red",alpha=alpha,marker=marker,ms=markersize,linestyle="None")#, label="Earth inner core")
        ax[time_index].plot(Off_axis_RA_outer_core[:,time_index],Off_axis_DEC_outer_core[:,time_index],color="orange",alpha=alpha,marker=marker,ms=markersize,linestyle="None")#,label="Earth outer core")
        ax[time_index].plot(Off_axis_RA_mantle[:,time_index],Off_axis_DEC_mantle[:,time_index],color="yellow",alpha=alpha,marker=marker,ms=markersize,linestyle="None")#, label="Earth mantle")
 
        ax[time_index].plot(direction_dec,0, markerfacecolor="gold", markeredgecolor="black", marker="D", markersize=6, linestyle="None")#, label="LIV-field direction")
        #add the time as text on the plot
        ax[time_index].text(0.05, 0.95, time.replace("July 16, 1999, ","")+" UTC", transform=ax[time_index].transAxes, fontsize=14,color="white", verticalalignment='top', bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))
        ax[time_index].tick_params(axis='both', which='major', labelsize=14)

    
    ax[3].imshow(P_Off_axis[1,:,:,3], origin="lower", extent=[ra_values_deg[0], ra_values_deg[-1], dec_values_deg[0], dec_values_deg[-1]], aspect="auto", cmap="RdPu", vmin=0., vmax=1.)

    ax[3].plot(Off_axis_RA_horizon[:,3],Off_axis_DEC_horizon[:,3],color="lime",alpha=alpha,marker=marker,ms=markersize,linestyle="None", label="Horizon")
    ax[3].plot(Off_axis_RA_core[:,3],Off_axis_DEC_core[:,3],color="red",alpha=alpha,marker=marker,ms=markersize,linestyle="None", label="Earth inner core")
    ax[3].plot(Off_axis_RA_outer_core[:,3],Off_axis_DEC_outer_core[:,3],color="orange",alpha=alpha,marker=marker,ms=markersize,linestyle="None",label="Earth outer core")
    ax[3].plot(Off_axis_RA_mantle[:,3],Off_axis_DEC_mantle[:,3],color="yellow",alpha=alpha,marker=marker,ms=markersize,linestyle="None", label="Earth mantle")

    ax[3].plot(direction_dec,0, markerfacecolor="gold", markeredgecolor="black", marker="D", markersize=6, linestyle="None", label="LIV-field direction")
    ax[3].tick_params(axis='both', which='major', labelsize=14)
    ax[2].set_xticks([0,90,180,270,360])
    ax[3].set_xticks([0,90,180,270,360])
    ax[2].set_xticklabels([" 0",90,180,270,"360   "])
    ax[3].set_xticklabels([" 0",90,180,270,"360   "])
    ax[0].set_yticks([-90,-45,0,45,90])
    ax[1].set_yticks([-90,-45,0,45,90])
    ax[0].set_yticklabels(["-90","-45"," 0","45","\n90"])
    

    ax[2].set_xlabel("RA[deg]", fontsize=15)
    ax[3].set_xlabel("RA[deg]", fontsize=15)
    ax[0].set_ylabel("DEC[deg]", fontsize=15)
    ax[2].set_ylabel("DEC[deg]", fontsize=15)



    #add the time as text on the plot
    ax[3].text(0.05, 0.95, time.replace("July 16, 1999, ","")+" UTC", transform=ax[3].transAxes, fontsize=14,color="white", verticalalignment='top', bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))




    #colorbar
    fig.subplots_adjust(right=0.88, wspace=0.07, hspace=0.07)
    cbar_ax = fig.add_axes([0.90, 0.107, 0.025, 0.775])
    cbar = fig.colorbar(ax[1].images[0], cax=cbar_ax, orientation="vertical", fraction=0.05, pad=0.05)
    cbar.set_label(r"$P(\nu_{\mu} \to \nu_{\mu})$", fontsize=15)
    cbar.ax.tick_params(labelsize=14)
    # fig.legend(loc="upper center", fontsize=11, ncol=5, bbox_to_anchor=(0.49, 0.98),fancybox=True, shadow=True)
    linewidth = 4
    egend_handeles, legend_labels = ax[3].get_legend_handles_labels()
    #only use one icon for the legend
    manuallist = [plt.Line2D([0], [0], color='gold', marker='D', linestyle='None', markersize=7, markeredgewidth=0.1, markeredgecolor="black"),
                plt.Line2D([0], [0], color='lime', marker='None',  markersize=10, markeredgewidth=0.0, linewidth=linewidth),
                plt.Line2D([0], [0], color='yellow', marker='None', linestyle='-', markersize=10, linewidth=linewidth),
                plt.Line2D([0], [0], color='orange', marker='None',  markersize=10, markeredgewidth=0.0, linewidth=linewidth),
                plt.Line2D([0], [0], color='red', marker='None', linestyle='-', markersize=10, linewidth=linewidth)]
                # plt.Line2D([0], [0], color='r', marker='None',  markersize=10, markeredgewidth=0.0)]
    legend_handeles = [manuallist[0], manuallist[1], manuallist[2], manuallist[3], manuallist[4]]#, manuallist[5]]
    legend_labels = ["LIV-field direction","Horizon","Mantle", "Outer Core", "Inner Core"]#,  r'$\sigma$ contour']
    lgnd = ax[3].legend(legend_handeles, legend_labels, fontsize=13, loc=(-1.2,2.10), ncol=5, handlelength=0.6,fancybox=True)#, shadow=True)


    # fig.tight_layout()





    
    # # plot RA vs dec oscillogram

    # fig, ax = plt.subplots(2,2, figsize=(8,6))
    # ax = ax.flatten()

    # fig.suptitle( r"Sidereal dependence of $cos(\theta_{zenith})$ in RA,DEC-oscillogram"+f", Time: {time} UTC", fontsize=12 )

    # for time_index, time in enumerate(daytimes):
            
    #     ax[time_index].imshow(cosz_Off_axis[:,:,time_index], origin="lower", extent=[ra_values_deg[0], ra_values_deg[-1], dec_values_deg[0], dec_values_deg[-1]], aspect="auto", cmap="RdYlBu", vmin=-1., vmax=1.)

    #     #plot horizontal lines 
    #     ax[time_index].plot(Off_axis_RA_horizon[:,time_index],Off_axis_DEC_horizon[:,time_index],color="black",marker=marker,ms=markersize,linestyle="None", label="Horizon")
    #     #plot earth core boundary
    #     ax[time_index].plot(Off_axis_RA_core[:,time_index],Off_axis_DEC_core[:,time_index],color="red",marker=marker,ms=markersize,linestyle="None", label="Earth inner core")
    #     #plot earth outer core boundary
    #     ax[time_index].plot(Off_axis_RA_outer_core[:,time_index],Off_axis_DEC_outer_core[:,time_index],color="orange",marker=marker,ms=markersize,linestyle="None", label="Earth outer core")
    #     #plot earth mantle boundary
    #     ax[time_index].plot(Off_axis_RA_mantle[:,time_index],Off_axis_DEC_mantle[:,time_index],color="yellow",marker=marker,ms=markersize,linestyle="None", label="Earth mantle")



    # for i in range(len(ax)):
    #     #plot ra,dec=0,0
    #     ax[i].plot(direction_dec,0, markerfacecolor="gold", markeredgecolor="black", marker="D", markersize=6, linestyle="None",label="LIV-field direction")
    #     # ax[i].legend(fontsize=8)



    
    # ax[0].set(xlabel="RA[deg]",ylabel="DEC[deg]",title="IceCube")
    # ax[1].set(xlabel="RA[deg]",ylabel="DEC[deg]",title=off_axis_detector)
    # cbar0 = fig.colorbar(ax[1].images[0], ax=ax[1], orientation="vertical", fraction=0.05, pad=0.05)
    # cbar0.set_label(r"$Cos(\theta_z)$", fontsize=12)

    # fig.tight_layout()







    #
    # Done
    #

    print("")
    dump_figures_to_pdf( __file__.replace(".py","_" + solver + ".pdf") )
