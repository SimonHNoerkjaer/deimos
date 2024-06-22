'''
Reproduce plots from external papers, using our implementation

Tom Stuttard
'''

import sys, os, collections

from deimos.wrapper.osc_calculator import *
from deimos.utils.plotting import *
from deimos.utils.constants import *
from deimos.utils.oscillations import get_coszen_from_path_length
from deimos.models.liv.sme import *


#
# Plotting functions
#

def reproduce_1410_4267(solver, num_points=100) :
    '''
    Reproduce plots from arXiv:1410.4267 - SuperK, isotropic LIV

    Cannot directly reproduce all their plots as as some show changes in SuperK event rates rather than 
    oscillation probabilities, but can look to reproduce similar effects at least (e.gh. same periodicity, etc)
    '''

    print("\n>>> Reproduce arXiv:1410.4267 ...")


    #
    # Define oscillogram grid
    #

    E_values_GeV = np.logspace(-1., 3., num=num_points)
    L_values_km = np.geomspace(1., EARTH_DIAMETER_km, num=num_points+1)
    coszen_values = get_coszen_from_path_length(L_values_km)


    #
    # Create model
    #

    # For nuSQuIDS case, need to specify energy nodes covering full space
    kw = {}
    if solver == "nusquids" :
        kw["energy_nodes_GeV"] = E_values_GeV
        kw["nusquids_variant"] = "sme"

    # Create calculator
    calculator = OscCalculator(
        solver=solver,
        atmospheric=True,
        **kw
    )

    #TODO Not sure what matter they use in the paper?
    calculator.set_matter("vacuum")


    #
    # Plot oscillograms (Fig. 1)
    #

    # Define physics cases
    cases = collections.OrderedDict()
    cases["No LIV"] = { "a_eV":0., "c":0., "color":"black" }
    cases[r"$a^t_{\mu \tau}$"] = { "a_eV":1e-22*1e9, "c":0., "color":"blue" }
    cases[r"$c^{tt}_{\mu \tau}$"] = { "a_eV":0., "c":7.5e-23, "color":"red" }

    #TODO more in appendices

    # Make figure
    fig, ax = plt.subplots(nrows=len(cases), figsize=(5, 4*len(cases)))
    fig.suptitle("arXiv:1410.4267 Fig. 1")

    # Loop over physics cases
    for y, (case_key, case_params) in enumerate(cases.items()) :

        ax[y].set_title(case_key)

        # Get SME matrices, and configure model
        calculator.set_sme_isotropic(
            basis="flavor",
            a_eV=get_sme_state_matrix(p23=case_params["a_eV"]), # a_mu_tau
            c=get_sme_state_matrix(p23=case_params["c"]), # c_mu_tau
        )

        # Plot oscillogram
        calculator.plot_oscillogram(
            initial_flavor=1,
            final_flavor=1,
            energy_GeV=E_values_GeV,
            distance_km=L_values_km,
            nubar=False,
            cmap="hot",
            ax=ax[y],
        )

    # Format
    fig.tight_layout()


    #
    # Plot 1D (vs coszen) plots (Fig. 2)
    #

    # Cannot directly reproduce as these are event rate, not P, but can try and see same shape at least...

    # Define coszen
    coszen_values = np.linspace(-1., 0., num=num_points)

    # Define the peak energies for each sample shown
    peak_energy_values_GeV = [10, 100, 1000]

    # Create figure
    fig, ax = plt.subplots(ncols=3, figsize=(12, 4))
    fig.suptitle("arXiv:1410.4267 Fig. 2")

    # Loop over physics cases
    for case_key, case_params in cases.items() :

        print(case_key)

        # Get SME matrices, and configure model
        calculator.set_sme_isotropic(
            basis="flavor",
            a_eV=get_sme_state_matrix(p23=case_params["a_eV"]), # a_mu_tau
            c=get_sme_state_matrix(p23=case_params["c"]), # c_mu_tau
        )

        # Loop over peak energy values
        for x, peak_E_GeV in enumerate(peak_energy_values_GeV) :

            # Title
            ax[x].set_title(r"$E$ = %i GeV" % peak_E_GeV)

            # Plot vs coszen
            calculator.plot_osc_prob_vs_coszen(
                initial_flavor=1, final_flavor=1, # Assuming numu survival dominates
                energy_GeV=peak_E_GeV, 
                coszen=coszen_values, 
                nubar=False, 
                fig=fig, ax=[ax[x]], 
                color=case_params["color"]
            )

    # Format
    fig.tight_layout()


def reproduce_2302_12005(solver, num_points) :
    '''
    Reproduce arXiv:2302.12005

    This is future LBL detectors (DUNE and T2HK)
    '''

    print("\n>>> Reproduce arXiv:2302.12005 ...")


    #
    # Detector steering
    #

    E_values_DUNE_GeV = np.linspace(0.5, 5., num=num_points)
    E_values_T2HK_GeV = np.linspace(0.25, 1.5, num=num_points)

    L_DUNE_km = DUNE_BASELINE_km 
    L_T2HK_km = T2HK_BASELINE_km 


    #
    # Create model
    #

    # For nuSQuIDS case, need to specify energy nodes covering full space
    kw = {}
    if solver == "nusquids" :
        kw["energy_nodes_GeV"] = np.linspace(min(E_values_DUNE_GeV[0], E_values_T2HK_GeV[0]), max(E_values_DUNE_GeV[-1], E_values_T2HK_GeV[-1]), num=100)
        kw["nusquids_variant"] = "sme"

    # Create calculator
    calculator = OscCalculator(
        solver=solver,
        atmospheric=False,
        **kw
    )

    # Set the oscillation params, as per Table 2
    calculator.set_mixing_angles(theta12=np.deg2rad(33.45), theta13=np.deg2rad(8.62), theta23=np.deg2rad(42.1), deltacp=np.deg2rad(230.))
    calculator.set_mass_splittings(7.42e-5, 2.51e-3)

    #TODO Not sure what matter they use in the paper?
    calculator.set_matter("vacuum")



    #
    # Fig 1 : nue appearance, SME a params
    #

    # Steer plot
    initial_flavor, final_flavor, nubar = 1, 0, False

    # Define physics cases
    cases = collections.OrderedDict()
    a_mag_eV = 2e-23 * 1e9
    cases[r"$a_{e \mu}$"] = { "a_eV":get_sme_state_matrix(p12=a_mag_eV) }
    cases[r"$a_{e \tau}$"] = { "a_eV":get_sme_state_matrix(p13=a_mag_eV) }
    cases[r"$a_{\mu \tau}$"] = { "a_eV":get_sme_state_matrix(p23=a_mag_eV) }

    # Create figure
    fig, ax = plt.subplots(ncols=3, nrows=2, figsize=(12, 8))
    fig.suptitle("arXiv:2302.12005 Fig. 1")

    # Loop over physics cases
    for x, (case_key, case_params) in enumerate(cases.items()) :

        print("Fig 1, %s" % case_key)

        # Loop over detector
        for y, (detector_label, E_values_GeV, L_km) in enumerate( zip(["DUNE", "T2HK"], [E_values_DUNE_GeV, E_values_T2HK_GeV], [L_DUNE_km, L_T2HK_km]) ) : 

            # Common plotting args
            plot_kw = dict(
                initial_flavor=initial_flavor, 
                final_flavor=final_flavor,
                nubar=nubar, 
                energy_GeV=E_values_GeV, 
                distance_km=L_km, 
                fig=fig,
                ax=[ax[y, x]], 
            )

            # Plot std osc
            calculator.set_std_osc()
            calculator.plot_osc_prob_vs_energy(
                color="black",
                label="No LIV",
                **plot_kw
            )

            # Plot SME
            calculator.set_sme_isotropic(basis="flavor", **case_params)
            calculator.plot_osc_prob_vs_energy(
                color="blue",
                label=case_key,
                **plot_kw
            )

    # Format
    for x in range(len(cases)) :
        ax[0,x].set_ylim(0., 0.15)
        ax[1,x].set_ylim(0., 0.08)
    fig.tight_layout()


    #
    # Fig 2 : nue appearance, SME c params
    #

    # Steer plot
    initial_flavor, final_flavor, nubar = 1, 0, False

    # Define physics cases
    cases = collections.OrderedDict()
    c_mag = 1e-24
    cases[r"$c_{e \mu}$"] = { "c":get_sme_state_matrix(p12=c_mag) }
    cases[r"$c_{e \tau}$"] = { "c":get_sme_state_matrix(p13=c_mag) }
    cases[r"$c_{\mu \tau}$"] = { "c":get_sme_state_matrix(p23=c_mag) }

    # Create figure
    fig, ax = plt.subplots(ncols=len(cases), nrows=2, figsize=(4*len(cases), 8))
    fig.suptitle("arXiv:2302.12005 Fig. 2")

    # Loop over physics cases
    for x, (case_key, case_params) in enumerate(cases.items()) :

        print("Fig 2, %s" % case_key)

        # Loop over detector
        for y, (detector_label, E_values_GeV, L_km) in enumerate( zip(["DUNE", "T2HK"], [E_values_DUNE_GeV, E_values_T2HK_GeV], [L_DUNE_km, L_T2HK_km]) ) : 

            # Common plotting args
            plot_kw = dict(
                initial_flavor=initial_flavor, 
                final_flavor=final_flavor,
                nubar=nubar, 
                energy_GeV=E_values_GeV, 
                distance_km=L_km, 
                fig=fig,
                ax=[ax[y, x]], 
            )

            # Plot std osc
            calculator.set_std_osc()
            calculator.plot_osc_prob_vs_energy(
                color="black",
                label="No LIV",
                **plot_kw
            )

            # Plot SME
            calculator.set_sme_isotropic(basis="flavor", **case_params)
            calculator.plot_osc_prob_vs_energy(
                color="blue",
                label=case_key,
                **plot_kw
            )

    # Format
    for x in range(len(cases)) :
        ax[0,x].set_ylim(0., 0.1)
        ax[1,x].set_ylim(0., 0.06)
    fig.tight_layout()


    #
    # Fig 3 : numu disappearance
    #

    # Steer plot
    initial_flavor, final_flavor, nubar = 1, 1, False

    # Define physics cases
    cases = collections.OrderedDict()
    cases[r"$a_{\mu \tau}$"] = { "a_eV":get_sme_state_matrix(p23=a_mag_eV) }
    cases[r"$c_{\mu \tau}$"] = { "c":get_sme_state_matrix(p23=c_mag) }

    # Create figure
    fig, ax = plt.subplots(ncols=len(cases), nrows=2, figsize=(4*len(cases), 8))
    fig.suptitle("arXiv:2302.12005 Fig. 3")

    # Loop over physics cases
    for x, (case_key, case_params) in enumerate(cases.items()) :

        print("Fig 3, %s" % case_key)

        # Loop over detector
        for y, (detector_label, E_values_GeV, L_km) in enumerate( zip(["DUNE", "T2HK"], [E_values_DUNE_GeV, E_values_T2HK_GeV], [L_DUNE_km, L_T2HK_km]) ) : 

            # Common plotting args
            plot_kw = dict(
                initial_flavor=initial_flavor, 
                final_flavor=final_flavor,
                nubar=nubar, 
                energy_GeV=E_values_GeV, 
                distance_km=L_km, 
                fig=fig,
                ax=[ax[y, x]], 
            )

            # Plot std osc
            calculator.set_std_osc()
            calculator.plot_osc_prob_vs_energy(
                color="black",
                label="No LIV",
                **plot_kw
            )

            # Plot SME
            calculator.set_sme_isotropic(basis="flavor", **case_params)
            calculator.plot_osc_prob_vs_energy(
                color="blue",
                label=case_key,
                **plot_kw
            )

    # Format
    fig.tight_layout()


def reproduce_1907_09145(solver, num_points) :
    '''
    Isotropic LIV in NOvA
    '''

    raise NotImplementedError("TODO")


def reproduce_hep_ph_0406255(solver, num_points) :
    '''
    Sidereal LIV in LSND
    '''


    print("\n>>> Reproduce arXiv:hep-ph/0406255 ...")


    #
    # Detector steering
    #

    E_values_GeV = np.linspace(20., 60, num=num_points) * 1e-3
    L_km = 30e-3 # LSND has 30 m baseline 


    #
    # Create model
    #

    # For nuSQuIDS case, need to specify energy nodes covering full space
    kw = {}
    if solver == "nusquids" :
        kw["energy_nodes_GeV"] = E_values_GeV
        kw["nusquids_variant"] = "sme"

    # Create calculator
    calculator = OscCalculator(
        solver=solver,
        atmospheric=False,
        **kw
    )

    #TODO Not sure what matter they use in the paper?
    calculator.set_matter("vacuum")

    # Place detector
    calculator.set_detector("LSND")


    #
    # Fig. 1
    #

    # Looking at antinue appearance
    initial_flavor, final_flavor, nubar = 1, 0, True


    #TODO
    #TODO
    #TODO
    #TODO
    #TODO
    #TODO
    ra_rad = 0.
    dec_rad = 0.
    #TODO
    #TODO
    #TODO
    #TODO
    #TODO
    #TODO
    #TODO

    # Define time scan
    start_time = datetime.datetime(2021, 1, 1, 0, 0, 0, 0) # Midnight, Jan 1st 2021  #TODO use a date LSND was actually running!
    hr_values = np.linspace(0., SIDEREAL_DAY_hr, num=48) # One sidereal day
    time_values = [ start_time + datetime.timedelta(hours=hr)  for hr in hr_values ]

    # Get LIV osc probs
    P_sme, coszen_values, azimuth_values = calculator.calc_osc_prob_sme_directional_beam(
        initial_flavor=initial_flavor,
        nubar=nubar,
        energy_GeV=E_values_GeV,
        ra_rad=ra_rad,
        dec_rad=dec_rad,
        time=time_values,
        sme_params=sme_params, 
    )

    # Selected chosen final flavor
    P_sme = P_sme[...,final_flavor]

    # Make fig
    fig, ax = plt.subplots( nrows=2, figsize=(6, 6) )

    # fig.suptitle(args.detector)
    # fig.suptitle( r"$E$ = %0.3g GeV // $\delta$ = %0.3g deg // %s" % (args.detector, E_GeV, dec_deg, date), fontsize=12 )  #TODO text box
    
    # Plot osc probs
    # ax[0].plot(hr_values, P_std, color="black", linestyle="-", label="Std. osc.")
    ax[0].plot(hr_values, P_sme, color="orange", linestyle="--", label="SME")

    # Plot coszen
    ax[1].plot(hr_values, coszen_values, color="blue", linestyle="-")

    # Formatting
    for this_ax in ax :
        this_ax.set_xlabel("t [hr]")
        this_ax.set_xlim(hr_values[0], hr_values[-1])
        this_ax.grid(True)
    ax[0].set_ylabel(P_label)
    ax[0].set_ylim(-0.01, 1.01)
    ax[0].legend(fontsize=12)
    ax[1].set_ylabel( "coszen" )
    ax[1].set_ylim(-1.01, 1.01)
    fig.tight_layout()



def reproduce_2309_01756(solver, num_points) :
    '''
    Sidereal LIV in NOvA
    '''

    #
    # Detector steering
    #

    E_values_GeV = np.linspace(0.5, 5., num=num_points)
    L_km = NOvA_BASELINE_km


    #
    # Create model
    #

    # For nuSQuIDS case, need to specify energy nodes covering full space
    kw = {}
    if solver == "nusquids" :
        kw["energy_nodes_GeV"] = E_values_GeV
        kw["nusquids_variant"] = "sme"

    # Create calculator
    calculator = OscCalculator(
        solver=solver,
        atmospheric=False,
        **kw
    )

    #TODO Not sure what matter they use in the paper?
    calculator.set_matter("vacuum")

    # Place detector
    calculator.set_detector("NOvA")


    #
    # Fig. 1 : Standard oscillations (no LIV)
    #

    # Define time scan
    start_time = datetime.datetime(2021, 1, 1, 0, 0, 0, 0) # Midnight, Jan 1st 2021
    hr_values = np.linspace(0., SIDEREAL_DAY_hr, num=num_points+1) # One sidereal day
    time_values = [ start_time + datetime.timedelta(hours=hr)  for hr in hr_values ]

    # Make figure
    fig = plt.figure(figsize=(6, 6))
    gs = fig.add_gridspec(ncols=2, nrows=2, width_ratios=(1, 3), wspace=0.2, hspace=0.5)#left=0.1, right=0.9, bottom=0.1, top=0.9, hspace=0.05)
    ax_1d = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0])]
    ax_2d = [fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1])]

    # Loop over final state (appearance and disappearance channels)
    initial_flavor, nubar = 1, False
    for y, final_flavor in enumerate([0, 1]) :

        # Get LIV osc probs
        osc_prob, ra_values_rad, dec_values_rad = calculator.calc_osc_prob_sme_directional_beam(
            initial_flavor=initial_flavor,
            nubar=nubar,
            energy_GeV=E_values_GeV,
            distance_km=L_km,
            time=time_values,
            std_osc=True, 
        )

        # Select chosen final flavor
        osc_prob = osc_prob[...,final_flavor]

        # Plot 1D probability vs energy (select any t, since t-independent)
        ax_1d[y].plot(osc_prob[0,:], E_values_GeV, color="red", linestyle="-")

        # PLot prob vs [E, t] (even though t-independent - to confirm)
        # plot_colormap( ax=ax, x=energy_GeV, y=y, z=osc_probs, vmin=vmin, vmax=vmax, cmap=cmap, zlabel=r"$%s$"%transition_prob_tex )

    # # Plot coszen
    # ax[1].plot(hr_values, coszen_values, color="blue", linestyle="-")

    # # Formatting
    # for this_ax in ax :
    #     this_ax.set_xlabel("t [hr]")
    #     this_ax.set_xlim(hr_values[0], hr_values[-1])
    #     this_ax.grid(True)
    # ax[0].set_ylabel(P_label)
    # ax[0].set_ylim(-0.01, 1.01)
    # ax[0].legend(fontsize=12)
    # ax[1].set_ylabel( "coszen" )
    # ax[1].set_ylim(-1.01, 1.01)
    fig.tight_layout()



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
    parser.add_argument("-n", "--num-points", type=int, required=False, default=1000, help="Num scan point")
    args = parser.parse_args()


    #
    # Plotting
    #

    # Isotropic LIV
    reproduce_1410_4267(solver=args.solver, num_points=args.num_points)
    reproduce_2302_12005(solver=args.solver, num_points=args.num_points)

    # Sidereal LIV
    # reproduce_hep_ph_0406255(solver=args.solver, num_points=args.num_points)
    # reproduce_2309_01756(solver=args.solver, num_points=args.num_points)


    print("")
    dump_figures_to_pdf( __file__.replace(".py",".pdf") )
