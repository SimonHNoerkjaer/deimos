'''
Comparing a range of QG-motivated decoherence scenarios

This is part of addressing comments on the review of the IceCube TeV decoherence paper

Refs:
  [1] https://arxiv.org/pdf/2306.14778.pdf
  [2] https://arxiv.org/abs/2208.12062

Tom Stuttard
'''

import sys, os, collections, copy, datetime

from deimos.wrapper.osc_calculator import *
from deimos.utils.plotting import *
from deimos.utils.constants import *

from deimos.density_matrix_osc_solver.density_matrix_osc_solver import GeV_to_eV


#
# Helper functions
#

def calc_qg_damping_coefficient(qg_model, E_GeV, mass_splitting_eV2, theta_rad, **qg_params) :
    '''
    Calculate the damping coefficient for the various models considered, alpha

    Definition : D = alpha * L
    Damping term: exp{-D}

    This is only valid for the 2-flavor vacuum case
    '''

    # Convert units
    E_eV = E_GeV * GeV_to_eV

    # Get QG scale
    E_qg_eV = qg_params["E_qg_eV"]

    # Checks
    assert E_qg_eV is not None
    assert qg_model is not None
    assert isinstance(qg_model, str)

    # Some elements are common to multiple models
    if qg_model in ["minimal_length_fluctuations", "metric_fluctuations"] :

        # Mass of lightest neutrino state is a free parameter
        m1_eV = qg_params["m1_eV"]
        m2_eV = np.sqrt(  mass_splitting_eV2 + np.square(m1_eV) ) #TODO think I might have made a function for this already? if so use it, it not then make one

        # Calc average group velocity, p/E, where p is average of the two states
        # Using E^2 = p^2 + m^2 to get p, where again m is the average
        # Note that this is basically always = 1 for the relevent energies (as also mentioned in footnote of page 12 of https://arxiv.org/pdf/2306.14778.pdf)
        #TODO cross-check calc via Lorentz boost method?
        m_mean_eV = ( m1_eV + m2_eV ) / 2.
        p_mean_eV = np.sqrt( np.square(E_eV) - np.square(m_mean_eV) )
        v_g = p_mean_eV / E_eV
        
        # Calc (delta m)^2   --> NOT delta (m^2)   (e.g. NOT mass splitting)
        dm_squared_eV = np.square( m2_eV - m1_eV )  #TODO try alt calc
    
    # Calc damping term: Minimal length fluctuations case
    if qg_model == "minimal_length_fluctuations" :
        alpha = 16. * np.power(E_eV, 4.) * dm_squared_eV
        alpha /= (v_g * np.power(E_qg_eV, 5.) )

    # Calc damping term: Stochastic fluctuations of the metric case
    #TODO eqn 29 vs 30? think 30 is just a simplification of 29 in some cases. For now using eqn 29
    elif qg_model == "metric_fluctuations" :
        alpha = 1. / ( 8. * v_g * E_qg_eV )
        alpha *= np.square( 1. + ( np.square(E_eV) / np.square(m1_eV*m2_eV) ) ) 
        alpha *= dm_squared_eV

    # Calc damping term: nu-VBH interactions
    elif qg_model == "nu_vbh_interactions" : #TODO which interaction type?
        # https://arxiv.org/pdf/2007.00068.pdf eqn 20, with zeta = 1 for natural scale (and using E_qg as free param as with other models here, e.g. not necessarily equal to M_P)
        n = qg_params["n"]
        alpha = np.power(E_eV, n) / np.power(E_qg_eV, n-1.)

    else :
        raise Exception("Unknown model : %s" % qg_model)

    return alpha


def calc_coherence_length(*args, **kwargs) :

    '''
    Get the coherence length for a given QG model

    This is when exp{-alpah*L} = 1/e   ->  -alpha*L = -1   -> L_coh = 1/alpha
    '''
    return 1. / calc_qg_damping_coefficient(*args, **kwargs)


    
#
# Plot functions
#


def plot_LoE(ax, E_GeV, L_km, P, color, linestyle, label=None) :
    '''
    Plot an L/E distribution
    '''

    # Calc L/E
    LoE = L_km / E_GeV

    # Check P
    #TODO handle 2D
    assert P.ndim == 1

    # Plot
    ax.plot(LoE, P, color=color, linestyle=linestyle, label=label)

    # Format ax
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlim(LoE[-1], LoE[0]) # Reverse x axis (since L/E is inverse of E)
    ax.set_xlabel("%s / %s" % (DISTANCE_LABEL, ENERGY_LABEL))


def calc_2flav_survival_prob(
    calculator, 
    E_GeV, 
    L_km, 
    flavor, 
    nubar=False, 
    # Solver options
    use_deimos=False,
    # QG model
    qg_model=None, 
    **qg_params
) :
    '''
    Equation 24 from [1], which gives 2-flavor vacuum surival probablity with a damping factor included
    '''

    #TODO Just implement D[rho] operator instead...

 
    # This is for 2 flavor only
    assert calculator.num_neutrinos == 2

    # This is for vacuum only
    #TODO


    # Extract standard osc params
    mass_splitting_eV2 = calculator.get_mass_splittings()
    assert len(mass_splitting_eV2) == 1
    mass_splitting_eV2 = mass_splitting_eV2[0]

    theta_rad = calculator.get_mixing_angles()
    assert len(theta_rad) == 1
    theta_rad = theta_rad[0]


    #
    # Calc standard osc
    #

    if use_deimos : #TODO not sure I am getting the correct answer here when using my own solver, why?

        calculator.set_std_osc()

        Pstd = calculator.calc_osc_prob(
            energy_GeV=E_GeV,
            initial_flavor=flavor,
            distance_km=L_km,
            nubar=nubar,
        )

        # Remove distance dimension   #TODO make more general
        Pstd = Pstd[:,0,...]

        # Get final flavor
        Pstd = Pstd[...,flavor]

    else :

        Pstd = 1. - calc_disappearance_prob_2flav_vacuum(E_GeV=E_GeV, L_km=L_km, mass_splitting_eV2=mass_splitting_eV2, theta_rad=theta_rad)


    # Done now if user doesn't ant to add QG effects
    if qg_model is None :
        return Pstd


    #TODO could also use simple analytic expression


    #
    # Calc damping term
    #

    # Get the coefficient
    alpha = calc_qg_damping_coefficient(E_GeV=E_GeV, mass_splitting_eV2=mass_splitting_eV2, theta_rad=theta_rad, qg_model=qg_model, **qg_params)

    # Convert units
    L_eV = L_km * km_to_eV

    # Calc damping term
    D = alpha * L_eV
    damping_term = np.exp(-D)


    #
    # Calc overall survival prob.
    #

    P_qg = ( damping_term * Pstd ) + ( (1. - damping_term) * (1. - (np.square(np.sin(2.*theta_rad)) / 2.)) )

    return P_qg

    

#
# Plot functions
#

def reproduce_2306_14778(solver) :
    '''
    Reproducing plots from https://arxiv.org/pdf/2306.14778.pdf to test my implementation of their scenario
    '''


    #
    # Create model
    #

    # Create calculator
    # Using 2-flavor approximation to match paper
    calculator = OscCalculator(
        solver=solver,
        atmospheric=False,
        flavors=["e", "mu"],
        mixing_angles_rad=[np.arcsin(np.sqrt(0.85))/2.], # Match paper
        mass_splittings_eV2=[7.53e-5], # Match paper
    )

    # Use vacuum
    calculator.set_matter("vacuum")




    #
    # Fig 1 : KamLAND
    #

    # This is for KamLAND
    L_km = KAMLAND_BASELINE_km
    LoE_values_km_per_MeV = np.linspace(20., 105, num=100) # Match range of Figure (corresponds to roughly 1-10 MeV)
    E_GeV_values = np.sort( L_km / (LoE_values_km_per_MeV*1e3) )
    initial_flavor, final_flavor, nubar = 0, 0, True # antinu survival

    # Set osc params from figure caption

    # Calculations assume survival
    assert initial_flavor == final_flavor

    # Make fig
    fig1, ax1 = plt.subplots( figsize=(6, 4) )

    # Plot std osc
    Pstd = calc_2flav_survival_prob(calculator=calculator, E_GeV=E_GeV_values, L_km=L_km, flavor=initial_flavor, nubar=nubar, E_qg_eV=None)
    plot_LoE(ax=ax1, E_GeV=E_GeV_values, L_km=L_km, P=Pstd, color="red", linestyle="--", label="Std osc")

    #TODO REMOVE
    # Pstd = calc_2flav_survival_prob(calculator=calculator, E_GeV=E_GeV_values, L_km=L_km, flavor=initial_flavor, nubar=nubar, E_qg_eV=None, use_deimos=True)
    # plot_LoE(ax=ax1, E_GeV=E_GeV_values, L_km=L_km, P=Pstd, color="green", linestyle="-.", label="Std osc (DEIMOS)")

    # Plot QG
    m1_eV = 1. # From caption
    E_qg_eV = 1e24 * GeV_to_eV # From caption
    qg_model = "metric_fluctuations"

    #TODO I need to apply this correctiuon to reproduce KamLAND plot, but not for SuperK. Why?
    E_qg_eV *= 1e-5

    Pqg = calc_2flav_survival_prob(calculator=calculator, E_GeV=E_GeV_values, L_km=L_km, flavor=initial_flavor, nubar=nubar, E_qg_eV=E_qg_eV, m1_eV=m1_eV, qg_model=qg_model)
    plot_LoE(ax=ax1, E_GeV=E_GeV_values, L_km=L_km, P=Pqg, color="blue", linestyle=":", label="QG")

    # Format
    ax1.grid(True)
    ax1.legend()
    fig1.tight_layout()



    #
    # Fig 3 : SuperK
    #

    # This is for KamLAND
    L_km = 10. # From caption
    LoE_values_km_per_GeV = np.geomspace(1., 1e4, num=100) # Match range of Figure
    E_GeV_values = np.sort( L_km / (LoE_values_km_per_GeV) )
    initial_flavor, final_flavor, nubar = 0, 0, True # antinumu survival

    # Set osc params from figure caption
    calculator.set_mixing_angles( np.arcsin(np.sqrt(0.99)) / 2. )
    calculator.set_mass_splittings( 2.45e-3 )

    # Calculations assume survival
    assert initial_flavor == final_flavor

    # Make fig
    fig3, ax3 = plt.subplots( figsize=(6, 4) )

    # Plot std osc
    Pstd = calc_2flav_survival_prob(calculator=calculator, E_GeV=E_GeV_values, L_km=L_km, flavor=initial_flavor, nubar=nubar, E_qg_eV=None)
    plot_LoE(ax=ax3, E_GeV=E_GeV_values, L_km=L_km, P=Pstd, color="red", linestyle="--", label="Std osc")


    # Plot QG case from paper
    E_qg_eV = 1e30 * GeV_to_eV # From caption
    qg_model = "metric_fluctuations"
    Pqg = calc_2flav_survival_prob(calculator=calculator, E_GeV=E_GeV_values, L_km=L_km, flavor=initial_flavor, nubar=nubar, E_qg_eV=E_qg_eV, m1_eV=m1_eV, qg_model=qg_model)
    plot_LoE(ax=ax3, E_GeV=E_GeV_values, L_km=L_km, P=Pqg, color="blue", linestyle=":", label=qg_model.replace("_", " ").title())

    # Add another QG case of interest
    E_qg_eV = 1e0 * GeV_to_eV # Choosing something with reasonable signal
    qg_model = "minimal_length_fluctuations"
    Pqg = calc_2flav_survival_prob(calculator=calculator, E_GeV=E_GeV_values, L_km=L_km, flavor=initial_flavor, nubar=nubar, E_qg_eV=E_qg_eV, m1_eV=m1_eV, qg_model=qg_model)
    plot_LoE(ax=ax3, E_GeV=E_GeV_values, L_km=L_km, P=Pqg, color="magenta", linestyle=":", label=qg_model.replace("_", " ").title())

    # Format
    ax3.set_xscale("log")
    ax3.grid(True)
    ax3.legend()
    fig3.tight_layout()

    # Cross check standard oscillogram
    if False :
        E_GeV = np.geomspace(0.1, 100., num=100)
        coszen = np.linspace(-1., 1., num=100)
        E_GeV_grid, coszen_grid = np.meshgrid(E_GeV, coszen, indexing="ij")
        L_km_grid = calc_path_length_from_coszen(coszen)
        P = calc_disappearance_prob_2flav_vacuum(E_GeV=E_GeV_grid, L_km=L_km_grid, mass_splitting_eV2=calculator.get_mass_splittings()[0], theta_rad=calculator.get_mixing_angles()[0])
        fig, ax = plt.subplots(figsize=(7,7))
        plot_colormap( ax=ax, x=E_GeV, y=coszen, z=P, vmin=0., vmax=1., cmap="jet" )
        ax.set_xscale("log")
        fig.tight_layout()


def compare_qg_models(solver) :
    '''
    Compare decoherence resulting from a range on QG models, for the case of atmospheric neutrinos
    '''

    pass #TOOD plot osc probs for each model


def compare_qg_models_coherence_length() :
    '''
    Compare the natural coherence length for a range of QG models
    '''

    # Steer standard osc physics
    mass_splitting_eV2 = MASS_SPLITTINGS_eV2[-1] # atmo
    mixing_angle_rad = MIXING_ANGLES_rad[-1] # atmo

    # Steer QG
    E_qg_eV = PLANCK_MASS_eV
    m1_eV = 1.

    # Decide E range
    E_eV = np.logspace(0., 30, num=100)

    # Make figure
    fig, ax = plt.subplots( figsize=(6, 4) )

    # Mark Planck scale
    ax.axvline(PLANCK_MASS_eV, linestyle="-", lw=1, color="black", label="Planck scale")
    ax.axhline(PLANCK_LENGTH_m, linestyle="-", lw=1, color="black")

    # Mark Earth diameter
    ax.axhline(EARTH_DIAMETER_km*1e3, linestyle="-", lw=1, color="brown", label="Earth diameter")

    # Calc and plot coherence length for each model
    L_coh_m = calc_coherence_length(E_GeV=(E_eV/GeV_to_eV), theta_rad=mixing_angle_rad, mass_splitting_eV2=mass_splitting_eV2, qg_model="minimal_length_fluctuations", E_qg_eV=E_qg_eV, m1_eV=m1_eV) * 1e3 # km -> m
    ax.plot(E_eV, L_coh_m, color="red", label="Minimal length fluctuations", linestyle="-", lw=2)

    L_coh_m = calc_coherence_length(E_GeV=(E_eV/GeV_to_eV), theta_rad=mixing_angle_rad, mass_splitting_eV2=mass_splitting_eV2, qg_model="metric_fluctuations", E_qg_eV=E_qg_eV, m1_eV=m1_eV) * 1e3 # km -> m
    ax.plot(E_eV, L_coh_m, color="blue", label="Metric fluctuations", linestyle="-", lw=2)

    for n, linestyle in zip([0, 1, 2, 3], ["-", "--", "-.", ":"]) :
        L_coh_m = calc_coherence_length(E_GeV=(E_eV/GeV_to_eV), theta_rad=mixing_angle_rad, mass_splitting_eV2=mass_splitting_eV2, qg_model="nu_vbh_interactions", E_qg_eV=E_qg_eV, n=n) * 1e3 # km -> m
        ax.plot(E_eV, L_coh_m, color="orange", label=r"$\nu$-VBH ($n$=%i)"%n, linestyle=linestyle, lw=2)

    # Format
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$E_{\nu}$ [eV]")
    ax.set_ylabel(r"$L_{coh}$ [m]")
    ax.grid(True)
    ax.legend(fontsize=8)
    fig.tight_layout()




#
# Main
#

if __name__ == "__main__" :

    solver = "deimos"

    reproduce_2306_14778(solver=solver)
    compare_qg_models_coherence_length()
    compare_qg_models(solver=solver)

    print("")
    dump_figures_to_pdf( __file__.replace(".py",".pdf") )
