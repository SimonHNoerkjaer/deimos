'''
Plot neutrino oscillations with (isotropic) SME parameters activated

Tom Stuttard
'''

import sys, os, collections

from deimos.wrapper.osc_calculator import *
from deimos.utils.plotting import *
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
    parser.add_argument("-n", "--num-points", type=int, required=False, default=100, help="Num scan point")
    args = parser.parse_args()


    #
    # Define neutrino
    #

    initial_flavor, nubar = 1, False # muon neutrino
    E_GeV = np.geomspace(1., 1e4, num=args.num_points)
    coszen = -1.


    #
    # Create model
    #

    # For nuSQuIDS case, need to specify energy nodes covering full space
    kw = {}
    if args.solver == "nusquids" :
        kw["energy_nodes_GeV"] = E_GeV
        kw["nusquids_variant"] = "sme"
    elif args.solver == "oscprob" :
        kw["oscprob_variant"] = "liv"

    # Create calculator
    calculator = OscCalculator(
        solver=args.solver,
        atmospheric=False,
        **kw
    )

    # Use vacuum (earth is only for atmospheric neutrinos)
    calculator.set_matter("vacuum")


    #
    # Define physics cases
    #


    # CHoose basis
    sme_basis = "mass"    # "mass" or "flavor" (nuSQuIDS only supports mass basis)

    # Choose which flavor/state element to make non-zero
    # Can be multiple, but here only doing one for a simple example
    state_indices = (2,3)
    state_indices_label = "%i%i"%state_indices if (sme_basis == "mass") else "%s%s" % tuple([ calculator.get_flavor_tex(i-1) for i in state_indices ])

    # Define some operators to test
    cases = collections.OrderedDict()
    a_mag_eV = 1e-14
    c_mag = 1e-26
    cases[r"$a^t_{%s}$ = %0.3g eV"%(state_indices_label, a_mag_eV)] = ( get_sme_state_matrix(**{"p%i%i"%state_indices:a_mag_eV}), None ) # Non-zero a
    cases[r"$c^{tt}_{%s}$ = %0.3g"%(state_indices_label, c_mag)] = ( None, get_sme_state_matrix(**{"p%i%i"%state_indices:c_mag}) ) # Non-zero c


    #
    # Loop over cases
    #

    for case_label, (a_eV, c) in cases.items() :

        # Report
        print("")
        print("Model : %s" % case_label)


        #
        # Plot oscillation vs energy
        #

        # Calc osc probs and plot, without new physics
        calculator.set_std_osc()
        fig, ax, _, = calculator.plot_osc_prob_vs_energy(initial_flavor=initial_flavor, energy_GeV=E_GeV, distance_km=EARTH_DIAMETER_km, xscale="log", color="black", label="Standard osc", title=r"coszen = %0.3g"%coszen)

        # Calc osc probs and plot, with SME
        calculator.set_sme_isotropic(basis=sme_basis, a_eV=a_eV, c=c)
        calculator.plot_osc_prob_vs_energy(initial_flavor=initial_flavor, energy_GeV=E_GeV, distance_km=EARTH_DIAMETER_km, xscale="log", color="orange", label=r"SME : %s"%(case_label), linestyle="--", fig=fig, ax=ax)


    #
    # Done
    #

    print("")
    dump_figures_to_pdf( __file__.replace(".py",".pdf") )
