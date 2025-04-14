"""
Calculate neutrino oscillation probability using density matrices
Optionally can define:
    - Matter potential
    - Decoherent open quantum system term

Useful references:
  [1] https://ocw.mit.edu/courses/chemistry/5-74-introductory-quantum-mechanics-ii-spring-2009/lecture-notes/MIT5_74s09_lec12.pdf

Tom Stuttard
"""

#TODO Would a class be better here?

import numpy as np
from scipy.integrate import solve_ivp
try :
    from odeintw import odeintw
except :
    raise Exception("ERROR : Could not import `odeintw`, ensure it is installed and visible in your `PATH`/`PYTHONPATH` ")

# DEIMOS basic tools
from deimos.utils.constants import *
from deimos.utils.matrix_algebra import *
from deimos.models.decoherence.decoherence_operators import get_complete_sun_matrix, get_decoherence_operator_nxn_basis
from deimos.utils.coordinates import *

# Import new physics models
from deimos.models.decoherence.decoherence_operators import get_complete_sun_matrix, get_decoherence_operator_nxn_basis
from deimos.models.liv.sme import get_sme_hamiltonian


#
# Globals
#

# Unit conversions
km_to_eV = 5.06773093741e9 # [km] -> [1/eV]
GeV_to_eV = 1.e9 # [GeV] -> [eV]
hbar = 6.5821195691e-16 # [eV s]


#
# Basic mathematical operations for density matrices
#

'''
Here providing the basic mathematical functions for using density matrices.
Everything is treated as complex matrices.

Neutrino-specific stuff like the PMNS matrix and mass <-> flavor basis transformations are also provided.

Have kept these outside of the actual calculator class below since they are very general and can be imported 
easily and used elsewhere.
'''

#TODO move these somewhere common

def get_rho(psi) :
    """Get denisty matrix, rho, given a state (ket), psi: rho = |psi><psi| (outer product) """
    return np.outer(psi,psi).astype(np.complex128)


def get_pmns_matrix(theta, dcp=0.) :
    """ Get the PMNS matrix (rotation from mass to flavor basis)"""
    #TODO Majorana phases
    if len(theta) == 1 :
        assert (dcp is None) or np.isclose(dcp, 0.), "CP phase not defined for 2-flavor systems"
        # This is just the standard unitary rotation matrix in 2D
        pmns = np.array( [ [np.cos(theta[0]),np.sin(theta[0])], [-np.sin(theta[0]),np.cos(theta[0])] ], dtype=np.complex128 )
    elif len(theta) == 3 :
        # Using definition from https://en.wikipedia.org/wiki/Neutrino_oscillation
        pmns = np.array( [
            [   np.cos(theta[0])*np.cos(theta[1]),                                                                        np.sin(theta[0])*np.cos(theta[1]),                                                                        np.sin(theta[1])*np.exp(-1.j*dcp)  ], 
            [  -np.sin(theta[0])*np.cos(theta[2]) -np.cos(theta[0])*np.sin(theta[2])*np.sin(theta[1])*np.exp(1.j*dcp),    np.cos(theta[0])*np.cos(theta[2]) -np.sin(theta[0])*np.sin(theta[2])*np.sin(theta[1])*np.exp(1.j*dcp),    np.sin(theta[2])*np.cos(theta[1])  ], 
            [  np.sin(theta[0])*np.sin(theta[2]) -np.cos(theta[0])*np.cos(theta[2])*np.sin(theta[1])*np.exp(1.j*dcp),    -np.cos(theta[0])*np.sin(theta[2]) -np.sin(theta[0])*np.cos(theta[2])*np.sin(theta[1])*np.exp(1.j*dcp),    np.cos(theta[2])*np.cos(theta[1])  ],
        ], dtype=np.complex128 )
    else :
        raise Exception("Only 2x2 or 3x3 PMNS matrix supported")
    assert is_unitary(pmns), "PMNS matrix must be unitary"
    return pmns


def get_mass_splittings_squared_array(mass_splittings_squared) :
    """ Get the N-1 mass splittings and fill the NxN matrix (as seen in the Hamiltonian), where N is num neutrino states """
    num_states = len(mass_splittings_squared) + 1
    assert num_states > 1
    ms_array = np.zeros( (num_states,num_states) )
    for i in range(num_states-1) :
        ms_array[i+1,i+1] = mass_splittings_squared[i]
    return ms_array


def psi_flav_to_mass_basis(psi_flav,pmns) :
    """ Rotate wavefunction from flavor to mass basis: U^dagger psi"""
    return np.dot( dagger(pmns), psi_flav ).astype(np.complex128)


def psi_mass_to_flav_basis(psi_mass,pmns) :
    """ Rotate wavefunction from mass to flavor basis: U psi"""
    return np.dot( pmns, psi_mass ).astype(np.complex128)


def rho_flav_to_mass_basis(rho_flav,pmns) :
    """ Rotate density matrix from flavor to mass basis: U^dagger rho U """
    assert is_square(rho_flav)
    assert is_square(pmns)
    return np.dot( dagger(pmns), np.dot(rho_flav,pmns) ).astype(np.complex128)


def rho_mass_to_flav_basis(rho_mass,pmns) :
    """ Rotate density matrix from mass to flavor basis: U rho U^dagger """
    assert is_square(rho_mass)
    assert is_square(pmns)
    return np.dot( pmns, np.dot(rho_mass,dagger(pmns)) ).astype(np.complex128)


def rho_flavor_prob(rho_flav,flav) :
    """ 
    Get the projection of a flavor basis vector onto a density matrix (which is itself expressed in the flavor basis).
    This gives the probability of detecting the neutrino as this flavor
    Can use either: P = <psi|rho|psi>
    Or: P(a->b) = Tr[rho_a(t) * rho_b(0)]
    """
    psi_flav = np.zeros( rho_flav.shape[0], dtype=np.complex128 ) # Create empty flvor vector, using rho shape to get num flavors
    psi_flav[flav] = 1. + 0.j # Set the desired flavor to 1
    #flav_prob = np.dot( psi_flav.T, np.dot(rho_flav,psi_flav) )
    flav_prob = np.abs(np.trace( np.dot(rho_flav,get_rho(psi_flav)) ))
    #assert np.isreal(flav_prob), "Falvor probability must be real, something has gone wrong (P=%s)"%flav_prob
    assert np.isclose(flav_prob.imag,0.), "Flavor probability must be real, something has gone wrong (P=%s)"%flav_prob #np.isreal doesn't work here, as struggles with ~0 values (results from numerical integration precision)
    flav_prob = flav_prob.real
    return flav_prob


def psi_flavor_prob(psi_flav,flav) : 
    """ 
    Project flavor basis vector onto current state of wavefunction to get osc prob
    P(a->b) = |<psi_b(t)|psi_a(0)>|^2
    """
    target_psi_flav = np.zeros_like( psi_flav, dtype=np.complex128 ) # Create empty flvor vector, using rho shape to get num flavors
    target_psi_flav[flav] = 1. + 0.j # Set the desired flavor to 1
    inner_prod = np.inner(psi_flav,target_psi_flav)
    flav_prob = inner_prod * np.conj(inner_prod)
    assert np.isclose(flav_prob.imag,0.), "Flavor probability must be real, something has gone wrong (P=%s)"%flav_prob #np.isreal doesn't work here, as struggles with ~0 values (results from numerical integration precision)
    flav_prob = flav_prob.real
    return flav_prob


def get_mass_projection(rho_mass,mass_index) :
    """ 
    Get the projection of a mass basis vector onto a density matrix (which is itself expressed in the mass basis).
    This is analagous to the flavor probability, but gets the current mass composition of the density matrix rather than flavor composition
    """
    psi_mass = np.zeros( rho_mass.shape[0], dtype=np.complex128 ) # Create empty flvor vector, using rho shape to get num flavors
    psi_mass[mass_index] = 1. + 0.j # Set the desired mass eigenstate to 1
    mass_proj = np.dot( psi_mass.T, np.dot(rho_mass,psi_mass) )
    assert np.isclose(mass_proj.imag,0.), "Mass projection must be real, something has gone wrong" #np.isreal doesn't work here, as struggles with ~0 values (results from numerical integration precision)
    mass_proj = mass_proj.real
    return mass_proj


#
# Matter effects
#

def get_electron_density_per_m3(matter_density_g_per_cm3, electron_fraction) : #TODO is this really N_e (e.g. number, not density?) need Avogadro's number in here then
    '''
    Calculate an SI units electron number density from :
      - the overall matter density (in g/cm^2 as it is usually reported)
      - electron fraction 
    '''
    # return matter_density_g_per_cm3 * np.power(1.e-2,-3) * electron_fraction
    return AVOGADROS_NUMBER * matter_density_g_per_cm3 * 7.68351e-15 * electron_fraction #TODO lifted this conversion factor from nuSQuIDS (pow(cm, -3)) but need to verify it and implement properly, think it is for mol -> eV conversion but


def get_matter_potential_flav(flavors, matter_density_g_per_cm3, electron_fraction, nsi_matrix=None) :
    '''
    Calculate the matter potential (in the flavor basis) corresponding to a given matter density and electron fraction
    This assumes constant density matter

    Important: NOT flipping sign for antineutrinos (let solver do that)


    Matter potentiual due to scattering on quarks and electrons is:

    V = [ CC+NC   0     0   ]
        [   0     NC    0   ]
        [   0     0     NC  ]

    where the CC term only affects electron neutrinos (since there are no muons/taus in the Earth).

    For a 2/3 neutrino system (e.g. only active neutrinos), the NC is a global phase and has no impact on oscillations (but matters for steriles) and therefore can be neglected, giving:

    V_e = [  CC   0    0  ]
          [  0    0    0  ]
          [  0    0    0  ]

    and the CC term is: sqrt(2) * G_F * n_e        (Fermi constant, and number of electrons)

    Get number of electrons from the matter density as follows:

    n_e = n_A * matter density * electron fraction     (Avogadro's number)    - TODO this is wrong, fix it
    '''

    # Calculate the CC term
    V_CC = np.sqrt(2.) * FERMI_CONSTANT * get_electron_density_per_m3(matter_density_g_per_cm3, electron_fraction)

    # Create the matter potentiual matrix, marking the [e,e] term as V_CC. This is generally [0,0], but user might choose a 2-nu system with mu-tau only.
    num_states = len(flavors)
    V = np.zeros( (num_states,num_states), dtype=np.complex128 )
    for i, f in enumerate(flavors) :
        if f == "e" :
            V[i, i] = 1.

    # Include the NSI matrix, if present
    if nsi_matrix is not None :
        V = V + nsi_matrix

    # Now apply the constants
    V *= V_CC                       #TODO Is this wrong for NSI, since assumes CC rather than NC?

    return V


#
# Solver
#

# def get_rho_dot(L, E, H, rho, calc_basis, L_power=None, E_power=None, **kw) :
def get_rho_dot(H, rho, calc_basis, D_matrix=None, D_matrix_basis=None) :
    '''
    Compute the time derivative of the density matrix, rho, e.g. d(rho)/dt.

    This is what we will numerically solve to get the time dependence of the system, e.g. rho(t), 
    and compute oscillation probabilities.

    Standard (mass splitting + mixing) oscillations, as well as matter effects (including NSI), 
    are contained in the Hamiltonian (H). A (potentially) non-unitary Lindblad operator D[rho]
    can also be included.

    Expansion of the various Hermitian operators involved (rho, H, D[rho]) into the SU(N) basis
    is a common method to solve this. Can expand the whole term of just the D[rho] part (have checked
    that this is equvialent).

    There are two equivalent methods in the literature, depending on the basis used:

      1) d(rho)/dt = i[H,rho] - D[rho]   - NxN basis

      2) d(rho)/dt = M * rho             - SU(N) basis   (M = M_H + M_D)

    Note that for case 1, one is still able to define your decoherence free parameters in the 
    SU(N) basis, and thus compute D[rho] in the SU(N) basis, as long as you convert D[rho] back 
    to NxN basis to calculate D[rho].

    Ultimately, return d(rho)/dt in the NxN basis (as this solver solves in that basis)
    '''

    #
    # Case 1 - NxN basis
    #

    if calc_basis == "nxn" :

        # Checks on rho (basically just sanity checking for myself that these conditions are real properties of a density matrix)
        if False :
            for i, j in [ [0, 1], [0, 2], [1, 2]] :
                assert np.isclose(rho[i,j].real, rho[j,i].real), "rho[%i,%i].real != rho[%i,%i].real (%0.3g != %0.3g)" % (i,j, j,i, rho[i,j].real, rho[j,i].real)
                assert np.isclose(rho[i,j].imag, -rho[j,i].imag), "rho[%i,%i].imag != -rho[%i,%i].imag (%0.3g != %0.3g)" % (i,j, j,i, rho[i,j].imag, rho[j,i].imag)

        # Calculate the Hamiltonian contribution to the time derivative
        rho_dot = -1j * commutator(H, rho)

        # Get the decoherence operator, and subtract it from the time derivative
        if D_matrix is not None :
            assert D_matrix_basis is not None
            D_rho = get_decoherence_operator_nxn_basis(rho=rho, D_matrix=D_matrix, D_matrix_basis=D_matrix_basis).astype(np.complex128)
            rho_dot -= D_rho


    #
    # Case 2 - SU(N) basis
    #

    #TODO Am I missing the rho0 coefficient somewhow? See 2001.09250 eq 2.4 and 1711.03680 page 2 text

    elif calc_basis == "sun" :

        # Check inputs
        assert "gamma_matrix" in kw
        gamma_matrix = kw["gamma_matrix"]

        # I user omitted the 0'th rw./col, add it (it is all zeros)
        gamma_matrix_sun = get_complete_sun_matrix(num_states=rho.shape[0], sun_matrix=gamma_matrix)

        # Decompose rho
        rho_sun = convert_matrix_to_sun_decomposition_coefficients(rho)

        # Get the decoherence matrix
        M_D = -1. * gamma_matrix_sun

        # Get expanded Hamiltonian matrix
        # Have two separate methods here, need to sort this out a bit...
        M_H = np.zeros_like(M_D)
        if True :
            # See 2001.09250 eq 2.7
            assert rho.shape[0] == 3, "Only implemented for 3 neutrino flavors currently"
            Delta_21 = H[1,1]
            Delta_31 = H[2,2]
            Delta_32 = Delta_31 - Delta_21
            M_H[1,2] = Delta_21 * -1.
            M_H[2,1] = Delta_21
            M_H[4,5] = Delta_31 * -1.
            M_H[5,4] = Delta_31
            M_H[6,7] = Delta_32 * -1.
            M_H[7,6] = Delta_32
        else :
            # 1711.03680 eq 8 (see also 2001.07580 eq 9)
            H_sun = convert_matrix_to_sun_decomposition_coefficients(H)
            for j in range(len(H_sun)) : #TODO general
                M_H[k,j] += 2. * np.sum([ SU3_STRUCTURE_CONSTANTS[i,j,k] * H_sun[i] for i in range(9) ])

        # Get overall matrix
        M = M_H + M_D

        # Compute d(rho)/dt
        rho_dot_sun = np.array([
            np.sum([ ( M[i,j] * rho_sun[j] ) for j in range(rho_sun.size) ]) for i in range(rho_sun.size)
        ], dtype=np.complex128)

        # Convert back to NxN
        rho_dot = convert_sun_decomposition_coefficients_to_matrix(rho_dot_sun)


    #
    # Done
    #

    else :
        raise Exception("Unknown `calc_basis` '%s', choose from ['nxn','sun']" % calc_basis)

    return rho_dot


class DensityMatrixOscSolver(object) :
    '''
    Class for calculating the time-dependence of a neutrino oscillation system,
    expressed in the density matrix formalism.

    Solution is obtained numerically, by default using `odetintw` but other options are implemented

    Includes support for:
     - Matter effects
     - Decoherence
     - Lightcone fluctuations
     - Standard Model Extension (SME)
    '''

    def __init__(self,
        # Define physics
        num_states=NUM_STATES,
        dm2_eV2=None, # Mass spittings
        theta_rad=None, # Mixing angles
        # Solver options
        rtol=1.e-12,
        atol=1.e-12,
        mxstep=10000000, # Crank this up if get "Solver failed : Excess work done on this call (perhaps wrong Dfun type)" errors
        ode_solver=None,
        ode_solver_method=None,
    ) :

        # Store args
        self.num_states = num_states
        self.rtol = rtol
        self.atol = atol
        self.mxstep = mxstep
        self.ode_solver = ode_solver
        self.ode_solver_method = ode_solver_method
        
        # Init and check solver definition
        if self.ode_solver is None :
            self.ode_solver = "odeintw"
        assert self.ode_solver in ["solve_ivp", "odeintw"], "solver not implemented. Use either 'solve_ivp' or 'odeintw'. "
            
        if self.ode_solver == "solve_ivp" :
            if self.ode_solver_method is None :
                self.ode_solver_method = 'RK45'
        else :
            assert self.ode_solver_method is None, "`ode_solver_method` arg not valid for %s" % self.ode_solver
            
        # Init some internal states
        self.matter_potential_flav_eV = None
        if theta_rad is None :
            theta_rad =  MIXING_ANGLES_rad[:(1 if self.num_states == 2 else 3)]
        self.set_mixing_angles(theta_rad) #TODO More general function to support arbitrary num states
        if dm2_eV2 is None :
            dm2_eV2 = MASS_SPLITTINGS_eV2[:self.num_states-1] 
        self.set_mass_splittings(dm2_eV2)

        # Checks
        assert self.num_states > 1


    def set_mixing_angles(self, theta_rad, deltacp=0.) :
        self.theta_rad = theta_rad
        self.deltacp = deltacp
        self.PMNS = get_pmns_matrix(self.theta_rad, dcp=self.deltacp)
        # self.PMNS_inv = get_pmns_matrix(-1.*self.theta_rad, dcp=deltacp) #TODO check this
        assert self.PMNS.shape == (self.num_states,self.num_states,)


    def set_mass_splittings(self,mass_splittings_eV2) :
        assert mass_splittings_eV2.size == 1 if self.num_states == 2 else 3
        self.dm2_eV2 = get_mass_splittings_squared_array(mass_splittings_eV2)
        assert self.dm2_eV2.shape == (self.num_states,self.num_states,)


    def get_mass_splittings(self) :
        ret = [ self.dm2_eV2[1, 1] ]
        if self.num_states == 3 :
            ret.append( self.dm2_eV2[2, 2] )
        return tuple(ret)


    def set_matter_potential(self, matter_potential_flav_eV) :
        '''
        Set the matter potential, in the flavor basis
        '''
        self.matter_potential_flav_eV = matter_potential_flav_eV
    
    
    def time_evolution_operator(self, H, L):
        return np.exp(-1j*H*L)
    

    def _solve(self,
        # Standard neutrino solver 
        initial_rho_mass, 
        E, 
        L, 
        H, 
        calc_basis,
        # Decoherence operator
        D_matrix=None,
        D_matrix_basis=None,
    ) :
        '''
        Solve the density matrix time evolution to get the final state for this system
        '''

        #
        # Derive function
        #

        # Define the time derivative of the density matrix: d(rho)/dt = -i[H,rho] (-D[rho])
        # The additional -D[rho] term if for the case of decoherence: D[rho]_ij = rho_ij * Gamma_ij
        # We will use this to numerically solve for rho and integrate
        # Anything depending on `rho` or `L` must be done in here

        def derive(L, rho, decoh_D_matrix_basis=None, decoh_D_matrix=None, flatten=False): #TODO DO the decoh D martrix things actually need to be args?

            # Handle lightcone fluctuations here (since depends on L, not just E)
            # if include_lightcone_fluctuations :
            #     from deimos.utils.model.lightcone_fluctuations.lightcone_fluctuation_model import get_lightcone_decoherence_D_matrix
            #     decoh_D_matrix_basis, decoh_D_matrix = get_lightcone_decoherence_D_matrix(num_states=self.num_states, H=H, E=E_val, L=L, m=lightcone_m, n=lightcone_n, dL0=lightcone_dL0, L0=lightcone_L0, E0=lightcone_E0) 

            # Handle cases where the solver can only pss 1D arrays
            if flatten :
                rho = rho.reshape(self.num_states, self.num_states)

            # Compute rho derivative
            rho_dot = get_rho_dot(H=H, rho=rho, calc_basis=calc_basis, D_matrix=decoh_D_matrix, D_matrix_basis=decoh_D_matrix_basis)
            
            # Handle cases where the solver can only pss 1D arrays
            if flatten :
                rho_dot = rho_dot.flatten()

            return rho_dot


        #
        # Solve ODE to get rho (mass basis) for each L (e.g. time evolution of system)
        #

        # This depends a little on the solver in question
        if self.ode_solver == "odeintw" :

            #
            # odeintw
            #

            # Solve
            solved_rho_mass, infodict = odeintw(
                derive, # d(rho)/dL
                initial_rho_mass, # rho(0)
                L, # L
                args=(D_matrix_basis, D_matrix, False),  # Args to pass to `derive` (other than `L, rho`). Note that `False` here means "don't flatten arrays"
                full_output=True,
                rtol=self.rtol,
                atol=self.atol,
                mxstep=self.mxstep,
                tfirst=True, # This means `derive` is derive(L,rho), not derive(rho,L). Doing this for consistency with solve_ivp
            )

            # Check
            assert infodict["message"] == "Integration successful.", "Solver failed : %s" % infodict["message"]


        elif self.ode_solver == "solve_ivp" :

            #
            # solve ivp
            #

            # Get the span of L values for the solver
            L_range = (np.nanmin(L), np.nanmax(L)) #TODO once only for speed?    #TODO get 0'tn and -1'th elements instead for speed?
            if L_range[0] == L_range[1] :
                L_range = (0., L_range[1])  #TODO better way of doing this?

            # Solve
            # Note that we are having to flatten rho matrices into 1D arrays here
            solved_rho_mass = solve_ivp(
                derive, # d(rho)/dL
                L_range, # range for which to solve ODE
                initial_rho_mass.flatten(), # rho(0) - shape is (L_size,), where N is number of L values    #TODO is the N correct?
                t_eval=L, # Define values of L for which want specific solutions
                method=self.ode_solver_method,
                args=(D_matrix_basis, D_matrix, True), # Args to pass to `derive` (other than `L, rho`). Note that `True` here means "flatten arrays"
                rtol=self.rtol,
                atol=self.atol,  # atol + rtol * abs(y)
                # mxstep=self.mxstep,
            )

            # Check solved successfully
            assert solved_rho_mass.success == True, "Solved failed : %s" % solved_rho_mass.message

            # Get the actual result, unflatten the rho matrices, and re-order to same dimensions as odeintw produces for consistency
            solved_rho_mass = np.transpose(solved_rho_mass.y).reshape(L.size, self.num_states, self.num_states)

        else :
            raise Exception("Unknown solver : %s" % self.ode_solver)

        # Track the entropy of the system
        #TODO

        return solved_rho_mass # Shape is (L.size, N, N)



    def calc_osc_probs(self,
        # Neutrino properties
        E_GeV,
        L_km,
        initial_state,
        initial_basis="flavor", # Optionally can set initial state in mass basis
        nubar=False,

        # Options to be passed to the decoherence calculator
        decoh_opts=None,
        lightcone_opts=None,

        # Options to be passed to the SME calculator
        sme_opts=None,
        
        # Misc
        calc_basis="nxn", # Optionally can choose which basis to perform calculation in (nxn, sun)
        verbose=False,
        plot_density_matrix_evolution=False, # Flag for plotting the solved density matrix vs distance/energy
    ) :
        '''

        Returns:
            osc_probs : array
                Oscillation probability from the chosen initial flavor to all possible final 
                flavors, for each distamce and energy value requested.
                Indexing : [ E value, L value, final state flavor ]
        '''

        #
        # Check inputs
        #

        # Check basis
        assert initial_basis in BASES, "Unknown `initial_basis` '%s', choose from %s" % (initial_basis,BASES)

        # Get states
        states = np.array(range(0,self.num_states))

        # Check initial state is valid
        assert initial_state in states, "`%i` is not a valid initial flavor, choose from %s" % (initial_state,flavors)

        # Check energies and distances are arrays
        # If found a single value, turn into an array
        if np.isscalar(E_GeV) : 
            E_GeV = [E_GeV]
        E_GeV = np.asarray(E_GeV)
        assert E_GeV.ndim == 1

        if np.isscalar(L_km) : 
            L_km = [L_km]
        L_km = np.asarray(L_km)
        assert L_km.ndim == 1

        # If L array is not ascending run odeintw on each node separately 
        #(get problems if it isn't, not sure why, perhaps odeintw is requires sorted arrays?)
        calc_L_nodes_separately = False
        if np.all(np.diff(L_km) >= 0) == False:
            calc_L_nodes_separately = True

        # Will need to pass some kwargs to `derive` later, populate as I go
        rhodot_kw = {}

        # Report
        if verbose :
            print("Num flavors : %i" % self.num_states)
            print("Initial state : %i (%s basis)" % (initial_state,initial_basis))
            print("L : %s km" % L_km)
            print("E : %s GeV" % E_GeV)


        #
        # Decoherence parameters
        #

        decoh_D_matrix = None
        decoh_D_matrix_basis = None
        include_decoherence = False

        # Handle the optional decoherence effects
        if decoh_opts is not None :

            include_decoherence = True

            # User provides a D matrix and energy-depenedence
            assert "D_matrix0_eV" in decoh_opts # D matrix for E0
            assert "D_matrix_basis" in decoh_opts # The basis the D matrix is defined in (NxN or SU(N))
            assert "E0_eV" in decoh_opts
            assert "n" in decoh_opts

            # Grab the vars, handling units
            decoh_opts = copy.deepcopy(decoh_opts)
            decoh_D_matrix0 = decoh_opts.pop("D_matrix0_eV")
            #decoh_gamma0 = decoh_opts.pop("gamma0_eV") # Removed this line because decoh_gamma0 and decoh_D_matrix0 are redundant
            decoh_n = decoh_opts.pop("n")
            decoh_E0 = decoh_opts.pop("E0_eV")
            decoh_D_matrix_basis = decoh_opts.pop("D_matrix_basis") # Added this line!
            assert len(decoh_opts) == 0, "Unused decoherence arguments!?!"


        #
        # Lightcone fluctuation parameters
        #

        include_lightcone_fluctuations = False

        # Handle the optional lightcone fluctuation effects
        if lightcone_opts is not None :

            raise NotImplemented("TODO: Need to reintegrated lightcone fluctuations into updated solver code")

            include_lightcone_fluctuations = True

            # Check args
            assert "dL0_km" in lightcone_opts
            assert "L0_km" in lightcone_opts
            assert "E0_eV" in lightcone_opts
            assert "m" in lightcone_opts
            assert "n" in lightcone_opts

            # Grab vars, handling units
            lightcone_opts = copy.deepcopy(lightcone_opts)
            lightcone_dL0 = lightcone_opts.pop("dL0_km") * km_to_eV
            lightcone_L0 = lightcone_opts.pop("L0_km") * km_to_eV
            lightcone_E0 = lightcone_opts.pop("E0_eV")
            lightcone_n = lightcone_opts.pop("n")
            lightcone_m = lightcone_opts.pop("m")

        # Check not doubling up on decoherence
        assert not (include_lightcone_fluctuations and include_decoherence), "Currently only supporting a single type of decoherence at once"


        #
        # SME (LIV) parameters
        #

        include_sme = False

        if sme_opts is not None :
            
            # To include SME parameters in calculation of the hamiltonian
            include_sme = True

            # Copy the opts to avoid modifying
            sme_opts = copy.deepcopy(sme_opts)

            # Handle basis in which flavor/mass structure is defined
            assert "basis" in sme_opts
            sme_basis = sme_opts.pop("basis")
            assert sme_basis in ["mass", "flavor"]
            sme_basis_is_flavor = sme_basis == "flavor" # Bool fast checking during solving

            # User provides a^(3) and c^(4) coefficients
            sme_a_t = sme_opts.pop("a_t_eV", None)
            sme_a_x = sme_opts.pop("a_x_eV", None)
            sme_a_y = sme_opts.pop("a_y_eV", None)
            sme_a_z = sme_opts.pop("a_z_eV", None)
            sme_c_tt = sme_opts.pop("c_tt", None)
            sme_c_tx = sme_opts.pop("c_tx", None)
            sme_c_ty = sme_opts.pop("c_ty", None)
            sme_c_tz = sme_opts.pop("c_tz", None)
            sme_c_xx = sme_opts.pop("c_xx", None)
            sme_c_xy = sme_opts.pop("c_xy", None)
            sme_c_xz = sme_opts.pop("c_xz", None)
            sme_c_yy = sme_opts.pop("c_yy", None)
            sme_c_yz = sme_opts.pop("c_yz", None)
            sme_c_zz = sme_opts.pop("c_zz", None)

            # Get neutrino direction (only required for directional/sidereal LIV)
            ra_rad = sme_opts.pop("ra_rad", None)
            dec_rad = sme_opts.pop("dec_rad", None)

            # Handle antineutrinos
            if nubar:
                if sme_a_t is not None :
                    sme_a_t *= -1.
                if sme_a_x is not None :
                    sme_a_x *= -1.
                if sme_a_y is not None :
                    sme_a_y *= -1.
                if sme_a_z is not None :
                    sme_a_z *= -1.

            # Check for additional SME arguments
            assert len(sme_opts) == 0, "Unused SME arguments!?! : %s" % list(sme_opts.keys())


        #
        # Unit conversions
        #

        # Convert units (everything into energy-equivalent units, e.g. eV, /eV, etc)
        L = L_km * km_to_eV
        E = E_GeV * GeV_to_eV
        dm2 = self.dm2_eV2 # No conversion required


        #
        # Handle nuances of solver
        #
        
        # `odeintw` needs a number of `t` values (in this case `L`) to solve
        # These also need not to be bunched up too closely
        # Add nodes either side of the given L value to ensure good solution
        #TODO Can this be done better? what about if user defines a list of closely bnched values????
        #TODO Could I just solve for energy rather than distance in this case?
        L_nodes = L
        pad_L = False
        if self.ode_solver == 'odeintw':
            pad_L = L.size == 1
            if pad_L :
                L_nodes = np.array( [0., L[0]] )
            else :
                L_nodes = L


        #
        # Earth rotation
        #

        #TODO

        # max_L = np.max(L)

        # # The sidereal frequency is ~ 1e-20 eV. Therefore for L<1e19 the rotation of the earth during
        # # the propagation of the neutrinos can be neglected
        # include_earth_rotation_during_neutrino_prop = False
        # if include_sme:
        #     if max_L >= 1e19:
        #         print("Note: Taking Earth's rotation into account during neutrino propagation")
        #         include_earth_rotation_during_neutrino_prop = True
        

        #
        # Define physics
        #

        # Get the PMNS matrix
        PMNS = self.PMNS
        if verbose :
            print("\nPMNS matrix %s :" % (PMNS.shape,))
            print(PMNS)

        # Conjugate the PMNS matrix for antineutrinos
        if nubar :
            PMNS = np.conj(PMNS)

        # Define the mass component of the vacuum Hamiltonian
        # Do not consider the energy depenednce yet
        M = dm2.astype(np.complex128)
        if verbose :
            print("\nDelta mass matrix :")
            print(M)

        # Add a matter potential
        # Remember to rotate it to the mass basis
        V = self.matter_potential_flav_eV # Use V for shorthand...
        if V is not None :
            assert V.shape == M.shape, "Matter potential V shape does not match Hamiltonian shape"
            if verbose :
                print("\nMatter potential (flavor basis) :")
                print(V)
            V = rho_flav_to_mass_basis(V, PMNS) # Believe this is the correct way to rotate but need to double check (cross ref with nuSQuIDS)
            if verbose :
                print("\nMatter potential (mass basis) :")
                print(V)

        #TODO report effective mixing angles and mass splittings in matter potential


        #
        # Set initial conditions
        #

        # Get the initial state in the mass basis (we will calculate in the mass basis)
        # If user specified a flavor basis initial state, perform this conversion first

        if initial_basis == "flavor" :

            # Flavor wave function
            initial_psi_flav = np.zeros((self.num_states))
            initial_psi_flav[initial_state] = 1
            if verbose :
                print("\nInitial flavor state:")
                print(initial_psi_flav)

            # Flavor density matrix
            initial_rho_flav = get_rho(initial_psi_flav)
            if verbose :
                print("\nInitial density matrix (flavor basis):")
                print(initial_rho_flav)

            # Convert to mass basis
            initial_rho_mass = rho_flav_to_mass_basis(initial_rho_flav, PMNS)

        else :

            # Mass wave function
            initial_psi_mass = np.zeros((self.num_states))
            initial_psi_mass[initial_state] = 1
            if verbose :
                print("\nInitial mass state:")
                print(initial_psi_mass)

            # Mass density matrix
            initial_rho_mass = get_rho(initial_psi_mass)

        if verbose :
            print("\nInitial density matrix (mass basis):")
            print(initial_rho_mass)


        #
        # Solve for each energy
        #
        
        osc_prob_results = []

        # Loop over each energy value
        for E_val in E :


            #
            # Set energy-dependent properties
            #

            # Get the vacuum Hamiltonian at this energy
            H = M / (2. * E_val)

            # Add/subtract the matter hamiltonian for nu/nubar
            if V is not None :
                H = H - V if nubar else H + V

            # Add/subtract SME term to Hamiltonian     #TODO what about antineutrinos?
            if include_sme : 
                H_eff = get_sme_hamiltonian(
                    ra=ra_rad,
                    dec=dec_rad,
                    a_eV_t=sme_a_t,
                    a_eV_x=sme_a_x,
                    a_eV_y=sme_a_y,
                    a_eV_z=sme_a_z,
                    c_tt=sme_c_tt,
                    c_tx=sme_c_tx,
                    c_ty=sme_c_ty,
                    c_tz=sme_c_tz,
                    c_xx=sme_c_xx,
                    c_xy=sme_c_xy,
                    c_xz=sme_c_xz,
                    c_yy=sme_c_yy,
                    c_yz=sme_c_yz,
                    c_zz=sme_c_zz,
                    E=E_val,
                    num_states=self.num_states,
                )
                if sme_basis_is_flavor :
                    H_eff = rho_flav_to_mass_basis(H_eff, PMNS) # Rotate to mass basis
                H += H_eff

            # Handle decoherence gamma param (or D matrix) energy-depenedence
            # Using the `gamma` function, but actually applying to the whole matrix rather than the individual elements (which is equivalent)
            if include_decoherence :
                from deimos.models.decoherence.nuVBH_model import get_gamma
                decoh_D_matrix = get_gamma(gamma0_eV=decoh_D_matrix0, E_eV=E_val, E0_eV=decoh_E0, n=decoh_n)


            #
            # Run solver
            #

            # Call the solver function
            solved_rho_mass = self._solve(
                initial_rho_mass=initial_rho_mass, 
                E=E_val, 
                L=L_nodes, 
                H=H, 
                calc_basis=calc_basis,
                D_matrix=decoh_D_matrix,
                D_matrix_basis=decoh_D_matrix_basis,
            )

            # Remove the extra nodes added for solver stability
            if pad_L :
                mask = np.array( [False, True], dtype=bool )
                solved_rho_mass = solved_rho_mass[mask]

            # Convert solutions to flavor basis
            solved_rho_flav = np.array([ rho_mass_to_flav_basis(rm, PMNS) for rm in solved_rho_mass ])

            #TODO Remove, clean up, or add to `test`
            if False :
                if verbose :
                    print("\nSolved rho (flavor basis):")
                    for r in solved_rho_flav : 
                        print(r)

            # Get oscillation probabilties for each final state flavor
            osc_probs = np.full( (len(L),len(states),), np.nan ) # Indexing [L value, final state flavor]
            for i_L in range(0,len(L)) :
                for i_f,final_flav in enumerate(states) :
                    osc_probs[i_L,i_f] = rho_flavor_prob(solved_rho_flav[i_L,...],final_flav)

            # Store the results for this energy value
            osc_prob_results.append(osc_probs)


            #
            # Plot density matrix evolution
            #

            # Here can optionally plot the density matrix evolution for debugging/understanding purposes
            if plot_density_matrix_evolution :
                assert np.isscalar(E) or (len(E) == 1)
                self._plot_density_matrix_evolution(self, L_km, solved_rho_mass, solved_rho_flav)


        #
        # Done
        #
          
        # Numpy-ify
        osc_prob_results = np.asarray(osc_prob_results) # Indexing [E value, L value, final state flavor]

        if verbose :
            print("\nOscillation probabilities calculated")
            
        return osc_prob_results


    def _plot_density_matrix_evolution(self, L_km, solved_rho_mass, solved_rho_flav) :
        '''
        Can optionally plot the evolution of the density matrix elements
        '''

        # Check inputs
        assert L_km.size > 1
        assert solved_rho_mass.shape == (L_km.size, self.num_states, self.num_states)
        assert solved_rho_flavs.shape == (L_km.size, self.num_states, self.num_states)

        rho_mass_fig = Figure(nx=self.num_states, ny=self.num_states, title=r"$\rho_{\rm{mass}}$", sharex=True, sharey=True, figsize=(9,7), row_headings=True, col_headings=True) 
        rho_flav_fig = Figure(nx=self.num_states, ny=self.num_states, title=r"$\rho_{\rm{flavor}}$", sharex=True, sharey=True, figsize=(9,7), row_headings=True, col_headings=True) 
        figs = [rho_mass_fig, rho_flav_fig]
        for i in range(self.num_states) :

            for fig in figs :
                fig.set_row_heading(i, str(i) )

            for j in range(self.num_states) :

                if i == 0 :
                    for fig in figs :
                        fig.set_col_heading(j, str(j) )

                num_L = solved_rho_mass.shape[0]

                rho_mass_ij = np.array([ solved_rho_mass[k,i,j] for k in range(num_L) ], dtype=np.complex128)
                rho_flav_ij = np.array([ solved_rho_flav[k,i,j] for k in range(num_L) ], dtype=np.complex128)

                rho_mass_ax = rho_mass_fig.get_ax(x=i, y=j)
                rho_flav_ax = rho_flav_fig.get_ax(x=i, y=j)

                rho_mass_ax.plot(L_km, rho_mass_ij.real, color="red", linestyle="-", zorder=5)
                rho_mass_ax.plot(L_km, rho_mass_ij.imag, color="orange", linestyle="--", zorder=6)

                rho_flav_ax.plot(L_km, rho_flav_ij.real, color="blue", linestyle="-", zorder=5)
                rho_flav_ax.plot(L_km, rho_flav_ij.imag, color="lightblue", linestyle="--", zorder=6)

                for ax in [rho_mass_ax, rho_flav_ax] :
                    ax.axhline(0., color="black", linestyle="--", linewidth=1, zorder=4)
                    ax.axhline(1./float(self.num_states), color="black", linestyle="-.", linewidth=1, zorder=4)

        for fig in figs :
            fig.quick_format( ylim=(-1., +1.), xlim=(L_km[0],L_km[-1]) )
        rho_mass_fig.save("rho_mass.png")
        rho_flav_fig.save("rho_flav.png")


#
# Test the code
#

def tests() :

    # This tests the how the solver works for small numbers of L values

    fig = Figure()
    E_GeV = np.linspace(1.,5.,num=100)
    initial_state = 1
    initial_basis = "flavor"
    for i,L_km in enumerate([ [1299.,1300.,1301.], [100.,1300.,2000.], [100.,1299.,1300.,1301.,2000.] ]) :
        osc_probs = calc_osc_probs( E_GeV=E_GeV,
                                    initial_state=initial_state,
                                    initial_basis=initial_basis,
                                    L_km=L_km)
        center = ( len(L_km) - 1 ) / 2 
        print(center)
        osc_probs = osc_probs[:,center,:]
        fig.get_ax().plot( E_GeV, osc_probs[:,1], linestyle="--", label=str(i), alpha=0.2 )
    fig.get_ax().grid(True)
    fig.get_ax().legend()

    dumpFiguresToPDF("osc_solver_tests.pdf")


run_tests = False
if run_tests : 
    test()

    # TODO trace
    #TODO rho^2 = rho



#
# Main function
#

if __name__ == "__main__" :


    '''
    Show some basic oscillation probabilities using these tools
    '''

    # Control numpy printing
    #np.set_printoptions(suppress=True,precision=2)

    # Create solver (using default settings)
    solver = DensityMatrixOscSolver( num_states=3 )


    #
    # DUNE
    #
    
    # Define physics
    E_GeV = np.linspace(0.6,10.,num=1000)
    L_km = np.array([1300.])
    initial_state = 1

    # Calculate oscillation probability
    osc_probs = solver.calc_osc_probs( 
        E_GeV=E_GeV,
        initial_state=initial_state,
        L_km=L_km,
    )

    # Squeeze out the single-valued distance dimension
    osc_probs = np.squeeze(osc_probs)

    # Plot
    fig = Figure( ny=2, title="DUNE")
    fig.get_ax(y=0).plot( E_GeV,osc_probs[:,0], color="red", linestyle="-", label=r"$%s \rightarrow %s$"%(NU_FLAVORS_TEX[initial_state],NU_FLAVORS_TEX[0]) )
    fig.get_ax(y=1).plot( E_GeV,osc_probs[:,1], color="blue", linestyle="-", label=r"$%s \rightarrow %s$"%(NU_FLAVORS_TEX[initial_state],NU_FLAVORS_TEX[1]) )
    fig.get_ax(y=-1).set_xlabel(r'$E$ [GeV]')
    for ax in fig.get_all_ax() :
        ax.set_ylabel('Probability')
        ax.grid(True)
        ax.legend()
    fig.tight_layout()


    #
    # Daya Bay
    #

    # See http://www1.phys.vt.edu/~dayabay/about.html

    L_km = np.logspace(-1,2,num=1000)
    E_GeV = np.array([4.e-3])
    initial_state = 0
    
    # Calculate oscillation probability
    osc_probs = solver.calc_osc_probs(
        E_GeV=E_GeV,
        initial_state=initial_state,
        L_km=L_km,
    )

    # Squeeze out the single-valued energy dimension
    osc_probs = np.squeeze(osc_probs)

    # Plot
    fig = Figure( ny=1, title="Daya Bay")
    fig.get_ax(y=0).plot( L_km,osc_probs[:,0], color="red", linestyle="-", label=r"$%s \rightarrow %s$"%(NU_FLAVORS_TEX[initial_state],NU_FLAVORS_TEX[0]) )
    fig.get_ax(y=-1).set_xlabel(r'$L$ [km]')
    for ax in fig.get_all_ax() :
        ax.set_xscale('log')
        ax.set_ylim(0.,1.1)
        ax.set_ylabel('Probability')
        ax.grid(True)
        ax.legend()
    fig.tight_layout()


    #
    # Atmospheric neutrinos
    #

    # Define neutrinos
    L_km = np.linspace(0.,EARTH_DIAMETER_km,num=100)
#        E_GeV = np.linspace(1.,50.,num=100)
    E_GeV = np.array([10.,25.,100.])
    initial_state = 1
    
    # Calculate oscillation probability
    osc_probs = solver.calc_osc_probs(
        E_GeV=E_GeV,
        initial_state=initial_state,
        L_km=L_km,
    )

    # Plot
    fig = Figure( ny=1, title="Atmospheric")
    for i_E in range(0,E_GeV.size) :
        fig.get_ax(y=0).plot( L_km,osc_probs[i_E,:,1], linestyle="-", label=r"$E = %0.3g$ GeV"%E_GeV[i_E] )
    fig.get_ax(y=-1).set_xlabel(r'$L$ [km]')
    for ax in fig.get_all_ax() :
        ax.set_ylim(0.,1.1)
        ax.set_ylabel (r"$P( %s \rightarrow %s)$"%(NU_FLAVORS_TEX[initial_state],NU_FLAVORS_TEX[1]) )
        ax.grid(True)
        ax.legend()
    fig.tight_layout()


    #
    # Done
    #

    #Dump figures to PDF
    print("\nGenerating PDF...")
    dump_figures_to_pdf( replace_file_ext(__file__,".pdf") )

