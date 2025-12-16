'''
Wrapper class providing a common interface to a range of oscillation solvers,
both from this project and external.

Tom Stuttard
'''

# Import plotting tools
# Do this before anything else, as want the matplotlib backend handling dealt with before any other packages called
from deimos.utils.plotting import *

import sys, os, collections, numbers, copy, re
import numpy as np
import warnings
# import healpy as hp

# Import nuSQuIDS
NUSQUIDS_AVAIL = False
try:
    import nuSQuIDS as nsq # Modern
    NUSQUIDS_AVAIL = True
except ImportError as e:
    try:
        import nuSQUIDSpy as nsq # Old (backwards compatibility)
        NUSQUIDS_AVAIL = True
    except ImportError as e:
        pass

# Import nuSQuIDS decoherence implementation
NUSQUIDS_DECOH_AVAIL = False
try:
    from nuSQUIDSDecohPy import nuSQUIDSDecoh, nuSQUIDSDecohAtm
    NUSQUIDS_DECOH_AVAIL = True
except ImportError as e:
    pass

# Import prob3
PROB3_AVAIL = False
try:
    from BargerPropagator import * 
    PROB3_AVAIL = True
except ImportError as e:
    pass

# General DEIMOS imports
from deimos.utils.constants import *
# from deimos.models.decoherence.decoherence_operators import get_model_D_matrix
from deimos.density_matrix_osc_solver.density_matrix_osc_solver import DensityMatrixOscSolver, get_pmns_matrix, get_matter_potential_flav
from deimos.utils.oscillations import calc_path_length_from_coszen
from deimos.utils.coordinates import *


#
# Globals
#

DEFAULT_CALC_BASIS = "nxn"
DEFAULT_DECOHERENCE_GAMMA_BASIS = "sun"


#
# Calculator
#

class OscCalculator(object) :
    '''
    A unified interface to a range of oscillation + decoherence calculation tools.
    Allows easy comparison of methods.
    ''' 

    def __init__(self,
        tool, # Name of the underlying calculion tool
        atmospheric, # Bool indicating calculating in atmospheric parameter space (e.g. zenith instead of baseline)
        flavors=None,
        # Osc params
        mixing_angles_rad=None,
        mass_splittings_eV2=None,
        deltacp_rad=None,
        **kw
    ) :

        # Store args
        self.tool = tool
        self.atmospheric = atmospheric
        self.flavors = flavors

        # User must specify flavors, or take default
        if self.flavors is None :
            self.flavors = FLAVORS
        assert isinstance(self.flavors, list)
        assert len(self.flavors) == len(set(self.flavors)), "Duplicate flavors provided"
        assert all([ (f in FLAVORS) for f in self.flavors ]), "Unknown flavors provided"
        self.num_neutrinos = len(self.flavors)

        # Checks
        assert self.num_neutrinos in [2,3]

        # Useful derived values
        self.num_sun_basis_vectors = self.num_neutrinos ** 2

        # Init
        if self.tool == "nusquids" :
            self._init_nusquids(**kw)
        elif self.tool == "deimos" :
            self._init_deimos(**kw)
        elif self.tool == "prob3" :
            self._init_prob3(**kw)
        else :
            raise Exception("Unrecognised tool : %s" % self.tool)

        # Set some default values for parameters
        self.set_matter("vacuum")

        if mass_splittings_eV2 is None :
            if self.num_neutrinos == 3 :
                mass_splittings_eV2 = MASS_SPLITTINGS_eV2
            else :
                raise Exception("Must specify 'mass_splittings_eV2' when not in 3 flavor mode")

        if mixing_angles_rad is None :
            if self.num_neutrinos == 3 :
                mixing_angles_rad = MIXING_ANGLES_rad
            else :
                raise Exception("Must specify 'mixing_angles_rad' when not in 3 flavor mode")

        if deltacp_rad is None :
            if self.num_neutrinos == 3 :
                deltacp_rad = DELTACP_rad
            else :
                raise Exception("Must specify 'deltacp_rad' when not in 3 flavor mode")

        # Update osc params
        self.set_mixing_angles(*mixing_angles_rad, deltacp=deltacp_rad)
        self.set_mass_splittings(*mass_splittings_eV2)
        self.set_calc_basis(DEFAULT_CALC_BASIS)
        # self.set_decoherence_D_matrix_basis(DEFAULT_DECOHERENCE_GAMMA_BASIS)

        # Init some variables related to astrohysical coordinates   #TODO are thes DEIMOS-specific? If so, init in _init_deimos()
        self.detector_coords = None
        self._neutrino_source_kw = None



    def parse_pisa_config(self,config) :
        '''
        Parse settings from a PISA config file and apply them
        '''

        pass #TODO


    def _init_nusquids(self,
        energy_nodes_GeV=None,
        coszen_nodes=None,
        interactions=False,
        nusquids_variant=None, # Specify nuSQuIDS variants (nuSQuIDSDecoh, nuSQUIDSLIV, etc)
        error=1.e-6,
    ) :

        assert NUSQUIDS_AVAIL, "Cannot use nuSQuIDS, not installed"


        #
        # Handle nuSQuIDS variants
        #

        # Store arg
        self._nusquids_variant = nusquids_variant

        # Aliases
        if self._nusquids_variant in ["decoh", "decoherence" ] :
            self._nusquids_variant = "nuSQUIDSDecoh"
        if self._nusquids_variant in ["liv", "LIV", "sme", "SME" ] :
            self._nusquids_variant = "nuSQUIDSLIV"


        #
        # Calculation nodes
        #

        # Energy node definition
        self.energy_nodes_GeV = energy_nodes_GeV
        if self.energy_nodes_GeV is False :
            pass # Single-energy mode
        elif self.energy_nodes_GeV is None :
            # Provide default nodes if none provided
            self.energy_nodes_GeV = np.logspace(0.,3.,num=100)

        # cos(zenith) node definition
        # Only relevant in atmospheric mode
        if self.atmospheric :
            self.coszen_nodes = coszen_nodes
            if self.coszen_nodes is None :
                # Provide default nodes if none provided
                self.coszen_nodes = np.linspace(-1.,1.,num=100)
        else :
            assert coszen_nodes is None, "`coszen_nodes` argument only valid in `atmospheric` mode"


        #
        # Instantiate nuSQuIDS
        #

        # Get nuSQuiDS units
        self.units = nsq.Const()

        # Get neutrino type
        # Alwys do both, not the most efficient but simplifies things
        nu_type = nsq.NeutrinoType.both 

        # Toggle between atmo. vs regular modes
        if self.atmospheric :

            # Instantiate nuSQuIDS atmospheric calculator
            args = [
                self.coszen_nodes,
                self.energy_nodes_GeV * self.units.GeV,
                self.num_neutrinos,
                nu_type,
                interactions,
            ]

            if self._nusquids_variant is None :
                self.nusquids = nsq.nuSQUIDSAtm(*args)

            elif self._nusquids_variant == "nuSQUIDSDecoh" :
                assert NUSQUIDS_DECOH_AVAIL, "Could not find nuSQuIDS decoherence implementation"
                self.nusquids = nuSQUIDSDecohAtm(*args) #TODO Needs updating to modern nuSQuIDS pybindings format

            elif self._nusquids_variant == "nuSQUIDSLIV" :
                assert hasattr(nsq, "nuSQUIDSLIVAtm"), "Could not find nuSQuIDS LIV implementation"
                self.nusquids = nsq.nuSQUIDSLIVAtm(*args)

            else :
                raise Exception("Unknown nusquids varint : %s" % self._nusquids_variant)
            
            # Add tau regeneration
            # if interactions :
            #     self.nusquids.Set_TauRegeneration(True) #TODO results look wrong, disable for no w and investigate #TODO what about NC regeneration?

        else :

            # Instantiate nuSQuIDS regular calculator
            if self.energy_nodes_GeV is False :
                # Single-energy mode
                assert not interactions, "`interactions` cannot be set in single energy mode"
                assert nu_type in [nsq.NeutrinoType.neutrino, nsq.NeutrinoType.antineutrino], "Single-energy mode does not support neutrino and anitneutrino calculation simultaneously" 
                args = [
                    self.num_neutrinos,
                    nu_type,
                ]
            else :
                args = [
                    self.energy_nodes_GeV * self.units.GeV,
                    self.num_neutrinos,
                    nu_type,
                    interactions,
                ]

            if self._nusquids_variant is None :
                self.nusquids = nsq.nuSQUIDS(*args)

            elif self._nusquids_variant == "nuSQUIDSDecoh" :
                assert NUSQUIDS_DECOH_AVAIL, "Could not find nuSQuIDS decoherence implementation"
                self.nusquids = nuSQUIDSDecoh(*args) #TODO Needs updating to modern nuSQuIDS pybindings format

            elif self._nusquids_variant == "nuSQUIDSLIV" :
                assert hasattr(nsq, "nuSQUIDSLIV"), "Could not find nuSQuIDS LIV implementation"
                self.nusquids = nsq.nuSQUIDSLIV(*args)

            else :
                raise Exception("Unknown nusquids varint : %s" % self._nusquids_variant)

        #
        # Various settings
        #

        self.nusquids.Set_rel_error(error)
        self.nusquids.Set_abs_error(error)

        self.nusquids.Set_ProgressBar(False)




    def _init_deimos(self,
        **kw
    ) :

        # Init persistent state variables required
        # These are things that are passed to `calc_osc_probs` basically
        # self._decoh_D_matrix_eV = None
        self._decoh_n = None
        self._decoh_E0_eV = None
        self._calc_basis = None
        # self._decoherence_D_matrix_basis = None
        self._decoh_model_kw = None
        self._lightcone_model_kw = None
        self._sme_model_kw = None
        self.skymap_use = None
        
        # Instantiate solver
        self.solver = DensityMatrixOscSolver(
            num_states=self.num_neutrinos,
            **kw
        )


    def _init_prob3(self,
        **kw
    ) :

        assert PROB3_AVAIL, "Prob3 not installed"

        # Create a dict to hold settings
        self._prob3_settings = {}

        # Create propagator
        self._propagator = BargerPropagator()




    def set_matter(self, matter,
                   **kw) :

        # Re-initalise any persistent matter-related setting
        # Mostly don't use this, only for "layers" mode currently 
        self._matter_settings = { "matter":matter }

        #
        # Vacuum
        #

        if (matter == "vacuum") or (matter is None) :

            if self.tool == "nusquids" :
                self.nusquids.Set_Body(nsq.Vacuum())

            elif self.tool == "deimos" :
                self.solver.set_matter_potential(None)

            elif self.tool == "prob3" :
                self._prob3_settings["matter"] = None

        #
        # Earth
        #

        elif matter == "earth" :

            if self.tool == "nusquids" :
                if self.atmospheric :
                    self.nusquids.Set_EarthModel(nsq.EarthAtm())
                else :
                    raise Exception("`earth` is only an option in atmospheric mode")

            elif self.tool == "deimos" :
                raise Exception("`%s` does have an Earth model" % self.tool)

            elif self.tool == "prob3" :
                self._prob3_settings["matter"] = "earth"


        #
        # Uniform matter density
        #

        elif matter == "constant" :

            # Check required kwargs present
            assert "matter_density_g_per_cm3" in kw
            assert "electron_fraction" in kw

            if self.tool == "nusquids" :
                self.nusquids.Set_Body(nsq.ConstantDensity(kw["matter_density_g_per_cm3"], kw["electron_fraction"]))

            elif self.tool == "deimos" :
                V = get_matter_potential_flav(flavors=self.flavors, matter_density_g_per_cm3=kw["matter_density_g_per_cm3"], electron_fraction=kw["electron_fraction"], nsi_matrix=None)
                self.solver.set_matter_potential(V)


            elif self.tool == "prob3" :
                self._prob3_settings["matter"] = "constant"
                self._prob3_settings["matter_density_g_per_cm3"] = kw["matter_density_g_per_cm3"]

            
        elif matter == "variable" :

            assert "radius_fraction_array" in kw     # Array of radii, in units of Earth radius, i.e. 0 at centre and 1 at surface
            assert "matter_density_array_g_per_cm3" in kw
            assert "electron_fraction_array" in kw

            if self.tool == "nusquids" :
                # print(kw["radius_fraction_array"], kw["matter_density_array_g_per_cm3"], kw["electron_fraction_array"])

                self.nusquids.Set_Body(nsq.VariableDensity(kw["radius_fraction_array"], kw["matter_density_array_g_per_cm3"], kw["electron_fraction_array"]))
            
            elif self.tool == "deimos" :
                raise Exception("`%s` does not have a variable density matter model implemented" % self.tool)

            elif self.tool == "prob3" :
                raise Exception("`%s` does not have a variable density matter model implemented" % self.tool)
            


        

        # elif (self._matter == "three layer") and (self.matter_opts is not None):

        #     # define the three layers
            
        #     matter_density_1 = self.matter_opts["matter_density_1"]
        #     matter_density_2 = self.matter_opts["matter_density_2"]
        #     matter_density_3 = self.matter_opts["matter_density_3"]
        #     electron_fraction_1 = self.matter_opts["electron_fraction_1"]
        #     electron_fraction_2 = self.matter_opts["electron_fraction_2"]
        #     electron_fraction_3 = self.matter_opts["electron_fraction_3"]


        #     # Evolve the state in three layers
        #     # In first if-statement: use body1 until 1/3 of the distance
        #     # In second if-statement: fist use body1 until 1/3 of the distance and then use body2 for the remaining distance (L-(1/3)*totalt_distance)
        #     # In third if-statement: fist use body1 until 1/3 of the total distance, then use body2 until 1/3 of the total distance and then use body3 for the remaining distance (L-(2/3)*totalt_distance)
        #     # Evolve the state between each layer but only set initial state once
            
        #     # LAYER 1:
        #     if i_L < (1/3)*len(distance_km) :
        #         self.nusquids.Set_Body(nsq.ConstantDensity(matter_density_1, electron_fraction_1))
        #         self.nusquids.Set_Track(nsq.ConstantDensity(matter_density_1, electron_fraction_1).Track(L*self.units.km))
        #         self.nusquids.Set_initial_state( initial_state, nsq.Basis.flavor )
        #         self.nusquids.EvolveState()

        #     # LAYER 2:
        #     elif (i_L < (2/3)*len(distance_km)) and (i_L >= (1/3)*len(distance_km)):
        #         self.nusquids.Set_Body(nsq.ConstantDensity(matter_density_1, electron_fraction_1))
        #         self.nusquids.Set_Track(nsq.ConstantDensity(matter_density_1, electron_fraction_1).Track((1/3)*distance_km[-1]*self.units.km))
        #         self.nusquids.Set_initial_state( initial_state, nsq.Basis.flavor )
        #         self.nusquids.EvolveState()

        #         self.nusquids.Set_Body(nsq.ConstantDensity(matter_density_2, electron_fraction_2))
        #         self.nusquids.Set_Track(nsq.ConstantDensity(matter_density_2, electron_fraction_2).Track((L-(1/3)*distance_km[-1])*self.units.km))
        #         self.nusquids.EvolveState()

        #     # LAYER 3:
        #     else :
        #         self.nusquids.Set_Body(nsq.ConstantDensity(matter_density_1, electron_fraction_1))
        #         self.nusquids.Set_Track(nsq.ConstantDensity(matter_density_1, electron_fraction_1).Track((1/3)*distance_km[-1]*self.units.km))
        #         self.nusquids.Set_initial_state( initial_state, nsq.Basis.flavor )
        #         self.nusquids.EvolveState()

        #         self.nusquids.Set_Body(nsq.ConstantDensity(matter_density_2, electron_fraction_2))
        #         self.nusquids.Set_Track(nsq.ConstantDensity(matter_density_2, electron_fraction_2).Track(((1/3)*distance_km[-1])*self.units.km))
        #         self.nusquids.EvolveState()

        #         self.nusquids.Set_Body(nsq.ConstantDensity(matter_density_3, electron_fraction_3))
        #         self.nusquids.Set_Track(nsq.ConstantDensity(matter_density_3, electron_fraction_3).Track((L-(2/3)*distance_km[-1])*self.units.km))
        #         self.nusquids.EvolveState()


        




        # elif (self._matter == "simple earth"):
        #     # assert self.matter_opts is not None, "matter_opts are preset for simple earth model"
        #     assert np.isclose(distance_km[-1], 12742.0, atol=1.), "distance_km[-1] must be equal to the radius of the earth (12742 km)" 

        #     # Simple Earth goes through 3 layers: mantle, outer core and inner core

        #     # Reference to layer data:
        #     # Preliminary reference Earth model - Adam M. Dziewonski and Don L. Anderson

        #     # define propagation distances through each of the earts layers 
        #     inner_core_thickness_km = 1221.5*2          #x2 because 1221 is the radius
        #     outer_core_thicknes_km = 3480.0-1221.5      #3480 is the outer core radius
        #     mantle_thickness_km = 5701.0-3480.0         #5701 is the mantle radius
        #     transition_and_crust_thickness_km = 6371.0-5701.0 #6371 is the earth radius

        #     #for simplicity is the transition and crust layer treated as part of the mantle (quite thin layers) #TODO maybe implement layers for both transition and crust, as densities vary a lot
        #     mantle_thickness_km = mantle_thickness_km + transition_and_crust_thickness_km

            

        #     # define the matter densities (g/cm3) and electron fractions for each of the earths layers
        #     electron_fraction = 0.5
        #     matter_density_mantle = 7.957
        #     matter_density_outer_core = 12.58
        #     matter_density_inner_core = 13.08

        #     # Evolve the state through the layers: mantle, outer core, inner core, outer core, mantle

        #     D_earth = distance_km[-1]
        #     N_L = len(distance_km)


        #     # layer index boundaries for if-statements
        #     mantle_index_boundary = (mantle_thickness_km/D_earth)*N_L 
        #     outer_core_index_boundary = ((mantle_thickness_km+outer_core_thicknes_km)/D_earth)*N_L
        #     inner_core_index_boundary = ((mantle_thickness_km+outer_core_thicknes_km+inner_core_thickness_km)/D_earth)*N_L
        #     second_outer_core_index_boundary = ((mantle_thickness_km+outer_core_thicknes_km+inner_core_thickness_km+outer_core_thicknes_km)/D_earth)*N_L


        #         # LAYER 1: MANTLE
            
        #     if i_L < mantle_index_boundary :
        #         prop_dist = L*self.units.km
        #         self.nusquids.Set_Body(nsq.ConstantDensity(matter_density_mantle, electron_fraction))
        #         self.nusquids.Set_Track(nsq.ConstantDensity(matter_density_mantle, electron_fraction).Track(prop_dist))
        #         self.nusquids.Set_initial_state( initial_state, nsq.Basis.flavor )
        #         self.nusquids.EvolveState()
            


        #     # LAYER 2: OUTER CORE
            
        #     elif (i_L < outer_core_index_boundary) and (i_L >= mantle_index_boundary):
            
        #         prop_dist_mantle = mantle_thickness_km*self.units.km
        #         prop_dist_outer_core = (L-mantle_thickness_km)*self.units.km

        #         self.nusquids.Set_Body(nsq.ConstantDensity(matter_density_mantle, electron_fraction))
        #         self.nusquids.Set_Track(nsq.ConstantDensity(matter_density_mantle, electron_fraction).Track(prop_dist_mantle))
        #         self.nusquids.Set_initial_state( initial_state, nsq.Basis.flavor )
        #         self.nusquids.EvolveState()

        #         self.nusquids.Set_Body(nsq.ConstantDensity(matter_density_outer_core, electron_fraction))
        #         self.nusquids.Set_Track(nsq.ConstantDensity(matter_density_outer_core, electron_fraction).Track(prop_dist_outer_core))
        #         self.nusquids.EvolveState()


        #     # LAYER 3: INNER CORE

        #     elif (i_L < inner_core_index_boundary) and (i_L >= outer_core_index_boundary):
            
        #         prop_dist_mantle = mantle_thickness_km*self.units.km
        #         prop_dist_outer_core = outer_core_thicknes_km*self.units.km
        #         prop_dist_inner_core = (L-(mantle_thickness_km+outer_core_thicknes_km))*self.units.km

        #         self.nusquids.Set_Body(nsq.ConstantDensity(matter_density_mantle, electron_fraction))
        #         self.nusquids.Set_Track(nsq.ConstantDensity(matter_density_mantle, electron_fraction).Track(prop_dist_mantle))
        #         self.nusquids.Set_initial_state( initial_state, nsq.Basis.flavor )
        #         self.nusquids.EvolveState()

        #         self.nusquids.Set_Body(nsq.ConstantDensity(matter_density_outer_core, electron_fraction))
        #         self.nusquids.Set_Track(nsq.ConstantDensity(matter_density_outer_core, electron_fraction).Track(prop_dist_outer_core))
        #         self.nusquids.EvolveState()

        #         self.nusquids.Set_Body(nsq.ConstantDensity(matter_density_inner_core, electron_fraction))
        #         self.nusquids.Set_Track(nsq.ConstantDensity(matter_density_inner_core, electron_fraction).Track(prop_dist_inner_core))
        #         self.nusquids.EvolveState()
            

        #     # LAYER 4: SECOND OUTER CORE

        #     elif (i_L < second_outer_core_index_boundary) and (i_L >= inner_core_index_boundary):
            
        #         prop_dist_mantle = mantle_thickness_km*self.units.km
        #         prop_dist_outer_core = outer_core_thicknes_km*self.units.km
        #         prop_dist_inner_core = inner_core_thickness_km*self.units.km
        #         prop_dist_second_outer_core = (L-(mantle_thickness_km+outer_core_thicknes_km+inner_core_thickness_km))*self.units.km

        #         self.nusquids.Set_Body(nsq.ConstantDensity(matter_density_mantle, electron_fraction))
        #         self.nusquids.Set_Track(nsq.ConstantDensity(matter_density_mantle, electron_fraction).Track(prop_dist_mantle))
        #         self.nusquids.Set_initial_state( initial_state, nsq.Basis.flavor )
        #         self.nusquids.EvolveState()

        #         self.nusquids.Set_Body(nsq.ConstantDensity(matter_density_outer_core, electron_fraction))
        #         self.nusquids.Set_Track(nsq.ConstantDensity(matter_density_outer_core, electron_fraction).Track(prop_dist_outer_core))
        #         self.nusquids.EvolveState()

        #         self.nusquids.Set_Body(nsq.ConstantDensity(matter_density_inner_core, electron_fraction))
        #         self.nusquids.Set_Track(nsq.ConstantDensity(matter_density_inner_core, electron_fraction).Track(prop_dist_inner_core))
        #         self.nusquids.EvolveState()

        #         self.nusquids.Set_Body(nsq.ConstantDensity(matter_density_outer_core, electron_fraction))
        #         self.nusquids.Set_Track(nsq.ConstantDensity(matter_density_outer_core, electron_fraction).Track(prop_dist_second_outer_core))
        #         self.nusquids.EvolveState()


        #     # LAYER 5: SECOND MANTLE
        #     elif i_L >= second_outer_core_index_boundary:
            
        #         prop_dist_mantle = mantle_thickness_km*self.units.km
        #         prop_dist_outer_core = outer_core_thicknes_km*self.units.km
        #         prop_dist_inner_core = inner_core_thickness_km*self.units.km
        #         prop_dist_second_outer_core = outer_core_thicknes_km*self.units.km
        #         prop_dist_second_mantle = (L-(mantle_thickness_km+outer_core_thicknes_km+inner_core_thickness_km+outer_core_thicknes_km))*self.units.km

                

        #         self.nusquids.Set_Body(nsq.ConstantDensity(matter_density_mantle, electron_fraction))
        #         self.nusquids.Set_Track(nsq.ConstantDensity(matter_density_mantle, electron_fraction).Track(prop_dist_mantle))
        #         self.nusquids.Set_initial_state( initial_state, nsq.Basis.flavor )
        #         self.nusquids.EvolveState()

        #         self.nusquids.Set_Body(nsq.ConstantDensity(matter_density_outer_core, electron_fraction))
        #         self.nusquids.Set_Track(nsq.ConstantDensity(matter_density_outer_core, electron_fraction).Track(prop_dist_outer_core))
        #         self.nusquids.EvolveState()

        #         self.nusquids.Set_Body(nsq.ConstantDensity(matter_density_inner_core, electron_fraction))
        #         self.nusquids.Set_Track(nsq.ConstantDensity(matter_density_inner_core, electron_fraction).Track(prop_dist_inner_core))
        #         self.nusquids.EvolveState()

        #         self.nusquids.Set_Body(nsq.ConstantDensity(matter_density_outer_core, electron_fraction))
        #         self.nusquids.Set_Track(nsq.ConstantDensity(matter_density_outer_core, electron_fraction).Track(prop_dist_second_outer_core))
        #         self.nusquids.EvolveState()

        #         self.nusquids.Set_Body(nsq.ConstantDensity(matter_density_mantle, electron_fraction))
        #         self.nusquids.Set_Track(nsq.ConstantDensity(matter_density_mantle, electron_fraction).Track(prop_dist_second_mantle))
        #         self.nusquids.EvolveState()








        #
        # Matter layers (of constant density)
        #

        elif matter == "layers" :

            # Check required kwargs present
            assert "layer_endpoint_km" in kw # The endpoint of each layer. The startpoint is either L=0 (first layer) or the end of the previous layer
            assert "matter_density_g_per_cm3" in kw # Density in each layer
            assert "electron_fraction" in kw # Electron fraction in each layer

            # Check their format (e.g. one value per layer)
            assert isinstance(kw["matter_density_g_per_cm3"], np.ndarray) and (kw["matter_density_g_per_cm3"].ndim == 1), "'matter_density_g_per_cm3' should be an array of float values in 'layers' mode"
            assert isinstance(kw["electron_fraction"], np.ndarray) and (kw["electron_fraction"].ndim == 1), "'electron_fraction' should be an array of float values in 'layers' mode"
            assert kw["layer_endpoint_km"].size == kw["electron_fraction"].size, "'layer_endpoint_km', 'matter_density_g_per_cm3' and 'electron_fraction' do not have the same length (should be one per layer)"
            assert kw["layer_endpoint_km"].size == kw["matter_density_g_per_cm3"].size, "'layer_endpoint_km', 'matter_density_g_per_cm3' and 'electron_fraction' do not have the same length (should be one per layer)"

            # Check layer endpoints as ascending
            assert np.all(kw["layer_endpoint_km"][:-1] <= kw["layer_endpoint_km"][1:]), "'layer_endpoint_km' must be ascending"

            if self.tool == "nusquids" :
                # Store the laters for use during state evolution
                self._matter_settings["layer_endpoint_km"] = kw["layer_endpoint_km"]
                self._matter_settings["matter_density_g_per_cm3"] = kw["matter_density_g_per_cm3"]
                self._matter_settings["electron_fraction"] = kw["electron_fraction"]

            elif self.tool == "deimos" :
                raise NotImplemented("'layers' mode for matter effects not implemented for deimos")

            elif self.tool == "prob3" :
                raise NotImplemented("'layers' mode for matter effects not implemented for prob3")


        #
        # Error handling
        #

        else :
            raise Exception("Unrecognised `matter` : %s" % matter)



    def set_mixing_angles(self,theta12, theta13=None, theta23=None, deltacp=0.) :
        '''
        Units: radians
        '''

        if self.num_neutrinos == 2 :
            assert theta13 is None
            assert theta23 is None
        else :
            assert theta13 is not None
            assert theta23 is not None

        if self.tool == "nusquids" :
            self.nusquids.Set_CPPhase( 0, 2, deltacp ) #TODO check indices
            self.nusquids.Set_MixingAngle( 0, 1, theta12 )
            if self.num_neutrinos > 2 :
                self.nusquids.Set_MixingAngle( 0, 2, theta13 )
                self.nusquids.Set_MixingAngle( 1, 2, theta23 )

        elif self.tool == "deimos" :
            self.solver.set_mixing_angles( np.array([ t for t in [theta12,theta13,theta23] if t is not None ]), deltacp=deltacp )
            # self.solver.set_mixing_angles( -1. * np.array([ t for t in [theta12,theta13,theta23] if t is not None ]) ) #TODO

        elif self.tool == "prob3" :
            # Just store for passing an propagation time to solver
            self._prob3_settings["theta12"] = theta12
            self._prob3_settings["theta13"] = theta13
            self._prob3_settings["theta23"] = theta23
            self._prob3_settings["deltacp"] = deltacp


    def get_mixing_angles(self) :

        if self.tool == "deimos" :
            return self.solver.theta_rad

        elif self.tool == "prob3" :
            assert self.num_neutrinos == 3
            return (self._prob3_settings["theta12"],  self._prob3_settings["theta13"], self._prob3_settings["theta23"])

        else :
            raise Exception("TODO")


    def get_deltacp(self) :

        if self.tool == "deimos" :
            return self.solver.deltacp

        elif self.tool == "prob3" :
            return self._prob3_settings["deltacp"]

        else :
            raise Exception("TODO")


    def set_deltacp(self, deltacp) :

        if self.tool == "nusquids" :
            self.nusquids.Set_CPPhase( 0, 2, deltacp )

        elif self.tool == "deimos" :
            raise Exception("Cannot set delta CP on its own for `deimos`, use `set_mixing_angles`")

        elif self.tool == "prob3" :
            self._prob3_settings["deltacp"] = deltacp


    def set_mass_splittings(self, deltam21, deltam31=None) :
        '''
        # Note: deltam31 is +ve for normal ordering and -ve for inverted ordering

        Units: eV**2
        '''

        if self.num_neutrinos == 2 :
            assert deltam31 is None
        else :
            assert deltam31 is not None

        if self.tool == "nusquids" :
            self.nusquids.Set_SquareMassDifference( 1, deltam21*self.units.eV*self.units.eV )
            if deltam31 is not None :
                self.nusquids.Set_SquareMassDifference( 2, deltam31*self.units.eV*self.units.eV )

        elif self.tool == "deimos" :
            self.solver.set_mass_splittings( np.array([ dm2 for dm2 in [deltam21, deltam31] if dm2 is not None ]) )

        elif self.tool == "prob3" :
            self._prob3_settings["deltam21"] = deltam21
            self._prob3_settings["deltam31"] = deltam31


    def get_mass_splittings(self) :
        '''
        Units: eV**2
        '''

        if self.tool == "nusquids" :
            mass_splittings_eV2 = [ self.nusquids.Get_SquareMassDifference(1)/(self.units.eV*self.units.eV) ]
            if self.num_neutrinos > 2 :
                mass_splittings_eV2.append( self.nusquids.Get_SquareMassDifference(2)/(self.units.eV*self.units.eV) )
            return tuple(mass_splittings_eV2)

        elif self.tool == "deimos" :
            return self.solver.get_mass_splittings()

        elif self.tool == "prob3" :
            return ( self._prob3_settings["deltam21"], self._prob3_settings["deltam31"] )


    def set_std_osc(self) :
        '''
        Use standard oscillations (e.g. disable any BSM effects)
        '''

        if self.tool == "nusquids" :
            self.set_calc_basis(DEFAULT_CALC_BASIS)

            if self._nusquids_variant == "nuSQUIDSDecoh" :
                # self.set_decoherence_D_matrix_basis(DEFAULT_CALC_BASIS)
                self.set_decoherence_D_matrix(D_matrix_eV=np.zeros((self.num_sun_basis_vectors,self.num_sun_basis_vectors)), n=0, E0_eV=1.)

            elif self._nusquids_variant == "nuSQUIDSLIV" :
                null_matrix = np.zeros((3,self.num_neutrinos,self.num_neutrinos))
                self.set_sme(directional=True, basis="mass", a_eV=null_matrix, c=null_matrix, e=null_matrix, ra_rad=0., dec_rad=0.)
        
        else :
            self._decoh_model_kw = None
            self._lightcone_model_kw = None
            self._sme_model_kw = None
            self._neutrino_source_kw = None


    def set_calc_basis(self, basis) :

        if self.tool == "nusquids" :
            assert basis == "nxn" #TODO is this correct?

        elif self.tool == "deimos" :
            self._calc_basis = basis # Store for use later

        elif self.tool == "prob3" :
            pass # Basis not relevent here, not solving Linblad master equation

        else :
            raise Exception("`%s` does not support setting calculation basis" % self.tool)


    # def set_decoherence_D_matrix_basis(self, basis) :

    #     if self.tool == "nusquids" :
    #         assert basis == "sun"

    #     elif self.tool == "deimos" :
    #         self._decoherence_D_matrix_basis = basis # Store for use later

    #     else :
    #         raise Exception("`%s` does not support setting decoherence gamma matrix basis" % self.tool)



    #
    # Decoherence member functions
    #

    def set_decoherence_D_matrix(self,
        D_matrix_eV,
        n, # energy-dependence
        E0_eV,
    ) :
        '''
        Set the decoherence D matrix, plus energy dependence

        Definitions in arXiv:2007.00068, e.g.:
          - D matrix -> eqn 10
          - energy-dependence (steered by n and E9) -> eqn 18
        '''

        #
        # Check inputs
        #

        # If user specified the full matrix, check dimensions
        assert isinstance(D_matrix_eV, np.ndarray)

        # Check all relevent matrix conditions
        self.check_decoherence_D_matrix(D_matrix_eV)


        #
        # Set values
        #

        if self.tool == "nusquids" :
            assert self._nusquids_variant == "nuSQUIDSDecoh"
            assert np.allclose(D_matrix_eV.imag, 0.), "nuSQuIDS decoherence implementation currently does not support imaginary gamma matrix"
            self.nusquids.Set_DecoherenceGammaMatrix(D_matrix_eV.real * self.units.eV)
            self.nusquids.Set_DecoherenceGammaEnergyDependence(n)
            self.nusquids.Set_DecoherenceGammaEnergyScale(E0_eV)

        elif self.tool == "deimos" :
            self._decoh_model_kw = {
                "D_matrix0_eV" : D_matrix_eV, # Put 0 in this line!
                "n" : n,
                "E0_eV" : E0_eV,
                "D_matrix_basis" : "sun" # Added this line!
            }


    def check_decoherence_D_matrix(self, D) :
        '''
        There exist inequalities between the elements of the D matrix, meaning that the elements are not fully independent

        Enforcing these inequalities here:

         - 2 flavor: https://arxiv.org/pdf/hep-ph/0105303.pdf
         - 3 flavor: https://arxiv.org/pdf/1811.04982.pdf Appendix B
        '''

        #TODO Move this function out of this class, into models dir

        if self.num_neutrinos == 3 :

            #
            # SU(3) case
            #

            #TODO What enforces g1=g1, g4=g5, g6=g7 ?

            assert D.shape == (9, 9)

            #TODO what about 0th row/col?

            # Check everything is real
            assert np.all( D.imag == 0. )

            # Check everything is positive or zero
            assert np.all( D >= 0. )

            # Extract diagonal elements (gamma)
            g1 = D[1,1]
            g2 = D[2,2]
            g3 = D[3,3]
            g4 = D[4,4]
            g5 = D[5,5]
            g6 = D[6,6]
            g7 = D[7,7]
            g8 = D[8,8]

            # Extract off-diagonal elements (beta)
            # Enforce pairs either side of the diagonal match in the process
            b12 = D[1,2]
            assert D[2,1] == b12
            b13 = D[1,3]
            assert D[3,1] == b13
            b14 = D[1,4]
            assert D[4,1] == b14
            b15 = D[1,5]
            assert D[5,1] == b15
            b16 = D[1,6]
            assert D[6,1] == b16
            b17 = D[1,7]
            assert D[7,1] == b17
            b18 = D[1,8]
            assert D[8,1] == b18
            b23 = D[2,3]
            assert D[3,2] == b23
            b24 = D[2,4]
            assert D[4,2] == b24
            b25 = D[2,5]
            assert D[5,2] == b25
            b26 = D[2,6]
            assert D[6,2] == b26
            b27 = D[2,7]
            assert D[7,2] == b27
            b28 = D[2,8]
            assert D[8,2] == b28
            b34 = D[3,4]
            assert D[4,3] == b34
            b35 = D[3,5]
            assert D[5,3] == b35
            b36 = D[3,6]
            assert D[6,3] == b36
            b37 = D[3,7]
            assert D[7,3] == b37
            b38 = D[3,8]
            assert D[8,3] == b38
            b45 = D[4,5]
            assert D[5,4] == b45
            b46 = D[4,6]
            assert D[6,4] == b46
            b47 = D[4,7]
            assert D[7,4] == b47
            b48 = D[4,8]
            assert D[8,4] == b48
            b56 = D[5,6]
            assert D[6,5] == b56
            b57 = D[5,7]
            assert D[7,5] == b57
            b58 = D[5,8]
            assert D[8,5] == b58
            b67 = D[6,7]
            assert D[7,6] == b67
            b68 = D[6,8]
            assert D[8,6] == b68
            b78 = D[7,8]
            assert D[8,7] == b78

            # Now implement all inequalities
            a1 = -g1 + g2 + g3 - (g8/3.) 
            a2 =  g1 - g2 + g3 - (g8/3.)
            a3 =  g1 + g2  -g3 - (g8/3.)

            a4 = -g4 + g5 + g3 + (2.*g8/3.) - (2.*b38/np.sqrt(3.)) # See here that beta38 is somehwat special (since it relates to the special gamma3/8 params)
            a5 =  g4 - g5 + g3 + (2.*g8/3.) - (2.*b38/np.sqrt(3.))
            a6 = -g6 + g7 + g3 + (2.*g8/3.) + (2.*b38/np.sqrt(3.))
            a7 =  g6 - g7 + g3 + (2.*g8/3.) + (2.*b38/np.sqrt(3.))

            a8 = -(g1/3.) - (g2/3.) - (g3/3.) + (2.*g4/3.) + (2.*g5/3.) + (2.*g6/3.) + (2.*g7/3.) - g8

            assert a1 >= 0., "Inequality failure (a1)"
            assert a2 >= 0., "Inequality failure (a2)"
            assert a3 >= 0., "Inequality failure (a3)"
            assert a4 >= 0., "Inequality failure (a4)"
            assert a5 >= 0., "Inequality failure (a5)"
            assert a6 >= 0., "Inequality failure (a1)"
            assert a7 >= 0., "Inequality failure (a7)"
            assert a8 >= 0., "Inequality failure (a8)"

            assert (4.*np.square(b12)) <= ( np.square(g3 - (g8/3.)) - np.square(g1 - g2) )
            assert (4.*np.square(b13)) <= ( np.square(g2 - (g8/3.)) - np.square(g1 - g3) )
            assert (4.*np.square(b23)) <= ( np.square(g1 - (g8/3.)) - np.square(g2 - g3) )

            assert np.square( 4.*np.square(b38) + (g4/np.sqrt(3.)) + (g5/np.sqrt(3.)) - (g6/np.sqrt(3.)) - (g7/np.sqrt(3.)) ) <= (a3*a8)

            #TODO there are still quite a few more involving beta....

        else :
            print("Checks on decoherence D matrix inequalities not yet implemented for a %i neutrino system" % self.num_neutrinos)
            pass


    def set_decoherence_model(self, model_name, **kw) :
        '''
        Set the decoherence model to be one of the pre-defined models
        '''

        from deimos.models.decoherence.nuVBH_model import get_randomize_phase_decoherence_D_matrix, get_randomize_state_decoherence_D_matrix, get_neutrino_loss_decoherence_D_matrix
        from deimos.models.decoherence.generic_models import get_generic_model_decoherence_D_matrix

        #
        # Unpack args
        #

        kw = copy.deepcopy(kw)

        assert "gamma0_eV" in kw
        gamma0_eV = kw.pop("gamma0_eV")

        assert "n" in kw
        n = kw.pop("n")

        assert "E0_eV" in kw
        E0_eV = kw.pop("E0_eV")

        assert len(kw) == 0


        #
        # nu-VBH interaction models
        #

        get_D_matrix_func = None

        # Check if model is one of the nuVBH models, and get the D matrix definition function if so
        if model_name == "randomize_phase" :
            get_D_matrix_func = get_randomize_phase_decoherence_D_matrix

        elif model_name == "randomize_state" :
            get_D_matrix_func = get_randomize_state_decoherence_D_matrix

        elif model_name == "neutrino_loss" :
            get_D_matrix_func = get_neutrino_loss_decoherence_D_matrix

        # Check if found a match
        if get_D_matrix_func is not None :
            D_matrix_basis, D_matrix0_eV = get_D_matrix_func(num_states=self.num_neutrinos, gamma=gamma0_eV)


        #
        # Generic models
        #

        # Otherwise, try the generic models
        else :
            D_matrix0_eV = get_generic_model_decoherence_D_matrix(name=model_name, gamma=gamma0_eV)
            # D_matrix_basis = "mass" #TODO


        #
        # Pass to the solver
        #

        self.set_decoherence_D_matrix( D_matrix_eV=D_matrix0_eV, n=n, E0_eV=E0_eV ) #TODO what about the basis?


    #
    # Lightcone flucutation member functions
    #

    def set_lightcone_fluctuations(
        self,
        dL0_km,
        L0_km,
        E0_eV,
        n,
        m,
    ) :
        '''
        Set lightcone fluctuation model parameters
        '''

        from deimos.utils.model.lightcone_fluctuations.lightcone_fluctuation_model import get_lightcone_decoherence_D_matrix

        if self.tool == "nusquids" :
            print("NotImplemented. This is placeholder code:")
            damping_power = 2
            D_matrix = np.diag([0,1,1,0,1,1,1,1,0])
            self.nusquids.Set_DecoherenceGammaMatrix(D_matrix.real)
            self.nusquids.Set_DecoherenceGammaEnergyDependence(n)
            self.nusquids.Set_DecoherenceGammaEnergyScale(E0_eV * self.units.eV)
            self.nusquids.Set_UseLightconeFluctuations(True)
            self.nusquids.Set_mLengthDependenceIndex(m)
            self.nusquids.Set_dL0(dL0_km * self.units.km)
            self.nusquids.Set_L0LengthScale(L0_km * self.units.km)
            self.nusquids.Set_DampingPower(damping_power)


        elif self.tool == "deimos" :
            self._lightcone_model_kw = {
                "dL0_km" : dL0_km,
                "L0_km" : L0_km,
                "E0_eV" : E0_eV,
                "n" : n,
                "m" : m,
            }

    #
    # SME member functions
    #

    def set_sme(self,
        directional, # bool
        basis=None,       # string: "mass" or "flavor"
        a_eV=None,        # 3 x Num_Nu x Num_nu
        c=None,           # 3 x Num_Nu x Num_nu
        e=None,           # 3 x Num_Nu x Num_nu
        ra_rad=None,
        dec_rad=None,
    ) :
        '''
        TODO
        '''



        #
        # Check inputs
        #

        if basis is None :
            basis = "mass"
        assert basis in ["flavor", "mass"]

        if directional :   #TODO Maybe not relevant anymore? (Non-directional does currently not work in nuSQuIDS)
            operator_shape = (3, self.num_neutrinos, self.num_neutrinos) # shape is (num spatial dims, N, N), where N is num neutrino states
            if a_eV is None: 
                a_eV = np.zeros(operator_shape)
            if c is None:
                c = np.zeros(operator_shape)
            if e is None :
                e = np.zeros(operator_shape)

            assert isinstance(a_eV, np.ndarray) and (a_eV.shape == operator_shape)
            assert isinstance(c, np.ndarray) and (c.shape == operator_shape) 
            assert isinstance(e, np.ndarray) and (e.shape == operator_shape) 

            assert (ra_rad is not None) and (dec_rad is not None), "Must provide ra and dec when using directional SME"

        else :
            operator_shape = (self.num_neutrinos, self.num_neutrinos) # shape is (N, N), where N is num neutrino states
            assert isinstance(a_eV, np.ndarray) and (a_eV.shape == operator_shape)
            assert isinstance(c, np.ndarray) and (c.shape == operator_shape) 
            assert e is None, "e not implemented yet for isotropic SME"

            assert (ra_rad is None) and (dec_rad is None), "ra and dec not relevent for isotropic SME"


        #
        # Set values
        #

        if self.tool == "nusquids" :
            assert directional, "Isotropic SME not implemented in nuSQuIDS yet"
            assert basis == "mass", "Only mass basis SME implemented in nuSQuIDS currently"
            self.nusquids.Set_LIVCoefficient(a_eV, c, e, ra_rad, dec_rad)

            # self.sme_opts = {
            #         "basis" : basis,
            #         "a_eV" : a_eV,
            #         "c" : c,
            #         "e": e,
            #         "ra_rad" : ra_rad,
            #         "dec_rad" : dec_rad,

        elif self.tool == "deimos" :
            if directional :
                self._sme_model_kw = {
                    "directional" : True,
                    "basis" : basis,
                    "a_eV" : a_eV,
                    "c" : c,
                    "e": e,
                    "ra_rad" : ra_rad,
                    "dec_rad" : dec_rad,
                }
            else :
                self._sme_model_kw = {
                    "directional" : False,
                    "basis" : basis,
                    "a_eV" : a_eV,
                    "c" : c,
                    # "e": e,
                }

        else :
            raise NotImplemented("SME not yet wrapped for %s" % self.tool) #TODO this is already supported by prob3, just need to wrap it


    def set_detector_location(
        self,
        lat_deg,
        long_deg, 
        height_m,
    ) :
        '''
        Define detector position
        '''

        self.detector_coords = DetectorCoords(
            detector_lat=lat_deg, 
            detector_long=long_deg, 
            detector_height_m=height_m,  #TODO consistency with detector depth in the L<->coszen calculation
        )
        

    def set_detector(
        self,
        name,
    ) :
        '''
        Set detector (position, etc), choosing from known detectors
        '''

        if name.lower() == "icecube" :
            self.set_detector_location(
                lat_deg="89°59′24″S",
                long_deg="63°27′11″W",
                height_m=-1400.,
            )

        elif name.lower() == "dune" :
            self.set_detector_location(
                lat_deg=44.3517,
                long_deg=-103.7513,
                height_m=-1.5e3,
            )

        elif name.lower() == "arca" :
            self.set_detector_location(
                lat_deg="36°15′36″N",
                long_deg="16°06′00″E",
                height_m=-1500.,
            )

        elif name.lower() == "equator" :
            self.set_detector_location(
                lat_deg=0.,
                long_deg=0.,
                height_m=0.,
            )

        else :
            raise NotImplemented("Unknown detector : %s" % name)




    
    # def set_neutrino_source(self,
    #                         # Date, Time and Timezone
    #                         date_str,
    #                         # Location on the sky
    #                         ra_deg=None, 
    #                         dec_deg=None,
    #                         ):

    #     if self.tool == "nusquids" :
    #         raise NotImplementedError()

    #     elif self.tool == "deimos" :
    #         #Set date, time and location of neutrino source
    #         coszen_neutrino_source, altitude_neutrino_source, azimuth_neutrino_source = self.detector_coords.get_coszen_altitude_and_azimuth(
    #             ra_deg = ra_deg, 
    #             dec_deg = dec_deg,
    #             date_str = date_str
    #             )
            
    #         self._neutrino_source_kw = {
    #             # Horizontal Coordinate System
    #             "coszen" : coszen_neutrino_source,
    #             "altitude" : altitude_neutrino_source,
    #             "azimuth" : azimuth_neutrino_source,
    #             # Equatorial Coordinate System
    #             "ra" : ra_deg,
    #             "dec" : dec_deg,
    #             # Store date_str for skymap
    #             "date_str" : date_str,
    #             "sidereal_time" : self.detector_coords.get_local_sidereal_time(date_str)
    #         }
         
    #     else :
    #         raise NotImplemented()


    def calc_osc_prob(self,
        energy_GeV,
        initial_flavor=None,
        initial_state=None,
        distance_km=None,
        coszen=None,
        nubar=False,
        **kw
    ) :
        '''
        For the given model state, calcukate oscillation probabilities for neutrinos as specified in the inputs
        '''

        #TODO caching
        #TODO Option for different final rho to allow nu->nubar transitions

        #
        # Check inputs
        # 
 
         # Handle arrays vs single values for energy
        if isinstance(energy_GeV, (list, np.ndarray)) :
            single_energy, energy_size = False, len(energy_GeV)
        else :
            assert isinstance(energy_GeV, numbers.Number)
            single_energy, energy_size = True, 1

        # Indexing
        if initial_flavor is not None :
            initial_flavor = self._get_flavor_index(initial_flavor)

        #
        # Handle atmospheric mode
        #
        
        if self.atmospheric :

            # Want coszen, not distance
            assert ( (coszen is not None) and (distance_km is None) ), "Must provide `coszen` (and not `distance_km`) in atmospheric mode"  #TODO option to provide distance still in atmo mode

            # Handle single vs array of distances
            if isinstance(coszen, (list, np.ndarray)) :
                coszen = np.array(coszen)
                assert coszen.ndim == 1
                single_dist, dist_size = False, len(coszen)
            else :
                assert isinstance(coszen, numbers.Number)
                coszen = [coszen]
                single_dist, dist_size = True, 1

        else :

            # Want distance, not coszen
            assert ( (distance_km is not None) and (coszen is None) ), "Must provide `distance_km` (and not `coszen`) in non-atmospheric mode" 

            # Handle single vs array of distances
            if isinstance(distance_km, (list, np.ndarray)) :
                single_dist, dist_size = False, len(distance_km)
            else :
                assert isinstance(distance_km, numbers.Number)
                single_dist, dist_size = True, 1



        #
        # Calculate
        #

        # Call sub-function for relevent solver
        if self.tool == "nusquids" :
            osc_probs = self._calc_osc_prob_nusquids( initial_flavor=initial_flavor, initial_state=initial_state, energy_GeV=energy_GeV, distance_km=distance_km, coszen=coszen, nubar=nubar, **kw ) #TODO use single E value for single E mode

        elif self.tool == "deimos" :
            assert initial_flavor is not None, "must provide `initial_flavor` (`initial_state` not currently supported for %s" % self.tool
            osc_probs = self._calc_osc_prob_deimos( initial_flavor=initial_flavor, nubar=nubar, energy_GeV=energy_GeV, distance_km=distance_km, coszen=coszen, **kw )
       
        elif self.tool == "prob3" :
            osc_probs = self._calc_osc_prob_prob3( initial_flavor=initial_flavor, energy_GeV=energy_GeV, distance_km=distance_km, coszen=coszen, nubar=nubar, **kw )



        #
        # Done
        #

        # Check shape of output array
        expected_shape = ( energy_size, dist_size, self.num_neutrinos )
        assert osc_probs.shape == expected_shape

        # Remove single-valued dimensions, and check shape again
        # osc_probs = np.squeeze(osc_probs)
        # expected_shape = []
        # if not single_energy :
        #     expected_shape.append(energy_size)
        # if not single_dist :
        #     expected_shape.append(dist_size)
        # expected_shape.append(self.num_neutrinos)
        # expected_shape = tuple(expected_shape)
        # assert osc_probs.shape = expected_shape
        if single_energy and single_dist :
            osc_probs = osc_probs[0,0,:]
        elif single_energy :
            osc_probs = osc_probs[0,:,:]
        elif single_dist :
            osc_probs = osc_probs[:,0,:]

        # Checks
        assert np.all( np.isfinite(osc_probs) ), "Found non-finite osc probs"

        return osc_probs



    def calc_osc_prob_sme(self,
        # Neutrino properties
        energy_GeV,
        ra_rad,
        dec_rad,
        time,
        initial_flavor,
        nubar=False,
        # SME properties
        std_osc=False, # Can toggle standard oscillations (rather than SME)
        basis=None,
        a_eV=None,
        c=None,
        e=None,
        # Args to pass down to the standard osc prob calc
        **kw
    ) :
        '''
        Similar to calc_osc_prob, but for the specific case of the SME where there is also a RA/declination/time dependence 

        Aswell as osc probs, also return the computed direction information
        '''

        #TODO option to provide detector coord info (coszen, azimuth) instead of ra/dec
        #TODO anything required to support skymaps?


        #
        # Check inputs
        #

        # Handle arrays vs single values for RA/dec     #TODO option to pass one of RA/dec as single valued and one as array
        if isinstance(ra_rad, (list, np.ndarray)) :
            assert isinstance(dec_rad, (list, np.ndarray)), "ra_rad and dec_rad must either both be array-like or both scalars"
            ra_rad_values = np.array(ra_rad)
            dec_rad_values = np.array(dec_rad)
            assert ra_rad_values.ndim == 1
            assert dec_rad_values.ndim == 1
            assert ra_rad_values.size == dec_rad_values.size
            single_dir = False
        else :
            assert isinstance(ra_rad, numbers.Number)
            assert isinstance(dec_rad, numbers.Number)
            ra_rad_values = [ra_rad]
            dec_rad_values = [dec_rad]
            single_dir = True

        # Handle arrays vs single values for time
        if isinstance(time, (list, np.ndarray)) :
            time_values = time
            assert np.ndim(time_values) == 1
            single_time = False
        else :
            time_values = [time]
            single_time = True

        # Handle SME vs standard osc
        if std_osc :
            assert basis is None
            assert a_eV is None
            assert c is None
            assert e is None


        #
        # Loop over directions
        #

        osc_probs = []
        coszen_values, azimuth_values = [], []

        # Loop over directions
        for ra_rad, dec_rad in zip(ra_rad_values, dec_rad_values) :

            osc_probs_vs_time = []
            coszen_values_vs_time, azimuth_values_vs_time = [], []


            #
            # Set SME model params 
            #

            # Cannot do this before calling this function as for most oscillation models, due to the RA/declination/time dependence of the Hamiltonian
            # Also might use standard oscillations here, depending on what user requestes

            if std_osc :
                self.set_std_osc()

            else :
                self.set_sme(
                    directional=True,
                    basis=basis,
                    a_eV=a_eV,
                    c=c,
                    e=e,
                    ra_rad=ra_rad,
                    dec_rad=dec_rad,
                )


            # 
            # Loop over times
            #

            for time in time_values :


                #
                # Handle atmospheric vs regular case
                #

                if self.atmospheric :

                    #
                    # Atmospheric case
                    #

                    # Need to know the detector location to get coszen/azimuth from RA/dec
                    assert self.detector_coords is not None, "Must set detector position"

                    # Get local direction coords
                    coszen, altitude, azimuth = self.detector_coords.get_coszen_altitude_and_azimuth(ra_deg=np.rad2deg(ra_rad), dec_deg=np.rad2deg(dec_rad), time=time)

                    # Standard osc prob calc, so this particular direction/time
                    _osc_probs = self.calc_osc_prob(
                        initial_flavor=initial_flavor,
                        nubar=nubar,
                        energy_GeV=energy_GeV,
                        coszen=coszen,
                        **kw # Pass down kwargs
                    )



                else :

                    #
                    # Regular (1D) case
                    #

                    raise NotImplemented("Non-atmospheric case not yet implemented for celestial coords")


                # Merge into the overall output array
                if single_time :
                    osc_probs_vs_time = _osc_probs
                    coszen_values_vs_time = coszen
                    azimuth_values_vs_time = azimuth
                else :
                    osc_probs_vs_time.append( _osc_probs )
                    coszen_values_vs_time.append( coszen )
                    azimuth_values_vs_time.append( azimuth )

            # Merge into the overall output array
            if single_dir :
                osc_probs = osc_probs_vs_time
                coszen_values = coszen_values_vs_time
                azimuth_values = azimuth_values_vs_time
            else :
                osc_probs.append( osc_probs_vs_time )
                coszen_values.append( coszen_values_vs_time )
                azimuth_values.append( azimuth_values_vs_time )

        #
        # Done
        #

        # Array-ify
        osc_probs = np.array(osc_probs)
        coszen_values = np.array(coszen_values)
        azimuth_values = np.array(azimuth_values)

        # Check size
        #TODO

        # Checks
        assert np.all( np.isfinite(osc_probs) ), "Found non-finite osc probs"

        # Return
        return_values = [osc_probs]
        if self.atmospheric :
            return_values.extend([ coszen_values, azimuth_values ])
        return tuple(return_values)



    def _calc_osc_prob_nusquids(self,
        energy_GeV,
        initial_flavor=None,
        initial_state=None,
        nubar=False,
        distance_km=None,
        coszen=None,

        # Neutrino direction in celestial coords - only required for certain models (such as the SME)
        ra_rad=None,
        dec_rad=None,
    ) :
        '''
        Calculate oscillation probability for the model

        Returned result has following structure: [ energy, coszen, final flavor ]
        '''


        #
        # Prepare
        #

        assert not ( (initial_flavor is None) and (initial_state is None) ), "Must provide `initial_flavor` or `initial_state`"
        assert not ( (initial_flavor is not None) and (initial_state is not None) ), "Must provide `initial_flavor` or `initial_state`, not both"

        # Calculate all final state flavors
        final_flavors = self.states

        # Handle scalars vs arrays
        energy_GeV = np.asarray( [energy_GeV] if np.isscalar(energy_GeV) else energy_GeV )
        if distance_km is not None :
            distance_km = np.asarray( [distance_km] if np.isscalar(distance_km) else distance_km )
        if coszen is not None :
            coszen = np.asarray( [coszen] if np.isscalar(coszen) else coszen )

        # Arrays must be 1D
        assert energy_GeV.ndim == 1
        if distance_km is not None :
            assert distance_km.ndim == 1
        if coszen is not None :
            assert coszen.ndim == 1

        # Handle nubar
        rho = 1 if nubar else 0

        # # Handle nubar
        # if initial_flavor is not None :
        #     if initial_flavor < 0 :
        #         assert include_nubar
        #         initial_flavor = -1* initial_flavor
        #         rho = 1
        #     else :
        #         rho = 0


        #
        #  SME Case
        # #

        # if self.sme_opts is not None :
            
        #     # To include SME parameters in calculation of the hamiltonian
        #     include_sme = True

        #     # Handle isotropic vs directional
        #     assert "directional" in self.sme_opts
        #     sme_is_directional = self.sme_opts.pop("directional")

        #     # Handle basis in which flavor/mass structure is defined
        #     assert "basis" in self.sme_opts
        #     sme_basis = self.sme_opts.pop("basis")
        #     assert sme_basis in ["mass", "flavor"]
        #     sme_basis_is_flavor = sme_basis == "flavor" # Bool fast checking during solving

        #     # User provides a(3) and c(4) coefficients, plus a possible mass-dependent non-renomalizable term
        #     self.sme_opts = copy.deepcopy(self.sme_opts)
        #     assert "a_eV" in self.sme_opts
        #     sme_a = self.sme_opts.pop("a_eV")
        #     assert "c" in self.sme_opts
        #     sme_c = self.sme_opts.pop("c") # dimensionless
        #     # if sme_is_directional : # e term only implemented for direction SME currently
        #     assert "e" in self.sme_opts
        #     sme_e = self.sme_opts.pop("e") # dimensionless


        #     # Handle antineutrinos
        #     if nubar:
        #         sme_a = - sme_a


        #     # Get neutrino direction in celestial coords
        #     if sme_is_directional :
        #         assert ra_rad is not None
        #         assert dec_rad is not None
        #         assert np.isscalar(ra_rad)
        #         assert np.isscalar(dec_rad)
        #         assert (ra_rad >= 0) and (ra_rad <= 2 * np.pi)
        #         assert (dec_rad >= -np.pi / 2) and (dec_rad <= np.pi / 2)

            
            
        #     # Check for additional SME arguments
        #     assert len(self.sme_opts) == 0, "Unused SME arguments!?!"


        #     # Set SME parameters in nusquids #TODO e_term not yet implemented
        #     self.nusquids.Set_LIVCoefficient(sme_a,sme_c,sme_e,ra_rad, dec_rad)
            
                
           


        #
        # Atmospheric case
        #

        if self.atmospheric :
            

            randomize_atmo_prod_height = False #TODO support

            # Init results container
            # results = np.full( (energy_GeV.size, coszen.size, final_flavors.size, 2 ), np.NaN )
            results = np.full( (energy_GeV.size, coszen.size, final_flavors.size ), np.NaN )

            # Determine shape of initial state vector
            state_shape = [ self.nusquids.GetNumCos(), self.nusquids.GetNumE() ]
            state_shape.append( 2 )
            state_shape.append( final_flavors.size )
            state_shape = tuple(state_shape)

            # Define initial state if not provided, otherwise verify the one provided
            if initial_state is None :
                initial_state = np.full( state_shape, 0. )
                initial_state[ :, :, rho, initial_flavor ] = 1. # dims = [ cz node, E node, nu(bar), flavor ]
            else :
                assert initial_state.shape == state_shape, "Incompatible shape for initial state : Expected %s, found %s" % (state_shape, initial_state.shape)

            # Set the intial state
            self.nusquids.Set_initial_state(initial_state, nsq.Basis.flavor)

            # Evolve the state
            self.nusquids.EvolveState()

            # Evaluate the flavor at each grid point to get oscillation probabilities
            for i_E,E in enumerate(energy_GeV) :
                for i_cz,cz in enumerate(coszen) :
                    for i_f,final_flavor in enumerate(final_flavors) :
                        # results[i_E,i_cz,i_f] = self.nusquids.EvalFlavor( final_flavor, cz, E*self.units.GeV )#, rho ) #TODO Add randomize prod height arg
                        results[i_E,i_cz,i_f] = self.nusquids.EvalFlavor( int(final_flavor), cz, E*self.units.GeV, int(rho), randomize_atmo_prod_height) #TODO add nubar


            return results
        
       






        #
        # Distance case
        #

        else :

            
            # Init results container
            results = np.full( (energy_GeV.size, distance_km.size, final_flavors.size), np.NaN )
            # results = np.full( (energy_GeV.size, distance_km.size, final_flavors.size, 2), np.NaN )

            # Determine shape of initial state vector
            state_shape = [ self.nusquids.GetNumE() ]
            state_shape.append(2)
            state_shape.append( final_flavors.size )
            state_shape = tuple(state_shape)

            # Define initial state if not provided, otherwise verify the one provided
            if initial_state is None :
                initial_state = np.full( state_shape, 0. )
                initial_state[ :, rho, initial_flavor ] = 1. # dims = [ E node, nu(bar), flavor ]
            else :
                assert initial_state.shape == state_shape, "Incompatible shape for initial state : Expected %s, found %s" % (state_shape, initial_state.shape)

            # Loop over distance nodes
            for i_L, L in enumerate(distance_km) :


                #
                # Propagate the neutrino in 1D
                #

                #TODO keep evolving from previous (shorter) distance node rather than re-calculating from 0 every time (for efficiency)?

                # Set the track (e.g. neutrino travel path), taking medium into account. Then propagate
                if self._matter_settings["matter"] == "vacuum" :

                    # Vacuum is easy: Just propagate in vacuum
                    self.nusquids.Set_Track(nsq.Vacuum.Track(L*self.units.km))
                    self.nusquids.Set_initial_state( initial_state, nsq.Basis.flavor )
                    self.nusquids.EvolveState()

                elif self._matter_settings["matter"] == "constant" :

                    # Constant density is easy: Just propagate in constant density medium
                    self.nusquids.Set_Track(nsq.ConstantDensity.Track(L*self.units.km))
                    self.nusquids.Set_initial_state( initial_state, nsq.Basis.flavor )
                    self.nusquids.EvolveState()

                elif self._matter_settings["matter"] == "layers" :

                    # Layers on constant density are a bit more tricky. Step through them, evolving the state though each later, then changing density and continuing the state evolution (without resetting it)
                    # Take care to cut off when reach the requested propagation distance

                    # Check the layers cover the full path length
                    assert self._matter_settings["layer_endpoint_km"][-1] >= L, "Matter layers do not cover the full baseline"

                    # Loop through layers
                    L_so_far = 0.
                    for endpoint, density, efrac in zip(self._matter_settings["layer_endpoint_km"], self._matter_settings["matter_density_g_per_cm3"], self._matter_settings["electron_fraction"]) :

                        # Bail out if have reached travel distance
                        if L_so_far > endpoint :
                            break

                        # Figure out how far we will travel in this layer
                        if L < endpoint :
                            endpoint = L # Do not step past endpoint
                        L_layer = endpoint - L_so_far

                        # Set the body and track, and propagate
                        self.nusquids.Set_Body(nsq.ConstantDensity(density, efrac))
                        self.nusquids.Set_Track(nsq.ConstantDensity.Track(L_layer*self.units.km))
                        if L_so_far == 0 :
                            self.nusquids.Set_initial_state( initial_state, nsq.Basis.flavor ) # Only first step
                        self.nusquids.EvolveState()

                        # Update distance counter
                        L_so_far += L_layer

                else :
                    raise Exception("Unknown matter : %s" % self._matter_settings["matter"]) 


                #
                # Evaluate final state
                #

                # Loop over energies
                for i_e, E in enumerate(energy_GeV) :

                    # Evaluate final state flavor composition
                    for i_f, final_flavor in enumerate(final_flavors) :
                        # for rho in [0, 1] :
                        #     results[i_e,i_L,i_f,rho] = self.nusquids.EvalFlavor( int(final_flavor), float(E*self.units.GeV), int(rho) )
                        results[i_e,i_L,i_f] = self.nusquids.EvalFlavor( int(final_flavor), float(E*self.units.GeV), int(rho) )

            return results








    def _calc_osc_prob_prob3(self,
        initial_flavor,
        energy_GeV,
        distance_km=None,
        coszen=None,
        nubar=False,
    ) :


        #
        # Check inputs
        #

        # Check num flavors
        assert self.num_neutrinos == 3, "prob3 wrapper only supporting 3-flavor oscillations currently" #TODO probably can add supoort for N != 3

        # Note that coszen vs distance handling already done in level above

        #TODO coszen->distance conversion for vacuum atmo case



        #
        # Define system
        #

        # Get osc propeties
        theta12, theta13, theta23 = self.get_mixing_angles()
        sin2_theta12, sin2_theta13, sin2_theta23 = np.square(np.sin(theta12)), np.square(np.sin(theta13)), np.square(np.sin(theta23))
        deltacp = self.get_deltacp()
        dm21, dm31 = self.get_mass_splittings()
        dm32 = dm31 - dm21 #TODO careful with mass ordering
        KNuType = -1 if nubar else +1

        # Dertemine all final states
        final_flavors = self.states

        # Determine matter
        earth, matter_density_g_per_cm3 = False, None
        vacuum = self._prob3_settings["matter"] is None
        if not vacuum :
            if self._prob3_settings["matter"] == "earth" :
                earth = True
            else :
                assert self._prob3_settings["matter"] == "constant"
                matter_density_g_per_cm3 = self._prob3_settings["matter_density_g_per_cm3"]


        #
        # Loop over energy/coszen
        #

        # Array-ify
        energy_GeV = np.asarray( [energy_GeV] if np.isscalar(energy_GeV) else energy_GeV )
        distance_km = np.asarray( [distance_km] if np.isscalar(distance_km) else distance_km )
 
        # Init outputs container
        energy_dim = np.size(energy_GeV)
        distance_dim = np.size(distance_km)
        results = np.full( (energy_dim, distance_dim, self.num_neutrinos), np.NaN )

        # Loop over energy
        for i_E in range(energy_dim) :

            # Update propagator settings
            # Must do this each time energy changes, but don't need to for distance
            self._propagator.SetMNS(
                sin2_theta12, # sin2_theta12,
                sin2_theta13, # sin2_theta13,
                sin2_theta23, # sin2_theta23,
                dm21, # dm12,
                dm32, # dm23,
                deltacp, # delta_cp [rad]   #TODO get diagreeement between prob3 and other solvers (DEIMOS, nuSQuIDS) when this is >0, not sure why?
                energy_GeV[i_E], # Energy
                True, # True means expect sin^2(theta), False means expect sin^2(2*theta)
                KNuType,
            )

            # Loop over distance f
            for i_L in range(distance_dim) :

                # Loop over flavor
                for i_f, final_flavor in enumerate(final_flavors) :


                    #
                    # Calc osc probs
                    #

                    # Handle prob3 flavor index format: uses [1,2,3], not [0,1,3]
                    initial_flavor_prob3 = initial_flavor + 1 # prob3 
                    final_flavor_prob3 = final_flavor + 1

                    # Calculation depends of matter type
                    if vacuum :

                        # Run propagation and calc osc probs
                        P = self._propagator.GetVacuumProb( initial_flavor_prob3, final_flavor_prob3 , energy_GeV[i_E],  distance_km[i_L])

                    else :

                        # Toggle between Earth vs constant density
                        if earth :

                            # Propagate in Earth
                            raise Exception("Not yet implemented") #TODO need to handle coszen, etc
                            self._propagator.DefinePath( cosineZ, prod_height )
                            self._propagator.propagate( KNuType )

                        else :

                            # Propagate in constant density matter
                            self._propagator.propagateLinear( KNuType, distance_km[i_L] , matter_density_g_per_cm3 )

                        # Calc osc probs for mater case (after propagation done abopve already)
                        P = self._propagator.GetProb( initial_flavor_prob3, final_flavor_prob3 )


                    # Set to output array
                    assert np.isscalar(P)
                    results[i_E, i_L, i_f] = P


        return results


    def _calc_osc_prob_deimos(self,

        # Neutrino definition
        initial_flavor,
        energy_GeV,
        distance_km=None,
        coszen=None,
        nubar=False,

        # Neutrino direction in celestial coords - only required for certain models (such as the SME)
        ra_rad=None,
        dec_rad=None,

    ) :

        #
        # Prepare
        #

        # Calculate all final state flavors
        final_flavors = self.states

        # Handle scalars vs arrays
        energy_GeV = np.asarray( [energy_GeV] if np.isscalar(energy_GeV) else energy_GeV )
        if distance_km is not None :
            distance_km = np.asarray( [distance_km] if np.isscalar(distance_km) else distance_km )
        if coszen is not None :
            coszen = np.asarray( [coszen] if np.isscalar(coszen) else coszen )

        # Arrays must be 1D
        assert energy_GeV.ndim == 1
        if distance_km is not None :
            assert distance_km.ndim == 1
        if coszen is not None :
            assert coszen.ndim == 1


        #
        # Calculate
        #

        # coszen -> L conversion (for atmospheric case)
        if self.atmospheric :
            production_height_km = DEFAULT_ATMO_PROD_HEIGHT_km #TODO steerable, randomizable
            detector_depth_km = DEFAULT_ATMO_DETECTOR_DEPTH_km if self.detector_coords is None else self.detector_coords.detector_depth_m*1e-3 # Use detector position, if available    #TODO should we really be defining this as height?
            distance_km = calc_path_length_from_coszen(cz=coszen, h=production_height_km, d=detector_depth_km)

        # DensityMatrixOscSolver doesn't like decending distance values in the input arrays,
        # and this is what you get from coszen arrays often
        flip = False
        if distance_km[-1] < distance_km[0] : 
            flip = True
            distance_km = np.flip(distance_km)

        # Run solver
        # 'results' has shape [N energy, N distance, N flavor]
        results = self.solver.calc_osc_probs(
            E_GeV=energy_GeV,
            L_km=distance_km,
            initial_state=initial_flavor,
            initial_basis="flavor",
            nubar=nubar,
            calc_basis=self._calc_basis,
            # D_matrix_basis=self._decoherence_D_matrix_basis,
            decoh_opts=self._decoh_model_kw,
            lightcone_opts=self._lightcone_model_kw,
            sme_opts=self._sme_model_kw,
            verbose=False
        )

        # Handle flip in results (L dimension)
        if flip :
            results = np.flip(results, axis=1)

        return results


    def _get_flavor_index(self,flavor) :

        index = None

        if isinstance(flavor, str) :
            if flavor in ["e","nue"] :
                index = 0
            elif flavor in ["mu","numu"] :
                index = 1
            elif flavor in ["tau","nutau"] :
                index = 2

        else :
            assert flavor in [0,1,2]
            index = flavor

        assert flavor < self.num_neutrinos

        return index 

    def set_colors(self, nu_colors) :
        self.nu_colors = nu_colors


    @property
    def states(self) :
        return np.array(range(self.num_neutrinos))


    def get_flavor_tex(self, i) :
        '''
        Get tex representation of flavor i (e.g. e, mu, tau)
        '''

        assert i < self.num_neutrinos
        flavor = self.flavors[i]

        if flavor == "e" :
            return r"e"
        elif flavor == "mu" :
            return r"\mu"
        elif flavor == "tau" :
            return r"\tau"
        else :
            raise Exception("Unknown flavor : %s" % flavor)


    def get_nu_flavor_tex(self, i=None, nubar=False) :
        '''
        Get tex representation of neutrino flavor i (e.g. nue, numu, nutau)
        '''

        nu_tex = r"\nu"

        if nubar :
            nu_tex = r"\bar{" + nu_tex + r"}"

        if i is None :
            nu_tex += r"_{\rm{all}}"
        else :
            flavor_tex = self.get_flavor_tex(i)
            nu_tex += r"_{" + flavor_tex + r"}"

        return nu_tex


    def get_nu_mass_tex(self, i=None, nubar=False) :
        '''
        Get tex representation of neutrino mass state i (e.g. nu_1, numu_2, nutau_3)
        '''

        nu_tex = r"\nu"

        if nubar :
            nu_tex = r"\bar{" + nu_tex + r"}"

        if i is None :
            nu_tex += r"_{\rm{all}}"
        else :
            nu_tex += r"_{" + (i+1) + r"}"

        return nu_tex


    # @property
    # def flavors_tex(self) :
    #     return [ self.get_flavor_tex(i) for i in self.states ]


    # @property
    # def masses_tex(self) :
    #     return [ self.get_mass_tex(i) for i in self.states ]


    # @property
    # def flavors_color(self) :
    #     return [ self.get_flavor_color(i) for i in self.states ]

    # def get_flavor_color(self, flavor) :
    #     return self.nu_colors[flavor]


    def get_transition_prob_tex(self, initial_flavor, final_flavor, nubar=False) :
        return r"P(%s \rightarrow %s)" % ( self.get_nu_flavor_tex(initial_flavor, nubar), self.get_nu_flavor_tex(final_flavor, nubar) )


    @property
    def PMNS(self) :
        '''
        Return the PMNS matrix
        '''

        if self.tool == "nusquids" :
            if self.num_neutrinos == 2 :
                theta = [ self.nusquids.Get_MixingAngle(0,1) ]
            elif self.num_neutrinos == 3 :
                theta = [ self.nusquids.Get_MixingAngle(0,1), self.nusquids.Get_MixingAngle(0,2), self.nusquids.Get_MixingAngle(1,2) ]
            else :
                raise Exception("`PMNS` function only supports 2/3 flavors")
            deltacp = self.nusquids.Get_CPPhase(0,2) #TODO check indices
            return get_pmns_matrix( theta=theta, dcp=deltacp )


        elif self.tool == "deimos" :
            return self.solver.PMNS


    def plot_osc_prob_vs_distance(self, 
        # Steer physics
        initial_flavor, 
        energy_GeV, 
        distance_km=None, coszen=None, 
        nubar=False, 
        final_flavor=None,
        # Plotting
        fig=None, ax=None, 
        label=None, 
        title=None,
        xscale="linear",
        ylim=None,
        **plot_kw
    ) :
        '''
        Compute and plot the oscillation probability, vs propagation distance
        '''

        # Handle distance vs coszen
        if self.atmospheric :
            assert coszen is not None
            dist_kw = {"coszen" : coszen}
            x = coszen
            xlabel = COSZEN_LABEL
        else :
            assert distance_km is not None
            dist_kw = {"distance_km" : distance_km}
            x = distance_km
            xlabel = DISTANCE_LABEL

        # Check inputs
        assert isinstance(initial_flavor, int)
        assert isinstance(x, np.ndarray)
        assert np.isscalar(energy_GeV)
        assert isinstance(nubar, bool)
        if final_flavor is not None :
            assert isinstance(final_flavor, int)

        # User may provide a figure, otherwise make one
        ny = ( self.num_neutrinos + 1 ) if final_flavor is None else 1
        if fig is None : 
            fig, ax = plt.subplots( nrows=ny, sharex=True, figsize=( 6, 4 if ny == 1 else 2*ny) )
            if ny == 1 :
                ax = [ax]

        else :
            assert ax is not None
            assert len(ax) == ny

        # Handle title
        if title is not None :
            fig.suptitle(title) 

        # Calc osc probs
        osc_probs = self.calc_osc_prob(
            initial_flavor=initial_flavor,
            energy_GeV=energy_GeV,
            **dist_kw
        )

        # Plot oscillations to all possible final states
        final_flavor_values = self.states if final_flavor is None else [final_flavor]
        for i, final_flavor in enumerate(final_flavor_values) :
            ax[i].plot( x, osc_probs[:,final_flavor], label=label, **plot_kw )
            ax[i].set_ylabel( r"$%s$" % self.get_transition_prob_tex(initial_flavor, final_flavor, nubar) )

        # Plot total oscillations to any final state
        if len(final_flavor_values) > 1 :
            osc_probs_flavor_sum = np.sum(osc_probs,axis=1)
            ax[-1].plot( x, osc_probs_flavor_sum, label=label, **plot_kw ) # Dimension 2 is flavor
            ax[-1].set_ylabel( r"$%s$" % self.get_transition_prob_tex(initial_flavor, None, nubar) )

        # Formatting
        if ylim is None :
            ylim = (-0.05, 1.05)
        ax[-1].set_xlabel(xlabel)
        if label is not None :
            ax[0].legend(fontsize=12) # loc='center left', bbox_to_anchor=(1, 0.5), 
        for this_ax in ax :
            this_ax.set_ylim(ylim)
            this_ax.set_xlim(x[0], x[-1])
            this_ax.set_xscale(xscale)
            this_ax.grid(True)
        fig.tight_layout()

        return fig, ax, osc_probs


    def plot_osc_prob_vs_cozen(self, coszen, *args, **kwargs) : # Alias
        return self.plot_osc_prob_vs_distance(coszen=coszen, *args, **kwargs)



    def plot_osc_prob_vs_energy(self, 
        # Steer physics
        initial_flavor, 
        energy_GeV, 
        distance_km=None, coszen=None, 
        nubar=False, 
        final_flavor=None,
        # Plotting
        fig=None, ax=None, 
        label=None, 
        title=None,
        xscale="linear",
        ylim=None,
        plot_LoE=False,
        **plot_kw
    ) :
        '''
        Compute and plot the oscillation probability, vs neutrino energy
        '''

        # Handle distance vs coszen
        if self.atmospheric :
            assert coszen is not None
            dist_kw = {"coszen" : coszen}
            x = coszen
        else :
            assert distance_km is not None
            dist_kw = {"distance_km" : distance_km}
            x = distance_km

        # Check inputs
        assert isinstance(initial_flavor, int)
        assert isinstance(energy_GeV, np.ndarray)
        assert np.isscalar(x)
        assert isinstance(nubar, bool)
        if final_flavor is not None :
            assert isinstance(final_flavor, int)

        # User may provide a figure, otherwise make one
        ny = ( self.num_neutrinos + 1 ) if final_flavor is None else 1
        if fig is None : 
            fig, ax = plt.subplots( nrows=ny, sharex=True, figsize=( 6, 4 if ny == 1 else 2*ny) )
            if ny == 1 :
                ax = [ax]
        else :
            assert ax is not None
            assert len(ax) == ny

        # Handle title
        if title is not None :
            fig.suptitle(title) 

        # Calc osc probs
        osc_probs = self.calc_osc_prob(
            initial_flavor=initial_flavor,
            energy_GeV=energy_GeV,
            **dist_kw
        )

        # Convert to L/E, if requested
        xplot = energy_GeV
        if plot_LoE :
            assert not self.atmospheric, "Need to handle coszen conversion for L/E plot"
            LoE = dist_kw["distance_km"] / energy_GeV
            xplot = LoE

        # Plot oscillations to all possible final states
        final_flavor_values = self.states if final_flavor is None else [final_flavor]
        for i, final_flavor in enumerate(final_flavor_values) :
            ax[i].plot( xplot, osc_probs[:,final_flavor], label=label, **plot_kw )
            ax[i].set_ylabel( r"$%s$" % self.get_transition_prob_tex(initial_flavor, final_flavor, nubar) )

        # Plot total oscillations to any final state
        if len(final_flavor_values) > 1 :
            osc_probs_flavor_sum = np.sum(osc_probs,axis=1)
            ax[-1].plot( xplot, osc_probs_flavor_sum, label=label, **plot_kw ) # Dimension 2 is flavor
            ax[-1].set_ylabel( r"$%s$" % self.get_transition_prob_tex(initial_flavor, None, nubar) )

        # Formatting
        if ylim is None :
            ylim = (-0.05, 1.05)
        if plot_LoE :
            ax[-1].set_xlabel("%s / %s" % (DISTANCE_LABEL, ENERGY_LABEL))
        else :
            ax[-1].set_xlabel(ENERGY_LABEL)
        if label is not None :
            ax[0].legend(fontsize=10) # loc='center left', bbox_to_anchor=(1, 0.5), 
        for this_ax in ax :
            if plot_LoE :
                this_ax.set_xlim(xplot[-1], xplot[0]) # Reverse
            else :
                this_ax.set_xlim(xplot[0], xplot[-1])
            this_ax.set_ylim(ylim)
            this_ax.grid(True)
            this_ax.set_xscale(xscale)
        fig.tight_layout()

        return fig, ax, osc_probs


    def plot_cp_asymmetry() :
        '''
        Plot the CP(T) asymmetry
        '''

        raise NotImplemented("TODO")


    def plot_oscillogram(
        self,
        initial_flavor,
        final_flavor,
        energy_GeV,
        coszen,
        nubar=False,
        title=None,
    ) :
        '''
        Helper function for plotting an atmospheric neutrino oscillogram
        '''
        from deimos.utils.plotting import plot_colormap, value_spacing_is_linear

        assert self.atmospheric, "`plot_oscillogram` can only be called in atmospheric mode"

        #
        # Steerig
        #

        # Plot steering
        transition_prob_tex = self.get_transition_prob_tex(initial_flavor, final_flavor, nubar)
        continuous_map = "jet" # plasma jet
        # diverging_cmap = "seismic" # PuOr_r RdYlGn Spectral


        #
        # Compute osc probs
        #

        # Define osc prob calc settings
        calc_osc_prob_kw = dict(
            initial_flavor=initial_flavor,
            nubar=nubar,
            energy_GeV=energy_GeV,
            coszen=coszen, 
        )

        # Calc osc probs 
        osc_probs = self.calc_osc_prob( **calc_osc_prob_kw )

        # Get chose flavor/rho
        osc_probs = osc_probs[:, :, final_flavor]

        #
        # Plot
        #

        # Create fig
        fig, ax = plt.subplots( figsize=(7, 6) )
        if title is not None :
            fig.suptitle(title) 

        # Plot oscillogram
        plot_colormap( ax=ax, x=energy_GeV, y=coszen, z=osc_probs, vmin=0., vmax=1., cmap=continuous_map, zlabel=r"$%s$"%transition_prob_tex )

        # Format
        xscale = "linear" if value_spacing_is_linear(energy_GeV) else "log"
        yscale = "linear" if value_spacing_is_linear(coszen) else "log"
        ax.set_xscale(xscale)
        ax.set_yscale(yscale)
        ax.set_xlim(energy_GeV[0], energy_GeV[-1])
        ax.set_ylim(coszen[0], coszen[-1])
        ax.set_xlabel(ENERGY_LABEL)
        ax.set_ylabel(COSZEN_LABEL)
        fig.tight_layout()

        return fig, ax, osc_probs


    def plot_right_ascension_vs_energy_2D(
            self,
            # Steer physics
            initial_flavor,
            energy_GeV,
            distance_km=None, coszen=None,
            nubar=False,
            final_flavor=None,
            # Plotting
            fig=None, ax=None,
            label=None,
            title=None,
            xscale="linear",
            ylim=None,
            **plot_kw
    ):
        '''
        Make a 2D plot of oscillation probabilities vs neutrino energy (x-axis) and right ascension (y-axis).
        '''
    
        # Check inputs
        assert isinstance(initial_flavor, int)
        assert isinstance(energy_GeV, np.ndarray)
        assert isinstance(nubar, bool)
        if final_flavor is not None:
            assert isinstance(final_flavor, int)
    
        # User may provide a figure, otherwise make one
        ny = self.num_neutrinos + 1 if final_flavor is None else 1
        if fig is None:
            fig, ax = plt.subplots(nrows=ny, sharex=True, sharey=False, figsize=(6, 4 * ny))
            if ny == 1:
                ax = [ax]
            if title is not None:
                for this_ax in ax:
                    this_ax.set_title(title)  # Set the same title for all subplots
        else:
            assert ax is not None
            assert len(ax) == ny
            assert title is None
            
        # Get a_eV, c and ra for naming the plot
        if self._sme_model_kw:    
            a_eV = self._sme_model_kw.get("a_eV")
            c = self._sme_model_kw.get("c")
            dec_0 = np.deg2rad(self._neutrino_source_kw["dec"][0])
    
        # Set title of figure     
        if self._sme_model_kw:
            fig.suptitle("SME",
                # r"$\delta \sim {:.2f}$".format(dec_0)
                # + r", $a^X = {:.2e} \, \rm GeV$".format(a_eV[0])
                # + r", $a^Y = {:.2e} \, \rm GeV$".format(a_eV[1])
                # + r", $c^X = {:.2e}$".format(c[0])
                # + r", $c^Y = {:.2e}$".format(c[1]),
                fontsize=14,
            )
        else:
            fig.suptitle("Standard osc",
                fontsize=14,
            )
        
        # Handle distance vs coszen
        if self.atmospheric:
            assert coszen is not None
            dist_kw = {"coszen": coszen}
        else:
            assert distance_km is not None
            dist_kw = {"distance_km": distance_km}
    
        # Calculate probabilities
        probabilities2d = self.calc_osc_prob(
            initial_flavor = initial_flavor,
            energy_GeV = energy_GeV,
            **dist_kw
        )
        
        # Transpose array to plot right ascension vs. energy
        probabilities2d = np. transpose(probabilities2d, (1,0,2) )
    
        # Define the possible final states
        final_states = ["e", "\u03BC", "\u03C4"]  # Use unicode characters for mu and tau
        
        # Loop over each final state and create the corresponding plot
        for i, final_flavor in enumerate(final_states):
            
            # Check for values outside the range [0.9, 1.1]
            if np.any(probabilities2d[:, :, i] < -0.1) or np.any(probabilities2d[:, :, i] > 1.1):
                warnings.warn("Values of oscillation probabilities outside the range [0, 1].", UserWarning)
                
            # Plot the results
            if self._sme_model_kw: 
                im = ax[i].pcolormesh(energy_GeV,
                                      np.deg2rad(self._neutrino_source_kw["ra"]),
                                      probabilities2d[:, :, i],
                                      vmin=0, vmax=1.0, 
                                      cmap='RdPu')
                ax[i].set_ylabel("Right Ascension (rad)")
            
            else:
                im = ax[i].pcolormesh(energy_GeV,
                                      coszen,
                                      probabilities2d[:, :, i],
                                      vmin=0, vmax=1.0, 
                                      cmap='RdPu')
                ax[i].set_ylabel("Coszen")
            ax[i].set_xscale(xscale)
    
            # Add colorbar
            cbar = fig.colorbar(im, ax=ax[i], label=r"$P(\nu_{\mu}\rightarrow \nu_{" + final_flavor + r"})$")
    
        # Plot total oscillations to any final state
        if final_flavor is not None:
            osc_probs_flavor_sum = probabilities2d.sum(axis=-1)
            
            # Check for values outside the range [0.9, 1.1]
            if np.any(osc_probs_flavor_sum < 0.9) or np.any(osc_probs_flavor_sum > 1.1):
                warnings.warn("Values outside the range [0.9, 1.1] in osc_probs_flavor_sum.", UserWarning)
            
            if self._sme_model_kw: 
                ax[-1].pcolormesh(energy_GeV,
                    np.deg2rad(self._neutrino_source_kw["ra"]),
                    osc_probs_flavor_sum,
                    vmin=0.9, vmax=1.1,
                    cmap="RdPu")
                
                ax[-1].set_ylabel("Right Asceionsion (rad)")
            
            else:
                im = ax[-1].pcolormesh(energy_GeV,
                    coszen,
                    osc_probs_flavor_sum,
                    vmin=0.9, vmax=1.1,
                    cmap="RdPu")
                
                ax[-1].set_ylabel("Coszen")
            ax[-1].set_xlabel(ENERGY_LABEL)
            ax[-1].set_xscale(xscale)
            
            # Add colorbar
            cbar = fig.colorbar(im, ax=ax[-1], label=r"$P(\nu_{\mu}\rightarrow \nu_{all})$")
    
        plt.tight_layout()
        plt.show()

        return fig, ax, probabilities2d
    
    
    def plot_declination_vs_energy_2D(
        self,
        # Steer physics
        initial_flavor,
        energy_GeV,
        distance_km=None, coszen=None,
        nubar=False,
        final_flavor=None,
        # Plotting
        fig=None, ax=None,
        label=None,
        title=None,
        xscale="linear",
        ylim=None,
        **plot_kw
    ):
        '''
        Make a 2D plot of oscillation probabilities vs neutrino energy (x-axis) and declination (y-axis).
        '''
    
        # Check inputs
        assert isinstance(initial_flavor, int)
        assert isinstance(energy_GeV, np.ndarray)
        assert isinstance(nubar, bool)
        if final_flavor is not None:
            assert isinstance(final_flavor, int)
    
        # User may provide a figure, otherwise make one
        ny = self.num_neutrinos + 1 if final_flavor is None else 1
        if fig is None:
            fig, ax = plt.subplots(nrows=ny, sharex=True, sharey=False, figsize=(6, 4 * ny))
            if ny == 1:
                ax = [ax]
            if title is not None:
                for this_ax in ax:
                    this_ax.set_title(title)  # Set the same title for all subplots
        else:
            assert ax is not None
            assert len(ax) == ny
            assert title is None
    
        # Set title of figure     
        # TODO adjust title of plots
        if self._sme_model_kw:
            fig.suptitle("SME",
                # r"$\alpha \sim {:.2f}$".format(ra_0)
                # + r", $a^X = {:.2e} \, \rm GeV$".format(a_eV[0])
                # + r", $a^Y = {:.2e} \, \rm GeV$".format(a_eV[1])
                # + r", $c^X = {:.2e}$".format(c[0])
                # + r", $c^Y = {:.2e}$".format(c[1]),
                fontsize=14,
            )
        else:
            fig.suptitle("Standard osc",
                fontsize=14,
            )
        
        # Handle distance vs coszen
        if self.atmospheric:
            assert coszen is not None
            dist_kw = {"coszen": coszen}
        else:
            assert distance_km is not None
            dist_kw = {"distance_km": distance_km}
    
        # Calculate probabilities
        probabilities2d = self.calc_osc_prob(
            initial_flavor=initial_flavor,
            energy_GeV=energy_GeV,
            **dist_kw
        )
    
        # Transpose array to plot declination vs. energy
        probabilities2d = np.transpose(probabilities2d, (1, 0, 2))
    
        # Define the possible final states
        final_states = ["e", "\u03BC", "\u03C4"]  # Use unicode characters for mu and tau
            
        # Loop over each final state and create the corresponding plot
        for i, final_flavor in enumerate(final_states):
            
            # Check for values outside the range [0.9, 1.1]
            if np.any(probabilities2d[:, :, i] < -0.1) or np.any(probabilities2d[:, :, i] > 1.1):
                warnings.warn("Values of oscillation probabilities outside the range [0, 1].", UserWarning)
              
            # Plot the results
            if self._sme_model_kw: 
                im = ax[i].pcolormesh(energy_GeV,
                                      np.deg2rad(self._neutrino_source_kw["dec"]),
                                      probabilities2d[:, :, i],
                                      vmin=0, vmax=1.0, 
                                      cmap='RdPu')
                ax[i].set_ylabel("Declination (rad)")
            
            else:
                im = ax[i].pcolormesh(energy_GeV,
                                      coszen,
                                      probabilities2d[:, :, i],
                                      vmin=0, vmax=1.0, 
                                      cmap='RdPu')
                ax[i].set_ylabel("Coszen")
                
            ax[i].set_xscale(xscale)
    
            # Add colorbar
            cbar = fig.colorbar(im, ax=ax[i], label=r"$P(\nu_{\mu}\rightarrow \nu_{" + final_flavor + r"})$")
    
        # Plot total oscillations to any final state
        if final_flavor is not None:
            osc_probs_flavor_sum = probabilities2d.sum(axis=-1)
            
            # Check for values outside the range [0.9, 1.1]
            if np.any(osc_probs_flavor_sum < 0.9) or np.any(osc_probs_flavor_sum > 1.1):
                warnings.warn("Values outside the range [0.9, 1.1] in osc_probs_flavor_sum.", UserWarning)
            
            if self._sme_model_kw: 
                ax[-1].pcolormesh(energy_GeV,
                    np.deg2rad(self._neutrino_source_kw["dec"]),
                    osc_probs_flavor_sum,
                    vmin=0.9, vmax=1.1,
                    cmap="RdPu")
                
                ax[-1].set_ylabel("Declination (rad)")
            
            else:
                im = ax[-1].pcolormesh(energy_GeV,
                    coszen,
                    osc_probs_flavor_sum,
                    vmin=0.9, vmax=1.1,
                    cmap="RdPu")
                
                ax[-1].set_ylabel("Coszen")
            ax[-1].set_xlabel(ENERGY_LABEL)
            ax[-1].set_xscale(xscale)
        
            #Add colorbar
            cbar = fig.colorbar(im, ax=ax[-1], label=r"$P(\nu_{\mu}\rightarrow \nu_{all})$")
    
        plt.tight_layout()
        plt.show()
    
        return fig, ax, probabilities2d
    
    
    def plot_declination_vs_energy_2D_diff(
    self,
    # Steer physics
    initial_flavor,
    energy_GeV,
    distance_km=None, coszen=None,
    nubar=False,
    final_flavor=None,
    # Plotting
    fig=None, ax=None,
    label=None,
    title=None,
    xscale="linear",
    ylim=None,
    **plot_kw
    ):
        '''
        Make a 2D plot of the difference of oscillation probabilities between standard osc and with SME
        vs neutrino energy (x-axis) and declination (y-axis).
        '''
    
        # Check inputs
        assert isinstance(initial_flavor, int)
        assert isinstance(energy_GeV, np.ndarray)
        assert isinstance(nubar, bool)
        if final_flavor is not None:
            assert isinstance(final_flavor, int)
    
        # User may provide a figure, otherwise make one
        ny = self.num_neutrinos + 1 if final_flavor is None else 1
        if fig is None:
            fig, ax = plt.subplots(nrows=ny, sharex=True, sharey=False, figsize=(6, 4 * ny))
            if ny == 1:
                ax = [ax]
            if title is not None:
                for this_ax in ax:
                    this_ax.set_title(title)  # Set the same title for all subplots
        else:
            assert ax is not None
            assert len(ax) == ny
            assert title is None
    
        # Get a_eV, c and ra for naming the plot
        if self._sme_model_kw:    
            a_eV = self._sme_model_kw.get("a_eV")
            c = self._sme_model_kw.get("c")
            ra_0 = np.deg2rad(self._neutrino_source_kw["ra"][0])
    
        # Set title of figure     
        if self._sme_model_kw:
            fig.suptitle("SME",
                # r"$\alpha \sim {:.2f}$".format(ra_0)
                # + r", $a^X = {:.2e} \, \rm GeV$".format(a_eV[0])
                # + r", $a^Y = {:.2e} \, \rm GeV$".format(a_eV[1])
                # + r", $c^X = {:.2e}$".format(c[0])
                # + r", $c^Y = {:.2e}$".format(c[1]),
                fontsize=14,
            )
        else:
            fig.suptitle("Standard osc",
                fontsize=14,
            )
        
        # Handle distance vs coszen
        if self.atmospheric:
            assert coszen is not None
            dist_kw = {"coszen": coszen}
        else:
            assert distance_km is not None
            dist_kw = {"distance_km": distance_km}
            
        # Calculate probabilities for the non-standard case
        sme_probabilities2d = self.calc_osc_prob(
            initial_flavor=initial_flavor,
            energy_GeV=energy_GeV,
            **dist_kw
        )
        
        # Call method to set parameters for the standard oscillation case
        self.set_std_osc()
        
        # Calculate probabilities for the standard case
        standard_probabilities2d = self.calc_osc_prob(
            initial_flavor=initial_flavor,
            energy_GeV=energy_GeV,
            **dist_kw
        )

        # Calculate the difference of probabilities between non-standard and standard cases
        diff_probabilities2d = sme_probabilities2d - standard_probabilities2d

        # Select a colormap
        cmap_diff = 'bwr'

        # Define the possible final states
        final_states = ["e", "\u03BC", "\u03C4"]  # Use unicode characters for mu and tau
        
        # Loop over each final state and create the corresponding plot
        for i, final_flavor in enumerate(final_states):
            
            # Check for values outside the range [0.9, 1.1]
            if np.any(diff_probabilities2d[:, :, i] < -1.1) or np.any(diff_probabilities2d[:, :, i] > 1.1):
                warnings.warn("Values of the difference of the oscillation probabilities outside the range [-1, 1].", UserWarning)
            
            # Use the custom colormap to plot the difference of probabilities
            if self._sme_model_kw: 
                im = ax[i].pcolormesh(energy_GeV,
                                      np.deg2rad(self._neutrino_source_kw["dec"]),
                                      diff_probabilities2d[:, :, i],
                                      vmin=-1.0, vmax=1.0, 
                                      cmap=cmap_diff)
                ax[i].set_ylabel("Declination (rad)")
            
            else:
                im = ax[i].pcolormesh(energy_GeV,
                                      coszen,
                                      diff_probabilities2d[:, :, i],
                                      vmin=-1.0, vmax=1.0, 
                                      cmap=cmap_diff)
                ax[i].set_ylabel("Coszen")
            
            ax[i].set_xscale(xscale)
            
            # Add colorbar
            cbar = fig.colorbar(im, ax=ax[i], label=r"$\Delta P(\nu_{\mu}\rightarrow \nu_{" + final_flavor + r"})$")
    
            
        # Plot total oscillations to any final state
        if final_flavor is not None:
            osc_probs_flavor_sum = diff_probabilities2d.sum(axis=-1)
            
            # Check for values outside the range [0.9, 1.1]
            if np.any(osc_probs_flavor_sum < -0.1) or np.any(osc_probs_flavor_sum > 0.1):
                warnings.warn("Values outside the range [-0.1, 0.1] in osc_probs_flavor_sum.", UserWarning)

            if self._sme_model_kw: 
                ax[-1].pcolormesh(energy_GeV,
                    np.deg2rad(self._neutrino_source_kw["dec"]),
                    osc_probs_flavor_sum,
                    vmin=-0.1, vmax=0.1,
                    cmap=cmap_diff)
                
                ax[-1].set_ylabel("Declination (rad)")
            
            else:
                im = ax[-1].pcolormesh(energy_GeV,
                    coszen,
                    osc_probs_flavor_sum,
                    vmin=-0.1, vmax=0.1,
                    cmap=cmap_diff)
                
                ax[-1].set_ylabel("Coszen")
            ax[-1].set_xlabel(ENERGY_LABEL)
            ax[-1].set_xscale(xscale)
        
            #Add colorbar
            cbar = fig.colorbar(im, ax=ax[-1], label=r"$\Delta P(\nu_{\mu}\rightarrow \nu_{all})$")
    

        return fig, ax, diff_probabilities2d
    
    
    def plot_right_ascension_vs_energy_2D_diff(
    self,
    # Steer physics
    initial_flavor,
    energy_GeV,
    distance_km=None, coszen=None,
    nubar=False,
    final_flavor=None,
    # Plotting
    fig=None, ax=None,
    label=None,
    title=None,
    xscale="linear",
    ylim=None,
    **plot_kw
    ):
        '''
        Make a 2D plot of the difference of oscillation probabilities between standard osc and with SME
        vs neutrino energy (x-axis) and declination (y-axis).
        '''
    
        # Check inputs
        assert isinstance(initial_flavor, int)
        assert isinstance(energy_GeV, np.ndarray)
        assert isinstance(nubar, bool)
        if final_flavor is not None:
            assert isinstance(final_flavor, int)
    
        # User may provide a figure, otherwise make one
        ny = self.num_neutrinos + 1 if final_flavor is None else 1
        if fig is None:
            fig, ax = plt.subplots(nrows=ny, sharex=True, sharey=False, figsize=(6, 4 * ny))
            if ny == 1:
                ax = [ax]
            if title is not None:
                for this_ax in ax:
                    this_ax.set_title(title)  # Set the same title for all subplots
        else:
            assert ax is not None
            assert len(ax) == ny
            assert title is None
    
        # Get a_eV, c and ra for naming the plot
        if self._sme_model_kw:    
            a_eV = self._sme_model_kw.get("a_eV")
            c = self._sme_model_kw.get("c")
            dec_0 = np.deg2rad(self._neutrino_source_kw["dec"][0])
    
        # Set title of figure     
        if self._sme_model_kw:
            fig.suptitle("SME",
                # r"$\delta \sim {:.2f}$".format(dec_0)
                # + r", $a^X = {:.2e} \, \rm GeV$".format(a_eV[0])
                # + r", $a^Y = {:.2e} \, \rm GeV$".format(a_eV[1])
                # + r", $c^X = {:.2e}$".format(c[0])
                # + r", $c^Y = {:.2e}$".format(c[1]),
                fontsize=14,
            )
        else:
            fig.suptitle("Standard osc",
                fontsize=14,
            )
        
        # Handle distance vs coszen
        if self.atmospheric:
            assert coszen is not None
            dist_kw = {"coszen": coszen}
        else:
            assert distance_km is not None
            dist_kw = {"distance_km": distance_km}
            
        # Calculate probabilities for the non-standard case
        sme_probabilities2d = self.calc_osc_prob(
            initial_flavor=initial_flavor,
            energy_GeV=energy_GeV,
            **dist_kw
        )
        
        # Call method to set parameters for the standard oscillation case
        self.set_std_osc()
        
        # Calculate probabilities for the standard case
        standard_probabilities2d = self.calc_osc_prob(
            initial_flavor=initial_flavor,
            energy_GeV=energy_GeV,
            **dist_kw
        )

        # Calculate the difference of probabilities between non-standard and standard cases
        diff_probabilities2d = sme_probabilities2d - standard_probabilities2d

        # Create a custom colormap
        cmap_diff = 'bwr'
        
        # Define the possible final states
        final_states = ["e", "\u03BC", "\u03C4"]  # Use unicode characters for mu and tau
        
        # Loop over each final state and create the corresponding plot
        for i, final_flavor in enumerate(final_states):
            
            # Check for values outside the range [0.9, 1.1]
            if np.any(diff_probabilities2d[:, :, i] < -1.1) or np.any(diff_probabilities2d[:, :, i] > 1.1):
                warnings.warn("Values of the difference of the oscillation probabilities outside the range [-1, 1].", UserWarning)
               
            # Use the custom colormap to plot the difference of probabilities
            if self._sme_model_kw: 
                im = ax[i].pcolormesh(energy_GeV,
                                      np.deg2rad(self._neutrino_source_kw["ra"]),
                                      diff_probabilities2d[:, :, i],
                                      vmin=-1.0, vmax=1.0, 
                                      cmap=cmap_diff)
                ax[i].set_ylabel("Declination (rad)")
            
            else:
                im = ax[i].pcolormesh(energy_GeV,
                                      coszen,
                                      diff_probabilities2d[:, :, i],
                                      vmin=-1.0, vmax=1.0, 
                                      cmap=cmap_diff)
                ax[i].set_ylabel("Coszen")
            ax[i].set_xscale(xscale)
            
            # Add colorbar
            cbar = fig.colorbar(im, ax=ax[i], label=r"$\Delta P(\nu_{\mu}\rightarrow \nu_{" + final_flavor + r"})$")
    
            
        # Plot total oscillations to any final state
        if final_flavor is not None:
            osc_probs_flavor_sum = diff_probabilities2d.sum(axis=-1)
            
            # Check for values outside the range [0.9, 1.1]
            if np.any(osc_probs_flavor_sum < -0.1) or np.any(osc_probs_flavor_sum > 0.1):
                warnings.warn("Values outside the range [-0.1, 0.1] in osc_probs_flavor_sum.", UserWarning)
            
            if self._sme_model_kw: 
                ax[-1].pcolormesh(energy_GeV,
                    np.deg2rad(self._neutrino_source_kw["ra"]),
                    osc_probs_flavor_sum,
                    vmin=-0.1, vmax=0.1,
                    cmap=cmap_diff)
                
                ax[-1].set_ylabel("Right Ascension (rad)")
            
            else:
                im = ax[-1].pcolormesh(energy_GeV,
                    coszen,
                    osc_probs_flavor_sum,
                    vmin=-0.1, vmax=0.1,
                    cmap="RdPu")
                
                ax[-1].set_ylabel("Coszen")
            ax[-1].set_xlabel(ENERGY_LABEL)
            ax[-1].set_xscale(xscale)
        
            #Add colorbar
            cbar = fig.colorbar(im, ax=ax[-1], label=r"$\Delta P(\nu_{\mu}\rightarrow \nu_{all})$")
    

        return fig, ax, diff_probabilities2d
    

    def plot_healpix_map(
            self,
            healpix_map,
            visible_sky_map,
            nside,
            title,
            cbar_label,
            cmap='viridis',
            min_val = -1,
            max_val = 1,
            ):
        
        #Plot in the mollview projection
        projected_map = hp.mollview(
                            map = healpix_map, 
                            title=title + "\n", 
                            cmap=cmap,
                            xsize=2000,
                            # rotation in the form (lon, lat, psi) (unit: degrees) : the point at longitude lon and latitude lat will be at the center.
                            # An additional rotation of angle psi around this direction is applied.
                            rot=(180, 0, 0),
                            # equatorial (celestial) coordinate system
                            coord='C',
                            # east towards left, west towards right
                            flip = 'astro',
                            min=min_val,
                            max=max_val,
                            cbar=False,
                            return_projected_map=True,
                            # Allow overlaying
                            hold = True
                            )
        # Overlay the visible_sky map
        hp.mollview(
            map=visible_sky_map,  # Add the visible_sky map as an overlay
            cmap='Greys',  # Set the colormap to 'Greys'
            xsize=2000,
            # An additional rotation of angle psi around this direction is applied.
            rot=(180, 0, 0),
            # equatorial (celestial) coordinate system
            coord='C',
            # east towards left, west towards right
            flip = 'astro',
            min = 0,
            max =1,
            # Set opacity to 0.2
            alpha=visible_sky_map,  
            # Allow overlaying
            reuse_axes=True,
            cbar=False,
            )
        
        # Add meridians and parallels
        hp.graticule()
        
        # Add declination labels
        for dec in np.arange(-75, 0, 15):
            #lonlat If True, theta and phi are interpreted as longitude and latitude in degree, otherwise, as colatitude and longitude in radian
            hp.projtext(359.9, dec, "\n" + f"{dec}°   ", lonlat=True, color="black", ha="right", va="center")  
        for dec in np.arange(0, 76, 15):
            if dec == 0:
                hp.projtext(359.9, dec, r"Declination $\delta$" + "\n\n", lonlat=True, color="black", ha="right", va="center", rotation ='vertical')
                continue
            hp.projtext(359.9, dec, f"{dec}°   " + "\n", lonlat=True, color="black", ha="right", va="center")
            
        # Add the right ascension labels
        hp.projtext(359.9, 0, "24h ", lonlat=True, color="black", ha="right", va="center")
        hp.projtext(0, 0, " 0h", lonlat=True, color="black", ha="left", va="center")
        hp.projtext(180, -90, "\n\n\n12h" + "\nRight Ascension " +  r"$\alpha$", lonlat=True, color="black", ha="center", va="center")
    
        # Create an empty image plot as a mappable for the colorbar
        img = plt.imshow(projected_map, cmap=cmap, vmin=min_val, vmax=max_val)
        cb = plt.colorbar(img, shrink=0.7)  # You can adjust the size of the colorbar using 'shrink' parameter
        cb.set_label(label=cbar_label)  
        
        # Save the plot to a file
        # Replace Greek symbols with English letters
        cbar_label = cbar_label.replace(r"$\Delta", "Delta")
        cbar_label = cbar_label.replace(r"\nu", "nu")
        cbar_label = cbar_label.replace(r"\mu", "mu")
        cbar_label = cbar_label.replace(r"\rightarrow", "to")
        cbar_label = cbar_label.replace(r"_", " ")

        # Remove any remaining LaTeX commands (e.g., "{", "}", "$")
        cbar_label = re.sub(r"{|}|\$", "", cbar_label)

        # Remove spaces and add underscores between words
        cbar_label = cbar_label.replace(" ", "_")
        title = title.replace(" ", "_")
        plt.savefig(title + cbar_label + ".png", bbox_inches='tight')
        plt.show()
        
        # Close the current plot to free memory
        plt.close()  
        
        
    def plot_osc_prob_skymap_2D(
        self,
        # Steer physics
        initial_flavor,
        energy_GeV,
        distance_km=None, coszen=None,
        date_str = None,
        nubar=False,
        final_flavor=None,
        #Plotting
        resolution= 8,
        cmap='RdPu',
        ) :
        
        '''
        Make a 2D plot of neutrino oscillation probabilities vs right ascension and declination
        for a fixed energy.
        '''
        
        # Check inputs
        assert isinstance(initial_flavor, int)
        assert isinstance(energy_GeV, np.ndarray)
        assert isinstance(nubar, bool)
        if final_flavor is not None:
            assert isinstance(final_flavor, int)
        assert resolution > 0 and (resolution & (resolution - 1)) == 0, "resolution needs to be a power of 2."
    
        # Get a_eV, c and ra for naming the plot
        if self._sme_model_kw:    
            a_eV = self._sme_model_kw.get("a_eV")
            c = self._sme_model_kw.get("c")
        
        # Handle distance vs coszen
        if self.atmospheric:
            assert coszen is not None
            dist_kw = {"coszen": coszen}
        else:
            assert distance_km is not None
            dist_kw = {"distance_km": distance_km}
        
        # Generate minimal ra and dec to cover all pixels of healpy map
        # Number of pixels of healpy map
        npix = hp.nside2npix(nside=resolution)
        
        # Convert pixel to polar coordinates (in deg)
        right_ascension_flat, declination_flat = hp.pix2ang(nside=resolution, ipix=np.arange(npix), lonlat=True)
        ra_rad, dec_rad = np.deg2rad(right_ascension_flat), np.deg2rad(declination_flat)

        # NEUTRINO SOURCE NO LONGER IMPLEMENTED
        # date_str = self._neutrino_source_kw["date_str"]
        # self._neutrino_source_kw = None
        # self.set_neutrino_source(# Date, Time and Timezone
        #                         date_str = date_str,
        #                         # Location on the sky
        #                         ra_deg = right_ascension_flat, 
        #                         dec_deg = declination_flat,
        #                         )
        
        #Store dictionaries for later use
        # neutrinos_dict = self._neutrino_source_kw
        sme_dict = self._sme_model_kw
        
        # Evaluate which pixels are above the horizon 
        _, alt, _ = self.detector_coords.get_coszen_altitude_and_azimuth(time = date_str, ra_deg = right_ascension_flat, dec_deg = declination_flat)
        # Create a mask for altitudes between 0 and 90 degrees
        mask = (alt >= 0) # & (alt <= 90)
        # Create an array of zeros with the same shape as alt
        visible_sky = np.zeros_like(alt)
        
        # Set the elements where the condition is met to .4
        visible_sky[mask] = .4

        
        
        # Calculate probabilities with SME model
        sme_probabilities2d = self.calc_osc_prob(
            initial_flavor=initial_flavor,
            energy_GeV=energy_GeV,
            ra_rad = ra_rad,
            dec_rad = dec_rad,
            **dist_kw
        )
       
        #
        # Plot each healpix map
        #
        
        #number of plots
        ny = self.num_neutrinos if final_flavor is None else 1
        
        # Define the possible final states
        if final_flavor is None:
            final_flavor = ["e", "\u03BC", "\u03C4"]  # Use unicode characters for mu and tau
        else:
            final_flavor = [final_flavor] 
            
        for i in range(len(energy_GeV)):
            healpix_maps_flavours = sme_probabilities2d[i,:,:]
            
            # Round to two significant digits
            rounded_energy = round(energy_GeV[i], -int(np.floor(np.log10(abs(energy_GeV[i]))) - 1))
            
            # Display in scientific notation
            formatted_energy = f"{rounded_energy:.2e} GeV"
        
            for j in range(ny):
                single_healpix_map = healpix_maps_flavours[:,j]
                # Check if any values are outside the range [-1, 1]
                if np.any(single_healpix_map < -.1) or np.any(single_healpix_map > 1.1):
                    warnings.warn("Values of the difference of the oscillation probabilities outside the range [-1, 1].", UserWarning)

                #Plot the difference in oscillation probabilities for all flavours
                self.plot_healpix_map(
                    healpix_map=single_healpix_map, 
                    visible_sky_map=visible_sky,
                    nside=resolution, 
                    title=formatted_energy,
                    cbar_label=r"$P(\nu_{\mu}\rightarrow \nu_{" + final_flavor[j] + r"})$", 
                    cmap=cmap,
                    min_val=0
                )
               
            healpix_maps_sum_flavours = np.sum(healpix_maps_flavours, axis=1)
            healpix_maps_sum_flavours = np.squeeze(healpix_maps_sum_flavours)
            
            # Check if any values are outside the range [-0.1, 0.1]
            if np.any(single_healpix_map < -0.1) or np.any(single_healpix_map > 0.1):
                warnings.warn("Values of the sum of the difference of the oscillation probabilities outside the range [-0.1, 0.1].", UserWarning)
            
            # Plot sum of flavours 
            self.plot_healpix_map(
                healpix_map=healpix_maps_sum_flavours,
                visible_sky_map=visible_sky, 
                nside=resolution, 
                title=formatted_energy,
                cbar_label=r"$P(\nu_{\mu}\rightarrow \nu_{all})$", 
                cmap=cmap,
                max_val=1.1,
                min_val=0.9,
            )
            
        return sme_probabilities2d
    
    def plot_osc_prob_skymap_2D_diff(
        self,
        # Steer physics
        initial_flavor,
        energy_GeV,
        distance_km=None, coszen=None,
        nubar=False,
        final_flavor=None,
        #Plotting
        resolution= 8,
        cmap='bwr',
        ) :
        
        '''
        Make a 2D plot of neutrino oscillation probabilities vs right ascension and declination
        for a fixed energy.
        '''
        
        # Check inputs
        assert isinstance(initial_flavor, int)
        assert isinstance(energy_GeV, np.ndarray)
        assert isinstance(nubar, bool)
        if final_flavor is not None:
            assert isinstance(final_flavor, int)
        assert resolution > 0 and (resolution & (resolution - 1)) == 0, "resolution needs to be a power of 2."
    
        # Get a_eV, c and ra for naming the plot
        if self._sme_model_kw:    
            a_eV = self._sme_model_kw.get("a_eV")
            c = self._sme_model_kw.get("c")
        
        # Handle distance vs coszen
        if self.atmospheric:
            assert coszen is not None
            dist_kw = {"coszen": coszen}
        else:
            assert distance_km is not None
            dist_kw = {"distance_km": distance_km}
        
        # Generate minimal ra and dec to cover all pixels of healpy map
        # Number of pixels of healpy map
        npix = hp.nside2npix(nside=resolution)
        
        # Convert pixel to polar coordinates (in deg)
        right_ascension_flat, declination_flat = hp.pix2ang(nside=resolution, ipix=np.arange(npix), lonlat=True)
        date_str = self._neutrino_source_kw["date_str"]
        self._neutrino_source_kw = None
        self.set_neutrino_source(# Date, Time and Timezone
                                date_str = date_str,
                                # Location on the sky
                                ra_deg = right_ascension_flat, 
                                dec_deg = declination_flat,
                                )
        
        #Store dictionaries for later use
        neutrinos_dict = self._neutrino_source_kw
        sme_dict = self._sme_model_kw
        
        # Set coszen values to the values corresponding to the different pixels of the healpix map
        self.skymap_use = False
        
        # Calculate probabilities with SME model
        sme_probabilities2d = self.calc_osc_prob(
            initial_flavor=initial_flavor,
            energy_GeV=energy_GeV,
            **dist_kw
        )
        # Call method to set parameters for the standard oscillation case
        self.set_std_osc()
        
        # Set _neutrino_source_kw values again and _sme_model_kw to zero to ensure 
        # that standard_probabilities2d has the same shape as sme_probabilities2d
        self._neutrino_source_kw = neutrinos_dict
        
        # Calculate probabilities for the standard case
        standard_probabilities2d = self.calc_osc_prob(
            initial_flavor=initial_flavor,
            energy_GeV=energy_GeV,
            **dist_kw
        )
        
        # Calculate the difference of probabilities between non-standard and standard cases
        diff_probabilities = sme_probabilities2d - standard_probabilities2d
        
        #
        # Plot each healpix map
        #
        
        #number of plots
        ny = self.num_neutrinos if final_flavor is None else 1
        
        # Define the possible final states
        if final_flavor is None:
            final_flavor = ["e", "\u03BC", "\u03C4"]  # Use unicode characters for mu and tau
        else:
            final_flavor = [final_flavor] 
            
        for i in range(len(energy_GeV)):
            healpix_maps_flavours = diff_probabilities[i,:,:]
            
            # Round to two significant digits
            rounded_energy = round(energy_GeV[i], -int(np.floor(np.log10(abs(energy_GeV[i]))) - 1))
            
            # Display in scientific notation
            formatted_energy = f"{rounded_energy:.2e} GeV"
        
            for j in range(ny):
                single_healpix_map = healpix_maps_flavours[:,j]
                # Check if any values are outside the range [-1, 1]
                if np.any(single_healpix_map < -1.1) or np.any(single_healpix_map > 1.1):
                    warnings.warn("Values of the difference of the oscillation probabilities outside the range [-1, 1].", UserWarning)

                #Plot the difference in oscillation probabilities for all flavours
                self.plot_healpix_map(
                    healpix_map=single_healpix_map, 
                    nside=resolution, 
                    title=formatted_energy,
                    cbar_label=r"$\Delta P(\nu_{\mu}\rightarrow \nu_{" + final_flavor[j] + r"})$", 
                    cmap=cmap
                )
               
            healpix_maps_sum_flavours = np.sum(healpix_maps_flavours, axis=1)
            healpix_maps_sum_flavours = np.squeeze(healpix_maps_sum_flavours)
            
            # Check if any values are outside the range [-0.1, 0.1]
            if np.any(single_healpix_map < -0.1) or np.any(single_healpix_map > 0.1):
                warnings.warn("Values of the sum of the difference of the oscillation probabilities outside the range [-0.1, 0.1].", UserWarning)
            
            # Plot sum of flavours 
            self.plot_healpix_map(
                healpix_map=healpix_maps_sum_flavours, 
                nside=resolution, 
                title=formatted_energy,
                cbar_label=r"$\Delta P(\nu_{\mu}\rightarrow \nu_{all})$", 
                cmap=cmap
            )
            
        return sme_probabilities2d, standard_probabilities2d 
            
    def compare_models(
        self,
        model_defs,
        initial_flavors,
        energy_GeV,
        distance_km, 
        include_std_osc=True,
    ) :
        '''
        Compare the different models/cases specified by `model_defs`

        model_defs : list of dicts
            Each dict must/can have the following entries:
                "calc_basis" (required)
                "D_matrix_basis" (required)
                "D_matrix" (required)
                "n" (required)
                "label" (required)
                "color" (optional)
                "linestyle" (optional)
        '''

        #TODO add comparison w.r.t. energy

        # Check inputs
        #TODO

        # Plot steering
        color_scale = ColorScale("hsv", len(model_defs))

        # Output containers
        figures = []

        # Loop over initial flavors
        for initial_flavor in initial_flavors :

            fig, ax = plt.subplots( nrows=self.num_neutrinos+1, sharex=True, figsize=(6,7) )
            figures.append(fig)

            # Plot std osc
            if include_std_osc :
                self.set_std_osc()
                self.plot_osc_prob_vs_distance(fig=fig, initial_flavor=initial_flavor, energy_GeV=energy_GeV, distance_km=distance_km, color="lightgrey", label="Std osc")

            # Loop over models/cases
            for i_model, model_dict in enumerate(model_defs) :

                # Plot steering
                label = model_dict["label"]
                color = model_dict["color"] if "color" in model_dict else color_scale.get(i_model)
                linestyle = model_dict["linestyle"] if "linestyle" in model_dict else "-"

                # Set physics params
                self.set_calc_basis(model_dict["calc_basis"])
                self.set_decoherence_D_matrix_basis(model_dict["D_matrix_basis"])
                self.set_decoherence_D_matrix( D_matrix_eV=model_dict["D_matrix"], n=model_dict["n"] )

                # Plot
                self.plot_osc_prob_vs_distance(fig=fig, initial_flavor=initial_flavor, energy_GeV=energy_GeV, distance_km=distance_km, color=color, linestyle=linestyle, label=label ) 

            # Add long range behaviour lines
            #TODO

            # Format
            ax[-1].set_xlabel(DISTANCE_LABEL)
            ax[0].legend( loc="upper right", fontsize=10 ) #TODO put to the right of the ax
            fig.quick_format( ylim=(-0.01,1.01), legend=False )

        return figures




def define_matching_perturbation_and_lindblad_calculators(num_neutrinos=3) :
    '''
    Create DecoherenceToyModel and OscCalculator instances
    with common osc parmas, etc.

    Allows for easy comparison between the models.
    '''

    #TODO Make this more flexible by making this a member function of OscCalculator
    # which returns a compatible DecoherenceToyModel instance.


    from deimos.utils.toy_model.decoherence_toy_model import DecoherenceToyModel, get_neutrino_masses

    #
    # Define system
    #

    # Get the system definition
    flavors = FLAVORS
    mass_splittings_eV2 = MASS_SPLITTINGS_eV2
    mixing_angles_rad = MIXING_ANGLES_rad
    deltacp = DELTACP_rad

    #TODO store flavor labels in class, or integrate with OscCalculator

    # Assuming some (arbitrary, at least w.r.t. oscillations) lightest neutrino mass, get masses from mass splittings
    lowest_neutrino_mass_eV = 0. #1.e-3
    masses_eV = get_neutrino_masses(lowest_neutrino_mass_eV, mass_splittings_eV2)

    # Get PMNS
    PMNS = get_pmns_matrix(mixing_angles_rad, dcp=deltacp)


    #
    # Create toy model
    #h

    perturbation_toy_model = DecoherenceToyModel(
        num_states=num_neutrinos,
        mass_state_masses_eV=masses_eV,
        PMNS=PMNS,
        seed=12345,
    )


    #
    # Create Lindblad lindblad_calculator
    #

    lindblad_calculator = OscCalculator(
        tool="dmos", #TODO nusquids
        atmospheric=False,
        num_neutrinos=num_neutrinos,
    )

    lindblad_calculator.set_matter("vacuum")

    lindblad_calculator.set_mass_splittings(*mass_splittings_eV2)
    lindblad_calculator.set_mixing_angles(*mixing_angles_rad, deltacp=deltacp)

    # Handle basis choice
    lindblad_calculator.set_calc_basis("nxn")


    #
    # Checks
    #

    assert np.array_equal(perturbation_toy_model.PMNS, lindblad_calculator.PMNS)
    assert np.isclose(perturbation_toy_model.get_mass_splittings()[0], mass_splittings_eV2[0]) # 21
    if num_neutrinos > 2 :
        assert np.isclose(perturbation_toy_model.get_mass_splittings()[1], mass_splittings_eV2[1]) # 31

    return perturbation_toy_model, lindblad_calculator

