# DEnsItyMatrixOscillationSolver - DEIMOS

A simple python-based neutrino oscillation solver using the density matrix formalism.

Primary author: Tom Stuttard (stuttard@icecube.wisc.edu)

Other contributors: Johann Nikolaides, Simon Hilding-Nørkjær and Martin Langgård Ravn.


## Installation

DEIMOS can be installed from source on a Linux system following the instructions below. We recommend installing DEIMOS within a conda env or similar.

1) **Clone repository:**

```
cd <path/to/install/location>
git clone https://github.com/ts4051/deimos.git
```

2) **Add to python env**:

There are two options to add DEIMOS to your python env:

*2a. Using `pip`*

```
cd <path/to/install/location>/deimos
python -m pip install --upgrade -e . [--user]
```

or 


*2b. Using `PYTHONPATH`*

```
export PYTHONPATH=<path/to/install/location>/deimos:$PYTHONPATH
```


## External integrating tools

*Note that this section is optional.*

DEIMOS includes a class `OscCalculator` that connects several oscillation probability calculators under a common interface, as well as other useful tools. These are:

* `nuSQuIDS` [link](https://github.com/arguelles/nuSQuIDS) - alternative density matrix based oscillation solver
  * Note that a [fork/branch](https://github.com/ts4051/nuSQuIDS/tree/bsm) of nuSQuIDS with comparable model implementations to DEIMOS is available.
* `Prob3` [link](https://github.com/rogerwendell/Prob3plusplusS) - alternative oscillation solver - **coming soon...**
* `OscProb` [link](https://github.com/joaoabcoelho/OscProb) - alternative oscillation solver - **coming soon...**
* `MCEq` [link](https://github.com/mceq-project/MCEq) - atmopsheric neutrino flux calculator

DEIMOS will activate support for these features (at run-time) if they are available (e.g. if it find it can import the packages). Therefore, to use these features within DEIMOS, simpy install these packages as per the instructions provided by the package developers, making sure to add them to your python environment.

### External tool installation helper script

Alternatively, a script for installing these external tools within a conda environment is available and can be run as follows. However, note that this is in beta, and may not work for all operating systems or tool/package versions (in which, manually installing the external tools as per their documentation is recommended). As external packages are developed in non-python languages such as C++, make sure to have the required compilers installed (on Ubuntu/Debian by running `sudo apt install build-essential`)

```
cd <path/to/install/location>/deimos
python ./install.py
python -c "import deimos ; import MCEq ; import nuSQuIDS" # This is to test the packages were sauccesfully installed and added to the python env
```

## Project overview:

The core components of the DEIMOS project are as follows:

* `density_matrix_osc_solver` - core density matrix solver
* `deimos/models` - implementation of new physics models, and scripts for generatintg oscillation probability plots
  * `osc` - standard neutrino oscillations [plotting scripts only]
  * `decoherence` - quantum gravity inspired neutrino decoherence [model implementations and plotting scripts]
  * `liv` - Lorentz Invariance violating neutrino oscillations, using the Standard Model Extension (SME) [model implementations and plotting scripts]
* `deimos/wrapper` - contains a class `OscCalculator` that pulls together various oscillation solvers and related tools into a single interface, and provides useful helper functios for e.g. detector definition, flux calcualtion, plotting, etc


## Examples:

See the following examples to guide how to use DEIMOS and the assoicated wrapper code.

* Standard oscillations
  * `deimos/models/osc/plot_std_osc.py` - basic oscillation probability plots for long baseline, atmospheric and reactor neutrino scenarios (assuming vacuum in all cases)
  * `deimos/models/osc/plot_atmo_osc.py` - plot atmopsheric neutrino oscillogram
  * `deimos/models/osc/plot_matter_effects.py` - plot impact of matter effects on neutrino oscillations
  * `deimos/models/osc/plot_flux.py` - propagate a neutrino flux across the Earth
  * `deimos/models/osc/compare_solvers.py` - compare oscillation calculations between solvers (e.g. DEIMOS and nuSQuIDS)

* Decoherence (see formalism in arXiv:2007.00068)
  * `deimos/models/decoherence/plot_reactor_and_lbl_decoherence.py` - plot decoherence in long-baseline and reactor experiments
  * `deimos/models/decoherence/plot_atmo_decoherence.py` - plot decoherence in atmospheric neutrinos (2D oscillogram)

* Lorentz Invariance Violation (phenomenology paper on the way)
  * `deimos/models/liv/plot_sme_isotropic.py` - plot modified oscillations due to isotropic Standard Model Extension coefficients
  * `deimos/models/liv/plot_sme_sidereal.py` - plot modified oscillations due to non-isotropic Standard Model Extension coefficients, such as sidereal modulations
  * `deimos/models/liv/paper_plots/...` - plotting scripts used for paper phenomenology paper
  * `deimos/models/liv/reproduce_external_plots.py` - reproduce plots from other litterature 


