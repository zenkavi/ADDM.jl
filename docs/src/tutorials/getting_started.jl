# # Getting started with ADDM.jl

# ## Load package
using ADDM

# ## Parameter recovery on simulated data

# ### Define model
MyModel = ADDM.define_model(d = 0.007, σ = 0.03, θ = .6, barrier = 1, decay = 0, nonDecisionTime = 100, bias = 0.0)

# ### Define stimuli

# This should be of type NamedTuple
# Required field names (case sensitive): `valueLeft` and `valueRight` 

# Option 1: Read in from CSV
# Note that the modules must be loaded beforehand
# These are dependencies for the ADDM module *but* the precompiled module gives access to 
# these dependencies only in the scope of ADDM.
# `ADDM.load_data_from_csv` that requires both of these packages would still work but
# the code below would not without importing these modules to the current interactive scope

using CSV
using DataFrames

fn = "./data/stims.csv"
tmp = DataFrame(CSV.File(fn, delim=","))
MyStims = (valueLeft = tmp.valueLeft, valueRight = tmp.valueRight)

# Option 2: Create random stimuli
# Random.seed!(38535)
# MyStims = (valueLeft = randn(1000), valueRight = randn(1000))

# ### Define fixationData
# This should be of type `FixationData` 
# If `fixDistType` is not `simple` it must also have fixations for the same value difference values in the stimuli

data = load_data_from_csv("./data/expdata.csv", "./data/fixations.csv")
fixationData = process_fixations(data, fixDistType="fixations")

# ### Simulate data
# Defining only required args without defaults
MyArgs = (fixationData = fixationData)

# Note that these are *positional* arguments for code efficiency
SimData = ADDM.simulate_data(MyModel, MyStims, ADDM.aDDM_simulate_trial, MyArgs)

# ### Recover parameters 

# Option 1: Grid Search

# Option 1.1: Read in grid
fn = ...
ParGrid = DataFrame(CSV.File(fn, delim=","))
OptimPars = ADDM.grid_search(aDDM = MyModel, data = SimData, grid = ParGrid, returnAll = false)

# Option 1.2: Define grid 
ParGrid = ...
OptimPars = ADDM.grid_search(aDDM = MyModel, data = SimData, grid = ParGrid, returnAll = false)

# Option 2: Narrow in from starting points
OptimPars = ADDM.grid_search(aDDM = MyModel, data = SimData, StartPoints = ..., Tolerance = ...)

# ## Parameter recovery on empirical data

# ### Read in empirical data
fn = "./data/subject_data.csv"
SubjectData = DataFrame(CSV.File(fn, delim=","))

# Grid Search
fn = "./data/parameter_grid.csv"
ParGrid = DataFrame(CSV.File(fn, delim=","))
OptimPars, LogLikelihoods = ADDM.grid_search(aDDM = MyModel, data = SimData, grid = ParGrid, returnAll = true)

# ## Visualizations

# ### Marginal posteriors
ADDM.plot_marginal_posteriors(LogLikelihoods = ...)

# ### True vs. simulated data
BestModel = ADDM.define_model(d = OptimPars.d, σ = OptimPars.σ, θ = OptimPars.θ, barrier = 1, decay = 0, nonDecisionTime = 100, bias = 0.0)
SimData = ADDM.simulate_data(aDDM = BestModel, stimuli = SubjectData)

ADDM.plot_true_vs_sim(true_data = SubjectData, sim_data = SimData)
