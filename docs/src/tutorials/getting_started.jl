# # Getting started with ADDM.jl

# ## Load package
using ADDM

# ## Parameter recovery on simulated data

# ### Define model
MyModel = ADDM.define_model(d = 0.007, σ = 0.03, θ = .6, barrier = 1, 
                decay = 0, nonDecisionTime = 100, bias = 0.0)

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
# Note that if you're going to create random stimuli you should make sure to have value differences
# that correspond to what you plan to fit in for fixation data

# Random.seed!(38535)
# MyStims = (valueLeft = randn(1000), valueRight = randn(1000))

# ### Define fixationData
# Fixation information that will be fed in to the model for simulations should be of 
# type `FixationData`. This type organizes empirical fixations to distributions

# This organizes both the behavioral and the fixation data as a dictionary of Trial objects indexed by subject
data = ADDM.load_data_from_csv("./data/expdata.csv", "./data/fixations.csv")

# Extract value difference information from the dataset to use in processing the fixations
vDiffs = sort(unique([x.valueLeft - x.valueRight for x in data["1"]]))

# When simulating an aDDM we need to input fixations. 
# But instead of using the fixation data from any given subject we summarize the empricial data 
# from all subjects as distributions from which the model samples from depending on the value 
# difference and the fixation type (1st, 2nd etc.)
MyFixationData = ADDM.process_fixations(data, fixDistType="fixation", valueDiffs = vDiffs)

# ### Simulate data
# Defining additional args
# Note this needs to me a NamedTuple, i.e. must have at least two elements.
# Otherwise it tries to apply `iterate` to the single element which would likely end with a 
# `MethodError`. In this example I specify `timeStep` and `cutoff` in addition to the
# only required argument without a default `fixationData` to avoid this
MyArgs = (timeStep = 10.0, cutOff = 20000, fixationData = MyFixationData)

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
