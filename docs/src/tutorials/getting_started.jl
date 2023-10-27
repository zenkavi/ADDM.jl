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
fn = "./data/stims.csv"
tmp = DataFrame(CSV.File(fn, delim=","))
MyStims = (valueLeft = tmp.valueLeft, valueRight = tmp.valueRight)

# Option 2: Create random stimuli
# Random.seed!(38535)
# MyStims = (valueLeft = randn(1000), valueRight = randn(1000))

# ### Define fixationData
# This should be of type `FixationData`
# Required keys for aDDM_simulate_trial: latencies, probFixLeftFirst, fixDistType, fixations, transitions
# these are also the outputs of get_empirical_distributions 
# Previously the usage had been
# ```
# data = load_data_from_csv("expdata.csv", "fixations.csv", convertItemValues=convert_item_values)
# fixationData = get_empirical_distributions(data, fixDistType="simple")
# ```
fixationData = FixationData(probFixLeftFirst, latencies, transitions, fixations; fixDistType="fixation")

# ### Simulate data
# Defining only required args without defaults
MyArgs = (fixationData = fixationData)

# Note that these are positional arguments for code efficiency
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
