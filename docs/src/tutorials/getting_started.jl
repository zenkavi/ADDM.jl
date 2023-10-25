# # Getting started with ADDM.jl

# ## Load package
using ADDM

# ## Parameter recovery on simulated data

# ### Define model
MyModel = ADDM.define_model(d = 0.007, σ = 0.03, θ = .6, barrier = 1, decay = 0, nonDecisionTime = 100, bias = 0.0)

# ### Define stimuli

# Option 1: Read in from CSV
MyStims = CSV.File(..., delim=",")

# Option 2: Create random stimuli
MyStims = ...

# ### Simulate data
SimData = ADDM.simulate_data(aDDM = MyModel, stimuli = MyStims, simulator_fn = ...)

# ### Recover parameters 

# Option 1: Grid Search

# Option 1.1: Read in grid
ParGrid = CSV.File(..., delim=",")
OptimPars = ADDM.grid_search(aDDM = MyModel, data = SimData, grid = ParGrid, returnAll = false)

# Option 1.2: Define grid 
ParGrid = ...
OptimPars = ADDM.grid_search(aDDM = MyModel, data = SimData, grid = ParGrid, returnAll = false)

# Option 2: Narrow in from starting points
OptimPars = ADDM.grid_search(aDDM = MyModel, data = SimData, StartPoints = ..., Tolerance = ...)

# ## Parameter recovery on empirical data

# ### Read in empirical data
SubjectData = CSV.File(..., delim=",")

# Grid Search
ParGrid = CSV.File(..., delim=",")
OptimPars, LogLikelihoods = ADDM.grid_search(aDDM = MyModel, data = SimData, grid = ParGrid, returnAll = true)

# ## Visualizations

# ### Marginal posteriors
ADDM.plot_marginal_posteriors(LogLikelihoods = ...)

# ### True vs. simulated data
BestModel = ADDM.define_model(d = OptimPars.d, σ = OptimPars.σ, θ = OptimPars.θ, barrier = 1, decay = 0, nonDecisionTime = 100, bias = 0.0)
SimData = ADDM.simulate_data(aDDM = BestModel, stimuli = SubjectData)

ADDM.plot_true_vs_sim(true_data = SubjectData, sim_data = SimData)
