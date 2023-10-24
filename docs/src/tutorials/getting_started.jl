# # Getting started with ADDM.jl

# ## Load package
using ADDM

# ## Parameter recovery on simulated data

# ### Define model
MyModel = ADDM.define_model(d = 0.007, σ = 0.03, θ = .6, barrier = 1, decay = 0, nonDecisionTime = 100, bias = 0.0)

# ### Define stimuli

# Option 1: Read in from CSV
MyStims = ...

# Option 2: Create random stimuli
MyStims = ...

# ### Simulate data
SimData = ADDM.simulate_data(aDDM = MyModel, stimuli = MyStims)

# ### Recover parameters 

# Option 1: Grid Search

# Option 1.1: Read in grid

# Option 1.2: Define grid 

# Option 2: Narrow in from starting points

# ## Parameter recovery on empirical data

# ### Read in empirical data

# Grid Search