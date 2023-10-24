module ADDM

# Don't think we should expose these functions directly to the scope
# Not exporting them as below would require more explicit calling 
# e.g. ADDM.DefineModel(...) or ADDM.GridSearch(...)
# export define_model, simulate_data, grid_search

# Todo: Load packages with `import` instead of `using` for explicit function calls
# Need to identify where they are referred to in each script before changing
import Plots
import Printf
import Random
import Distributions
import Base.Threads
import CSV
import DataFrames
import Statistics
import LinearAlgebra
import ProgressMeter
import BenchmarkTools

# If you want functions exposed to the global scope when importing the package
# through `using ADDM` then you would add `export ...` statements here

include("define_model.jl")
include("simulate_data.jl")
include("compute_likelihood.jl")
include("grid_search.jl")
include("util.jl")

end