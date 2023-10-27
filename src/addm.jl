module ADDM

# Don't think we should expose these functions directly to the scope
# Not exporting them as below would require more explicit calling 
# e.g. ADDM.DefineModel(...) or ADDM.GridSearch(...)
# export define_model, simulate_data, grid_search

using Plots
using Printf
using Random
using Distributions
using Base.Threads
using CSV
using DataFrames
using Statistics
using LinearAlgebra
using ProgressMeter
using BenchmarkTools

# If you want functions exposed to the global scope when usinging the package
# through `using ADDM` then you would add `export ...` statements here

include("define_model.jl")
include("fixation_data.jl")
include("simulate_data.jl")
# include("compute_likelihood.jl")
# include("grid_search.jl")
include("util.jl")

end