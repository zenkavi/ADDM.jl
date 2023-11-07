module ADDM

# Don't think we should expose these functions directly to the scope
# Not exporting them as below would require more explicit calling 
# e.g. ADDM.define_model(...) or ADDM.grid_search(...)
# export define_model, simulate_data, grid_search

using Base.Threads
using BenchmarkTools
using CSV
using DataFrames
using Distributions
using LinearAlgebra
using Plots
using Random
using Statistics
using StatsBase

# If you want functions exposed to the global scope when usinging the package
# through `using ADDM` then you would add `export ...` statements here

include("define_model.jl")
include("fixation_data.jl")
include("simulate_data.jl")
include("compute_likelihood.jl")
include("util.jl")
# include("grid_search.jl")

end