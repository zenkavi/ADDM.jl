module ADDM

# Don't think we should expose these functions directly to the scope
# Not exporting them as below would require more explicit calling 
# e.g. ADDM.DefineModel(...) or ADDM.GridSearch(...)
# export define_model, simulate_data, grid_search

# Todo: Load packages with `import` instead of `using` for explicit function calls
# Need to identify where they are referred to in each script before changing
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

include("addm_functions.jl")
include("ddm_functions.jl")
include("util.jl")

end