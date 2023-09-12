"""
#!/usr/bin/env julia
Copyright (C) 2023, California Institute of Technology

This file is part of addm_toolbox.

addm_toolbox is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

addm_toolbox is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with addm_toolbox. If not, see <http://www.gnu.org/licenses/>.

---

Module: addm_grid_search.jl
Author: Lynn Yang, lynnyang@caltech.edu

Testing functions in aDDM Toolbox.
"""

using LinearAlgebra
using ProgressMeter
using BenchmarkTools

include("addm.jl")
include("util.jl")


function aDDM_grid_search(addm::aDDM, fixationData::FixationData, dList::LinRange{Float64, Int64}, σList::LinRange{Float64, Int64},
                          θList::LinRange{Float64, Int64}, n::Int64; trials::Int64=1000, cutOff::Int64=30000)
    """
    """
    dMLEList = Vector{Float64}(undef, n)
    σMLEList = Vector{Float64}(undef, n)
    θMLEList = Vector{Float64}(undef, n)

    NNL_List = Vector{}(undef, n)

    # Create an array of tuples for all parameter combinations.
    param_combinations = [(d, σ, θ) for d in dList, σ in σList, θ in θList]
    
    @showprogress for i in 1:n
        addmTrials = aDDM_simulate_trial_data_threads(addm, fixationData, trials, cutOff=cutOff)
        
        # Vectorized calculation of negative log-likelihood for all parameter combinations
        neg_log_like_array = [aDDM_negative_log_likelihood_threads(addm, addmTrials, d, σ, θ) for (d, σ, θ) in param_combinations]
        
        # Find the index of the minimum negative log-likelihood and obtain the MLE parameters
        minIdx = argmin(neg_log_like_array)
        dMin, σMin, θMin = param_combinations[minIdx]
        
        println("obtained MLE parameters")
        
        # Store results in the preallocated arrays
        dMLEList[i] = dMin
        σMLEList[i] = σMin
        θMLEList[i] = θMin

        #NNL: NNL_List[i] = neg_log_like_array
    end

    return dMLEList, σMLEList, θMLEList #NNL:, NNL_List
end

"""
dLow = parse(Float64, ARGS[1])
dHigh = parse(Float64, ARGS[2])
σLow = parse(Float64, ARGS[3])
σHigh = parse(Float64, ARGS[4])
θLow = parse(Float64, ARGS[5])
θHigh = parse(Float64, ARGS[6])
gridSize = parse(Int64, ARGS[7])
n = parse(Int64, ARGS[8])

# Set default values for trials and cutOff
trials = length(ARGS) >= 9 ? parse(Int64, ARGS[9]) : 1000
cutOff = length(ARGS) >= 10 ? parse(Int64, ARGS[10]) : 30000
"""

println("Enter dLow:")
dLow = parse(Float64, readline())
println("Enter dHigh:")
dHigh = parse(Float64, readline())
println("Enter σLow:")
σLow = parse(Float64, readline())
println("Enter σHigh:")
σHigh = parse(Float64, readline())
println("Enter uLow:")
θLow = parse(Float64, readline())
println("Enter θHigh:")
θHigh = parse(Float64, readline())
println("Enter grid size:")
gridSize = parse(Int64, readline())
println("Enter number of datasets:")
n = parse(Int64, readline())

dTrue = 0.005
σTrue = 0.07
θTrue = 0.3

addm = aDDM(dTrue, σTrue, θTrue)
data = load_data_from_csv("expdata.csv", "fixations.csv", convertItemValues=convert_item_values)
fixationData = get_empirical_distributions(data, fixDistType="simple")

dList = LinRange(dLow, dHigh, gridSize)
σList = LinRange(σLow, σHigh, gridSize)
θList = LinRange(θLow, θHigh, gridSize)

@time begin
    dMLEList, σMLEList, θMLEList = aDDM_grid_search(addm, fixationData, dList, σList, θList, n)
    #NNL: dMLEList, σMLEList, θMLEList, NNL_List = aDDM_grid_search(addm, fixationData, dList, σList, θList, n)
end

println("d: ", dMLEList)
println("σ: ", σMLEList)
println("θ: ", θMLEList)
#NNL: println("NNL: ", NNL_List)