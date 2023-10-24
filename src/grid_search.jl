"""
    aDDM_grid_search(addm::aDDM, fixationData::FixationData, dList::LinRange{Float64, Int64}, σList::LinRange{Float64, Int64},
                          θList::LinRange{Float64, Int64}, n::Int64; trials::Int64=1000, cutOff::Int64=30000, simData::Bool = false)

Compute the likelihood of either observed or simulated data for all parameter combinations in paramGrid.

# Arguments
- `simData`: boolean specifying whether to simulated data
- `expData`: path to observed data if not simulating
- `fixationData`: path to either observed fixation data if not simulating or output of `get_empirical_distributions` if simulating
- `paramGrid`: grid of parameter combinations for which likelihoods of data will be computed
- `addm`: addm object specifying the true model of simulating data
- `nTrials`: number of trials to simulate for each dataset if simulating data
- `rtCutOff`: max allowed RT in ms if simulating data 

# Returns:
- MLE estimates and likelihoods for 
      
# Todo
- Specify how to read data in when not simulated
- Make parameter grid more general; not limited to only d, sigma and theta but also include bias, barrierDecay

Planned usage:
```
aDDM_grid_search(simData::Bool = false, expData::..., fixationData::FixationData, paramGrid::..., # this is all you need if fitting to true data
                addm::aDDM, nTrials::Int64=1000, rtCutOff::Int64=30000) # additional args if simulating
```
"""
function grid_search(addm::aDDM, fixationData::FixationData, dList::LinRange{Float64, Int64}, σList::LinRange{Float64, Int64},
                          θList::LinRange{Float64, Int64}, n::Int64; trials::Int64=1000, cutOff::Int64=30000, simData::Bool = false)
    
    dMLEList = Vector{Float64}(undef, n)
    σMLEList = Vector{Float64}(undef, n)
    θMLEList = Vector{Float64}(undef, n)

    NNL_List = Vector{}(undef, n)

    # Create an array of tuples for all parameter combinations.
    param_combinations = [(d, σ, θ) for d in dList, σ in σList, θ in θList]
    
    @showprogress for i in 1:n

        if simData
          addmTrials = aDDM_simulate_trial_data_threads(addm, fixationData, trials, cutOff = cutOff)
        else
          addmTrials = ...
        end

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
