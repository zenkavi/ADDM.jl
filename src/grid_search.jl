"""
    grid_search(data, param_grid, likelihood_fn, return_grid_likelihoods = false; 
                likelihood_args =  (timeStep = 10.0, approxStateStep = 0.1), return_trial_likelihoods = false)

Compute the likelihood of either observed or simulated data for all parameter combinations in paramGrid.

# Arguments
- `data`: 

# Returns:
- ...

"""
function grid_search(data, likelihood_fn, param_grid, 
                    fixed_params = Dict(:θ=>1.0, :η=>0.0, :barrier=>1, :decay=>0, :nonDecisionTime=>0, :bias=>0.0); 
                    return_grid_likelihoods = false,
                    likelihood_args = (timeStep = 10.0, approxStateStep = 0.1), 
                    return_trial_likelihoods = false)

  n = length(param_grid) # number of parameter combinations specified in param_grid
  all_nll = Vector{}(undef, n)

  # Pass fixed parameters to the model
  # These don't need to be for each combination of the parameter grid
  model = ADDM.aDDM()
  for (k,v) in fixed_params setproperty!(model, k, v) end

  # Problem: how to define rest of the param_grid and iterate over it?
  # What should the structure of param_grid be?

  for (i, cur_grid_params) in enumerate(param_grid)
    
    for (k,v) in cur_grid_params setproperty!(model, k, v) end

    if return_trial_likelihoods
      all_nll[i], trial_likelihoods[i] = ADDM.compute_trials_nll(model, data, likelihood_fn, likelihood_args = likelihood_args; 
                              return_trial_likelihoods = return_trial_likelihoods)
    else
      all_nll[i] = ADDM.compute_trials_nll(model, data, likelihood_fn, likelihood_args = likelihood_args)
    end
  
  end

  # return(all_nlls) 

  # minIdx = argmin(all_nll)
  # best_pars = 
  # return best_pars

end

function grid_search(addm::aDDM, fixationData::FixationData, dList::LinRange{Float64, Int64}, σList::LinRange{Float64, Int64},
                          θList::LinRange{Float64, Int64}, n::Int64; trials::Int64=1000, cutOff::Int64=30000, simData::Bool = false)
    

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
