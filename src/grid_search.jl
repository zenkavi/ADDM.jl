"""
    grid_search(data, param_grid, likelihood_fn = nothing, 
                    fixed_params = Dict(:θ=>1.0, :η=>0.0, :barrier=>1, :decay=>0, :nonDecisionTime=>0, :bias=>0.0); 
                    return_grid_nlls = true,
                    likelihood_args = (timeStep = 10.0, approxStateStep = 0.1), 
                    return_model_posteriors = false,
                    model_priors = nothing,
                    likelihood_fn_module = Main,
                    sequential_model = false)
                    fixed_params = Dict(:θ=>1.0, :η=>0.0, :barrier=>1, :decay=>0, :nonDecisionTime=>0, :bias=>0.0); 
                    return_grid_nlls = true,
                    likelihood_args = (timeStep = 10.0, approxStateStep = 0.1), 
                    return_model_posteriors = false,
                    model_priors = nothing,
                    likelihood_fn_module = Main,
                    sequential_model = false)
                likelihood_args =  (timeStep = 10.0, approxStateStep = 0.1), return_trial_likelihoods = false)

Compute the likelihood of either observed or simulated data for all parameter combinations in `param_grid`.

# Arguments

## Required 

- `data`: Data for which the sum of negative log likelihoods will be computed for each trial.
  Should be a vector of `ADDM.Trial` objects.
- `likelihood_fn`: Name of likelihood function to be used to compute likelihoods. 
  The toolbox has `ADDM.aDDM_get_trial_likelihood` and `ADDM.DDM_get_trial_likelihood` defined.
  If comparing different generative processes then leave at default value of `nothing`
  and make sure to define a `likelihood_fn` in the `param_grid`.
- `param_grid`: Parameter combinations for which the sum of nll's for the `data` is 
  computed. Vector of NamedTuples.
- `fixed_params`: Default `Dict(:θ=>1.0, :η=>0.0, :barrier=>1, :decay=>0, :nonDecisionTime=>0, :bias=>0.0)`.
  Parameters required by the `likelihood_fn` that are not specified to vary across likelihood 
  computations.

## Optional 

- `likelihood_args`: Default `(timeStep = 10.0, approxStateStep = 0.1)`. Additional 
  arguments to be passed onto `likelihood_fn`. 
- `return_grid_nlls`: Default `false`. If true, will return the sum of nll's for 
  each parameter combination in the grid search.
- `return_model_posteriors`: Default `false`. If true, will return the posterior probability 
  for each parameter combination in `param_grid`.
- `return_trial_posteriors`: Default `false`. If true, will return the posterior probability 
  for each parameter combination in `param_grid` after each trial in `data`.  
- `model_priors`: priors for each model probability if not assummed to be uniform. Should be
  specified as a `Dict` with values of probabilities matching the keys for the correct model
  specified in `param_grid`.
- `likelihood_fn_module`: If an alternative likelihood fn is 
- `sequential_model`: Boolean to specify if the model requires all data concurrently (e.g. RL-DDM). If `true` model cannot be multithreaded

# Returns
- `best_pars`: `Dict` containing the parameter combination with the lowest nll.
- `grid_nlls`: `DataFrame` containing sum of nll's for each parameter combination.
- `trial_posteriors`: Posterior probability for each parameter combination after each trial.
- `model_posteriors`: Posterior probability for each parameter combination after all trials.

"""
function grid_search(data, param_grid, likelihood_fn = nothing, 
                    fixed_params = Dict(:θ=>1.0, :η=>0.0, :barrier=>1, :decay=>0, :nonDecisionTime=>0, :bias=>0.0); 
                    likelihood_args = (timeStep = 10.0, approxStateStep = 0.1), 
                    return_grid_nlls = false,
                    return_model_posteriors = false,
                    return_trial_posteriors = false,
                    model_priors = nothing,
                    likelihood_fn_module = Main,
                    sequential_model = false)

  # Indexed with model param information instead of param_grid rows using NamedTuple keys.
  # Defined with a specific length for performance.
  n = length(param_grid)
  all_nll = Dict(zip(param_grid, zeros(n)))
  
  # compute_trials_nll returns a dict indexed by trial numbers 
  # so trial_likelihoods are initialized with keys as parameter combinations and values of empty dictionaries
  n_trials = length(data)
  trial_likelihoods = Dict(k => Dict(zip(1:n_trials, zeros(n_trials))) for k in param_grid)
  # this is a bit more flexible but might not be as performant
  # Dict(k => Dict{Int64, Float64}() for k in param_grid) 
  
  # Pass fixed parameters to the model
  # These don't need to be updated for each combination of the parameter grid
  model = ADDM.aDDM()
  for (k,v) in fixed_params setproperty!(model, k, v) end

  # What should the structure of param_grid be?
  # It used to be a dictionary of NamedTuples 
  # because when all_nll was defined as a Vector above
  # then this loop used/relied on the keys of param_grid being numbers
  # e.g. 1 => (d = 0.003, sigma = 0.01, theta = 0.2)
  # because the keys index k is used to index the all_nll vector as well
  # But this always led to making sure that outputs were aligned with the parameters
  # Now param_grid is a Vector of NamedTuples that contain the param combination 
  # These are used as the keys for the output Dicts as well

  # !!!!!!!!! This is what I want to parallelize as MPI jobs on top of the threaded compute_trials_nll
  for cur_grid_params in param_grid
    
    # If likelihood_fn is not defined as argument to the function 
    # it should be defined in the param_grid
    # Extract that info and create variable that contains executable function
    if likelihood_fn === nothing
      if :likelihood_fn in keys(cur_grid_params)
        likelihood_fn_str = cur_grid_params[:likelihood_fn]
        if (occursin(".", likelihood_fn_str))
          space, func = split(likelihood_fn_str, ".")
          likelihood_fn = getfield(getfield(Main, Symbol(space)), Symbol(func))
        else
          likelihood_fn = getfield(likelihood_fn_module, Symbol(likelihood_fn_str))
        end
      else
        println("likelihood_fn not specified or defined in param_grid")
      end
    end

    # Update the model with the current parameter combination
    # By default cur_grid_params are NamedTuples so it needs pairs to map k, v pairs
    if !(cur_grid_params isa Dict)
      for (k,v) in pairs(cur_grid_params) setproperty!(model, k, v) end
    else
      for (k,v) in cur_grid_params setproperty!(model, k, v) end
    end
    
    # Make sure param names are converted to Greek symbols
    model = ADDM.convert_param_text_to_symbol(model)

    # !!!!!!!!! Add condition when sequential_model is false and threading should not be done
    if return_model_posteriors
      # Now trial likelihoods will be a dict indexed by trial numbers
      all_nll[cur_grid_params], trial_likelihoods[cur_grid_params] = ADDM.compute_trials_nll(model, data, likelihood_fn, likelihood_args; 
                              return_trial_likelihoods = true,  sequential_model = false)

    else
      all_nll[cur_grid_params] = ADDM.compute_trials_nll(model, data, likelihood_fn, likelihood_args)
    end
  
  end

  # Wrangle likelihood data and extract best pars

  # Extract best fit pars
  best_fit_pars = argmin(all_nll)
  best_pars = Dict(pairs(best_fit_pars))
  best_pars = merge(best_pars, fixed_params)
  best_pars = ADDM.convert_param_text_to_symbol(best_pars)

  # Begin collecting output
  output = Dict()
  output[:best_pars] = best_pars

  if return_grid_nlls
      # Add param info to all_nll
      all_nll_df = DataFrame()
      for (k, v) in all_nll
        row = DataFrame(Dict(pairs(k)))
        row.nll .= v
        # vcat avoids issues with append! and can bind rows with different columns
        all_nll_df = vcat(all_nll_df, row, cols=:union)
      end

      output[:grid_nlls] = all_nll_df
  end
    
  if return_model_posteriors
      
    # Process trial likelihoods to compute model posteriors for each parameter combination
    nTrials = length(data)
    nModels = length(param_grid)

    # Define posteriors as a dictionary with models as keys and Dicts with trial numbers keys as values
    trial_posteriors = Dict(k => Dict(zip(1:n_trials, zeros(n_trials))) for k in param_grid)

    if isnothing(model_priors)          
      model_priors = Dict(zip(keys(trial_likelihoods), repeat([1/nModels], outer = nModels)))
    end

    # Trialwise posterior updating
    for t in 1:nTrials

      # Reset denominator p(data) for each trial
      denom = 0

      # Update denominator summing
      for comb_key in keys(trial_likelihoods)
        if t == 1
          # Changed the initial posteriors so use model_priors for first trial
          denom += (model_priors[comb_key] * trial_likelihoods[comb_key][t])
        else
          denom += (trial_posteriors[comb_key][t-1] * trial_likelihoods[comb_key][t])
        end
      end

      # Calculate the posteriors after this trial.
      for comb_key in keys(trial_likelihoods)
        if t == 1
          prior = model_priors[comb_key]
        else
          prior = trial_posteriors[comb_key][t-1]
        end
        trial_posteriors[comb_key][t] = (trial_likelihoods[comb_key][t] * prior / denom)
      end
    end

    if return_trial_posteriors
      output[:trial_posteriors] = trial_posteriors
    end

    model_posteriors = Dict(k => trial_posteriors[k][nTrials] for k in keys(trial_posteriors))
    output[:model_posteriors] = model_posteriors

  end

  return output

end