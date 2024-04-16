@everywhere function grid_search_floop_thread(data, param_grid, likelihood_fn = nothing, 
  fixed_params = Dict(:θ=>1.0, :η=>0.0, :barrier=>1, :decay=>0, :nonDecisionTime=>0, :bias=>0.0); 
  likelihood_args = (timeStep = 10.0, approxStateStep = 0.1), 
  return_grid_nlls = false,
  return_model_posteriors = false,
  return_trial_posteriors = false,
  model_priors = nothing,
  likelihood_fn_module = Main,
  sequential_model = false,
  save_intermediate = true)

  # Indexed with model param information instead of param_grid rows using NamedTuple keys.
  # Defined with a specific length for performance.
  n = length(param_grid)
  all_nll = Ref(Dict(zip(param_grid, zeros(n))))

  # compute_trials_nll returns a dict indexed by trial numbers 
  # so trial_likelihoods are initialized with keys as parameter combinations and values of empty dictionaries
  n_trials = length(data)
  trial_likelihoods = Ref(Dict(k => Dict(zip(1:n_trials, zeros(n_trials))) for k in param_grid))
  
  #### START OF PARALLELIZABLE PROCESSES

  @floop ThreadedEx() for cur_grid_params in param_grid

  ## ADD WARNING ON IF STEPSIZE RATIO IS GOOD FOR THE NOISE LEVEL BEING TESTED
    if save_intermediate
      println(cur_grid_params)
      flush(stdout)
    end

    cur_model, cur_likelihood_fn = setup_fit_for_params(fixed_params, likelihood_fn, cur_grid_params, likelihood_fn_module)

    if return_model_posteriors
    # Trial likelihoods will be a dict indexed by trial numbers
      all_nll[][cur_grid_params], trial_likelihoods[][cur_grid_params] = compute_trials_nll_threads(cur_model, data, cur_likelihood_fn, likelihood_args; 
                  return_trial_likelihoods = true,  sequential_model = sequential_model)

      # THIS CURRENTLY WON'T WORK FOR MODELS WITH DIFFERENT PARAMETERS
      if save_intermediate
        save_intermediate_likelihoods(trial_likelihoods[][cur_grid_params], cur_grid_params)
      end
      
    else
      all_nll[][cur_grid_params] = compute_trials_nll_threads(cur_model, data, cur_likelihood_fn, likelihood_args, sequential_model = sequential_model)
    end

  end

  #### END OF PARALLELIZABLE PROCESSES

  # Wrangle likelihood data and extract best pars

  # Extract best fit pars
  best_fit_pars = argmin(all_nll[])
  best_pars = Dict(pairs(best_fit_pars))
  best_pars = merge(best_pars, fixed_params)
  best_pars = ADDM.convert_param_text_to_symbol(best_pars)

  # Begin collecting output
  output = Dict()
  output[:best_pars] = best_pars

  if return_grid_nlls
    # Add param info to all_nll
    all_nll_df = DataFrame()
    for (k, v) in all_nll[]
      row = DataFrame(Dict(pairs(k)))
      row.nll .= v
      # vcat can bind rows with different columns
      all_nll_df = vcat(all_nll_df, row, cols=:union)
    end

    output[:grid_nlls] = all_nll_df
  end

  if return_model_posteriors

    trial_posteriors = get_trial_posteriors(data, param_grid, model_priors, trial_likelihoods[])

    if return_trial_posteriors
      output[:trial_posteriors] = trial_posteriors
    end

    model_posteriors = Dict(k => trial_posteriors[k][n_trials] for k in keys(trial_posteriors))
    output[:model_posteriors] = model_posteriors

    end

  return output

end

@everywhere function grid_search_floop_floop(data, param_grid, likelihood_fn = nothing, 
  fixed_params = Dict(:θ=>1.0, :η=>0.0, :barrier=>1, :decay=>0, :nonDecisionTime=>0, :bias=>0.0); 
  likelihood_args = (timeStep = 10.0, approxStateStep = 0.1), 
  return_grid_nlls = false,
  return_model_posteriors = false,
  return_trial_posteriors = false,
  model_priors = nothing,
  likelihood_fn_module = Main,
  sequential_model = false,
  save_intermediate = true)

  # Indexed with model param information instead of param_grid rows using NamedTuple keys.
  # Defined with a specific length for performance.
  n = length(param_grid)
  all_nll = Ref(Dict(zip(param_grid, zeros(n))))

  # compute_trials_nll returns a dict indexed by trial numbers 
  # so trial_likelihoods are initialized with keys as parameter combinations and values of empty dictionaries
  n_trials = length(data)
  trial_likelihoods = Ref(Dict(k => Dict(zip(1:n_trials, zeros(n_trials))) for k in param_grid))
  
  #### START OF PARALLELIZABLE PROCESSES

  @floop ThreadedEx() for cur_grid_params in param_grid

  ## ADD WARNING ON IF STEPSIZE RATIO IS GOOD FOR THE NOISE LEVEL BEING TESTED

    if save_intermediate
      println(cur_grid_params)
      flush(stdout)
    end

    cur_model, cur_likelihood_fn = setup_fit_for_params(fixed_params, likelihood_fn, cur_grid_params, likelihood_fn_module)

    if return_model_posteriors
    # Trial likelihoods will be a dict indexed by trial numbers
      all_nll[][cur_grid_params], trial_likelihoods[][cur_grid_params] = compute_trials_nll_floop(cur_model, data, cur_likelihood_fn, likelihood_args; 
                  return_trial_likelihoods = true,  sequential_model = sequential_model)

      # THIS CURRENTLY WON'T WORK FOR MODELS WITH DIFFERENT PARAMETERS
      if save_intermediate
        save_intermediate_likelihoods(trial_likelihoods[][cur_grid_params], cur_grid_params)
      end
      
    else
      all_nll[][cur_grid_params] = compute_trials_nll_floop(cur_model, data, cur_likelihood_fn, likelihood_args, sequential_model = sequential_model)
    end

  end

  #### END OF PARALLELIZABLE PROCESSES

  # Wrangle likelihood data and extract best pars

  # Extract best fit pars
  best_fit_pars = argmin(all_nll[])
  best_pars = Dict(pairs(best_fit_pars))
  best_pars = merge(best_pars, fixed_params)
  best_pars = ADDM.convert_param_text_to_symbol(best_pars)

  # Begin collecting output
  output = Dict()
  output[:best_pars] = best_pars

  if return_grid_nlls
    # Add param info to all_nll
    all_nll_df = DataFrame()
    for (k, v) in all_nll[]
      row = DataFrame(Dict(pairs(k)))
      row.nll .= v
      # vcat can bind rows with different columns
      all_nll_df = vcat(all_nll_df, row, cols=:union)
    end

    output[:grid_nlls] = all_nll_df
  end

  if return_model_posteriors

    trial_posteriors = get_trial_posteriors(data, param_grid, model_priors, trial_likelihoods[])

    if return_trial_posteriors
      output[:trial_posteriors] = trial_posteriors
    end

    model_posteriors = Dict(k => trial_posteriors[k][n_trials] for k in keys(trial_posteriors))
    output[:model_posteriors] = model_posteriors

    end

  return output

end

@everywhere function grid_search_floop_serial(data, param_grid, likelihood_fn = nothing, 
  fixed_params = Dict(:θ=>1.0, :η=>0.0, :barrier=>1, :decay=>0, :nonDecisionTime=>0, :bias=>0.0); 
  likelihood_args = (timeStep = 10.0, approxStateStep = 0.1), 
  return_grid_nlls = false,
  return_model_posteriors = false,
  return_trial_posteriors = false,
  model_priors = nothing,
  likelihood_fn_module = Main,
  sequential_model = false,
  save_intermediate = true)

  # Indexed with model param information instead of param_grid rows using NamedTuple keys.
  # Defined with a specific length for performance.
  n = length(param_grid)
  all_nll = Ref(Dict(zip(param_grid, zeros(n))))

  # compute_trials_nll returns a dict indexed by trial numbers 
  # so trial_likelihoods are initialized with keys as parameter combinations and values of empty dictionaries
  n_trials = length(data)
  trial_likelihoods = Ref(Dict(k => Dict(zip(1:n_trials, zeros(n_trials))) for k in param_grid))
  
  #### START OF PARALLELIZABLE PROCESSES

  @floop ThreadedEx() for cur_grid_params in param_grid

  ## ADD WARNING ON IF STEPSIZE RATIO IS GOOD FOR THE NOISE LEVEL BEING TESTED

    if save_intermediate
      println(cur_grid_params)
      flush(stdout)
    end

    cur_model, cur_likelihood_fn = setup_fit_for_params(fixed_params, likelihood_fn, cur_grid_params, likelihood_fn_module)

    if return_model_posteriors
    # Trial likelihoods will be a dict indexed by trial numbers
      all_nll[][cur_grid_params], trial_likelihoods[][cur_grid_params] = compute_trials_nll_serial(cur_model, data, cur_likelihood_fn, likelihood_args; 
                  return_trial_likelihoods = true,  sequential_model = sequential_model)

      # THIS CURRENTLY WON'T WORK FOR MODELS WITH DIFFERENT PARAMETERS
      if save_intermediate
        save_intermediate_likelihoods(trial_likelihoods[][cur_grid_params], cur_grid_params)
      end
      
    else
      all_nll[][cur_grid_params] = compute_trials_nll_serial(cur_model, data, cur_likelihood_fn, likelihood_args, sequential_model = sequential_model)
    end

  end

  #### END OF PARALLELIZABLE PROCESSES

  # Wrangle likelihood data and extract best pars

  # Extract best fit pars
  best_fit_pars = argmin(all_nll[])
  best_pars = Dict(pairs(best_fit_pars))
  best_pars = merge(best_pars, fixed_params)
  best_pars = ADDM.convert_param_text_to_symbol(best_pars)

  # Begin collecting output
  output = Dict()
  output[:best_pars] = best_pars

  if return_grid_nlls
    # Add param info to all_nll
    all_nll_df = DataFrame()
    for (k, v) in all_nll[]
      row = DataFrame(Dict(pairs(k)))
      row.nll .= v
      # vcat can bind rows with different columns
      all_nll_df = vcat(all_nll_df, row, cols=:union)
    end

    output[:grid_nlls] = all_nll_df
  end

  if return_model_posteriors

    trial_posteriors = get_trial_posteriors(data, param_grid, model_priors, trial_likelihoods[])

    if return_trial_posteriors
      output[:trial_posteriors] = trial_posteriors
    end

    model_posteriors = Dict(k => trial_posteriors[k][n_trials] for k in keys(trial_posteriors))
    output[:model_posteriors] = model_posteriors

    end

  return output

end