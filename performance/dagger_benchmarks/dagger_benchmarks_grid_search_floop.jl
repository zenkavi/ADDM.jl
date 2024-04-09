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

    model = ADDM.aDDM()
    for (k,v) in fixed_params setproperty!(model, k, v) end

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

    if return_model_posteriors
    # Trial likelihoods will be a dict indexed by trial numbers
      all_nll[][cur_grid_params], trial_likelihoods[][cur_grid_params] = compute_trials_nll_thread(model, data, likelihood_fn, likelihood_args; 
                  return_trial_likelihoods = true,  sequential_model = sequential_model)

      # THIS CURRENTLY WON'T WORK FOR MODELS WITH DIFFERENT PARAMETERS
      if save_intermediate
        # Process intermediate output
        cur_df = DataFrame(Symbol(i) => j for (i, j) in pairs(trial_likelihoods[][cur_grid_params]))
      
        rename!(cur_df, :first => :trial_num, :second => :likelihood)
      
        # Unpack parameter info
        for (a, b) in pairs(cur_grid_params)
          cur_df[!, a] .= b
        end
      
        # Change type of trial num col to sort by
        cur_df[!, :trial_num] = [parse(Int, (String(i))) for i in cur_df[!,:trial_num]]
      
        sort!(cur_df, :trial_num)
    
        # Save intermediate output
        int_save_path = "outputs/dagger_benchmarks_" * grid_search_fn * '_' * compute_trials_fn * '_'
        trial_likelihoods_path = int_save_path * "trial_likelihoods_int_save.csv"
        CSV.write(trial_likelihoods_path, cur_df, writeheader = !isfile(trial_likelihoods_path), append = true)
      end
      
    else
      all_nll[][cur_grid_params] = compute_trials_nll_thread(model, data, likelihood_fn, likelihood_args, sequential_model = sequential_model)
    end

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

    model = ADDM.aDDM()
    for (k,v) in fixed_params setproperty!(model, k, v) end

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

    if return_model_posteriors
    # Trial likelihoods will be a dict indexed by trial numbers
      all_nll[][cur_grid_params], trial_likelihoods[][cur_grid_params] = compute_trials_nll_floop(model, data, likelihood_fn, likelihood_args; 
                  return_trial_likelihoods = true,  sequential_model = sequential_model)

      # THIS CURRENTLY WON'T WORK FOR MODELS WITH DIFFERENT PARAMETERS
      if save_intermediate
        # Process intermediate output
        cur_df = DataFrame(Symbol(i) => j for (i, j) in pairs(trial_likelihoods[][cur_grid_params]))
      
        rename!(cur_df, :first => :trial_num, :second => :likelihood)
      
        # Unpack parameter info
        for (a, b) in pairs(cur_grid_params)
          cur_df[!, a] .= b
        end
      
        # Change type of trial num col to sort by
        cur_df[!, :trial_num] = [parse(Int, (String(i))) for i in cur_df[!,:trial_num]]
      
        sort!(cur_df, :trial_num)
    
        # Save intermediate output
        int_save_path = "outputs/dagger_benchmarks_" * grid_search_fn * '_' * compute_trials_fn * '_'
        trial_likelihoods_path = int_save_path * "trial_likelihoods_int_save.csv"
        CSV.write(trial_likelihoods_path, cur_df, writeheader = !isfile(trial_likelihoods_path), append = true)
      end

    else
      all_nll[][cur_grid_params] = compute_trials_nll_floop(model, data, likelihood_fn, likelihood_args, sequential_model = sequential_model)
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
      model_priors = Dict(zip(keys(trial_likelihoods[]), repeat([1/nModels], outer = nModels)))
    end

    # Trialwise posterior updating
    for t in 1:nTrials

      # Reset denominator p(data) for each trial
      denom = 0

      # Update denominator summing
      for comb_key in keys(trial_likelihoods[])
        if t == 1
          # Changed the initial posteriors so use model_priors for first trial
          denom += (model_priors[comb_key] * trial_likelihoods[][comb_key][t])
        else
          denom += (trial_posteriors[comb_key][t-1] * trial_likelihoods[][comb_key][t])
        end
      end

      # Calculate the posteriors after this trial.
      for comb_key in keys(trial_likelihoods[])
        if t == 1
          prior = model_priors[comb_key]
        else
          prior = trial_posteriors[comb_key][t-1]
        end
        trial_posteriors[comb_key][t] = (trial_likelihoods[][comb_key][t] * prior / denom)
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

@everywhere function grid_search_floop_dagger(data, param_grid, likelihood_fn = nothing, 
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

    model = ADDM.aDDM()
    for (k,v) in fixed_params setproperty!(model, k, v) end

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

    if return_model_posteriors
    # Trial likelihoods will be a dict indexed by trial numbers
      all_nll[][cur_grid_params], trial_likelihoods[][cur_grid_params] = compute_trials_nll_dagger(model, data, likelihood_fn, likelihood_args; 
                  return_trial_likelihoods = true,  sequential_model = sequential_model)

      # THIS CURRENTLY WON'T WORK FOR MODELS WITH DIFFERENT PARAMETERS
      if save_intermediate
        # Process intermediate output
        cur_df = DataFrame(Symbol(i) => j for (i, j) in pairs(trial_likelihoods[][cur_grid_params]))
      
        rename!(cur_df, :first => :trial_num, :second => :likelihood)
      
        # Unpack parameter info
        for (a, b) in pairs(cur_grid_params)
          cur_df[!, a] .= b
        end
      
        # Change type of trial num col to sort by
        cur_df[!, :trial_num] = [parse(Int, (String(i))) for i in cur_df[!,:trial_num]]
      
        sort!(cur_df, :trial_num)
    
        # Save intermediate output
        int_save_path = "outputs/dagger_benchmarks_" * grid_search_fn * '_' * compute_trials_fn * '_'
        trial_likelihoods_path = int_save_path * "trial_likelihoods_int_save.csv"
        CSV.write(trial_likelihoods_path, cur_df, writeheader = !isfile(trial_likelihoods_path), append = true)
      end
      
    else
      all_nll[][cur_grid_params] = compute_trials_nll_dagger(model, data, likelihood_fn, likelihood_args, sequential_model = sequential_model)
    end

  end

@everywhere function grid_search_floop_serial(data, param_grid, likelihood_fn = nothing, 
  fixed_params = Dict(:θ=>1.0, :η=>0.0, :barrier=>1, :decay=>0, :nonDecisionTime=>0, :bias=>0.0); 
  likelihood_args = (timeStep = 10.0, approxStateStep = 0.1), 
  return_grid_nlls = false,
  return_model_posteriors = false,
  return_trial_posteriors = false,
  model_priors = nothing,
  likelihood_fn_module = Main,
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

    model = ADDM.aDDM()
    for (k,v) in fixed_params setproperty!(model, k, v) end

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

    if return_model_posteriors
    # Trial likelihoods will be a dict indexed by trial numbers
      all_nll[][cur_grid_params], trial_likelihoods[][cur_grid_params] = compute_trials_nll_serial(model, data, likelihood_fn, likelihood_args; 
                  return_trial_likelihoods = true,  sequential_model = sequential_model)

      # THIS CURRENTLY WON'T WORK FOR MODELS WITH DIFFERENT PARAMETERS
      if save_intermediate
        # Process intermediate output
        cur_df = DataFrame(Symbol(i) => j for (i, j) in pairs(trial_likelihoods[][cur_grid_params]))
      
        rename!(cur_df, :first => :trial_num, :second => :likelihood)
      
        # Unpack parameter info
        for (a, b) in pairs(cur_grid_params)
          cur_df[!, a] .= b
        end
      
        # Change type of trial num col to sort by
        cur_df[!, :trial_num] = [parse(Int, (String(i))) for i in cur_df[!,:trial_num]]
      
        sort!(cur_df, :trial_num)
    
        # Save intermediate output
        int_save_path = "outputs/dagger_benchmarks_" * grid_search_fn * '_' * compute_trials_fn * '_'
        trial_likelihoods_path = int_save_path * "trial_likelihoods_int_save.csv"
        CSV.write(trial_likelihoods_path, cur_df, writeheader = !isfile(trial_likelihoods_path), append = true)
      end
      
    else
      all_nll[][cur_grid_params] = compute_trials_nll_serial(model, data, likelihood_fn, likelihood_args, sequential_model = sequential_model)
    end

  end
end