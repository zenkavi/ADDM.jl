
@everywhere function grid_search_dagger_serial(data, param_grid, likelihood_fn = nothing, 
  fixed_params = Dict(:θ=>1.0, :η=>0.0, :barrier=>1, :decay=>0, :nonDecisionTime=>0, :bias=>0.0); 
  likelihood_args = (timeStep = 10.0, approxStateStep = 0.1), 
  return_grid_nlls = false,
  return_model_posteriors = false,
  return_trial_posteriors = false,
  model_priors = nothing,
  likelihood_fn_module = Main,
  sequential_model = false,
  save_intermediate = true)
   
  # n = length(param_grid)
  # all_nll = Dict(zip(param_grid, zeros(n)))
  all_nll = DataFrame()

  # n_trials = length(data)
  # trial_likelihoods = Dict(k => Dict(zip(1:n_trials, zeros(n_trials))) for k in param_grid)
  trial_likelihoods = EmptyDict()
  trial_likelihoods = DataFrame()

  #### START OF PARALLELIZABLE PROCESSES

  @sync for cur_grid_params in param_grid

    println(cur_grid_params)
    flush(stdout)

    cur_model, cur_likelihood_fn = setup_fit_for_params(fixed_params, likelihood_fn, cur_grid_params, likelihood_fn_module)

    # if return_model_posteriors
    # Trial likelihoods will be a dict indexed by trial numbers
      # Dagger.@spawn CAN'T RETURN MULTIPLE OBJECTS?! OR ITERABLES?
      # Maybe related https://github.com/JuliaParallel/Dagger.jl/issues/480
      # cur_nll, cur_trial_likelihoods = Dagger.@spawn compute_trials_nll_serial(cur_model, data, cur_likelihood_fn, likelihood_args; return_trial_likelihoods = true)

      # cur_nll, cur_trial_likelihoods = Dagger.spawn(compute_trials_nll_serial, cur_model, data, cur_likelihood_fn, likelihood_args; return_trial_likelihoods = true)      
      # trial_likelihoods = merge!!(trial_likelihoods, Dict(cur_grid_params => cur_trial_likelihoods))
      
      # # cur_trial_likelihoods = Dagger.@spawn compute_trials_nll_serial(cur_model, data, cur_likelihood_fn, likelihood_args; return_trial_likelihoods = true) #this works if only one likelihood value is returned
      # cur_grid_params_dict = Dict(pairs(cur_grid_params))
      # cur_grid_params_dict[:one_trial_likelihood] = cur_trial_likelihoods #Adding fetch here slowed everything down
      # push!(trial_likelihoods, NamedTuple(cur_grid_params_dict))

      # cur_nll = Dagger.@spawn compute_trials_nll_serial(cur_model, data, cur_likelihood_fn, likelihood_args) #works as below            
      # cur_grid_params_dict = Dict(pairs(cur_grid_params))
      # cur_grid_params_dict[:nll] = cur_nll #Adding fetch here slowed everything down
      # push!(all_nll, NamedTuple(cur_grid_params_dict))


      # THIS CURRENTLY WON'T WORK FOR MODELS WITH DIFFERENT PARAMETERS
      # if save_intermediate
      #   # This needs the first argument to be a dictionary
      #   save_intermediate_likelihoods(cur_trial_likelihoods, cur_grid_params)
      # end
      
    # else
      cur_nll = Dagger.@spawn compute_trials_nll_serial(cur_model, data, cur_likelihood_fn, likelihood_args)
      cur_grid_params_dict = Dict(pairs(cur_grid_params))
      cur_grid_params_dict[:nll] = cur_nll #Adding fetch here slowed everything down
      push!(all_nll, NamedTuple(cur_grid_params_dict))

    # end

  end

  all_nll.nll = Dagger.@spawn fetch.(all_nll.nll)
  
  #### END OF PARALLELIZABLE PROCESSES

  # Wrangle likelihood data and extract best pars

  # Extract best fit pars
  best_fit_pars = all_nll[argmin(all_nll.nll),:]
  best_pars = Dict(pairs(best_fit_pars))
  best_pars = merge(best_pars, fixed_params)
  best_pars = ADDM.convert_param_text_to_symbol(best_pars)

  # Begin collecting output
  output = Dict()
  output[:best_pars] = best_pars

  if return_grid_nlls
    output[:grid_nlls] = all_nll
  end

  # if return_model_posteriors

  #   trial_posteriors = get_trial_posteriors(data, param_grid, model_priors, trial_likelihoods)

  #   if return_trial_posteriors
  #     output[:trial_posteriors] = trial_posteriors
  #   end

  #   model_posteriors = Dict(k => trial_posteriors[k][nTrials] for k in keys(trial_posteriors))
  #   output[:model_posteriors] = model_posteriors

  #   end

  return output

end

@everywhere function setup_fit_for_params(fixed_params, likelihood_fn, cur_grid_params, likelihood_fn_module=Main)
  
  model = ADDM.aDDM()
  for (k,v) in fixed_params setproperty!(model, k, v) end

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

  if !(cur_grid_params isa Dict)
    for (k,v) in pairs(cur_grid_params) setproperty!(model, k, v) end
  else
    for (k,v) in cur_grid_params setproperty!(model, k, v) end
  end

  # Make sure param names are converted to Greek symbols
  model = ADDM.convert_param_text_to_symbol(model)

  return model, likelihood_fn

end

@everywhere function get_trial_posteriors(data, param_grid, model_priors, trial_likelihoods) 
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

  return trial_posteriors
end

@everywhere function save_intermediate_likelihoods(trial_likelihoods_for_grid_params, cur_grid_params)
  # Process intermediate output
  cur_df = DataFrame(Symbol(i) => j for (i, j) in pairs(trial_likelihoods_for_grid_params))
        
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
