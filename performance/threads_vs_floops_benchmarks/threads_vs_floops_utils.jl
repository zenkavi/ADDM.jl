function setup_fit_for_params(fixed_params, likelihood_fn, cur_grid_params, likelihood_fn_module=Main)
  
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

function get_trial_posteriors(data, param_grid, model_priors, trial_likelihoods) 
  # Process trial likelihoods to compute model posteriors for each parameter combination
  nTrials = length(data)
  nModels = length(param_grid)

  # Define posteriors as a dictionary with models as keys and Dicts with trial numbers keys as values
  trial_posteriors = Dict(k => Dict(zip(1:nTrials, zeros(nTrials))) for k in param_grid)

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

function save_intermediate_likelihoods(trial_likelihoods_for_grid_params, cur_grid_params)
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
  int_save_path = "../outputs/threads_vs_floops_" * grid_search_fn * '_' * compute_trials_fn * '_'
  trial_likelihoods_path = int_save_path * "trial_likelihoods_int_save.csv"
  CSV.write(trial_likelihoods_path, cur_df, writeheader = !isfile(trial_likelihoods_path), append = true)
end
