function compute_trials_nll_threads(model::ADDM.aDDM, data, likelihood_fn, likelihood_args = (timeStep = 10.0, approxStateStep = 0.1); 
  return_trial_likelihoods = false, sequential_model = false)

  n_trials = length(data)
  likelihoods = Dict(zip(1:n_trials, zeros(n_trials)))
  data_dict = Dict(zip(1:n_trials, data))

  if sequential_model
    for (trial_number, one_trial) in data_dict 
      cur_lik = likelihood_fn(;model = model, trial = one_trial, likelihood_args...)
      likelihoods[trial_number] = cur_lik
    end
  else
    # Note using threads doesn't guaranteee likelihoods are returned in same order
    # That's why the likelihoods are stored as key, value pairs so they can be rearranged later if needed
    @threads for trial_number in collect(eachindex(data_dict))
      one_trial = data_dict[trial_number]
      cur_lik = likelihood_fn(;model = model, trial = one_trial, likelihood_args...)
      likelihoods[trial_number] = cur_lik
    end
  end

  # If likelihood is 0, set it to 1e-64 to avoid taking the log of a 0.
  likelihoods = Dict(k => max(v, 1e-64) for (k,v) in likelihoods)

  # Sum over all of the negative log likelihoods.
  negative_log_likelihood = -sum(log.(values(likelihoods)))

  if return_trial_likelihoods
    return negative_log_likelihood, likelihoods
  else
    return negative_log_likelihood
  end
end

function compute_trials_nll_floop(model::ADDM.aDDM, data, likelihood_fn, likelihood_args = (timeStep = 10.0, approxStateStep = 0.1); 
  return_trial_likelihoods = false, sequential_model = false, executor = ThreadedEx())

  n_trials = length(data)
  data_dict = Dict(zip(1:length(data), data))
  likelihoods = Ref{Dict{Int64, Float64}}(Dict(zip(1:n_trials, zeros(n_trials))))

  # Redundant but maybe more foolproof in case there is confusion about the executor
  if sequential_model
   cur_exe = SequentialEx()
  else
   cur_exe = executor
  end

  @floop cur_exe for trial_number in collect(eachindex(data_dict))
    likelihoods[][trial_number] = likelihood_fn(;model = model, trial = data_dict[trial_number], likelihood_args...)
  end

  # If likelihood is 0, set it to 1e-64 to avoid taking the log of a 0.
  likelihoods[] = Dict(k => max(v, 1e-64) for (k,v) in likelihoods[])

  # Sum over all of the negative log likelihoods.
  negative_log_likelihood = -sum(log.(values(likelihoods[])))

  if return_trial_likelihoods
    return negative_log_likelihood, likelihoods[]
  else
    return negative_log_likelihood
  end
end

function compute_trials_nll_serial(model::ADDM.aDDM, data, likelihood_fn, likelihood_args = (timeStep = 10.0, approxStateStep = 0.1); 
  return_trial_likelihoods = false, sequential_model = false)

  n_trials = length(data)
  likelihoods = Dict(zip(1:n_trials, zeros(n_trials)))
  data_dict = Dict(zip(1:n_trials, data))

  for (trial_number, one_trial) in data_dict 
    cur_lik = likelihood_fn(;model = model, trial = one_trial, likelihood_args...)
    likelihoods[trial_number] = cur_lik
  end

  # If likelihood is 0, set it to 1e-64 to avoid taking the log of a 0.
  likelihoods = Dict(k => max(v, 1e-64) for (k,v) in likelihoods)

  # Sum over all of the negative log likelihoods.
  negative_log_likelihood = -sum(log.(values(likelihoods)))

  if return_trial_likelihoods
    return negative_log_likelihood, likelihoods
    # return likelihoods[1]
  else
    return negative_log_likelihood
  end
end
