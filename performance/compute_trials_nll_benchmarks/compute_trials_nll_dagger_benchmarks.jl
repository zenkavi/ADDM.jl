using Distributed
@everywhere using ADDM
@everywhere using Base.Threads
@everywhere using BenchmarkTools
@everywhere using Dagger
@everywhere using DataFrames

#########################
# Define functions
#########################

@everywhere function compute_trials_nll_dagger(model::ADDM.aDDM, data, likelihood_fn, likelihood_args = (timeStep = 10.0, approxStateStep = 0.1); 
  return_trial_likelihoods = false, sequential_model = false)

  n_trials = length(data)
  # likelihoods = Dict(zip(1:n_trials, zeros(n_trials)))
  likelihoods = DataFrame()
  data_dict = Dict(zip(1:n_trials, data))

  if sequential_model
    for (trial_number, one_trial) in data_dict 
      cur_lik = likelihood_fn(;model = model, trial = one_trial, likelihood_args...)
      likelihoods[trial_number] = cur_lik
    end
  else
    # Note using threads doesn't guaranteee likelihoods are returned in same order
    # That's why the likelihoods are stored as key, value pairs so they can be rearranged later if needed
    @sync for trial_number in collect(eachindex(data_dict))
      one_trial = Dagger.@spawn data_dict[trial_number]
      cur_lik = Dagger.@spawn likelihood_fn(;model = model, trial = one_trial, likelihood_args...)
      # cur_lik = Dagger.@spawn likelihood_fn(;model = model, trial = one_trial)
      # likelihoods[trial_number] = fetch.(cur_lik)
      push!(likelihoods, (;trial_number, cur_lik))
    end
    likelihoods.cur_lik = fetch.(likelihoods.cur_lik)
  end

  # If likelihood is 0, set it to 1e-64 to avoid taking the log of a 0.
  likelihoods.cur_lik = [max(v, 1e-64) for v in likelihoods.cur_lik]

  # Sum over all of the negative log likelihoods.
  negative_log_likelihood = -sum(log.(likelihoods.cur_lik))

  if return_trial_likelihoods
    return negative_log_likelihood, likelihoods
  else
    return negative_log_likelihood
  end
end


#########################
# Setup
#########################

# Read in data
dp = "/Users/zenkavi/Documents/RangelLab/aDDM-Toolbox/ADDM.jl/data/"
krajbich_data = ADDM.load_data_from_csv(dp*"Krajbich2010_behavior.csv", dp*"Krajbich2010_fixations.csv");
data = krajbich_data["18"];

# Pass fixed parameters to the model
model = ADDM.aDDM()
cur_params = Dict(:d=>.00085, :sigma=> .055, :θ=>1.0, :η=>0.0, :barrier=>1, :decay=>0, :nonDecisionTime=>0, :bias=>0.0)
for (k,v) in cur_params setproperty!(model, k, v) end
ADDM.convert_param_text_to_symbol(model)

likelihood_fn = ADDM.aDDM_get_trial_likelihood

#########################
# Run Benchmarks
#########################

## Again returning trial likelihoods does not make much of a difference
# b1 = @benchmark compute_trials_nll_threads(model, data, likelihood_fn; return_trial_likelihoods = true)
# println("compute_trials_nll_threads (w likelihoods) = $(median(b1.times)/10^6) ms")

## Can't remember if I compared times of storing likelihoods with trial number info as dict compared to an array
## This version in the package is faster
b2 = @benchmark ADDM.compute_trials_nll(model, data, likelihood_fn; return_trial_likelihoods = true)
println("ADDM.compute_trials_nll = $(median(b2.times)/10^6) ms")

b3 = @benchmark compute_trials_nll_dagger(model, data, likelihood_fn; return_trial_likelihoods = true)
println("compute_trials_nll_dagger = $(median(b3.times)/10^6) ms")

b4 = @benchmark compute_trials_nll_dagger(model, data, likelihood_fn)
println("compute_trials_nll_dagger (w/out likelihoods) = $(median(b4.times)/10^6) ms")

# Usage
# julia --project --threads 4 performance/compute_trials_nll_benchmarks/compute_trials_nll_dagger_benchmarks.jl

# Output
# ADDM.compute_trials_nll = 21.30125 ms
# compute_trials_nll_dagger = 33.856167 ms
# compute_trials_nll_dagger (w/out likelihoods) = 35.927917 ms

# Dagger can't beat base threads at this level. What about at the level of parameter combinations in grid_search?