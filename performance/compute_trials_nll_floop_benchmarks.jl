using ADDM
using Base.Threads
using BenchmarkTools
using FLoops

#########################
# Define functions
#########################

function compute_trials_nll_floop(model::ADDM.aDDM, data, likelihood_fn, likelihood_args = (timeStep = 10.0, approxStateStep = 0.1); 
  return_trial_likelihoods = false, sequential_model = false)

  n_trials = length(data)
  data_dict = Dict(zip(1:length(data), data))
  # likelihoods = Ref(Dict(zip(1:n_trials, zeros(n_trials))))
  likelihoods = Ref{Dict{Int64, Float64}}(Dict(zip(1:n_trials, zeros(n_trials))))

  if sequential_model
    for (trial_number, one_trial) in data_dict 
      cur_lik = likelihood_fn(;model = model, trial = one_trial, likelihood_args...)
      likelihoods[][trial_number] = cur_lik
    end
  else
    @floop for trial_number in collect(eachindex(data_dict))
        likelihoods[][trial_number] = likelihood_fn(;model = model, trial = data_dict[trial_number], likelihood_args...)
    end
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


function compute_trials_nll_floop_exe(model::ADDM.aDDM, data, likelihood_fn, likelihood_args = (timeStep = 10.0, approxStateStep = 0.1); 
  return_trial_likelihoods = false, sequential_model = false, executor = ThreadedEx())

  n_trials = length(data)
  data_dict = Dict(zip(1:length(data), data))
  likelihoods = Ref{Dict{Int64, Float64}}(Dict(zip(1:n_trials, zeros(n_trials))))

  if sequential_model
    # for (trial_number, one_trial) in data_dict 
    #   cur_lik = likelihood_fn(;model = model, trial = one_trial, likelihood_args...)
    #   likelihoods[][trial_number] = cur_lik
    # end
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

b1 = @benchmark compute_trials_nll_floop(model, data, likelihood_fn; return_trial_likelihoods = true)
println("compute_trials_nll_floop (w likelihoods) = $(median(b1.times)/10^6)")

b2 = @benchmark compute_trials_nll_floop_exe(model, data, likelihood_fn; return_trial_likelihoods = true, executor = ThreadedEx())
println("compute_trials_nll_floop (w likelihoods) = $(median(b2.times)/10^6)")

# Don't understand how basesize works. The three tried here did not change times.
b3 = @benchmark compute_trials_nll_floop_exe(model, data, likelihood_fn; return_trial_likelihoods = true, executor = ThreadedEx(basesize = 1))
println("compute_trials_nll_floop (w likelihoods) = $(median(b3.times)/10^6)")

b4 = @benchmark compute_trials_nll_floop_exe(model, data, likelihood_fn; return_trial_likelihoods = true, executor = ThreadedEx(basesize = 2))
println("compute_trials_nll_floop (w likelihoods) = $(median(b4.times)/10^6)")

# As expected this slows down to sequential
b5 = @benchmark compute_trials_nll_floop_exe(model, data, likelihood_fn; return_trial_likelihoods = true, executor = SequentialEx())
println("compute_trials_nll_floop (w likelihoods) = $(median(b5.times)/10^6)")

# As expected this slows down to sequential
b6 = @benchmark compute_trials_nll_floop_exe(model, data, likelihood_fn; return_trial_likelihoods = true, sequential_model = true)
println("compute_trials_nll_floop (w likelihoods) = $(median(b6.times)/10^6)")