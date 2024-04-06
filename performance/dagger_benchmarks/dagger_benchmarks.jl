using Distributed

@everywhere using ADDM
@everywhere using ArgParse
@everywhere using Base.Threads
@everywhere using BenchmarkTools
@everywhere using CSV
@everywhere using Dagger
@everywhere using DataFrames
@everywhere using Dates
@everywhere using FLoops

#########################
# Initialize arguments
#########################

@everywhere function parse_commandline()
  s = ArgParseSettings()

  @add_arg_table! s begin
      "dp"
          help = "first positional argument; data path"
          arg_type = String
          required = true
      "gsf"
          help = "second positional argument; grid search function"
          arg_type = String
          required = true
      "te"
          help = "fourth positional argument; trials executor"
          arg_type = String
          required = true
  end

  parsed_args = parse_args(s)
  dp = parsed_args["dp"]
  gsf = parsed_args["gsf"]
  te = parsed_args["te"]

  return dp, gsf, gse, te 
end

println("Parsing arguments...")
flush(stdout)
dp, grid_search_fn, grid_search_exec, trials_exec = parse_commandline()


println("Defining functions...")
flush(stdout)


#########################
# Define functions
#########################

#########################
# compute_trials_nll versions
#########################

@everywhere function compute_trials_nll_threads(model::ADDM.aDDM, data, likelihood_fn, likelihood_args = (timeStep = 10.0, approxStateStep = 0.1); 
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

@everywhere function compute_trials_nll_floop(model::ADDM.aDDM, data, likelihood_fn, likelihood_args = (timeStep = 10.0, approxStateStep = 0.1); 
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

@everywhere function compute_trials_nll_serial(model::ADDM.aDDM, data, likelihood_fn, likelihood_args = (timeStep = 10.0, approxStateStep = 0.1); 
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
  else
    return negative_log_likelihood
  end
end

#########################
# grid_search versions
#########################

@everywhere function grid_search_thread_thread()

@everywhere function grid_search_thread_floop()

@everywhere function grid_search_thread_dagger()

@everywhere function grid_search_thread_serial()

@everywhere function grid_search_floop_thread()

@everywhere function grid_search_floop_floop()

@everywhere function grid_search_floop_dagger()

@everywhere function grid_search_floop_serial()

@everywhere function grid_search_dagger_thread()

@everywhere function grid_search_dagger_floop()

@everywhere function grid_search_dagger_dagger()

@everywhere function grid_search_dagger_serial()

@everywhere function grid_search_serial_thread()

@everywhere function grid_search_serial_floop()

@everywhere function grid_search_serial_dagger()
    

#########################
# Setup
#########################

# Read in data
# dp = "/Users/zenkavi/Documents/RangelLab/aDDM-Toolbox/ADDM.jl/data/"
println("Reading in data...")
flush(stdout)
data = ADDM.load_data_from_csv(dp*"sim_data_beh.csv", dp*"sim_data_fix.csv");
data = data["1"]; # Sim data is saved with parcode "1"

# Read in parameter space
# fn = dp*"/sim_data_grid_tst.csv"; #15000 combinations
# fn = dp*"/sim_data_grid.csv";
fn = dp*"/sim_data_grid2.csv"; #8000 combinations
tmp = DataFrame(CSV.File(fn, delim=","));
param_grid = NamedTuple.(eachrow(tmp));

my_likelihood_args = (timeStep = 10.0, approxStateStep = 0.05);
fixed_params = Dict(:η=>0.0, :barrier=>1, :decay=>0, :nonDecisionTime=>100, :bias=>0.0);

#########################
# Select benchmark function
#########################

println("Selecting benchmark function...")
flush(stdout)

...

#########################
# Run Benchmark
#########################

println("Starting benchmarking...")
flush(stdout)
output, b_time, b_mem = BenchmarkTools.@btimed f()

#########################
# Save outputs
#########################

println("Done benchmarking! Starting output processing...")
flush(stdout)
base_path = "outputs/grid_search_" * grid_search_fn * '_' * grid_search_exec * '_' * trials_exec * '_' * string(Dates.today()) * "_"

println("Done benchmarking at " * string(now()))
println("Starting output processing...")
flush(stdout)
base_path = "outputs/small_stepsize_" * grid_search_exec * '_' * string(now()) * "_"

b_time_df = DataFrame(:grid_search_fn => grid_search_fn, :grid_search_exec => grid_search_exec, :trials_exec => trials_exec, :b_time => b_time, :b_mem => b_mem)
b_time_df[!, :nthreads] .= nthreads()
b_time_path = base_path * "b_time.csv"
CSV.write(b_time_path, b_time_df)

best_pars_path = base_path * "best_pars.csv"
CSV.write(best_pars_path, DataFrame(output[:best_pars]))

trial_posteriors_df = DataFrame()
for (k,v) in output[:trial_posteriors]
  cur_df = DataFrame(Symbol(i) => j for (i, j) in pairs(v))

  rename!(cur_df, :first => :trial_num, :second => :posterior)

  for (a, b) in pairs(k)
    cur_df[!, a] .= b
  end

  cur_df[!, :trial_num] = [parse(Int, (String(i))) for i in cur_df[!,:trial_num]]

  sort!(cur_df, :trial_num)

  # trial_posteriors_df = vcat(trial_posteriors_df, cur_df, cols=:union)
  append!(trial_posteriors_df, cur_df, cols=:union)
end

trial_posteriors_path = base_path * "trial_posteriors.csv"
CSV.write(trial_posteriors_path, trial_posteriors_df)
println("Done!")
flush(stdout)

#########################
# Usage
#########################

# julia --project=../ --threads 4 grid_search_benchmarks.jl /Users/zenkavi/Documents/RangelLab/aDDM-Toolbox/ADDM.jl/data/ floop2 thread thread 
# julia --project=../ --threads 8 grid_search_benchmarks.jl /central/groups/rnl/zenkavi/ADDM.jl/data/ floop2 thread thread 