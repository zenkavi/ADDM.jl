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
      "tf"
          help = "third positional argument; compute_trials function"
          arg_type = String
          required = true
  end

  parsed_args = parse_args(s)
  dp = parsed_args["dp"]
  gsf = parsed_args["gsf"]
  ctf = parsed_args["tf"]

  return dp, gsf, ctf 
end

println("Parsing arguments...")
flush(stdout)
dp, grid_search_fn, compute_trials_fn = parse_commandline()


println("Defining functions...")
flush(stdout)


#########################
# Define functions
#########################

#########################
# macro within BenchmarkTools for saving outputs
#########################

@eval BenchmarkTools macro btimed(args...)
  _, params = prunekwargs(args...)
  bench, trial, result = gensym(), gensym(), gensym()
  trialmin, trialallocs = gensym(), gensym()
  tune_phase = hasevals(params) ? :() : :($BenchmarkTools.tune!($bench))
  return esc(quote
      local $bench = $BenchmarkTools.@benchmarkable $(args...)
      $BenchmarkTools.warmup($bench)
      $tune_phase
      local $trial, $result = $BenchmarkTools.run_result($bench)
      local $trialmin = $BenchmarkTools.minimum($trial)
      $result, $BenchmarkTools.time($trialmin), $BenchmarkTools.memory($trialmin)
  end)
end

#########################
# compute_trials_nll versions
#########################

@everywhere include("dagger_benchmarks_compute_trials.jl")

#########################
# grid_search versions
#########################

@everywhere include("dagger_benchmarks_grid_search_thread.jl")
@everywhere include("dagger_benchmarks_grid_search_floop.jl")
@everywhere include("dagger_benchmarks_grid_search_dagger.jl")
@everywhere include("dagger_benchmarks_grid_search_serial.jl")
    

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
# fn = dp*"/sim_data_grid_tst.csv";
# fn = dp*"/sim_data_grid.csv"; #15000 combinations
fn = dp*"/sim_data_grid2.csv"; #8000 combinations
tmp = DataFrame(CSV.File(fn, delim=","));
tmp[!,"likelihood_fn"] .= "ADDM.aDDM_get_trial_likelihood";
param_grid = NamedTuple.(eachrow(tmp));

my_likelihood_args = (timeStep = 10.0, approxStateStep = 0.05);
fixed_params = Dict(:η=>0.0, :barrier=>1, :decay=>0, :nonDecisionTime=>100, :bias=>0.0);

#########################
# Select benchmark function
#########################

println("Selecting benchmark function...")
flush(stdout)

# Case 1
if grid_search_fn == "thread"
  if compute_trials_fn == "thread"
    f() = grid_search_thread_thread(data, param_grid, nothing, 
    fixed_params, 
    likelihood_args = my_likelihood_args, 
    return_grid_nlls = false,
    return_model_posteriors = true,
    return_trial_posteriors = true)
  end

  if compute_trials_fn == "floop"
    f() = grid_search_thread_floop(data, param_grid, nothing, 
    fixed_params, 
    likelihood_args = my_likelihood_args, 
    return_grid_nlls = false,
    return_model_posteriors = true,
    return_trial_posteriors = true)
  end

  if compute_trials_fn == "dagger"
    f() = grid_search_thread_dagger(data, param_grid, nothing, 
    fixed_params, 
    likelihood_args = my_likelihood_args, 
    return_grid_nlls = false,
    return_model_posteriors = true,
    return_trial_posteriors = true)
  end

  if compute_trials_fn == "serial"
    f() = grid_search_thread_serial(data, param_grid, nothing, 
    fixed_params, 
    likelihood_args = my_likelihood_args, 
    return_grid_nlls = false,
    return_model_posteriors = true,
    return_trial_posteriors = true)
  end
end

# Case 2
if grid_search_fn == "floop"
  if compute_trials_fn == "thread"
    f() = grid_search_floop_thread(data, param_grid, nothing, 
    fixed_params, 
    likelihood_args = my_likelihood_args, 
    return_grid_nlls = false,
    return_model_posteriors = true,
    return_trial_posteriors = true)
  end

  if compute_trials_fn == "floop"
    f() = grid_search_floop_floop(data, param_grid, nothing, 
    fixed_params, 
    likelihood_args = my_likelihood_args, 
    return_grid_nlls = false,
    return_model_posteriors = true,
    return_trial_posteriors = true)
  end

  if compute_trials_fn == "dagger"
    f() = grid_search_floop_dagger(data, param_grid, nothing, 
    fixed_params, 
    likelihood_args = my_likelihood_args, 
    return_grid_nlls = false,
    return_model_posteriors = true,
    return_trial_posteriors = true)
  end

  if compute_trials_fn == "serial"
    f() = grid_search_floop_serial(data, param_grid, nothing, 
    fixed_params, 
    likelihood_args = my_likelihood_args, 
    return_grid_nlls = false,
    return_model_posteriors = true,
    return_trial_posteriors = true)
  end
end

# Case 3
if grid_search_fn == "dagger"
  if compute_trials_fn == "thread"
    f() = grid_search_dagger_thread(data, param_grid, nothing, 
    fixed_params, 
    likelihood_args = my_likelihood_args, 
    return_grid_nlls = false,
    return_model_posteriors = true,
    return_trial_posteriors = true)
  end

  if compute_trials_fn == "floop"
    f() = grid_search_dagger_floop(data, param_grid, nothing, 
    fixed_params, 
    likelihood_args = my_likelihood_args, 
    return_grid_nlls = false,
    return_model_posteriors = true,
    return_trial_posteriors = true)
  end

  if compute_trials_fn == "dagger"
    f() = grid_search_dagger_dagger(data, param_grid, nothing, 
    fixed_params, 
    likelihood_args = my_likelihood_args, 
    return_grid_nlls = false,
    return_model_posteriors = true,
    return_trial_posteriors = true)
  end

  if compute_trials_fn == "serial"
    f() = grid_search_dagger_serial(data, param_grid, nothing, 
    fixed_params, 
    likelihood_args = my_likelihood_args, 
    return_grid_nlls = false,
    return_model_posteriors = true,
    return_trial_posteriors = true)
  end
end

# Case 4
if grid_search_fn == "serial"
  if compute_trials_fn == "thread"
    f() = grid_search_serial_thread(data, param_grid, nothing, 
    fixed_params, 
    likelihood_args = my_likelihood_args, 
    return_grid_nlls = false,
    return_model_posteriors = true,
    return_trial_posteriors = true)
  end

  if compute_trials_fn == "floop"
    f() = grid_search_serial_floop(data, param_grid, nothing, 
    fixed_params, 
    likelihood_args = my_likelihood_args, 
    return_grid_nlls = false,
    return_model_posteriors = true,
    return_trial_posteriors = true)
  end

  if compute_trials_fn == "dagger"
    f() = grid_search_serial_dagger(data, param_grid, nothing, 
    fixed_params, 
    likelihood_args = my_likelihood_args, 
    return_grid_nlls = false,
    return_model_posteriors = true,
    return_trial_posteriors = true)
  end

  if compute_trials_fn == "serial"
    f() = grid_search_serial_serial(data, param_grid, nothing, 
    fixed_params, 
    likelihood_args = my_likelihood_args, 
    return_grid_nlls = false,
    return_model_posteriors = true,
    return_trial_posteriors = true)
  end
end

#########################
# Run Benchmark
#########################

println("Starting benchmarking...")
flush(stdout)
output, b_time, b_mem = BenchmarkTools.@btimed f()

#########################
# Save outputs
#########################

println("Done benchmarking at " * string(now()))
println("Starting output processing...")
flush(stdout)
base_path = "outputs/dagger_benchmarks_" * grid_search_fn * '_' * compute_trials_fn * '_' * string(now()) * "_"

b_time_df = DataFrame(:grid_search_fn => grid_search_fn, :compute_trials_fn => compute_trials_fn, :b_time => b_time, :b_mem => b_mem)
b_time_df[!, :nthreads] .= nthreads()
b_time_path = base_path * "b_time.csv"
CSV.write(b_time_path, b_time_df)

best_pars_path = base_path * "best_pars.csv"
CSV.write(best_pars_path, DataFrame(output[:best_pars]))

trial_posteriors_df = DataFrame()
for (k,v) in output[:trial_posteriors]
  cur_df = DataFrame(Symbol(i) => j for (i, j) in pairs(v))

  rename!(cur_df, :first => :trial_num, :second => :posterior)

  # Unpack parameter info
  for (a, b) in pairs(k)
    cur_df[!, a] .= b
  end

  # Change type of trial num col to sort by
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

# julia --project=../ --threads 4 dagger_benchmarks.jl /Users/zenkavi/Documents/RangelLab/aDDM-Toolbox/ADDM.jl/data/ thread dagger
