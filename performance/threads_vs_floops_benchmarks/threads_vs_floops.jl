using ADDM
using ArgParse
using Base.Threads
using BenchmarkTools
using CSV
using DataFrames
using Dates
using FLoops

#########################
# Initialize arguments
#########################

function parse_commandline()
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
      "ctf"
          help = "third positional argument; compute_trials function"
          arg_type = String
          required = true
      "n_bench"
          help = "fourth positional argument; number of times to run the function"
          arg_type = Int
          required = true
      "test"
          help = "fifth positional argument; boolean for whether to use small test data"
          arg_type = String
          required = false
          default = "false"
  end

  parsed_args = parse_args(s)
  dp = parsed_args["dp"]
  gsf = parsed_args["gsf"]
  ctf = parsed_args["ctf"]
  n_bench = parsed_args["n_bench"]
  test = parsed_args["test"]

  return dp, gsf, ctf, n_bench, test 
end

println("Parsing arguments...")
flush(stdout)
dp, grid_search_fn, compute_trials_fn, n_bench, test = parse_commandline()
if test == "true"
  test = true
else
  test = false
end
# dp = "/Users/zenkavi/Documents/RangelLab/aDDM-Toolbox/ADDM.jl/data/"
# grid_search_fn = "thread"
# compute_trials_fn = "floop"
# test = true

println("Defining functions...")
flush(stdout)

#########################
# Define functions
#########################

#########################
# compute_trials_nll versions
#########################

# cd("./performance/threads_vs_floops_benchmarks")
include("threads_vs_floops_compute_trials.jl")

#########################
# grid_search versions
#########################

include("threads_vs_floops_utils.jl")
include("threads_vs_floops_grid_search_thread.jl")
include("threads_vs_floops_grid_search_floop.jl")
include("threads_vs_floops_grid_search_serial.jl")

#########################
# Output saving function
#########################

function save_output(output, grid_search_fn, compute_trials_fn)

  base_path = "../outputs/threads_vs_floops_" * grid_search_fn * '_' * compute_trials_fn * '_' * string(now()) * "_"

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
end


#########################
# Read in data and parameter space
#########################

# Read in data
println("Reading in data...")
flush(stdout)

if test
  data = ADDM.load_data_from_csv(dp*"sim_data_beh_tst.csv", dp*"sim_data_fix_tst.csv");
  fn = dp*"/sim_data_grid_tst.csv";
else
  data = ADDM.load_data_from_csv(dp*"sim_data_beh.csv", dp*"sim_data_fix.csv");
  fn = dp*"/sim_data_grid3.csv"; # 1980 combinations
end

data = data["1"]; # Sim data is saved with parcode "1"
# Read in parameter space
tmp = DataFrame(CSV.File(fn, delim=","));
tmp[!,"likelihood_fn"] .= "ADDM.aDDM_get_trial_likelihood";
param_grid = NamedTuple.(eachrow(tmp));

my_likelihood_args = (timeStep = 10.0, approxStateStep = 0.01);
fixed_params = Dict(:η=>0.0, :barrier=>1, :decay=>0, :nonDecisionTime=>100, :bias=>0.0);

#########################
# Select function to benchmark
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
    return_trial_posteriors = true,
    save_intermediate = false)
  end

  if compute_trials_fn == "floop"
    f() = grid_search_thread_floop(data, param_grid, nothing, 
    fixed_params, 
    likelihood_args = my_likelihood_args, 
    return_grid_nlls = false,
    return_model_posteriors = true,
    return_trial_posteriors = true,
    save_intermediate = false)
  end

  if compute_trials_fn == "serial"
    f() = grid_search_thread_serial(data, param_grid, nothing, 
    fixed_params, 
    likelihood_args = my_likelihood_args, 
    return_grid_nlls = false,
    return_model_posteriors = true,
    return_trial_posteriors = true,
    save_intermediate = false)
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
    return_trial_posteriors = true,
    save_intermediate = false)
  end

  if compute_trials_fn == "floop"
    f() = grid_search_floop_floop(data, param_grid, nothing, 
    fixed_params, 
    likelihood_args = my_likelihood_args, 
    return_grid_nlls = false,
    return_model_posteriors = true,
    return_trial_posteriors = true,
    save_intermediate = false)
  end

  if compute_trials_fn == "serial"
    f() = grid_search_floop_serial(data, param_grid, nothing, 
    fixed_params, 
    likelihood_args = my_likelihood_args, 
    return_grid_nlls = false,
    return_model_posteriors = true,
    return_trial_posteriors = true,
    save_intermediate = false)
  end
end

# Case 3
if grid_search_fn == "serial"
  if compute_trials_fn == "thread"
    f() = grid_search_serial_thread(data, param_grid, nothing, 
    fixed_params, 
    likelihood_args = my_likelihood_args, 
    return_grid_nlls = false,
    return_model_posteriors = true,
    return_trial_posteriors = true,
    save_intermediate = false)
  end

  if compute_trials_fn == "floop"
    f() = grid_search_serial_floop(data, param_grid, nothing, 
    fixed_params, 
    likelihood_args = my_likelihood_args, 
    return_grid_nlls = false,
    return_model_posteriors = true,
    return_trial_posteriors = true,
    save_intermediate = false)
  end

  if compute_trials_fn == "serial"
    f() = grid_search_serial_serial(data, param_grid, nothing, 
    fixed_params, 
    likelihood_args = my_likelihood_args, 
    return_grid_nlls = false,
    return_model_posteriors = true,
    return_trial_posteriors = true,
    save_intermediate = false)
  end
end

#########################
# Run Benchmark
#########################

println("Starting benchmarking at "* string(now()))
flush(stdout)

b_time_df = DataFrame();
for i in 1:n_bench
  t1 = now();
  output = f();
  t2 = now();
  b_time = t2-t1;

  if i==1
    println("Saving output of first iteration...")
    flush(stdout)
    save_output(output, grid_search_fn, compute_trials_fn)
  end

  println("Iteration "* string(i) * " time: " * string(b_time))
  flush(stdout)
  push!(b_time_df, (;:grid_search_fn => grid_search_fn, :compute_trials_fn => compute_trials_fn, :b_time => b_time, :nthreads => nthreads()))
end

base_path = "../outputs/threads_vs_floops_" * grid_search_fn * '_' * compute_trials_fn * '_' * string(now()) * "_"

b_time_path = base_path * "b_time.csv"
CSV.write(b_time_path, b_time_df)

println("Done benchmarking at " * string(now()))
flush(stdout)

#########################
# Usage
#########################

# Local
# julia --project=../../ --threads 4 threads_vs_floops.jl /Users/zenkavi/Documents/RangelLab/aDDM-Toolbox/ADDM.jl/data/ thread thread 10 true

# Remote
# julia --project=$JPROJ_PATH --threads $NUM_THREADS threads_vs_floops.jl $DATA_PATH thread thread 4 true