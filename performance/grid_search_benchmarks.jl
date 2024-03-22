using ADDM
using ArgParse
using Base.Threads
using BenchmarkTools
using CSV
using DataFrames
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
      "gse"
          help = "third positional argument; grid search executor"
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
  gse = parsed_args["gse"]
  te = parsed_args["te"]

  return dp, gsf, gse, te 
end

println("Parsing arguments...")
flush(stdout)
dp, grid_search_fn, grid_search_exec, trials_exec = parse_commandline()


println("Defining functions...")
flush(stdout)
#########################
# Define helper
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
# Define functions
#########################

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

# Use sequential_model argument to turn threading on/off for compute_trials_nll

function grid_search_thread(data, param_grid, likelihood_fn = nothing, 
  fixed_params = Dict(:θ=>1.0, :η=>0.0, :barrier=>1, :decay=>0, :nonDecisionTime=>0, :bias=>0.0); 
  likelihood_args = (timeStep = 10.0, approxStateStep = 0.1), 
  return_grid_nlls = false,
  return_model_posteriors = false,
  return_trial_posteriors = false,
  model_priors = nothing,
  likelihood_fn_module = Main,
  sequential_model = false, threaded = true)

  # Indexed with model param information instead of param_grid rows using NamedTuple keys.
  # Defined with a specific length for performance.
  n = length(param_grid)
  all_nll = Dict(zip(param_grid, zeros(n)))

  # compute_trials_nll returns a dict indexed by trial numbers 
  # so trial_likelihoods are initialized with keys as parameter combinations and values of empty dictionaries
  n_trials = length(data)
  trial_likelihoods = Dict(k => Dict(zip(1:n_trials, zeros(n_trials))) for k in param_grid)
  # this is a bit more flexible but might not be as performant
  # Dict(k => Dict{Int64, Float64}() for k in param_grid) 

  # Pass fixed parameters to the model
  # These don't need to be updated for each combination of the parameter grid
  # model = ADDM.aDDM()
  # for (k,v) in fixed_params setproperty!(model, k, v) end


  #### START OF PARALLELIZABLE PROCESSES

  if threaded
    @threads for cur_grid_params in param_grid

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
        all_nll[cur_grid_params], trial_likelihoods[cur_grid_params] = ADDM.compute_trials_nll(model, data, likelihood_fn, likelihood_args; 
                    return_trial_likelihoods = true,  sequential_model = sequential_model)
        # println(trial_likelihoods)
        # flush(stdout)
      else
        all_nll[cur_grid_params] = ADDM.compute_trials_nll(model, data, likelihood_fn, likelihood_args, sequential_model = sequential_model)
      end
  
    end
  else #SEQUENTIAL

    for cur_grid_params in param_grid

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
        all_nll[cur_grid_params], trial_likelihoods[cur_grid_params] = ADDM.compute_trials_nll(model, data, likelihood_fn, likelihood_args; 
                    return_trial_likelihoods = true,  sequential_model = sequential_model)
      # println(trial_likelihoods)
      # flush(stdout)
      else
        all_nll[cur_grid_params] = ADDM.compute_trials_nll(model, data, likelihood_fn, likelihood_args, sequential_model = sequential_model)
      end
  
    end
  end
  
  #### END OF PARALLELIZABLE PROCESSES

  # Wrangle likelihood data and extract best pars

  # Extract best fit pars
  best_fit_pars = argmin(all_nll)
  best_pars = Dict(pairs(best_fit_pars))
  best_pars = merge(best_pars, fixed_params)
  best_pars = ADDM.convert_param_text_to_symbol(best_pars)

  # Begin collecting output
  output = Dict()
  output[:best_pars] = best_pars

  if return_grid_nlls
    # Add param info to all_nll
    all_nll_df = DataFrame()
    for (k, v) in all_nll
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

    if return_trial_posteriors
      output[:trial_posteriors] = trial_posteriors
    end

    model_posteriors = Dict(k => trial_posteriors[k][nTrials] for k in keys(trial_posteriors))
    output[:model_posteriors] = model_posteriors

    end

  return output

end

function grid_search_floop(data, param_grid, likelihood_fn = nothing, 
  fixed_params = Dict(:θ=>1.0, :η=>0.0, :barrier=>1, :decay=>0, :nonDecisionTime=>0, :bias=>0.0); 
  likelihood_args = (timeStep = 10.0, approxStateStep = 0.1), 
  return_grid_nlls = false,
  return_model_posteriors = false,
  return_trial_posteriors = false,
  model_priors = nothing,
  likelihood_fn_module = Main,
  sequential_model = false,
  grid_exec = ThreadedEx())

  # Indexed with model param information instead of param_grid rows using NamedTuple keys.
  # Defined with a specific length for performance.
  n = length(param_grid)
  all_nll = Ref(Dict(zip(param_grid, zeros(n))))

  # compute_trials_nll returns a dict indexed by trial numbers 
  # so trial_likelihoods are initialized with keys as parameter combinations and values of empty dictionaries
  n_trials = length(data)
  trial_likelihoods = Ref(Dict(k => Dict(zip(1:n_trials, zeros(n_trials))) for k in param_grid))
  
  #### START OF PARALLELIZABLE PROCESSES

  @floop grid_exec for cur_grid_params in param_grid

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
      all_nll[][cur_grid_params], trial_likelihoods[][cur_grid_params] = ADDM.compute_trials_nll(model, data, likelihood_fn, likelihood_args; 
                  return_trial_likelihoods = true,  sequential_model = sequential_model)
      # println(trial_likelihoods[])
      # flush(stdout)
    else
      all_nll[][cur_grid_params] = ADDM.compute_trials_nll(model, data, likelihood_fn, likelihood_args, sequential_model = sequential_model)
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


function grid_search_floop2(data, param_grid, likelihood_fn = nothing, 
  fixed_params = Dict(:θ=>1.0, :η=>0.0, :barrier=>1, :decay=>0, :nonDecisionTime=>0, :bias=>0.0); 
  likelihood_args = (timeStep = 10.0, approxStateStep = 0.1), 
  return_grid_nlls = false,
  return_model_posteriors = false,
  return_trial_posteriors = false,
  model_priors = nothing,
  likelihood_fn_module = Main,
  sequential_model = false,
  grid_exec = ThreadedEx(),
  trials_exec = ThreadedEx())

  # Indexed with model param information instead of param_grid rows using NamedTuple keys.
  # Defined with a specific length for performance.
  n = length(param_grid)
  all_nll = Ref(Dict(zip(param_grid, zeros(n))))

  # compute_trials_nll returns a dict indexed by trial numbers 
  # so trial_likelihoods are initialized with keys as parameter combinations and values of empty dictionaries
  n_trials = length(data)
  trial_likelihoods = Ref(Dict(k => Dict(zip(1:n_trials, zeros(n_trials))) for k in param_grid))
  
  #### START OF PARALLELIZABLE PROCESSES

  @floop grid_exec for cur_grid_params in param_grid

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
                  return_trial_likelihoods = true,  sequential_model = sequential_model, executor = trials_exec)
      # println(trial_likelihoods[])
      # flush(stdout)
    else
      all_nll[][cur_grid_params] = compute_trials_nll_floop(model, data, likelihood_fn, likelihood_args, sequential_model = sequential_model, executor = trials_exec)
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
# fn = dp*"/sim_data_grid.csv";
fn = dp*"/sim_data_grid2.csv";
tmp = DataFrame(CSV.File(fn, delim=","));
param_grid = NamedTuple.(eachrow(tmp));

my_likelihood_args = (timeStep = 10.0, approxStateStep = 0.1);
fixed_params = Dict(:η=>0.0, :barrier=>1, :decay=>0, :nonDecisionTime=>100, :bias=>0.0);

#########################
# Select benchmark function
#########################

println("Selecting benchmark function...")
flush(stdout)
if grid_search_fn == "thread"
  if grid_search_exec == "thread"
    if trials_exec == "thread"
      f() = grid_search_thread(data, param_grid, ADDM.aDDM_get_trial_likelihood, 
                                fixed_params, 
                                likelihood_args = my_likelihood_args, 
                                return_grid_nlls = true, return_model_posteriors = true, return_trial_posteriors = true, 
                                sequential_model = false, threaded = true)
    else
      f() = grid_search_thread(data, param_grid, ADDM.aDDM_get_trial_likelihood, 
                                fixed_params, 
                                likelihood_args = my_likelihood_args, 
                                return_grid_nlls = true, return_model_posteriors = true, return_trial_posteriors = true, 
                                sequential_model = true, threaded = true)
    end
  else
    if trials_exec == "thread"
      f() = grid_search_thread(data, param_grid, ADDM.aDDM_get_trial_likelihood, 
                                fixed_params, 
                                likelihood_args = my_likelihood_args, 
                                return_grid_nlls = true, return_model_posteriors = true, return_trial_posteriors = true, 
                                sequential_model = false, threaded = false)
    else
      f() = grid_search_thread(data, param_grid, ADDM.aDDM_get_trial_likelihood, 
                                fixed_params, 
                                likelihood_args = my_likelihood_args, 
                                return_grid_nlls = true, return_model_posteriors = true, return_trial_posteriors = true, 
                                sequential_model = true, threaded = false)
    end
  end
end

if grid_search_fn == "floop"
  if grid_search_exec == "thread"
    if trials_exec == "thread"
      f() = grid_search_floop(data, param_grid, ADDM.aDDM_get_trial_likelihood, 
                                fixed_params, 
                                likelihood_args = my_likelihood_args, 
                                return_grid_nlls = true, return_model_posteriors = true, return_trial_posteriors = true, 
                                sequential_model = false, grid_exec = ThreadedEx())
    else
      f() = grid_search_floop(data, param_grid, ADDM.aDDM_get_trial_likelihood, 
                                fixed_params, 
                                likelihood_args = my_likelihood_args, 
                                return_grid_nlls = true, return_model_posteriors = true, return_trial_posteriors = true, 
                                sequential_model = true, grid_exec = ThreadedEx())
    end
  else
    if trials_exec == "thread"
      f() = grid_search_floop(data, param_grid, ADDM.aDDM_get_trial_likelihood, 
                                fixed_params, 
                                likelihood_args = my_likelihood_args, 
                                return_grid_nlls = true, return_model_posteriors = true, return_trial_posteriors = true, 
                                sequential_model = false, grid_exec = SequentialEx())
    else
      f() = grid_search_floop(data, param_grid, ADDM.aDDM_get_trial_likelihood, 
                                fixed_params, 
                                likelihood_args = my_likelihood_args, 
                                return_grid_nlls = true, return_model_posteriors = true, return_trial_posteriors = true, 
                                sequential_model = true, grid_exec = SequentialEx())
    end
  end
end

if grid_search_fn == "floop2"
  if grid_search_exec == "thread"
    if trials_exec == "thread"
      f() = grid_search_floop2(data, param_grid, ADDM.aDDM_get_trial_likelihood, 
                                fixed_params, 
                                likelihood_args = my_likelihood_args, 
                                return_grid_nlls = true, return_model_posteriors = true, return_trial_posteriors = true, 
                                trials_exec = ThreadedEx(), grid_exec = ThreadedEx())
    else
      f() = grid_search_floop2(data, param_grid, ADDM.aDDM_get_trial_likelihood, 
                                fixed_params, 
                                likelihood_args = my_likelihood_args, 
                                return_grid_nlls = true, return_model_posteriors = true, return_trial_posteriors = true, 
                                trials_exec = SequentialEx(), grid_exec = ThreadedEx())
    end
  else
    if trials_exec == "thread"
      f() = grid_search_floop2(data, param_grid, ADDM.aDDM_get_trial_likelihood, 
                                fixed_params, 
                                likelihood_args = my_likelihood_args, 
                                return_grid_nlls = true, return_model_posteriors = true, return_trial_posteriors = true, 
                                trials_exec = ThreadedEx(), grid_exec = SequentialEx())
    else
      f() = grid_search_floop2(data, param_grid, ADDM.aDDM_get_trial_likelihood, 
                                fixed_params, 
                                likelihood_args = my_likelihood_args, 
                                return_grid_nlls = true, return_model_posteriors = true, return_trial_posteriors = true, 
                                trials_exec = SequentialEx(), grid_exec = SequentialEx())
    end
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

println("Done benchmarking! Starting output processing...")
flush(stdout)
base_path = "outputs/grid_search_" * grid_search_fn * '_' * grid_search_exec * '_' * trials_exec * '_'

b_time_df = DataFrame(:grid_search_fn => grid_search_fn, :grid_search_exec => grid_search_exec, :trials_exec => trials_exec, :b_time => b_time, :b_mem => b_mem)
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

# 1000 trials - simulated
# 25 x 25 x 25 = 15625 size param_grid
# Idea: One subject with a lot of data and a thorough grid search
# Jobs
# Expected equivalences: (if @floops is not slower than @threads)

# grid_search_thread(..., sequential_model = true, threaded = false)
# grid_search_floop(..., sequential_model = true, grid_exec = SequentialEx())
# grid_search_floop2(..., trials_exec = SequentialEx(), grid_exec = SequentialEx())

# grid_search_thread(..., sequential_model = false, threaded = false)
# grid_search_floop(..., sequential_model = false, grid_exec = SequentialEx())
# grid_search_floop2(..., trials_exec = ThreadedEx(), grid_exec = SequentialEx())

# grid_search_thread(..., sequential_model = true, threaded = true)
# grid_search_floop(..., sequential_model = true, grid_exec = ThreadedEx())
# grid_search_floop2(..., trials_exec = SequentialEx(), grid_exec = ThreadedEx())

# grid_search_thread(..., sequential_model = false, threaded = true)
# grid_search_floop(..., sequential_model = false, grid_exec = ThreadedEx()) 
# grid_search_floop2(..., trials_exec = ThreadedEx(), grid_exec = ThreadedEx())

# Todo: 
# Progress/thread tracking
# Intermediate saving
# Boxed likelihood_fn and filling of output within parallel execution