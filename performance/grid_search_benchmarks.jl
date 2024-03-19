using ADDM
using Base.Threads
using BenchmarkTools
using CSV
using DataFrames
using FLoops

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
dp = "/Users/zenkavi/Documents/RangelLab/aDDM-Toolbox/ADDM.jl/data/"
krajbich_data = ADDM.load_data_from_csv(dp*"Krajbich2010_behavior.csv", dp*"Krajbich2010_fixations.csv");
data = krajbich_data["18"];

# Read in parameter space
fn = dp*"/Krajbich_grid3.csv";
tmp = DataFrame(CSV.File(fn, delim=","));
param_grid = NamedTuple.(eachrow(tmp));

my_likelihood_args = (timeStep = 10.0, approxStateStep = 0.1);
fixed_params = Dict(:η=>0.0, :barrier=>1, :decay=>0, :nonDecisionTime=>0, :bias=>0.0);

# Make sure functions are working
output = grid_search_thread(data, param_grid, ADDM.aDDM_get_trial_likelihood, 
    fixed_params, 
    likelihood_args = my_likelihood_args, 
    return_grid_nlls = true, return_model_posteriors = true, return_trial_posteriors = true);

output = grid_search_floop(data, param_grid, ADDM.aDDM_get_trial_likelihood, 
    fixed_params, 
    likelihood_args = my_likelihood_args, 
    return_grid_nlls = true, return_model_posteriors = true, return_trial_posteriors = true);

output = grid_search_floop2(data, param_grid, ADDM.aDDM_get_trial_likelihood, 
    fixed_params, 
    likelihood_args = my_likelihood_args, 
    return_grid_nlls = true, return_model_posteriors = true, return_trial_posteriors = true);

# best_pars = output[:best_pars];
# nll_df = output[:grid_nlls];
# trial_posteriors = output[:trial_posteriors];
# model_posteriors = output[:model_posteriors];

#########################
# Run Benchmarks
#########################

## Returning trial likelihoods does not slow things down
b1a = @benchmark grid_search_thread(data, param_grid, ADDM.aDDM_get_trial_likelihood, 
                          fixed_params, 
                          likelihood_args = my_likelihood_args, 
                          return_grid_nlls = true, return_model_posteriors = true, return_trial_posteriors = true, 
                          sequential_model = true, threaded = false);
println("sequential grid_search w sequential compute_trials_nll = $(median(b1.times)/10^6)")

b1b = @benchmark grid_search_thread(data, param_grid, ADDM.aDDM_get_trial_likelihood, 
                          fixed_params, 
                          likelihood_args = my_likelihood_args, 
                          return_grid_nlls = true, return_model_posteriors = true, return_trial_posteriors = true, 
                          sequential_model = false, threaded = false);
println("sequential grid_search w @threads compute_trials_nll = $(median(b2.times)/10^6)")


b1c = @benchmark grid_search_thread(data, param_grid, ADDM.aDDM_get_trial_likelihood, 
                          fixed_params, 
                          likelihood_args = my_likelihood_args, 
                          return_grid_nlls = true, return_model_posteriors = true, return_trial_posteriors = true, 
                          sequential_model = true, threaded = true);
println("@threads grid_search w sequential compute_trials_nll = $(median(b3.times)/10^6)")

b1d = @benchmark grid_search_thread(data, param_grid, ADDM.aDDM_get_trial_likelihood, 
                          fixed_params, 
                          likelihood_args = my_likelihood_args, 
                          return_grid_nlls = true, return_model_posteriors = true, return_trial_posteriors = true, 
                          sequential_model = false, threaded = true);
println("@threads grid_search @threads compute_trials_nll = $(median(b4.times)/10^6)")

# @floop to grid_serch + @threads compute_trials_nll
# Need to fix likelihood_fn box; check if it works when param_grid has likelihood_fn
b2a = @benchmark grid_search_floop(data, param_grid, ADDM.aDDM_get_trial_likelihood, 
                          fixed_params, 
                          likelihood_args = my_likelihood_args, 
                          return_grid_nlls = true, return_model_posteriors = true, return_trial_posteriors = true, 
                          sequential_model = false);
println("@floop to grid_serch + @threads compute_trials_nll = $(median(b5.times)/10^6)")

b2b = @benchmark grid_search_floop(data, param_grid, ADDM.aDDM_get_trial_likelihood, 
                          fixed_params, 
                          likelihood_args = my_likelihood_args, 
                          return_grid_nlls = true, return_model_posteriors = true, return_trial_posteriors = true, 
                          sequential_model = false,
                          grid_exec = SequentialEx());
println("SequentialEx to grid_serch + @threads compute_trials_nll = $(median(b5a.times)/10^6)")

b2c = @benchmark grid_search_floop(data, param_grid, ADDM.aDDM_get_trial_likelihood, 
                          fixed_params, 
                          likelihood_args = my_likelihood_args, 
                          return_grid_nlls = true, return_model_posteriors = true, return_trial_posteriors = true, 
                          sequential_model = false,
                          grid_exec = SequentialEx());
println("SequentialEx to grid_serch + @threads compute_trials_nll = $(median(b5a.times)/10^6)")

# Double floop
# Need to fix likelihood_fn box
b3a = @benchmark grid_search_floop2(data, param_grid, ADDM.aDDM_get_trial_likelihood, 
                          fixed_params, 
                          likelihood_args = my_likelihood_args, 
                          return_grid_nlls = true, return_model_posteriors = true, return_trial_posteriors = true, 
                          sequential_model = false);
println("@floop to grid_serch + @floop compute_trials_nll = $(median(b6.times)/10^6)")

b3b = @benchmark grid_search_floop2(data, param_grid, ADDM.aDDM_get_trial_likelihood, 
                          fixed_params, 
                          likelihood_args = my_likelihood_args, 
                          return_grid_nlls = true, return_model_posteriors = true, return_trial_posteriors = true, 
                          sequential_model = false,
                          grid_exec = SequentialEx());
println("SequentialEx to grid_serch + @floop compute_trials_nll = $(median(b6a.times)/10^6)")

b3c = @benchmark grid_search_floop2(data, param_grid, ADDM.aDDM_get_trial_likelihood, 
                          fixed_params, 
                          likelihood_args = my_likelihood_args, 
                          return_grid_nlls = true, return_model_posteriors = true, return_trial_posteriors = true, 
                          sequential_model = false,
                          trials_exec = SequentialEx());
println("@floop to grid_serch + SequentialEx compute_trials_nll = $(median(b6b.times)/10^6)")

# Usage
# julia --project=../aDDM-Toolbox/ADDM.jl --threads 3 grid_search_benchmarks.jl

# 1000 trials - simulated
# 25 x 25 x 25 = 15625 size param_grid
# Idea: One subject with a lot of data and a thorough grid search
# Jobs

grid_search_thread(..., sequential_model = true, threaded = false)
grid_search_thread(..., sequential_model = false, threaded = false)
grid_search_thread(..., sequential_model = true, threaded = true)
grid_search_thread(..., sequential_model = false, threaded = true)
grid_search_floop(..., sequential_model = false, grid_exec = ThreadedEx()) 
grid_search_floop(..., sequential_model = true, grid_exec = ThreadedEx())
grid_search_floop(..., sequential_model = false, grid_exec = SequentialEx())
grid_search_floop(..., sequential_model = true, grid_exec = SequentialEx())
grid_search_floop2(..., trials_exec = ThreadedEx(), grid_exec = ThreadedEx())
grid_search_floop2(..., trials_exec = SequentialEx(), grid_exec = ThreadedEx())
grid_search_floop2(..., trials_exec = ThreadedEx(), grid_exec = SequentialEx())
grid_search_floop2(..., trials_exec = SequentialEx(), grid_exec = SequentialEx())


# Expected equivalences: (if @floops is not slower than @threads)

grid_search_thread(..., sequential_model = true, threaded = false)
grid_search_floop(..., sequential_model = true, grid_exec = SequentialEx())
grid_search_floop2(..., trials_exec = SequentialEx(), grid_exec = SequentialEx())

grid_search_thread(..., sequential_model = false, threaded = false)
grid_search_floop(..., sequential_model = false, grid_exec = SequentialEx())
grid_search_floop2(..., trials_exec = ThreadedEx(), grid_exec = SequentialEx())

grid_search_thread(..., sequential_model = true, threaded = true)
grid_search_floop(..., sequential_model = true, grid_exec = ThreadedEx())
grid_search_floop2(..., trials_exec = SequentialEx(), grid_exec = ThreadedEx())

grid_search_thread(..., sequential_model = false, threaded = true)
grid_search_floop(..., sequential_model = false, grid_exec = ThreadedEx()) 
grid_search_floop2(..., trials_exec = ThreadedEx(), grid_exec = ThreadedEx())


# Job scripts for each
# Output organization for each - want time and output

