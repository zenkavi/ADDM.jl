# Defining custom models

Though the toolbox comes with the standard attentional DDM that allows for multiplicative and additive discounting of unattended items, users might also conceive of other generative processes (within the sequantial sampling to a bound framework) that give rise to observed choices and response times.  

In this tutorial we lay out the framework for how to incorporate such models within our toolbox to take advantage of Julia's processing speed.  

Broadly, this involves defining three parts: 

1. trial simulator describing how the new parameter changes the data generating process resulting in a choice and response time
    - this is then fed into `ADDM.simulate_data` along with the model object and stimuli to generate choice and response times.
2. model object with new parameter
    - this is only a container of key-value pairs of parameter names and values used as a convenient wrapper to feed into the simulator and likelihood computer.
3. trial likelihood function computing the probability of the observing a given choice and response time for a combinaton of parameter specified in the model object
    - this is then fed into `ADDM.grid_search` along with the data you want to compute the likelihoods for and the parameter search space.

Let's begin with importing the packages we'll use in this tutorial.

```@repl 5
using ADDM, CSV, DataFrames, DataFramesMeta, Dates, Distributions, LinearAlgebra, StatsPlots
```

## Define simulator

The built-in model has a `decay` parameter for a linear decay of the `barrier`. Let's build a model with an exponential decay of the barrier such that the barrier at each timestep is defined as `barrier(t) = exp(-λt)`.

Based on the [built-in trial simulators as defined here](https://github.com/aDDM-Toolbox/ADDM.jl/blob/main/src/simulate_data.jl#L39) the trial simulator would look like [this](https://github.com/aDDM-Toolbox/ADDM.jl/blob/main/data/my_trial_simulator.jl). The custom model trial simulator is identical to the built-in simulators except for where the barriers for the accummulation process is defined:

```@repl 5
data_path = joinpath(dirname(dirname(pathof(ADDM))), "data/"); # hide
include(data_path * "my_trial_simulator.jl"); nothing # hide
```

```julia
function my_trial_simulator(;model::ADDM.aDDM, fixationData::ADDM.FixationData, 
                        valueLeft::Number, valueRight::Number, 
                        timeStep::Number=10.0, numFixDists::Int64=3, cutOff::Number=100000)
    
   [...]

    # The values of the barriers can change over time.
    # In this case we include an exponential decay
    # Due to the shape of the exponential decay function the starting point for the decay is exp(0) = 1
    barrierUp = exp.(-model.λ .* (0:cutOff-1))
    barrierDown = -exp.(-model.λ .* (0:cutOff-1))
    
    [...]

    trial = ADDM.Trial(choice = choice, RT = RT, valueLeft = valueLeft, valueRight = valueRight)
    trial.fixItem = fixItem 
    trial.fixTime = fixTime 
    trial.fixRDV = fixRDV
    trial.uninterruptedLastFixTime = uninterruptedLastFixTime
    trial.RDV = tRDV
    return trial
end
```

## Define model container

Then we create a model object of class `aDDM` to store the parameters of our model. There are two ways of doing this. First, we could use the `ADDM.define_model` function. That would like:

```@repl 5
my_model = ADDM.define_model(d = 0.007, σ = 0.03, θ = .6, barrier = 1, nonDecisionTime = 100, bias = 0.0)
```

```@repl 5
my_model.λ = .0005;
```

The `ADDM.define_model` function is limited to the standard parameter names. So the new parameter `λ` is added to the model after its creation. Alternatively, we can create an empty model object and add our parameters individually.

```julia
my_model = ADDM.aDDM()
my_model.d = 0.007
my_model.σ = 0.03
my_model.θ = .6
my_model.η = 0
my_model.barrier = 1
my_model.nonDecisionTime = 100
my_model.bias = 0.0
my_model.λ = .0005
```

### Simulate data

Now that we have defined the generative process (the simulator function) and the model (the parameter values) there are two more necessary inputs for simulating data: stimuli (pairs of values for different objects) and fixation data (location and duration).

#### Define stimuli and fixation distribution

We will use data from Tavares et al. (2017) that [comes with the toolbox](https://github.com/aDDM-Toolbox/ADDM.jl/tree/main/data). Importantly, we will *only* be using the stimuli and fixations from this dataset, *not* the empirical choice and response times. This is ensured by the `stimsOnly` argument of the `ADDM.load_data_from_csv` function. By using the stimuli and the fixations to sample from, we will generate choice and response using our custom simulator function. 

```@repl 5
data = ADDM.load_data_from_csv(data_path * "stimdata.csv", data_path * "fixations.csv"; stimsOnly = true);
```

Extract stimulus values from this dataset and wrangle into the format expected by the simulator function.

```@repl 5
nTrials = 600;
my_stims = (valueLeft = reduce(vcat, [[i.valueLeft for i in data[j]] for j in keys(data)])[1:nTrials], valueRight = reduce(vcat, [[i.valueRight for i in data[j]] for j in keys(data)])[1:nTrials]);
```

Aggregate fixations from all subjects to create fixation duration distributions indexed by value difference and order (1st, 2nd etc.). Since the fixations will be indexed by the value difference, this is extracted from the stimuli and used as an input to the `ADDM.process_fixations` function. The simulator function will sample from this aggregate data.

```@repl 5
vDiffs = sort(unique(reduce(vcat, [[i.valueLeft - i.valueRight for i in data[j]] for j in keys(data)])));

my_fixations = ADDM.process_fixations(data, fixDistType="fixation", valueDiffs = vDiffs);
```

#### Simulate choice and response times

Now that we have read in each of the required inputs we can simulate data with our custom simulator function. To do so we specify the third positional argument to the wrapper function `ADDM.simulate_data` as `my_trial_simulator` so it knows to use this function to generate choice and response times.

```@repl 5
my_args = (timeStep = 10.0, cutOff = 20000, fixationData = my_fixations);

my_sim_data = ADDM.simulate_data(my_model, my_stims, my_trial_simulator, my_args);
```

## Define likelihood function

Based on the [built-in likelihood function as defined here](https://github.com/aDDM-Toolbox/ADDM.jl/blob/main/src/compute_likelihood.jl#L17) the custom likelihood function would look like [this](https://github.com/aDDM-Toolbox/ADDM.jl/blob/main/data/my_likelihood_fn.jl). The custom likelihood function is identical to the built-in function except for where the barriers for the accummulation process is defined:

```@repl 5
include(data_path * "my_likelihood_fn.jl"); nothing # hide
```

```julia
function my_likelihood_fn(;model::ADDM.aDDM, trial::ADDM.Trial, timeStep::Number = 10.0, 
                                   stateStep::Number = 0.01)
    
    [...]

    # The values of the barriers can change over time.
    barrierUp = exp.(-model.λ .* (0:numTimeSteps-1))
    barrierDown = -exp.(-model.λ .* (0:numTimeSteps-1))
    
    [...]
    
    # Compute the likelihood contribution of this trial based on the final
    # choice.
    likelihood = 0
    if trial.choice == -1 # Choice was left.
        if probUpCrossing[end] > 0
            likelihood = probUpCrossing[end]
        end
    elseif trial.choice == 1 # Choice was right.
        if probDownCrossing[end] > 0 
            likelihood = probDownCrossing[end]
        end
    end
    
    return likelihood
end
```

### Recover parameters for simulated data

Now that we have generated some data using known parameters with our custom simulator and defined the likelihood function to compute the likelihood of a choice and response time pair associated with a specific fixation pattern, we can compute the likelihoods for a range of parameter combinations to confirm that the likelihood function correctly recovers the true parameters.

#### Define search grid

```@repl 5
fn = data_path * "custom_model_grid.csv";
tmp = DataFrame(CSV.File(fn, delim=","));
param_grid = NamedTuple.(eachrow(tmp));
```

#### Run grid search on simulated data

```@repl 5
fixed_params = Dict(:η=>0.0, :barrier=>1, :decay=>0, :nonDecisionTime=>100, :bias=>0.0);

my_likelihood_args = (timeStep = 10.0, stateStep = 0.01);

t1 = now();
output = ADDM.grid_search(my_sim_data, param_grid, my_likelihood_fn,
    fixed_params, 
    likelihood_args=my_likelihood_args, 
    return_grid_nlls = true, return_trial_posteriors = true, return_model_posteriors = true,
    return_trial_likelihoods = true);
t2 = now();

mle = output[:mle];
nll_df = output[:grid_nlls];
trial_posteriors = output[:trial_posteriors];
model_posteriors = output[:model_posteriors];
```

The true parameters are `d = 0.007, σ = 0.03, θ = .6, λ = .0005`. Theta and lambda are recovered correctly but not the d and sigma.

```@repl 5
sort!(nll_df, [:nll]);

show(nll_df, allrows = true)
```

How do the model posteriors evolve with each observation?

```@repl 5
# Initialize empty df
trial_posteriors_df = DataFrame();

for i in 1:nTrials

  # Get the posterior for each model after the curent trial
  cur_trial_posteriors = DataFrame(keys(trial_posteriors))
  cur_trial_posteriors[!, :posterior] = [x[i] for x in values(trial_posteriors)]
  
  # Add the trial information
  cur_trial_posteriors[!, :trial_num] .= i

  # Add the current trial posterior to the initialized df
  trial_posteriors_df = vcat(trial_posteriors_df, cur_trial_posteriors, cols=:union)
end;

@transform!(trial_posteriors_df, @byrow :modelnum = string(:d) * string(:sigma) * string(:theta)* string(:lambda))

@df trial_posteriors_df plot(
      :trial_num,
      :posterior,
      group = :modelnum,
      xlabel = "Trial",
      ylabel = "Posterior p",
      legend = false
  )

savefig("plot_5_1.png"); nothing # hide
```
![plot](plot_5_1.png)

As expected from the trajectory of the model posteriors, there is no uncertainty in the parameter posteriors.

```@repl 5
marginal_posteriors = ADDM.marginal_posteriors(model_posteriors, two_d_marginals = true);

ADDM.marginal_posterior_plot(marginal_posteriors)

savefig("plot_5_2.png"); nothing # hide
```
![plot](plot_5_2.png)

How do the parameter posteriors change across trials?

```@repl 5
trial_param_posteriors = DataFrame();

for i in 1:nTrials

  # Get the posterior for each model after the curent trial
  cur_trial_posteriors = Dict(zip(keys(trial_posteriors), [x[i] for x in values(trial_posteriors)]))

  # Use built-in function to marginalize for each parameter
  cur_param_posteriors = ADDM.marginal_posteriors(cur_trial_posteriors)

  # Wrangle the output to be a single df and add trial number info
  for j in 1:length(cur_param_posteriors)
    df = cur_param_posteriors[j][:,:] #assign a copy
    
    df[!, :par_name] .= names(df)[1]
    df[!, :trial_num] .= i
    rename!(df, Symbol(names(df)[1]) => :par_value)

    trial_param_posteriors = vcat(trial_param_posteriors, df, cols=:union)

  end

end;

par_names = unique(trial_param_posteriors[:,:par_name]);

plot_array = Any[];

for cur_par_name in par_names

  plot_df = @rsubset(trial_param_posteriors, :par_name == cur_par_name)

  cur_plot = @df plot_df plot(
      :trial_num,
      :posterior_sum,
      group = :par_value,
      title = cur_par_name,
      xlabel = "Trial",
      ylabel = "Posterior p",
  )

  push!(plot_array, cur_plot)

end;

plot(plot_array...)

savefig("plot_5_3.png"); nothing # hide
```
![plot](plot_5_3.png)

How about the posterior predictive data?

```@repl 5
posteriors_df = DataFrame();

for (k, v) in model_posteriors
  cur_row = DataFrame([k])
  cur_row.posterior = [v]
  posteriors_df = vcat(posteriors_df, cur_row, cols=:union)
end;

bestModelPars = @chain posteriors_df begin
    combine(_) do sdf
        sdf[argmax(sdf.posterior), :]
    end
  end;

est_model = ADDM.define_model(d = bestModelPars.d[1], σ = bestModelPars.sigma[1], θ = bestModelPars.theta[1], barrier = 1, nonDecisionTime = 100, bias = 0.0)

est_model.λ = bestModelPars.lambda[1];

est_sim_data = ADDM.simulate_data(est_model, my_stims, my_trial_simulator, my_args);

## Define the limit for the x-axis based on true data
rts = [i.RT * i.choice for i in my_sim_data]; #left choice rt's are negative
l = abs(minimum(rts)) > abs(maximum(rts)) ? abs(minimum(rts)) : abs(maximum(rts))

## Split the RTs for left and right choice. Left is on the left side of the plot
rts_pos = [i.RT for i in my_sim_data if i.choice > 0];
rts_neg = [i.RT * (-1) for i in my_sim_data if i.choice < 0];

rts_pos_est = [i.RT for i in est_sim_data if i.choice > 0];
rts_neg_est = [i.RT * (-1) for i in est_sim_data if i.choice < 0];

histogram(rts_pos, normalize=true, bins = range(-l, l, length=41), fillcolor = "gray", yaxis = false, grid = false, label = "True data")
density!(rts_pos_est, label = "Best model", linewidth = 3, linecolor = "blue")

histogram!(rts_neg, normalize=true, bins = range(-l, l, length=41), fillcolor = "gray", label = "")
density!(rts_neg_est, linewidth = 3, linecolor = "blue", label = "")

vline!([0], linecolor = "red", label = "")

savefig("plot_5_4.png"); nothing # hide
```
![plot](plot_5_4.png)