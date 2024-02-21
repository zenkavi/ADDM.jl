# Defining custom models

Though the packages comes with the standard attentional DDM that allows for multiplicative and additive discounting of unattended items, users might also conceive of other generative processes (within the sequantial sampling to a bound framework) that give rise to observed choices, response times.  

In this tutorial we lay out the framework for how to incorporate such models within our toolbox to take advantage of Julia's processing speed.  

Broadly, this involves defining three parts: 

1. trial simulator describing how the new parameter changes the data generating process resulting in a choice and response time
    - this is then fed into `ADDM.simulate_data` along with the model object and stimuli to generate choice and response times.
2. model object with new parameter
    - this is only a container of key-value pairs of parameter names and values used a convenient wrapper to feed into the simulator and likelihood computer.
3. trial likelihood calculator computing the probability of the observed choice and response time
    - this is then fed into `ADDM.grid_search` along with the data you want to compute the likelihoods for and the parameter search space.

Let's begin with importing the packages we'll use in this tutorial.

```@repl 4
using ADDM
using CSV
using DataFrames
using Distributions
using LinearAlgebra
using StatsPlots
```

## Define simulator

The built-in model has a `decay` parameter for a linear decay of the `barrier`. Let's build a model with an exponential decay of the barrier such that the barrier at each timestep is defined as `barrier(t) = exp(-λt)`.

Based on the [built-in trial simulators as defined here](https://github.com/aDDM-Toolbox/ADDM.jl/blob/main/src/simulate_data.jl#L39) the trial simulator would look like [this]((https://github.com/aDDM-Toolbox/ADDM.jl/blob/main/docs/src/tutorials/my_trial_simulator.jl)). The custom model trial simulator is identical to the built-in simulators except for where the barriers for the accummulation process is defined:

```@repl 4
include("my_trial_simulator.jl"); nothing # hide
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

Now create a model object of class `aDDM` to store the parameters of our model. There are two ways of doing this. First, we could use the `ADDM.define_model` function. That would like:

```@repl 4
my_model = ADDM.define_model(d = 0.007, σ = 0.03, θ = .6, barrier = 1, nonDecisionTime = 100, bias = 0.0)
```

```@repl 4
my_model.λ = .05
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
my_model.λ = .05
```

### Simulate data

#### Define stimuli and fixation distribution

We will use ...

```
fn = "../../../data/stimdata.csv"
tmp = DataFrame(CSV.File(fn, delim=","))
my_stims = (valueLeft = tmp.item_left, valueRight = tmp.item_right)
```

Create `data` object that contains info on stimuli and fixations.

```
data = ADDM.load_data_from_csv("../../../data/stimdata.csv", "../../../data/fixations.csv"; stimsOnly = true)
vDiffs = sort(unique(my_stims.valueLeft - my_stims.valueRight));
```

Process fixations to become a single pool from which simulations sample from.

```
my_fixations = ADDM.process_fixations(data, fixDistType="fixation", valueDiffs = vDiffs);
```

#### Simulate choice and response times


```@repl 4
my_args = (timeStep = 10.0, cutOff = 20000, fixationData = my_fixations);
```

Simuluate one set of data with the stimuli.

```@repl 4
my_sim_data = ADDM.simulate_data(my_model, my_stims, my_trial_simulator, my_args);
```

```@repl 4
length(my_sim_data)
```

## Define likelihood function

Based on the [built-in likelihood function as defined here](https://github.com/aDDM-Toolbox/ADDM.jl/blob/main/src/compute_likelihood.jl#L17) the custom likelihod function would look like [this](https://github.com/aDDM-Toolbox/ADDM.jl/blob/main/docs/src/tutorials/my_likelihood_fn.jl). The custom likelihood function is identical to the built-in function except for where the barriers for the accummulation process is defined:

```@repl 4
include("my_likelihood_fn.jl"); nothing # hide
```

```julia
function my_likelihood_fn(;model::ADDM.aDDM, trial::ADDM.Trial, timeStep::Number = 10.0, 
                                   approxStateStep::Number = 0.1)
    
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

#### Define search grid

```@repl 4
fn = "../../../data/custom_model_grid.csv";
tmp = DataFrame(CSV.File(fn, delim=","));
param_grid = Dict(pairs(NamedTuple.(eachrow(tmp))))
```

#### Run grid search on simulated data

```@repl 4
fixed_params = Dict(:η=>0.0, :barrier=>1, :decay=>0, :nonDecisionTime=>100, :bias=>0.0)

my_likelihood_args = (timeStep = 10.0, approxStateStep = 0.1)

best_pars, nll_df, trial_posteriors = ADDM.grid_search(my_sim_data, param_grid, my_likelihood_fn,
    fixed_params, 
    likelihood_args=my_likelihood_args, 
    return_model_posteriors = true);

nTrials = length(my_sim_data);
model_posteriors = Dict(zip(keys(trial_posteriors), [x[nTrials] for x in values(trial_posteriors)]));
```

The true parameters are `d = 0.007, σ = 0.03, θ = .6, λ = .05`. Even with smaller state space step size the correct decay is not recovered. Instead, the fast response times are attributed to faster drift rates and larger sigmas.

```@repl 4
sort!(nll_df, [:nll])

show(nll_df, allrows = true)
```

The posteriors have no uncertainty either.

```@repl 4
marginal_posteriors = ADDM.marginal_posteriors(param_grid, model_posteriors, true)

ADDM.margpostplot(marginal_posteriors)

savefig("plot_4_1.png"); nothing # hide
```
![plot](plot_4_1.png)