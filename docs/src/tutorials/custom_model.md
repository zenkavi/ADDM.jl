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

## Load package

```julia
using ADDM
using Distributions
```

## Define simulator

The built-in model has a `decay` parameter for a linear decay of the `barrier`. Let's build a model with an exponential decay of the barrier such that the barrier at each timestep is defined as `barrier(t) = exp(-λt)`.

Based on the [built-in trial simulators as defined here](https://github.com/aDDM-Toolbox/ADDM.jl/blob/main/src/simulate_data.jl) the trial simulator would look like the following:

```julia
function my_trial_simulator(;model::ADDM.aDDM, fixationData::ADDM.FixationData, 
                        valueLeft::Number, valueRight::Number, 
                        timeStep::Number=10.0, numFixDists::Int64=3, cutOff::Number=100000)
    
    fixUnfixValueDiffs = Dict(1 => valueLeft - valueRight, 2 => valueRight - valueLeft)
    
    fixItem = Number[]
    fixTime = Number[]
    fixRDV = Number[]

    RDV = model.bias
    trialTime = 0
    choice = 0
    tRDV = Number[RDV]
    RT = 0
    uninterruptedLastFixTime = 0
    ndtTimeSteps = Int64(model.nonDecisionTime ÷ timeStep)

    # The values of the barriers can change over time.
    # In this case we include an exponential decay
    # Due to the shape of the exponential decay function the starting point for the decay is exp(0) = 1
    barrierUp = exp.(-model.λ .* (0:cutOff-1))
    barrierDown = -exp.(-model.λ .* (0:cutOff-1))
    
    # Sample and iterate over the latency for this trial.
    latency = rand(fixationData.latencies)
    remainingNDT = model.nonDecisionTime - latency

    # This will not change anything (i.e. move the RDV) if there is no latency data in the fixations
    for t in 1:Int64(latency ÷ timeStep)
        # Sample the change in RDV from the distribution.
        RDV += rand(Normal(0, model.σ))
        push!(tRDV, RDV)

        # If the RDV hit one of the barriers, the trial is over.
        # No barrier decay before decision-related accummulation
        if abs(RDV) >= model.barrier
            choice = RDV >= 0 ? -1 : 1
            push!(fixRDV, RDV)
            push!(fixItem, 0)
            push!(fixTime, t * timeStep)
            trialTime += t * timeStep
            RT = trialTime
            uninterruptedLastFixTime = latency
            trial = Trial(choice = choice, RT = RT, valueLeft = valueLeft, valueRight = valueRight)
            trial.fixItem = fixItem 
            trial.fixTime = fixTime 
            trial.fixRDV = fixRDV
            trial.uninterruptedLastFixTime = uninterruptedLastFixTime
            trial.RDV = tRDV
            return trial
        end
    end

    # Add latency to this trial's data
    push!(fixRDV, RDV)
    push!(fixItem, 0)
    push!(fixTime, latency - (latency % timeStep))
    trialTime += latency - (latency % timeStep)

    fixNumber = 1
    prevFixItem = -1
    currFixLocation = 0
    decisionReached = false

    # Begin decision related accummulation
    cumTimeStep = 0
    while true
        if currFixLocation == 0
            # This is an item fixation; sample its location.
            if prevFixItem == -1
                # Sample the first item fixation for this trial.
                currFixLocation = rand(Bernoulli(1 - fixationData.probFixLeftFirst)) + 1
            elseif prevFixItem in [1, 2]
                currFixLocation = abs(3 - prevFixItem)
            end
            prevFixItem = currFixLocation

            # Sample the duration of this item fixation.
            valueDiff = fixUnfixValueDiffs[currFixLocation]
            #[1] is here to make sure it's not sampling from 1-element Vector but from the array inside it
            currFixTime = rand(fixationData.fixations[fixNumber][valueDiff][1]) 
            

            if fixNumber < numFixDists
                fixNumber += 1
            end

        else
            # This is a transition.
             currFixLocation = 0
            # Sample the duration of this transition. The fixation data used below does not have transition information so ignoring this.
            # currFixTime = rand(fixationData.transitions)
            currFixTime = 0
        end

        # Iterate over the remaining non-decision time remaining after the latency
        # This can span more than first fixation depending on the first fixation duration
        # That's why it's not conditioned over the fixation number
        if remainingNDT > 0
            for t in 1:Int64(remainingNDT ÷ timeStep)
                # Sample the change in RDV from the distribution.
                RDV += rand(Normal(0, model.σ))
                push!(tRDV, RDV)

                # If the RDV hit one of the barriers, the trial is over.
                # No barrier decay before decision-related accummulation
                if abs(RDV) >= model.barrier
                    choice = RDV >= 0 ? -1 : 1
                    push!(fixRDV, RDV)
                    push!(fixItem, currFixLocation)
                    push!(fixTime, t * timeStep)
                    trialTime += t * timeStep
                    RT = trialTime
                    uninterruptedLastFixTime = currFixTime
                    decisionReached = true
                    break
                end
            end
        end

        # Break out of the while loop if decision reached during NDT
        # The break above only breaks from the NDT for loop
        if decisionReached
            break
        end

        remainingFixTime = max(0, currFixTime - max(0, remainingNDT))
        remainingNDT -= currFixTime

        # Iterate over the duration of the current fixation.
        # Does not move RDV if there is no fixation time left due to NDT
        for t in 1:Int64(remainingFixTime ÷ timeStep)
            # We use a distribution to model changes in RDV
            # stochastically. The mean of the distribution (the change
            # most likely to occur) is calculated from the model
            # parameters and from the values of the two items.
            if currFixLocation == 0
                μ = 0
            elseif currFixLocation == 1
                μ = model.d * ( (valueLeft + model.η) - (model.θ * valueRight))
            elseif currFixLocation == 2
                μ = model.d * ((model.θ * valueLeft) - (valueRight + model.η))
            end

            # Sample the change in RDV from the distribution.
            RDV += rand(Normal(μ, model.σ))
            push!(tRDV, RDV)

            # Increment cumulative timestep to look up the correct barrier value in case there has been a decay
            # Decay in this case only happens during decision-related accummulation (not before)
            # Don't want to use t here because this is reset for each fixation throughout a trial but the barrier is not
            cumTimeStep += 1

            # If the RDV hit one of the barriers, the trial is over.
            # Decision related accummulation here so barrier might have decayed
            if abs(RDV) >= barrierUp[cumTimeStep]
                choice = RDV >= 0 ? -1 : 1
                push!(fixRDV, RDV)
                push!(fixItem, currFixLocation)
                push!(fixTime, t * timeStep)
                trialTime += t * timeStep
                RT = trialTime
                uninterruptedLastFixTime = currFixTime
                decisionReached = true
                break
            end
        end

        # Break out of the while loop if decision reached during NDT
        # The break above only breaks from the curFixTime for loop
        if decisionReached
            break
        end

        # Add fixation to this trial's data.
        push!(fixRDV, RDV)
        push!(fixItem, currFixLocation)
        push!(fixTime, currFixTime - (currFixTime % timeStep))
        trialTime += currFixTime - (currFixTime % timeStep)

    end

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

```julia
my_model = ADDM.define_model(d = 0.007, σ = 0.03, θ = .6, barrier = 1, nonDecisionTime = 100, bias = 0.0)
my_model.λ = .05
```

The `ADDM.define_model` function is limited to the standard parameter names. So the new parameter `λ` is added to the model after its creation. Alternatively, we can create an empty model object and add our parameters individually.

```julia
my_model = ADDM.aDDM()
my_model.d = 0.007
my_model.σ = 0.03
my_model.θ = .6
my_model.barrier = 1
my_model.nonDecisionTime = 100
my_model.bias = 0.0
my_model.λ = .05
```

### Simulate data

#### Define stimuli and fixation distribution

We will use sample empirical data from Krajbich et al. (2010) to create sample stimuli and fixation distributions. Importantly, we will *not* be using the empirical choices and response times but instead simulate our own data given the generative process we defined in our custom model and the parameter values we specify for it (i.e. in this notebook we do not fit this custom model to the empirical data from Krajbich et al.).

```julia
using CSV
using DataFrames

fn = "./data/Krajbich2010_stims.csv"
tmp = DataFrame(CSV.File(fn, delim=","))
my_stims = (valueLeft = tmp.item_left, valueRight = tmp.item_right)
```

```julia
data = ADDM.load_data_from_csv("./data/Krajbich2010_behavior.csv", "./data/Krajbich2010_fixations.csv")
vDiffs = sort(unique(my_stims.valueLeft - my_stims.valueRight))
my_fixations = ADDM.process_fixations(data, fixDistType="fixation", valueDiffs = vDiffs)

# Why is there no second fixation durations for vDiff == 7 when there are later fixations?
my_fixations.fixations[2][7] = my_fixations.fixations[3][7]
```

#### Simulate choice and response times

```julia
my_args = (timeStep = 10.0, cutOff = 20000, fixationData = my_fixations)

my_sim_data = ADDM.simulate_data(my_model, my_stims, my_trial_simulator, my_args)
```

## Define likelihood function

```julia
my_likelihood_fn
```

### Recover parameters for simulated data

#### Define search grid

```julia
fn = "./data/custom_model_grid.csv"
tmp = DataFrame(CSV.File(fn, delim=","))
param_grid = Dict(pairs(NamedTuple.(eachrow(tmp))))
```

#### Run grid search on simulated data

```julia
fixed_params = Dict(:η=>0.0, :barrier=>1, :decay=>0, :nonDecisionTime=>0, :bias=>0.0)

best_pars, nll_df = ADDM.grid_search(my_sim_data, my_likelihood_fn, param_grid, fixed_params)
```