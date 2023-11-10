# Getting started with ADDM.jl

## Load package

```julia
using ADDM
```

## Parameter recovery on simulated data

### Define model

```julia
MyModel = ADDM.define_model(d = 0.007, σ = 0.03, θ = .6, barrier = 1, 
                decay = 0, nonDecisionTime = 100, bias = 0.0)
```

### Define stimuli

This should be of type `NamedTuple` with required field names (case sensitive): `valueLeft` and `valueRight` 

**Option 1: Read in from CSV**  

Note that the `CSV` and `DataFrames` modules must be loaded beforehand. These are dependencies for the ADDM module *but* the precompiled module gives access to these dependencies only in the scope of ADDM. In other words, `ADDM.load_data_from_csv` that requires both of these packages would still work but the code below would not without importing these modules to the current interactive scope

```julia
using CSV
using DataFrames

fn = "./data/stims.csv"
tmp = DataFrame(CSV.File(fn, delim=","))
MyStims = (valueLeft = tmp.valueLeft, valueRight = tmp.valueRight)
```

**Option 2: Create random stimuli**

Note that if you're going to create random stimuli you should make sure to have value differences that correspond to what you plan to fit in for fixation data

```julia
Random.seed!(38535)
MyStims = (valueLeft = randn(1000), valueRight = randn(1000))
```

### Define fixationData

Fixation information that will be fed in to the model for simulations should be of type `FixationData`. This type organizes empirical fixations to distributions conditional on fixation type (first, second etc.) and value difference.

This organizes both the behavioral and the fixation data as a dictionary of Trial objects indexed by subject

```julia
data = ADDM.load_data_from_csv("./data/expdata.csv", "./data/fixations.csv")
```

Extract value difference information from the dataset to use in processing the fixations

```julia
vDiffs = sort(unique([x.valueLeft - x.valueRight for x in data["1"]]))
```

When simulating an aDDM we need to input fixations. But instead of using the fixation data from any given subject we summarize the empricial data from all subjects as distributions from which the model samples from depending on the value difference and the fixation type (1st, 2nd etc.).

```julia
MyFixationData = ADDM.process_fixations(data, fixDistType="fixation", valueDiffs = vDiffs)
```

### Simulate data

Defining additional arguments for aDDM trial simulator (e.g. fixation data). Note this needs to me a NamedTuple, i.e. must have at least two elements Otherwise it tries to apply `iterate` to the single element which would likely end with a  `MethodError`. In this example I specify `timeStep` and `cutoff` in addition to the  only required argument without a default `fixationData` to avoid this.

```julia
MyArgs = (timeStep = 10.0, cutOff = 20000, fixationData = MyFixationData)
```

Note that these are *positional* arguments for code efficiency

```julia
SimData = ADDM.simulate_data(MyModel, MyStims, ADDM.aDDM_simulate_trial, MyArgs)
```

Data can also be simulated from probability distributions of fixation data.

```julia
MyFixationDist, MyTimeBins = ADDM.convert_to_fixationDist(MyFixationData)

MyBlankFixationData = ADDM.FixationData(MyFixationData.probFixLeftFirst, MyFixationData.latencies, MyFixationData.transitions, Dict())

MyArgs = (fixationData = MyBlankFixationData, fixationDist = MyFixationDist, timeBins = MyTimeBins)

SimData = ADDM.simulate_data(MyModel, MyStims, ADDM.aDDM_simulate_trial, MyArgs)
```

### Recover parameters 

**Option 1: Grid Search**

```julia
ffn = "./data/addm_grid.csv"
tmp = DataFrame(CSV.File(fn, delim=","))
param_grid = Dict(pairs(NamedTuple.(eachrow(tmp))))

best_pars, all_nll_df = ADDM.grid_search(SimData, ADDM.aDDM_get_trial_likelihood, param_grid, Dict(:η=>0.0, :barrier=>1, :decay=>0, :nonDecisionTime=>100, :bias=>0.0))
```

**Option 2: Narrow in from starting points**

TBD

## Parameter recovery on empirical data

TBD

## Visualizations

### Marginal posteriors

TBD

### True vs. simulated data

TBD