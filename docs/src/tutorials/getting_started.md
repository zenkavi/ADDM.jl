# Getting started with ADDM.jl

## Load package

```@repl 1
using ADDM
```

## Parameter recovery on simulated data

### Define model

```@repl 1
MyModel = ADDM.define_model(d = 0.007, σ = 0.03, θ = .6, barrier = 1, 
                decay = 0, nonDecisionTime = 100, bias = 0.0)
```

### Define stimuli

This should be of type `NamedTuple` with required field names (case sensitive): `valueLeft` and `valueRight` 

**Option 1: Read in from CSV**  

!!! note

    Note that the `CSV` and `DataFrames` modules must be loaded beforehand. These are dependencies for the ADDM module *but* the precompiled module gives access to these dependencies only in the scope of ADDM. In other words, `ADDM.load_data_from_csv` that requires both of these packages would still work but the code below would not without importing these modules to the current interactive scope.    

```@repl 1
using CSV
using DataFrames

# fn = "./data/stims.csv"
fn = "../../../data/stims.csv"
tmp = DataFrame(CSV.File(fn, delim=","))
MyStims = (valueLeft = tmp.valueLeft, valueRight = tmp.valueRight)
```

**Option 2: Create random stimuli**

!!! note

    If you're going to create random stimuli you should make sure to have value differences that correspond to what you plan to fit in for fixation data.

```
Random.seed!(38535)
MyStims = (valueLeft = randn(1000), valueRight = randn(1000))
```

### Define fixationData

Fixation information that will be fed in to the model for simulations should be of type [`FixationData`](https://addm-toolbox.github.io/ADDM.jl/dev/apireference/#Fixation-data). This type organizes empirical fixations to distributions conditional on fixation type (first, second etc.) and value difference.

This organizes both the behavioral and the fixation data as a dictionary of `Trial` objects indexed by subject. Here, we are reading in empirical data that comes with the package but we will not be making use of the observed choices and response times. The empirical data is only used to extract value difference information to index the fixation data correctly. The choices and response times will be simulated below based on the parameters we specified above.

Note also that the `ADDM.load_data_from_csv()` will expect columns `parcode`,`trial`, `rt`, `choice`, `item_left`, `item_right` and convert `item_left` and`item_right` to `valueLeft` and `valueRight`. 

```@repl 1
data = ADDM.load_data_from_csv("../../../data/expdata.csv", "../../../data/fixations.csv")
```

Extract value difference information from the dataset to use in processing the fixations

```@repl 1
vDiffs = sort(unique([x.valueLeft - x.valueRight for x in data["1"]]))
```

When simulating an aDDM we need to input fixations. But instead of using the fixation data from any given subject we summarize the empricial data from all subjects as distributions from which the model samples from depending on the value difference and the fixation type (1st, 2nd etc.).

```@repl 1
MyFixationData = ADDM.process_fixations(data, fixDistType="fixation", valueDiffs = vDiffs)
```

### Simulate data

First we define additional arguments for aDDM trial simulator (e.g. fixation data, time step for simulations). Note these need to be specified as a `NamedTuple`, and must have at least two elements. Otherwise it tries to apply `iterate` to the single element which would likely end with a  `MethodError`. In this example I specify `timeStep` and `cutoff` in addition to the  only required argument without a default `fixationData` to avoid this.

```@repl 1
MyArgs = (timeStep = 10.0, cutOff = 20000, fixationData = MyFixationData)
```

Note that these are *positional* arguments for code efficiency

```@repl 1
SimData = ADDM.simulate_data(MyModel, MyStims, ADDM.aDDM_simulate_trial, MyArgs)
```

Data can also be simulated from [probability distributions of fixation data](https://addm-toolbox.github.io/ADDM.jl/dev/apireference/#ADDM.convert_to_fixationDist).

```
MyFixationDist, MyTimeBins = ADDM.convert_to_fixationDist(MyFixationData)

MyBlankFixationData = ADDM.FixationData(MyFixationData.probFixLeftFirst, MyFixationData.latencies, MyFixationData.transitions, Dict())

MyArgs = (fixationData = MyBlankFixationData, fixationDist = MyFixationDist, timeBins = MyTimeBins)

SimData = ADDM.simulate_data(MyModel, MyStims, ADDM.aDDM_simulate_trial, MyArgs)
```

### Recover parameters using a grid search

The `ADDM.grid_search` function computes

```@repl 1
fn = "../../../data/addm_grid.csv"
tmp = DataFrame(CSV.File(fn, delim=","))
param_grid = Dict(pairs(NamedTuple.(eachrow(tmp))))

best_pars, all_nll_df = ADDM.grid_search(SimData, ADDM.aDDM_get_trial_likelihood, param_grid, Dict(:η=>0.0, :barrier=>1, :decay=>0, :nonDecisionTime=>100, :bias=>0.0))
```

Examine the sum of negative log likelihoods for each parameter combination.

```@repl 1
sort!(all_nll_df, [:nll])
```

Save data frame containing the negative log likelihood info for all parameter combinations you searched for. 

!!! note

    Make sure that you have mounted a local directory to your container if you're working through this tutorial in a docker container. The output path below is the one specified in the installation instructions. You should change it if you want to save your output elsewhere.

    ```
    output_path = '/home/jovyan/work/all_nll_df.csv'
    CSV.write(output_path, all_nll_df)
    ```

You might have noticed that the grid search did not identify the true parameters (`d = 0.007, σ = 0.03, θ = .6`) as the ones with the highest likelihood. This highlights the importance of choosing good stepsizes for the temporal and spatial discretization.

The default stepsizes are defined as `timeStep = 10.0, approxStateStep = 0.1`. Let's reduce the spatial step size and see if we can recover the corect parameter combination.

```@repl 1
my_likelihood_args = (timeStep = 10.0, approxStateStep = 0.01)

best_pars, all_nll_df = ADDM.grid_search(SimData, ADDM.aDDM_get_trial_likelihood, param_grid, Dict(:η=>0.0, :barrier=>1, :decay=>0, :nonDecisionTime=>100, :bias=>0.0), likelihood_args=my_likelihood_args)

sort!(all_nll_df, [:nll])
```