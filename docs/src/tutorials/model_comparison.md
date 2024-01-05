
What metrics can you use to compare non-nested models with different number of parameters?

WAIC

Cross validation

Posterior model probability/Bayes factor

# Uncertainty in the best fitting parameters of a single generative process

## Posterior model probability

```@repl 1
using ADDM
using CSV
using DataFrames
using DataFramesMeta
using StatsPlots
```

```@repl 1
krajbich_data = ADDM.load_data_from_csv("./data/Krajbich2010_behavior.csv", "./data/Krajbich2010_fixations.csv")
```

Run grid search for a single subject. This computes the nll for 64 parameter combinations for a single subject.

```@repl 1
fn = "./data/Krajbich_grid.csv"
tmp = DataFrame(CSV.File(fn, delim=","))
param_grid = Dict(pairs(NamedTuple.(eachrow(tmp))))

my_likelihood_args = (timeStep = 10.0, approxStateStep = 0.01)

subj_data = krajbich_data["14"]
  
best_pars, nll_df, model_posteriors = ADDM.grid_search(subj_data, ADDM.aDDM_get_trial_likelihood, param_grid, 
    Dict(:η=>0.0, :barrier=>1, :decay=>0, :nonDecisionTime=>0, :bias=>0.0), 
    likelihood_args=my_likelihood_args, 
    return_model_posteriors = true)

```

Merge model posteriors with the model parameters they refer to

```@repl 1
posteriors_df = DataFrame()

for (k, v) in param_grid
  cur_row = DataFrame([v])
  cur_row.posterior = [model_posteriors[k]]
  append!(posteriors_df, cur_row)
end
```

Plot model posteriors

```@repl 1
plot_df = @chain posteriors_df begin
  @rsubset :posterior > 1e-10
  @rtransform :x_label = "d: " * string(:d) * ", \nσ: " * string(:sigma) * ", \nθ: " * string(:theta) 
  @orderby -:posterior
  end

@df plot_df bar(:x_label, :posterior, legend = false, xrotation = 45, ylabel = "p(model|data)",bottom_margin = (5, :mm))
```

## Marginal posteriors for parameters


Compute marginal posteriors

```@repl 1
ADDM.marginal_posteriors(param_grid, model_posteriors)
```

Plot marginal posteriors

```@repl 1
```

# Comparing fit of different generative processes

## Estimate best fitting parameters separately for each process

This isn't actually necessary. You only need trial likelihoods and priors for each model that are indexed in a way that leaves no ambiguity about which model generated with trial likelihoods

You can't compute marginal parameter distributions across different generative processes (I don't think) but you could compare the two best fitting parameter combinations from one generative process to an entirely different generative process, as long as you have the trial likelihoods for each model.

## Compute trial likelihoods plugging in best fitting parameters


# True vs. simulated data