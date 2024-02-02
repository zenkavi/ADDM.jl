# Uncertainty in the best fitting parameters of a single generative process

## Posterior model probability

```@repl 1
using ADDM
using CSV
using DataFrames
using DataFramesMeta
using StatsPlots
```

Read in data from Krajbich et al. (2010)

```@repl 1
krajbich_data = ADDM.load_data_from_csv("../../../data/Krajbich2010_behavior.csv", "../../../data/Krajbich2010_fixations.csv")
```

Run grid search for a single subject. This computes the nll for 64 parameter combinations for a single subject.

```@repl 1
# fn = "./data/Krajbich_grid.csv"
fn = "../../../data/Krajbich_grid3.csv"
tmp = DataFrame(CSV.File(fn, delim=","))
param_grid = Dict(pairs(NamedTuple.(eachrow(tmp))))

my_likelihood_args = (timeStep = 10.0, approxStateStep = 0.01)

subj_data = krajbich_data["18"]
  
best_pars, nll_df, model_posteriors = ADDM.grid_search(subj_data, ADDM.aDDM_get_trial_likelihood, param_grid, 
    Dict(:η=>0.0, :barrier=>1, :decay=>0, :nonDecisionTime=>0, :bias=>0.0), 
    likelihood_args=my_likelihood_args, 
    return_model_posteriors = true)

```

Merge model posteriors with the model parameters they refer to. This creates a dataframe instead of the `model_posteriors` dictionary that is easier to make plots with.

```@repl 1
posteriors_df = DataFrame()

for (k, v) in param_grid
  cur_row = DataFrame([v])
  cur_row.posterior = [model_posteriors[k]]
  append!(posteriors_df, cur_row)
end
```

Plot model posteriors. Note the use of `@chain` and other operations such as `@rsubset` etc. for `dplyr` like functionality in Julia through `DataFrameMeta.jl`.  

Note also that we're only plotting the posteriors for models that have a meaningful amount of probability mass instead of all the 64 models that were tested.

```@repl 1
plot_df = @chain posteriors_df begin
  @rsubset :posterior > 1e-10
  @rtransform :x_label = "d: " * string(:d) * ", \nσ: " * string(:sigma) * ", \nθ: " * string(:theta) 
  @orderby -:posterior
  end

@df plot_df bar(:x_label, :posterior, legend = false, xrotation = 45, ylabel = "p(model|data)",bottom_margin = (5, :mm))

savefig("plot2.png"); nothing # hide
```
![plot](plot2.png)

## Marginal posteriors for parameters


Compute and plot posteriors for each parameter

```@repl 1
param_posteriors = ADDM.marginal_posteriors(param_grid, model_posteriors)

plot_array = Any[]
for plot_df in param_posteriors
  x_lab = names(plot_df)[1]
  cur_plot = @df plot_df bar(plot_df[:, x_lab], :posterior_sum, leg = false, ylabel = "p(" * x_lab * " = x|data)", xlabel = x_lab )
  push!(plot_array, cur_plot) 
end
plot(plot_array...)

savefig("plot3.png"); nothing # hide
```
![plot](plot3.png)

Compute and plot marginal posteriors with heatmaps

```@repl 1
marginal_posteriors = ADDM.marginal_posteriors(param_grid, model_posteriors, true)

margpostplot(marginal_posteriors)

savefig("plot4.png"); nothing # hide
```
![plot](plot4.png)


# Comparing fit of different generative processes

## Estimate best fitting parameters separately for each process

This isn't actually necessary. You only need trial likelihoods and priors for each model that are indexed in a way that leaves no ambiguity about which model generated with trial likelihoods

You can't compute marginal parameter distributions across different generative processes (I don't think) but you could compare the two best fitting parameter combinations from one generative process to an entirely different generative process, as long as you have the trial likelihoods for each model.

## Compute trial likelihoods plugging in best fitting parameters


# True vs. simulated data