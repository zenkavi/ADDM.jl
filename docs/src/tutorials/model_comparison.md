# Model comparison

## Comparing parameters of a single generative processes

### Posterior model probability

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

my_likelihood_args = (timeStep = 10.0, approxStateStep = 0.01);

subj_data = krajbich_data["18"];
  
best_pars, nll_df, model_posteriors = ADDM.grid_search(subj_data, param_grid, ADDM.aDDM_get_trial_likelihood, 
    Dict(:η=>0.0, :barrier=>1, :decay=>0, :nonDecisionTime=>0, :bias=>0.0), 
    likelihood_args=my_likelihood_args, 
    return_model_posteriors = true);

```

### Individual parameter posteriors

Merge model posteriors with the model parameters they refer to. This creates a dataframe instead of the `model_posteriors` dictionary that is easier to make plots with.

```@repl 1
posteriors_df = DataFrame();

for (k, v) in param_grid
  cur_row = DataFrame([v])
  cur_row.posterior = [model_posteriors[k]]
  # append!(posteriors_df, cur_row)
  all_nll_df = vcat(all_nll_df, row, cols=:union)
end;
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

savefig("plot3.png"); nothing # hide
```
![plot](plot3.png)

### Marginal posteriors for parameters

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

savefig("plot4.png"); nothing # hide
```
![plot](plot4.png)

Compute and plot marginal posteriors with heatmaps

```@repl 1
marginal_posteriors = ADDM.marginal_posteriors(param_grid, model_posteriors, true)

ADDM.margpostplot(marginal_posteriors)

savefig("plot5.png"); nothing # hide
```
![plot](plot5.png)

## Comparing different generative processes

```@repl 1
# fn = "../../../data/Krajbich_grid3.csv"
fn = "./data/Krajbich_grid3.csv"
tmp = DataFrame(CSV.File(fn, delim=","))
tmp.likelihood_fn .= "ADDM.aDDM_get_trial_likelihood"
param_grid1 = Dict(pairs(NamedTuple.(eachrow(tmp))))

# fn = "../../../data/custom_model_grid.csv"
fn = "./data/custom_model_grid.csv"
tmp = DataFrame(CSV.File(fn, delim=","))
tmp.likelihood_fn .= "my_likelihood_fn"
param_grid2 = Dict(pairs(NamedTuple.(eachrow(tmp))))

# Increase the indices of the second model's parameter combinations 
# This avoid overwriting the parameter combinations with the same index 
# in the first parameter grid
param_grid2 = Dict(keys(param_grid2) .+ length(param_grid1) .=> values(param_grid2))

param_grid = Dict(param_grid1..., param_grid2...)


my_likelihood_args = (timeStep = 10.0, approxStateStep = 0.01);
  
# Haven't tested this part yet
best_pars, nll_df, model_posteriors = ADDM.grid_search(subj_data, param_grid, nothing
    Dict(:η=>0.0, :barrier=>1, :decay=>0, :nonDecisionTime=>0, :bias=>0.0), 
    likelihood_args=my_likelihood_args, 
    return_model_posteriors = true);

posteriors_df = DataFrame();
for (k, v) in param_grid
  cur_row = DataFrame([v])
  cur_row.posterior = [model_posteriors[k]]
  # append!(posteriors_df, cur_row)
  posteriors_df = vcat(posteriors_df, cur_row, cols=:union)
end;


```


## More?

- True vs. simulated data
    - RT distributions conditional on choice