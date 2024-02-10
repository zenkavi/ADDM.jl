# Model comparison

The parameter combination that has the highest likelihood to have generated a given dataset is often what is used in downstream analyses and related to other variables of interest. While a fast estimation these parameters is therefore very useful, it is valueable to get a sense of the uncertainty associated with the estimation. In this tutorial we introduce some of the toolbox's capabilities to assess this.

When estimating the best-fitting parameters for a model (aDDM or otherwise) our ability to recover them is *always* limited to the parameter space we explore. Therefore, any computation of the uncertainty associated with specific parameters values is only with respect to other values that we have tried.

In other words, the uncertainty is not some divine measure that accounts for all possible models. It is a comparative measure that tells us how much better a specific combination of parameters is compared to other combinations in the parameter space we have defined. In this toolbox, we make the parameter space explicit by specifying the grid in the `ADDM.grid_search` function. 

The uncertainty associated with each parameter value and/or parameter combination is quantified as a probability distribution. Specifically, a posterior probability distribution that reflects both the prior beliefs on how likely each parameter value is and how much to update them based on how much evidence each trial provides in favor of a parameter combination.

## Comparing parameters of a single generative processes

In this section we will demonstrate how to compute posterior probabilities associated with each parameter combination and each parameter type for a single generative process. A generative process, in this context, refers to the computational model we believe gives rise to observable data (in this case, choices and response times). Here, we compute the uncertainty over different parameter combinations of one specific computational model, the standard aDDM. In the next section we compute the uncertainty over different computational models, accounting for the uncertainty within the parameter spaces of each model.

### Posterior model probability

```@repl 1
using ADDM
using CSV
using DataFrames
using DataFramesMeta
using StatsPlots
```

Read in a subset of the data from Krajbich et al. (2010) that comes with the toolbox.

```@repl 1
krajbich_data = ADDM.load_data_from_csv("../../../data/Krajbich2010_behavior.csv", "../../../data/Krajbich2010_fixations.csv")
```

Run grid search for a single subject. This computes the negative log-likelihood (nll) for each parameter combination for a single subject. Moreover, here we introduce the `return_model_posteriors` argument when running `ADDM.grid_search`, which expands the output to include a `model_posteriors` dictionary. 

!!! note

    `model_posteriors` contains the posterior probability associated with each model (i.e. parameter combination) **for the set of models that were fit**. Since it is a probability distribution it sums to 1. In other words, the posterior probabilities associated with the models would change if they were being compared to different combinations of parameters.


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

### Model posteriors

The `model_posteriors` variable returned above when running the grid search is a dictionary indexed by the model number as listed in the `param_grid` input and does not contain information on the specific combination of parameters for each model. Here, we convert that `model_posteriors` dictionary to a dataframe with the specific parameter information so it is easier to make plots with.

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

Note also that we're only plotting the posteriors for models that have a meaningful amount of probability mass instead of all the models that were tested by excluding rows without a posterior probability greater than `1e-10`.

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

### Parameter posteriors

As described above the `model_posteriors` dictionary contains the probability distribution associated with each parameter *combination* but does *not* include the information on the individual parameter values. The `ADDM.marginal_posteriors` function adds this information and summarizes the probability distribution collapsing over levels of different parameters. Below, we first summarize the distribution for each of the three parameters separately.

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

We can also use the `ADDM.marginal_posteriors` function to compute parameter posteriors with respect to each other by specifying the third positional argument. When set to `true`, the `ADDM.marginal_posteriors` function returns pairwise marginal distributions that can be plotted as heatmaps to visualize conditional distributions of the parameters.   

The toolbox includes a visualization function, `ADDM.margpostplot` that creates a grid of plots with individual parameter posteriors on the diagonal and the conditional posteriors as heatmaps below the diagonal.

```@repl 1
marginal_posteriors = ADDM.marginal_posteriors(param_grid, model_posteriors, true)

ADDM.margpostplot(marginal_posteriors)

savefig("plot5.png"); nothing # hide
```
![plot](plot5.png)

## Comparing different generative processes

Aside from comparing different parameter combinations for a single model, we can also compare how likely one computational model is compared to another, in generating the observed data. Since any specific value of a given parameter involves uncertainty as we computed above, we need to account for this when comparing different generative processes to each other.

This again involves computing the comparative advantage, the posterior probability, for each point in the parameter space that we examine but now the parameter space does not only contain the parameters within each model, but also which model they belong to. 

Here, we'll use the same participant's data from before and examine if it can be explained better by a standard aDDM (that we fit above) or another model where the boundaries of the evidence accummulation decay exponentially throughout the decision. This model is detailed further in the [Defining custom models](https://addm-toolbox.github.io/ADDM.jl/dev/tutorials/custom_model/) tutorial.

The comparison of these two generative processes is operationalized by specifying them in the same `param_grid` as we had previously used to specify different values for the parameters of a single generative process. In this case, we add the information on which generative process the parameter combination belongs to in a new column called `likelihood_fn`.

First we read in the file that defines the parameter space for the first model, the standard aDDM.

```@repl 1
# fn = "../../../data/Krajbich_grid3.csv"
fn = "./data/Krajbich_grid3.csv"
tmp = DataFrame(CSV.File(fn, delim=","))
tmp.likelihood_fn .= "ADDM.aDDM_get_trial_likelihood"
param_grid1 = Dict(pairs(NamedTuple.(eachrow(tmp))))
```

Then we define the likelihood function for the second model along with the parameter space we will examine for this second model. Note also that we modify the indices of the specific parameter combinations for this second model to avoid over-writing the parameters from the first model.

```@repl 1
# include("./my_likelihood_fn.jl")
include("./docs/src/tutorials/my_likelihood_fn.jl")

# fn = "../../../data/custom_model_grid.csv"
fn = "./data/custom_model_grid.csv"
tmp = DataFrame(CSV.File(fn, delim=","))
tmp.likelihood_fn .= "my_likelihood_fn"
param_grid2 = Dict(pairs(NamedTuple.(eachrow(tmp))))

# Increase the indices of the second model's parameter combinations 
# This avoid overwriting the parameter combinations with the same index 
# in the first parameter grid
param_grid2 = Dict(keys(param_grid2) .+ length(param_grid1) .=> values(param_grid2))
```

Now that we have defined the parameter space for both models, we combine them both in a single `param_grid`, over which we'll compute the posterior distribution.

```@repl 1
param_grid = Dict(param_grid1..., param_grid2...)
```

With this expanded `param_grid` that includes information on the different likelihood functions we call the `ADDM.grid_search` function setting the third position argument to `nothing`. This argument is where we define the likelihood function in the case of a single model but now this is specified in the `param_grid`.

```@repl 1
my_likelihood_args = (timeStep = 10.0, approxStateStep = 0.1);
  
# Haven't tested this part yet
best_pars, nll_df, model_posteriors = ADDM.grid_search(subj_data, param_grid, nothing,
    Dict(:η=>0.0, :barrier=>1, :decay=>0, :nonDecisionTime=>0, :bias=>0.0), 
    likelihood_args=my_likelihood_args, 
    return_model_posteriors = true);
```

Just as before, the `model_posteriors` dictionary does not contain information on the parameter, so we combine it with the `param_grid` in a `DataFrame` for visualization purposes.

```@repl 1
posteriors_df = DataFrame();
for (k, v) in param_grid
  cur_row = DataFrame([v])
  cur_row.posterior = [model_posteriors[k]]
  # append!(posteriors_df, cur_row)
  posteriors_df = vcat(posteriors_df, cur_row, cols=:union)
end;
```

We can take a look at the most likely parameter combinations across the generative processes.

```@repl 1
sort(posteriors_df, :posterior, rev=true)
```

We can also collapse the posterior distribution across the generative processes and compare how much better one processes is compared to the other in giving rise to the observed data.  

```@repl 1
gdf = groupby(posteriors_df, :likelihood_fn);
combdf = combine(gdf, :posterior => sum)

@df combdf bar(:likelihood_fn, :posterior_sum, legend = false, xrotation = 45, ylabel = "p(model|data)",bottom_margin = (5, :mm))

savefig("plot6.png"); nothing # hide
```
![plot](plot6.png)


## True vs. simulated data