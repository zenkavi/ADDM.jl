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
using Distributions
using LinearAlgebra
using StatsPlots
```

Read in a subset of the data from Krajbich et al. (2010) that comes with the toolbox.

```@repl 1
krajbich_data = ADDM.load_data_from_csv("../../../data/Krajbich2010_behavior.csv", "../../../data/Krajbich2010_fixations.csv");
```

Run grid search for a single subject. This computes the negative log-likelihood (nll) for each parameter combination for a single subject. Moreover, here we introduce the `return_model_posteriors` argument when running `ADDM.grid_search`, which expands the output to include a `model_posteriors` dictionary. 

!!! note

    `model_posteriors` contains the posterior probability associated with each model (i.e. parameter combination) **for the set of models that were fit**. Since it is a probability distribution it sums to 1. In other words, the posterior probabilities associated with the models would change if they were being compared to different combinations of parameters.


```@repl 1
fn = "../../../data/Krajbich_grid3.csv"
tmp = DataFrame(CSV.File(fn, delim=","))
param_grid = Dict(pairs(NamedTuple.(eachrow(tmp))))

my_likelihood_args = (timeStep = 10.0, approxStateStep = 0.1);

subj_data = krajbich_data["18"];
  
best_pars, nll_df, trial_posteriors = ADDM.grid_search(subj_data, param_grid, ADDM.aDDM_get_trial_likelihood, 
    Dict(:η=>0.0, :barrier=>1, :decay=>0, :nonDecisionTime=>0, :bias=>0.0), 
    likelihood_args=my_likelihood_args, 
    return_model_posteriors = true);

nTrials = length(subj_data)
model_posteriors = Dict(zip(keys(trial_posteriors), [x[nTrials] for x in values(trial_posteriors)]))

```

### Model posteriors

The `model_posteriors` variable returned above when running the grid search is a dictionary indexed by the model number as listed in the `param_grid` input and does not contain information on the specific combination of parameters for each model. Here, we convert that `model_posteriors` dictionary to a dataframe with the specific parameter information so it is easier to make plots with.

```@repl 1
posteriors_df = DataFrame();

for (k, v) in param_grid
  cur_row = DataFrame([v])
  cur_row.posterior = [model_posteriors[k]]
  posteriors_df = vcat(posteriors_df, cur_row, cols=:union)
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

savefig("plot_3_1.png"); nothing # hide
```
![plot](plot_3_1.png)

#### Trialwise changes to the model posteriors

```@repl 1
trial_model_posteriors = DataFrame();
for i in 1:nTrials
  # Get the posterior for each model after the curent trial
  cur_trial_posteriors = Dict(zip(keys(trial_posteriors), [x[i] for x in values(trial_posteriors)]))
  cur_trial_posteriors = DataFrame(model_num = collect(keys(cur_trial_posteriors)), posterior = collect(values(cur_trial_posteriors)))
  cur_trial_posteriors[!, :trial_num] .= i
  trial_model_posteriors = vcat(trial_model_posteriors, cur_trial_posteriors, cols=:union)
end
```

Plot changes to posteriors of each model across trials

```@repl 1
@df trial_model_posteriors plot(
      :trial_num,
      :posterior,
      group = :model_num,
      xlabel = "Trial",
      ylabel = "Posterior p",
      legend = false
  )
```


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

savefig("plot_3_2.png"); nothing # hide
```
![plot](plot_3_2.png)

We can also use the `ADDM.marginal_posteriors` function to compute parameter posteriors with respect to each other by specifying the third positional argument. When set to `true`, the `ADDM.marginal_posteriors` function returns pairwise marginal distributions that can be plotted as heatmaps to visualize conditional distributions of the parameters.   

The toolbox includes a visualization function, `ADDM.margpostplot` that creates a grid of plots with individual parameter posteriors on the diagonal and the conditional posteriors as heatmaps below the diagonal.

```@repl 1
marginal_posteriors = ADDM.marginal_posteriors(param_grid, model_posteriors, true)

ADDM.margpostplot(marginal_posteriors)

savefig("plot_3_3.png"); nothing # hide
```
![plot](plot_3_3.png)

#### Trialwise changes to the parameter posteriors

```@repl 1
trial_param_posteriors = DataFrame();
for i in 1:nTrials
  # Get the posterior for each model after the curent trial
  cur_trial_posteriors = Dict(zip(keys(trial_posteriors), [x[i] for x in values(trial_posteriors)]))

  # Use built-in function to marginalize for each parameter
  cur_param_posteriors = ADDM.marginal_posteriors(param_grid, cur_trial_posteriors)

  # Wrangle the output to be a single df and add trial number info
  for j in 1:length(cur_param_posteriors)
    df = cur_param_posteriors[j][:,:] #assign a copy
    
    df[!, :par_name] .= names(df)[1]
    df[!, :trial_num] .= i
    rename!(df, Symbol(names(df)[1]) => :par_value)

    trial_param_posteriors = vcat(trial_param_posteriors, df, cols=:union)

  end

end
```

Plot trialwise marginal posteriors for each parameter

```@repl 1
par_names = unique(trial_param_posteriors[:,:par_name])

plot_array = Any[]

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

end

plot(plot_array...)

savefig("plot_3_4.png"); nothing # hide
```
![plot](plot_3_4.png)


## Comparing different generative processes

Aside from comparing different parameter combinations for a single model, we can also compare how likely one computational model is compared to another, in generating the observed data. Since any specific value of a given parameter involves uncertainty as we computed above, we need to account for this when comparing different generative processes to each other.

This again involves computing the comparative advantage, the posterior probability, for each point in the parameter space that we examine but now the parameter space does not only contain the parameters within each model, but also which model they belong to. 

Here, we'll use the same participant's data from before and examine if it can be explained better by a standard aDDM (that we fit above) or another model where the boundaries of the evidence accummulation decay exponentially throughout the decision. This model is detailed further in the [Defining custom models](https://addm-toolbox.github.io/ADDM.jl/dev/tutorials/custom_model/) tutorial.

The comparison of these two generative processes is operationalized by specifying them in the same `param_grid` as we had previously used to specify different values for the parameters of a single generative process. In this case, we add the information on which generative process the parameter combination belongs to in a new column called `likelihood_fn`.

First we read in the file that defines the parameter space for the first model, the standard aDDM.

```@repl 1
fn = "../../../data/Krajbich_grid3.csv";
tmp = DataFrame(CSV.File(fn, delim=","));
tmp.likelihood_fn .= "ADDM.aDDM_get_trial_likelihood";
param_grid1 = Dict(pairs(NamedTuple.(eachrow(tmp))))
```

Then we define the likelihood function for the second model along with the parameter space we will examine for this second model. Note also that we modify the indices of the specific parameter combinations for this second model to avoid over-writing the parameters from the first model.

```@repl 1
include("./my_likelihood_fn.jl")

fn = "../../../data/custom_model_grid.csv";
tmp = DataFrame(CSV.File(fn, delim=","));
tmp.likelihood_fn .= "my_likelihood_fn";
param_grid2 = Dict(pairs(NamedTuple.(eachrow(tmp))));

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

savefig("plot_3_5.png"); nothing # hide
```
![plot](plot_3_5.png)


## Comparing true data with simulated data

The comparison of the generative processes above strongly favors the standard aDDM over the custom model in generating the observed data (within the ranges of the parameter space we explored).

Another way to examine how well a model describes observed data is by comparing how well it predicts observed patterns. In this case, this would involve inspecting response time distributions conditional on choice as these are the two outputs of the generative models.

One can choose different features and statistics about the observed data to compare with model predictions. Below, we plot how the response time distributions for the best fitting model from each generative process compares to the true data.  

First, we get best fitting parameters for each model.

```@repl 1
bestModelPars = @chain posteriors_df begin
    groupby(:likelihood_fn) 
    combine(_) do sdf
        sdf[argmax(sdf.posterior), :]
    end
  end
```

Using these parameters for each model we simulate data for the stimuli used in the true data.

```@repl 1
## Prepare inputs for simulator

vDiffs = sort(unique([x.valueLeft - x.valueRight for x in subj_data]));
fixData = ADDM.process_fixations(krajbich_data, fixDistType="fixation", valueDiffs = vDiffs);
MyArgs = (timeStep = 10.0, cutOff = 20000, fixationData = fixData);
MyStims = (valueLeft = [x.valueLeft for x in subj_data], valueRight = [x.valueRight for x in subj_data])

## Define standard model with the best fitting parameters
standPars = @rsubset bestModelPars :likelihood_fn == "ADDM.aDDM_get_trial_likelihood";
standModel = ADDM.define_model(d = standPars.d[1], σ = standPars.sigma[1], θ = standPars.theta[1]);

## Simulate data for the best standard model
simStand = ADDM.simulate_data(standModel, MyStims, ADDM.aDDM_simulate_trial, MyArgs);

## Read in trial simulator function for alternative model
include("./my_trial_simulator.jl")

## Define standard model with the best fitting parameters
altPars = @rsubset bestModelPars :likelihood_fn == "my_likelihood_fn";
altModel = ADDM.define_model(d = 0.007, σ = 0.03, θ = .6, barrier = 1, nonDecisionTime = 100, bias = 0.0)
altModel.λ = .05

## Simulate data for the best alternative model
simAlt = ADDM.simulate_data(altModel, MyStims, my_trial_simulator, MyArgs);
```

Finally we extract and plot the response time data for the true and simulated data.

```@repl 1
# Plot true RT histograms overlaid with simulated RT histograms

## Define the limit for the x-axis based on true data
rts = [i.RT * i.choice for i in subj_data]; #left choice rt's are negative
l = abs(minimum(rts)) > abs(maximum(rts)) ? abs(minimum(rts)) : abs(maximum(rts))

## Split the RTs for left and right choice. Left is on the left side of the plot
rts_pos = [i.RT for i in subj_data if i.choice > 0];
rts_neg = [i.RT * (-1) for i in subj_data if i.choice < 0];

rts_pos_stand = [i.RT for i in simStand if i.choice > 0];
rts_pos_alt = [i.RT for i in simAlt if i.choice > 0];

rts_neg_stand = [i.RT * (-1) for i in simStand if i.choice < 0];
rts_neg_alt = [i.RT * (-1) for i in simAlt if i.choice < 0];

## Make plot

histogram(rts_pos, normalize=true, bins = range(-l, l, length=41), fillcolor = "gray", yaxis = false, grid = false, label = "True data")
density!(rts_pos_stand, label = "ADDM predictions", linewidth = 3, linecolor = "blue")
density!(rts_pos_alt, label = "Custom model predictions", linewidth = 3, linecolor = "green")

histogram!(rts_neg, normalize=true, bins = range(-l, l, length=41), fillcolor = "gray", label = "")
density!(rts_neg_stand, linewidth = 3, linecolor = "blue", label = "")
density!(rts_neg_alt, linewidth = 3, linecolor = "green", label = "")

vline!([0], linecolor = "red", label = "")

savefig("plot_3_6.png"); nothing # hide
```
![plot](plot_3_6.png)