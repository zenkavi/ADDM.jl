# Parameter estimation on empirical data

## Load packages

```@repl 1
using ADDM
using CSV
using DataFrames
```

## Read in data

Data in this tutorial are from 10 subjects in Kraijbich et al (2010). We will use the built-in data loading function `ADDM.load_data_from_csv` that expects a behavioral file with columns `parcode, trial, rt, choice, item_left, item_right` and fixations file with columns `parcode, trial, fix_item, fix_time`

If your data is not organized in this way you could either preprocess it so it does or you can read in the data however you want and reshape it with Julia to ensure it is organized as a dictionary of `Trial` objects *indexed by subject/parcode*. A `Trial` looks like

```
ADDM.Trial(1, 1474.0, -5, 5, Number[3, 0, 1, 0, 2, 0], Number[270.0, 42.0, 246.0, 62.0, 558.0, 296.0], #undef, #undef, #undef)
```  

where the first element is choice (-1 for left, +1 for right), second element is response time in ms, third is value of left option, fourth is value of right option. Fixation data is specified in the fourth and fifth elements as fixation location (1 for left, 2 for right) and fixation duration (in ms) respectively.  


```@repl 1
krajbich_data = ADDM.load_data_from_csv("../../../data/Krajbich2010_behavior.csv", "../../../data/Krajbich2010_fixations.csv")
```

## Grid search

Using a grid of 64 parameter combinations with `d` in {0.0001, 0.00015, 0.0002, 0.00025}, `μ` in {80, 100, 120, 140}, `θ` in {0.3, 0.5, 0.7, 0.9}  and `σ = d*μ`   

```@repl 1
fn = "../../../data/Krajbich_grid.csv"
tmp = DataFrame(CSV.File(fn, delim=","))
param_grid = Dict(pairs(NamedTuple.(eachrow(tmp))))

all_nll_df = DataFrame()
best_pars = Dict()

for k in keys(krajbich_data)
  cur_subj_data = krajbich_data[k]
  
  subj_best_pars, subj_nll_df = ADDM.grid_search(cur_subj_data, ADDM.aDDM_get_trial_likelihood, param_grid, Dict(:η=>0.0, :barrier=>1, :decay=>0, :nonDecisionTime=>0, :bias=>0.0))

  best_pars[k] = subj_best_pars

  subj_nll_df[!, "parcode"] .= k
  append!(all_nll_df, subj_nll_df)
  
end
```

To view best parameter estimates for each subject

```@repl 1
best_pars
```

Plot variability in the negative log likelihoods for each parameter combination for each subject

```@repl 1
using StatsPlots

wide_nll_df = unstack(all_nll_df, :parcode, :nll)
select!(wide_nll_df, Not([:d, :sigma, :theta]))
colnames = names(wide_nll_df)
colnames = string.("subj-", colnames)
N = length(colnames)

@df wide_nll_df histogram(cols(1:N); layout=grid(2,5), legend=false, title=permutedims(colnames), frame=:box, titlefontsize=11, c=:blues, bins = 20, size=(1800,1000), xrotation = 45)
```
