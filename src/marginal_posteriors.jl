"""
    marginal_posteriors(param_grid, posteriors
                fixed_params = Dict(:θ=>1.0, :η=>0.0, :barrier=>1, :decay=>0, :nonDecisionTime=>0, :bias=>0.0))

Compute the marginal posterior distributions for the fitted parameters specified in `param_grid`.

# Arguments

## Required 

- `param_grid`: Grid of parameter combinations for which the sum of nll's for the `data` is 
  computed.
- `posteriors`: Dictionary of posterior model probabilities. Keys of this dictionary should match
  the keys of the `param_grid` for the models the probabilities refer to.

# Returns
- 

"""
function marginal_posteriors(param_grid, posteriors_dict)

  posteriors_df = DataFrame()

  for (k, v) in param_grid
    cur_row = DataFrame([v])
    cur_row.posterior = [posteriors_dict[k]]
    append!(posteriors_df, cur_row)
  end

  par_names = names(posteriors_df)[names(posteriors_df) .!= "posterior"]

  out = Vector{}(undef, length(par_names))

  for (i,n) in enumerate(par_names)
    gdf = groupby(posteriors_df, n)
    combdf = combine(gdf, :posterior => sum)
    out[i] = combdf
  end

  return out


