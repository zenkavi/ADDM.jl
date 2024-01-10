"""
    marginal_posteriors(param_grid, posteriors_dict, two_d_marginals)

Compute the marginal posterior distributions for the fitted parameters specified in `param_grid`.

# Arguments

## Required 

- `param_grid`: Grid of parameter combinations for which the sum of nll's for the `data` is 
  computed.
- `posteriors_dict`: Dictionary of posterior model probabilities. Keys of this dictionary should match
  the keys of the `param_grid` for the models the probabilities refer to.
- `two_d_marginals`: Boolean. Whether to compute posteriors to plot heatmaps of posteriors.

# Returns
- Vector of `DataFrame`s. If `two_d_marginals` is false, return only dataframes containing
  posteriors for each parameter. Otherwise, also includes posteriors for pairwise combinations of 
  parameters as well.

"""
function marginal_posteriors(param_grid, posteriors_dict, two_d_marginals = false)

  posteriors_df = DataFrame()

  for (k, v) in param_grid
    cur_row = DataFrame([v])
    cur_row.posterior = [posteriors_dict[k]]
    append!(posteriors_df, cur_row)
  end

  par_names = names(posteriors_df)[names(posteriors_df) .!= "posterior"]
  
  if two_d_marginals
    par_combs = combinations(par_names, 2)
    out = Vector{}(undef, (length(par_names)+length(par_combs)))
  else
    out = Vector{}(undef, length(par_names))
  end

  # this is only for single parameters, diagonal plots
  for (i,n) in enumerate(par_names)
    gdf = groupby(posteriors_df, n)
    combdf = combine(gdf, :posterior => sum)
    out[i] = combdf
  end

  if two_d_marginals
    l = length(par_names)
    for (j,c) in enumerate(par_combs)
      gdf = groupby(posteriors_df, c)
      combdf = combine(gdf, :posterior => sum)
      out[l + j] = combdf
    end
  end

  return out

end