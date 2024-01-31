"""
    margpostplot

This plot type shows the posteriors for each parameter individually,
    as well as the posterior probabilities of pairwise combinations.  

The input is an array of dataframes resulting from 

```
best_pars, nll_df, model_posteriors = ADDM.grid_search(subj_data, ADDM.aDDM_get_trial_likelihood, param_grid, 
    Dict(:η=>0.0, :barrier=>1, :decay=>0, :nonDecisionTime=>0, :bias=>0.0), 
    likelihood_args=my_likelihood_args, 
    return_model_posteriors = true)

ADDM.marginal_posteriors(param_grid, model_posteriors, true)
```

Recipe modified from 
https://github.com/JuliaPlots/StatsPlots.jl/blob/master/src/corrplot.jl
"""
import Plots: _cycle
using Plots.PlotMeasures

@userplot MargPostPlot

recipetype(::Val{:margpostplot}, args...) = MargPostPlot(args)

function update_ticks_guides(d::KW, labs, i, j, n)
    # d[:title]  = (i==1 ? _cycle(labs,j) : "")
    # d[:xticks] = (i==n)
    d[:xguide] = (i == n ? _cycle(labs, j) : "")
    # d[:yticks] = (j==1)
    d[:yguide] = (j == 1 ? _cycle(labs, i) : "")
end

@recipe function f(mpp::MargPostPlot)
    mps = mpp.args[1]
    n = 0
    for i in mps
      if length(names(i)) == 2
        n += 1
      end
    end
    mps1 = mps[1:n]
    mps2 = mps[n+1:length(mps)]
    labs = pop!(plotattributes, :label, [""])

    g = grid(n, n)
    indices = zeros(Int8, (n, n))
    s = 1
    # Make upper triangle blank
    for i = 1:n, j = 1:n
      isblank = i < j
      g[i, j].attr[:blank] = isblank
      if !isblank
        indices[i, j] = s
        s += 1
      end
    end

    link := :x  # need custom linking for y
    # layout := g
    layout := grid(n, n)
    legend := false
    foreground_color_border := nothing
    margin := 1mm
    titlefont := font(11)

    title = get(plotattributes, :title, "")
    title_location = get(plotattributes, :title_location, :center)
    title := "" # does this over-write user-specific titles?

    # barplots for individual parameters on the diagonal
    for i = 1:n
        @series begin
            if title != "" && title_location === :left && i == 1
                title := title
            end
            seriestype := :bar
            subplot := indices[i, i]
            # legend := false
            # subplot := i
            grid := false
            xformatter --> ((i == n) ? :auto : (x -> ""))
            yformatter --> ((i == 1) ? :auto : (y -> ""))
            # update_ticks_guides(plotattributes, labs, i, i, n)
            # data that will be plotted using the seriestype
            vx = view(mps1[i], :, 1) # param column
            vy = view(mps1[i], :, 2) # posterior_sum column
            vx, vy
        end
    end

    # heatmaps below diagonal
    for j = 1:n
      # this fills things by row. should i change it depending on how mpp2 is indexed?
        # ylink := setdiff(vec(indices[i, :]), indices[i, i])
      # filling by column would be
        # xlink:= setdiff(vec(indices[:, j]), indices[j, j])
        for i = 1:n
            j == i && continue
            subplot := indices[i, j]
            # update_ticks_guides(plotattributes, labs, i, j, n)
            if i > j
                # heatmaps below diagonal
                @series begin
                    seriestype := :heatmap
                    xformatter --> ((i == n) ? :auto : (x -> ""))
                    yformatter --> ((j == 1) ? :auto : (y -> ""))

                    # Reshape data for heatmap
                    cur_mp2 = popfirst!(mps2)
                    vx = sort(unique(view(cur_mp2, :, 1)))
                    vy = sort(unique(view(cur_mp2, :, 2)))
                    vz = fill(NaN, size(vy, 1), size(vx, 1))
                    for k in eachindex(vx), l in eachindex(vy)
                      cur_row = subset(cur_mp2, 1 => a -> a .== vx[k],  2 => b -> b .== vy[l])
                      if nrow(cur_row) > 0
                        vz[l, k] = cur_row.posterior_sum[1]
                      end
                    end
                    vx, vy, vz
                end
              else
                @series begin
                  blank := 1
                end
            end
        end
    end
  end