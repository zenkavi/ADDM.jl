using ADDM, CSV, DataFrames

# Create SimData

MyModel = ADDM.define_model(d = 0.007, σ = 0.03, θ = .6, barrier = 1, decay = 0, nonDecisionTime = 100, bias = 0.0)

dp = "/Users/zenkavi/Documents/RangelLab/aDDM-Toolbox/ADDM.jl/data/"
data = ADDM.load_data_from_csv(dp*"stimdata.csv", dp*"fixations.csv"; stimsOnly = true);

nTrials = 1500

MyStims = (valueLeft = reduce(vcat, [[i.valueLeft for i in data[j]] for j in keys(data)])[1:nTrials], valueRight = reduce(vcat, [[i.valueRight for i in data[j]] for j in keys(data)])[1:nTrials]);

vDiffs = sort(unique([x.valueLeft - x.valueRight for x in data["1"]]));

MyFixationData = ADDM.process_fixations(data, fixDistType="fixation", valueDiffs = vDiffs);

MyArgs = (timeStep = 10.0, cutOff = 20000, fixationData = MyFixationData);

SimData = ADDM.simulate_data(MyModel, MyStims, ADDM.aDDM_simulate_trial, MyArgs);


# Save SimData
SimDataBehDf = DataFrame()
SimDataFixDf = DataFrame()

for (i, cur_trial) in enumerate(SimData)
  # "parcode","trial","fix_time","fix_item"
  cur_fix_df = DataFrame(:fix_item => cur_trial.fixItem, :fix_time => cur_trial.fixTime)
  cur_fix_df[!, :parcode] .= 1
  cur_fix_df[!, :trial] .= i  
  SimDataFixDf = vcat(SimDataFixDf, cur_fix_df, cols=:union)

  # "parcode","trial","rt","choice","item_left","item_right"
  cur_beh_df = DataFrame(:parcode => 1, :trial => i, :choice => cur_trial.choice, :rt => cur_trial.RT, :item_left => cur_trial.valueLeft, :item_right => cur_trial.valueRight)
  SimDataBehDf = vcat(SimDataBehDf, cur_beh_df, cols=:union)

end

CSV.write(dp * "sim_data_beh.csv", SimDataBehDf)
CSV.write(dp * "sim_data_fix.csv", SimDataFixDf)

# Create and save param_grid
# True params: d = 0.007, σ = 0.03, θ = .6

# ds = collect(.001:.001:.025);
# sigmas = collect(.01:.01:.25);
# thetas = collect(.12:.03:.85);
# sim_data_grid = allcombinations(DataFrame, "d" => ds, "sigma" => sigmas, "theta" => thetas);
# CSV.write(dp * "sim_data_grid.csv", sim_data_grid)

# ds = collect(.001:.001:.02);
# sigmas = collect(.01:.01:.2);
# thetas = collect(.27:.03:.85);
# sim_data_grid = allcombinations(DataFrame, "d" => ds, "sigma" => sigmas, "theta" => thetas);
# CSV.write(dp * "sim_data_grid2.csv", sim_data_grid)

ds = collect(.001:.001:.015);
sigmas = collect(.005:.005:.06);
thetas = collect(0:.1:1);
sim_data_grid = allcombinations(DataFrame, "d" => ds, "sigma" => sigmas, "theta" => thetas);
CSV.write(dp * "sim_data_grid3.csv", sim_data_grid)