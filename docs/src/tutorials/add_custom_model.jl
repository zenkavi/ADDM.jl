# # Add custom model

# ## Create model with new parameter
# Define model with the standard parameters
MyCustomModel = ADDM.define_model(d = ..., σ = ...)

# Add new parameter. In this case the parameter is called `NewParameter` 
# and has a value of 3.
MyCustomModel.NewParameter = 3

# ## Define simulator

# Required: 
# Input: model aDDM object, valueLeft, valueRight, timeStep
# Output: Trial object
# Optional:
# Input: cutOff
# Output: RDV

# Required kwargs model, valueLeft, valueRight, timeStep, cutOff
# Others kwaargs can be added and passed to `simulator_args` in `simulate_data`
function my_simulate_trial(;model::aDDM, valueLeft::Number, valueRight::Number,
  timeStep::Number = 10.0, cutOff::Int64 = 20000)

  ...

  choice = RDV >= 0 ? 1 : -1
  RT = cutOff * timeStep
  trial = Trial(choice = choice, RT = RT, valueLeft = valueLeft, valueRight = valueRight)
  # trial.RDV = ...
  return trial
end

# ## Define likelihood computer 
function my_get_trial_likelihood(;model::aDDM, trial::Trial, timeStep::Number = 10, 
  approxStateStep::Number = 0.1)
  ...
  return likelihood
end
