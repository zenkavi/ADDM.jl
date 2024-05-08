"""
Constructor for trial definitions that will contain choice RT and other trial info. 
  Not intended to be used alone but as part of `make_trial`

# Example

```julia
julia> t = ADDM.Trial()
ADDM.Trial(Dict{Symbol, Any}())

julia> t.choice = 1
1

julia> t.RT = 100
100

julia> propertynames(t)
KeySet for a Dict{Symbol, Any} with 2 entries. Keys:
  :choice
  :RT
```

"""
struct Trial
  properties::Dict{Symbol, Any}
end
Trial() = Trial(Dict{Symbol, Any}())

Base.getproperty(x::Trial, property::Symbol) = getfield(x, :properties)[property]
Base.setproperty!(x::Trial, property::Symbol, value) = getfield(x, :properties)[property] = value
Base.propertynames(x::Trial) = keys(getfield(x, :properties))

"""
    make_trial(;choice, RT)

# Arguments
## Required keyword arguments
- `choice`: either -1 (for left item) or +1 (for right item).
- `RT`: response time in milliseconds.

## Optional 
- `valueLeft`: value of the left item.
- `valueRight`: value of the right item.
- `fixItem`: list of items fixated during the trial in chronological
    order; 1 correponds to left, 2 corresponds to right, and any
    other value is considered a transition/blank fixation.
- `fixTime`: list of fixation durations (in milliseconds) in
    chronological order.
- `fixRDV`: list of Float64 corresponding to the RDV values at the end of
    each fixation in the trial.
- `uninterruptedLastFixTime`: Int64 corresponding to the duration, in
    milliseconds, that the last fixation in the trial would have if it
    had not been interrupted when a decision was made.
- `RDV`: vector of RDV over time.

# Example

```
julia> t = ADDM.make_trial(choice = 1, RT = 100)
ADDM.Trial(Dict{Symbol, Any}(:choice => 1, :RT => 100))

julia> propertynames(t)
KeySet for a Dict{Symbol, Any} with 2 entries. Keys:
  :choice
  :RT
```

"""
function make_trial(;choice::Number, RT::Number)
  
  # Required parameters
  t = Trial()

  ## Required definitions
  t.choice = choice
  t.RT = RT

  return t
end

"""
Constructor for model definitions that will contain model parameter and parameter value
  mapping. Not intended to be used alone but as part of `define_model`

# Example

```julia
julia> MyModel = ADDM.aDDM()
aDDM(Dict{Symbol, Any}())

julia> MyModel.d = 0.005
0.005

julia> MyModel.σ = .06
0.06

julia> MyModel
aDDM(Dict{Symbol, Any}(:σ => 0.06, :d => 0.005))
```
"""
struct aDDM
  properties::Dict{Symbol, Any}
end
aDDM() = aDDM(Dict{Symbol, Any}())

Base.getproperty(x::aDDM, property::Symbol) = getfield(x, :properties)[property]
Base.setproperty!(x::aDDM, property::Symbol, value) = getfield(x, :properties)[property] = value
Base.propertynames(x::aDDM) = keys(getfield(x, :properties))

"""
    define_model(d, σ, θ = 1, η = 0, barrier = 1, decay = 0, nonDecisionTime = 0, bias = 0.0)

Create attentional drift diffusion model with parameters described in 
  Krajbich et al. (2010).

# Arguments 
## Required parameters
- `d`: Number, parameter of the model which controls the speed of
    integration of the signal.
- `σ`: Number, parameter of the model, standard deviation for the
    normal distribution.

## Optional parameters
- `θ`: Float64 Traditionally between 0 and 1, parameter of the model which controls
    the attentional discounting. Default at 1 makes it a ddm.
- `η`: Float64 Additive attentional enhancement the attentional discounting. 
    Default at 0 makes it a ddm.
- `barrier`: positive Int64, boundary separation in each direction from 0. Default at 1.
- `decay`: constant for linear barrier decay at each time step. Default at 0.
- `nonDecisionTime`: non-negative Number, the amount of time in
    milliseconds during which processes other than evidence accummulation occurs. 
    Default at 0.
- `bias`: Number, corresponds to the initial value of the relative decision value
    variable. Must be smaller than barrier.


# Example

```julia
julia> MyModel = define_model(d = .006, σ = 0.05)
aDDM(Dict{Symbol, Any}(:nonDecisionTime => 0, :σ => 0.05, :d => 0.006, :bias => 0.0, :barrier => 1, :decay => 0, :θ => 1.0, :η => 0.0))

julia> propertynames(MyModel)
KeySet for a Dict{Symbol, Any} with 8 entries. Keys:
  :nonDecisionTime
  :σ
  :d
  :bias
  :barrier
  :decay
  :θ
  :η
````

"""
function define_model(;d::Number, σ::Number, θ::Float64 = 1.0, η::Float64 = 0.0, barrier::Number = 1, 
  decay::Number = 0, nonDecisionTime::Number = 0, bias::Number = 0.0)
  
  # Required parameters
  m = aDDM()

  ## Requires definitions
  m.d = d # drift rate
  m.σ = σ # sampling noise

  ## Has default value
  m.θ = θ # multiplicative attentional discounting
  m.η = η # additive attentional enhancement
  m.barrier = barrier # threshold
  m.decay = decay # barrier decay
  m.nonDecisionTime = nonDecisionTime 
  m.bias = bias # starting point bias

  return m
end