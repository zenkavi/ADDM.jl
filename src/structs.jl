"""
    Trial(choice, RT, valueLeft, valueRight)

# Arguments
## Required keyword arguments
- `choice`: either -1 (for left item) or +1 (for right item).
- `RT`: response time in milliseconds.
- `valueLeft`: value of the left item.
- `valueRight`: value of the right item.

## Optional 
- `RDV`: vector of RDV over time.
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

# Example

```julia
julia> t = Trial(choice = 1, RT = 2145, valueLeft = 1, valueRight = 3)
Trial(1, 2145, 1, 3, #undef, #undef, #undef, #undef, #undef)

julia> t.RT
2145

julia> t.uninterruptedLastFixTime
ERROR: UndefRefError: access to undefined reference
Stacktrace:
 [1] getproperty(x::Trial, f::Symbol)
   @ Base ./Base.jl:37
 [2] top-level scope
   @ REPL[4]:1

julia> t.uninterruptedLastFixTime = 189
189

julia> t
Trial(1, 2145, 1, 3, #undef, #undef, #undef, 189, #undef)
```

# Todo
## Tests
- Create a tmp Trial struct with only mandatory arguments
- Check it has the correct number elements
- Assign optional argument to the Trial struct
- Check that is assigned to the correct slot
"""
mutable struct Trial
    
    # Required components of a Trial
    # They are keyword arguments without defaults which makes them required
    choice::Number
    RT::Number
    valueLeft::Number
    valueRight::Number

    # Optional components
    fixItem::Vector{Number}
    fixTime::Vector{Number}
    fixRDV::Vector{Number}
    uninterruptedLastFixTime::Number
    RDV::Vector{Number}

    Trial(;choice, RT, valueLeft, valueRight) = new(choice, RT, valueLeft, valueRight)
end

"""
Implementation of simple attentional drift-diffusion model (aDDM), as described
by Kraijbich et al. (2010).

# Arguments:
- `d`: Number, parameter of the model which controls the speed of
    integration of the signal.
- `σ`: Number, parameter of the model, standard deviation for the
    normal distribution.
- `θ`: Float64 Traditionally between 0 and 1, parameter of the model which controls
    the attentional discounting. Default at 1 makes it a ddm.
- `barrier`: positive Int64, boundary separation in each direction from 0. Default at 1.
- `decay`: constant for linear barrier decay at each time step. Default at 0.
- `nonDecisionTime`: non-negative Number, the amount of time in
    milliseconds during which processes other than evidence accummulation occurs. 
    Default at 0.
- `bias`: Number, corresponds to the initial value of the relative decision value
    variable. Must be smaller than barrier.
- `params`: Tuple, parameters of the model. Order of parameters: d, σ, barrier, decay, 
    nonDecisionTime, bias

# Todo
- Tests
- Change decay parameter to function instead of scalar
"""
mutable struct aDDM
    
    d::Number
    σ::Number
    θ::Float64
    barrier::Number
    decay::Number
    nonDecisionTime::Number
    bias::Number

    function aDDM(;d, σ, θ = 1, barrier = 1, decay = 0, nonDecisionTime = 0, bias = 0.0)
      if barrier <= 0
        throw(ValueError("Error: barrier parameter must larger than zero."))
      elseif bias >= barrier
        throw(ValueError("Error: bias parameter must be smaller than barrier parameter."))
      end
        new(d, σ, θ, barrier, decay, nonDecisionTime, bias)
    end
end

"""
    FixationData(probFixLeftFirst, latencies, transitions, fixations; 
                 fixDistType="fixation")
    
# Arguments:
- `probFixLeftFirst`: Float64 between 0 and 1, empirical probability that
    the left item will be fixated first.
- `latencies`: Vector corresponding to the empirical distribution of
    trial latencies (delay before first fixation) in milliseconds.
- `transitions`: Vector corresponding to the empirical distribution
    of transitions (delays between item fixations) in milliseconds.
- `fixations`: Dict whose indexing is defined according to parameter
    fixDistType. Each entry is an array corresponding to the
    empirical distribution of item fixation durations in
    milliseconds.
- `fixDistType`: String, one of {'simple', 'difficulty', 'fixation'},
    determines how the fixation distributions are indexed. If
    'simple', fixation distributions are indexed only by type (1st,
    2nd, etc). If 'difficulty', they are indexed by type and by trial
    difficulty, i.e., the absolute value for the trial's value
    difference. If 'fixation', they are indexed by type and by the
    value difference between the fixated and unfixated items.
"""
struct FixationData
    
    probFixLeftFirst::Float64
    latencies::Vector{Number}
    transitions::Vector{Number}
    fixations::Dict 
    fixDistType::String 

    function FixationData(probFixLeftFirst, latencies, transitions, fixations; fixDistType="fixation")
        availableDistTypes = ["simple", "difficulty", "fixation"]
        if !(fixDistType in availableDistTypes)
            throw(RuntimeError("Argument fixDistType must be one of {simple, difficulty, fixation}"))
        end
        new(probFixLeftFirst, latencies, transitions, fixations, fixDistType)
    end
end