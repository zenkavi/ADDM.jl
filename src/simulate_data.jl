"""
    aDDM_simulate_trial(model::aDDM, fixationData::FixationData, 
                        valueLeft::Number, valueRight::Number; timeStep::Number=10.0, 
                        numFixDists::Int64=3 , fixationDist=nothing, timeBins=nothing, 
                        cutOff::Number=100000)

Generate a DDM trial given the item values.

# Arguments
- `model`: aDDM object.
- `fixationData`: FixationData object.
- `valueLeft`: value of the left item.
- `valueRight`: value of the right item.
- `timeStep`: Number, value in milliseconds to be used for binning
    time axis.
- `numFixDists`: Int64, number of fixation types to use in the fixation
    distributions. For instance, if numFixDists equals 3, then 3
    separate fixation types will be used, corresponding to the 1st,
    2nd and other (3rd and up) fixations in each trial.
- `fixationDist`: distribution of fixations which, when provided, will be
    used instead of fixationData.fixations. This should be a dict of
    dicts of dicts, corresponding to the probability distributions of
    fixation durations. Indexed first by fixation type (1st, 2nd, etc),
    then by the value difference between the fixated and unfixated 
    items, then by time bin. Each entry is a number between 0 and 1 
    corresponding to the probability assigned to the particular time
    bin (i.e. given a particular fixation type and value difference,
    probabilities for all bins should add up to 1).
- `timeBins`: array containing the time bins used in fixationDist.

# Returns
- An Trial object resulting from the simulation.

# Todo
- Response time cut off
- Additional fixation data distributions
"""
function aDDM_simulate_trial(;model::aDDM, fixationData::FixationData, valueLeft::Number, valueRight::Number, 
                        timeStep::Number=10.0, numFixDists::Int64=3 , fixationDist=nothing, 
                        timeBins=nothing, cutOff::Number=100000)
    
    fixUnfixValueDiffs = Dict(1 => valueLeft - valueRight, 2 => valueRight - valueLeft)
    
    fixItem = Number[]
    fixTime = Number[]
    fixRDV = Number[]

    RDV = model.bias
    trialTime = 0
    choice = 0
    tRDV = Number[RDV]
    RT = 0
    uninterruptedLastFixTime = 0
    
    # Sample and iterate over the latency for this trial.
    latency = rand(fixationData.latencies)
    remainingNDT = model.nonDecisionTime - latency
    for t in 1:Int64(latency ÷ timeStep)
        # Sample the change in RDV from the distribution.
        RDV += rand(Normal(0, model.σ))
        push!(tRDV, RDV)

        # If the RDV hit one of the barriers, the trial is over.
        if abs(RDV) >= model.barrier
            choice = RDV >= 0 ? -1 : 1
            push!(fixRDV, RDV)
            push!(fixItem, 0)
            push!(fixTime, t * timeStep)
            trialTime += t * timeStep
            RT = trialTime
            uninterruptedLastFixTime = latency
            trial = Trial(choice = choice, RT = RT, valueLeft = valueLeft, valueRight = valueRight)
            trial.fixItem = fixItem 
            trial.fixTime = fixTime 
            trial.fixRDV = fixRDV
            trial.uninterruptedLastFixTime = uninterruptedLastFixTime
            trial.RDV = tRDV
            return trial
        end
    end

    # Add latency to this trial's data
    push!(fixRDV, RDV)
    push!(fixItem, 0)
    push!(fixTime, latency - (latency % timeStep))
    trialTime += latency - (latency % timeStep)

    fixNumber = 1
    prevFixItem = -1
    currFixLocation = 0
    decisionReached = false

    while true
        if currFixLocation == 0
            # This is an item fixation; sample its location.
            if prevFixItem == -1
                # Sample the first item fixation for this trial.
                currFixLocation = rand(Bernoulli(1 - fixationData.probFixLeftFirst)) + 1
            elseif prevFixItem in [1, 2]
                currFixLocation = abs(3 - prevFixItem)
            end
            prevFixItem = currFixLocation

            # Sample the duration of this item fixation.
            if fixationDist === nothing
                if fixationData.fixDistType == "simple"
                    currFixTime = rand(reduce(vcat, fixationData.fixations[fixNumber]))
                elseif fixationData.fixDistType == "difficulty" # maybe add reduce() like in simple
                    valueDiff = abs(valueLeft - valueRight)
                    currFixTime = rand(fixationData.fixations[fixNumber][valueDiff])
                elseif fixationData.fixDistType == "fixation"
                    valueDiff = fixUnfixValueDiffs[currFixLocation]
                    currFixTime = rand(fixationData.fixations[fixNumber][valueDiff])
                end
            else 
                # TODO
                throw(error("I HAVE NOT CODED THIS PART JUST YET"))
            end

            if fixNumber < numFixDists
                fixNumber += 1
            end

        else
            # This is a transition.
            currFixLocation = 0
            #Sample the duration of this transition.
            currFixTime = rand(fixationData.transitions)
        end

        # Iterate over the remaining non-decision time
        if remainingNDT > 0
            for t in 1:Int64(remainingNDT ÷ timeStep)
                # Sample the change in RDV from the distribution.
                RDV += rand(Normal(0, model.σ))
                push!(tRDV, RDV)

                # If the RDV hit one of the barriers, the trial is over.
                if abs(RDV) >= model.barrier
                    choice = RDV >= 0 ? -1 : 1
                    push!(fixRDV, RDV)
                    push!(fixItem, currFixLocation)
                    push!(fixTime, t * timeStep)
                    trialTime += t * timeStep
                    RT = trialTime
                    uninterruptedLastFixTime = currFixTime
                    decisionReached = true
                    break
                end
            end
        end

        if decisionReached
            break
        end

        remainingFixTime = max(0, currFixTime - max(0, remainingNDT))
        remainingNDT -= currFixTime

        # Iterate over the duration of the current fixation.
        for t in 1:Int64(remainingFixTime ÷ timeStep)
            # We use a distribution to model changes in RDV
            # stochastically. The mean of the distribution (the change
            # most likely to occur) is calculated from the model
            # parameters and from the values of the two items.
            if currFixLocation == 0
                μ = 0
            elseif currFixLocation == 1
                μ = model.d * ( (valueLeft + model.η) - (model.θ * valueRight))
            elseif currFixLocation == 2
                μ = model.d * ((model.θ * valueLeft) - (valueRight + model.η))
            end

            # Sample the change in RDV from the distribution.
            RDV += rand(Normal(μ, model.σ))
            push!(tRDV, RDV)

            # If the RDV hit one of the barriers, the trial is over.
            if abs(RDV) >= model.barrier
                choice = RDV >= 0 ? -1 : 1
                push!(fixRDV, RDV)
                push!(fixItem, currFixLocation)
                push!(fixTime, t * timeStep)
                trialTime += t * timeStep
                RT = trialTime
                uninterruptedLastFixTime = currFixTime
                decisionReached = true
                break
            end
        end

        if decisionReached
            break
        end

        # Add fixation to this trial's data.
        push!(fixRDV, RDV)
        push!(fixItem, currFixLocation)
        push!(fixTime, currFixTime - (currFixTime % timeStep))
        trialTime += currFixTime - (currFixTime % timeStep)

    end

    trial = Trial(choice = choice, RT = RT, valueLeft = valueLeft, valueRight = valueRight)
    trial.fixItem = fixItem 
    trial.fixTime = fixTime 
    trial.fixRDV = fixRDV
    trial.uninterruptedLastFixTime = uninterruptedLastFixTime
    trial.RDV = tRDV
    return trial
end

"""
    DDM_simulate_trial(model::aDDM, valueLeft::Number, valueRight::Number; timeStep::Number = 10.0, 
                       cutOff::Int64 = 20000)

Generate a DDM trial given the item values.

# Arguments
- `model`: aDDM object.
- `valueLeft`: value of the left item.
- `valueRight`: value of the right item.
- `timeStep`: Number, value in milliseconds to be used for binning the
    time axis.
- `cutOff`: Number, value in milliseconds to be used as a cap if trial
    response time is too long.

    # Returns
- A Trial object resulting from the simulation.
"""
function DDM_simulate_trial(;model::aDDM, valueLeft::Number, valueRight::Number,
                            timeStep::Number = 10.0, cutOff::Int64 = 20000)
    
    RDV = model.bias
    elapsedNDT = 0
    tRDV = Vector{Number}(undef, cutOff)
    valueDiff = model.d * (valueLeft - valueRight)

    for time in 0:cutOff-1
        tRDV[time + 1] = RDV

        # If the RDV hit one of the barriers, the trial is over.
        if abs(RDV) >= model.barrier
            choice = RDV >= 0 ? -1 : 1
            RT =  time * timeStep
            trial = Trial(choice = choice, RT = RT, valueLeft = valueLeft, valueRight = valueRight)
            trial.RDV = tRDV[1:time + 1]
            return trial
        end

        # If the response time is higher than the cutoff, the trial is over.
        if time * timeStep >= cutOff
            choice = RDV >= 0 ? 1 : -1
            RT =  time * timeStep
            trial = Trial(choice = choice, RT = RT, valueLeft = valueLeft, valueRight = valueRight)
            trial.RDV = tRDV[1:time + 1]
            return trial
        end

        # Sample the change in RDV from the distribution.
        if elapsedNDT < (model.nonDecisionTime ÷ timeStep)
            μ = 0
            elapsedNDT += 1
        else
            μ = valueDiff
        end

        RDV += rand(Normal(μ, model.σ))
    end

    choice = RDV >= 0 ? 1 : -1
    RT = cutOff * timeStep
    trial = Trial(choice = choice, RT = RT, valueLeft = valueLeft, valueRight = valueRight)
    trial.RDV = tRDV
    return trial
end

"""
    simulate_trial(model::aDDM, stimuli, simulator_fn, simulator_args = (timeStep = 10.0, cutOff = 20000))

Simulate data using the model for the given stimuli.

# Arguments
- `model`: aDDM object.
- `stimuli`: Named tuple with `valueLeft` and `valueRight` specifying the values of options.
- `simulator_fn`: Name of the function that simulates a trial for the given model.
- `simulator_args`: Named tuple containing kwargs that should be fed to `simulator_fn`

# Returns
- Vector of Trial objects containing simulated data.
"""
function simulate_data(model::aDDM, stimuli, simulator_fn, simulator_args = (timeStep = 10.0, cutOff = 20000))
  
   # Check stimuli has the valueLeft and valueRight
  if !(:valueLeft in keys(stimuli))
    # throw(RuntimeError("valueLeft not specified in stimuli"))
    error("valueLeft not specified in stimuli")
  end

  if !(:valueRight in keys(stimuli))
    # throw(RuntimeError("valueRight not specified in stimuli"))
    error("valueRight not specified in stimuli")
  end

  # Extract valueLeft and valueRight from stimuli
  valueLefts = stimuli.valueLeft
  valueRights = stimuli.valueRight

  # Feed the model and the stimuli to the simulator function
  n = length(valueLefts) # length of stimuli
  SimData = Vector{Trial}(undef, n)

  # how does this define how many threads are used?
  # afaiu this needs to be specific *before* julia is started e.g. by `julia --threads 4`
  @threads for i in 1:n 

    # The `simulator_args...` notation maps the NamedTuple to the kwargs of the simulator_fn
    # The named tuple `simulator_args` does not have to have all the kwargs
    # Additional kwargs can be specified before or after `simulator_args`
    SimData[i] = simulator_fn(;model = model, valueLeft = valueLefts[i], valueRight = valueRights[i], simulator_args...)
  end

  return SimData

end