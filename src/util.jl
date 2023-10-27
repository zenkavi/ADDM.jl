"""
    load_data_from_csv(expdataFileName, fixationsFileName; convertItemValues=nothing)

Load experimental data from two CSV files: an experimental data file and a
fixations file. Format expected for experimental data file: parcode, trial,
rt, choice, item_left, item_right. Format expected for fixations file:
parcode, trial, fix_item, fix_time.

# Arguments
- `expdataFileName`: String, name of experimental data file.
- `fixationsFileName`: String, name of fixations file.
  - `parcode`: Subject identifier
  - `trial`: Trial number
  - `fix_item`: Fixation location. 0 = transition, 1 = left, 2 = right,
  3 = latency.
  - `fix_time`: Fixation duration.

# Return
  A dict, indexed by subjectId, where each entry is a list of Trial
      objects.
"""
function load_data_from_csv(expdataFileName, fixationsFileName = nothing)
    
    # Load experimental data from CSV file.
    try 
        df = DataFrame(CSV.File(expdataFileName, delim=","))
    catch
        print("Error while reading experimental data file " * expdataFileName)
        return nothing
    end
    
    df = DataFrame(CSV.File(expdataFileName, delim=","))
    
    if (!("parcode" in names(df)) || !("trial" in names(df)) || !("rt" in names(df)) || !("choice" in names(df))
        || !("item_left" in names(df)) || !("item_right" in names(df)))
        throw(error("Missing field in experimental data file. Fields required: parcode, trial, rt, choice, item_left, item_right"))
    end
    
    data = Dict()
    subjectIds = unique(df.parcode)
    for subjectId in subjectIds
        data[subjectId] = Trial[]
        parcode_df = df[df.parcode .== subjectId, [:trial, :rt, :choice, :item_left, :item_right]]
        trialIds = unique(parcode_df.trial) 
        for trialId in trialIds
            trial_df = parcode_df[parcode_df.trial .== trialId, [:rt, :choice, :item_left, :item_right]]
            itemLeft = trial_df.item_left[1]
            itemRight = trial_df.item_right[1]
            push!(data[subjectId], Trial(choice = trial_df.choice[1], RT = trial_df.rt[1], valueLeft = itemLeft, valueRight = itemRight) ) 
            end
        end
    
    # Load fixation data from CSV file if specified.
    if fixationsFileName != nothing
      try
          df = DataFrame(CSV.File(fixationsFileName, delim=","))
      catch
          print("Error while reading fixations file " * fixationsFileName)
          return nothing
      end
    
      df = DataFrame(CSV.File(fixationsFileName, delim=","))
    
      if (!("parcode" in names(df)) || !("trial" in names(df)) || !("fix_item" in names(df)) || !("fix_time" in names(df)))
          throw(error("Missing field in fixations file. Fields required: parcode, trial, fix_item, fix_time"))
      end
      
      subjectIds = unique(df.parcode)
      for subjectId in subjectIds
          if !(subjectId in keys(data))
              continue
          end
          parcode_df = df[df.parcode .== subjectId, [:trial, :fix_item, :fix_time]]
          trialIds = unique(parcode_df.trial)
          for (t, trialId) in enumerate(trialIds)
              trial_df = parcode_df[parcode_df.trial .== trialId, [:fix_item, :fix_time]]
              dataTrial = Matrix(trial_df)
              data[subjectId][t].fixItem = trial_df.fix_item
              data[subjectId][t].fixTime = trial_df.fix_time
          end
      end
    end
    

    data = Dict(string(subjectId) => trials for (subjectId, trials) in data)

    return data
end