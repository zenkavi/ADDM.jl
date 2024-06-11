# Parallelization

As detailed in the [tutorial on likelihood computation](https://addm-toolbox.github.io/ADDM.jl/dev/tutorials/02_likelihood_computation/) ADDM.jl uses likelihood as the optimization criterion and grid computation as the optimization algorithm. To decrease computation time ADDM.jl relies on embarassingly parallel data parallelization based on [FLoops.jl](https://github.com/JuliaFolds/FLoops.jl) on both of these levels.

Parallelization at the trial likelihood computation level depends on most models' assumption that each trial is independent from each other[^1]. When this assumption is true, trials likelihoods can be computed in parallel. By default, ADDM.jl uses multithreading to do this using the `sequential_model = false`, or `compute_trials_exec = ThreadedEx()` arguments in `ADDM.grid_search()` or `ADDM.compute_trials_nlls()`.  

At the grid computation level, the sum of negative log likelihoods for each parameter combination can be computed in parallel. ADDM.jl uses multithreading for this as well via the `grid_search_exec = ThreadedEx()` argument to `ADDM.grid_search()`.  

See the [Additional resources](#additional-resources) section for pointers on how to use other distributed computing methods with ADDM.jl.    

## Local computations on your own machine

By default, ADDM.jl will try to use all the threads available to the Julia kernel it is running in when using `ADDM.grid_search()` or `ADDM.compute_trials_nlls()`. Importantly, the number of threads for the Julia kernel will default to 1 unless specified by the user when starting the Julia session. The number of threads can be specified using the `--threads` argument: 

```sh
export NUM_THREADS=4
export JPROJ_PATH='~/ADDM.jl'

julia --project=$JPROJ_PATH --threads $NUM_THREADS
```

You don't need to specify anything else for `ADDM.grid_search()` or `ADDM.compute_trials_nlls()`. They will work with all the available resources.

To estimate parameters locally for a dataset with multiple participants you can then loop through each subject's data. An example of this is provided in a [previous tutorial](https://addm-toolbox.github.io/ADDM.jl/dev/tutorials/03_empirical_data/#Grid-search)

## Remote computations on an HPC

If you have access to a high performance computing cluster, you can submit separate jobs to estimate the best fitting parameters for each participant. Here we'll detail an example workflow and highlight some points to keep in mind when modifying this for your purposes.   

The workflow consists of three parts:  
1. Julia script that defines the steps for parameter estimation for a single participant  
2. Job script submitted to the scheduler that runs the Julia script for a single participant  
3. Shell script that loops through participant numbers and submits separate jobs  

### Julia script

Below we provide code snippets for an example script, `estimate_parameters.jl`, that estimates the best fitting parameters for a single subject. The subject identifier, likelihood function arguments timestep and statestep and the path that contains the data for each participant is specified as positional arguments to this script. This script will be run using the job script described in the next section.  

This script:  

- Expects paths: `/home/data/` and `/home/ADDM.jl`  
- Expects files: `/home/data/sub-{SUBNUM}_beh.csv`, `/home/data/sub-{SUBNUM}_fix.csv`, `/home/data/param_space.csv`  
- Expects arguments: subnum, timestep, statestep, data path
- Outputs files: `/home/data/sub-{SUBNUM}_dt-{TIMESTEP}_dx-{STATESTEP}_mle.csv`, `/home/data/sub-{SUBNUM}_dt-{TIMESTEP}_dx-{STATESTEP}_trial_posteriors.csv`, `/home/.out/estimate_parameters_sub-{SUBNUM}_dt-{TIMESTEP}_dx-{STATESTEP}_%j.out` (path specified in the job script below)  

#### Part 1: Initialize arguments

In the first part we import the necessary packages and write a function that parses the input arguments to the script. This parsing function is called to assign the arguments to objects within the Julia session.

```julia
using ADDM, ArgParse, Base.Threads, CSV, DataFrames, Dates

#########################
# Usage
#########################

# export NUM_THREADS=4
# export JPROJ_PATH="/home/ADDM.jl"
# export DATA_PATH="/home/data/"

# julia --project=$JPROJ_PATH --threads $NUM_THREADS estimate_parameters.jl {SUBNUM} {TIMESTEP} {STATESTEP} $DATA_PATH

#########################
# Part 1: Initialize arguments
#########################

function parse_commandline()
  s = ArgParseSettings()

  @add_arg_table! s begin
      "subnum"
          help = "subject number"
          arg_type = String
          required = true
      "timestep"
          help = "timestep for likelihood_fn"
          arg_type = String
          required = true
      "statestep"
          help = "statestep for likelihood_fn"
          arg_type = String
          required = true
      "dp"
          help = "data path"
          arg_type = String
          required = true
  end

  parsed_args = parse_args(s)
  subnum = parsed_args["subnum"]
  timestep = parsed_args["timestep"]
  statestep = parsed_args["statestep"]
  dp = parsed_args["dp"]

  return subnum, timestep, statestep, dp
end

println("Parsing arguments...")
flush(stdout)
subnum, timestep, statestep, dp = parse_commandline()

timestep = parse(Float64, timestep)
statestep = parse(Float64, statestep)
```

#### Part 2: Output saving function

In the second part we define a function that will be called at the end of the script to save the desired outputs of the parameter estimation. In this case, the function will save the MLE and the trial posteriors for all parameter combinations for a single subject. The file names will be determined by the input arguments to the function (e.g. `sub-9_dt-10_dx-0.01_mle.csv`)

Note also the `println` and the `flush` commands that will update the job output file (e.g. `estimate_parameters_sub-9_dt-10_dx-0.01_%j.out`) to provide updates on the progress of the job.

```julia
#########################
# Part 2: Output saving function
#########################

function save_output(output, subnum, timestep, statestep, dp)

  base_path = dp * "sub-" * subnum * "_dt-" * string(timestep) * "_dx-" * string(statestep) * "_"
  
  mle_path = base_path * "mle.csv"
  CSV.write(mle_path, DataFrame(output[:mle]))

  
  trial_posteriors_df = DataFrame()
  for (k,v) in output[:trial_posteriors]
    cur_df = DataFrame(Symbol(i) => j for (i, j) in pairs(v))

    rename!(cur_df, :first => :trial_num, :second => :posterior)

    # Unpack parameter info
    for (a, b) in pairs(k)
      cur_df[!, a] .= b
    end

    # Change type of trial num col to sort by
    cur_df[!, :trial_num] = [parse(Int, (String(i))) for i in cur_df[!,:trial_num]]

    sort!(cur_df, :trial_num)

    # trial_posteriors_df = vcat(trial_posteriors_df, cur_df, cols=:union)
    append!(trial_posteriors_df, cur_df, cols=:union)
  end

  trial_posteriors_path = base_path * "trial_posteriors.csv"
  CSV.write(trial_posteriors_path, trial_posteriors_df)

end
```

#### Part 3: Read in data and parameter space

In this section we read in the data, both behavioral and fixations, from the data path (`dp`) for a specific subject specified in the arguments to the script and parsed in the first part of the script. If your data is not saved in individual files for each subject, you could read in all your data and subset only the relevant datapoints in this section.

We also read in a file that specified the parameter space (i.e. the combinations of parameters) the grid search will compute the sum of negative log likelihoods for. This will be done using multithreaded by default. Your parameter space does not have to be read in from a CSV. It could be defined in this section in whatever way you prefer, as long as the `param_grid` used by `ADDM.grid_search()` is shaped into a Vector of NamedTuples eventually.

Finally, we specify the additional arguments for `ADDM.grid_search()`.

```julia
#########################
# Part 3: Read in data and parameter space
#########################

# Read in data
println("Reading in data...")
flush(stdout)

data = ADDM.load_data_from_csv(dp * "sub-" * subnum * "_beh.csv", dp * "sub-" * subnum * "_fix.csv");
data = data["sub-" * subnum]; 

# Read in parameter space
fn = dp * "param_space.csv";
tmp = DataFrame(CSV.File(fn, delim=","));
tmp[!,"likelihood_fn"] .= "ADDM.aDDM_get_trial_likelihood";
param_grid = NamedTuple.(eachrow(tmp));

my_likelihood_args = (timeStep = timestep, stateStep = statestep);
fixed_params = Dict(:η=>0.0, :barrier=>1, :decay=>0, :nonDecisionTime=>200, :bias=>0.0);

```

#### Part 4: Run grid_search

In this section we run the grid computation for the data using the parameter space. Note that we set `return_trial_posteriors` as `true` because the output saving function defined in part 2 will be expecting this.  

We can add additional information to the output too. Here we add the computation and number of threads used for the computation.   

```julia
#########################
# Part 4: Run grid_search
#########################

println("Starting grid_search at "* string(now()))
flush(stdout)

t1 = now();
output = ADDM.grid_search(data, param_grid, ADDM.aDDM_get_trial_likelihood, fixed_params, likelihood_args = my_likelihood_args, return_trial_posteriors = true, return_model_posteriors = true);
t2 = now();

comp_time = t2-t1;
output[:mle][:comp_time] = comp_time
output[:mle][:nthreads] = nthreads()
```

#### Part 5: Save output

In this last section we call the previously defined output saving function to have a record of the best fitting parameters and other outputs we are interested in before the job terminates.

```julia
#########################
# Part 5: Save output
#########################

println("Saving output...")
flush(stdout)

save_output(output, subnum, timestep, statestep, dp)
```

### Job script

Here we provide code snippets for an example script, `estimate_parameters_job.batch`, that submits a job to run `estimate_parameters.jl` for a single subject. The subject identifier, likelihood function arguments timestep and statestep are specified as by the shell script described in the next section.   

This script:  

- Expects paths: `/home/data/`, `/home/ADDM.jl`, `/home/.out/`, `/home/.err/`
- Expects modules: `julia/1.10.2`
- Expects files: `estimate_parameters.jl`  
- Expects arguments: subnum, timestep, statestep (these will be fed in from shell script described below)
- Outputs files: `/home/.out/estimate_parameters_sub-{SUBNUM}_dt-{TIMESTEP}_dx-{STATESTEP}_%j.out`, `/home/.err/estimate_parameters_sub-{SUBNUM}_dt-{TIMESTEP}_dx-{STATESTEP}_%j.err`

```sh
#!/bin/bash

#SBATCH --time=3:00:00 # walltime
#SBATCH --job-name=estimate_parameters_sub-{SUBNUM}_dt-{TIMESTEP}_dx-{STATESTEP}
#SBATCH --nodes=1 # number of nodes
#SBATCH --ntasks=1 # number of tasks
#SBATCH --cpus-per-task=8 # cores per task
#SBATCH --mem-per-cpu=8G # memory per CPU core
#SBATCH --output=/home/.out/estimate_parameters_sub-{SUBNUM}_dt-{TIMESTEP}_dx-{STATESTEP}_%j.out
#SBATCH --error=/home/.err/estimate_parameters_sub-{SUBNUM}_dt-{TIMESTEP}_dx-{STATESTEP}_%j.err

export NUM_THREADS=8
export DATA_PATH='/home/data/'
export JPROJ_PATH='/home/ADDM.jl'

module load julia/1.10.2 
julia --project=$JPROJ_PATH -e 'import Pkg; Pkg.instantiate()'
julia --project=$JPROJ_PATH --threads $NUM_THREADS estimate_parameters.jl {SUBNUM} {TIMESTEP} {STATESTEP} $DATA_PATH
```

### Shell script

`estimate_parameters.sh`

Here we provide code snippets for an example script, `estimate_parameters.sh`, that loops through and replaces values for the arguments  in `estimate_parameter_job.batch` (subject identifier, timestep, and statestep) and submits a separate job for each combination of the arguments. To estimate the best fitting parameters for each subject in separate jobs (and for two different statesteps) you only need to run this script (`sh estimate_parameter.sh`)

This script:  

- Expects files: `estimate_parameters_job.batch`  

```sh
set -e
for timestep in 10
do
  for statestep in 0.01 0.1
  do
    for subnum in 1 2 3 4 5 6 7 8 9 10 
    do
      sed -e "s/{TIMESTEP}/$timestep/g" -e "s/{STATESTEP}/$statestep/g" -e "s/{SUBNUM}/$subnum/g" estimate_parameters_job.batch | sbatch
    done
  done
done
```

## Additional resources

Parallelization and distributed computing requires careful thinking of concurrency and data storage. The workflows we provide here have been accurate in our testing. Additional gains in performance are possible but have not been tested extensively and therefore not built-in to ADDM.jl for now. For interested and more experienced users we recommend the following resources:  

- [This tutorial](https://enccs.github.io/julia-for-hpc/) provides an excellent overview of parallel computing options in Julia, with some background on parallel computing concepts as well. It includes pointers for `MPI.jl` and `Dagger.jl` (similar to Dask in Python) for distribited computing that can implement hierarchical structures.  
- If you're interested in more advanced distributed computing in HPC setting [this discussion might be helpful](https://discourse.julialang.org/t/is-clustermanagers-jl-maintained-or-how-to-do-multi-node-calculations-in-julia/110050/24)


[^1]: This is not true for models with sequential dependence between the trials, such as the reinforcement learning DDM (Fontanesi, L. et al. (2019)., Pedersen, M. L., Frank, M. J., & Biele, G. (2017)). Such models can still be specified within ADDM.jl but trial likelihood computation cannot be parallelized. For models with such dependence change the `sequential_model = false`, or `compute_trials_exec = ThreadedEx()` arguments in `ADDM.grid_search()` or `ADDM.compute_trials_nlls()`.