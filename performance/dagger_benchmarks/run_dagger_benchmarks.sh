set -e
for grid_search_fn in thread floop serial
do
  for compute_trials_fn in thread floop dagger serial
  do
    sed -e "s/{GRID_SEARCH_FN}/$grid_search_fn/g" -e "s/{COMPUTE_TRIALS_FN}/$compute_trials_fn/g" dagger_benchmarks_job.batch | sbatch
  done
done