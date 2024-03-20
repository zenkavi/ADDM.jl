set -e
for grid_search_fn in thread floop floop2
do
  for grid_search_exec in thread seq
  do
    for trial_exec in thread seq
    do
      sed -e "s/{GRID_SEARCH_FN}/$grid_search_fn/g" -e "s/{GRID_SEARC_EXEC}/$grid_search_exec/g" -e "s/{TRIALS_EXEC}/$trials_exec/g" grid_search_benchmark_job.batch | sbatch
    done
  done
done