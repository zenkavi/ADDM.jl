set -e
for grid_search_exec in thread seq
do
  sed -e "s/{GRID_SEARCH_EXEC}/$grid_search_exec/g" small_stepsize_recovery_job.batch | sbatch
done
