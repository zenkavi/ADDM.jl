set -e
for grid_search_exec in thread seq
do
  for trials_exec in thread seq
  do
    sed -e "s/{GRID_SEARCH_EXEC}/$grid_search_exec/g" -e "s/{TRIALS_EXEC}/$trials_exec/g" gs_floop_improvements_job.batch | sbatch
  done
done
