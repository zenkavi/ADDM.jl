

Can we do this through a config (file or inline julia) depending on available resources?

Thread for compute_trials_nll
MPI for grid_search

If no mpi then 
thread grid_search?
or is nested threading possible? 
  nest threads with @spawn? https://discourse.julialang.org/t/multithreading-for-nested-loops/36002


Which level of parallelization is more important? param_grid or across trials?