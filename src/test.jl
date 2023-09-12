"""
#!/usr/bin/env julia
Copyright (C) 2023, California Institute of Technology

This file is part of addm_toolbox.

addm_toolbox is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

addm_toolbox is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with addm_toolbox. If not, see <http://www.gnu.org/licenses/>.

---

Module: test.jl
Author: Lynn Yang, lynnyang@caltech.edu

Testing functions in aDDM Toolbox.
"""

include("addm.jl")
include("addm_grid_search.jl")

arg1 = parse(Int, ARGS[1])
arg2 = parse(Int, ARGS[2])

result = aDDM_get_trial_likelihood(arg1, arg2)

println("Result: ", result)