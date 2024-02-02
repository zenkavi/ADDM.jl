# push!(LOAD_PATH,"../src/")

ENV["GKSwstype"] = "100"
using Documenter, ADDM

makedocs(
    sitename = "ADDM.jl",
    clean = true,
    format = Documenter.HTML(
        collapselevel = 2
    ),
    pages = [
        "Home" => "index.md",
        "Tutorials" => ["tutorials/getting_started.md",
                        "tutorials/empirical_data.md",
                        "tutorials/custom_model.md",
                        "tutorials/model_comparison.md"],
        "API Reference" => "apireference.md",
    ],
    doctestfilters = [r"[\s\-]?\d\.\d{6}e[\+\-]\d{2}"],
)

deploydocs(
    repo = "github.com/aDDM-Toolbox/ADDM.jl.git",
)