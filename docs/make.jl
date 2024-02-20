# push!(LOAD_PATH,"../src/")

# ENV["GKSwstype"] = "100"
using Documenter, ADDM

makedocs(
    sitename = "ADDM.jl",
    clean = true,
    format = Documenter.HTML(
        collapselevel = 2
    ),
    pages = [
        "Home" => "index.md",
        "Tutorials" => ["tutorials/01_getting_started.md",
                        "tutorials/02_empirical_data.md",
                        "tutorials/03_model_comparison.md",
                        "tutorials/04_custom_model.md"],
        "API Reference" => "apireference.md",
    ],
    doctestfilters = [r"[\s\-]?\d\.\d{6}e[\+\-]\d{2}"],
)

deploydocs(
    repo = "github.com/aDDM-Toolbox/ADDM.jl.git",
)