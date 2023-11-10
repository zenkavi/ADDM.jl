# push!(LOAD_PATH,"../src/")

using Documenter, ADDM

# makedocs(sitename="ADDM.jl")

makedocs(
    sitename = "ADDM.jl",
    clean = true,
    format = Documenter.HTML(
        collapselevel = 1
    ),
    pages = [
        "Home" => "index.md",
        "Tutorials" => ["tutorials/getting_started.md"],
        "API Reference" => "apireference.md",
    ],
    doctestfilters = [r"[\s\-]?\d\.\d{6}e[\+\-]\d{2}"],
)