using Documenter
using Jimm

DocMeta.setdocmeta!(Jimm, :DocTestSetup, :(using Jimm); recursive = true)

makedocs(;
    modules = [Jimm],
    sitename = "Jimm.jl",
    authors = "Carroll Vance and contributors",
    repo = Documenter.Remotes.GitHub("csvance", "Jimm.jl"),
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://csvance.github.io/Jimm.jl",
        edit_link = "master",
        assets = String[],
    ),
    pages = [
        "Home" => "index.md",
        "Getting Started" => "getting_started.md",
        "Porting Backbones" => "porting.md",
        "Testing" => "testing.md",
        "API Reference" => [
            "Overview" => "api/index.md",
            "Models" => "api/models.md",
            "Layers" => "api/layers.md",
            "Interop" => "api/interop.md",
        ],
    ],
    checkdocs = :exports,
    doctest = false,
    warnonly = [:missing_docs],
)

deploydocs(;
    repo = "github.com/csvance/Jimm.jl",
    devbranch = "master",
    push_preview = false,
)
