using Basinhopping
using Documenter

DocMeta.setdocmeta!(Basinhopping, :DocTestSetup, :(using Basinhopping); recursive=true)

makedocs(;
    modules=[Basinhopping],
    authors="Gabriel Matos <pygdfm@leeds.ac.uk> and contributors",
    repo="https://github.com/gamatos/Basinhopping.jl/blob/{commit}{path}#{line}",
    sitename="Basinhopping.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://gamatos.github.io/Basinhopping.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "List of functions" => "functions.md",
    ],
)

deploydocs(;
    repo="github.com/gamatos/Basinhopping.jl",
    devbranch="main",
)
