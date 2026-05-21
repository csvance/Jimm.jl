using Test
using JimmCI
using JimmCI.PathFilter

@testset "JimmCI" begin
    include("test_path_filter.jl")
    include("test_jobs.jl")
end
