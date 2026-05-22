using Test
using Dates

# `using JimmCI` is already in runtests.jl. Reach into the submodule by
# its fully-qualified name; `_classify_pr_head` is intentionally unexported.
const _Jobs     = JimmCI.Jobs
const _classify = _Jobs._classify_pr_head

_pr(head_full, base_full) = Dict(
    "number" => 42,
    "title"  => "test pr",
    "head"   => Dict("sha" => "a"^40,
                     "repo" => Dict("full_name" => head_full)),
    "base"   => Dict("sha" => "b"^40,
                     "repo" => Dict("full_name" => base_full)),
)

@testset "Jobs" begin
    @testset "_classify_pr_head" begin
        @testset "same-repo PR" begin
            head_repo, is_fork = _classify(_pr("owner/Jimm.jl", "owner/Jimm.jl"))
            @test head_repo == "owner/Jimm.jl"
            @test is_fork === false
        end

        @testset "fork PR" begin
            head_repo, is_fork = _classify(_pr("contrib/Jimm.jl", "owner/Jimm.jl"))
            @test head_repo == "contrib/Jimm.jl"
            @test is_fork === true
        end

        @testset "missing head repo treated as non-fork" begin
            # GitHub returns head.repo == null on PRs whose source fork has
            # been deleted. Don't flag those as forks; downstream code can't
            # do anything useful with them anyway.
            pr = Dict("number" => 1,
                      "head" => Dict("sha" => "a"^40, "repo" => nothing),
                      "base" => Dict("sha" => "b"^40,
                                     "repo" => Dict("full_name" => "owner/Jimm.jl")))
            head_repo, is_fork = _classify(pr)
            @test head_repo === nothing
            @test is_fork === false
        end
    end

    @testset "Job constructor stores head_repo + is_fork" begin
        j = _Jobs.Job("a"^40, "b"^40, ["resnet"], false, "pr-1@aaaaaaaa",
                     _Jobs.PR_JOB;
                     pr_number  = 1, pr_title  = "t",
                     head_repo  = "contrib/Jimm.jl", is_fork = true,
                     created_at = DateTime(2026, 1, 1))
        @test j.head_repo == "contrib/Jimm.jl"
        @test j.is_fork === true

        master = _Jobs.Job("c"^40, "c"^40, ["resnet"], true, "master@cccccccc",
                          _Jobs.MASTER_JOB; created_at = DateTime(2026, 1, 1))
        @test master.head_repo === nothing
        @test master.is_fork === false
    end
end
