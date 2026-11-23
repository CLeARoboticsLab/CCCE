using ChanceConstrainedCE
using Test
using Statistics: mean

const GOOD = ("LOCALLY_SOLVED", "OPTIMAL")
fair_obj(c, Δ) = sum(max(ci + Δ, maximum(c)) for ci in c) - length(c) * Δ

@testset "ChanceConstrainedCE" begin

    @testset "exclusive set F" begin
        for (r, n) in ((2, 3), (3, 4))
            S, m = ExclusiveJointSet(r, n)
            @test m == 2^r
            @test length(S) == n^r
            aset = BuildRunwayActionSet(r)
            # every resource has exactly one occupant in every profile of F
            @test all(all(==(1), NvVector(aset, collect(x), r)) for x in S)
        end
    end

    # scalability-paper setting, small n
    r, c, σ, z, Δ = 3, 10.0, 2.0, 1.6449, 3.0
    kw = (zalpha = z, unit = 1.0, objective = :fair)

    @testset "Restricted-CCCE = RR_F = Compact (feasible instance)" begin
        n = 4
        ρ = [1.5, 4.0, 5.5, 2.5]                    # max ρ ≥ z·σ ≈ 3.29 → feasible
        W = UniformAgentYieldWeights(ρ, r)
        sig = fill(σ, n)
        red = SearchCorrReducedLPHet(r, n, c, Δ, W; sigma = sig, kw...)
        rrf = HullLPHetOverF(r, n, c, Δ, W; sigma = sig, kw...)
        cmp = CompactLPHet(r, n, c, Δ, ρ; sigma = sig, kw...)
        @test string(red.status) in GOOD
        @test string(rrf.status) in GOOD
        @test string(cmp.status) in GOOD
        # Theorem 1: same feasible set → same optimal value
        @test fair_obj(red.cvals, Δ) ≈ fair_obj(rrf.cvals, Δ) rtol = 1e-5
        @test fair_obj(red.cvals, Δ) ≈ fair_obj(cmp.cvals, Δ) rtol = 1e-5
        # agents with ρ_i < z·σ never hold a resource (Lemma 1)
        @test all(cmp.y[i] < 1e-6 for i in 1:n if ρ[i] < z * σ)
    end

    @testset "infeasible when max ρ < z·σ" begin
        n = 3
        ρ = [1.2, 2.0, 3.0]
        W = UniformAgentYieldWeights(ρ, r)
        sig = fill(σ, n)
        @test_throws ErrorException HullLPHetOverF(r, n, c, Δ, W; sigma = sig, kw...)
        @test !(string(SearchCorrReducedLPHet(r, n, c, Δ, W; sigma = sig, kw...).status) in GOOD)
        @test !(string(CompactLPHet(r, n, c, Δ, ρ; sigma = sig, kw...).status) in GOOD)
    end

    @testset "general CC-CE search (PATH)" begin
        res = SearchCorr(1, 2, ones(2), 1.0; zalpha = 1.2816, sigma = [0.3, 0.3])
        x = res.primals[1:res.l]                    # joint-action probabilities
        @test string(res.status) == "MCP_Solved"
        @test sum(x) ≈ 1 atol = 1e-6
        @test all(x .>= -1e-8)

        # LP formulation of the same problem reaches the same optimal value
        lp = SearchCorrLP(1, 2, ones(2), 1.0; zalpha = 1.2816, sigma = [0.3, 0.3])
        @test string(lp.status) == "OPTIMAL"
        @test CalcJ(lp.primals[1:lp.l], lp.C, lp.m, lp.n) ≈ CalcJ(x, res.C, res.m, res.n) rtol = 1e-6
        @test length(lp.mu_ineq) == length(res.mu_ineq)
    end
end
