using ChanceConstrainedCE
using Test
using Statistics: mean
using JuMP: MOI

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

    @testset "general API: small game" begin
        # chicken, as costs (1 = dare, 2 = yield)
        g = zeros(2, 2, 2)
        g[1, :, :] = [10 0; 7 2]
        g[2, :, :] = [10 7; 0 2]
        ce = solve_ccce(g)
        @test ce.status == MOI.OPTIMAL
        @test sum(ce.z) ≈ 1
        @test ce.objective_value ≈ 40 / 7 atol = 1e-8          # best CE beats both pure NE (7)
        @test Set(find_ccpne(g)) == Set([CartesianIndex(1, 2), CartesianIndex(2, 1)])
        @test solve_rr(g).objective_value ≈ 7
        cc = solve_ccce(g; alpha = 0.9, sigma = 1.0)
        @test cc.objective_value > ce.objective_value          # robustness costs efficiency
        @test all(c.dual >= -1e-9 for c in cc.constraints)
        @test solve_rr(g; alpha = 0.999, sigma = 3.0).status == :NO_CCPNE
        # infogain is the first-order value of the uncertainty (paper, Prop. 1):
        # dJ*/dσ_i = q(α) Σ_{c ∈ C_i} λ_c π_c
        h = 1e-6
        up = solve_ccce(g; alpha = 0.9, sigma = [1.0 + h, 1.0])
        fd = (up.objective_value - cc.objective_value) / h
        pred = cc.zalpha * sum(c.dual * c.π for c in cc.constraints if c.agent == 1)
        @test fd ≈ pred rtol = 1e-3
    end

    @testset "general API reproduces the paper solvers" begin
        # CC-CE / InfoGain game (SetC), aggregate objective
        r1, n1, γ1 = 2, 3, 2.0
        Cg = SetC(r1, n1, ones(n1); mult = γ1)
        m1 = 2^r1
        costs1 = zeros(n1, ntuple(_ -> m1, n1)...)
        for a in CartesianIndices(ntuple(_ -> m1, n1)), i in 1:n1
            costs1[i, Tuple(a)...] = J_def(i, collect(Tuple(a)), Cg)
        end
        σ1 = [56.0, 7.0, 40.0]
        old = SearchCorrLP(r1, n1, ones(n1), 1.0; zalpha = 1.2816, sigma = σ1, mult = γ1)
        new = solve_ccce(costs1; zalpha = 1.2816, sigma = σ1)
        @test new.objective_value ≈ old.score rtol = 1e-8

        # scalability game (exclusive resource assignment), :fair objective
        n2 = 4
        ρ = [1.5, 4.0, 5.5, 2.5]
        W = UniformAgentYieldWeights(ρ, r)
        aset = BuildRunwayActionSet(r)
        m2 = length(aset)
        costs2 = zeros(n2, ntuple(_ -> m2, n2)...)
        for a in CartesianIndices(ntuple(_ -> m2, n2)), i in 1:n2
            costs2[i, Tuple(a)...] = VertiportCostHet(i, collect(Tuple(a)), aset, r, c, W; unit = 1.0)
        end
        F = exclusive_support(r, n2)
        sig = fill(σ, n2)
        gen = (zalpha = z, sigma = sig, objective = :fair, Δ = Δ)
        red = SearchCorrReducedLPHet(r, n2, c, Δ, W; sigma = sig, kw...)
        @test solve_ccce(costs2; support = F, gen...).objective_value ≈ fair_obj(red.cvals, Δ) rtol = 1e-5
        rrf = HullLPHetOverF(r, n2, c, Δ, W; sigma = sig, kw...)
        rrx = HullLPHetOverX(r, n2, c, Δ, W; sigma = sig, kw...)
        gF = solve_rr(costs2; support = F, gen...)
        gX = solve_rr(costs2; gen...)
        @test length(gF.ccpne) == rrf.d
        @test length(gX.ccpne) == rrx.d
        @test gF.objective_value ≈ fair_obj(rrf.cvals, Δ) rtol = 1e-5
        @test gX.objective_value ≈ fair_obj(rrx.cvals, Δ) rtol = 1e-5
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
