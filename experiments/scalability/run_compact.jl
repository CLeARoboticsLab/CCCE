# =============================================================================
# run_compact.jl -- compact expected-allocation LP on the same instances
#
# CompactLPHet solves Restricted-CCCE exactly with n variables (y_i = expected
# number of resources held by agent i) when (a) the yield cost is uniform per
# agent and (b) the coordinator objective depends only on expected costs. It
# is run on the SAME instances as run_mc.jl (same constants, seed, draw order)
# with the same timing protocol (end to end, per-n warm-up).
#
# Exactness check: Restricted-CCCE is re-solved (untimed) on every instance and
# the :fair objective values are compared. AC/Gini may differ slightly because
# the LP can have several optimal points.
#
# Usage (from the repository root):
#   julia --project=. experiments/scalability/run_compact.jl [--quick]
# Output: experiments/scalability/results/mc_compact_results[_quick].mat
# =============================================================================

using ChanceConstrainedCE
using Statistics: mean, median
using Random
using Printf
using MAT

const R         = 3
const C         = 10.0
const SIGMA     = 2.0
const ZALPHA    = 1.6449
const DELTA     = 3.0
const OBJECTIVE = :fair
const RHO_LO    = 1.0
const RHO_HI    = 6.0
const RHO_WARM  = 5.0
const SEED      = 2026
const RESULTS   = joinpath(@__DIR__, "results")
const GOOD      = ("LOCALLY_SOLVED", "OPTIMAL")

kw(sigma) = (zalpha = ZALPHA, sigma = sigma, unit = 1.0, objective = OBJECTIVE)
# closed form of the :fair objective at a given expected-cost vector
fair_obj(c) = sum(max(ci + DELTA, maximum(c)) for ci in c) - length(c) * DELTA

function run_compact(; ns = 3:8, n_trials = 100, seed = SEED)
    rows = NamedTuple[]
    rng = MersenneTwister(seed)
    for n in ns
        sigma = fill(SIGMA, n)
        CompactLPHet(R, n, C, DELTA, fill(RHO_WARM, n); kw(sigma)...)      # warm-up
        Sset, _ = ExclusiveJointSet(R, n)
        for trial in 1:n_trials
            ρ = RHO_LO .+ rand(rng, n) .* (RHO_HI - RHO_LO)

            local rc
            t_cmp = @elapsed (rc = CompactLPHet(R, n, C, DELTA, ρ; kw(sigma)...))
            rr = SearchCorrReducedLPHet(R, n, C, DELTA, UniformAgentYieldWeights(ρ, R), Sset; kw(sigma)...)

            st_cmp, st_red = string(rc.status), string(rr.status)
            ok = st_cmp in GOOD && st_red in GOOD
            push!(rows, (n = n, trial = trial, rho_min = minimum(ρ), rho_max = maximum(ρ),
                t_cmp = t_cmp, build_cmp = rc.buildTime, solve_cmp = rc.solverTime,
                AC_cmp = ok ? mean(rc.cvals) : NaN, GI_cmp = ok ? rc.giniScore : NaN,
                obj_cmp = ok ? fair_obj(rc.cvals) : NaN, obj_red = ok ? fair_obj(rr.cvals) : NaN,
                nElig = count(rc.canHold), status_cmp = st_cmp, status_red = st_red))
        end
        println("  n=$n done"); flush(stdout)
    end
    return rows
end

function report(rows)
    agree = count(r -> (r.status_cmp in GOOD) == (r.status_red in GOOD), rows)
    println("\nFeasibility agrees with Restricted-CCCE on $agree / $(length(rows)) instances")
    both = filter(r -> !isnan(r.obj_cmp), rows)
    gap = [abs(r.obj_cmp - r.obj_red) / max(1.0, abs(r.obj_red)) for r in both]
    @printf("Relative :fair objective gap over %d solved instances: max %.2e, median %.2e\n",
            length(both), maximum(gap), median(gap))
    @printf("\n%2s | %12s %12s %12s\n", "n", "t_cmp", "build", "solve")
    for n in sort(unique(r.n for r in rows))
        s = filter(r -> r.n == n && r.status_cmp in GOOD, rows)
        @printf("%2d | %12.3g %12.3g %12.3g\n", n, median(r.t_cmp for r in s),
                median(r.build_cmp for r in s), median(r.solve_cmp for r in s))
    end
    println("(median seconds over feasible instances)")
end

if abspath(PROGRAM_FILE) == @__FILE__
    quick = "--quick" in ARGS
    cfg = quick ? (ns = 3:6, n_trials = 10) : (ns = 3:8, n_trials = 100)
    rows = run_compact(; cfg...)
    report(rows)
    cols = Dict{String,Any}()
    for key in keys(rows[1])
        v = [getfield(r, key) for r in rows]
        cols[String(key)] = eltype(v) <: AbstractString ? v : Float64.(v)
    end
    out = joinpath(RESULTS, quick ? "mc_compact_results_quick.mat" : "mc_compact_results.mat")
    mkpath(RESULTS)
    matwrite(out, Dict{String,Any}("tbl" => cols, "n_list" => Float64.(collect(cfg.ns)),
                                    "n_trials" => Float64(cfg.n_trials)))
    println("Wrote $out")
end
