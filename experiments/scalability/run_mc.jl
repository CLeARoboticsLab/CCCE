# =============================================================================
# run_mc.jl -- Monte Carlo scalability benchmark (Figs. 2-3 of the paper)
#
# Exclusive resource-assignment game, r = 3 resources (m = 8 actions/agent),
# c = 10, sigma_i = 2, alpha = 0.95 (z = 1.6449), Delta = 3, :fair objective,
# rho_i ~ U(1, 6) i.i.d. per agent, 100 instances per n = 3..8, seed 2026.
#
# Methods (all solved with JuMP + Ipopt):
#   Restricted-CCCE  CC-CE LP over the exclusive set F (|F| = n^r)
#   RR_F             CC-PNE screening over F, then hull LP over the CC-PNEs found
#   RR_X             CC-PNE screening over X (|X| = m^n), then hull LP
#   Full-CCCE        CC-CE LP over X (n <= 5 only)
#
# Timing protocol (wall clock, end to end):
#   * t_* includes building the support set (F or X), computing every
#     incentive-constraint / cost coefficient, building the model, and the
#     solve. RR_* additionally include the CC-PNE screening (search_*), which
#     itself includes building F (RR_F) or enumerating X (RR_X).
#   * build_* / solve_* split the LP part into model construction and Ipopt.
#   * One untimed warm-up solve per n and method (rho_i = 5 for all agents)
#     runs before the 100 instances, so no instance carries JIT compilation.
#
# Infeasible instances: if max_i rho_i < z_alpha*sigma, no CC-CE (hence no
# CC-PNE) exists; every method then fails (status NO_CCPNE / LOCALLY_INFEASIBLE).
# Those instances are recorded here and excluded at plotting time, without
# resampling (20 of 600 with the default seed).
#
# Usage (from the repository root):
#   julia --project=. experiments/scalability/run_mc.jl            # paper setting (~1-1.5 h)
#   julia --project=. experiments/scalability/run_mc.jl --quick    # 10 instances, n = 3..6
# Output: experiments/scalability/results/mc_rho_noise_results[_quick].mat
# =============================================================================

using ChanceConstrainedCE
using Statistics: mean, median
using Random
using Printf
using MAT

const R         = 3
const C         = 10.0
const SIGMA     = 2.0
const ZALPHA    = 1.6449      # Phi^-1(0.95)
const KAPPA     = ZALPHA * SIGMA
const DELTA     = 3.0
const OBJECTIVE = :fair
const RHO_LO    = 1.0
const RHO_HI    = 6.0
const RHO_WARM  = 5.0         # all agents robust to holding and yielding -> CC-PNE exist
const SEED      = 2026
const RESULTS   = joinpath(@__DIR__, "results")
const GOOD      = ("LOCALLY_SOLVED", "OPTIMAL")

kw(sigma) = (zalpha = ZALPHA, sigma = sigma, unit = 1.0, objective = OBJECTIVE)

# --- one timed call per method; F / X construction is inside the timer ------
function time_red(n, W, sigma)
    local res, tF
    t = @elapsed begin
        tF = @elapsed ((Sset, _) = ExclusiveJointSet(R, n))
        res = SearchCorrReducedLPHet(R, n, C, DELTA, W, Sset; kw(sigma)...)
    end
    return res, t, tF
end

function time_rrf(n, W, sigma)
    res = nothing; tF = NaN
    t = @elapsed begin
        tF = @elapsed ((Sset, _) = ExclusiveJointSet(R, n))
        try
            res = HullLPHetOverF(R, n, C, DELTA, W, Sset; kw(sigma)...)
        catch e
            occursin("No CC-PNE", sprint(showerror, e)) || rethrow(e)
        end
    end
    return res, t, tF
end

function time_rrx(n, W, sigma)
    res = nothing
    t = @elapsed try
        res = HullLPHetOverX(R, n, C, DELTA, W; kw(sigma)...)
    catch e
        occursin("No CC-PNE", sprint(showerror, e)) || rethrow(e)
    end
    return res, t
end

function time_full(n, W, sigma)
    local res
    t = @elapsed (res = SearchCorrLPHet(R, n, C, DELTA, W; kw(sigma)...))
    return res, t
end

function run_mc(; ns = 3:8, n_trials = 100, full_nmax = 5, seed = SEED)
    rng = MersenneTwister(seed)
    rows = NamedTuple[]
    for n in ns
        sigma = fill(SIGMA, n)
        Fsize = n^R

        Wwarm = UniformAgentYieldWeights(fill(RHO_WARM, n), R)       # warm-up (untimed, no rng)
        time_red(n, Wwarm, sigma); time_rrf(n, Wwarm, sigma); time_rrx(n, Wwarm, sigma)
        n <= full_nmax && time_full(n, Wwarm, sigma)

        for trial in 1:n_trials
            ρ = RHO_LO .+ rand(rng, n) .* (RHO_HI - RHO_LO)
            W = UniformAgentYieldWeights(ρ, R)

            rr, t_red, tF_red = time_red(n, W, sigma)
            rf, t_rrf, tF_rrf = time_rrf(n, W, sigma)
            rx, t_rrx         = time_rrx(n, W, sigma)
            okf = rf !== nothing; okx = rx !== nothing

            t_full, AC_full, GI_full, status_full = NaN, NaN, NaN, "N/A"
            build_full, solve_full = NaN, NaN
            if n <= full_nmax
                ru, t_full = time_full(n, W, sigma)
                AC_full, GI_full, status_full = mean(ru.cvals), ru.giniScore, string(ru.status)
                build_full, solve_full = ru.buildTime, ru.solverTime
            end

            dF = okf ? rf.d : 0
            push!(rows, (n = n, trial = trial,
                rho_min = minimum(ρ), rho_max = maximum(ρ),
                Fsize = Fsize, PF = dF, ratio = dF / Fsize, dX = okx ? rx.d : 0, dF = dF,
                t_red = t_red, t_rrf = t_rrf, t_rrx = t_rrx, t_full = t_full,
                search_rrf = okf ? tF_rrf + rf.searchTime : NaN,
                search_rrx = okx ? rx.searchTime : NaN,
                tF_red = tF_red, tF_rrf = tF_rrf,
                build_red = rr.buildTime, solve_red = rr.solverTime,
                build_rrf = okf ? rf.buildTime : NaN, solve_rrf = okf ? rf.solverTime : NaN,
                build_rrx = okx ? rx.buildTime : NaN, solve_rrx = okx ? rx.solverTime : NaN,
                build_full = build_full, solve_full = solve_full,
                AC_red = mean(rr.cvals), GI_red = rr.giniScore,
                AC_rrf = okf ? mean(rf.cvals) : NaN, GI_rrf = okf ? rf.giniScore : NaN,
                AC_rrx = okx ? mean(rx.cvals) : NaN, GI_rrx = okx ? rx.giniScore : NaN,
                AC_full = AC_full, GI_full = GI_full,
                status_red = string(rr.status),
                status_rrf = okf ? string(rf.status) : "NO_CCPNE",
                status_rrx = okx ? string(rx.status) : "NO_CCPNE",
                status_full = status_full))
        end
        println("  n=$n done ($n_trials instances)"); flush(stdout)
    end
    return rows
end

function summarize(rows)
    @printf("\n%2s | %5s %6s | %10s %10s %10s %10s\n", "n", "valid", "failed",
            "Restricted", "RR_F", "RR_X", "Full")
    for n in sort(unique(r.n for r in rows))
        sub = filter(r -> r.n == n, rows)
        ok  = filter(r -> r.status_red in GOOD && r.status_rrf in GOOD && r.status_rrx in GOOD, sub)
        med(f) = isempty(ok) || all(isnan, f.(ok)) ? NaN : median(filter(!isnan, f.(ok)))
        @printf("%2d | %5d %6d | %10.3g %10.3g %10.3g %10.3g\n", n, length(ok), length(sub) - length(ok),
                med(r -> r.t_red), med(r -> r.t_rrf), med(r -> r.t_rrx), med(r -> r.t_full))
    end
    println("(median total time [s] over valid instances)")
end

function write_mat(rows, outfile; ns, n_trials, full_nmax)
    cols = Dict{String,Any}()
    for key in keys(rows[1])
        v = [getfield(r, key) for r in rows]
        cols[String(key)] = eltype(v) <: AbstractString ? v : Float64.(v)
    end
    mkpath(dirname(outfile))
    matwrite(outfile, Dict{String,Any}(
        "n_list" => Float64.(collect(ns)), "n_trials" => Float64(n_trials),
        "params" => Dict{String,Any}("r" => Float64(R), "c" => C, "sigma" => SIGMA,
            "zalpha" => ZALPHA, "kappa" => KAPPA, "delta" => DELTA,
            "rho_lo" => RHO_LO, "rho_hi" => RHO_HI, "full_nmax" => Float64(full_nmax),
            "seed" => Float64(SEED)),
        "tbl" => cols))
    println("Wrote $(length(rows)) rows to $outfile")
end

if abspath(PROGRAM_FILE) == @__FILE__
    quick = "--quick" in ARGS
    cfg = quick ? (ns = 3:6, n_trials = 10, full_nmax = 4) : (ns = 3:8, n_trials = 100, full_nmax = 5)
    println("MC scalability benchmark: r=$R, c=$C, sigma=$SIGMA, alpha=0.95, rho ~ U($RHO_LO,$RHO_HI), " *
            "n=$(collect(cfg.ns)), $(cfg.n_trials) instances/n, Full-CCCE for n<=$(cfg.full_nmax)" *
            (quick ? "  [QUICK]" : ""))
    rows = run_mc(; cfg...)
    summarize(rows)
    write_mat(rows, joinpath(RESULTS, quick ? "mc_rho_noise_results_quick.mat" : "mc_rho_noise_results.mat"); cfg...)
end
