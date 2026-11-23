# =============================================================================
# search_corr_lp.jl
#
# SearchCorrLP: the same CC-CE problem as SearchCorr (search_corr.jl), solved
# directly as a linear program with JuMP instead of as an MCP with PATH.
#
#   min_x  Σ_a x_a Σ_i J̄_i(a)                       (CalcJ: expected total cost)
#   s.t.   Σ_a x_a = 1,  x ≥ 0
#          for every (i, a_i, ā_i ≠ a_i):            (CalcH, same order)
#            Σ_{a: a_i = a_i} x_a [J̄_i(ā_i, a_-i) − J̄_i(a)] − z_α σ_eff p_i(a_i) ≥ 0
#
# SearchCorr's problem is exactly this LP. The fairness auxiliaries (v, w with
# T1/T2/T3) appear only in its constraints, never in its objective, so they do
# not affect x. SearchCorr compiles a symbolic KKT system on every call, which
# costs ~20 s per solve; this LP solves in milliseconds.
#
# The returned named tuple uses SearchCorr's layout, so every downstream helper
# (GetCEConstraintInfo, TopKCEBy*, ...) works unchanged:
#   primals   = [x; v; w]     (v, w set to a feasible point of T1-T3)
#   ineq_vals = [x; CE rows; T1; T2; T3]   (BuildCorrConstraintMap order)
#   mu_ineq   = [reduced costs of x ≥ 0; CE duals; zeros for T1-T3]
# Duals are the LP multipliers of the ≥ constraints (nonnegative for Min).
#
# With a simplex solver (default GLPK) the solution is a vertex. When the LP has
# several optimal x or several optimal duals, SearchCorr (PATH) and SearchCorrLP
# can return different, equally optimal points, and anything computed from x or
# μ (realized costs, InfoGain rankings) can then differ.
# =============================================================================

function SearchCorrLP(r, n, λ, Δ;
                      zalpha,
                      sigma,
                      zero_sigma_ce_keys = Set{Tuple{Int,Int,Int}}(),
                      sigma_scale_keys   = Dict{Tuple{Int,Int,Int},Float64}(),
                      mult = 2.0,
                      optimizer = GLPK.Optimizer,
                      verbose = false)

    C = SetC(r, n, λ; mult = mult)
    n = blocksize(C)[1]
    m = size(C[Block(1)])[1]
    l = m^n
    CI = CartesianIndices(ntuple(_ -> m, n))     # same column-major order as reshape(x, m, …, m)
    σi(i) = isa(sigma, Number) ? sigma : sigma[i]

    model = Model(optimizer)
    verbose || set_silent(model)

    local x, ce_refs, Jtot
    buildTime = @elapsed begin
        @variable(model, x[1:l] >= 0)
        @constraint(model, sum(x) == 1)

        Jik = [J_def(i, collect(Tuple(CI[k])), C) for i in 1:n, k in 1:l]
        Jtot = vec(sum(Jik; dims = 1))
        @objective(model, Min, sum(Jtot[k] * x[k] for k in 1:l))

        ce_refs = ConstraintRef[]
        for i in 1:n, ai in 1:m
            ks = [k for k in 1:l if CI[k][i] == ai]
            for aibar in 1:m
                aibar == ai && continue
                key = (i, ai, aibar)
                σeff = haskey(sigma_scale_keys, key) ? σi(i) * sigma_scale_keys[key] :
                       key in zero_sigma_ce_keys    ? zero(σi(i)) : σi(i)
                expr = zero(AffExpr)
                for k in ks
                    a_dev = collect(Tuple(CI[k])); a_dev[i] = aibar
                    add_to_expression!(expr, J_def(i, a_dev, C) - Jik[i, k] - zalpha * σeff, x[k])
                end
                push!(ce_refs, @constraint(model, expr >= 0))
            end
        end
    end

    solverTime = @elapsed optimize!(model)
    status = termination_status(model)
    ok = has_values(model)

    xv = ok ? value.(x) : fill(NaN, l)
    ce_vals = ok ? value.(ce_refs) : fill(NaN, length(ce_refs))
    has_d = ok && has_duals(model)
    mu_x  = has_d ? reduced_cost.(x) : fill(NaN, l)
    mu_ce = has_d ? dual.(ce_refs) : fill(NaN, length(ce_refs))

    # fairness auxiliaries at a feasible point (they do not enter the objective)
    c = [CalcIndividualJ(xv, i, C, m, n) for i in 1:n]
    w = maximum(c)
    v = max.(c .+ Δ, w)

    primals   = [xv; v; w]
    ineq_vals = [xv; ce_vals; w .- c; v .- c .- Δ; v .- w]
    mu_ineq   = [mu_x; mu_ce; zeros(3n)]

    score = ok ? sum(Jtot .* xv) : NaN
    (; primals,
       x_sol = primals,
       ineq_vals,
       mu_ineq,
       score,
       avgDelayScore = score / n,
       fairScore = ok ? EvalFairness(xv, C, m, n, Δ) : NaN,
       giniScore = ok ? EvalGini(xv, C, m, n, Δ) : NaN,
       varsize = length(primals),
       buildTime,
       solverTime,
       status,
       m, n, l, C)
end
