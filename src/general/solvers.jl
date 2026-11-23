# =============================================================================
# General-purpose CC-CE and RR solvers for any finite normal-form game.
#
# A game is given by a cost array
#
#     costs[i, a_1, …, a_n] = J_i(a)        (agent i's cost; agents minimize)
#
# with size (n, m_1, …, m_n). Agent i's deviation-cost uncertainty is
# η_i ~ N(0, σ_i²) (agent level, shared by all of i's deviation constraints).
# Every incentive constraint c = (i, a_i, a_i') of the CC-CE reads
#
#     Σ_{a: a_i fixed} z(a) [J_i(a_i', a_-i) − J_i(a)]  −  q(α) σ_i π_i(a_i)  ≥ 0,
#
# where π_i(a_i) = Σ_{a_-i} z(a_i, a_-i) and q(α) = Φ⁻¹(α). This is the
# deterministic equivalent of P(no profitable deviation) ≥ α. With α = 0.5
# (q = 0), or σ = 0, it reduces to the ordinary correlated equilibrium.
#
#   solve_ccce   CC-CE: LP over distributions z on X, or on a candidate
#                support S ⊆ X (Restricted-CCCE). Deviations always range
#                over the full individual action sets.
#   find_ccpne   pure profiles a with J_i(a_i', a_-i) − J_i(a) ≥ q(α) σ_i for
#                every agent and every deviation (chance-constrained pure NE).
#   solve_rr     RR: LP over mixtures of the CC-PNEs found in X or in S.
#                Every such mixture is a CC-CE, so no incentive constraints are
#                needed.
#
# Objectives (system cost J_sys(z), always linear in z):
#   :aggregate      Σ_i E_z[J_i]                          (utilitarian)
#   :fair           Σ_i max(C_i + Δ, max_j C_j) − nΔ      (fairness threshold Δ)
#   :minmax         max_i C_i
#   array W         Σ_a z(a) W[a], W of size (m_1, …, m_n)
# where C_i = E_z[J_i].
# =============================================================================

const _GOOD_LP = (MOI.OPTIMAL, MOI.LOCALLY_SOLVED)

function _check_costs(costs::AbstractArray)
    n = size(costs, 1)
    ndims(costs) == n + 1 || throw(ArgumentError(
        "costs must have size (n, m_1, …, m_n); got size $(size(costs)) for n = $n agents"))
    return n, size(costs)[2:end]
end

function _zalpha(alpha, zalpha)
    if zalpha !== nothing
        return Float64(zalpha)
    elseif alpha === nothing
        return 0.0
    else
        0 < alpha < 1 || throw(ArgumentError("alpha must lie in (0, 1)"))
        return quantile(Normal(), alpha)
    end
end

_sigmas(sigma, n) = sigma isa Number ? fill(Float64(sigma), n) : Float64.(collect(sigma))

function _support(support, ms)
    support === nothing && return vec(collect(CartesianIndices(ms)))
    return [CartesianIndex(Tuple(s)) for s in support]
end

J(costs, i, a::CartesianIndex) = costs[i, Tuple(a)...]
function _deviate(a::CartesianIndex, i, ai)
    t = collect(Tuple(a)); t[i] = ai
    return CartesianIndex(Tuple(t))
end

# expected-cost expressions C_i(z) and the objective
function _set_objective!(model, z, S, costs, n, objective, Δ)
    C = [@expression(model, sum(J(costs, i, S[k]) * z[k] for k in eachindex(S))) for i in 1:n]
    if objective === :aggregate
        @objective(model, Min, sum(C))
    elseif objective === :fair
        @objective(model, Min, _fairness_objective!(model, C, n, Δ))
    elseif objective === :minmax
        t = @variable(model)
        @constraint(model, [i = 1:n], t >= C[i])
        @objective(model, Min, t)
    elseif objective isa AbstractArray
        @objective(model, Min, sum(objective[S[k]] * z[k] for k in eachindex(S)))
    else
        throw(ArgumentError("objective must be :aggregate, :fair, :minmax or an array of size (m_1, …, m_n)"))
    end
    return C
end

function _distribution(zval, S, ms)
    Z = zeros(ms)
    for (k, a) in enumerate(S)
        Z[a] = max(zval[k], 0.0)
    end
    return Z
end

"""
    solve_ccce(costs; alpha = nothing, zalpha = nothing, sigma = 0.0,
               objective = :aggregate, Δ = 1.0, support = nothing,
               sigma_scale = Dict(), optimizer = GLPK.Optimizer, verbose = false)

Chance-constrained correlated equilibrium of the game `costs[i, a_1, …, a_n]` (agents
minimize cost), selected by minimizing `objective`; see the header of this file.

- `alpha` (confidence level) or `zalpha = Φ⁻¹(α)`. Both omitted: nominal CE.
- `sigma`: scalar or per-agent vector of uncertainty levels σ_i.
- `support`: optional collection of joint actions (tuples or CartesianIndex);
  z is restricted to it (Restricted-CCCE), deviations are not.
- `sigma_scale`: `Dict((i, a_i, a_i′) => s)` multiplies σ_i by `s` in that single
  constraint (e.g. `s = 0` removes its uncertainty margin).

Returns a named tuple with `z` (array of size (m_1, …, m_n)), `status`,
`objective_value`, `expected_costs`, and `constraints`: one entry per
incentive constraint with fields `agent, rec, dev, slack, dual, π, σ, infogain`,
where `dual` ≥ 0 is the LP multiplier λ_c and `infogain = σ λ_c π_c` (value of
reducing that constraint's uncertainty, to first order). When the LP has several
optimal solutions or multipliers, the solver returns one of them.
"""
function solve_ccce(costs::AbstractArray; alpha = nothing, zalpha = nothing, sigma = 0.0,
                    objective = :aggregate, Δ = 1.0, support = nothing,
                    sigma_scale = Dict{Tuple{Int,Int,Int},Float64}(),
                    optimizer = GLPK.Optimizer, verbose = false)
    n, ms = _check_costs(costs)
    q = _zalpha(alpha, zalpha)
    σ = _sigmas(sigma, n)
    S = _support(support, ms)

    model = Model(optimizer)
    verbose || set_silent(model)
    @variable(model, z[1:length(S)] >= 0)
    @constraint(model, sum(z) == 1)

    meta = NamedTuple[]
    refs = ConstraintRef[]
    for i in 1:n, ai in 1:ms[i]
        ks = [k for k in eachindex(S) if S[k][i] == ai]
        isempty(ks) && continue                     # a_i never recommended
        π = sum(z[k] for k in ks)
        for aj in 1:ms[i]
            aj == ai && continue
            σc = σ[i] * get(sigma_scale, (i, ai, aj), 1.0)
            margin = sum((J(costs, i, _deviate(S[k], i, aj)) - J(costs, i, S[k])) * z[k] for k in ks)
            push!(refs, @constraint(model, margin - q * σc * π >= 0))
            push!(meta, (agent = i, rec = ai, dev = aj, σ = σc, ks = ks))
        end
    end
    C = _set_objective!(model, z, S, costs, n, objective, Δ)

    solve_time = @elapsed optimize!(model)
    status = termination_status(model)
    ok = status in _GOOD_LP && has_values(model)

    zval = ok ? value.(z) : fill(NaN, length(S))
    duals = ok && has_duals(model) ? dual.(refs) : fill(NaN, length(refs))
    constraints = [(agent = m.agent, rec = m.rec, dev = m.dev,
                    slack = ok ? value(refs[c]) : NaN,
                    dual = duals[c],
                    π = ok ? sum(zval[k] for k in m.ks) : NaN,
                    σ = m.σ,
                    infogain = ok ? m.σ * duals[c] * sum(zval[k] for k in m.ks) : NaN)
                   for (c, m) in enumerate(meta)]

    return (; z = ok ? _distribution(zval, S, ms) : fill(NaN, ms),
              status, objective_value = ok ? objective_value(model) : NaN,
              expected_costs = ok ? value.(C) : fill(NaN, n),
              constraints, zalpha = q, solve_time, model)
end

"""
    find_ccpne(costs; alpha = nothing, zalpha = nothing, sigma = 0.0, support = nothing)

Chance-constrained pure Nash equilibria: joint actions `a` (in `support`, or in all of
X) such that `J_i(a_i′, a_-i) − J_i(a) ≥ Φ⁻¹(α) σ_i` for every agent `i` and every
deviation `a_i′ ≠ a_i`. Returns a vector of `CartesianIndex`.
"""
function find_ccpne(costs::AbstractArray; alpha = nothing, zalpha = nothing, sigma = 0.0,
                    support = nothing)
    n, ms = _check_costs(costs)
    q = _zalpha(alpha, zalpha)
    σ = _sigmas(sigma, n)
    is_ccpne(a) = all(J(costs, i, _deviate(a, i, aj)) - J(costs, i, a) >= q * σ[i]
                      for i in 1:n for aj in 1:ms[i] if aj != a[i])
    return filter(is_ccpne, _support(support, ms))
end

"""
    solve_rr(costs; alpha = nothing, zalpha = nothing, sigma = 0.0,
             objective = :aggregate, Δ = 1.0, support = nothing,
             optimizer = GLPK.Optimizer, verbose = false)

RR: find the CC-PNEs (in `support`, or in all of X) with `find_ccpne`, then
choose the mixture of them that minimizes `objective`. Every such mixture is a
CC-CE. With `support = F`, this is RR_F; without a support, RR_X.

For a general candidate set S, every mixture of CC-PNEs in S is a CC-CE supported
on S, so RR_S is feasible for Restricted-CCCE on S but may be worse. The two
coincide (Theorem 1 of the scalability paper) for exclusive resource-assignment
games with S = F (see `exclusive_support`).

Returns `z`, `status` (`:NO_CCPNE` if none exist), `objective_value`,
`expected_costs`, `ccpne` (the equilibria found), `weights` (their mixture
weights), `search_time` and `solve_time`.
"""
function solve_rr(costs::AbstractArray; alpha = nothing, zalpha = nothing, sigma = 0.0,
                  objective = :aggregate, Δ = 1.0, support = nothing,
                  optimizer = GLPK.Optimizer, verbose = false)
    n, ms = _check_costs(costs)
    local P
    search_time = @elapsed (P = find_ccpne(costs; alpha, zalpha, sigma, support))
    if isempty(P)
        return (; z = fill(NaN, ms), status = :NO_CCPNE, objective_value = NaN,
                  expected_costs = fill(NaN, n), ccpne = P, weights = Float64[],
                  search_time, solve_time = 0.0, model = nothing)
    end

    model = Model(optimizer)
    verbose || set_silent(model)
    @variable(model, w[1:length(P)] >= 0)
    @constraint(model, sum(w) == 1)
    C = _set_objective!(model, w, P, costs, n, objective, Δ)

    solve_time = @elapsed optimize!(model)
    status = termination_status(model)
    ok = status in _GOOD_LP && has_values(model)
    wval = ok ? value.(w) : fill(NaN, length(P))

    return (; z = ok ? _distribution(wval, P, ms) : fill(NaN, ms),
              status, objective_value = ok ? objective_value(model) : NaN,
              expected_costs = ok ? value.(C) : fill(NaN, n),
              ccpne = P, weights = wval, search_time, solve_time, model)
end

"""
    exclusive_support(r, n)

Exclusive-assignment candidate set for a resource game encoded with
`BuildRunwayActionSet(r)` (each of n agents picks occupy/yield on r resources):
the joint actions in which every resource has exactly one occupant (|F| = n^r).
Returns a vector of `CartesianIndex`, ready for the `support` keyword.
"""
exclusive_support(r::Int, n::Int) = [CartesianIndex(s) for s in first(ExclusiveJointSet(r, n))]
