using Random, Statistics, MAT, Distributions
using BlockArrays: Block, blocksize

############################################################
# Helpers
############################################################

function joint_action_to_onehot(a::Tuple, m, n)
    z = zeros(m^n)
    Z = reshape(z, ntuple(_ -> m, n))
    Z[CartesianIndex(a)] = 1.0
    return vec(Z)
end

function realized_score(a::Tuple, C, m, n)
    z = joint_action_to_onehot(a, m, n)
    return CalcJ(z, C, m, n)
end

function alpha_to_z(alpha::Real)
    alpha <= 0 && return 0.0
    return quantile(Normal(), alpha)
end


function sanitize_prob_vector(x; tol = 1e-12)
    p = vec(copy(x))
    p[abs.(p) .< tol] .= 0.0
    p = max.(p, 0.0)

    s = sum(p)
    if s <= tol
        error("Probability vector collapsed to zero after sanitization.")
    end
    return p ./ s
end

function sample_joint_action_from_x(rng, x, m, n)
    probs = sanitize_prob_vector(x)
    idx = rand(rng, Categorical(probs))
    CI = CartesianIndices(ntuple(_ -> m, n))[idx]
    return Tuple(CI)
end

function check_deviation_solver_style(x, C, m, n, sigma, rng)
    ε_dev = 1e-7
    deviated = falses(n)

    for i in 1:n
        η = randn(rng) * sigma[i]

        for ai in 1:m
            pai = 0.0

            # marginal probability p(ai)
            for idx in 1:length(x)
                CI = CartesianIndices(ntuple(_->m,n))[idx]
                if CI[i] == ai
                    pai += x[idx]
                end
            end

            pai == 0 && continue

            for aibar in 1:m
                aibar == ai && continue

                mean_diff = 0.0

                # expected regret
                for idx in 1:length(x)
                    CI = CartesianIndices(ntuple(_->m,n))[idx]
                    a = Tuple(CI)

                    if a[i] == ai
                        a_dev = collect(a)
                        a_dev[i] = aibar

                        mean_diff += x[idx] *
                            (player_cost(i, Tuple(a_dev), C) - player_cost(i, a, C))
                    end
                end

                if mean_diff + η * pai < -ε_dev
                    deviated[i] = true
                    break
                end
            end

            deviated[i] && break
        end
    end

    return deviated
end

############################################################
# Cost at a fixed joint action
############################################################

function player_cost(i, a::Tuple, C)
    s = 0.0
    n = length(a)
    for j in 1:n
        s += C[Block(i,j)][a[i], a[j]]
    end
    return s
end

############################################################
# Realized action under a sampled recommendation
############################################################

"""
Given sampled recommendation a_rec, each player i chooses its realized action
by comparing the recommended action against unilateral deviations while
holding all other players at the recommended action.

Deviation if:
    rec_cost - dev_cost + eta[i] > 0
"""
function realized_profile_after_simul_deviation(a_rec::Tuple, C, m, eta, rng, sigma; tol = 1e-8)
    n = length(a_rec)
    a_real = collect(a_rec)
    deviated = falses(n)

    for i in 1:n
        rec_cost = player_cost(i, a_rec, C)

        best_action = a_rec[i]
        best_margin = 0.0

        for aibar in 1:m
            aibar == a_rec[i] && continue

            a_dev = collect(a_rec)
            a_dev[i] = aibar
            dev_cost = player_cost(i, Tuple(a_dev), C)

            eta_rec = randn(rng) * sigma[i]
            eta_dev = randn(rng) * sigma[i]

            margin = (rec_cost + eta_rec) - (dev_cost + eta_dev)

            if margin > best_margin + tol
                best_margin = margin
                best_action = aibar
            end
        end

        a_real[i] = best_action
        deviated[i] = (best_action != a_rec[i])
    end

    return Tuple(a_real), deviated
end

############################################################
# CE rollout evaluator
############################################################

function evaluate_ce_solution_realized(rng, x, C, m, n, sigma; noise_runs = 100)
    score_list = Float64[]
    dev_flags = Bool[]
    num_dev_list = Int[]

    for _ in 1:noise_runs
        # sample recommendation from CE distribution
        a_rec = sample_joint_action_from_x(rng, x, m, n)

        # realized perturbation on deviation side
        eta = randn(rng, n) .* sigma

        # realized action after agents respond to recommendation
        # a_real, deviated = realized_profile_after_simul_deviation(a_rec, C, m, eta, rng, sigma)
        deviated = check_deviation_solver_style(x, C, m, n, sigma, rng)
        a_rec = sample_joint_action_from_x(rng, x, m, n)
        push!(score_list, realized_score(a_rec, C, m, n))

        # evaluate realized action with the SAME weighted cost function
        # push!(score_list, realized_weighted_score(a_rec, C, m, n, sigma))
        # push!(score_list, realized_weighted_score(a_real, C, m, n, sigma))
        push!(dev_flags, any(deviated))
        push!(num_dev_list, count(deviated))
    end

    return (;
        score_mean = mean(score_list),
        score_std  = std(score_list),
        dev_rate = mean(dev_flags),
        mean_num_deviators = mean(num_dev_list),
    )
end

