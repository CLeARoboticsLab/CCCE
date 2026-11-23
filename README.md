# ChanceConstrainedCE.jl

Computation of **chance-constrained correlated equilibria (CC-CE)** in Julia, together with the code
and data needed to reproduce the experiments of two papers:

| Folder | Paper | Figures |
|---|---|---|
| `experiments/ccce_infogain/` | *Chance-Constrained Correlated Equilibria for Robust Noncooperative Coordination* (IEEE L-CSS, 2026) | Fig. 3 (confidence level α), Fig. 4 (information acquisition) |
| `experiments/scalability/` | *Scaling Chance-Constrained Correlated Equilibrium Computation for Exclusive Resource-Assignment Games* (arXiv:2609.13565) | Fig. 2 (computation time), Fig. 3 (average cost and Gini index) |

## Quick start

You need **Julia ≥ 1.10** ([install with juliaup](https://julialang.org/downloads/)). MATLAB is needed
only to redraw the figures.

```bash
# 1. Get the code
git clone -b public git@github.com:iamjaehan/ChanceConstrainedCE.git
cd ChanceConstrainedCE

# 2. Install the pinned dependencies (first time only, ~5-10 min including precompilation)
julia --project=. -e 'using Pkg; Pkg.instantiate()'

# 3. Check that everything works (~1 min)
julia --project=. -e 'using Pkg; Pkg.test()'

# 4. Smoke-test each experiment (a few minutes in total)
julia --project=. experiments/scalability/run_mc.jl --quick
julia --project=. experiments/scalability/run_compact.jl --quick
julia --project=. experiments/ccce_infogain/exp1_alpha.jl --quick
julia --project=. experiments/ccce_infogain/exp2_infogain.jl --quick
```

Run every command from the repository root. Each script prints a summary and writes a `.mat`
file to its `results/` folder. `--quick` runs write `*_quick.mat` and never overwrite the paper data.

### Running times

Measured on an Apple M4 Pro (12 cores, 24 GB). Times will differ on other machines.

| Command | `--quick` | Paper setting |
|---|---|---|
| `experiments/scalability/run_mc.jl` | ~30 s | ~1 h (dominated by Full-CCCE at n = 5 and RR_X at n = 8) |
| `experiments/scalability/run_compact.jl` | ~15 s | ~2 min |
| `experiments/ccce_infogain/exp1_alpha.jl` | ~3 min | ~4 h (700 PATH solves, ~20 s each) |
| `experiments/ccce_infogain/exp2_infogain.jl` | ~20 min | ~2-3 h (360 PATH solves) |

The CC-CE / InfoGain experiments are slow because every PATH solve rebuilds and compiles the
symbolic MCP. This affects only wall-clock time, not results. The data behind the published figures
is already in `results/`, so you can redraw every figure without rerunning anything (see below).

### Redrawing the figures (MATLAB)

```bash
cd experiments/scalability/plot  && matlab -batch PlotScalability     # -> experiments/scalability/figures/
cd experiments/ccce_infogain/plot && matlab -batch PlotCCCEInfoGain   # -> experiments/ccce_infogain/figures/
```

By default the plot scripts read the paper data in `results/`. To plot your own run, rerun the
experiment without `--quick` (this overwrites `results/`), or change the data path at the top of the
plot script.

Key package versions are pinned in `Project.toml` (`[compat]`) to the versions used for the papers:
JuMP 1.29.4, Ipopt.jl 1.14.0 (Ipopt 3.14.19), PATHSolver.jl 1.7.9, ParametricMCPs.jl 0.1.17,
Symbolics.jl 6.58.0. PATH is used under its free courtesy license, which is set in `src/ChanceConstrainedCE.jl`.

## Using the solvers in your own game

Describe a finite game by a cost array `costs[i, a_1, …, a_n]` (agent `i`'s cost at joint action
`a`; agents minimize). Each agent's deviation costs carry Gaussian uncertainty with standard
deviation `σ_i`, and incentive compatibility must hold with probability `α`.

```julia
using ChanceConstrainedCE

# Game of chicken (actions: 1 = dare, 2 = yield)
costs = zeros(2, 2, 2)
costs[1, :, :] = [10 0; 7 2]
costs[2, :, :] = [10 7; 0 2]

res = solve_ccce(costs; alpha = 0.9, sigma = [1.0, 1.0])
res.z                   # recommendation distribution over joint actions (2×2)
res.expected_costs      # E_z[J_i] per agent
res.constraints         # per incentive constraint (i, a_i, a_i'): slack, dual λ, π, infogain = σ λ π

rr = solve_rr(costs; alpha = 0.9, sigma = 1.0)   # mixture of chance-constrained pure NE
rr.ccpne                # the CC-PNEs found
```

| Function | What it solves |
|---|---|
| `solve_ccce(costs; alpha, sigma, objective, support)` | CC-CE linear program. With `support = S`, z is restricted to S (Restricted-CCCE). |
| `find_ccpne(costs; alpha, sigma, support)` | Chance-constrained pure Nash equilibria in X or in S. |
| `solve_rr(costs; alpha, sigma, objective, support)` | Best mixture of CC-PNEs: RR_X, or RR_S with `support = S` (RR_F with `S = F`). |
| `exclusive_support(r, n)` | The exclusive-assignment set F for the occupy/yield resource game of the scalability paper. |

Options:
- `alpha` (or `zalpha = Φ⁻¹(α)`): omit both for the nominal correlated equilibrium.
- `sigma`: a scalar or a per-agent vector.
- `objective`: `:aggregate` (sum of expected costs, default), `:fair` (fairness threshold `Δ`),
  `:minmax`, or an array `W` of system costs per joint action.
- `sigma_scale`: `Dict((i, a_i, a_i′) => s)` scales one constraint's uncertainty, for example `s = 0`
  to remove it after acquiring information.
- `optimizer`: any JuMP LP solver (default GLPK).

For a general candidate set S, RR_S is feasible for Restricted-CCCE on S but can be worse. The two
coincide for exclusive resource-assignment games with S = F, as shown in the scalability paper.
When the LP has several optimal solutions or multipliers, the solver returns one of them.

## Package layout

```
src/
  ChanceConstrainedCE.jl          module definition and exports
  general/solvers.jl              solve_ccce, find_ccpne, solve_rr, exclusive_support (any finite game)
  mcp/parametric_optimization_problem.jl
                                  optimization problem -> MCP (ParametricMCPs) -> PATH
  ccce/game_setup.jl              runway/queue game (SetC), CC-CE constraints (CalcH, CorrPacker),
                                  constraint/dual bookkeeping used by InfoGain
  ccce/search_corr.jl             SearchCorr: general CC-CE solved as an MCP with PATH
  exclusive/action_sets.jl        action encoding, exclusive set F (ExclusiveJointSet, |F| = n^r)
  exclusive/exclusive_game.jl     exclusive resource-assignment game and its solvers (JuMP + Ipopt):
                                    SearchCorrReducedLPHet   Restricted-CCCE (CC-CE LP over F)
                                    SearchCorrLPHet          Full-CCCE (CC-CE LP over X)
                                    HullLPHetOverF / OverX   RR_F / RR_X (CC-PNE screening + hull LP)
                                    CompactLPHet             compact O(n) LP (see below)
test/runtests.jl                  correctness checks (F construction, Restricted = RR_F = Compact,
                                  infeasibility condition, PATH-based CC-CE)
```

## Reproducing the CC-CE / InfoGain experiments

```bash
julia --project=. experiments/ccce_infogain/exp1_alpha.jl      # Fig. 3: γ = 2, α ∈ {0 (Naive-CE), 0.66, …, 0.999}, 100 trials, 100 noise samples
julia --project=. experiments/ccce_infogain/exp2_infogain.jl   # Fig. 4: α = 0.9, γ ∈ {1.3, 1.5, 1.8, 2.0}, 10 trials, 10 noise samples

cd experiments/ccce_infogain/plot && matlab -batch PlotCCCEInfoGain
```

Outputs go to `experiments/ccce_infogain/results/` and figures to `experiments/ccce_infogain/figures/`:
`fig3_alpha_test.pdf` and `fig4_infogain.pdf`. The scripts use exactly the setting of the paper:
n = 3 agents, 2 vertiports (4 actions per agent), seed 42, and the CC-CE problem solved as an MCP with
PATH (`SearchCorr`). The `.mat` files and logs in `results/` are the data behind the published figures.

The package also provides `SearchCorrLP`, which solves the same CC-CE problem as a linear program
(JuMP + GLPK) in milliseconds instead of ~20 s. It reaches the same optimal value. The LP can have
several optimal solutions and dual multipliers, though, so the recommendation distribution, realized
costs and dual-based (InfoGain) rankings can differ from PATH's. The experiment scripts therefore use
PATH, as in the paper.

## Reproducing the scalability experiments

```bash
julia --project=. experiments/scalability/run_mc.jl          # Figs. 2-3: 6 values of n x 100 instances
julia --project=. experiments/scalability/run_compact.jl     # compact LP on the same instances

cd experiments/scalability/plot && matlab -batch PlotScalability
```

Figures are written to `experiments/scalability/figures/`:
`fig2_total_time.pdf`, `fig3_cost_fairness.pdf`, and `fig2_total_time_compact.pdf` (Fig. 2 with the compact LP added).

**Setting.** r = 3 resources (m = 8 actions per agent), c = 10, σ_i = 2, α = 0.95, Δ = 3, coordinator
objective `:fair`. Yield costs ρ_i ~ U(1, 6) i.i.d. per agent. 100 instances per n = 3,…,8, seed 2026.
Full-CCCE is run for n ≤ 5 only.

**Timing protocol.** Reported times are end-to-end wall-clock times, not solver time alone:

- CC-CE methods include building the support set (F or X), computing all incentive-constraint
  coefficients, building the LP, and the solve.
- RR methods include the CC-PNE screening over F or X, building the hull LP over the CC-PNEs found, and
  the solve.
- `build_*` and `solve_*` in the results split the LP part into model construction and Ipopt time.
- One untimed warm-up per n and method runs first, so no instance includes Julia JIT compilation.

**Infeasible instances.** When max_i ρ_i < z_α σ, no CC-CE (and hence no CC-PNE) exists, and every
method reports infeasibility. These instances (20 of 600 with the default seed) are excluded from all
figures without resampling. All statistics use the same retained instances for every method.

**Compact LP.** `CompactLPHet` solves Restricted-CCCE exactly with n variables (the expected number of
resources held by each agent). It is exact only when (a) each agent's yield cost is the same on every
resource and (b) the coordinator objective depends on the joint distribution only through the agents'
expected costs. `run_compact.jl` checks the objective value against Restricted-CCCE on every instance.
It does not apply to resource-dependent costs, to objectives on the realized-cost distribution (e.g.,
variance or tail risk), or to Full-CCCE.

**Reference hardware** for the timings in `results/`: Apple M4 Pro (12 cores, 24 GB RAM), macOS, Julia 1.12.
Absolute times will differ on other machines. The relative ordering of the methods should not.

## Citation

If you use this code, please cite the corresponding paper(s).

```bibtex
@article{im2026chance,
  author  = {Im, Jaehan and Topcu, Ufuk and Fridovich-Keil, David},
  title   = {Chance-Constrained Correlated Equilibria for Robust Noncooperative Coordination},
  journal = {IEEE Control Systems Letters},
  volume  = {10},
  pages   = {1273--1278},
  year    = {2026},
  doi     = {10.1109/LCSYS.2026.3705300}
}

@misc{im2026scaling,
  author        = {Im, Jaehan and Topcu, Ufuk and Fridovich-Keil, David},
  title         = {Scaling Chance-Constrained Correlated Equilibrium Computation for Exclusive Resource-Assignment Games},
  year          = {2026},
  eprint        = {2609.13565},
  archivePrefix = {arXiv},
  primaryClass  = {cs.GT},
  url           = {https://arxiv.org/abs/2609.13565}
}
```
