# ChanceConstrainedCE.jl

Computation of **chance-constrained correlated equilibria (CC-CE)** in Julia, together with the code
and data needed to reproduce the experiments of two papers:

| Folder | Paper | Figures |
|---|---|---|
| `experiments/ccce_infogain/` | *Chance-Constrained Correlated Equilibria for Robust Noncooperative Coordination* (IEEE L-CSS, 2026) | α sweep (NaiveCE vs. CC-CE), InfoGain strategy comparison |
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
| `experiments/ccce_infogain/exp1_alpha.jl` | ~3 min | several hours (700 PATH solves, ~20 s each) |
| `experiments/ccce_infogain/exp2_infogain.jl` | ~7 min | ~3 h (200 PATH solves plus noise rollouts) |

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

## Package layout

```
src/
  ChanceConstrainedCE.jl          module definition and exports
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
julia --project=. experiments/ccce_infogain/exp1_alpha.jl      # α sweep: 100 trials, 100 noise runs each
julia --project=. experiments/ccce_infogain/exp2_infogain.jl   # InfoGain: γ ∈ {1.1, 1.3, 1.7, 2.0}, 10 trials, 50 noise runs

cd experiments/ccce_infogain/plot && matlab -batch PlotCCCEInfoGain
```

Outputs go to `experiments/ccce_infogain/results/` and figures to `experiments/ccce_infogain/figures/`:
`fig_alpha_test.pdf` (α sweep) and `fig_infogain_test.pdf` (InfoGain strategies). Both runs use a fixed
seed (42). The `.mat` files and logs in `results/` are the data behind the published figures. Running the
scripts regenerates them exactly: the solver is deterministic, so the logs match line by line.

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
