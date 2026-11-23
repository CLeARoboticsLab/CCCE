module ChanceConstrainedCE

# =============================================================================
# Chance-constrained correlated equilibrium (CC-CE) computation.
#
#   ccce/       general CC-CE search on the runway/queue game (SetC), solved as
#               an MCP with PATH -- used by the CC-CE + InfoGain experiments.
#   exclusive/  exclusive resource-assignment games: Restricted-CCCE over F,
#               Full-CCCE over X, CC-PNE search + hull LPs (RR_X, RR_F) and the
#               compact expected-allocation LP, solved with JuMP + Ipopt --
#               used by the scalability experiments.
#   mcp/        ParametricOptimizationProblem -> ParametricMCP -> PATH wrapper.
# =============================================================================

using LinearAlgebra: norm_sqr
using Statistics: mean, std
using Random
using BlockArrays
using BlockArrays: Block, BlockArray, mortar, blocks, blocksize
using Combinatorics
using Symbolics: Symbolics, @variables
using ParametricMCPs: ParametricMCPs, ParametricMCP
using PATHSolver
using JuMP
import Ipopt
import GLPK

function __init__()
    # PATH's free "courtesy" license, as published with PATHSolver.jl.
    PATHSolver.c_api_License_SetString(
        "1259252040&Courtesy&&&USR&GEN2035&5_1_2026&1000&PATH&GEN&31_12_2035&0_0_0&6000&0_0")
end

include("mcp/parametric_optimization_problem.jl")
include("ccce/game_setup.jl")
include("ccce/search_corr.jl")
include("ccce/search_corr_lp.jl")
include("exclusive/action_sets.jl")
include("exclusive/exclusive_game.jl")

export ActionIndex, BuildCorrConstraintMap, BuildRunwayActionSet, CEKeyList, CalcEFJ,
       CalcH, CalcIndividualJ, CalcJ, CalcJNash, CalcJNashSet, CalcMarginalP, CalcPhi,
       CalcWeightedJ, CheckSetCCPNEHet, CompactLPHet, CorrPacker, EvalAverageDelay,
       EvalFairness, EvalGini, ExclusiveJointSet, FindAllCCPNEHet,
       GetActiveCEConstraints, GetActiveConstraints, GetCEConstraintIndices,
       GetCEConstraintInfo, HetYieldWeights, HullLPHetOverF, HullLPHetOverX,
       IsCCPNEProfileHet, J_def, NashPacker, NumCorrEqConstraints,
       NumCorrIneqConstraints, NumCorrPrimalVars, NvVector, OccupancyVec,
       ParametricOptimizationProblem, PrintCEList, RandomKCE, SearchCorr, SearchCorrLP,
       SearchCorrLPHet, SearchCorrReducedLPHet, SetC, SmoothMax, SplitCorrSolution,
       SwitchIndtoJoint, T1Const, T1ConstN, T2Const, T2ConstN, T3Const,
       TopKActiveCEByMu, TopKCEByMu, TopKCEByMuPSigma, TopKCEByMuSigma, TopKCEBySigma,
       UniformAgentYieldWeights, VertiportCostHet, generateAseq, is,
       max_CE_violation_2p, of, total_dim, solve

end # module
