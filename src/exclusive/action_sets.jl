
# =============================================================================
# ReducedActionSet.jl
#
# Joint-action-space reduction for the runway/queue CC-CE game defined in
# 0_GameSetup.jl (SetC). Implements the "exclusive resource assignment"
# candidate set F from the reduced-joint-action formulation:
#
#   F := { x ∈ {0,1}^{n×r} : for every runway v, exactly one queue occupies it }
#
# For this game m = 2^r (each queue picks Occupy/Yield per runway independently),
# so |X| = m^n = 2^(rn), while |F| = n^r.
#
# Deviation costs for the reduced CC-CE LP are evaluated analytically via the
# cost-matrix blocks C[Block(i,j)] (J_def, already defined in 0_GameSetup.jl),
# so a deviation a_i' need NOT itself be an element of F -- exactly matching
# the AS[c,x] definition in the reduction note (a_i' ranges over the full
# individual action set Xi regardless of the recommendation set S).
# =============================================================================

# -----------------------------------------------------------------------------
# 1) Action-set construction (self-contained; does not depend on SetC globals)
# -----------------------------------------------------------------------------

"""
Reconstruct the per-player action ordering used internally by SetC, i.e. the
m = 2^r tuples of 'G'/'S' (Occupy/Stop) across r runways, in the same order
SetC uses to index its cost-matrix blocks.
"""
function BuildRunwayActionSet(r::Int)
    primeAction = "['G','S']"
    stringUnit = (primeAction * ",")^(r - 1) * primeAction
    stringSum = "vec(collect(Iterators.product(" * stringUnit * ")))"
    return eval(Meta.parse(stringSum))
end

"""
Index (1-based) of an occupancy tuple `tup::NTuple{r,Char}` within the action
ordering `actionSet` returned by BuildRunwayActionSet.
"""
function ActionIndex(actionSet, tup::Tuple)
    idx = findfirst(==(tup), actionSet)
    isnothing(idx) && error("Occupancy tuple $tup not found in action set.")
    return idx
end

# -----------------------------------------------------------------------------
# 2) Exclusive-occupancy reduced set F
# -----------------------------------------------------------------------------

"""
Build the exclusive-occupancy candidate set F for n queues and r runways.

Returns a Vector of NTuple{n,Int}, each entry giving, for every queue i, its
action index (1..m, m=2^r) in the SAME indexing SetC uses for its cost blocks.
|F| = n^r.
"""
function ExclusiveJointSet(r::Int, n::Int)
    actionSet = BuildRunwayActionSet(r)
    m = length(actionSet)
    @assert m == 2^r

    Sset = Vector{NTuple{n,Int}}(undef, n^r)
    k = 0
    for y in Iterators.product(ntuple(_ -> 1:n, r)...)
        # y[v] = index of the queue occupying runway v; all others yield on v
        k += 1
        occ = [ntuple(v -> (y[v] == i ? 'G' : 'S'), r) for i in 1:n]
        Sset[k] = ntuple(i -> ActionIndex(actionSet, occ[i]), n)
    end
    return Sset, m
end

