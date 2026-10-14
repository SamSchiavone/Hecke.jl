# Improving the computation of the odd theta gradients

Status of the suggestions for `odd_theta_gradient_kernel` (`ReconstructGenusG.jl`)
and of what has been built from them. Branch `theta-gradients`; timings on
Sam's MacBook Pro (M‑series, Julia 1.13.1 with `-t 3`) at 256 bits unless
stated, with the patched FLINT 3.6.0 (branch `theta-optimize`: cheap error
propagation in `acb_theta_all`, threaded `acb_theta_sum`).

## The problem

The reconstruction needs, twice, a matrix `K` whose rows are the gradients of
the odd theta functions at `z = 0` up to a common `GL_g`:

| | theta constants from | period matrix | gradients today |
|---|---|---|---|
| curve, genus g | `tau` via `acb_theta` | known | **`odd_theta_gradients(tau)`** (item 5) |
| Prym, genus g − 1 | Schottky–Jung products of the curve's constants | not available | `odd_theta_gradient_kernel(g − 1, thetas_prym)`: kernel of Fay's relations |

The Fay kernel is a sparse linear system with `n = 2^(g-1)(2^g - 1)` unknowns,
`m ≈ n` relations and at most `5·2^(g-3)` nonzeros per relation, solved by a
Float64 pivot selection followed by full‑precision iterative refinement
(`_sparse_kernel`). Everything in the Float64 phase used to be dense:

| g | n | nnz/row | nnz total | dense `A0` | dense QR/LU flops |
|---|---|---|---|---|---|
| 6 | 2016 | 40 | 8·10⁴ | 65 MB | 10¹¹ |
| 7 | 8128 | 80 | 6.5·10⁵ | 1.1 GB | 7·10¹² |
| 8 | 32640 | 160 | 5·10⁶ | 17 GB | 4·10¹⁴ |
| 9 | 130816 | 320 | 4·10⁷ | 274 GB | — |

For a curve of genus g the Prym system has genus g − 1, so genus‑8 curves need
the g = 7 row, genus‑9 curves the g = 8 row.

## 1. Sparse Float64 phase (SuiteSparse) — implemented

`_sparse_kernel(...; sparse = nothing, sparse_limit = 1500, sparse_tol = 1e-13)`:
for `n > sparse_limit` (genus ≥ 6) the Float64 matrix is a `SparseMatrixCSC`;
the pivot columns come from SPQR's rank‑revealing QR (`qr(A0; tol)`, column
permutation `pcol`, rank checked against `n − g`), the pivot block is all rows ×
pivot columns factored by sparse QR, the corrections of the refinement are
sparse least squares, `σ_min` by inverse iteration through the sparse `R`.
Refinement loop, radius estimate and residual checks unchanged. No BigFloat
fallback in sparse mode. Reachable as `odd_theta_gradient_kernel(g, thetas;
sparse = true/false, sparse_tol = ...)`.

Tested with `time_fay_kernel(g)` (`ThetaGradientsTests.jl`): same relations,
sparse vs dense, both compared with `odd_theta_gradients` up to the common
linear map (random reduced tau of `random_period_matrix`, seed 1):

| g | n | dense (before) | sparse (after) | residual (both) | kernel vs gradients |
|---|---|---|---|---|---|
| 5 | 496 | 0.62 s | 3.7 s (first call, includes compilation) | 2⁻²⁵⁵ / 2⁻²⁵³ | 2⁻²⁴⁵ |
| 6 | 2016 | 2.8 s | 1.2 s | 2⁻²⁵⁴ | 2⁻²²⁸ |
| 7 | 8128 | 51 s | **7.2 s** | 2⁻²⁵² | 2⁻²⁰³ |

The agreement with the gradients drops with g in both versions alike (the
conditioning of the Fay system, `σ_min ≈ 10⁻¹²` at g = 7, not the solver).
Not yet run: g = 8 (n = 32640; needs the relation set first, see item 3), so the
fill‑in of SPQR at that size is still unmeasured.

## 2. Relations ordered for fill‑free elimination — not implemented

Normalise g rows of `K` to the identity; a relation with exactly one unknown
gradient determines it by one scalar division, so an ordering of the relations
in which each introduces one new characteristic makes the system triangular and
the kernel a back‑substitution in O(nnz), with no factorization. The supports of
the relations (translates `base + σ`, `base + σ + ρ` over the bases of
`_fay_relation_structure`) are structured enough that such an order, or a
partial one (large triangular blocks plus a small dense remainder), should be
findable once per genus. Long chains amplify error, which the full‑precision
refinement repairs; `σ_min` reports if the order is too ill‑conditioned.
Fallback if item 1's fill‑in is bad at g ≥ 8; also the realistic replacement
of the BigFloat fallback (elimination in `Double64` or in `acb` at 128 bits
along this order).

## 3. Relation sets from the symmetry instead of a search — not implemented

Translating a Fay relation `(w, ρ)` by `t ∈ F₂^{2g}` gives another Fay
relation, so a basis is "a few `(w, ρ)` up to translation", computable for any
g, instead of a stored list found by `find_fay_relations` (a Float64 pivoted‑QR
search over `2^(2g-1)` candidates per ρ, O(n² × candidates): the `@info` in
`_fay_structure` warns from g = 6; estimated hours at g = 8, and no
`G8FayRelations` file exists). Needed before item 1 can be run at g = 8; its
coset structure is the natural starting point for item 2.

## 4. Smaller things — implemented

- `_sparse_fay_rows` caches the products `theta[c1] theta[c2]` over the pairs
  `(base, base + ρ)` (two ball multiplications per term instead of three).
- The residual loop of the refinement is `Threads.@threads` over the rows.
- Not separately timed (both are active in all runs of the table above;
  `odd_theta_gradient_kernel(6, thetas)` takes 3.9 s on a first call including
  compilation).
- Not done: the BigFloat fallback for undecidable Float64 pivots cannot exist
  in sparse mode (see item 2).

## 5. Gradients from the period matrix — implemented (curve only)

`odd_theta_gradients(tau)` (`ThetaGradients.jl`, see `ThetaGradientsSpec.md`):
for odd κ, `theta[κ](ε e_j, tau)/ε = ∂_j theta[κ](0, tau) + O(ε²)`, so g
evaluations of all theta values (one `acb_theta_jet` call, or one call per
thread with the points split into groups) give `K`; the remainder is bounded by
Cauchy's estimate from `acb_theta_ql_local_bound`, so the result is certified.
`reconstruct_quadrics_data(...; gradient_method = :theta)` (default) uses it
for the curve; `:fay` is the previous behaviour; the Prym always uses the kernel.
Tests: `test_theta_gradients(5)` passes (overlap with `acb_theta_jet` per
characteristic, radius ≈ 2⁻ᵖʳᵉᶜ); the Fay kernel is a linear image of `K`
(table above); `test_theta_gradients_pipeline` not yet run.

Timings (`random_period_matrix`, the paper's worst‑case reduced tau with
eigenvalues of Im(tau) ≈ 0.75):

| g | n | serial | 3 threads (grouped calls) |
|---|---|---|---|
| 5 | 496 | 0.9 s | |
| 6 | 2016 | 10 s | |
| 7 | 8128 | 146 s | 75 s |
| 8 | 32640 | 1893 s (32 min) | (est. ~16 min) |

The cost is dominated by FLINT's low‑precision sign determination
(`acb_theta_ql_setup`: with all characteristics and z ≠ 0 the last duplication
step cannot be "easy", so every step uses the auxiliary vector, and the 8‑bit
attempt is redone at 16 bits), hence **independent of the target precision**:
g = 6, 1000 bits: 26 s vs 25 s at 256 bits. FLINT's own random reduced tau runs
~12× faster at g = 7 than the paper's recipe; a timing on a real Jacobian's tau
is still to be done. Further gains need FLINT: sharing the sign roots across the
g vectors (they coincide with z = 0 to 2⁻¹⁵⁰), up to g× (upstream change to
`ql_setup`).

## Summary

| item | status | effect so far |
|---|---|---|
| 1 sparse Float64 phase | done, default for g ≥ 6 | Prym g = 7: 51 s → 7 s; g = 8 untested |
| 2 fill‑free ordering | not started | needed if SPQR fill‑in is bad at g ≥ 8 |
| 3 relations from symmetry | not started | needed for g = 8 (no stored relations; search ~hours) |
| 4 cache + threads | done | not separately measured |
| 5 gradients from tau | done (curve) | replaces the curve's Fay kernel; g = 8 in 32 min serial |

Next: item 3 (makes a genus‑8 Prym possible at all), then run item 1 at g = 8.
