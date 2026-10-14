# Spec: gradients of the odd theta functions from the period matrix

Status: proposal, not yet implemented. Target file: `src/RieSrf/Reconstruction/ThetaGradients.jl`
(new), with small changes in `ReconstructGenusG.jl`.

## 1. Why

`reconstruct_quadrics_from_thetas` needs, for the curve, the matrix `K` whose rows
are the gradients `l_kappa = grad theta[kappa](0, tau)` of the odd theta functions, up
to a common linear transformation (step 1 in the header of `ReconstructGenusG.jl`).
Today `K` is obtained by `odd_theta_gradient_kernel`: the kernel of Fay's linear
relations between the gradients, with coefficients that are cubic in the theta
constants. That is a sparse linear system with `n_odd = 2^(g-1) (2^g - 1)` unknowns
whose Float64 phase is dense: 1 GB and minutes at g = 7, 17 GB and hours at g = 8.

In practice we start from a period matrix `tau` and compute the theta constants from
it with `acb_theta`. When `tau` is known, the gradients themselves are computable,
and the actual gradients are a valid `K` (the kernel is only defined up to `GL_g`, and
the true gradients are one representative of it). The linear system is unnecessary
for the curve.

### Curve vs. Prym

The pipeline needs two such matrices:

| | theta constants from | period matrix known? | how to get the gradients |
|---|---|---|---|
| curve, genus g | `tau` via `acb_theta` | yes | **this spec**: evaluate, O(g) theta evaluations |
| Prym, genus g - 1 | Schottky-Jung products `theta[0 d1; 0 d2] theta[0 d1; 1 d2]` of the curve's constants (`prym_thetas`) | **no** | unchanged: Fay kernel, `odd_theta_gradient_kernel(g - 1, thetas_prym)` |

So the Fay machinery is not replaced; it moves to the Prym, one genus lower, where the
existing dense Float64 phase is adequate up to curves of genus 8 (Prym genus 7,
`n_odd = 8128`). A sparse rewrite of `_sparse_kernel` (SPQR/UMFPACK) is only needed
for curves of genus 9 and is out of scope here.

## 2. Method

Fix `tau` (reduced, `g x g`) and a working precision. For an odd characteristic
`kappa`, `theta[kappa](z, tau)` is an odd function of `z` with `theta[kappa](0) = 0`,
so for `eps > 0` and the unit vector `e_j`,

```
theta[kappa](eps e_j, tau) = eps d_j theta[kappa](0, tau) + eps^3 d_j^3 theta[kappa](0, tau)/6 + ...
```

(only odd powers). Hence the `j`-th component of every odd gradient at once:

```
d_j theta[kappa](0, tau) = theta[kappa](eps e_j, tau) / eps + R_j,
```

and **g evaluations** of all theta values (at `z = eps e_1, ..., eps e_g`) give the full
matrix `K`. This replaces the `(ord + 1)^g = 2^g` evaluations that
`acb_theta_jet(ord = 1)` performs internally (256 at g = 8), plus its order-3
derivative pass for error bounds.

### Certified remainder

`acb_theta_ql_local_bound(c, rho, z = 0, tau, ord = 0)` (FLINT) gives `c, rho` with
`|theta[a,b](z, tau)| <= c` for all characteristics on the polydisc `|z_i| <= rho`.
Cauchy's estimates give `|d_j^k theta(0)| / k! <= c / rho^k`, so

```
|R_j| <= (1/eps) sum_{k >= 3} c (eps/rho)^k = c eps^2 / (rho^3 (1 - eps/rho)).
```

This bound is added to the radius of every entry of column `j`. It is the same
Cauchy-bound device used for the error propagation in the patched `acb_theta` (branch
`theta-optimize` of flint), so the result is a rigorous ball.

### Choice of `eps` and working precision

Let `prec` be the precision of `tau`. The truncation error must be below the target:
`c eps^2 / rho^3 <= 2^-prec |grad|`, roughly `eps = 2^-e` with

```
e = ceil((prec + log2(c / rho^3) + 8) / 2).
```

The value `theta[kappa](eps e_j)` is of size `eps |grad|`, so an absolute error
`2^-w` of the evaluation becomes a relative error `2^(e - w)` of the gradient:
evaluate at

```
w = prec + e + 16  (about 1.5 prec + log2(c / rho^3)/2 + 24).
```

Example g = 8, prec = 256, `c / rho^3 ~ 2^40`: `e = 152`, `w = 424`.

The evaluation: one call `acb_theta_jet(th, zs, nb = g, tau, ord = 0, ab = 0,
all = 1, sqr = 0, w)` with the `g` vectors `zs = [eps e_1, ..., eps e_g]` (exact
dyadic entries). Passing all `g` vectors in one call shares the tau-dependent setup
(reduction, auxiliary vector, `ql_setup`) across the points (paper, Remark 4.12 (1)).
`theta[kappa](eps e_j)` is read off for the odd `kappa` in the order of
`odd_theta_characteristics(g)` (index `char_to_index`), divided by `eps`
(`acb_mul_2exp_si`, exact), the remainder bound added, and the result rounded to
`prec`.

Expected cost at g = 8, 256 bits with the patched FLINT: one `acb_theta_all`-like
evaluation at ~420 bits and z != 0 for 8 points, estimated 1-3 minutes single-thread
(to be measured; z != 0 exercises `reduce_z` and the auxiliary-vector path that the
z = 0 benchmarks did not).

## 3. API

```julia
odd_theta_gradients(tau::AcbMatrix; precision::Int = precision(base_ring(tau)),
                    verbose::Bool = false) -> AcbMatrix, NamedTuple
```

- `tau`: Siegel reduced (as `tau_red` in `reconstruct_quadrics_data`); exactly
  symmetric. The radii of `tau` are propagated by `acb_theta` as usual.
- Returns `K` (`n_odd x g`, rows in the order of `odd_theta_characteristics(g)`,
  `K[r, j] = d_j theta[odds[r]](0, tau)`, in the coordinates of `z`, i.e. of the
  normalized differentials) and `info = (eps_bits = e, working_precision = w,
  truncation_log2, relative_radius_log2, time)`.
- Precision of the entries of `K`: `base_ring(tau)` (`prec`).

Helper, exported for tests: `theta_values_at(tau, zs::Vector{Vector{AcbFieldElem}}, prec)
-> Vector{Dict}` wrapping the `nb > 1` call of `acb_theta_jet` (ord 0, all
characteristics), keyed like `Hecke.thetas`.

### Integration in `ReconstructGenusG.jl`

- `reconstruct_quadrics_data(tau; gradient_method = :theta, ...)`: new keyword,
  `:theta` (default; this spec) or `:fay` (current behaviour, kept for comparison and
  for the constants-only path). With `:theta`, `K = odd_theta_gradients(tau_red)` is
  computed in `reconstruct_quadrics_data` and passed to
  `reconstruct_quadrics_from_thetas(g, thetas; gradients = K, ...)` (new optional
  argument; `nothing` means compute the Fay kernel as now).
- `reconstruct_quadrics_from_thetas` (theta constants only, no `tau`) is unchanged in
  behaviour: it still uses the Fay kernel for the curve when no `gradients` are given.
- The Prym kernel `K_prym = odd_theta_gradient_kernel(g - 1, thetas_prym)` is unchanged
  in both paths.
- `coordinates = true`: with `:theta`, `K` already is the gradient matrix, so
  `B = identity` and `info.coordinates = -Inf`; `_gradient_coordinates` is still run
  when `verbose` or a new `check_coordinates = true` is set (it recomputes `g + 3`
  gradients with `acb_theta_jet` per characteristic: a cheap independent check of
  this spec's evaluation).
- `info.times` gains `:theta_gradients`; `info.gradients` (the Fay residual `q_K`)
  becomes `info.truncation_log2` of the evaluation when `:theta` is used.

### Usage

```julia
tau = small_period_matrix(riemann_surface(f, 300))            # genus 8 curve
T, tau_red = Hecke.siegel_reduction(tau)
K, info = odd_theta_gradients(tau_red)                         # 32640 x 8, ~minutes
# the full pipeline, curve gradients from tau, Prym gradients from Fay in genus 7
data = reconstruct_quadrics_data(tau; sign_data = pairs, gradient_method = :theta)
# comparison with the current method (genus <= 7: minutes; genus 8: hours)
data_fay = reconstruct_quadrics_data(tau; sign_data = pairs, gradient_method = :fay)
```

## 4. Tests (`ReconstructGenusGTests.jl`)

1. **Against `acb_theta_jet` per characteristic** (certified, independent code path):
   for g = 4, 5, 6 and a few odd characteristics, `odd_theta_gradients` and
   `theta_gradients_single` overlap, and the radii of the new values are at most a few
   bits larger.
2. **Against the Fay kernel**: for g = 5 (and 6), `K_fay = odd_theta_gradient_kernel(g,
   thetas)` and `K = odd_theta_gradients(tau_red)` satisfy `K_fay = K B` for the
   `g x g` matrix `B` solved from `g` well conditioned rows, with residual on all rows
   below `2^(-prec/2)` (this is `_gradient_coordinates` with the roles swapped).
3. **Pipeline**: `reconstruct_quadrics_data` with `:theta` and `:fay` give the same
   quadrics after the change of coordinates `B` (genus 5 test curve
   `canonical_test_curve_g5`), and `reconstruct_rational_quadrics` recognizes the
   same rational quadrics with `:theta`.
4. **Precision**: the relative radius of `K` is about `2^-prec` (not dominated by the
   truncation term) for prec = 128, 256, 512 at g = 5.
5. **Timing** (not a test; recorded in `TIMINGS.md` style): `odd_theta_gradients` vs
   `odd_theta_gradient_kernel` at g = 5, 6, 7, and `odd_theta_gradients` alone at g = 8.

## 5. Out of scope / later

- Prym gradients from a Prym period matrix (would need the Prym's period matrix,
  which the Schottky-Jung construction does not provide).
- Sparse Float64 phase of `_sparse_kernel` (SPQR / UMFPACK), fill-free ordering of
  Fay relations, orbit-structured relation sets: needed for the Prym at curve genus 9.
- Threading inside `acb_theta` beyond the current `acb_theta_sum` parallelization.

## 6. Open points

- The size of `c / rho^3` from `acb_theta_ql_local_bound` at z = 0: if it is much
  larger than 2^40 for reduced `tau` of genus 7-8, `w` grows and the evaluation gets
  slower; measure on real period matrices, and if needed use a tighter bound from
  the actual `|theta|` values (the Cauchy bound with `c = max |theta[kappa](z)|` on a
  smaller polydisc, obtained from the same evaluation at a second radius).
- Whether `acb_theta_jet` with `nb = g` at z != 0 is as fast as hoped after the
  `theta-optimize` changes; otherwise fall back to `g` separate `acb_theta_all` calls
  (same cost order, more setup).
- Exact symmetry of `tau` must hold (asymmetric input makes `acb_siegel_reduce` fail
  silently and `acb_theta` return indeterminate balls); `odd_theta_gradients`
  should symmetrize like `theta_constants_duplication` does.
