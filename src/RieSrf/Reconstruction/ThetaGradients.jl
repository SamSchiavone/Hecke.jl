################################################################################
#
#  ThetaGradients.jl : the gradients of the odd theta functions at z = 0 from
#  the period matrix (see ThetaGradientsSpec.md)
#
#  For an odd characteristic kappa, theta[kappa](z, tau) is an odd function of
#  z with theta[kappa](0) = 0, so
#
#    theta[kappa](eps e_j, tau) / eps = d_j theta[kappa](0, tau) + R_j,
#    |R_j| <= c eps^2 / (rho^3 (1 - eps/rho)),
#
#  where |theta[a, b](z, tau)| <= c on the polydisc |z_i| <= rho for all
#  characteristics (acb_theta_ql_local_bound) and R_j is bounded by Cauchy's
#  estimates (only odd powers of eps occur). One evaluation of all theta
#  values at z = eps e_j gives the j-th component of every odd gradient, so
#  g evaluations (one acb_theta_jet call with g vectors, sharing the setup
#  that depends on tau only) give the whole matrix K of odd_theta_gradient_kernel
#  (ReconstructGenusG.jl): the actual gradients, in the coordinates of z.
#
#  Curve vs. Prym: this replaces the Fay kernel for the curve, whose period
#  matrix we have. The Prym (genus g - 1) has theta constants (Schottky-Jung
#  products) but no period matrix: its gradient kernel stays
#  odd_theta_gradient_kernel(g - 1, thetas_prym).
#
#  Development code: load in Main after `using Hecke` (or Oscar), before
#  ReconstructGenusG.jl.
#
################################################################################

isdefined(@__MODULE__, :RSR) || (global RSR = Hecke.RiemannSurfaces)

using Hecke.RiemannSurfaces: odd_theta_characteristics

# c, rho with |theta[a, b](z, tau)| <= c for all characteristics and all z with
# |z_i| <= rho (FLINT's acb_theta_ql_local_bound at z = 0, ord = 0), as a
# certified upper bound of c and lower bound of rho in the 64 bit real field
function _theta_local_bound(tau::AcbMatrix)
  g = nrows(tau)
  CC = base_ring(tau)
  RR = ArbField(precision(CC))
  c = RR()
  rho = RR()
  z = Hecke.acb_vec([zero(CC) for _ in 1:g])
  ccall((:acb_theta_ql_local_bound, Hecke.libflint), Nothing,
        (Ref{ArbFieldElem}, Ref{ArbFieldElem}, Ptr{Hecke.acb_struct}, Ref{AcbMatrix}, Int),
        c, rho, z, tau, 0)
  Hecke.acb_vec_clear(z, g)
  RR64 = ArbField(64)
  c_up = RR64(Hecke.midpoint(c) + Hecke.radius(c))
  rho_low = RR64(Hecke.midpoint(rho) - Hecke.radius(rho))
  @req isfinite(c_up) && is_positive(rho_low) "acb_theta_ql_local_bound gave c = $c, rho = $rho (tau not reduced, or Im(tau) not positive definite?)."
  return c_up, rho_low
end

# All theta values theta[a, b](z, tau) at the vectors zs (each of length g),
# at precision w: the FLINT vector of acb_theta_jet (ord 0, all
# characteristics) as a Vector of length nb * 4^g, the values of the k-th
# vector at positions (k - 1) 4^g + 1 .. k 4^g in the order of
# Hecke.theta_characteristics_indices (bits of (a..., b...), most significant
# first).
function _theta_values_at(tau::AcbMatrix, zs::Vector{Vector{AcbFieldElem}}, w::Int)
  g = nrows(tau)
  nb = length(zs)
  nbth = 4^g
  CCw = AcbField(w)
  tau_w = change_base_ring(CCw, tau)
  flat = reduce(vcat, [[CCw(x) for x in z] for z in zs])
  @req length(flat) == nb * g "Each z must have length $g."
  th = Hecke.acb_vec(nb * nbth)
  zz = Hecke.acb_vec(flat)
  ccall((:acb_theta_jet, Hecke.libflint), Nothing,
        (Ptr{Hecke.acb_struct}, Ptr{Hecke.acb_struct}, Int, Ref{AcbMatrix}, Int, UInt, Cint, Cint, Int),
        th, zz, nb, tau_w, 0, UInt(0), Cint(1), Cint(0), w)
  values = Hecke.array(CCw, th, nb * nbth)
  Hecke.acb_vec_clear(th, nb * nbth)
  Hecke.acb_vec_clear(zz, nb * g)
  return values
end

# position (0-based, FLINT) of the characteristic (a..., b...) in the theta
# vectors: its bits, most significant first
_theta_position(kappa) = foldl((acc, x) -> (acc << 1) | x, kappa; init = 0)

@doc raw"""
    odd_theta_gradients(tau::AcbMatrix; parallel = Threads.nthreads() > 1, verbose = false) -> AcbMatrix, NamedTuple

The gradients grad theta[kappa](0, tau) of the odd theta functions, as the
matrix K (rows: the odd characteristics in the order of
odd_theta_characteristics(g); g columns, the derivatives d/dz_1, ..., d/dz_g),
with certified radii: a drop-in for the Fay kernel of
odd_theta_gradient_kernel(g, thetas) when the period matrix is known (the
kernel is the gradients up to a common linear transformation; these are the
gradients themselves, in the coordinates of z). From g evaluations of all
theta values at z = eps e_j, see the header of ThetaGradients.jl.

tau should be Siegel reduced (as tau_red of reconstruct_quadrics_data) and is
symmetrized. With `parallel` the g evaluations run as separate calls on the
Julia threads (the cost is dominated by low precision sign determination,
which is then split over the threads); otherwise one call with g vectors. The entries have the precision of tau. `info`: `eps_bits` (eps =
2^-eps_bits), `working_precision`, `bound_bits` (log2 of c / rho^3 of the
Cauchy bound: it decides eps and the working precision),
`truncation_log2` (log2 of the remainder bound, absolute), `relative_radius_log2`
(of K) and `time`.
"""
function odd_theta_gradients(tau::AcbMatrix; parallel::Bool = Threads.nthreads() > 1, verbose::Bool = false)
  start_time = time()
  g = nrows(tau)
  @req ncols(tau) == g "tau must be square."
  CC = base_ring(tau)
  prec = precision(CC)
  tau = (tau + transpose(tau)) * inv(CC(2))
  # the Cauchy bound: eps and the working precision
  c_up, rho_low = _theta_local_bound(tau)
  bound_bits = RSR._arb_mid_2exp(c_up) - 3 * (RSR._arb_mid_2exp(rho_low) - 1)   # >= log2(c / rho^3)
  e = cld(prec + max(bound_bits, 0) + 8, 2)
  # (the same Cauchy constant enters FLINT's error propagation at z != 0:
  # about bound_bits of the evaluation are lost, so add them)
  w = prec + e + 16 + max(bound_bits, 0)
  verbose && println("odd_theta_gradients: g = $g, prec = $prec, log2(c/rho^3) <= $bound_bits, eps = 2^-$e, working precision $w")
  # the remainder bound c eps^2 / (rho^3 (1 - eps/rho)) (ball arithmetic:
  # _add_error! uses its upper bound)
  RR64 = parent(c_up)
  eps64 = RR64(2)^(-e)
  remainder = c_up * eps64^2 / (rho_low^3 * (1 - eps64 / rho_low))
  @req isfinite(remainder) && is_positive(rho_low - eps64) "eps = 2^-$e is not below rho = $rho_low."
  # all theta values at eps e_1, ..., eps e_g
  CCw = AcbField(w)
  eps = one(CCw)
  RSR._acb_mul_2exp!(eps, eps, -e)
  zs = [[i == j ? eps : zero(CCw) for i in 1:g] for j in 1:g]
  nbth = 4^g
  if parallel
    # one acb_theta_jet call per thread, with the g points split into
    # Threads.nthreads() groups: each call repeats the tau-dependent setup
    # (so fewer, larger groups), while the per point sign determination,
    # which dominates, is shared within a group and split over the threads
    groups = [collect(k:Threads.nthreads():g) for k in 1:min(Threads.nthreads(), g)]
    t_eval = @elapsed begin
      parts = Vector{Vector{AcbFieldElem}}(undef, length(groups))
      Threads.@threads for i in eachindex(groups)
        parts[i] = _theta_values_at(tau, zs[groups[i]], w)
      end
      # back to the order of the points
      values = Vector{AcbFieldElem}(undef, g * nbth)
      for (i, group) in enumerate(groups), (k, j) in enumerate(group)
        values[(j - 1) * nbth + 1:j * nbth] = parts[i][(k - 1) * nbth + 1:k * nbth]
      end
    end
  else
    t_eval = @elapsed values = _theta_values_at(tau, zs, w)
  end
  verbose && println("theta values at $g points evaluated$(parallel ? " on $(Threads.nthreads()) threads" : "") [$(round(t_eval, digits = 2)) s]")
  @req all(isfinite, values) "acb_theta_jet returned non-finite values (tau not reduced, or not exactly symmetric?)."
  # K[r, j] = theta[odds[r]](eps e_j) / eps + remainder
  odds = odd_theta_characteristics(g)
  n = length(odds)
  K = zero_matrix(CC, n, g)
  for (r, kappa) in enumerate(odds)
    pos = _theta_position(kappa)
    for j in 1:g
      x = values[(j - 1) * nbth + pos + 1]
      RSR._acb_mul_2exp!(x, x, e)
      RSR._add_error!(x, remainder)
      K[r, j] = CC(x)
    end
  end
  info = (eps_bits = e, working_precision = w, bound_bits = bound_bits,
          truncation_log2 = RSR._arb_mid_2exp(remainder),
          relative_radius_log2 = RSR._log2_relative_radius(K),
          time = time() - start_time, time_evaluation = t_eval)
  verbose && println("K: $n x $g, relative radius 2^$(round(info.relative_radius_log2, digits = 1)) [$(round(info.time, digits = 2)) s]")
  return K, info
end
