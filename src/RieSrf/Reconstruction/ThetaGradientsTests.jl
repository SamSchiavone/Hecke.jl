################################################################################
#
#  ThetaGradientsTests.jl : tests and timings of odd_theta_gradients
#  (ThetaGradients.jl) against theta_gradients_single (FLINT's acb_theta_jet
#  per characteristic, certified) and against the Fay kernel
#  odd_theta_gradient_kernel (ReconstructGenusG.jl).
#
#  Development code: load in Main after ThetaGradients.jl (and, for the Fay
#  comparison and the pipeline test, after ReconstructGenusG.jl and
#  ReconstructGenusGTests.jl).
#
#    test_theta_gradients(g; prec = 256)           # @testset, g <= 6 quick
#    time_theta_gradients([5, 6, 7]; prec = 256)   # timings vs the Fay kernel
#    time_theta_gradients([8]; prec = 256, fay = false)
#    test_theta_gradients_pipeline(; prec = 300)   # genus 5 curve, :theta vs :fay
#
################################################################################

using Test
using Random

# A random reduced period matrix of genus g (as in the benchmarks of
# Elkies-Kieffer, section 5): Re uniform in [-1/2, 1/2], Im = B^T B with
# B = [I v; 0 sqrt(3)/2], v random in [-1/2, 1/2]^(g-1); then Siegel reduced
# and symmetrized. Deterministic for a given seed.
function random_period_matrix(g::Int, prec::Int; seed::Int = 1)
  rng = MersenneTwister(seed)
  CC = AcbField(prec)
  RR = ArbField(prec)
  B = zero_matrix(RR, g, g)
  for i in 1:g
    B[i, i] = i < g ? one(RR) : sqrt(RR(3)) / 2
    i < g && (B[i, g] = RR(rand(rng) - 0.5))
  end
  Y = transpose(B) * B
  X = zero_matrix(RR, g, g)
  for i in 1:g, j in i:g
    X[i, j] = RR(rand(rng) - 0.5)
    X[j, i] = X[i, j]
  end
  tau = zero_matrix(CC, g, g)
  for i in 1:g, j in 1:g
    tau[i, j] = CC(X[i, j], Y[i, j])
  end
  _, tau = Hecke.siegel_reduction(tau)
  tau = (tau + transpose(tau)) * inv(CC(2))
  # exact midpoints, exactly symmetric
  for i in 1:g, j in 1:g
    tau[i, j] = j < i ? tau[j, i] : RSR._acb_mid(tau[i, j])
  end
  return tau
end

# |z| at machine precision (RSR._abs64 refers to RSR inside the module, which
# is only defined in Main)
_abs64_(z::AcbFieldElem) = abs(RSR._c64(z))

# log2 of the largest relative difference |x - y| / max |y| over the entries
# (midpoints), and whether all balls overlap
function _compare_values(X::AcbMatrix, Y::AcbMatrix)
  scale = maximum(_abs64_(Y[i, j]) for i in 1:nrows(Y), j in 1:ncols(Y))
  difference = maximum(_abs64_(X[i, j] - Y[i, j]) for i in 1:nrows(Y), j in 1:ncols(Y))
  overlap = all(overlaps(X[i, j], Y[i, j]) for i in 1:nrows(Y), j in 1:ncols(Y))
  return RSR._safe_log2(difference / scale), overlap
end

# K_fay = K B for the g x g matrix B solved from g well conditioned rows:
# log2 of the relative residual on all rows (as _gradient_coordinates with
# the roles swapped)
function _kernel_vs_gradients(K_fay::AcbMatrix, K::AcbMatrix)
  g = ncols(K)
  CC = base_ring(K)
  rows = RSR._numerical_kernel_data(K; nullity = 0).pivot_rows
  Kr = matrix(CC, length(rows), g, [K[r, j] for r in rows for j in 1:g])
  Fr = matrix(CC, length(rows), g, [K_fay[r, j] for r in rows for j in 1:g])
  B = RSR._solve_precond(Kr, Fr)
  R = map_entries(RSR._acb_mid, K) * map_entries(RSR._acb_mid, B) - map_entries(RSR._acb_mid, K_fay)
  scale = maximum(_abs64_(K_fay[i, j]) for i in 1:nrows(K_fay), j in 1:g)
  residual = maximum(_abs64_(R[i, j]) for i in 1:nrows(R), j in 1:g)
  return RSR._safe_log2(residual / scale)
end

@doc raw"""
    test_theta_gradients(g; prec = 256, characteristics = 6, seed = 1, fay = (g <= 5))

A @testset for odd_theta_gradients on a random reduced period matrix of genus
g: the values overlap those of theta_gradients_single for `characteristics`
random odd characteristics and agree to about prec bits; the relative radius
of K is at most 2^(-prec + 40); with `fay` (needs ReconstructGenusG.jl and
the Fay relations of genus g) the Fay kernel is a linear image of K.
"""
function test_theta_gradients(g::Int; prec::Int = 256, characteristics::Int = 6, seed::Int = 1,
                              fay::Bool = (g <= 5), verbose::Bool = false)
  @testset "odd theta gradients, genus $g, $prec bits" begin
    tau = random_period_matrix(g, prec; seed = seed)
    K, info = odd_theta_gradients(tau; verbose = verbose)
    odds = odd_theta_characteristics(g)
    @test nrows(K) == length(odds) && ncols(K) == g
    @test info.relative_radius_log2 <= -prec + 40
    rng = MersenneTwister(seed)
    chosen = sort(randperm(rng, length(odds))[1:characteristics])
    # (ThetaDuplication.jl is loaded into Main, or into the module)
    @req isdefined(Main, :theta_gradients_single) || isdefined(RSR, :theta_gradients_single) "theta_gradients_single is not loaded: include ThetaDuplication.jl first."
    gradients_single = isdefined(Main, :theta_gradients_single) ? Main.theta_gradients_single : RSR.theta_gradients_single
    single = gradients_single(tau, odds[chosen])
    G = matrix(base_ring(tau), characteristics, g, [single[Tuple(odds[r])][j] for r in chosen for j in 1:g])
    Kc = matrix(base_ring(tau), characteristics, g, [K[r, j] for r in chosen for j in 1:g])
    log2_difference, overlap = _compare_values(Kc, G)
    verbose && println("vs theta_gradients_single: log2 difference $log2_difference, overlap $overlap")
    @test overlap
    @test log2_difference <= -prec + 40
    if fay
      thetas = Hecke.thetas([zero(base_ring(tau)) for _ in 1:g], tau)
      K_fay, q = odd_theta_gradient_kernel(g, thetas)
      residual = _kernel_vs_gradients(K_fay, K)
      verbose && println("Fay kernel = K B: log2 residual $residual (Fay residual $q)")
      @test residual <= -prec / 2
    end
  end
end

@doc raw"""
    time_theta_gradients(genera; prec = 256, seed = 1, fay = true, verbose = false) -> Vector

For each genus: the time of odd_theta_gradients on a random reduced period
matrix (and its eps, working precision, radius), and with `fay` the time of
Hecke.thetas plus odd_theta_gradient_kernel (the current method; needs the
Fay relations of the genus: _fay_rhos for g = 4, 5, the files
_default_fay_data_file(g) otherwise, or find_fay_relations, slow). Prints a
table and returns the rows.
"""
function time_theta_gradients(genera; prec::Int = 256, seed::Int = 1, fay::Bool = true, verbose::Bool = false)
  rows = NamedTuple[]
  for g in genera
    tau = random_period_matrix(g, prec; seed = seed)
    t_new = @elapsed (K, info) = odd_theta_gradients(tau; verbose = verbose)
    row = (g = g, n = nrows(K), time_gradients = t_new, eps_bits = info.eps_bits,
           working_precision = info.working_precision, bound_bits = info.bound_bits,
           radius_log2 = info.relative_radius_log2)
    if fay
      t_thetas = @elapsed thetas = Hecke.thetas([zero(base_ring(tau)) for _ in 1:g], tau)
      t_fay = @elapsed (K_fay, q) = odd_theta_gradient_kernel(g, thetas; verbose = verbose)
      residual = _kernel_vs_gradients(K_fay, K)
      row = merge(row, (time_thetas = t_thetas, time_fay_kernel = t_fay, fay_residual_log2 = q,
                        agreement_log2 = residual))
    end
    push!(rows, row)
    println("g = $g (n = $(row.n)): odd_theta_gradients $(round(t_new, digits = 2)) s ",
            "(eps 2^-$(info.eps_bits), $(info.working_precision) bits, radius 2^$(round(info.relative_radius_log2, digits = 1)))",
            fay ? "; thetas $(round(row.time_thetas, digits = 2)) s + Fay kernel $(round(row.time_fay_kernel, digits = 2)) s, agreement 2^$(round(row.agreement_log2, digits = 1))" : "")
  end
  return rows
end

@doc raw"""
    test_theta_gradients_pipeline(; prec = 300, g = 5, coefficients = -3:3, kw...)

reconstruct_quadrics_data on a random genus g curve (canonical_test_curve_g5
/ _g6 of ReconstructGenusGTests.jl) with gradient_method = :theta and
:fay: the same number of quadrics, and with coordinates = true the quadrics
of :theta are those of :fay moved by its coordinate change (compared in
numerical reduced echelon form via compare_reconstructed_quadrics, both
against the exact quadrics of the curve). Prints the times of both.
"""
function test_theta_gradients_pipeline(; prec::Int = 300, g::Int = 5, coefficients = -3:3, kw...)
  curve = g == 5 ? canonical_test_curve_g5(; coefficients = coefficients) :
          g == 6 ? canonical_test_curve_g6(; coefficients = coefficients) :
                   canonical_test_curve(g; coefficients = coefficients)
  @testset "reconstruction with theta gradients, genus $g" begin
    r_theta = compare_reconstructed_quadrics(curve, prec; gradient_method = :theta, kw...)
    r_fay = compare_reconstructed_quadrics(curve, prec; gradient_method = :fay, kw...)
    println("genus $g, $prec bits: :theta $(round(r_theta.time_reconstruction, digits = 2)) s ",
            "(log2 difference to the exact quadrics $(round(r_theta.log2_difference, digits = 1))), ",
            ":fay $(round(r_fay.time_reconstruction, digits = 2)) s ($(round(r_fay.log2_difference, digits = 1)))")
    @test r_theta.log2_difference <= -prec / 2
    @test r_fay.log2_difference <= -prec / 2
  end
end

################################################################################
#  The sparse Float64 phase of the Fay kernel (for the Prym)
################################################################################

@doc raw"""
    time_fay_kernel(g; prec = 256, seed = 1, stored = true, dense = true, unit = true, gradients = true) -> NamedTuple

The Fay kernel odd_theta_gradient_kernel(g, thetas) of a random reduced
period matrix, in up to three ways: `stored` relations (the basis of
find_fay_relations: _fay_rhos for g <= 5, the files G6FayRelations,
G7FayRelations, ... in SignMatrices otherwise) with the sparse Float64
phase (SuiteSparse) and, with `dense`, with the dense one (the code before;
skip for g >= 8); and `unit`: all relations of the unit rhos, no basis
selection (`rhos = :unit, select = false`; needs no stored file, the way
for g >= 8). Times, residuals, and log2 of the relative differences after
the common linear map (the kernels agree up to GL_g); with `gradients`
also against odd_theta_gradients (g = 8: minutes to an hour). Needs
ReconstructGenusG.jl.
"""
function time_fay_kernel(g::Int; prec::Int = 256, seed::Int = 1, stored::Bool = true, dense::Bool = true,
                         unit::Bool = true, gradients::Bool = true, verbose::Bool = false)
  tau = random_period_matrix(g, prec; seed = seed)
  t_thetas = @elapsed thetas = Hecke.thetas([zero(base_ring(tau)) for _ in 1:g], tau)
  n = length(odd_theta_characteristics(g))
  result = (g = g, n = n, time_thetas = t_thetas)
  println("g = $g (n = $n): thetas $(round(t_thetas, digits = 2)) s")
  K = nothing
  if gradients
    t_grad = @elapsed (K, _) = odd_theta_gradients(tau)
    result = merge(result, (time_gradients = t_grad,))
    println("          odd_theta_gradients $(round(t_grad, digits = 2)) s")
  end
  vs(Kx) = K === nothing ? NaN : _kernel_vs_gradients(Kx, K)
  K_sparse = nothing
  if stored
    t_sparse = @elapsed (K_sparse, q_sparse) = odd_theta_gradient_kernel(g, thetas; sparse = true, verbose = verbose)
    agree_sparse = vs(K_sparse)
    result = merge(result, (time_sparse = t_sparse, residual_sparse = q_sparse, agreement_sparse = agree_sparse))
    println("          Fay kernel, stored relations, sparse $(round(t_sparse, digits = 2)) s (residual 2^$(round(q_sparse, digits = 1)), vs gradients 2^$(round(agree_sparse, digits = 1)))")
  end
  if unit
    t_unit = @elapsed (K_unit, q_unit) = odd_theta_gradient_kernel(g, thetas; rhos = :unit, select = false, sparse = true, verbose = verbose)
    agree_unit = vs(K_unit)
    unit_vs_stored = K_sparse === nothing ? NaN : _kernel_vs_gradients(K_unit, K_sparse)
    result = merge(result, (time_unit = t_unit, residual_unit = q_unit, agreement_unit = agree_unit, unit_vs_stored = unit_vs_stored))
    println("          Fay kernel, all relations of the unit rhos, sparse $(round(t_unit, digits = 2)) s (residual 2^$(round(q_unit, digits = 1)), vs gradients 2^$(round(agree_unit, digits = 1)), vs stored 2^$(round(unit_vs_stored, digits = 1)))")
  end
  if dense && stored
    t_dense = @elapsed (K_dense, q_dense) = odd_theta_gradient_kernel(g, thetas; sparse = false, verbose = verbose)
    agree_dense = vs(K_dense)
    # (the two kernels are normalized on possibly different free columns:
    # compare up to the common linear map)
    diff = _kernel_vs_gradients(K_dense, K_sparse)
    result = merge(result, (time_dense = t_dense, residual_dense = q_dense, agreement_dense = agree_dense,
                            sparse_vs_dense = diff))
    println("          Fay kernel dense $(round(t_dense, digits = 2)) s (residual 2^$(round(q_dense, digits = 1)), vs gradients 2^$(round(agree_dense, digits = 1))); sparse vs dense kernel 2^$(round(diff, digits = 1))")
  end
  return result
end
