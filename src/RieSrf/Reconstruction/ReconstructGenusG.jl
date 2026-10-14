################################################################################
#
#  ReconstructGenusG.jl : the quadrics through the canonical model of a
#  generic curve of genus g >= 5 from its theta constants
#
#  The genus 4 method (ReconstructG4.jl) with the Prym of eta = (0..0 1 0..0)
#  (b_1 = 1), without tables:
#
#   1. The gradients l_kappa = grad theta[kappa](0) of the odd theta functions
#      satisfy linear relations whose coefficients are theta constants (Fay's
#      relations, construct_fay_matrix). The kernel K of these relations has
#      dimension g: its rows are the l_kappa up to a common linear
#      transformation (the tangent hyperplanes of the canonical model).
#   2. The same for the Prym (genus g - 1) with its theta constants
#      theta_P[d]^2 = theta[0 d1; 0 d2] theta[0 d1; 1 d2] (Schottky-Jung): the
#      rows m_d of K_P. The signs of the square roots come from Riemann's
#      relations with the precomputed (term, matrix) pairs of
#      SignCorrectionFromMatrices.jl.
#   3. A linear relation sum_d c_d m_d^2 = 0 between the squares gives the
#      quadric sum_d c_d l_kappa(d) l_kappa(d)+eta through the canonical model
#      (kappa(d) = (0 d1; 0 d2)). These span the (g - 2)(g - 3)/2 quadrics
#      that contain it (for g = 5 and a non-trigonal curve: the curve is their
#      intersection).
#
#  The quadrics are in the coordinates of the kernel K. With the big period
#  matrix and the gradients of the theta functions (theta_jets) they are
#  moved to the coordinates of the differentials, where they are defined over
#  the field of the curve (reconstruct_rational_quadrics; as
#  reconstruct_rational_curve_g4).
#
#  Development code: load in Main after `using Oscar` (the sign correction
#  uses Oscar's isometry groups and load), after ReconstructCurveAuxiliary.jl,
#  RiemannRelations.jl and SignCorrectionFromMatrices.jl. ThetaCharacteristics.jl
#  belongs to the module (RieSrf.jl); its functions are imported below. Uses the internals of Hecke.RiemannSurfaces
#  (numerical_kernel, _solve_precond, the recognition and normalization of
#  ReconstructG4.jl). The sign data of the Prym (genus g - 1) are read with
#  load_sign_data, by default from SignMatrices/G$(g - 1)Matrices next to this
#  file; pass `sign_data` (a file or the loaded pairs) to reuse them.
#
#  Usage:
#    pairs = load_sign_data("SignMatrices/G4Matrices")
#    Q = reconstruct_quadrics(small_period_matrix(RS); sign_data = pairs)
#    Q = reconstruct_rational_quadrics(big_period_matrix(RS); sign_data = pairs)
#    check_rational_quadrics(f, 500; sign_data = pairs)
#
################################################################################

isdefined(@__MODULE__, :RSR) || (global RSR = Hecke.RiemannSurfaces)

using Hecke.RiemannSurfaces: odd_theta_characteristics, even_theta_characteristics,
                             char_to_index, lift_prym_indices

_default_sign_data_file(h::Int) = joinpath(@__DIR__, "SignMatrices", "G$(h)Matrices")
_default_fay_data_file(g::Int) = joinpath(@__DIR__, "SignMatrices", "G$(g)FayRelations")

# The rho of the Fay matrices for genus 5 and its genus 4 Prym (enough for a
# kernel of the right dimension); nothing for other genera (then rho are
# added until the kernel has the right dimension).
function _fay_rhos(g::Int)
  F = GF(2)
  g == 5 && return [F.(v) for v in ([1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 1, 0, 0, 1],
                                     [0, 1, 0, 1, 0, 0, 1, 1, 0, 0], [0, 1, 0, 0, 1, 0, 0, 0, 0, 1],
                                     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 0, 1, 1, 1, 0, 1, 1, 0, 1],
                                     [1, 0, 1, 0, 1, 1, 0, 1, 0, 1], [0, 0, 1, 0, 1, 1, 0, 1, 0, 0])]
  g == 4 && return [F.(v) for v in ([1, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 1, 0, 1],
                                     [0, 1, 1, 0, 0, 1, 1, 0], [0, 1, 0, 1, 0, 0, 0, 1],
                                     [1, 1, 1, 1, 1, 1, 1, 1], [1, 0, 1, 1, 0, 1, 0, 1],
                                     [1, 0, 0, 1, 1, 1, 0, 1], [0, 0, 1, 1, 1, 0, 0, 0])]
  return nothing
end

# The nonzero rho in GF(2)^(2g) of weight at most max_weight, by weight (not
# all 2^(2g) of them: for g = 10 these would be 10^6 vectors; a few of weight
# 1 suffice in practice)
function _fay_rho_candidates(g::Int; max_weight::Int = 3)
  F = GF(2)
  n = 2*g
  vectors = Vector{Int}[]
  for w in 1:min(max_weight, n)
    level = Vector{Int}[]
    function subsets!(v, start, left)
      left == 0 && return push!(level, copy(v))
      for i in start:(n - left + 1)
        v[i] = 1
        subsets!(v, i + 1, left - 1)
        v[i] = 0
      end
    end
    subsets!(zeros(Int, n), 1, w)
    append!(vectors, sort!(level))
  end
  return [F.(v) for v in vectors]
end

# The w in {0,1}^(2g) with <rho, w> = 1 (symplectic), as integer vectors
# (naive_non_orth_complement without GF(2) arithmetic)
function _non_orthogonal_doubled(rho::Vector{Int})
  n = length(rho)
  g = div(n, 2)
  result = Vector{Int}[]
  for bits in 0:(2^n - 1)
    w = [(bits >> (n - k)) & 1 for k in 1:n]
    isodd(sum(rho[i] * w[g + i] + w[i] * rho[g + i] for i in 1:g)) && push!(result, w)
  end
  return result
end

# The Fay relations of all rho, of which only n_odd - g + extra are
# evaluated in ball arithmetic: those chosen at machine precision (pivoted QR
# of the row scaled ComplexF64 matrix, built from fay_matrix_structure), the
# first n_odd - g a basis of the relations if they have the expected rank,
# the others a check. Avoids the matrix of all relations (4096 x 496 balls
# for g = 5).
function _selected_fay_structure(g::Int, thetas, rhos, odd_indices::Vector{Int}; extra::Int = 2*g)
  LA = RSR.LinearAlgebra
  structure = reduce(vcat, [fay_matrix_structure(g, rho) for rho in rhos])
  n_odd = length(odd_indices)
  position = Dict(c => k for (k, c) in enumerate(odd_indices))
  values64 = Dict(k => RSR._c64(v) for (k, v) in thetas)
  A64 = zeros(ComplexF64, length(structure), n_odd)
  for (r, terms) in enumerate(structure), (column, coefficient, c1, c2, c3) in terms
    k = get(position, column, 0)
    k == 0 || (A64[r, k] += coefficient * values64[c1] * values64[c2] * values64[c3])
  end
  for r in axes(A64, 1)
    len = maximum(abs, view(A64, r, :))
    len > 0 && (A64[r, :] ./= len)
  end
  # (LAPACK's pivoted QR can hang on non-finite input)
  @req all(isfinite, A64) "Non-finite Fay relations at machine precision (theta constants with infinite radius?)."
  F = LA.qr(Matrix(transpose(A64)), LA.ColumnNorm())
  keep = sort(F.p[1:min(n_odd - g + extra, length(structure))])
  return structure[keep]
end

# The relations of the structure at the theta constants as sparse rows
# (columns: positions in `columns`, values; empty rows dropped)
function _sparse_fay_rows(structure, thetas, columns::Vector{Int})
  position = Dict(c => k for (k, c) in enumerate(columns))
  rows = Tuple{Vector{Int}, Vector{AcbFieldElem}}[]
  # the pairs (c1, c2) = (base, base + rho) repeat across the relations of
  # one rho: cache their products
  pair_products = Dict{Any, AcbFieldElem}()
  for terms in structure
    entries = Dict{Int, AcbFieldElem}()
    for (column, coefficient, c1, c2, c3) in terms
      k = get(position, column, 0)
      k == 0 && continue
      pair = get!(pair_products, (c1, c2)) do
        thetas[c1] * thetas[c2]
      end
      value = coefficient * (pair * thetas[c3])
      entries[k] = haskey(entries, k) ? entries[k] + value : value
    end
    filter!(e -> !iszero(e[2]), entries)
    isempty(entries) && continue
    cols = sort!(collect(keys(entries)))
    push!(rows, (cols, [entries[k] for k in cols]))
  end
  return rows
end

# In place, at precision p (FLINT): z += x y and z += x
_acb_addmul!(z::AcbFieldElem, x::AcbFieldElem, y::AcbFieldElem, p::Int) =
  ccall((:acb_addmul, Hecke.libflint), Nothing,
        (Ref{AcbFieldElem}, Ref{AcbFieldElem}, Ref{AcbFieldElem}, Int), z, x, y, p)
_acb_add!(z::AcbFieldElem, x::AcbFieldElem, y::AcbFieldElem, p::Int) =
  ccall((:acb_add, Hecke.libflint), Nothing,
        (Ref{AcbFieldElem}, Ref{AcbFieldElem}, Ref{AcbFieldElem}, Int), z, x, y, p)

# The kernel (dimension `nullity`) of the sparse matrix with the given rows
# ((columns, values)) and n columns, for the Fay relations (m ~ n rows, few
# entries per row):
#  - rows and columns are scaled by powers of 2 (exact; two rounds), which
#    equilibrates the pivot block (the kernel of A D is D^-1 times that of A);
#  - at machine precision (midpoints): pivot columns from the pivoted QR of A0
#    (its R diagonal checks the rank gap), pivot rows from the pivoted QR of
#    the transpose of the pivot columns (as in numerical_kernel);
#  - the kernel [X; I] at full precision by iterative refinement of
#    B X = -A[rows, free] (B = A[rows, pivots]) with the LU of B0; the sparse
#    residuals are computed at the precision they need (the bits reached so
#    far + 128), not at full precision;
#  - its radius estimated as in RSR._refined_solve (sigma_min(B0) by inverse
#    iteration).
# Returns K (n x nullity) and log2 of the relative residual of the scaled
# matrix on the midpoints of all rows.
function _sparse_kernel(rows, n::Int, nullity::Int; max_iterations::Int = 300, verbose::Bool = false,
                        basis_rows::Bool = false, dense_limit::Int = 4000, big_limit::Int = 600,
                        sparse::Union{Nothing, Bool} = nothing, sparse_limit::Int = 1500,
                        sparse_tol::Float64 = 1e-13)
  LA = RSR.LinearAlgebra
  SA = Hecke.SparseArrays
  start_time = time()
  elapsed() = round(time() - start_time, digits = 1)
  CC = parent(rows[1][2][1])
  prec = precision(CC)
  m = length(rows)
  r = n - nullity
  @req m >= r "Only $m relations for rank $r."
  ZERO = -(Int64(1) << 59)
  mid_exponent(z) = max(ZERO, RSR._arb_mid_2exp(real(z)), RSR._arb_mid_2exp(imag(z)))
  tmp = CC()
  # exact scaling of rows and columns by powers of 2
  scaled = [[RSR._scaled_entry(CC, v, 0) for v in vals] for (_, vals) in rows]
  column_shift = zeros(Int, n)             # A' = A_rows diag(2^-column_shift)
  for _ in 1:2
    for vals in scaled
      e = maximum(mid_exponent, vals)
      e > ZERO && foreach(v -> RSR._acb_mul_2exp!(v, v, -e), vals)
    end
    ce = fill(ZERO, n)
    for ((cols, _), vals) in zip(rows, scaled), (j, v) in zip(cols, vals)
      ce[j] = max(ce[j], mid_exponent(v))
    end
    ce[ce .== ZERO] .= 0
    for ((cols, _), vals) in zip(rows, scaled), (j, v) in zip(cols, vals)
      RSR._acb_mul_2exp!(v, v, -ce[j])
    end
    column_shift .+= ce
  end
  mids = [map(RSR._acb_mid, vals) for vals in scaled]
  wide = false                             # BigFloat pivots (see below)
  # the Float64 matrix: sparse (SuiteSparse) for large n, dense otherwise
  use_sparse = sparse === nothing ? n > sparse_limit : sparse
  if use_sparse
    Is, Js, Vs = Int[], Int[], ComplexF64[]
    for (i, (cols, _)) in enumerate(rows), (k, j) in enumerate(cols)
      push!(Is, i); push!(Js, j); push!(Vs, RSR._mid_c64_scaled(mids[i][k], 0, tmp))
    end
    @req all(isfinite, Vs) "Non-finite Fay relations at machine precision (theta constants with infinite radius?)."
    A0 = SA.sparse(Is, Js, Vs, m, n)
  else
    A0 = zeros(ComplexF64, m, n)
    for (i, (cols, _)) in enumerate(rows), (k, j) in enumerate(cols)
      A0[i, j] = RSR._mid_c64_scaled(mids[i][k], 0, tmp)
    end
    @req all(isfinite, A0) "Non-finite Fay relations at machine precision (theta constants with infinite radius?)."
  end
  verbose && println("A0 built and scaled ($m x $n$(use_sparse ? ", sparse, $(SA.nnz(A0)) entries" : ""); column scales 2^$(-maximum(column_shift)) to 2^$(-minimum(column_shift))) [$(elapsed()) s]")
  if use_sparse
    # sparse rank-revealing QR (SPQR, multifrontal): the column permutation
    # puts the r independent columns first (entries are scaled to about 1,
    # so sparse_tol is an absolute threshold on the R diagonal); then the
    # pivot columns on all rows as in the dense large-n branch, corrections
    # by sparse least squares. No BigFloat fallback here.
    Fa = SA.qr(A0; tol = sparse_tol)
    ra = LA.rank(Fa)
    verbose && println("sparse QR done: numerical rank $ra (expected $r), nnz(R) = $(SA.nnz(Fa.R)) [$(elapsed()) s]")
    @req ra == r "The Fay relations do not have rank $r at machine precision (sparse QR with tol = $sparse_tol finds $ra; adjust sparse_tol, or use sparse = false)."
    piv = sort(Fa.pcol[1:r])
    free = sort(Fa.pcol[r + 1:n])
    prows = collect(1:m)
    Fp = SA.qr(A0[:, piv]; tol = sparse_tol)
    @req LA.rank(Fp) == r "The pivot columns of the Fay relations are rank deficient at machine precision."
    Rp = LA.UpperTriangular(Fp.R)
    pcp = Fp.pcol                          # A0[:, piv][:, pcp] = Q Rp (up to row order)
    solve_correction = D -> Fp \ D
    inverse_normal = v -> (w = zeros(ComplexF64, r); w[pcp] = Rp \ (Rp' \ v[pcp]); w)
  elseif n > dense_limit
    # large n (pivoted QRs too slow: BLAS 2). The Float64 kernel from the
    # unpivoted (BLAS 3) Householder QR A0 = Q R: kernel(A0) = kernel(R), by
    # subspace inverse iteration with R^H R (triangular solves); the free
    # columns where it is well conditioned (pivoted QR of its g x n
    # transpose, cheap). The pivot columns A0[:, piv] (all rows) then have
    # full rank r and are well conditioned: corrections by least squares with
    # their QR, residuals on all rows. (A square block of r chosen rows can
    # be far worse conditioned than all rows together: the stored basis rows
    # gave sigma_min 1e-17, all rows 7e-12 in genus 7.)
    Fa = LA.qr(A0)
    Ra = Matrix(Fa.R)
    da = abs.(LA.diag(Ra))
    floor_value = eps() * maximum(da)
    for i in 1:n
      abs(Ra[i, i]) < floor_value && (Ra[i, i] = floor_value)
    end
    T = LA.UpperTriangular(Ra)
    Y = Matrix(LA.qr(randn(ComplexF64, n, nullity)).Q)
    for _ in 1:4
      Y = Matrix(LA.qr(T \ (T' \ Y)).Q)
    end
    kernel_residual = maximum(abs, A0 * Y)
    verbose && println("kernel QR done: R diagonal from $(maximum(da)) to $(minimum(da)); Float64 kernel residual $kernel_residual [$(elapsed()) s]")
    free = sort(LA.qr(Matrix(transpose(Y)), LA.ColumnNorm()).p[1:nullity])
    piv = setdiff(1:n, free)
    prows = collect(1:m)
    Fp = LA.qr(A0[:, piv])
    Tp = LA.UpperTriangular(Matrix(Fp.R))
    solve_correction = D -> Fp \ D
    inverse_normal = v -> Tp \ (Tp' \ v)
  else
    F = LA.qr(A0, LA.ColumnNorm())
    d = abs.(LA.diag(F.R))
    verbose && println("column QR done: R diagonal $(d[1]) first, $(d[r]) at $r, $(r < min(m, n) ? d[r + 1] : 0.0) after [$(elapsed()) s]")
    clear64 = d[r] > 1e-12 * d[1] && (r == min(m, n) || d[r + 1] < 1e-3 * d[r])
    if clear64
      piv = F.p[1:r]
      free = sort(F.p[r + 1:n])
      prows = LA.qr(Matrix(transpose(A0[:, piv])), LA.ColumnNorm()).p[1:r]
      F0 = LA.lu(A0[prows, piv]; check = false)
      @req LA.issuccess(F0) "The pivot block of the Fay relations is singular at machine precision."
      solve_correction = D -> F0 \ D
      inverse_normal = v -> F0' \ (F0 \ v)
    else
      # not decidable in Float64 (a dynamic range beyond 2^-53, e.g. theta
      # constants near the boundary of the Jacobian locus): the same choices
      # in BigFloat (its default precision), slower (generic pivoted QR);
      # the corrections of the refinement are solved there too
      # (generic BigFloat linear algebra is O(m n^2) without BLAS: hours for
      # the 2016 odd characteristics of genus 6, so only for small n)
      @req n <= big_limit "The Fay relations do not have a clear rank $r at machine precision (R diagonal $(d[r]), $(r < min(m, n) ? d[r + 1] : 0.0)), and n = $n > big_limit = $big_limit is too large for the BigFloat fallback."
      wide = true
      big_bits = precision(BigFloat)
      AB = zeros(Complex{BigFloat}, m, n)
      for (i, (cols, _)) in enumerate(rows), (k, j) in enumerate(cols)
        AB[i, j] = Complex(BigFloat(real(mids[i][k])), BigFloat(imag(mids[i][k])))
      end
      FB = LA.qr(AB, LA.ColumnNorm())
      dB = abs.(LA.diag(FB.R))
      verbose && println("column QR at $big_bits bits: R diagonal $(Float64(dB[r] / dB[1])) at $r, ",
                         "$(r < min(m, n) ? Float64(dB[r + 1] / dB[1]) : 0.0) after (relative) [$(elapsed()) s]")
      @req dB[r] > big(2.0)^(40 - big_bits) * dB[1] && (r == min(m, n) || dB[r + 1] < big(1e-3) * dB[r]) "The Fay relations do not have a clear rank $r at machine precision (R diagonal $(d[r]), $(r < min(m, n) ? d[r + 1] : 0.0)), nor at $big_bits bits (relative $(Float64(dB[r] / dB[1])), $(r < min(m, n) ? Float64(dB[r + 1] / dB[1]) : 0.0))."
      piv = FB.p[1:r]
      free = sort(FB.p[r + 1:n])
      prows = LA.qr(Matrix(transpose(AB[:, piv])), LA.ColumnNorm()).p[1:r]
      FB0 = LA.lu(AB[prows, piv]; check = false)
      @req LA.issuccess(FB0) "The pivot block of the Fay relations is singular at $big_bits bits."
      solve_correction = D -> ComplexF64.(FB0 \ Complex{BigFloat}.(D))
      inverse_normal = v -> ComplexF64.(FB0' \ (FB0 \ Complex{BigFloat}.(v)))
    end
  end
  nres = length(prows)
  # sigma_min of the pivot columns by inverse iteration (B^H B)^-1
  v = ones(ComplexF64, r) / sqrt(r)
  growth = 1.0
  for _ in 1:20
    w = inverse_normal(v)
    growth = LA.norm(w)
    v = w / growth
  end
  sigma = 1 / sqrt(growth)
  verbose && println("pivots chosen, sigma_min of the pivot columns about $sigma [$(elapsed()) s]")
  @req sigma > (wide ? 2.0^(60 - precision(BigFloat)) : 1e-14) "The Fay relations have rank < $r at machine precision (sigma_min of the pivot columns $sigma)."
  # the Float64 residual of the kernel estimate on all rows: rank check
  Xf = -solve_correction(Matrix(A0[prows, free]))
  Kf = zeros(ComplexF64, n, nullity)
  Kf[piv, :] = Xf
  for (c, j) in enumerate(free)
    Kf[j, c] = 1
  end
  residual64 = maximum(abs, A0 * Kf) / maximum(abs, Kf)
  @req residual64 < 1e-6 "The Fay relations have rank > $r at machine precision (relative residual $residual64)."
  verbose && println("Float64 kernel residual $residual64; refining [$(elapsed()) s]")
  # refinement of X (columns X[c], length r) with B X = -A[prows, free]
  pivot_position = Dict(j => k for (k, j) in enumerate(piv))
  free_position = Dict(j => c for (c, j) in enumerate(free))
  pivot_terms = [[(pivot_position[j], a) for (j, a) in zip(rows[i][1], mids[i]) if haskey(pivot_position, j)] for i in prows]
  free_terms = [[(free_position[j], a) for (j, a) in zip(rows[i][1], mids[i]) if haskey(free_position, j)] for i in prows]
  X = [[CC() for _ in 1:r] for _ in 1:nullity]
  R = [[CC() for _ in 1:nres] for _ in 1:nullity]
  vector_exponent(V) = maximum(mid_exponent, V)
  rhs_exponents = zeros(Int, nullity)
  bits = zeros(Int, nullity)                   # bits of the residual reached
  active = trues(nullity)
  previous = fill(typemax(Int), nullity)
  exponents = zeros(Int, nullity)
  # the right hand sides of the corrections: in BigFloat with the BigFloat
  # pivots (rounding them to Float64 would lose everything for a pivot block
  # with condition number beyond 2^53)
  D0 = wide ? zeros(Complex{BigFloat}, nres, nullity) : zeros(ComplexF64, nres, nullity)
  scaled_tmp = CC()
  for iteration in 1:max_iterations
    for c in 1:nullity
      active[c] || continue
      working = min(prec + 32, bits[c] + 128)
      Xc, Rc = X[c], R[c]
      # (the rows are independent: each k writes only Rc[k])
      Threads.@threads for k in 1:nres
        s = Rc[k]
        ccall((:acb_zero, Hecke.libflint), Nothing, (Ref{AcbFieldElem},), s)
        for (cf, a) in free_terms[k]
          cf == c && _acb_add!(s, s, a, working)
        end
        for (p, a) in pivot_terms[k]
          _acb_addmul!(s, a, Xc[p], working)
        end
      end
      e = vector_exponent(Rc)
      iteration == 1 && (rhs_exponents[c] = e)
      scale = max(rhs_exponents[c], vector_exponent(Xc))
      bits[c] = scale - e
      verbose && c == 1 && println("iteration $iteration: residual 2^$(e - scale) at $working bits [$(elapsed()) s]")
      # converged, or stagnating (less than 2 bits per step)
      if e <= ZERO || e <= scale - prec || e > previous[c] - 2
        active[c] = false
        D0[:, c] .= 0
        continue
      end
      previous[c] = e
      exponents[c] = e
      for k in 1:nres
        if wide
          RSR._acb_mul_2exp!(scaled_tmp, Rc[k], -e)
          D0[k, c] = -Complex(BigFloat(Hecke.midpoint(real(scaled_tmp))), BigFloat(Hecke.midpoint(imag(scaled_tmp))))
        else
          D0[k, c] = -RSR._mid_c64_scaled(Rc[k], e, tmp)
        end
      end
    end
    any(active) || break
    D = solve_correction(D0)
    for c in 1:nullity
      active[c] || continue
      for k in 1:r
        z = CC(real(D[k, c]), imag(D[k, c]))
        RSR._acb_mul_2exp!(z, z, exponents[c])
        X[c][k] = X[c][k] + z
      end
    end
  end
  for c in 1:nullity
    scale = max(rhs_exponents[c], vector_exponent(X[c]))
    previous[c] == typemax(Int) || previous[c] <= scale - div(prec, 2) ||
      error("The iterative refinement did not converge (residual 2^$(previous[c] - scale) relative).")
  end
  verbose && println("refined [$(elapsed()) s]")
  # radius estimate
  radius64(z) = Float64(Hecke.radius(real(z))) + Float64(Hecke.radius(imag(z)))
  relative = maximum(maximum(radius64, vals) for vals in scaled)
  RR = ArbField(prec)
  for c in 1:nullity
    largest = maximum(abs(RSR._c64(x)) for x in X[c])
    err = RR(sqrt(r) * (1 + largest) * relative / sigma)
    foreach(x -> RSR._add_error!(x, err), X[c])
  end
  # the kernel of the scaled matrix
  Ks = zero_matrix(CC, n, nullity)
  for (k, j) in enumerate(piv), c in 1:nullity
    Ks[j, c] = X[c][k]
  end
  for (c, j) in enumerate(free)
    Ks[j, c] = one(CC)
  end
  # log2 relative residual on the midpoints of all rows (scaled to 1)
  Kmid = [[RSR._acb_mid(Ks[j, c]) for j in 1:n] for c in 1:nullity]
  eN = maximum(maximum(mid_exponent, Kc) for Kc in Kmid)
  worst = -Inf
  for (i, (cols, _)) in enumerate(rows), c in 1:nullity
    s = zero(CC)
    for (j, a) in zip(cols, mids[i])
      _acb_addmul!(s, a, Kmid[c][j], prec)
    end
    es = mid_exponent(s)
    es > ZERO && (worst = max(worst, Float64(es - eN)))
  end
  # back to the kernel of A: K = diag(2^-column_shift) Ks
  K = zero_matrix(CC, n, nullity)
  for j in 1:n, c in 1:nullity
    x = Ks[j, c]
    RSR._acb_mul_2exp!(x, x, -column_shift[j])
    K[j, c] = x
  end
  return K, worst
end

@doc raw"""
    odd_theta_gradient_kernel(g, thetas; rhos = nothing, relations = nothing) -> AcbMatrix, Float64

The kernel K (rows: the odd characteristics in the order of
odd_theta_characteristics(g), g columns) of Fay's linear relations between
the gradients of the odd theta functions at 0, i.e. the gradients up to a
common linear transformation, and log2 of its relative residual on the
midpoints. The relations, in this order of preference: `relations` (a list
of find_fay_relations or a file of save_fay_relations); those of `rhos` or
of _fay_rhos(g) (genus 4 and 5), of which a basis is chosen for this curve;
the file _default_fay_data_file(g) if it exists; find_fay_relations for
this curve (slow for g >= 6: save its result). Further keywords go to
_sparse_kernel: `sparse` (true/false; by default for n > `sparse_limit` =
1500, i.e. g >= 6: the Float64 phase with SuiteSparse instead of dense QR;
g = 7: 7 s instead of 51 s), `sparse_tol`.
"""
function odd_theta_gradient_kernel(g::Int, thetas; rhos = nothing, relations = nothing, verbose::Bool = false,
                                   kernel_kw...)
  odd_indices = char_to_index.(collect.(odd_theta_characteristics(g)))
  structure, stored_relations = _fay_structure(g, thetas, odd_indices; rhos = rhos, relations = relations)
  verbose && println("structure: $(length(structure)) relations")
  rows = _sparse_fay_rows(structure, thetas, odd_indices)
  verbose && println("sparse rows: $(length(rows)), $(sum(r -> length(r[1]), rows)) entries")
  # stored relations (find_fay_relations): the first n_odd - g are a basis
  stored = stored_relations && length(rows) == length(structure)
  return _sparse_kernel(rows, length(odd_indices), g; verbose = verbose, basis_rows = stored, kernel_kw...)
end

# The Fay relations used by odd_theta_gradient_kernel (see there for the
# order of preference) and whether they are stored ones
function _fay_structure(g::Int, thetas, odd_indices::Vector{Int}; rhos = nothing, relations = nothing)
  rhos === nothing && relations === nothing && (rhos = _fay_rhos(g))
  if relations === nothing && rhos === nothing
    file = _default_fay_data_file(g)
    if isfile(file)
      relations = file
    else
      @info "No Fay relations for genus $g (_default_fay_data_file($g)); searching them for this curve. Save the result of find_fay_relations to skip this."
      relations = find_fay_relations(g, thetas)
    end
  end
  relations === nothing && return _selected_fay_structure(g, thetas, rhos, odd_indices), false
  relations isa AbstractString && (relations = load_fay_relations(relations))
  setup = _fay_setup(g)
  return [_fay_relation_structure(g, w, sigma, setup) for (w, sigma) in relations], true
end

# The smallest `count` singular values (relative to the largest, increasing)
# of the Float64 matrix of the Fay relations, rows and columns scaled by
# powers of 2: g at rounding level and a gap if the kernel has dimension g.
# A diagnostic (an SVD: only for moderate n).
function fay_singular_values64(g::Int, thetas; rhos = nothing, relations = nothing, count::Int = g + 6)
  odd_indices = char_to_index.(collect.(odd_theta_characteristics(g)))
  structure, _ = _fay_structure(g, thetas, odd_indices; rhos = rhos, relations = relations)
  rows = _sparse_fay_rows(structure, thetas, odd_indices)
  A = zeros(ComplexF64, length(rows), length(odd_indices))
  for (i, (cols, vals)) in enumerate(rows), (j, v) in zip(cols, vals)
    A[i, j] = RSR._c64(v)
  end
  for _ in 1:2
    for i in axes(A, 1)
      e = maximum(abs, view(A, i, :)); e > 0 && (A[i, :] ./= exp2(round(log2(e))))
    end
    for j in axes(A, 2)
      e = maximum(abs, view(A, :, j)); e > 0 && (A[:, j] ./= exp2(round(log2(e))))
    end
  end
  sv = RSR.LinearAlgebra.svdvals(A)
  return reverse(sv[end - min(count, length(sv)) + 1:end]) ./ sv[1]
end

# log2 of the singular values of M relative to the largest (decreasing), at
# the full precision: one-sided (Hestenes) Jacobi on the midpoints at
# precision + 64 bits, which also gets the small singular values with
# relative accuracy (no Gram matrix). -Inf for exactly zero ones. For the
# small matrices of the quadric step (columns: the monomials of degree 2).
function singular_values_log2(M::AcbMatrix; sweeps::Int = 40)
  CC = AcbField(precision(base_ring(M)) + 64)
  RR = ArbField(precision(CC))
  m, n = nrows(M), ncols(M)
  mid(z) = RSR._acb_mid(z)
  cols = [[mid(CC(M[i, j])) for i in 1:m] for j in 1:n]
  inner(x, y) = sum(conj(x[i]) * y[i] for i in 1:m)          # x^H y
  tol = RR(2)^(-precision(CC) + 16)
  for _ in 1:sweeps
    rotated = false
    for j in 1:n - 1, k in j + 1:n
      x, y = cols[j], cols[k]
      alpha, beta, gamma = real(inner(x, x)), real(inner(y, y)), inner(x, y)
      ag = abs(gamma)
      (iszero(ag) || ag <= tol * sqrt(alpha * beta)) && continue
      rotated = true
      e = gamma / ag                       # x^H (conj(e) y) = |gamma|
      ytil = [mid(conj(e) * v) for v in y]
      zeta = (beta - alpha) / (2 * ag)
      sz = RSR._arb_mid_f64(zeta) >= 0 ? 1 : -1
      t = sz / (abs(zeta) + sqrt(1 + zeta^2))
      c = 1 / sqrt(1 + t^2)
      s = c * t
      cols[j] = [mid(c * xi - s * yi) for (xi, yi) in zip(x, ytil)]
      cols[k] = [mid(s * xi + c * yi) for (xi, yi) in zip(x, ytil)]
    end
    rotated || break
  end
  log2_norm(x) = (nrm = sqrt(real(inner(x, x))); iszero(nrm) ? -Inf : Float64(log(nrm) / log(RR(2))))
  values = sort([log2_norm(x) for x in cols]; rev = true)
  return values .- values[1]
end


@doc raw"""
    find_fay_relations(g, thetas; extra = 2g, chunk = 256, tolerance = 1e-6, verbose = false)
        -> Vector{Tuple{Vector{Int}, Vector{Int}}}

A basis of Fay's relations between the gradients of the odd theta functions
(n_odd - g of them) and `extra` more as a check, as pairs (w, rho) of
doubled characteristics (the relation frobenius_fay_relation(g, thetas, w,
rho)). Chosen at machine precision from the theta constants `thetas` of a
generic curve: the relations of the rho of _fay_rho_candidates(g), in
chunks, are kept when they are independent of the ones kept before
(projection on the orthogonal complement and pivoted QR; relations
proportional to an earlier one are skipped). When the rank is reached, the
LU of the chosen relations checks that none depends on the earlier ones
(rounding in the orthogonal basis can make a dependent relation look new):
such relations are dropped and the search continues. Which relations
are independent does not depend on the (generic) curve: save the result
with save_fay_relations (by default to _default_fay_data_file(g)).
"""
function find_fay_relations(g::Int, thetas; extra::Int = 2*g, chunk::Int = 256,
                            tolerance::Float64 = 1e-6, verbose::Bool = false)
  LA = RSR.LinearAlgebra
  odd_indices = char_to_index.(collect.(odd_theta_characteristics(g)))
  n = length(odd_indices)
  target = n - g
  position = Dict(c => k for (k, c) in enumerate(odd_indices))
  values64 = Dict(k => RSR._c64(v) for (k, v) in thetas)
  setup = _fay_setup(g)
  # the relation as a normalized Float64 vector
  function relation_vector(w, sigma)
    v = zeros(ComplexF64, n)
    for (column, coefficient, c1, c2, c3) in _fay_relation_structure(g, w, sigma, setup)
      k = get(position, column, 0)
      k == 0 || (v[k] += coefficient * values64[c1] * values64[c2] * values64[c3])
    end
    len = LA.norm(v)
    len > 0 && (v ./= len)
    return v
  end
  # proportional relations (different (w, rho), the same row up to a factor)
  # are skipped
  seen = Set{Any}()
  function signature(v)
    j = findfirst(!iszero, v)
    j === nothing && return nothing
    u = v ./ v[j]
    return Tuple((k, round(real(u[k]), sigdigits = 10), round(imag(u[k]), sigdigits = 10)) for k in findall(!iszero, u))
  end
  U = zeros(ComplexF64, n, 0)                 # orthonormal basis of the span
  selected = Tuple{Vector{Int}, Vector{Int}}[]
  last_block = Tuple{Vector{Int}, Vector{Int}}[]
  last_rest = Int[]
  for rho in _fay_rho_candidates(g)
    sigma = Int.(lift.(Ref(ZZ), rho))
    ws = _non_orthogonal_doubled(sigma)
    for start in 1:chunk:length(ws)
      block = Tuple{Vector{Int}, Vector{Int}}[]
      columns = Vector{ComplexF64}[]
      for w in ws[start:min(start + chunk - 1, end)]
        v = relation_vector(w, sigma)
        key = signature(v)
        (key === nothing || key in seen) && continue
        push!(seen, key)
        push!(block, (w, sigma))
        push!(columns, v)
      end
      isempty(block) && continue
      V = reduce(hcat, columns)
      @req all(isfinite, V) "Non-finite Fay relations at machine precision (theta constants with infinite radius?)."
      W = V - U * (U' * V)
      W -= U * (U' * W)                       # twice (classical Gram-Schmidt)
      F = LA.qr(W, LA.ColumnNorm())
      new = min(count(>(tolerance), abs.(LA.diag(F.R))), target - size(U, 2))
      if new > 0
        Q = Matrix(F.Q)[:, 1:new]
        Q -= U * (U' * Q)
        Q -= U * (U' * Q)
        U = hcat(U, Matrix(LA.qr(Q).Q)[:, 1:new])
        append!(selected, [block[j] for j in F.p[1:new]])
      end
      last_block, last_rest = block, F.p[new + 1:end]
      size(U, 2) < target && continue
      # verification: the LU (partial pivoting) of the selected relations,
      # as columns; (nearly) zero pivots mark relations that depend on the
      # ones before them (lost orthogonality of U): drop them, rebuild U
      M = reduce(hcat, [relation_vector(w, s) for (w, s) in selected])
      Ft = LA.lu(M; check = false)
      du = abs.(LA.diag(Ft.U))
      bad = findall(<(1e-10 * maximum(du)), du)
      verbose && println("rank $(size(U, 2)) reached; verification: $(length(bad)) dependent relations")
      if isempty(bad)
        length(last_rest) >= extra || @info "Only $(length(last_rest)) check relations (asked for $extra)."
        append!(selected, [last_block[j] for j in last_rest[1:min(extra, end)]])
        return selected
      end
      deleteat!(selected, bad)
      U = Matrix(LA.qr(reduce(hcat, [relation_vector(w, s) for (w, s) in selected])).Q)[:, 1:length(selected)]
    end
    verbose && println("rho = $sigma: rank $(size(U, 2)) of $target")
  end
  error("The Fay relations of all rho have rank $(size(U, 2)), less than $target.")
end

function save_fay_relations(file::AbstractString, relations)
  save(file, ([w for (w, _) in relations], [rho for (_, rho) in relations]))
end

function load_fay_relations(file::AbstractString)
  ws, rhos = load(file)
  return [(Vector{Int}(w), Vector{Int}(rho)) for (w, rho) in zip(ws, rhos)]
end

# The quadrics from the gradients (rows of K) and those of the Prym (rows of
# K_prym): relations c between the squares m_d^2 give sum_d c_d l l'. The
# symmetric matrices are stored by their entries (i, j), i >= j.
function _quadrics_from_gradients(g::Int, K::AcbMatrix, K_prym::AcbMatrix; strict::Bool = true,
                                  diagnostics::Bool = false)
  CC = base_ring(K)
  prec = precision(CC)
  lower(n) = [(i, j) for j in 1:n for i in j:n]
  HiHis = lift_prym_indices(g)
  n = length(HiHis)
  HiHis = zero_matrix(CC, n, length(lower(g)))
  lis = zero_matrix(CC, n, length(lower(g - 1)))
  for (d, (i1, i2)) in enumerate(HiHis)
    for (k, (a, b)) in enumerate(lower(g))
      HiHis[d, k] = K[i1, a] * K[i2, b] + K[i2, a] * K[i1, b]
    end
    for (k, (a, b)) in enumerate(lower(g - 1))
      lis[d, k] = K_prym[d, a] * K_prym[d, b]
    end
  end
  # the squares span all quadrics in g - 1 variables
  O = RSR.numerical_kernel(transpose(lis); nullity = n - length(lower(g - 1)))[1]
  q_relations = RSR._log2_relative_residual(transpose(lis), O)
  @req !strict || q_relations < -prec / 4 "The squares of the Prym gradients do not span the quadrics (log2 residual $q_relations)."
  TT = transpose(O) * products
  e = div((g - 2) * (g - 3), 2)
  # the singular values of TT (rows: the relations, columns: the monomials;
  # the quadrics are its row space, of dimension e for a generic curve):
  # relative, decreasing, at Float64 (a diagnostic)
  rows_scaled = Matrix{ComplexF64}(undef, nrows(TT), ncols(TT))
  for i in 1:nrows(TT)
    row = [RSR._c64(TT[i, j]) for j in 1:ncols(TT)]
    m = maximum(abs, row)
    rows_scaled[i, :] = m > 0 ? row ./ m : row
  end
  singular_values = RSR.LinearAlgebra.svdvals(rows_scaled)
  singular_values ./= singular_values[1]
  data = RSR._numerical_kernel_data(TT; nullity = length(lower(g)) - e)
  q_quadrics = RSR._log2_relative_residual(TT, data.kernel)
  @req !strict || q_quadrics < -prec / 4 "The relations do not give $e quadrics (log2 residual $q_quadrics); wrong signs of the Prym theta constants, or a non-generic curve?"
  R, x = polynomial_ring(CC, ["x$i" for i in 1:g]; cached = false)
  half = inv(CC(2))
  quadrics = [sum((a == b ? half : one(CC)) * TT[r, k] * x[a] * x[b] for (k, (a, b)) in enumerate(lower(g)))
              for r in data.pivot_rows]
  info = (relations = q_relations, quadrics = q_quadrics, quadric_singular_values = singular_values)
  diagnostics || return quadrics, info
  # at full precision: the squares (columns: the quadratic monomials in g - 1
  # variables; all nonzero if they span the quadrics) and the relation matrix
  # (columns: the quadratic monomials in g; (g-2)(g-3)/2 nonzero for a
  # generic Jacobian)
  return quadrics, merge(info, (squares_singular_values_log2 = singular_values_log2(squares),
                                quadric_singular_values_log2 = singular_values_log2(TT)))
end

# B with K = G B, G the gradients of the odd theta functions at 0 (rows), and
# log2 of the relative residual of G B - K on the midpoints: the coordinates
# x of the quadrics are B^-1 z (z: the normalized differentials).
# Only the gradients of g well conditioned rows of K (and `checks` more, the
# largest rows of K, for the residual) are computed (theta_gradients_single).
function _gradient_coordinates(tau::AcbMatrix, K::AcbMatrix; checks::Int = 3)
  g = nrows(tau)
  CC = base_ring(tau)
  odds = odd_theta_characteristics(g)
  rows = RSR._numerical_kernel_data(K; nullity = 0).pivot_rows
  sizes = [maximum(RSR._abs64(K[r, i]) for i in 1:g) for r in 1:nrows(K)]
  extra = first(sort(setdiff(1:nrows(K), rows); by = r -> -sizes[r]), checks)
  used = [rows; extra]
  gradients = RSR.theta_gradients_single(tau, odds[used])
  G = matrix(CC, length(used), g, [gradients[Tuple(odds[r])][i] for r in used for i in 1:g])
  Ku = matrix(CC, length(used), g, [K[r, i] for r in used for i in 1:g])
  B = RSR._solve_precond(G[1:g, :], Ku[1:g, :])
  difference = map_entries(RSR._acb_mid, G) * map_entries(RSR._acb_mid, B) - map_entries(RSR._acb_mid, Ku)
  scale = maximum(RSR._abs64(Ku[i, j]) for i in 1:nrows(Ku), j in 1:g)
  residual = maximum(RSR._abs64(difference[i, j]) for i in 1:nrows(Ku), j in 1:g)
  return B, RSR._safe_log2(residual / scale)
end

################################################################################
#
#  The theta constants that are needed
#
################################################################################

# A dictionary of theta constants that records which characteristics are read
# (all values are the same nonzero number; the Fay relations only use the
# values arithmetically).
struct _RecordingThetas{N} <: AbstractDict{NTuple{N, Int}, AcbFieldElem}
  value::AcbFieldElem
  used::Set{NTuple{N, Int}}
end

Base.getindex(d::_RecordingThetas{N}, k::NTuple{N, Int}) where N = (push!(d.used, k); d.value)
Base.haskey(d::_RecordingThetas{N}, k::NTuple{N, Int}) where N = true
Base.length(d::_RecordingThetas) = length(d.used)
Base.iterate(d::_RecordingThetas, state...) = nothing

@doc raw"""
    used_theta_characteristics(g; rhos = nothing) -> Vector{NTuple{2g, Int}}

The characteristics whose theta constants reconstruct_quadrics_data reads for
genus g: those of the Fay relations of the curve (with the rho of
odd_theta_gradient_kernel; for the adaptive choice all rho candidates are
recorded up to the size needed, see below) and theta[0 d1; 0 d2],
theta[0 d1; 1 d2] for the Prym (d even of genus g - 1).
"""
function used_theta_characteristics(g::Int; rhos = nothing)
  rhos === nothing && (rhos = _fay_rhos(g))
  @req rhos !== nothing "No default rho for genus $g; pass rhos."
  recorder = _RecordingThetas{2*g}(AcbField(64)(1.3, 0.7), Set{NTuple{2*g, Int}}())
  for rho in rhos
    construct_fay_matrix(g, recorder, rho)
  end
  used = recorder.used
  for d in even_theta_characteristics(g - 1)
    push!(used, (0, d[1:g-1]..., 0, d[g:2*(g-1)]...))
    push!(used, (0, d[1:g-1]..., 1, d[g:2*(g-1)]...))
  end
  push!(used, ntuple(_ -> 0, 2*g))
  # odd constants vanish at z = 0: they are read, but cost nothing
  return sort!(filter(RSR._is_even, collect(used)))
end

@doc raw"""
    select_fay_relations(g, thetas; rhos = nothing, tolerance = 1e-8) -> NamedTuple

(rhos: by default those of _fay_rhos(g), otherwise the first max_rhos candidates.)

An experiment towards fewer theta constants: chooses Fay relations (rows
(w, rho) of construct_fay_matrix) greedily, each time the one that needs the
fewest even theta constants not used so far among those that raise the rank
of the selected rows (Float64 Gram-Schmidt on the rows restricted to the odd
characteristics, with the values of `thetas`), until the rank is
#odd - g. (Genus 5: 491 relations, 422 even constants for the curve, 478
with the Prym, of 528. Relations inside eta^perp do not exist: the
characteristics of a Fay relation differ by sums of pairs of an Aronhold
system, which span the whole space.) Returns the selected rows (w, rho), the even characteristics they
use (`curve`), together with those of the Prym (`total`), and the counts.
"""
function select_fay_relations(g::Int, thetas; rhos = nothing, tolerance::Float64 = 1e-8,
                              max_rhos::Int = 4*g)
  if rhos === nothing
    defaults = _fay_rhos(g)
    rhos = defaults === nothing ? _fay_rho_candidates(g)[1:max_rhos] : defaults
  end
  odd_indices = char_to_index.(collect.(odd_theta_characteristics(g)))
  target = length(odd_indices) - g
  CC = parent(first(values(thetas)))
  # all rows with their values and the even characteristics they read
  rows = Tuple{Any, Any, Vector{ComplexF64}, Set{NTuple{2*g, Int}}}[]
  for rho in rhos, w in naive_non_orth_complement(rho)
    recorder = _RecordingThetas{2*g}(CC(1.3, 0.7), Set{NTuple{2*g, Int}}())
    frobenius_fay_relation(g, recorder, w, rho)
    used = Set(filter(RSR._is_even, collect(recorder.used)))
    row_values = frobenius_fay_relation(g, thetas, w, rho)[odd_indices]
    v = [ComplexF64(RSR._c64(x)) for x in row_values]
    norm = sqrt(sum(abs2, v))
    norm > 0 && push!(rows, (w, rho, v / norm, used))
  end
  basis = Vector{ComplexF64}[]
  chosen = Tuple{Any, Any}[]
  covered = Set{NTuple{2*g, Int}}()
  remaining = collect(eachindex(rows))
  residual(v) = (r = copy(v); for q in basis; r .-= q * sum(conj.(q) .* r); end; r)
  while length(basis) < target && !isempty(remaining)
    # candidates by the number of new characteristics
    sort!(remaining; by = k -> length(setdiff(rows[k][4], covered)))
    added = false
    for (position, k) in enumerate(remaining)
      r = residual(rows[k][3])
      n = sqrt(sum(abs2, r))
      n > tolerance || continue
      push!(basis, r / n)
      push!(chosen, (rows[k][1], rows[k][2]))
      union!(covered, rows[k][4])
      deleteat!(remaining, position)
      added = true
      break
    end
    added || break
  end
  @req length(basis) >= target "The Fay relations of these rho reach rank $(length(basis)) of $target."
  total = copy(covered)
  for d in even_theta_characteristics(g - 1)
    push!(total, (0, d[1:g-1]..., 0, d[g:2*(g-1)]...))
    push!(total, (0, d[1:g-1]..., 1, d[g:2*(g-1)]...))
  end
  return (relations = chosen, curve = sort!(collect(covered)), total = sort!(collect(total)),
          counts = (relations = length(chosen), curve = length(covered), total = length(total),
                    even = length(even_theta_characteristics(g))))
end

@doc raw"""
    theta_constants_riemann_signs(tau; sign_data, fallback = true) -> Dict, NamedTuple

All theta constants of tau (genus g) with the signs from Riemann's relations
instead of Float64 sums: the squares by theta_constants_duplication (no final
sign choices), the signs of the g(2g + 1) fixed characteristics
(find_fixed_even_chars) from FLINT's certified acb_theta_one (in parallel;
they also fix the symplectic basis, so the constants belong to tau and not to
M tau for some M = 1 mod 2), the other signs by correct_signs_with_data! with
the sign data of genus g (a file for load_sign_data or the loaded pairs).
`info`: that of the duplication, the number of fixed signs that were flipped
and whether the stored sign data sufficed.
"""
function theta_constants_riemann_signs(tau::AcbMatrix; sign_data, fallback::Bool = true,
                                       certified_fixed_signs::Bool = true)
  g = nrows(tau)
  CC = base_ring(tau)
  squares, info = RSR.theta_constants_duplication(tau; squared = true)
  thetas = Dict{NTuple{2*g, Int}, AcbFieldElem}()
  for (key, square) in squares
    if !RSR._is_even(key)
      thetas[key] = zero(CC)
    elseif contains_zero(square)
      a = abs(square)
      root = zero(CC)
      RSR._add_error!(root, sqrt(Hecke.midpoint(a) + Hecke.radius(a)))
      thetas[key] = root
    else
      thetas[key] = sqrt(square)
    end
  end
  # (vanishing constants break the Riemann relations of the sign correction)
  vanishing = count(key -> RSR._is_even(key) && contains_zero(squares[key]), keys(squares))
  @req vanishing == 0 "$vanishing even theta constants vanish; only generic curves are supported."
  fixed, _ = find_fixed_even_chars(g)
  # only the signs are needed: acb_theta_one at low precision (raised for
  # the characteristics where it does not decide), certified
  flipped = 0
  undecided = [Tuple(c) for c in fixed]
  if !certified_fixed_signs
    # the heuristic signs of the duplication (Float64 approximations at the
    # top level; 78 acb_theta_one calls cost ~4.5 s in genus 6)
    signed, signed_info = RSR.theta_constants_duplication(tau)
    if signed_info.reliable
      remaining = eltype(undecided)[]
      for key in undecided
        x, reference = thetas[key], signed[key]
        plus, minus = overlaps(x, reference), overlaps(-x, reference)
        if plus != minus
          minus && (thetas[key] = -thetas[key]; flipped += 1)
        else
          push!(remaining, key)
        end
      end
      undecided = remaining
    end
  end
  low = 64
  while !isempty(undecided)
    @req low <= 2 * precision(CC) "acb_theta_one does not decide the signs of $(length(undecided)) fixed characteristics."
    CClow = AcbField(low)
    certified = RSR.theta_constants_single(change_base_ring(CClow, tau), undecided)
    remaining = eltype(undecided)[]
    for key in undecided
      x, reference = CClow(thetas[key]), certified[key]
      plus, minus = overlaps(x, reference), overlaps(-x, reference)
      if plus != minus
        if minus
          thetas[key] = -thetas[key]
          flipped += 1
        end
      else
        @req plus "The duplication and acb_theta_one disagree for the characteristic $key."
        push!(remaining, key)
      end
    end
    undecided = remaining
    low *= 2
  end
  pairs = sign_data isa AbstractString ? load_sign_data(sign_data) : sign_data
  _, sufficient = correct_signs_with_data!(thetas, g, pairs; fallback = fallback)
  return thetas, merge(info, (fixed_signs_flipped = flipped, sign_data_sufficed = sufficient))
end

# theta_constants_duplication, FLINT if its result is not reliable
function _theta_constants_duplication_or_flint(tau::AcbMatrix)
  thetas, info = RSR.theta_constants_duplication(tau)
  info.reliable && return thetas
  @info "theta_constants_duplication not reliable for this tau (final radius $(info.final_relative_radius), margin $(info.sign_margin_final)); using Hecke.thetas"
  CC = base_ring(tau)
  return Hecke.thetas([zero(CC) for _ in 1:nrows(tau)], tau)
end

# The theta constants of the (Siegel reduced) tau by the theta_method of
# reconstruct_quadrics_data
function _theta_constants_by_method(tau_red::AcbMatrix, theta_method::Symbol; rhos = nothing,
                                    curve_sign_data = nothing, certified_fixed_signs::Bool = true)
  g = nrows(tau_red)
  CC = base_ring(tau_red)
  @req theta_method in (:flint, :single, :duplication, :riemann) "theta_method must be :flint, :single, :duplication or :riemann."
  @req theta_method !== :riemann || curve_sign_data !== nothing "theta_method = :riemann needs curve_sign_data (the sign data of genus g)."
  theta_method === :flint && return Hecke.thetas([zero(CC) for _ in 1:g], tau_red)
  theta_method === :single && return RSR.theta_constants_single(tau_red, used_theta_characteristics(g; rhos = rhos))
  theta_method === :duplication && return _theta_constants_duplication_or_flint(tau_red)
  return theta_constants_riemann_signs(tau_red; sign_data = curve_sign_data,
                                       certified_fixed_signs = certified_fixed_signs)[1]
end

_loaded_sign_data(data, h::Int) = data === nothing ? load_sign_data(_default_sign_data_file(h)) :
                                  data isa AbstractString ? load_sign_data(data) : data

@doc raw"""
    reconstruct_quadrics_from_thetas(g, thetas; sign_data = nothing, rhos = nothing, prym_rhos = nothing,
                                     fay_relations = nothing, prym_fay_relations = nothing,
                                     sign_fallback = true, strict = true, verbose = false) -> NamedTuple

The quadrics of reconstruct_quadrics_data from the even theta constants
`thetas` of genus g alone (a dictionary characteristic => value, with
correct signs, e.g. from theta_constants_riemann_signs; the odd ones 0): the
theta constants of the Prym (Schottky-Jung, signs by Riemann's relations with
`sign_data` of genus g - 1), the kernels of Fay's relations for both (the
gradients up to a common linear map: `gradients`, rows the odd
characteristics) and the quadrics from the relations between the squares of
the Prym gradients, in the coordinates of `gradients`. With `gradients` (a
matrix as odd_theta_gradients gives it, from the period matrix) the Fay
kernel of the curve is skipped and these rows are used instead (the Prym
kernel is still computed from the Prym theta constants). The theta constants
need not come from a Jacobian: with `strict = false` nothing is required of
the residuals, and `info` reports them (about -precision if the relations
hold) and the singular values of the relation matrix (`quadric_singular_values`,
relative, decreasing: for a generic Jacobian (g-2)(g-3)/2 of them are not
small, the others at the level of the precision). With `diagnostics`
also: `squares_singular_values_log2` and `quadric_singular_values_log2` (log2
of all singular values of the squares of the Prym gradients and of the
relation matrix, relative, at the full precision: singular_values_log2) and
`fay_singular_values`, `prym_fay_singular_values` (the smallest singular
values of the Float64 Fay matrices: g resp. g - 1 at rounding level, then a
gap).
"""
function reconstruct_quadrics_from_thetas(g::Int, thetas; sign_data = nothing, rhos = nothing,
                                          prym_rhos = nothing, fay_relations = nothing,
                                          prym_fay_relations = nothing, sign_fallback::Bool = true,
                                          strict::Bool = true, diagnostics::Bool = false, verbose::Bool = false,
                                          gradients = nothing)
  @req g >= 5 "Only for g >= 5."
  times = Dict{Symbol, Float64}()
  stage(name) = (t = time(); () -> (times[name] = time() - t;
                                    verbose && println(rpad(string(name), 24), round(times[name], digits = 2), " s")))
  pairs_prym = _loaded_sign_data(sign_data, g - 1)
  done = stage(:prym_signs)
  thetas_prym = prym_thetas(g, thetas)
  _, sufficient = correct_signs_with_data!(thetas_prym, g - 1, pairs_prym; fallback = sign_fallback)
  done()
  done = stage(:gradient_kernel)
  if gradients === nothing
    K, q_K = odd_theta_gradient_kernel(g, thetas; rhos = rhos, relations = fay_relations, verbose = verbose)
  else
    # the gradients themselves (odd_theta_gradients, from the period matrix):
    # no Fay kernel for the curve; the Prym below still needs it
    K, q_K = gradients, -Inf
  end
  done()
  done = stage(:prym_gradient_kernel)
  K_prym, q_K_prym = odd_theta_gradient_kernel(g - 1, thetas_prym; rhos = prym_rhos, relations = prym_fay_relations)
  done()
  done = stage(:quadrics)
  quadrics, q = _quadrics_from_gradients(g, K, K_prym; strict = strict, diagnostics = diagnostics)
  done()
  info = merge((gradients = q_K, prym_gradients = q_K_prym, sign_data_sufficed = sufficient), q)
  if diagnostics
    info = merge(info, (fay_singular_values = fay_singular_values64(g, thetas; rhos = rhos, relations = fay_relations),
                        prym_fay_singular_values = fay_singular_values64(g - 1, thetas_prym; rhos = prym_rhos,
                                                                         relations = prym_fay_relations)))
  end
  info = merge(info, (times = (; times...),))
  return (quadrics = quadrics, gradients = K, prym_gradients = K_prym, prym_thetas = thetas_prym, info = info)
end

# The theta constants of the Prym (Schottky-Jung for eta = (0..0; 1 0..0)) of
# the theta constants of genus G, signs by Riemann's relations (sign data of
# genus G - 1); and whether the stored pairs sufficed
function _prym_theta_constants(G::Int, thetas, prym_sign_data; fallback::Bool = true)
  thetas_P = prym_thetas(G, thetas)
  _, sufficient = correct_signs_with_data!(thetas_P, G - 1, _loaded_sign_data(prym_sign_data, G - 1);
                                           fallback = fallback)
  return thetas_P, sufficient
end

@doc raw"""
    classify_jacobian(g, thetas; sign_data = nothing, fay_relations = nothing, prym_fay_relations = nothing,
                      rhos = nothing, prym_rhos = nothing, gap_bits = nothing,
                      allow_vanishing = false) -> NamedTuple

Whether the theta constants `thetas` (genus g >= 5, with signs) behave like
those of a generic Jacobian under the reconstruction, as a verdict that is
never an error: `verdict` is :jacobian, :not_jacobian or :inconclusive, `stage`
the step that decided, `evidence` the numbers. The steps:

 0. `:precision`: the even theta constants that do not vanish must be known
    to at least half the working precision (relative radii), else
    :inconclusive; vanishing ones (`:theta_constants`) are :inconclusive too,
    unless `allow_vanishing` (then the steps below are tried anyway, e.g.
    for hyperelliptic curves or products of Jacobians, also when the
    Schottky-Jung step fails: with vanishing theta constants Riemann's
    relations need not determine all the signs).
 1. `:schottky_jung`: the theta constants of the Prym for eta = (0..0; 1 0..0)
    by Schottky-Jung must satisfy Riemann's relations: the rank reached by
    the stored pairs (`sign_data`, genus g - 1) against the number of signs.
    For a Jacobian all of them (the random search completes a rank that is
    at least 90%); for a generic principally polarized abelian variety
    none. Below 90%: :not_jacobian.
 2. `:gradient_kernel`, `:prym_gradient_kernel`: the Fay kernels (dimension
    g resp. g - 1, residuals below -gap_bits). A failure for g itself (Fay's
    relations hold for every tau) is :inconclusive, for the Prym (its theta
    constants are not those of an abelian variety) :not_jacobian.
 3. `:squares`: the squares of the Prym gradients must span the quadrics in
    g - 1 variables (all singular values above -gap_bits): else :not_jacobian.
 4. `:quadrics`: the number k of quadrics, the singular values of the relation
    matrix above 2^(-precision/2), with a gap of at least gap_bits to the
    next one (else :inconclusive): k = (g-2)(g-3)/2 is :jacobian, any other k
    :not_jacobian (5 for the intermediate Jacobian of a cubic threefold in
    genus 5).

`gap_bits`: by default a quarter of the precision. Returns also the
quadrics (in the coordinates of the gradient kernel) when the verdict is
:jacobian.
"""
function classify_jacobian(g::Int, thetas; sign_data = nothing, fay_relations = nothing,
                           prym_fay_relations = nothing, rhos = nothing, prym_rhos = nothing,
                           gap_bits = nothing, allow_vanishing::Bool = false)
  CC = parent(first(values(thetas)))
  prec = precision(CC)
  gap_bits === nothing && (gap_bits = div(prec, 4))
  expected = div((g - 2) * (g - 3), 2)
  evidence = Dict{Symbol, Any}()
  result(v, stage; quadrics = nothing) = (verdict = v, stage = stage, expected = expected,
                                          quadrics = quadrics, evidence = (; evidence...))
  # 0. the theta constants themselves: enough precision (e.g. a period
  # matrix with large radii; of the ones that do not vanish), none vanishing
  # (unless allow_vanishing: e.g. hyperelliptic curves, products)
  vanishing = Set(RSR._vanishing_even_theta_constants(thetas, prec))
  evidence[:vanishing_theta_constants] = length(vanishing)
  worst = maximum(RSR._log2_relative_radius(v) for (k, v) in thetas if RSR._is_even(k) && !(k in vanishing))
  evidence[:theta_precision_bits] = -worst
  -worst < prec / 2 && return result(:inconclusive, :precision)
  isempty(vanishing) || allow_vanishing || return result(:inconclusive, :theta_constants)
  # 1. Schottky-Jung: Riemann's relations for the Prym theta constants
  pairs_prym = _loaded_sign_data(sign_data, g - 1)
  thetas_prym = prym_thetas(g, thetas)
  coverage = sign_data_coverage(g - 1, thetas_prym, pairs_prym)
  evidence[:schottky_jung_rank] = coverage.rank
  evidence[:schottky_jung_needed] = coverage.needed
  # (with allow_vanishing the steps below are tried anyway: with vanishing
  # theta constants the relations may not determine all signs)
  coverage.rank < 0.9 * coverage.needed && !allow_vanishing && return result(:not_jacobian, :schottky_jung)
  try
    correct_signs_with_data!(thetas_prym, g - 1, pairs_prym; fallback = coverage.rank < coverage.needed)
  catch err
    evidence[:schottky_jung_error] = sprint(showerror, err)
    allow_vanishing || return result(:not_jacobian, :schottky_jung)
  end
  # 2. the Fay kernels
  K, q_K = try
    odd_theta_gradient_kernel(g, thetas; rhos = rhos, relations = fay_relations)
  catch err
    evidence[:gradient_kernel_error] = sprint(showerror, err)
    return result(:inconclusive, :gradient_kernel)
  end
  evidence[:gradient_residual] = q_K
  q_K > -gap_bits && return result(:inconclusive, :gradient_kernel)
  K_prym, q_Kp = try
    odd_theta_gradient_kernel(g - 1, thetas_prym; rhos = prym_rhos, relations = prym_fay_relations)
  catch err
    message = sprint(showerror, err)
    evidence[:prym_gradient_kernel_error] = message
    # a rank that cannot be decided numerically (huge dynamic range, e.g. near
    # the boundary) is no evidence against a Jacobian
    return result(occursin("clear rank", message) ? :inconclusive : :not_jacobian, :prym_gradient_kernel)
  end
  evidence[:prym_gradient_residual] = q_Kp
  q_Kp > -gap_bits && return result(:not_jacobian, :prym_gradient_kernel)
  # 3., 4. the squares and the quadrics
  quadrics, q = _quadrics_from_gradients(g, K, K_prym; strict = false, diagnostics = true)
  squares_sv, quadric_sv = q.squares_singular_values_log2, q.quadric_singular_values_log2
  evidence[:squares_singular_values_log2] = squares_sv
  evidence[:quadric_singular_values_log2] = quadric_sv
  minimum(squares_sv) < -gap_bits && return result(:not_jacobian, :squares)
  k = count(>(-prec / 2), quadric_sv)
  gap = k == 0 ? Inf : k == length(quadric_sv) ? Inf : quadric_sv[k] - quadric_sv[k + 1]
  smallest_kept = k == 0 ? 0.0 : quadric_sv[k]
  evidence[:quadric_count] = k
  evidence[:quadric_gap_bits] = gap
  (gap < gap_bits || smallest_kept < -gap_bits) && return result(:inconclusive, :quadrics)
  k == expected && return result(:jacobian, :quadrics; quadrics = quadrics)
  return result(:not_jacobian, :quadrics)
end

@doc raw"""
    prym_quadrics(tau; prym_sign_data = nothing, sign_data = nothing, fay_relations = nothing,
                  prym_fay_relations = nothing, theta_method = :riemann, curve_sign_data = nothing,
                  certified_fixed_signs = true, strict = false, verbose = false) -> NamedTuple

The reconstruction of reconstruct_quadrics_from_thetas applied to the Prym
variety of the period matrix tau (genus G = nrows(tau) >= 6) for
eta = (0..0 1 0..0) instead of to tau itself: the theta constants of tau
(`theta_method`, `curve_sign_data` of genus G, as in reconstruct_quadrics_data),
those of the Prym P (genus g = G - 1, Schottky-Jung, signs by Riemann's
relations with `prym_sign_data` of genus g), then the quadrics as if P were
the Jacobian of a curve of genus g (`sign_data`: genus g - 1, for the Prym of
P; `fay_relations`: genus g, `prym_fay_relations`: genus g - 1). For G = 6
(P a generic principally polarized abelian variety of dimension 5, not a
Jacobian) this shows what the reconstruction does off the Jacobian locus:
see `info` (residuals and singular values; `diagnostics = true` by default,
see reconstruct_quadrics_from_thetas); `strict = false` by default, so that
it returns instead of stopping at the checks. Also returns the theta
constants of tau (`curve_thetas`, of the Siegel reduced `tau`) and of the
Prym (`thetas`).
"""
function prym_quadrics(tau::AcbMatrix; prym_sign_data = nothing, sign_data = nothing,
                       fay_relations = nothing, prym_fay_relations = nothing, rhos = nothing,
                       prym_rhos = nothing, theta_method::Symbol = :riemann, curve_sign_data = nothing,
                       certified_fixed_signs::Bool = true, strict::Bool = false,
                       diagnostics::Bool = true, curve_thetas = nothing, verbose::Bool = false)
  G = nrows(tau)
  @req ncols(tau) == G && G >= 6 "tau must be a G x G matrix with G >= 6 (a Prym of dimension >= 5)."
  g = G - 1
  _, tau_red = Hecke.siegel_reduction(tau)
  t = time()
  thetas = curve_thetas !== nothing ? curve_thetas :
           _theta_constants_by_method(tau_red, theta_method; curve_sign_data = curve_sign_data,
                                      certified_fixed_signs = certified_fixed_signs)
  verbose && println(rpad("theta_constants", 24), round(time() - t, digits = 2), " s")
  thetas_P = prym_thetas(G, thetas)
  _, sufficient = correct_signs_with_data!(thetas_P, g, _loaded_sign_data(prym_sign_data, g))
  vanishing = RSR._vanishing_even_theta_constants(thetas_P, precision(base_ring(tau)))
  @req isempty(vanishing) "$(length(vanishing)) even theta constants of the Prym vanish."
  result = reconstruct_quadrics_from_thetas(g, thetas_P; sign_data = sign_data, rhos = rhos,
                                            prym_rhos = prym_rhos, fay_relations = fay_relations,
                                            prym_fay_relations = prym_fay_relations, strict = strict,
                                            diagnostics = diagnostics, verbose = verbose)
  return merge(result, (thetas = thetas_P, curve_thetas = thetas, tau = tau_red,
                        info = merge(result.info, (prym_sign_data_sufficed = sufficient,))))
end

@doc raw"""
    reconstruct_quadrics_data(tau; sign_data = nothing, rhos = nothing, prym_rhos = nothing,
                              sign_fallback = true, coordinates = false, theta_method = :flint, curve_sign_data = nothing) -> NamedTuple

The quadrics through the canonical model of the generic curve of genus g >= 5
with small period matrix tau (see the header of ReconstructGenusG.jl):
`quadrics` (polynomials in g variables over the complex field), `tau` (the
Siegel reduced period matrix used), `transform` (tau = transform(input tau)),
`coordinates` (with `coordinates = true`: B with K = G B, G the theta
gradients, see reconstruct_rational_quadrics; otherwise nothing) and `info`
(log2 residuals on the midpoints, about -precision if all is well, and
whether the stored sign data sufficed). `theta_method`: `:flint` (all
constants with Hecke.thetas, certified), `:single` (only the constants of
used_theta_characteristics, each with acb_theta_one, in parallel; needs
`rhos` for genera without default rho) or `:duplication`
(theta_constants_duplication: all constants, heuristic signs, fast; falls
back to Hecke.thetas when its result is not reliable) or `:riemann`
(theta_constants_riemann_signs: squares by duplication, fixed signs
certified, the others from Riemann's relations with `curve_sign_data`, the
sign data of genus g). `sign_data`: the sign data of genus
g - 1 (a file for load_sign_data or the loaded pairs; default
_default_sign_data_file(g - 1)); `rhos`, `prym_rhos`: see
odd_theta_gradient_kernel; `sign_fallback`: see correct_signs_with_data!.
`gradient_method`: `:theta` (default; the gradients of the odd theta
functions evaluated from tau by odd_theta_gradients of ThetaGradients.jl,
which must be loaded: g theta evaluations instead of the Fay linear system
for the curve; the Prym kernel is still the Fay kernel of genus g - 1) or
`:fay` (the Fay kernel for the curve too, as before). With `:theta` and
`coordinates = true`, `coordinates` is the identity (the quadrics are in the
coordinates of z) unless `check_coordinates = true`, which recomputes a few
gradients with acb_theta_jet as an independent check (`info.coordinates`).
"""
function reconstruct_quadrics_data(tau::AcbMatrix; sign_data = nothing, rhos = nothing,
                                   prym_rhos = nothing, sign_fallback::Bool = true,
                                   coordinates::Bool = false, theta_method::Symbol = :flint,
                                   curve_sign_data = nothing, fay_relations = nothing,
                                   prym_fay_relations = nothing, certified_fixed_signs::Bool = true,
                                   gradient_method::Symbol = :theta, check_coordinates::Bool = false,
                                   verbose::Bool = false)
  g = nrows(tau)
  times = Dict{Symbol, Float64}()
  stage(name) = (t = time(); () -> (times[name] = time() - t;
                                    verbose && println(rpad(string(name), 24), round(times[name], digits = 2), " s")))
  @req ncols(tau) == g && g >= 5 "tau must be a g x g matrix with g >= 5."
  CC = base_ring(tau)
  prec = precision(CC)
  done = stage(:siegel_reduction)
  T, tau_red = Hecke.siegel_reduction(tau)
  done()
  @req theta_method in (:flint, :single, :duplication, :riemann) "theta_method must be :flint, :single, :duplication or :riemann."
  @req theta_method !== :riemann || curve_sign_data !== nothing "theta_method = :riemann needs curve_sign_data (the sign data of genus g)."
  done = stage(:theta_constants)
  thetas = _theta_constants_by_method(tau_red, theta_method; rhos = rhos, curve_sign_data = curve_sign_data,
                                      certified_fixed_signs = certified_fixed_signs)
  done()
  vanishing = RSR._vanishing_even_theta_constants(thetas, prec)
  @req isempty(vanishing) "$(length(vanishing)) even theta constants vanish; only generic curves are supported."
  @req gradient_method in (:theta, :fay) "gradient_method must be :theta or :fay."
  gradients = nothing
  if gradient_method === :theta
    # the gradients of the odd theta functions from tau (ThetaGradients.jl)
    # instead of the Fay kernel; the Prym kernel is unaffected
    done = stage(:theta_gradients)
    gradients, info_gradients = odd_theta_gradients(tau_red; verbose = verbose)
    done()
  end
  result = reconstruct_quadrics_from_thetas(g, thetas; sign_data = sign_data, rhos = rhos,
                                            prym_rhos = prym_rhos, fay_relations = fay_relations,
                                            prym_fay_relations = prym_fay_relations,
                                            sign_fallback = sign_fallback, verbose = verbose,
                                            gradients = gradients)
  # (not pairs(...): a global `pairs` in Main, e.g. sign data, would shadow it)
  for (k, v) in zip(keys(result.info.times), values(result.info.times))
    times[k] = v
  end
  quadrics, K = result.quadrics, result.gradients
  done = stage(:gradient_coordinates)
  if coordinates && gradient_method === :theta && !check_coordinates
    # K already is the matrix of gradients: the coordinates are those of z
    B, q_B = identity_matrix(CC, g), -Inf
  else
    B, q_B = coordinates ? _gradient_coordinates(tau_red, K) : (nothing, NaN)
  end
  done()
  info = merge(result.info, (coordinates = q_B, times = (; times...)))
  gradient_method === :theta && (info = merge(info, (theta_gradients = info_gradients,)))
  return (quadrics = quadrics, tau = tau_red, transform = T, coordinates = B, info = info)
end

reconstruct_quadrics(tau::AcbMatrix; kw...) = reconstruct_quadrics_data(tau; kw...).quadrics

################################################################################
#
#  A model over Q (or a number field) from the big period matrix
#
################################################################################

# x = L y: the coordinates x of the quadrics from the coordinates y of the
# differentials of the big period matrix Omega = [Omega_A Omega_B]: z =
# Omega_A'^-1 y (normalized differentials for tau' = T(tau), Omega' = Omega
# [D^T B^T; C^T A^T]) and x = B^-1 z (K = G B).
function _coordinates_from_differentials(Omega::AcbMatrix, data)
  g = nrows(data.tau)
  CC = base_ring(data.tau)
  T = change_base_ring(CC, data.transform)
  Cb, D = T[g+1:2*g, 1:g], T[g+1:2*g, g+1:2*g]
  OmegaA = Omega[:, 1:g] * transpose(D) + Omega[:, g+1:2*g] * transpose(Cb)
  return RSR._inv_precond(data.coordinates) * RSR._inv_precond(OmegaA)
end

# The reduced row echelon form of A (full row rank): the pivot columns from
# elimination on the midpoints (an entry below 2^(-prec/3) of the largest is
# zero), then A[:, pivots]^-1 A in ball arithmetic.
function _numerical_rref(A::AcbMatrix)
  CC = base_ring(A)
  m, n = nrows(A), ncols(A)
  B = map_entries(RSR._acb_mid, A)
  threshold = 2.0^(-precision(CC) / 3) * maximum(RSR._abs64(B[i, j]) for i in 1:m, j in 1:n)
  pivots = Int[]
  r = 0
  for c in 1:n
    r == m && break
    i = r + argmax([RSR._abs64(B[l, c]) for l in r+1:m])
    RSR._abs64(B[i, c]) < threshold && continue
    r += 1
    B = swap_rows(B, r, i)
    for l in r+1:m
      f = RSR._acb_mid(B[l, c] / B[r, c])
      for j in c:n
        B[l, j] = RSR._acb_mid(B[l, j] - f * B[r, j])
      end
    end
    push!(pivots, c)
  end
  @req r == m "The quadrics are not linearly independent."
  P = matrix(CC, m, m, [A[i, c] for i in 1:m for c in pivots])
  return RSR._solve_precond(P, A), pivots
end

@doc raw"""
    reconstruct_rational_quadrics(Pi::AcbMatrix, K = QQ; place = nothing, normalize = true,
                                  reduce_units = true, kw...) -> Vector{MPolyRingElem}

The quadrics through the canonical model over K (QQ or a number field,
embedded by `place`) of a generic curve of genus g >= 5 with big period
matrix Pi (g x 2g; the rows: a basis of the differentials defined over K, e.g.
big_period_matrix(RS) for a curve over K), in the coordinates of these
differentials: the reduced row echelon basis of the space of quadrics, its
coefficients recognized in K, and with `normalize` (and `reduce_units`) each
made integral and primitive as in reconstruct_rational_curve_g4. Other
keywords are passed to reconstruct_quadrics_data.
"""
function reconstruct_rational_quadrics(Pi::AcbMatrix, K = QQ; place = nothing, normalize::Bool = true,
                                       reduce_units::Bool = true, numerical::Bool = false, kw...)
  g = nrows(Pi)
  @req ncols(Pi) == 2*g "Pi must be a g x 2g matrix."
  CCin = base_ring(Pi)
  tau = RSR._solve_precond(Pi[:, 1:g], Pi[:, g+1:2*g])
  tau = (tau + transpose(tau)) * inv(CCin(2))
  data = reconstruct_quadrics_data(tau; coordinates = true, kw...)
  @req data.info.coordinates < -precision(CCin) / 4 "The gradient kernel is not a linear image of the theta gradients (log2 residual $(data.info.coordinates))."
  CC = base_ring(data.tau)
  L = _coordinates_from_differentials(change_base_ring(CC, Pi), data)
  ys = gens(parent(data.quadrics[1]))
  xy = [sum(L[i, j] * ys[j] for j in 1:g) for i in 1:g]
  e2 = RSR._exponent_vectors(g, 2)
  Qy = [evaluate(q, xy) for q in data.quadrics]
  A = matrix(CC, length(Qy), length(e2), [coeff(q, e) for q in Qy for e in e2])
  E, pivots = _numerical_rref(A)
  numerical && return (rref = E, pivots = pivots, monomials = e2, info = data.info)
  F, _, recognize, _ = RSR._recognition_field_data(K, place, CC)
  S, X = polynomial_ring(F, ["x$i" for i in 1:g]; cached = false)
  monomial_of(e) = prod(X[j]^e[j] for j in 1:g)
  result = elem_type(S)[]
  for r in 1:nrows(E)
    cs = [c in pivots ? (c == pivots[r] ? one(F) : zero(F)) : recognize(E[r, c]) for c in 1:ncols(E)]
    if any(isnothing, cs)
      failed = [c for c in 1:ncols(E) if cs[c] === nothing]
      bits = minimum(-RSR._log2_relative_radius(E[r, c]) for c in failed)
      error("Could not recognize $(length(failed)) of the coefficients of quadric $r in K: " *
            "they are known to only about $(round(bits, digits = 1)) bits (the period matrix has " *
            "$(precision(CCin)); the reconstruction loses the rest). More precision needed? " *
            "First unrecognized coefficient: $(E[r, failed[1]])")
    end
    q = sum(cs[k] * monomial_of(e2[k]) for k in eachindex(e2))
    push!(result, normalize ? RSR._normalize_model(q; reduce_units) : q)
  end
  return result
end

@doc raw"""
    check_rational_quadrics(f, prec = 500; place = nothing, kw...) -> NamedTuple

For a curve f(x, y) = 0 of genus g >= 5 over QQ or a number field whose
differentials have Baker's basis x^(i-1) y^(j-1) dx / f_y ((i, j) the interior
points of the Newton polygon): reconstruct_rational_quadrics of its big period
matrix (original model) and the exact check that the quadrics vanish on the
monomials m. Keywords are passed to reconstruct_rational_quadrics.
"""
function check_rational_quadrics(f::MPolyRingElem, prec::Int = 500; place = nothing, kw...)
  K = base_ring(f)
  if K isa QQField
    RS = RSR.riemann_surface(f, prec; superelliptic = false, model = :original)
  else
    place === nothing && (place = infinite_places(K)[1])
    RS = RSR.riemann_surface(f, place, prec; superelliptic = false, model = :original)
  end
  g = RSR.genus(RS)
  points = RSR._newton_polygon_interior_points(f)
  @req length(points) == g "The Newton polygon of f does not have g = $g interior points."
  Q = reconstruct_rational_quadrics(RSR.big_period_matrix(RS), K; place = place, kw...)
  x, y = gens(parent(f))
  m = [x^(p[1] - 1) * y^(p[2] - 1) for p in points]
  return (quadrics = Q, vanish = all(q -> iszero(evaluate(q, m)), Q))
end

@doc raw"""
    quadrics_residual_on_points(RS; npoints = 6, kw...) -> NamedTuple

A check of reconstruct_quadrics_data that does not need recognition: points
P of the plane model of RS (original model, `model = :original`) are mapped
to the coordinates of the quadrics, x = L y with y_k = omega_k(P) (the
differentials of the big period matrix, without dx) and L as in
reconstruct_rational_quadrics, and the quadrics are evaluated there.
Returns log2 of the largest relative value (`worst`, about -precision if the
quadrics contain the canonical curve and the coordinates are right) and the
residual of the coordinates. Keywords are passed to
reconstruct_quadrics_data.
"""
function quadrics_residual_on_points(RS; npoints::Int = 6, kw...)
  C = RSR.computational_model(RS)
  @req isone(C.transform) "The check needs the original model (model = :original)."
  Pi = RSR.big_period_matrix(RS)
  g = nrows(Pi)
  CCin = base_ring(Pi)
  tau = RSR._solve_precond(Pi[:, 1:g], Pi[:, g+1:2*g])
  tau = (tau + transpose(tau)) * inv(CCin(2))
  data = reconstruct_quadrics_data(tau; coordinates = true, kw...)
  CC = base_ring(data.tau)
  prec = precision(CC)
  L = _coordinates_from_differentials(change_base_ring(CC, Pi), data)
  scales = [maximum(RSR._abs64, coefficients(q)) for q in data.quadrics]
  factor_set, factor_matrix, _, _ = RSR.differential_form_data(C)
  v = RSR.embedding(C)
  factors = [RSR._embed_mpoly(p, v, prec) for p in factor_set]
  fC = RSR.complex_defining_polynomial(C, prec)
  worst = -Inf
  for k in 1:npoints
    x0 = CC(0.37 + 0.61*k, -0.29 + 0.43*k) / 3
    for y0 in RSR.fiber(fC, x0)[1:min(2, end)]
      values = [evaluate(p, [x0, y0]) for p in factors]
      y = [prod(values[l]^factor_matrix[l, j] for l in eachindex(values)) for j in 1:g]
      xpt = L * matrix(CC, g, 1, y)
      xs = [xpt[i, 1] for i in 1:g]
      norm = maximum(RSR._abs64, xs)
      for (q, s) in zip(data.quadrics, scales)
        worst = max(worst, RSR._safe_log2(RSR._abs64(evaluate(q, xs)) / (s * norm^2)))
      end
    end
  end
  return (worst = worst, coordinates = data.info.coordinates)
end

@doc raw"""
    exact_quadrics_from_differentials(RS) -> Vector{MPolyRingElem}

The quadrics through the canonical image of the curve of RS (over QQ),
computed exactly from the differentials that the rows of big_period_matrix(RS)
integrate: the products of powers of the factors of differential_form_data
(for the function field basis these are the basis elements without the
constant of the factorization of their numerators), as elements of the
function field. The linear relations over QQ between the products
y_i y_j (multiplied by a common denominator in x, written in the basis
1, y, .., y^(n-1)). In the variables x1, .., xg; the reference for
reconstruct_rational_quadrics.
"""
function exact_quadrics_from_differentials(RS)
  _, ys = _differentials_in_function_field(RS)
  g = length(ys)
  e2 = RSR._exponent_vectors(g, 2)
  products = [prod(ys[i]^e[i] for i in 1:g) for e in e2]
  N = kernel(_rational_coefficient_matrix(products); side = :left)
  S, Xs = polynomial_ring(QQ, ["x$i" for i in 1:g]; cached = false)
  return [sum(N[r, j] * prod(Xs[i]^e2[j][i] for i in 1:g) for j in eachindex(e2)) for r in 1:nrows(N)]
end

# The function field QQ(x)[a]/f of the computational model of RS (f over QQ,
# possibly as a number field of degree 1), the map to_F of polynomials over
# QQ in x, y to it, and the differentials of big_period_matrix(RS) divided by
# dx: the products of powers of the factors of differential_form_data.
function _differentials_in_function_field(RS)
  C = RSR.computational_model(RS)
  RSR._ensure_differentials!(C)
  rational(c) = c isa QQFieldElem ? c : coeff(c, 0)
  f = C.defining_polynomial
  kx, t = rational_function_field(QQ, "x"; cached = false)
  kxy, u = polynomial_ring(kx, "y"; cached = false)
  F, a = function_field(sum(rational(c) * t^e[1] * u^e[2] for (c, e) in zip(Hecke.coefficients(f), Hecke.exponent_vectors(f))), "a")
  to_F(p) = sum((rational(c) * F(t)^e[1] * a^e[2] for (c, e) in zip(Hecke.coefficients(p), Hecke.exponent_vectors(p))); init = zero(F))
  factor_set, factor_matrix, _, _ = RSR.differential_form_data(C)
  factors = [to_F(p) for p in factor_set]
  # (function field elements have no negative powers)
  power(z, e) = e >= 0 ? z^e : inv(z)^(-e)
  ys = [prod(power(factors[l], factor_matrix[l, k]) for l in eachindex(factors)) for k in 1:size(factor_matrix, 2)]
  return to_F, ys
end

# Function field elements as the rows of a matrix over QQ: multiplied by a
# common denominator in x, the coefficients of x^i a^j.
function _rational_coefficient_matrix(elements)
  R, (X, _) = polynomial_ring(QQ, [:x, :y]; cached = false)
  numerators = [RSR._to_mpoly(R, numerator(p)) for p in elements]
  denominators = [denominator(p) for p in elements]
  common = reduce(lcm, denominators)
  scaled = [n * divexact(common, d)(X) for (n, d) in zip(numerators, denominators)]
  monos = unique(vcat([collect(Hecke.monomials(p)) for p in scaled]...))
  return matrix(QQ, length(scaled), length(monos), [coeff(p, m) for p in scaled for m in monos])
end

@doc raw"""
    compare_with_exact_quadrics(RS; kw...) -> NamedTuple

The numerical reduced row echelon form of the reconstructed quadrics
(reconstruct_rational_quadrics with numerical = true) against the exact one
of exact_quadrics_from_differentials with the same pivots: the largest
difference, and the largest numerator and denominator (in bits) of the exact
coefficients.
"""
function compare_with_exact_quadrics(RS; kw...)
  numerical = reconstruct_rational_quadrics(RSR.big_period_matrix(RS); numerical = true, kw...)
  exact = exact_quadrics_from_differentials(RS)
  e2 = numerical.monomials
  k = base_ring(parent(exact[1]))
  A = matrix(k, length(exact), length(e2), [coeff(q, e) for q in exact for e in e2])
  P = A[:, numerical.pivots]
  Ex = inv(P) * A
  CC = base_ring(numerical.rref)
  # the field of the equation may be QQ as a number field of degree 1
  rational(c) = c isa QQFieldElem ? c : coeff(c, 0)
  difference = maximum(RSR._abs64(numerical.rref[r, c] - CC(rational(Ex[r, c])))
                       for r in 1:nrows(Ex), c in 1:ncols(Ex))
  heights = [(nbits(numerator(rational(c))), nbits(denominator(rational(c)))) for c in Ex]
  return (difference = difference, max_numerator_bits = maximum(first, heights),
          max_denominator_bits = maximum(last, heights), exact_rref = Ex)
end

################################################################################
#
#  A generic genus 5 test curve with known quadrics
#
################################################################################

@doc raw"""
    canonical_test_curve_g5(; coefficients = -3:3) -> NamedTuple

A random canonical genus 5 curve over QQ, the complete intersection of
Q_k = a_k X4 + b_k X5 + c_k (k = 1, 2; a_k, b_k linear and c_k quadratic in
X1, X2, X3) and a random quadric Q3, with its plane model F(x, y) = 0, the
projection to (X1 : X2 : X3) = (x : y : 1) (a sextic with 5 nodes). On the
curve X4 = (b1 c2 - b2 c1)/D and X5 = (a2 c1 - a1 c2)/D with D = a1 b2 - a2 b1,
so the canonical coordinates are the adjoint cubics
A = (x D, y D, D, b1 c2 - b2 c1, a2 c1 - a1 c2): the differentials
A_i dx / F_y satisfy Q1, Q2, Q3 exactly. Returns `quadrics`, `plane`
(F) and `numerators` (A). Retries until F is irreducible of degree 6.
"""
function canonical_test_curve_g5(; coefficients = -3:3)
  S, X = polynomial_ring(QQ, ["X$i" for i in 1:5]; cached = false)
  R, (x, y) = polynomial_ring(QQ, [:x, :y]; cached = false)
  random() = QQ(rand(coefficients))
  linear() = sum(random() * X[i] for i in 1:3)
  quadratic(n) = sum(random() * X[i] * X[j] for i in 1:n for j in i:n)
  plane(p) = evaluate(p, [x, y, one(R), zero(R), zero(R)])
  while true
    a, b, c = [linear() for _ in 1:2], [linear() for _ in 1:2], [quadratic(3) for _ in 1:2]
    quadrics = [[a[k] * X[4] + b[k] * X[5] + c[k] for k in 1:2]; quadratic(5)]
    A1, B1, C1, A2, B2, C2 = plane.((a[1], b[1], c[1], a[2], b[2], c[2]))
    D = A1 * B2 - A2 * B1
    numerators = [x * D, y * D, D, B1 * C2 - B2 * C1, A2 * C1 - A1 * C2]
    @assert all(q -> iszero(evaluate(q, numerators)), quadrics[1:2])
    F = evaluate(quadrics[3], numerators)     # D^2 Q3(x, y, 1, X4, X5)
    fac = factor(F)
    total_degree(F) == 6 && length(fac) == 1 && all(e == 1 for (_, e) in fac) || continue
    return (quadrics = quadrics, plane = F, numerators = numerators)
  end
end

@doc raw"""
    period_matrix_of_differentials(RS, F, numerators) -> AcbMatrix

The big period matrix of the differentials A dx / F_y (A in `numerators`,
polynomials in the variables of F, the plane model of RS, see
canonical_test_curve_g5): M big_period_matrix(RS) with M over QQ computed
exactly in the function field (A / F_y = sum_k M_k y_k, y_k the differentials
of RS divided by dx).
"""
function period_matrix_of_differentials(RS, F::MPolyRingElem, numerators)
  to_F, ys = _differentials_in_function_field(RS)
  @req iszero(to_F(F)) "F is not the plane model of RS (model = :original)."
  Fy = inv(to_F(derivative(F, 2)))
  Y = _rational_coefficient_matrix([ys; [to_F(A) * Fy for A in numerators]])
  g = length(ys)
  M = zero_matrix(QQ, length(numerators), g)
  for i in eachindex(numerators)
    N = kernel(Y[[1:g; g + i], :]; side = :left)
    @req nrows(N) == 1 && !iszero(N[1, g + 1]) "The differential $i is not holomorphic (or the basis of RS is not one)."
    for k in 1:g
      M[i, k] = -N[1, k] // N[1, g + 1]
    end
  end
  Pi = RSR.big_period_matrix(RS)
  CC = base_ring(Pi)
  return matrix(CC, nrows(M), g, [CC(M[i, k]) for i in 1:nrows(M) for k in 1:g]) * Pi
end

@doc raw"""
    canonical_test_curve_g6(; coefficients = -3:3) -> NamedTuple

A random genus 6 curve over QQ with known canonical model: a plane sextic F
with nodes at (0, 0), (1, 0), (0, 1), (1, 1) (the generic genus 6 curve):
F = sum c q m over q in {q1^2, q1 q2, q2^2} and m in {1, x, y, x^2, x y, y^2}
(q1 = x^2 - x, q2 = y^2 - y, the conics through the nodes; random c). The
adjoint cubics A = (q1, x q1, y q1, q2, x q2, y q2) give the canonical
differentials A_i dx / F_y; the quadrics are the relations
sum c_ij A_i A_j = lambda F (5 of them hold on the plane: the quintic del
Pezzo surface). Returns `quadrics` (6, in X1..X6), `plane` (F) and
`numerators` (A). Retries until F is irreducible.
"""
function canonical_test_curve_g6(; coefficients = -3:3)
  R, (x, y) = polynomial_ring(QQ, [:x, :y]; cached = false)
  q1, q2 = x^2 - x, y^2 - y
  monomials = [one(R), x, y, x^2, x*y, y^2]
  numerators = [q1, x*q1, y*q1, q2, x*q2, y*q2]
  while true
    F = sum(QQ(rand(coefficients)) * q * m for q in (q1^2, q1*q2, q2^2) for m in monomials)
    fac = factor(F)
    total_degree(F) == 6 && length(fac) == 1 && all(e == 1 for (_, e) in fac) || continue
    return (quadrics = _quadrics_through_numerators(numerators, F), plane = F, numerators = numerators)
  end
end

# The quadrics sum c_ij X_i X_j with sum c_ij A_i A_j = h F (A_i the adjoints
# of degree d - 3 of the plane curve F of degree d, h of degree d - 6)
function _quadrics_through_numerators(numerators, F)
  g = length(numerators)
  e2 = RSR._exponent_vectors(g, 2)
  x, y = gens(parent(F))
  k = 2 * maximum(total_degree, numerators) - total_degree(F)
  multiples = [F * x^i * y^j for i in 0:k for j in 0:k - i]
  polynomials = [[prod(numerators[i]^e[i] for i in 1:g) for e in e2]; multiples]
  monos = unique(vcat([collect(Hecke.monomials(p)) for p in polynomials]...))
  M = matrix(QQ, length(polynomials), length(monos), [coeff(p, m) for p in polynomials for m in monos])
  N = kernel(M; side = :left)
  S, X = polynomial_ring(QQ, ["X$i" for i in 1:g]; cached = false)
  return [sum(N[r, j] * prod(X[i]^e2[j][i] for i in 1:g) for j in eachindex(e2)) for r in 1:nrows(N)]
end

@doc raw"""
    check_rational_quadrics_canonical(curve, prec::Int = 500; kw...) -> NamedTuple

reconstruct_rational_quadrics for a curve of canonical_test_curve_g5 or
canonical_test_curve_g6 from the period matrix of its canonical
differentials (period_matrix_of_differentials of the plane model at `prec`
bits). `same_span`: whether the result spans the same space of quadrics as
the original ones. Keywords are passed to reconstruct_rational_quadrics.
"""
function check_rational_quadrics_canonical(curve, prec::Int = 500; kw...)
  g = length(curve.numerators)
  RS = RSR.riemann_surface(curve.plane, prec; superelliptic = false, model = :original)
  @req RSR.genus(RS) == g "The plane model has genus $(RSR.genus(RS)), not $g."
  Pi = period_matrix_of_differentials(RS, curve.plane, curve.numerators)
  Q = reconstruct_rational_quadrics(Pi; kw...)
  e2 = RSR._exponent_vectors(g, 2)
  coefficient_matrix(qs) = matrix(QQ, length(qs), length(e2), [QQ(coeff(q, e)) for q in qs for e in e2])
  A, B = coefficient_matrix(curve.quadrics), coefficient_matrix(Q)
  n = length(curve.quadrics)
  same_span = rank(A) == n && rank(B) == n && rank(vcat(A, B)) == n
  return (quadrics = Q, original = curve.quadrics, same_span = same_span, curve = curve)
end

@doc raw"""
    check_rational_quadrics_g5(prec::Int = 500; coefficients = -3:3, curve = nothing, kw...) -> NamedTuple

reconstruct_rational_quadrics for a random curve of canonical_test_curve_g5
(or `curve`, a result of it) from the period matrix of its canonical
differentials (period_matrix_of_differentials of the plane model at `prec`
bits). `same_span`: whether the result spans the same space of quadrics as
the original ones. Keywords are passed to reconstruct_rational_quadrics.
"""
function check_rational_quadrics_g5(prec::Int = 500; coefficients = -3:3, curve = nothing, kw...)
  curve === nothing && (curve = canonical_test_curve_g5(; coefficients = coefficients))
  return check_rational_quadrics_canonical(curve, prec; kw...)
end
