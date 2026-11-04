#=
using Hecke.RiemannSurfaces
using GenericLinearAlgebra
R, (x,y) = polynomial_ring(QQ, [:x,:y])
f = 2*x^5*y^3 + 5*x^4*y^3 - 9*x^4*y^2 + 3*x^3*y - 4*x^2*y^2 - 3*x*y^3 + 1
@time RS = riemann_surface(f, 200, integration_method = "heuristic")
g = genus(RS)
tau = small_period_matrix(RS)
CC = complex_field(RS)
z = zeros(CC, g)
@time thetas = Hecke.thetas(z, tau)

thetas_prym = prym_thetas(g, thetas)
fixed_thetas, non_fixed_thetas = find_fixed_even_chars(g-1)
terms = find_zero_sum_tetrads(fixed_thetas)

N = length(terms)
M = zero_matrix(GF(2), 4*N,  2^(2*(g-1)));
b = zero_matrix(GF(2), 4*N, 1);
for i in (1:4:4*N)
  test, rel = relation_with_term(g-1, thetas_prym, terms[div(i+3, 4)])
  if !test
    continue
  end
  c = rel[1][2]
  for k in (0:3)
    mon, coeff = rel[k + 1]
    b[i + k,1] = GF(2)(div(c*coeff-1,2))
    for j in mon
      M[i + k, j] = GF(2)(1)
    end
  end
end

Mbr = M[:,char_to_index.(non_fixed_thetas)];

v =  solve(Mbr, b, side = :right)

for j in (1:length(v))
  thetas_prym[non_fixed_thetas[j]...] *= ZZ(-1)^(lift(ZZ,v[j]))
end

for i in (1:N)
  println(relation_with_term(g-1, thetas_prym, terms[i], identity_matrix(GF(2), 2*(g-1))))
end


#z0 = CC.([1.4,0,3.3,1//2,1])
#@time thetas5_z0 = Hecke.thetas(z0, tau)
#Th_sq = map(x->x^2, theta_list(thetas5_z0, 5))


A = aronhold_system(g, azygetic_system(g));
A2 = aronhold_system(g, azygetic_system_flip(g));

even_indices = char_to_index.(even_theta_characteristics(g));
odd_indices = char_to_index.(odd_theta_characteristics(g));

I = collect(1:length(odd_indices));
I = collect(1:2^(2*g-1));

@time M1 = theta_square_relations(g, thetas, A, I);
@time M2 = theta_square_relations(g, thetas, A2, I);
M = [M1 ; M2];

M_even = M[:, even_indices];

#CC = parent(thetas[zeros(Int, 2*g)...])
#prec = precision(CC)
#setprecision(BigFloat, prec)
M_even_float = Complex{Float64}.(collect(M_even));
M_even_float = (x-> abs(x) < 10^(-50) ? zero(ComplexF64) : x).(M_even_float);
@time K = permutedims(nullspace(transpose(M_even_float)));
M_float = Complex{Float64}.(collect(M));
@time O = K*(M_float[:,odd_indices]);
@time rank(O)

#Sanity check:
#P = M * Th_sq;
#length(filter( x -> abs(x) > 10^(-30), P)) == 0


z0 = CC.([1.4,3.3,1//2,1])
@time thetas4_z0 = Hecke.thetas(z0, tau)
Th_sq = map(x->x^2, theta_list(thetas4_z0, 4))
P = M * Th_sq;
length(filter( x -> abs(x) > 10^(-30), P)) == 0


b = A[1:4]
B = sum(b)
A2 = [b;[B + v for v in A[5:end] ]]

g4_theta_sq = prym_thetas_sq(5, thetas)

rho = QQ.([1,1, 0,0,0,0,1,0,0,1])//2
sigma = [0,QQ(1//2),0,0,0,0,0,0,QQ(1//2),0]
SS = max_noether_relations(5, thetas5, rho, sigma)

v = [derivative(SS, i) for i in odd_indices]
Th = theta_list(thetas5, 5)
w = (x -> x(Th...)).(v)
filter( x -> x != zero(parent(x)), w)


=#

# The theta characteristics from the module (ThetaCharacteristics.jl is part
# of Hecke.RiemannSurfaces; this development file is loaded in Main).
using Hecke.RiemannSurfaces: integral_left_kernel   # (not exported from the module)
using Hecke.RiemannSurfaces: odd_theta_characteristics, even_theta_characteristics,
                             char_to_index, lift_prym_indices

#Use bits for speed.
function bitmask(v::Vector{Int})
  m = zero(UInt128)
  for (i, x) in enumerate(v)
      iszero(x) || (m |= (UInt128(1) << (i - 1)))
  end
  return m
end

#Computes squares of the thetas of the Prym. 
function prym_thetas_sq(g::Int, thetas)
  g_min_thetas = Dict{NTuple{2*(g-1), Int64}, AcbFieldElem}()
  chars_even = even_theta_characteristics(g-1)
  for char in chars_even
    delta_new1 = (0, char[1:g-1]..., 0, char[g:2*(g-1)]...)
    delta_new2 = (0, char[1:g-1]..., 1, char[g:2*(g-1)]...)
    #Formula from Lemma 1 (p. 148) of Farkas
    g_min_thetas[char] = thetas[delta_new1]*thetas[delta_new2]
  end

  return g_min_thetas
end

function prym_thetas(g::Int, thetas)
  g_min_thetas = Dict{NTuple{2*(g-1), Int64}, AcbFieldElem}()
  chars_even = even_theta_characteristics(g-1)
  for char in chars_even
    delta_new1 = (0, char[1:g-1]..., 0, char[g:2*(g-1)]...)
    delta_new2 = (0, char[1:g-1]..., 1, char[g:2*(g-1)]...)
    #Formula from Lemma 1 (p. 148) of Farkas
    g_min_thetas[char] = sqrt(thetas[delta_new1]*thetas[delta_new2])
  end
  chars_odd = odd_theta_characteristics(g-1)
  CC = parent(thetas[zeros(Int, 2*g)...])
  for char in chars_odd 
    g_min_thetas[char] = CC(0)
  end
  return g_min_thetas
end

function azygetic_system(g::Int)
  zer = zero_matrix(GF(2), g, g)
  zer1 = zero_matrix(GF(2), g, 1)
	zer2 = zero_matrix(GF(2), g, 2)
	id = identity_matrix(GF(2), g)
	triang = zer
	for i in (1:g)
		for j in (i:g)
			triang[i,j] = 1
		end
	end

	odd_chars = [id; triang]
	even_chars = [[id zer2] ; [zer1 triang zer1]]

  return [odd_chars even_chars]
end

function aronhold_system(g::Int, M)
  a = aronhold_system(g)
  apply_transformation_to_char.(Ref(M), a)
end

function aronhold_system(g::Int)
  azy_set = azygetic_system(g)
  zer = zero_matrix(GF(2), g, g)
	id = identity_matrix(GF(2), g)
	J = [zer id; id zer]
  J1 = [zer id; zer zer]

  w = diagonal(transpose(azy_set) * J1 * azy_set)

  for i in (1:2*g+2)
    e = zeros(GF(2),2*g+2)
    e[i] += g
    test, sol = can_solve_with_solution(J*azy_set, w+e)
    if test
      return [t + sol for t in [transpose(azy_set)[j,:] for j in (1:ncols(azy_set))][(1:end) .!= i]]
    end
  end
  error("Error in computation of aronhold system.")

end

function max_noether_data(g::Int, M)
  a = aronhold_system(g, M)
  a = [QQ.(lift.(Ref(ZZ), av))/2 for av in a]
  a0 = sum(a)


  l = Vector{QQFieldElem}[]
  for p in (1:g-3)
    push!(l, a[2*p+6] + a[2*p+7])
  end

  lambda = Vector{QQFieldElem}[]
  l_M = matrix(l)
  V = Iterators.product(repeat([[QQ(0),QQ(1)]], g-3)...)
  for v in V 
    v = [v...]
    push!(lambda, v * l_M)
  end

  A = Vector{QQFieldElem}[]

  for lam in lambda
    for ai in [[a0] ; a[1:7]]
      push!(A, lam+ai)
    end
  end

  L = sum(a[i] for i in (8:2:2*g))
  if mod(g, 2) == 1
    L += a0
  end
  return L, lambda, [[a0] ; a[1:7]], a[2*g]
end

function max_noether_characteristics(g::Int)
  L, lambda, A_rho, a2g = max_noether_data(g, identity_matrix(GF(2), 2*g))

  chars = []

  for h in (1:2^(g-3)) 
    v = L + lambda[h]
    push!(chars, v)
  end

  for a_rho in A_rho
    for j in (1:2^(g-4))
      v = L + lambda[j] + a_rho + a2g
      push!(chars, v)
    end
  end

  return chars
end


function max_noether_relations(g::Int, thetas::Dict{NTuple{N, Int64}, AcbFieldElem}, rho::Vector{QQFieldElem}, sigma::Vector{QQFieldElem}) where N
  @req g >= 4 "g needs to be bigger than 3."
  L, lambda, A_rho, a2g = compute_max_noether_characteristics(g::Int)

  CC = parent(thetas[zeros(Int, 2*g)...])
  R, X = polynomial_ring(CC, 2^(2*g))


  D = thetas

  theta_indices = Hecke.theta_characteristics_indices(g)
  Dv = Dict([(theta_indices[i+1], X[i]) for i in (1:2^(2*g)-1)])
  Dv[zeros(Int, 2*g)...] = X[2^(2*g)]

  first_term = R(0)

  factor = (-1)^(half_char_inner_prod(L, a2g) + half_char_inner_prod(a2g, L) )
 
  for h in (1:2^(g-3)) 
    first_term += (-1)^(half_char_inner_prod(lambda[h], lambda[h]) + 
    half_char_inner_prod(L+lambda[h] +a[1] + a2g, L + lambda[h] + a[1] + a2g)+half_char_inner_prod(rho+sigma, L + lambda[h])) * 
    mod(half_char_inner_prod(L + lambda[h] + rho) - 1, 2) *
    theta_HI(D, L + lambda[h]) * theta_HI(D, L + lambda[h] + rho) *
    theta_HI(Dv, L + lambda[h] + sigma) * theta_HI(Dv, L + lambda[h] + rho + sigma)

  end

  first_term *= factor

  second_term = 0
  for a_rho in A_rho
    for j in (1:2^(g-4))
      second_term += (-1)^((half_char_inner_prod(lambda[j] + a_rho, L)  + half_char_inner_prod(L, lambda[j] +a_rho) + half_char_inner_prod(rho + sigma, L + lambda[j] + a_rho + a2g))) *
      mod(half_char_inner_prod(L + lambda[j] + a_rho + a2g + rho) - 1, 2) * 
      theta_HI(D, L + lambda[j] + a_rho + a2g) * theta_HI(D, L + lambda[j] + a_rho + a2g + rho) *
      theta_HI(Dv, L + lambda[j] + a_rho + a2g + sigma) * theta_HI(Dv, L + lambda[j] + a_rho + a2g + rho + sigma)
    end
  end

  return first_term - second_term
end


function frobenius_fay_relation(g::Int, thetas::AbstractDict{NTuple{N, Int64}, AcbFieldElem}, rho::Vector{FqFieldElem}, sigma::Vector{FqFieldElem}) where N
  rho_Q = QQ.(lift.(Ref(ZZ), rho))/2
  sigma_Q = QQ.(lift.(Ref(ZZ), sigma))/2
  return frobenius_fay_relation(g, thetas, rho_Q, sigma_Q)
end
  

function frobenius_fay_relation(g::Int, thetas::AbstractDict{NTuple{N, Int64}, AcbFieldElem}, rho::Vector{QQFieldElem}, sigma::Vector{QQFieldElem}) where N
  @req g >= 4 "g needs to be bigger than 3."
  @req mod(half_char_inner_prod(rho, sigma) + half_char_inner_prod(sigma, rho), 2) == 1 "Rho and sigma must not be orthogonal."
  a = aronhold_system(g)
  a = [QQ.(lift.(Ref(ZZ), av))/2 for av in a]
  a0 = sum(a)


  l = Vector{QQFieldElem}[]
  for p in (1:g-3)
    push!(l, a[2*p+6] + a[2*p+7])
  end

  lambda = Vector{QQFieldElem}[]
  l_M = matrix(l)
  V = Iterators.product(repeat([[QQ(0),QQ(1)]], g-3)...)
  for v in V 
    v = [v...]
    push!(lambda, v * l_M)
  end

  A = Vector{QQFieldElem}[]

  for lam in lambda
    for ai in [[a0] ; a[1:7]]
      push!(A, lam+ai)
    end
  end

  L = sum(a[i] for i in (8:2:2*g))
  if mod(g, 2) == 1
    L += a0
  end

  CC = parent(thetas[zeros(Int, 2*g)...])

  D = thetas

  result = zeros(CC, 2^(2*g))


  factor = (-1)^(half_char_inner_prod(L, a[2*g]) + half_char_inner_prod(a[2*g], L) )
 
  for h in (1:2^(g-3)) 
 
    v = L + lambda[h] + sigma
    if mod(half_char_inner_prod(v), 2) == 1
      result[char_to_index(v)] += (-1)^(half_char_inner_prod(lambda[h], lambda[h]) + 
      half_char_inner_prod(L+lambda[h] +a0 + a[2*g], L + lambda[h] + a0 + a[2*g])+half_char_inner_prod(rho+sigma, L + lambda[h])) * 
      mod(half_char_inner_prod(L + lambda[h] + rho) - 1, 2) *
      theta_HI(D, L + lambda[h]) * theta_HI(D, L + lambda[h] + rho) *
      _sign_flip(v) * theta_HI(D, v + rho) * factor
    else 
      result[char_to_index(v + rho)] += (-1)^(half_char_inner_prod(lambda[h], lambda[h]) + 
      half_char_inner_prod(L+lambda[h] +a0 + a[2*g], L + lambda[h] + a0 + a[2*g])+half_char_inner_prod(rho+sigma, L + lambda[h])) * 
      mod(half_char_inner_prod(L + lambda[h] + rho) - 1, 2) *
      theta_HI(D, L + lambda[h]) * theta_HI(D, L + lambda[h] + rho) *
      _sign_flip(v + rho) * theta_HI(D, v) * factor
    end

  end

  for a_rho in [[a0] ; a[1:7]]
    for j in (1:2^(g-4))
      v = L + lambda[j] + a_rho + a[2*g] + sigma
      if mod(half_char_inner_prod(v), 2) == 1
        result[char_to_index(v)] -= (-1)^((half_char_inner_prod(lambda[j] + a_rho, L)  + half_char_inner_prod(L, lambda[j] +a_rho) + half_char_inner_prod(rho + sigma, L + lambda[j] + a_rho + a[2*g]))) *
        mod(half_char_inner_prod(L + lambda[j] + a_rho + a[2*g] + rho) - 1, 2) * 
        theta_HI(D, L + lambda[j] + a_rho + a[2*g]) * theta_HI(D, L + lambda[j] + a_rho + a[2*g] + rho) *
       _sign_flip(v) * theta_HI(D, v + rho)
      else 
        result[char_to_index(v + rho)] -= (-1)^((half_char_inner_prod(lambda[j] + a_rho, L)  + half_char_inner_prod(L, lambda[j] +a_rho) + half_char_inner_prod(rho + sigma, L + lambda[j] + a_rho + a[2*g]))) *
        mod(half_char_inner_prod(L + lambda[j] + a_rho + a[2*g] + rho) - 1, 2) * 
        theta_HI(D, L + lambda[j] + a_rho + a[2*g]) * theta_HI(D, L + lambda[j] + a_rho + a[2*g] + rho) *
        _sign_flip(v + rho) * theta_HI(D, v)
      end
    end
  end

  return result
end


function naive_non_orth_complement(rho::Vector{FqFieldElem})
  V = Iterators.product(repeat([[GF(2)(0),GF(2)(1)]], length(rho))...)
  result = Vector{FqFieldElem}[]
  for v in V 
    v = collect(v)
    if half_char_inner_prod(rho, v) + half_char_inner_prod(v, rho) == 1
      push!(result, v)
    end
  end
  return result
end

function construct_fay_matrix(g::Int, thetas, rho::Vector{FqFieldElem})
  W = naive_non_orth_complement(rho)
  M = Vector{AcbFieldElem}[]
  for w in W 
    v = frobenius_fay_relation(g, thetas, w, rho)
    push!(M, v)
  end
  return M
end

################################################################################
#
#  The structure of construct_fay_matrix, without theta constants
#
#  The same relations as frobenius_fay_relation(g, thetas, w, rho) for the w
#  of naive_non_orth_complement(rho), with the characteristics doubled
#  (integer vectors 2 * char) instead of in QQ: for each relation the terms
#  (column, coefficient, three characteristics), the entry of the column
#  being coefficient * theta[c1] theta[c2] theta[c3] (the sign flips of
#  theta_HI included in the coefficient). Independent of the curve, so it
#  can be computed once per (g, rho); fay_rows evaluates it.
#
################################################################################

# half_char_inner_prod for doubled characteristics
_doubled_inner(n::Vector{Int}, m::Vector{Int}, g::Int) = sum(n[i] * m[g + i] for i in 1:g)
_doubled_inner(n::Vector{Int}, g::Int) = _doubled_inner(n, n, g)
# _sign_flip for doubled characteristics
_doubled_flip(c::Vector{Int}, g::Int) = isodd(sum(c[i] * fld(c[g + i], 2) for i in 1:g)) ? -1 : 1
_doubled_key(c::Vector{Int}) = Tuple(mod.(c, 2))

function _fay_relation_structure(g::Int, rho::Vector{Int}, sigma::Vector{Int}, setup)
  a, a0, lambda, L, factor = setup
  hp(n) = _doubled_inner(n, g)
  hp(n, m) = _doubled_inner(n, m, g)
  flip(c) = _doubled_flip(c, g)
  terms = Tuple{Int, Int, NTuple{2*g, Int}, NTuple{2*g, Int}, NTuple{2*g, Int}}[]
  function add!(sign, base, v)
    # the term theta[base] theta[base + rho] grad theta[v or v + rho] theta[v + rho or v]
    s = sign * flip(base) * flip(base + rho) * flip(v) * flip(v + rho)
    if isodd(hp(v))
      push!(terms, (char_to_index(mod.(v, 2)), s, _doubled_key(base), _doubled_key(base + rho), _doubled_key(v + rho)))
    else
      push!(terms, (char_to_index(mod.(v + rho, 2)), s, _doubled_key(base), _doubled_key(base + rho), _doubled_key(v)))
    end
  end
  for h in 1:2^(g - 3)
    base = L + lambda[h]
    iszero(mod(hp(base + rho) - 1, 2)) && continue
    e = hp(lambda[h], lambda[h]) + hp(base + a0 + a[2*g]) + hp(rho + sigma, base)
    add!((-1)^e * factor, base, base + sigma)
  end
  for a_rho in [[a0]; a[1:7]], j in 1:2^(g - 4)
    base = L + lambda[j] + a_rho + a[2*g]
    iszero(mod(hp(base + rho) - 1, 2)) && continue
    e = hp(lambda[j] + a_rho, L) + hp(L, lambda[j] + a_rho) + hp(rho + sigma, base)
    add!(-(-1)^e, base, base + sigma)
  end
  return terms
end

_fay_setup(g::Int) = _fay_setup(g, [Int.(lift.(Ref(ZZ), av)) for av in aronhold_system(g)])

# the data of the relations for the (doubled) Aronhold system a
function _fay_setup(g::Int, a::Vector{Vector{Int}})
  a0 = sum(a)
  l = [a[2*p + 6] + a[2*p + 7] for p in 1:g - 3]
  lambda = [sum((v[p] * l[p] for p in 1:g - 3); init = zeros(Int, 2*g))
            for v in Iterators.product(ntuple(_ -> 0:1, g - 3)...)]
  lambda = vec(lambda)
  L = sum(a[i] for i in 8:2:2*g)
  isodd(g) && (L += a0)
  factor = (-1)^(_doubled_inner(L, a[2*g], g) + _doubled_inner(a[2*g], L, g))
  return (a, a0, lambda, L, factor)
end

@doc raw"""
    fay_matrix_structure(g, rho) -> Vector

The relations of construct_fay_matrix(g, thetas, rho) without the theta
constants: per relation the terms (column, coefficient, c1, c2, c3) (see
fay_rows).
"""
function fay_matrix_structure(g::Int, rho::Vector{FqFieldElem})
  @req g >= 4 "g needs to be bigger than 3."
  setup = _fay_setup(g)
  sigma = Int.(lift.(Ref(ZZ), rho))
  return [_fay_relation_structure(g, Int.(lift.(Ref(ZZ), w)), sigma, setup)
          for w in naive_non_orth_complement(rho)]
end

@doc raw"""
    fay_rows(structure, thetas, columns) -> Vector{Vector{AcbFieldElem}}

The relations of fay_matrix_structure evaluated at the theta constants,
restricted to `columns` (theta indices, e.g. those of the odd
characteristics; terms in other columns are dropped).
"""
function fay_rows(structure, thetas, columns::Vector{Int})
  CC = parent(first(values(thetas)))
  position = Dict(c => k for (k, c) in enumerate(columns))
  rows = Vector{Vector{AcbFieldElem}}(undef, length(structure))
  for (r, terms) in enumerate(structure)
    row = [zero(CC) for _ in columns]
    for (column, coefficient, c1, c2, c3) in terms
      k = get(position, column, 0)
      k == 0 && continue
      row[k] += coefficient * (thetas[c1] * thetas[c2] * thetas[c3])
    end
    rows[r] = row
  end
  return rows
end

# The largest difference between fay_rows and construct_fay_matrix (a check
# of the port).
function check_fay_rows(g::Int, thetas, rho::Vector{FqFieldElem})
  columns = collect(1:2^(2*g))
  new = fay_rows(fay_matrix_structure(g, rho), thetas, columns)
  old = construct_fay_matrix(g, thetas, rho)
  return maximum(Float64(abs(new[r][c] - old[r][c])) for r in eachindex(old) for c in columns)
end

function number_of_odd_thetas(g)
  return 2^(g-1)*(2^g - 1) 
end

function number_of_even_thetas(g)
  return 2^(g-1)*(2^g + 1) 
end


# The characteristic part of relation_with_term for the matrix M1 = T M (no
# theta constants): for each term of the relation the sorted indices of its
# four characteristics (`term_tup`), its integer coefficient including the
# sign flips of theta_HI (0 for a term with an odd characteristic) and the
# four characteristics reduced to bits (the keys of the thetas).
function _relation_terms(g::Int, terms, M1::FqMatrix)
  rho = QQ.(terms[1])//2 + QQ.(terms[2])//2
  sigma = QQ.(terms[1])//2 + QQ.(terms[3])//2
  a = aronhold_system(g, M1)
  a = [QQ.(lift.(Ref(ZZ), av))/2 for av in a]
  a0 = sum(a)
  l = Vector{QQFieldElem}[]
  for p in (1:g-3)
    push!(l, a[2*p+6] + a[2*p+7])
  end
  lambda = Vector{QQFieldElem}[]
  l_M = matrix(l)
  for v in Iterators.product(repeat([[QQ(0),QQ(1)]], g-3)...)
    push!(lambda, [v...] * l_M)
  end
  L = sum(a[i] for i in (8:2:2*g))
  if mod(g, 2) == 1
    L += a0
  end
  factor = (-1)^(half_char_inner_prod(L, a[2*g]) + half_char_inner_prod(a[2*g], L))
  term_tup = Vector{Int}[]
  term_coefficients = Int[]
  theta_keys = Vector{Vector{Int}}[]
  function add_term!(c, v1)
    chars = [v1, v1 + rho, v1 + sigma, v1 + sigma + rho]
    c *= mod(half_char_inner_prod(chars[2]) - 1, 2) * prod(_sign_flip(v) for v in chars)
    push!(term_coefficients, c)
    push!(theta_keys, [mod.(Int.(2*v), Ref(2)) for v in chars])
    push!(term_tup, sort(char_to_index.(chars)))
  end
  for h in (1:2^(g-3))
    v1 = L + lambda[h]
    add_term!(factor * (-1)^(half_char_inner_prod(lambda[h], lambda[h]) +
              half_char_inner_prod(v1 + a0 + a[2*g], v1 + a0 + a[2*g]) +
              half_char_inner_prod(rho + sigma, v1)), v1)
  end
  for a_rho in [[a0] ; a[1:7]]
    for j in (1:2^(g-4))
      v1 = L + lambda[j] + a_rho + a[2*g]
      add_term!(-(-1)^(half_char_inner_prod(lambda[j] + a_rho, L) + half_char_inner_prod(L, lambda[j] + a_rho) +
                half_char_inner_prod(rho + sigma, v1)), v1)
    end
  end
  return (term_tup = term_tup, coefficients = term_coefficients, keys = theta_keys)
end

# _relation_terms with doubled integer characteristics (no QQ arithmetic):
# the same terms, coefficients and keys.
function _relation_terms_int(g::Int, terms, M1::FqMatrix)
  t = [collect(Int, c) for c in terms]
  rho, sigma = t[1] + t[2], t[1] + t[3]
  a, a0, lambda, L, factor = _fay_setup(g, [Int.(lift.(Ref(ZZ), av)) for av in aronhold_system(g, M1)])
  hp(n) = _doubled_inner(n, g)
  hp(n, m) = _doubled_inner(n, m, g)
  term_tup = Vector{Int}[]
  term_coefficients = Int[]
  theta_keys = Vector{Vector{Int}}[]
  function add_term!(c, v1)
    chars = [v1, v1 + rho, v1 + sigma, v1 + sigma + rho]
    c *= mod(hp(chars[2]) - 1, 2) * prod(_doubled_flip(v, g) for v in chars)
    push!(term_coefficients, c)
    push!(theta_keys, [mod.(v, 2) for v in chars])
    push!(term_tup, sort([char_to_index(mod.(v, 2)) for v in chars]))
  end
  for h in 1:2^(g - 3)
    v1 = L + lambda[h]
    add_term!(factor * (-1)^(hp(lambda[h], lambda[h]) + hp(v1 + a0 + a[2*g]) + hp(rho + sigma, v1)), v1)
  end
  for a_rho in [[a0]; a[1:7]], j in 1:2^(g - 4)
    v1 = L + lambda[j] + a_rho + a[2*g]
    add_term!(-(-1)^(hp(lambda[j] + a_rho, L) + hp(L, lambda[j] + a_rho) + hp(rho + sigma, v1)), v1)
  end
  return (term_tup = term_tup, coefficients = term_coefficients, keys = theta_keys)
end

# Whether _relation_terms_int agrees with _relation_terms for the pairs (a
# check of the port).
check_relation_terms(g::Int, pairs) =
  all(_relation_terms_int(g, term, M1) == _relation_terms(g, term, M1) for (term, M1) in pairs)

# The numerical part: the integral relation between the values of the terms
# (LLL); (false, ...) unless it has more than two terms and no term has a
# repeated characteristic. Two terms only when the other terms vanish (theta
# constants that vanish, e.g. of hyperelliptic curves or products: then the
# relation of Riemann reduces to these two terms; otherwise a relation with
# two terms is spurious).
function _relation_from_terms(thetas, data)
  all(allunique, data.term_tup) || return false, [(Vector{Int64}[], ZZ(0))]
  CC = parent(first(values(thetas)))
  term_values = [c == 0 ? zero(CC) : c * prod(thetas[Tuple(k)] for k in ks)
                 for (c, ks) in zip(data.coefficients, data.keys)]
  # LLL only needs the digits that separate the terms (the relation has
  # coefficients +-1): the precision of the size range of the terms plus a
  # margin; the relation is then checked at full precision (full LLL if not)
  sol = _small_relation(term_values)
  result = filter(x -> x[2] != 0, collect(zip(data.term_tup, sol)))
  nonvanishing = count(v -> !contains_zero(v), term_values)
  length(result) > 2 || (length(result) == 2 && nonvanishing == 2) || return false, [(Vector{Int64}[], ZZ(0))]
  return true, sort!(result)
end

# The last row of integral_left_kernel of the values, computed at the
# precision of their size range + 128 bits instead of the full precision when
# the relation found there also holds at full precision.
function _small_relation(values::Vector{AcbFieldElem})
  CC = parent(values[1])
  exponent(z) = max(_arb_mid_2exp_rr(real(z)), _arb_mid_2exp_rr(imag(z)))
  nonzero = [exponent(v) for v in values if !iszero(v)]
  if length(nonzero) > 1
    low = (maximum(nonzero) - minimum(nonzero)) + 128
    if low < precision(CC) - 64
      # (under a lock: called from threads, and a new precision is a new
      # entry in the cache of fields, see _embedding_lock)
      CClow = lock(() -> AcbField(low), Hecke.RiemannSurfaces._embedding_lock)
      sol = integral_left_kernel(matrix(CClow.(values)))[1][end, :]
      total = sum((c * v for (c, v) in zip(sol, values)); init = zero(CC))
      count(!iszero, sol) > 2 && contains_zero(total) && return sol
    end
  end
  return integral_left_kernel(matrix(values))[1][end, :]
end

_arb_mid_2exp_rr(x::ArbFieldElem) = max(-(Int64(1) << 59),
  ccall((:arf_abs_bound_lt_2exp_si, Hecke.libflint), Int, (Ref{ArbFieldElem},), x))

function relation_with_term(g::Int, thetas::Dict{NTuple{N, Int64}, AcbFieldElem}, terms, T::FqMatrix) where N
  @req g >= 4 "g needs to be bigger than 3."
  v1 = max_noether_characteristics(g)[1]
  M = map_even_characteristic(GF(2).(ZZ.((v1*2))), GF(2).(terms[1]))
  test, result = _relation_from_terms(thetas, _relation_terms(g, terms, T*M))
  return test, result, T*M
end
