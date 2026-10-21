# julia -t 4 --gcthreads=4

using Oscar
include("src/RieSrf/Reconstruction/ReconstructCurveAuxiliary.jl")
include("src/RieSrf/Reconstruction/RiemannRelations.jl")
include("src/RieSrf/Reconstruction/SignCorrectionFromMatrices.jl")
include("src/RieSrf/Reconstruction/ReconstructGenusG.jl")
include("src/RieSrf/Reconstruction/ReconstructGenusGTests.jl")
include("src/RieSrf/Reconstruction/ThetaGradientsTests.jl")

leading_coefficient = Hecke.leading_coefficient
g = 8
curve = canonical_test_curve(g; coefficients = -3:3, node_range = -5:5)
prec = 128
RS = riemann_surface_with_differentials(curve.plane, curve.numerators, prec)
_, tau = Hecke.siegel_reduction(_small_period_matrix(RSR.big_period_matrix(RS)))
CC = complex_field(RS)
z = zeros(CC,8)
thetas = Hecke.thetas(z,tau)
