# julia -t 4 --gcthreads=4

using Oscar
R = "src/RieSrf/Reconstruction"
for f in ["ReconstructCurveAuxiliary.jl", "RiemannRelations.jl", "SignCorrectionFromMatrices.jl",
          "ThetaGradients.jl", "ReconstructGenusG.jl", "ReconstructGenusGTests.jl", "ThetaGradientsTests.jl"]
  include(joinpath(R, f))
end
leading_coefficient = Hecke.leading_coefficient
ccall((:flint_set_num_threads, Hecke.libflint), Nothing, (Cint,), Threads.nthreads())

results = test_random_quadrics(6; count = 1, prec = 500)




using Pkg; Pkg.precompile()
using Oscar
include("src/RieSrf/Reconstruction/ReconstructCurveAuxiliary.jl")
include("src/RieSrf/Reconstruction/RiemannRelations.jl")
include("src/RieSrf/Reconstruction/SignCorrectionFromMatrices.jl")
include("src/RieSrf/Reconstruction/ReconstructGenusG.jl")
include("src/RieSrf/Reconstruction/ReconstructGenusGTests.jl")
include("src/RieSrf/Reconstruction/ThetaGradientsTests.jl")

include("ReconstructCurveAuxiliary.jl")
include("RiemannRelations.jl")
include("SignCorrectionFromMatrices.jl")
include("ReconstructGenusG.jl")
include("ReconstructGenusGTests.jl")
include("ThetaGradients.jl")
include("ThetaGradientsTests.jl")


leading_coefficient = Hecke.leading_coefficient
g = 8
curve = canonical_test_curve(g; coefficients = -3:3, node_range = -5:5)
prec = 128
RS = riemann_surface_with_differentials(curve.plane, curve.numerators, prec)
_, tau = Hecke.siegel_reduction(_small_period_matrix(RSR.big_period_matrix(RS)))
CC = complex_field(RS)
z = zeros(CC,8)
thetas = Hecke.thetas(z,tau)
