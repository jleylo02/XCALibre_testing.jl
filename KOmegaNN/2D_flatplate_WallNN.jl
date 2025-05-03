# using Plots
using XCALibre
# using CUDA
using Flux
# using Lux
using BSON: @load
using StaticArrays
using Statistics
using LinearAlgebra
using KernelAbstractions
using Adapt
using Zygote

grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
grid = "flatplate_2D_highRe.unv"

mesh_file = joinpath(grids_dir, grid)
mesh = UNV2D_mesh(mesh_file, scale=0.001)

includet("k_struct.jl")
includet("k_update_user_boundary.jl")
# Using Flux NN
# includet("KOmegaNN_Flux.jl")
@load "KOmegaNN/WallNormNN_Flux.bson" network
@load "KOmegaNN/NNmean.bson" data_mean
@load "KOmegaNN/NNstd.bson" data_std

# Using Lux NN
# includet("KOmegaNN_Lux.jl")
# @load "WallNormNN_Lux.bson" network
# @load "WallNormNN_ls.bson" layer_states
# @load "WallNormNN_p.bson" parameters
# @load "NNmean.bson" data_mean
# @load "NNstd.bson" data_std

backend = CPU(); # activate_multithread(backend)
mesh_dev = mesh; workgroup = 1024
# backend = CUDABackend()
# mesh_dev = adapt(backend, mesh); workgroup= 32

velocity = [10, 0.0, 0.0]
nu = 1e-5
Re = velocity[1]*1/nu
k_inlet = 0.375
ω_inlet = 1000
cmu = 0.09

# here we need a function to extract the y values of all boundary cells for that patch
patchID = 3 # this is the wall ID
wall_faceIDs = mesh.boundaries[patchID].IDs_range

## initialise memory
nbfaces = wall_faceIDs |> length
Uplus = Float32.(zeros(1,nbfaces)) 
y = Float32.(zeros(1,nbfaces))
yPlus = Float32.(zeros(1,nbfaces))
yPlus_s = Float32.(zeros(1,nbfaces)) 

for fi ∈ eachindex(wall_faceIDs)
    fID = wall_faceIDs[fi]
    face = mesh.faces[fID]
    y[1,fi] = face.delta
end


Uplus = network(yPlus_s)
NNgradient(y_plus) = Zygote.gradient(x -> network(x)[1], y_plus)[1] # maybe make var name better
NNgradient(yPlus_s[:, 500])[1] # this is how you call a single value

model = Physics(
    time = Steady(),
    fluid = Fluid{Incompressible}(nu = nu),
    turbulence = RANS{KOmega}(),
    energy = Energy{Isothermal}(),
    domain = mesh_dev
    )

k_w = NNKWallFunction(
    Uplus, NNgradient, network, model.turbulence.k, nu, data_mean, data_std, cmu, y,yPlus, yPlus_s, false
)


@. yPlus = (0.09^0.25)*y*sqrt(model.turbulence.k.values[wall_faceIDs]')/nu
@. yPlus_s = (yPlus - data_mean)/data_std

Uplus = network(yPlus_s)


@assign! model momentum U (
    Dirichlet(:inlet, velocity),
    Neumann(:outlet, 0.0),
    Wall(:wall, [0.0, 0.0, 0.0]),
    # Dirichlet(:wall, [0.0, 0.0, 0.0]),
    Neumann(:top, 0.0)
)

@assign! model momentum p (
    Neumann(:inlet, 0.0),
    Dirichlet(:outlet, 0.0),
    Neumann(:wall, 0.0),
    Neumann(:top, 0.0)
)

@assign! model turbulence k (
    Dirichlet(:inlet, k_inlet),
    Neumann(:outlet, 0.0),
    NeumannFunction(:wall, k_w),
    # Neumann(:wall, 0.0),
    Neumann(:top, 0.0)
)

@assign! model turbulence omega (
    Dirichlet(:inlet, ω_inlet),
    Neumann(:outlet, 0.0),
    OmegaWallFunction(:wall),
    Neumann(:top, 0.0)
)

@assign! model turbulence nut (
    Dirichlet(:inlet, k_inlet/ω_inlet),
    Neumann(:outlet, 0.0),
    NeumannFunction(:wall, k_w), 
    Neumann(:top, 0.0)
)

schemes = (
    U = set_schemes(divergence=Upwind, gradient=Orthogonal),
    p = set_schemes(divergence=Upwind, gradient=Orthogonal),
    k = set_schemes(divergence=Upwind, gradient=Orthogonal),
    omega = set_schemes(divergence=Upwind, gradient=Orthogonal)
)


solvers = (
    U = set_solver(
        model.momentum.U;
        solver      = BicgstabSolver, # BicgstabSolver, GmresSolver
        preconditioner = Jacobi(), 
        convergence = 1e-7,
        relax       = 0.8,
    ),
    p = set_solver(
        model.momentum.p;
        solver      = CgSolver, # BicgstabSolver, GmresSolver
        preconditioner = Jacobi(), 
        convergence = 1e-7,
        relax       = 0.2,
    ),
    k = set_solver(
        model.turbulence.k;
        solver      = BicgstabSolver, # BicgstabSolver, GmresSolver
        preconditioner = Jacobi(), 
        convergence = 1e-7,
        relax       = 0.3,
    ),
    omega = set_solver(
        model.turbulence.omega;
        solver      = BicgstabSolver, # BicgstabSolver, GmresSolver
        preconditioner = Jacobi(), 
        convergence = 1e-7,
        relax       = 0.3,
    )
)

runtime = set_runtime(iterations=500, write_interval=100, time_step=1)
# runtime = set_runtime(iterations=1, write_interval=100, time_step=1)

hardware = set_hardware(backend=backend, workgroup=workgroup)

config = Configuration(
    solvers=solvers, schemes=schemes, runtime=runtime, hardware=hardware)

GC.gc()

initialise!(model.momentum.U, velocity)
initialise!(model.momentum.p, 0.0)
initialise!(model.turbulence.k, k_inlet)
initialise!(model.turbulence.omega, ω_inlet)
initialise!(model.turbulence.nut, k_inlet/ω_inlet)

residuals = run!(model, config) # 9.39k allocs
