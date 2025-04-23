# using Plots
using XCALibre
# using CUDA
using Flux
using BSON: @load

grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
grid = "flatplate_2D_highRe.unv"

mesh_file = joinpath(grids_dir, grid)
mesh = UNV2D_mesh(mesh_file, scale=0.001)

# Using Flux NN
# includet("KOmegaNN_Flux.jl")
@load "WallNormNN_Flux.bson" network
@load "NNmean.bson" data_mean
@load "NNstd.bson" data_std

# Using Lux NN
# includet("KOmegaNN_Lux.jl")
@load "WallNormNN_Lux.bson" network
@load "WallNormNN_ls.bson" layer_states
@load "WallNormNN_p.bson" parameters
@load "NNmean.bson" data_mean
@load "NNstd.bson" data_std

## JL: do I need this bit?? ##
(bc::Inflow)(vec, t, i) = begin
    velocity = @view bc.output[:,i]
    return @inbounds SVector{3}(velocity[1], velocity[2], velocity[3])
    # return @inbounds SVector{3}(velocity)
end

backend = CPU(); # activate_multithread(backend)
mesh_dev = mesh; workgroup = 1024
# backend = CUDABackend()
# mesh_dev = adapt(backend, mesh); workgroup= 32

velocity = [10, 0.0, 0.0]
nu = 1e-5
Re = velocity[1]*1/nu
k_inlet = 0.375
ω_inlet = 1000

ncells = mesh.boundary_cellsID[mesh.boundaries[1].IDs_range] |> length
input = zeros(1,ncells)
input .= (input .- data_mean) ./ data_std
output = network(input)

# K Functor
k_w= NNKWallFunction(
    input,
    output, 
    gradient, 
    network, 
    false
)

k_w_dev = k_w

model = Physics(
    time = Steady(),
    fluid = Fluid{Incompressible}(nu = nu),
    turbulence = RANS{KOmegaNN}(),
    energy = Energy{Isothermal}(),
    domain = mesh_dev
    )

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
    DirichletFunction(:wall, k_w_dev),
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
    DirichletFunction(:wall, nut_w_dev), 
    Neumann(:top, 0.0)
)

schemes = (
    U = set_schemes(divergence=Upwind, gradient=Midpoint),
    p = set_schemes(divergence=Upwind, gradient=Midpoint),
    k = set_schemes(divergence=Upwind, gradient=Midpoint),
    omega = set_schemes(divergence=Upwind, gradient=Midpoint)
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
