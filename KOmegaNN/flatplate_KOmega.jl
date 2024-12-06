
using XCALibre


mesh_file = "flatplate_v2_scaled.unv"
mesh = UNV2D_mesh(mesh_file, scale=0.001)

# Select backend and setup hardware
backend = CPU()
# backend = CUDABackend() # ru non NVIDIA GPUs
# backend = ROCBackend() # run on AMD GPUs

hardware = set_hardware(backend=backend, workgroup=1024)
# hardware = set_hardware(backend=backend, workgroup=32) # use for GPU backends

mesh_dev = mesh # use this line to run on CPU
# mesh_dev = adapt(backend, mesh)  # Uncomment to run on GPU 

velocity = [1.8, 0.0, 0.0]
nu = 1.5e-5
Re = velocity[1]*2/nu
I = 0.1 #turbulent intensity
k_inlet = 3/2 * (velocity[1]*I)^2 #turbulent kinetic energy

nut_inlet = nu*25#higher the ratio, more stable it is
omega_inlet = k_inlet/nut_inlet #eddy viscocity

model = Physics(
    time = Steady(),
    fluid = Fluid{Incompressible}(nu = nu,),
    turbulence = RANS{KOmega}(),
    energy = Energy{Isothermal}(),
    domain = mesh_dev
    )

@assign! model momentum U (
    Dirichlet(:inlet, velocity), #dirichlet - fixing value
    Neumann(:outlet, 0.0), #neumann - gradient condition
    Wall(:plate, [0.0, 0.0, 0.0]), #preserves shear stress
    Neumann(:top, 0.0),
)

@assign! model momentum p (
    Neumann(:inlet, 0.0),
    Dirichlet(:outlet, 0.0),
    Neumann(:plate, 0.0),
    Neumann(:top, 0.0)
)

@assign! model turbulence k (
    Dirichlet(:inlet, k_inlet), #dirichlet - fixing value
    Neumann(:outlet, 0.0), #neumann - gradient condition
    OmegaWallFunction(:plate), #preserves shear stress
    Neumann(:top, 0.0),
)

@assign! model turbulence omega (
    Dirichlet(:inlet, omega_inlet), #dirichlet - fixing value
    Neumann(:outlet, 0.0), #neumann - gradient condition
    OmegaWallFunction(:plate), #preserves shear stress
    Neumann(:top, 0.0),
)

schemes = ( #the numerics
    U = set_schemes(divergence = Upwind), #upwind- low accuracy, high stability setup, good for making sure things are running as they should
    p = set_schemes(), # no input provided (will use defaults)
    k = set_schemes(divergence = Upwind), #k and omega transport equations similar to momentum equations, so can keep all of the parameters for k and omega the same as U
    omega = set_schemes(divergence = Upwind)
     
)

solvers = (
    U = set_solver(
        model.momentum.U;
        solver      = BicgstabSolver, # Options: GmresSolver
        preconditioner = Jacobi(), # Options: NormDiagonal(), DILU(), ILU0()
        convergence = 1e-7,
        relax       = 0.7,
        rtol = 1e-4,
        atol = 1e-10
    ),
    p = set_solver(
        model.momentum.p;
        solver      = CgSolver, # Options: CgSolver, BicgstabSolver, GmresSolver
        preconditioner = Jacobi(), # Options: NormDiagonal(), LDL() (with GmresSolver)
        convergence = 1e-7,
        relax       = 0.7,
        rtol = 1e-4,
        atol = 1e-10
    ),
    k = set_solver(
        model.turbulence.k;
        solver      = BicgstabSolver, # Options: GmresSolver
        preconditioner = Jacobi(), # Options: NormDiagonal(), DILU(), ILU0()
        convergence = 1e-7,
        relax       = 0.7,
        rtol = 1e-4,
        atol = 1e-10
    ),
    omega = set_solver(
        model.turbulence.omega;
        solver      = BicgstabSolver, # Options: GmresSolver
        preconditioner = Jacobi(), # Options: NormDiagonal(), DILU(), ILU0()
        convergence = 1e-7,
        relax       = 0.7,
        rtol = 1e-4,
        atol = 1e-10
    )
)

runtime = set_runtime(iterations=2000, time_step=1, write_interval=100)  #new simulation so save more often


config = Configuration(
    solvers=solvers, schemes=schemes, runtime=runtime, hardware=hardware)

initialise!(model.momentum.U, velocity) #gives the inital guess for velocity, could be where my ML comes in handy
initialise!(model.momentum.p, 0.0) #inital guess for pressure (incompressible so should be zero)
initialise!(model.turbulence.k, k_inlet) #inital guess for k 
initialise!(model.turbulence.omega, omega_inlet) #inital guess for omega

residuals = run!(model, config);

tauw, pos = wall_shear_stress(:plate, model)

x = [pos[i][1] for i ∈ eachindex(pos)]
 
using Plots

 plot(x,tauw.x.values)

 x_ss = [x[125],x[67], x[29],x[1],]

 tauw_ss = [tauw.x.values[125],tauw.x.values[67],tauw.x.values[29],tauw.x.values[1]] # values are different than when
 # simulation last ran (before installing XCALibre as dev), seem to be off by an order of 10 

 plot(x, tauw.x.values/(0.5*velocity[1]^2)) # Cf plot seems to show same error, with values off by order of 10
