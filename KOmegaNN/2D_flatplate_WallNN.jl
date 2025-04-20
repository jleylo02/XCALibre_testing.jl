# using Plots
using XCALibre
# using CUDA
using KernelAbstractions
using Adapt
using Flux
using BSON: @load

grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
grid = "flatplate_2D_highRe.unv"

mesh_file = joinpath(grids_dir, grid)
mesh = UNV2D_mesh(mesh_file, scale=0.001)

backend = CPU(); # activate_multithread(backend)
mesh_dev = mesh; workgroup = 1024
# backend = CUDABackend()
# mesh_dev = adapt(backend, mesh); workgroup= 32

velocity = [10, 0.0, 0.0]
nu = 1e-5
Re = velocity[1]*1/nu
k_inlet = 0.375
ω_inlet = 1000

####################################################
struct NNWallFunction{I,O,G,N,T} <: XCALibreUserFunctor
    input::I # vector to hold input yplus value
    output::O # vector to hold network prediction
    gradient::G # vector to hold scaled gradient
    network::N # neural network
    steady::T
end
Adapt.@adapt_structure NNWallFunction

@load "WallNormNN_Flux.bson" network
@load "NNmean.bson" data_mean
@load "NNstd.bson" data_std

XCALibre.Discretise.update_user_boundary!(
    BC::DirichletFunction{I,V}, P, BC, eqn, model, config 
    ) where{I,V <:NNWallFunction} = begin
    # backend = _get_backend(mesh)
    (; hardware) = config
    (; backend, workgroup) = hardware
    
    # Deconstruct mesh to required fields
    mesh = model.domain
    (; faces, boundary_cellsID, boundaries) = mesh

    # Extract physics models
    (; fluid, momentum, turbulence) = model

    # facesID_range = get_boundaries(BC, boundaries)
    boundaries_cpu = get_boundaries(boundaries)
    facesID_range = boundaries_cpu[BC.ID].IDs_range
    start_ID = facesID_range[1]

    # Access equation data and deconstruct sparse array
    # JL: The arguments here need to be altered
    A = _A(eqn)
    b = _b(eqn, nothing)
    colval = _colval(A)
    rowptr = _rowptr(A)
    nzval = _nzval(A)

    (; output, input, network, gradient) = BC.value
    output = network(input) 
    Pk = model.turbulence.Pk

    # calcualte gradient du+/dy+
    compute_gradient(y) = Zygote.gradient(x -> network(x)[1], y)[1] # needs to be Zygote.jacobian for Lux model
    # for loop to calculate gradient for all values in input
    gradient = [compute_gradient(input[:, i]) for i in 1:size(input, 2)]
    gradient = hcat(gradient...)

    # Execute apply boundary conditions kernel
    kernel! = _update_user_boundary!(backend, workgroup)
    kernel!(
        P.values, BC, fluid, momentum, turbulence, faces, boundary_cellsID, start_ID, gradU, ndrange=length(facesID_range)
    )

    #correct_production!(Pk, k.BCs, model, S.gradU, config) # need to change the arguments in this
    
end

# Using Flux NN
@kernel function _update_user_boundary!(values, BC, fluid, momentum, turbulence, faces, boundary_cellsID, start_ID) # arguments defined in the struct need to go here also
        i = @index(Global)
        fID = i + start_ID - 1 # Redefine thread index to become face ID
    
        (; input, output, cmu, B, E, gradient) = BC.value # JL: looking at the KWallFunction struct definition, these values are defined here and exported, so would I have to do this in the above?
        (; nu) = fluid
        (; U) = momentum
        (; k, nut) = turbulence
    
        Uw = U.BCs[BC.ID].value # JL: recently changed
        cID = boundary_cellsID[fID]
        face = faces[fID]
        nuc = nu[cID]
        (; delta, normal)= face
        #uStar = cmu^0.25*sqrt(k[cID])
        #dUdy = uStar/(kappa*delta)
        yplus = y_plus(k[cID], nuc, delta, cmu)
        BC.value.input = (yplus .- data_mean) ./ data_std
        
        dUdy = ((cmu^0.25*sqrt(k[cID]))^2/nuc)*BC.value.gradient
        nutw = nuc*(BC.value.input/BC.value.output)
        mag_grad_U = mag(sngrad(U[cID], Uw, delta, normal))
        values[cID] = (nutw)*mag_grad_U*dUdy # corrected Pk
        # JL: based on conversation with Humberto, I think what needs to be done here is similar to the OmegaWallFunction:
        # JL: need to alter this to update the Pk value that is used in the k equation (I think, need to sit and think about this)
        # Classic approach
        # b[cID] += A[cID,cID]*ωc
        # A[cID,cID] += A[cID,cID]
        
        # nzIndex = spindex(rowptr, colval, cID, cID)
        # Atomix.@atomic b[cID] += nzval[nzIndex]*ωc
        # Atomix.@atomic nzval[nzIndex] += nzval[nzIndex] 

        z = zero(eltype(nzval))
        for nzi ∈ rowptr[cID]:(rowptr[cID+1] - 1)
            nzval[nzi] = z
        end
        cIndex = spindex(rowptr, colval, cID, cID)
        nzval[cIndex] = one(eltype(nzval))
        b[cID] = ωc
    end

model = Physics(
    time = Steady(),
    fluid = Fluid{Incompressible}(nu = nu),
    turbulence = RANS{KOmega}(),
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
    KWallFunction(:wall),
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
    NutWallFunction(:wall), 
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
