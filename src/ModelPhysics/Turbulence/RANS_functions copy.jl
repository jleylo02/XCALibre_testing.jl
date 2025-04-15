# TO DO: These functions needs to be organised in a more sensible manner
function bound!(field, config)
    # Extract hardware configuration
    (; hardware) = config
    (; backend, workgroup) = hardware

    (; values, mesh) = field
    (; cells, cell_neighbours) = mesh

    # set up and launch kernel
    kernel! = _bound!(backend, workgroup)
    kernel!(values, cells, cell_neighbours, ndrange = length(values))
    # KernelAbstractions.synchronize(backend)
end

@kernel function _bound!(values, cells, cell_neighbours)
    i = @index(Global)

    sum_flux = 0.0
    sum_area = 0
    average = 0.0
    @uniform mzero = eps(eltype(values)) # machine zero

    @inbounds begin
        for fi ∈ cells[i].faces_range
            cID = cell_neighbours[fi]
            sum_flux += max(values[cID], mzero) # bounded sum
            sum_area += 1
        end
        average = sum_flux/sum_area

        values[i] = max(
            max(
                values[i],
                average*signbit(values[i])
            ),
            mzero
        )
    end
end

y_plus_laminar(E, kappa) = begin
    yL = 11.0; for i ∈ 1:10; yL = log(max(yL*E, 1.0))/kappa; end
    yL
end

ω_vis(nu, y, beta1) = 6*nu/(beta1*y^2)

ω_log(k, y, cmu, kappa) = sqrt(k)/(cmu^0.25*kappa*y)

y_plus(k, nu, y, cmu) = cmu^0.25*y*sqrt(k)/nu

sngrad(Ui, Uw, delta, normal) = begin
    Udiff = (Ui - Uw)
    Up = Udiff - (Udiff⋅normal)*normal 
    grad = Up/delta 
    return grad
end

mag(vector) = sqrt(vector[1]^2 + vector[2]^2 + vector[3]^2) 

nut_wall(nu, yplus, kappa, E::T) where T = begin
    max(nu*(yplus*kappa/log(max(E*yplus, 1.0 + 1e-4)) - 1), zero(T))
end

@generated constrain_equation!(eqn, fieldBCs, model, config) = begin
    BCs = fieldBCs.parameters
    func_calls = Expr[]
    for i ∈ eachindex(BCs)
        BC = BCs[i]
        if BC <: OmegaWallFunction
            call = quote
                constrain!(eqn, fieldBCs[$i], model, config)
            end
            push!(func_calls, call)
        end
    end
    quote
    $(func_calls...)
    nothing
    end 
end

function constrain!(eqn, BC, model, config)

    # backend = _get_backend(mesh)
    (; hardware) = config
    (; backend, workgroup) = hardware

    # Access equation data and deconstruct sparse array
    A = _A(eqn)
    b = _b(eqn, nothing)
    colval = _colval(A)
    rowptr = _rowptr(A)
    nzval = _nzval(A)
    
    # Deconstruct mesh to required fields
    mesh = model.domain
    (; faces, boundaries, boundary_cellsID) = mesh

    fluid = model.fluid 
    # turbFields = model.turbulence.fields
    turbulence = model.turbulence

    # facesID_range = get_boundaries(BC, boundaries)
    boundaries_cpu = get_boundaries(boundaries)
    facesID_range = boundaries_cpu[BC.ID].IDs_range
    start_ID = facesID_range[1]

    # Execute apply boundary conditions kernel
    kernel! = _constrain!(backend, workgroup)
    kernel!(
        turbulence, fluid, BC, faces, start_ID, boundary_cellsID, colval, rowptr, nzval, b, ndrange=length(facesID_range)
    )
end

@kernel function _constrain!(turbulence, fluid, BC, faces, start_ID, boundary_cellsID, colval, rowptr, nzval, b)
    i = @index(Global)
    fID = i + start_ID - 1 # Redefine thread index to become face ID

    @uniform begin
        nu = fluid.nu
        k = turbulence.k
        (; kappa, beta1, cmu, B, E, yPlusLam) = BC.value
    end
    ωc = zero(eltype(nzval))
    
    @inbounds begin
        cID = boundary_cellsID[fID]
        face = faces[fID]
        y = face.delta
        ωvis = ω_vis(nu[cID], y, beta1)
        ωlog = ω_log(k[cID], y, cmu, kappa)
        yplus = y_plus(k[cID], nu[cID], y, cmu) 

        if yplus > yPlusLam 
            ωc = ωlog
        else
            ωc = ωvis
        end
        # Line below is weird but worked
        # b[cID] = A[cID,cID]*ωc

        
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
end

@generated constrain_boundary!(field, fieldBCs, model, config) = begin
    BCs = fieldBCs.parameters
    func_calls = Expr[]
    for i ∈ eachindex(BCs)
        BC = BCs[i]
        if BC <: OmegaWallFunction
            call = quote
                set_cell_value!(field, fieldBCs[$i], model, config)
            end
            push!(func_calls, call)
        end
    end
    quote
    $(func_calls...)
    nothing
    end 
end

function set_cell_value!(field, BC, model, config)
    # backend = _get_backend(mesh)
    (; hardware) = config
    (; backend, workgroup) = hardware
    
    # Deconstruct mesh to required fields
    mesh = model.domain
    (; faces, boundaries, boundary_cellsID) = mesh
    (; fluid, turbulence) = model
    # turbFields = turbulence.fields

    # facesID_range = get_boundaries(BC, boundaries)
    boundaries_cpu = get_boundaries(boundaries)
    facesID_range = boundaries_cpu[BC.ID].IDs_range
    start_ID = facesID_range[1]

    # Execute apply boundary conditions kernel
    kernel! = _set_cell_value!(backend, workgroup)
    kernel!(
        field, turbulence, fluid, BC, faces, start_ID, boundary_cellsID, ndrange=length(facesID_range)
    )
end

@kernel function _set_cell_value!(field, turbulence, fluid, BC, faces, start_ID, boundary_cellsID)
    i = @index(Global)
    fID = i + start_ID - 1 # Redefine thread index to become face ID

    @uniform begin
        (; nu) = fluid
        (; k) = turbulence
        (; kappa, beta1, cmu, B, E, yPlusLam) = BC.value
        (; values) = field
        ωc = zero(eltype(values))
    end


    @inbounds begin
        cID = boundary_cellsID[fID]
        face = faces[fID]
        y = face.delta
        ωvis = ω_vis(nu[cID], y, beta1)
        ωlog = ω_log(k[cID], y, cmu, kappa)
        yplus = y_plus(k[cID], nu[cID], y, cmu) 

        if yplus > yPlusLam 
            ωc = ωlog
        else
            ωc = ωvis
        end

        values[cID] = ωc # needs to be atomic?
    end
end

correct_production!(Pk, k.BCs, model, S.gradU, config)

@generated correct_production!(P, fieldBCs, model, gradU, config) = begin
    BCs = fieldBCs.parameters
    func_calls = Expr[]
    for i ∈ eachindex(BCs)
        BC = BCs[i]
        if BC <: KWallFunction 
            call = quote
                set_production!(P, fieldBCs[$i], model, gradU, config)
            end
            push!(func_calls, call)
        end
    end
    quote
    $(func_calls...)
    nothing
    end 
end

function set_production!(P, BC, model, gradU, config)
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

    # Execute apply boundary conditions kernel
    kernel! = _set_production!(backend, workgroup)
    kernel!(
        P.values, BC, fluid, momentum, turbulence, faces, boundary_cellsID, start_ID, gradU, ndrange=length(facesID_range)
    )
end

@kernel function _set_production!(
    values, BC, fluid, momentum, turbulence, faces, boundary_cellsID, start_ID, gradU)
    i = @index(Global)
    fID = i + start_ID - 1 # Redefine thread index to become face ID

    (; kappa, beta1, cmu, B, E, yPlusLam) = BC.value
    (; nu) = fluid
    (; U) = momentum
    (; k, nut) = turbulence

    Uw = SVector{3}(0.0,0.0,0.0)
    cID = boundary_cellsID[fID]
    face = faces[fID]
    nuc = nu[cID]
    (; delta, normal)= face
    uStar = cmu^0.25*sqrt(k[cID])
    dUdy = uStar/(kappa*delta)
    yplus = y_plus(k[cID], nuc, delta, cmu)
    nutw = nut_wall(nuc, yplus, kappa, E)
    mag_grad_U = mag(sngrad(U[cID], Uw, delta, normal)) 
    # mag_grad_U = mag(gradU[cID]*normal)
    if yplus > yPlusLam
        values[cID] = (nu[cID] + nutw)*mag_grad_U*dUdy 
    else
        values[cID] = 0.0
    end
end

@generated correct_eddy_viscosity!(νtf, nutBCs, model, config) = begin
    BCs = nutBCs.parameters
    func_calls = Expr[]
    for i ∈ eachindex(BCs)
        BC = BCs[i]
        if BC <: NutWallFunction
            call = quote
                correct_nut_wall!(νtf, nutBCs[$i], model, config)
            end
            push!(func_calls, call)
        end
    end
    quote
    $(func_calls...)
    nothing
    end 
end

function correct_nut_wall!(νtf, BC, model, config)
    # backend = _get_backend(mesh)
    (; hardware) = config
    (; backend, workgroup) = hardware
    
    # Deconstruct mesh to required fields
    mesh = model.domain
    (; faces, boundary_cellsID, boundaries) = mesh

    # Extract physics models
    (; fluid, turbulence) = model

    # facesID_range = get_boundaries(BC, boundaries)
    boundaries_cpu = get_boundaries(boundaries)
    facesID_range = boundaries_cpu[BC.ID].IDs_range
    start_ID = facesID_range[1]

    # Execute apply boundary conditions kernel
    kernel! = _correct_nut_wall!(backend, workgroup)
    kernel!(
        νtf.values, fluid, turbulence, BC, faces, boundary_cellsID, start_ID, ndrange=length(facesID_range)
    )
end

@kernel function _correct_nut_wall!(
    values, fluid, turbulence, BC, faces, boundary_cellsID, start_ID)
    i = @index(Global)
    fID = i + start_ID - 1 # Redefine thread index to become face ID

    (; kappa, beta1, cmu, B, E, yPlusLam) = BC.value
    (; nu) = fluid
    (; k) = turbulence
    
    cID = boundary_cellsID[fID]
    face = faces[fID]
    # nuf = nu[fID]
    (; delta)= face
    # yplus = y_plus(k[cID], nuf, delta, cmu)
    nuc = nu[cID]
    yplus = y_plus(k[cID], nuc, delta, cmu)
    nutw = nut_wall(nuc, yplus, kappa, E)
    if yplus > yPlusLam
        values[fID] = nutw
    else
        values[fID] = 0.0
    end
end

# JL: Initial Neural Network BC integration 

struct WallNormNN{I,O,G,N,T} <: XCALibreUserFunctor
    input::I # vector to hold input yplus value
    output::O # vector to hold network prediction
    gradient::G # vector to hold scaled gradient
    network::N # neural network
    steady::T
end
Adapt.@adapt_structure WallNormNN

using BSON: @save
@save "WallNormNN_Flux.bson" model
@save "NNmean.bson" data_mean
@save "NNstd.bson" data_std

XCALibre.Discretise.update_user_boundary!(
    BC::DirichletFunction{I}, P, BC, model,config 
    ) where{I <:WallNormNN} = begin
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



    (; output, input, network, gradient) = BC.value
    output = network(input) 
    Pk = model.turbulence.Pk

    # Execute apply boundary conditions kernel
    kernel! = _update_user_boundary!(backend, workgroup)
    kernel!(
        P.values, BC, fluid, momentum, turbulence, faces, boundary_cellsID, start_ID, gradU, ndrange=length(facesID_range)
    )

    #correct_production!(Pk, k.BCs, model, S.gradU, config) # need to change the arguments in this
    
end

# Using Flux NN
@kernel function _update_user_boundary!(values, BC, fluid, momentum, turbulence, faces, boundary_cellsID, start_ID)
        i = @index(Global)
        fID = i + start_ID - 1 # Redefine thread index to become face ID
    
        (; input, output, cmu, B, E, yPlusLam) = BC.value
        (; nu) = fluid
        (; U) = momentum
        (; k, nut) = turbulence
    
        Uw = SVector{3}(0.0,0.0,0.0)
        cID = boundary_cellsID[fID]
        face = faces[fID]
        nuc = nu[cID]
        (; delta, normal)= face
        #uStar = cmu^0.25*sqrt(k[cID])
        #dUdy = uStar/(kappa*delta)
        yplus = y_plus(k[cID], nuc, delta, cmu)
        BC.value.input = (yplus .- data_mean) ./ data_std

        # calcualte gradient du+/dy+
        compute_gradient(y) = Zygote.gradient(x -> network(x)[1], y)[1] # needs to be Zygote.jacobian for Lux model
        # for loop to calculate gradient for all values in input
        BC.value.gradient = [compute_gradient(BC.value.input[:, i]) for i in 1:size(BC.value.input, 2)]
        BC.value.gradient = hcat(BC.value.gradient...)
        
        dUdy = ((cmu^0.25*sqrt(k[cID]))^2/nuc)*BC.value.gradient
        nutw = nuc*(BC.value.input/BC.value.output)
        mag_grad_U = mag(sngrad(U[cID], Uw, delta, normal))
        values[cID] = (nutw)*mag_grad_U*dUdy # corrected Pk
    end