struct NNKWallFunction{I,O,G,N,T} <: XCALibreUserFunctor
    input::I # vector to hold input yplus value
    output::O # vector to hold network prediction
    gradient::G # vector to hold scaled gradient
    network::N # neural network
    steady::T # this will need to be false to run at every timestep
end
Adapt.@adapt_structure NNKWallFunction

@generated correct_production_NN!(fieldBCs, eqnModel, component, faces, cells, facesID_range, time, config) = begin 
    BCs = fieldBCs.parameters
    func_calls = Expr[]
    for i ∈ eachindex(BCs)
        BC = BCs[i]
        if BC <: NeumannFunction 
            call = quote
                update_user_boundary!(fieldBCs[$i], eqnModel, component, faces, cells, facesID_range, time, config) 
            end
            push!(func_calls, call)
        end
    end
    quote
    $(func_calls...)
    nothing
    end 
end

update_user_boundary!(
    BC::NeumannFunction{I,V}, eqnModel, component, faces, cells, facesID_range, time, config ) where{I,V <:NNKWallFunction} = begin
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

    A = _A(eqnModel)
    b = _b(eqnModel, nothing)
    colval = _colval(A)
    rowptr = _rowptr(A)
    nzval = _nzval(A)

    (; output, input, network, gradient) = BC.value
    output = network(input) 
    

    # calcualte gradient du+/dy+
    compute_gradient(y_plus) = Zygote.gradient(x -> network(x)[1], y_plus)[1] # needs to be Zygote.jacobian for Lux model
    # for loop to calculate gradient for all values in input
    gradient = [compute_gradient(input[:, i]) for i in 1:size(input, 2)]
    gradient = hcat(gradient...)

    # Execute apply boundary conditions kernel
    kernel_range = length(facesID_range)
    kernel! = _update_user_boundary!(backend, workgroup, kernel_range)
    kernel!(BC, fluid, momentum, turbulence, eqnModel, component, faces, cells, facesID_range, 
    start_ID, boundary_cellsID, time, config, ndrange=kernel_range, colval, rowptr, nzval, b)
end

# Using Flux NN
@kernel function _update_user_boundary!(BC, fluid, momentum, turbulence, eqnModel, component, 
    faces, cells, facesID_range, start_ID, boundary_cellsID, time, config, colval, rowptr, nzval, b)
    i = @index(Global)
    fID = i + start_ID - 1 # Redefine thread index to become face ID
    
    (; input, output, gradient) = BC.value 
    (; nu) = fluid
    (; U) = momentum
    (; k, Pk) = turbulence

    Uw = SVector{3}(0.0,0.0,0.0)
    cID = boundary_cellsID[fID]
    face = faces[fID]
    volume = cells[cID].volume
    nuc = nu[cID]
    (; delta, normal)= face
    cmu = 0.09
    yplus = y_plus(k[cID], nuc, delta, cmu)
    input = (yplus .- data_mean) ./ data_std
        
    dUdy = ((cmu^0.25*sqrt(k[cID]))^2/nuc)*gradient
    nutw = nuc.*(input./output)
    mag_grad_U = mag(sngrad(U[cID], Uw, delta, normal))
    Pk_corrected[cID] = (nutw).*mag_grad_U.*dUdy

    z = zero(eltype(nzval))
    for nzi ∈ rowptr[cID]:(rowptr[cID+1] - 1)
        nzval[nzi] = z
    end
    cIndex = spindex(rowptr, colval, cID, cID)
    nzval[cIndex] = one(eltype(nzval))

    b[cID] = b[cID] - Pk[cID]*volume + Pk_corrected[cID]*volume # JL: this is what needs to be done once the model is passed
end

struct NNNutwWallFunction{I,O,N,T} <: XCALibreUserFunctor
    input::I # vector to hold input yplus value
    output::O # vector to hold network prediction
    network::N # neural network
    steady::T # this will need to be false to run at every timestep
end
Adapt.@adapt_structure NNNutwWallFunction

@generated correct_eddy_viscosity_NN!(fieldBCs, eqnModel, component, faces, cells, facesID_range, time, config) = begin
    BCs = fieldBCs.parameters
    func_calls = Expr[]
    for i ∈ eachindex(BCs)
        BC = BCs[i]
        if BC <: NeumannFunction # JL: This may have to change when new Neumann type is defined
            call = quote
                update_user_boundary!(fieldBCs[$i], eqnModel, component, faces, cells, facesID_range, time, config) # JL: args here must mate those in the generated function
            end
            push!(func_calls, call)
        end
    end
    quote
    $(func_calls...)
    nothing
    end 
end

update_user_boundary!(
    BC::NeumannFunction{I,V}, eqnModel, component, faces, cells, facesID_range, time, config ) where{I,V <:NNNutwWallFunction} = begin
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

    (; output, input, network) = BC.value
    
    output = network(input) 

    # Execute apply boundary conditions kernel
    kernel! = _update_user_boundary!(backend, workgroup)
    kernel!(BC, eqnModel, component, faces, cells, facesID_range, time, config, fluid, turbulence, boundary_cellsID, start_ID, ndrange=length(facesID_range))
end

# Using Flux NN
@kernel function _update_user_boundary!(BC, eqnModel, component, faces, cells, facesID_range, time, 
    config, values, fluid, turbulence, boundary_cellsID, start_ID)
    i = @index(Global)
    fID = i + start_ID - 1 # Redefine thread index to become face ID
    
    (; input, output) = BC.value 
    (; nu) = fluid
    (; k, nutf) = turbulence
    
    cID = boundary_cellsID[fID]
    face = faces[fID]
    nuc = nu[cID]
    (; delta, normal)= face
    cmu = 0.09
    yplus = y_plus(k[cID], nuc, delta, cmu)
    input = (yplus .- data_mean) ./ data_std
        
    nutw = nuc.*(input./output)
    nutf[fID] = nutw
end
