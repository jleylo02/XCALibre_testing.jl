XCALibre.Discretise.update_user_boundary!(
    BC::NeumannFunction{I,V}, faces, cells, facesID_range, time, config) where{I,V<:NNKWallFunction}= begin
   

    # (; hardware) = config
    # (; backend, workgroup) = hardware
    # kernel_range = length(facesID_range)
    # kernel! = _update_user_boundary!(backend, workgroup, kernel_range)
    # kernel!(BC, eqnModel, component, faces, cells, facesID_range, time, ndrange=kernel_range) 

    # Actually, here you will need to update your output vector field
    (; input, k, nu, network) = BC.value

    @. input= (0.09^0.25)*input*sqrt(k.values[facesID_range]')/nu
    @. input = (input - data_mean)/data_std # here we scale to use properly with network
    output .= network(input) # updateing U+
    nothing


end

function XCALibre.ModelPhysics.set_production!(P, BC::NeumannFunction, model, gradU, config)
    # # backend = _get_backend(mesh)
    # (; hardware) = config
    # (; backend, workgroup) = hardware

    # # Deconstruct mesh to required fields
    # mesh = model.domain
    # (; faces, boundary_cellsID, boundaries) = mesh

    # # Extract physics models
    # (; fluid, momentum, turbulence) = model

    # # facesID_range = get_boundaries(BC, boundaries)
    # boundaries_cpu = get_boundaries(boundaries)
    # facesID_range = boundaries_cpu[BC.ID].IDs_range
    # start_ID = facesID_range[1]

    # (; output, input, network, gradient) = BC.value
    
    # # Execute apply boundary conditions kernel
    # kernel! = _set_production_NN!(backend, workgroup)
    # kernel!(
    #     P.values, BC, fluid, momentum, turbulence, faces, boundary_cellsID, start_ID, gradU, ndrange=length(facesID_range)
    # )
    nothing
end



@kernel function _set_production_NN!(
    values, BC, fluid, momentum, turbulence, faces, boundary_cellsID, start_ID, gradU)
    i = @index(Global)
    fID = i + start_ID - 1 # Redefine thread index to become face ID

    (; input, output, gradient, data_mean, data_std) = BC.value 
    (; nu) = fluid
    (; U) = momentum
    (; k) = turbulence

    Uw = SVector{3}(0.0,0.0,0.0)
    cID = boundary_cellsID[fID]
    face = faces[fID]
    nuc = nu[cID]
    (; delta, normal)= face
    cmu = 0.09 # minor thing, maybe you want to give this to your user
    yplus = XCALibre.ModelPhysics.y_plus(k[cID], nuc, delta, cmu)
    input = (yplus .- data_mean) ./ data_std
        
    dUdy = ((cmu^0.25*sqrt(k[cID]))^2/nuc)*gradient
    nutw = nuc.*(input./output)
    mag_grad_U = mag(sngrad(U[cID], Uw, delta, normal))
    values[cID] = nutw.*mag_grad_U.*dUdy
end

function XCALibre.ModelPhysics.correct_nut_wall!(νtf, BC::NeumannFunction, model, config)
    # # backend = _get_backend(mesh)
    # (; hardware) = config
    # (; backend, workgroup) = hardware
    
    # # Deconstruct mesh to required fields
    # mesh = model.domain
    # (; faces, boundary_cellsID, boundaries) = mesh

    # # Extract physics models
    # (; fluid, turbulence) = model

    # # facesID_range = get_boundaries(BC, boundaries)
    # boundaries_cpu = get_boundaries(boundaries)
    # facesID_range = boundaries_cpu[BC.ID].IDs_range
    # start_ID = facesID_range[1]

    # (; output, input, network) = BC.value

    # # Execute apply boundary conditions kernel
    # kernel! = _correct_nut_wall_NN!(backend, workgroup)
    # kernel!(
    #     νtf.values, fluid, turbulence, BC, faces, boundary_cellsID, start_ID, ndrange=length(facesID_range)
    # )

    nothing
end

@kernel function _correct_nut_wall_NN!(
    values, fluid, turbulence, BC, faces, boundary_cellsID, start_ID)
i = @index(Global)
    fID = i + start_ID - 1 # Redefine thread index to become face ID
    
    (; input, output, data_mean, data_std) = BC.value  
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
    values[fID] = nutw
end