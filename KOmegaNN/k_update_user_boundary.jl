XCALibre.Discretise.update_user_boundary!(
    BC::NeumannFunction{I,V}, faces, cells, facesID_range, time, config) where{I,V<:NNKWallFunction}= begin
   

    # (; hardware) = config
    # (; backend, workgroup) = hardware
    # kernel_range = length(facesID_range)
    # kernel! = _update_user_boundary!(backend, workgroup, kernel_range)
    # kernel!(BC, eqnModel, component, faces, cells, facesID_range, time, ndrange=kernel_range) 

    # Actually, here you will need to update your output vector field
    (; yplus, yplus_s, y, k, nu, network, output, cmu) = BC.value

    @. yplus = (cmu^0.25)*y*sqrt(k.values[facesID_range]')/nu
    @. yplus_s = (yplus - data_mean)/data_std # here we scale to use properly with network, creating a local variable so not to overwrite the y_plus values
    output .= network(yplus_s) # updateing U+
    nothing


end

function XCALibre.ModelPhysics.set_production!(P, BC::NeumannFunction, model, gradU, config)
    (; hardware) = config
    (; backend, workgroup) = hardware

    # Deconstruct mesh to required fields
    mesh = model.domain
    (; faces, boundary_cellsID, boundaries) = mesh
    (; fluid, momentum, turbulence) = model

    # get boundary information
    boundaries_cpu = get_boundaries(boundaries)
    facesID_range = boundaries_cpu[BC.ID].IDs_range
    start_ID = facesID_range[1]
    
    # Execute apply boundary conditions kernel
    kernel! = _set_production_NN!(backend, workgroup)
    kernel!(
        P.values, BC, fluid, momentum, turbulence, faces, boundary_cellsID, start_ID, ndrange=length(facesID_range)
    )
    nothing
end

@kernel function _set_production_NN!(
    values, BC, fluid, momentum, turbulence, faces, boundary_cellsID, start_ID)
    i = @index(Global)
    fID = i + start_ID - 1 # Redefine thread index to become face ID

    (; yplus, yplus_s, y, output, gradient, data_mean, data_std, cmu) = BC.value 
    (; nu) = fluid
    (; U) = momentum
    (; k) = turbulence

    Uw = SVector{3}(0.0,0.0,0.0)
    cID = boundary_cellsID[fID]
    face = faces[fID]
    nuc = nu[cID]
    (; delta, normal)= face
    # yplus = XCALibre.ModelPhysics.y_plus(k[cID], nuc, delta, cmu) # might be able to revmoe
    # input = (yplus .- data_mean) ./ data_std

    # yplusi = yplus[i] # if time allows a quick a dirty performance trick

    dUdy_s = gradient(yplus_s[:, i])[1]
    dUdy = dUdy_s/data_std
    # dUdy = ((cmu^0.25*sqrt(k[cID]))^2/nuc)*gradient
    nutw = nuc*(yplus[i]/output[i])
    mag_grad_U = XCALibre.ModelPhysics.mag(
        XCALibre.ModelPhysics.sngrad(U[cID], Uw, delta, normal)
        ) # JL: add the XCALibre.ModelPhysics to this line also?
    

    if yplus[i] > 11.25
        values[cID] = (nu[cID] + nutw)*mag_grad_U*dUdy 
        # values[cID] = nutw*mag_grad_U*dUdy
    else
        values[cID] = 0.0
    end
end

function XCALibre.ModelPhysics.correct_nut_wall!(νtf, BC::NeumannFunction, model, config)
    (; hardware) = config
    (; backend, workgroup) = hardware
    
    # Deconstruct mesh to required fields
    mesh = model.domain
    (; faces, boundary_cellsID, boundaries) = mesh
    (; fluid, turbulence) = model

    # Get boundary information
    boundaries_cpu = get_boundaries(boundaries)
    facesID_range = boundaries_cpu[BC.ID].IDs_range
    start_ID = facesID_range[1]

    # Execute apply boundary conditions kernel
    kernel! = _correct_nut_wall_NN!(backend, workgroup)
    kernel!(
        νtf.values, fluid, turbulence, BC, faces, boundary_cellsID, start_ID, ndrange=length(facesID_range)
    )

    nothing
end

@kernel function _correct_nut_wall_NN!(
    values, fluid, turbulence, BC, faces, boundary_cellsID, start_ID)
i = @index(Global)
    fID = i + start_ID - 1 # Redefine thread index to become face ID
    
    (; yplus, y, output, gradient, data_mean, data_std, cmu) = BC.value 
    (; nu) = fluid
    (; k, nutf) = turbulence
    
    cID = boundary_cellsID[fID]
    face = faces[fID]
    nuc = nu[cID]
    (; delta, normal)= face
    # yplus = XCALibre.ModelPhysics.y_plus(k[cID], nuc, delta, cmu)
    # input = (yplus - data_mean) ./ data_std
        
    nutw = nuc*(yplus[i]/output[i])

    if yplus[i] > 11.25
        values[fID] = nutw
    else
        values[fID] = 0.0
    end
end