XCALibre.Discretise.update_user_boundary!(
    BC::NeumannFunction{I,V}, faces, cells, facesID_range, time, config) where{I,V<:NNKWallFunction}= begin

    (; yplus, yplus_s, y, k, nu, network, Uplus, cmu, parameters, layer_states) = BC.value

    @. yplus = (cmu^0.25)*y*sqrt(k.values[facesID_range]')/nu
    @. yplus_s = (yplus - data_mean)/data_std # here we scale to use properly with network, creating a local variable so not to overwrite the y_plus values
    Uplus, layer_states = network(yplus_s, parameters, layer_states)
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
end

@kernel function _set_production_NN!(
    values, BC, fluid, momentum, turbulence, faces, boundary_cellsID, start_ID)
    i = @index(Global)
    fID = i + start_ID - 1 # Redefine thread index to become face ID

    (; yplus, yplus_s, y, Uplus, gradient, data_mean, data_std, cmu) = BC.value 
    (; nu) = fluid
    (; U) = momentum
    (; k) = turbulence

    Uw = SVector{3}(0.0,0.0,0.0)
    cID = boundary_cellsID[fID]
    face = faces[fID]
    nuc = nu[cID]
    (; delta, normal)= face
    yplusi = yplus[i] 
    Uplusi= Uplus[i]
    dUdy_s = gradient(yplus_s[:, i])[1]
    Uscaling = (((cmu^0.25)*sqrt(k[cID]))^2)/nuc
    dUdy = (dUdy_s/data_std)*Uscaling
    nutw = nuc*(yplusi/Uplusi)
    mag_grad_U = XCALibre.ModelPhysics.mag(
        XCALibre.ModelPhysics.sngrad(U[cID], Uw, delta, normal)
        )  
        values[cID] = nutw*mag_grad_U*dUdy
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
end

@kernel function _correct_nut_wall_NN!(
    values, fluid, turbulence, BC, faces, boundary_cellsID, start_ID)
i = @index(Global)
    fID = i + start_ID - 1 # Redefine thread index to become face ID
    
    (; yplus, y, Uplus, gradient, data_mean, data_std, cmu) = BC.value 
    (; nu) = fluid
    (; k, nutf) = turbulence
    
    cID = boundary_cellsID[fID]
    face = faces[fID]
    nuc = nu[cID]
    (; delta, normal)= face
    yplusi = yplus[i] 
    Uplusi= Uplus[i]
        
    nutw = nuc*(yplusi/Uplusi)
    values[fID] = nutw
end