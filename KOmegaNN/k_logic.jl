struct NNKWallFunction{I,O,G,N,T} <: XCALibreUserFunctor
    input::I # vector to hold input yplus value
    output::O # vector to hold network prediction
    gradient::G # vector to hold scaled gradient
    network::N # neural network
    steady::T # this will need to be false to run at every timestep
end
Adapt.@adapt_structure NNKWallFunction

@generated correct_production_NN!(eqnModel, component, faces, cells, facesID_range, time, config) = begin 
    BCs = fieldBCs.parameters
    func_calls = Expr[]
    for i ∈ eachindex(BCs)
        BC = BCs[i]
        if BC <: DirichletFunction # JL: This may have to change when new Neumann type is defined
            call = quote
                update_user_boundary!(eqnModel, component, faces, cells, facesID_range, time, config) 
            end
            push!(func_calls, call)
        end
    end
    quote
    $(func_calls...)
    nothing
    end 
end

XCALibre.Discretise.update_user_boundary!(
    BC::DirichletFunction{I,V}, eqnModel, component, faces, cells, facesID_range, time, config ) 
    where{I,V <:NNKWallFunction} = begin
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
    A = _A(eqnModel)
    b = _b(eqnModel, nothing)
    colval = _colval(A)
    rowptr = _rowptr(A)
    nzval = _nzval(A)

    (; output, input, network, gradient, cmu) = BC.value
    cmu = 0.09
    output = network(input) 
    Pk = model.turbulence.Pk

    # calcualte gradient du+/dy+
    compute_gradient(y) = Zygote.gradient(x -> network(x)[1], y_plus)[1] # needs to be Zygote.jacobian for Lux model
    # for loop to calculate gradient for all values in input
    gradient = [compute_gradient(input[:, i]) for i in 1:size(input, 2)]
    gradient = hcat(gradient...)

    # Execute apply boundary conditions kernel
    kernel! = _update_user_boundary!(backend, workgroup)
    kernel!(eqnModel, component, faces, cells, facesID_range, time, config)
end

# Using Flux NN
@kernel function _update_user_boundary!(eqnModel, component, faces, cells, facesID_range, time, config)
    i = @index(Global)
    fID = i + start_ID - 1 # Redefine thread index to become face ID
    
    (; input, output, cmu, gradient) = BC.value 
    (; nu) = fluid
    (; U) = momentum
    (; k, Pk) = turbulence
    
    Uw = SVector{3}(0.0,0.0,0.0)
    cID = boundary_cellsID[fID]
    face = faces[fID]
    nuc = nu[cID]
    (; delta, normal)= face
    
    yplus = y_plus(k[cID], nuc, delta, cmu)
    input = (yplus .- data_mean) ./ data_std
        
    dUdy = ((cmu^0.25*sqrt(k[cID]))^2/nuc)*gradient
    nutw = nuc*(input/output)
    mag_grad_U = mag(sngrad(U[cID], Uw, delta, normal))
    values[cID] = (nutw)*mag_grad_U*dUdy # corrected Pk
    b[cID] = b[cID] - Pk[cID]*Volume + values[cID]*Volume # JL: this is what needs to be done once the model is passed
end

# K Functor
k_w= NNKWallFunction(
    input,
    output, 
    gradient, 
    network, 
    false
)

k_w_dev = k_w