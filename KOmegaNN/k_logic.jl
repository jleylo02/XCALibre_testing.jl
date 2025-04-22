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

@generated correct_productionNN!() = begin
    BCs = fieldBCs.parameters
    func_calls = Expr[]
    for i ∈ eachindex(BCs)
        BC = BCs[i]
        if BC <: DirichletFunction
            call = quote
                update_user_boundary!() # JL: args here must mate those in the generated function
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

    (; output, input, network, gradient, cmu) = BC.value
    cmu = 0.09
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
    
        #(; input, output, cmu, B, E, gradient) = BC.value # JL: looking at the KWallFunction struct definition, these values are defined here and exported, so would I have to do this in the above?
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
        
        dUdy = ((cmu^0.25*sqrt(k[cID]))^2/nuc)*gradient
        nutw = nuc*(input/output)
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

       # z = zero(eltype(nzval))
        #for nzi ∈ rowptr[cID]:(rowptr[cID+1] - 1)
         #   nzval[nzi] = z
        #end
        #cIndex = spindex(rowptr, colval, cID, cID)
        ##nzval[cIndex] = one(eltype(nzval))
        #b[cID] = ωc
    end