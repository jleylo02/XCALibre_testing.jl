struct NNNutwWallFunction{I,O,G,N,T} <: XCALibreUserFunctor
    input::I # vector to hold input yplus value
    output::O # vector to hold network prediction
    network::N # neural network
    steady::T # this will need to be false to run at every timestep
end
Adapt.@adapt_structure NNNutwWallFunction

@generated correct_eddy_viscosity_NN!(vtf, BC, eqnModel, component, faces, cells, facesID_range, time, config) = begin
    BCs = fieldBCs.parameters
    func_calls = Expr[]
    for i ∈ eachindex(BCs)
        BC = BCs[i]
        if BC <: NeumannFunction # JL: This may have to change when new Neumann type is defined
            call = quote
                update_user_boundary!(vtf, BC, eqnModel, component, faces, cells, facesID_range, time, config) # JL: args here must mate those in the generated function
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
    BC::NeumannFunction{I,V}, BC, eqnModel, component, faces, cells, facesID_range, time, config ) 
    where{I,V <:NNNutwWallFunction} = begin
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

    (; output, input, network, cmu) = BC.value
    cmu = 0.09
    output = network(input) 

    # Execute apply boundary conditions kernel
    kernel! = _update_user_boundary!(backend, workgroup)
    kernel!(BC, eqnModel, component, faces, cells, facesID_range, time, config, 
    νtf.values, fluid, turbulence, boundary_cellsID, start_ID, ndrange=length(facesID_range))
end

# Using Flux NN
@kernel function _update_user_boundary!(BC, eqnModel, component, faces, cells, facesID_range, time, 
    config, values, fluid, turbulence, boundary_cellsID, start_ID)
    i = @index(Global)
    fID = i + start_ID - 1 # Redefine thread index to become face ID
    
    (; input, output, cmu) = BC.value 
    (; nu) = fluid
    (; k) = turbulence
    
    Uw = SVector{3}(0.0,0.0,0.0)
    cID = boundary_cellsID[fID]
    face = faces[fID]
    nuc = nu[cID]
    (; delta, normal)= face
    yplus = y_plus(k[cID], nuc, delta, cmu)
    input = (yplus .- data_mean) ./ data_std
        
    nutw = nuc*(input/output)
    values[fID] = nutw
end

# Nutw Functor
nut_w= NNNutwWallFunction(
    input,
    output,  
    network, 
    false
)

nut_w_dev = nut_w