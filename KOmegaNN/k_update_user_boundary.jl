XCALibre.Discretise.update_user_boundary!(
    BC::NeumannFunction{I,V}, eqnModel, component, faces, cells, facesID_range, time, config) where{I,V<:NNKWallFunction}= begin)
   

    (; hardware) = config
    (; backend, workgroup) = hardware
    kernel_range = length(facesID_range)
    kernel! = _update_user_boundary!(backend, workgroup, kernel_range)
    kernel!(BC, eqnModel, component, faces, cells, facesID_range, time, ndrange=kernel_range) 
    
    