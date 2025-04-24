export XCALibreUserFunctor
export DirichletFunction

abstract type XCALibreUserFunctor end

"""
    DirichletFunction(ID, value) <: AbstractDirichlet

Dirichlet boundary condition defined with user-provided function.

# Input
- `ID` Boundary name provided as symbol e.g. :inlet
- `value` Custom function for Dirichlet boundary condition.

# Function requirements

The function passed to this boundary condition must have the following signature:

    f(coords, time, index) = SVector{3}(ux, uy, uz)

Where, `coords` is a vector containing the coordinates of a face, `time` is the current time in transient simulations (and the iteration number in steady simulations), and `index` is the local face index (from 1 to `N`, where `N` is the number of faces in a given boundary). The function must return an SVector (from StaticArrays.jl) representing the velocity vector. 
"""
struct DirichletFunction{I,V} <: AbstractDirichlet
    ID::I
    value::V
end
Adapt.@adapt_structure DirichletFunction

function fixedValue(BC::DirichletFunction, ID::I, value::V) where {I<:Integer,V}
    # Exception 1: Value is scalar
    if V <: Number
        return DirichletFunction{I,typeof(value)}(ID, value)
    # Exception 2: value is a function
    elseif V <: Function
        return DirichletFunction{I,V}(ID, value)
    # Exception 3: value is a user provided XCALibre functor
    elseif V <: XCALibreUserFunctor
        return DirichletFunction{I,V}(ID, value)
    # Error if value is not scalar or tuple
    else
        throw("The value provided should be a scalar or a tuple") #= JL: extended the functionality 
        of this to allow vectors, so may have to change this to allow vector and not throw error =#
    end
end

@define_boundary DirichletFunction Laplacian{Linear} VectorField begin
    J = term.flux[fID]
    (; area, delta, centre) = face 
    flux = J*area/delta
    ap = term.sign*(-flux)
    # bc.value.update!(bc.value, centre, time, i)
    value = bc.value(centre, time, i)[component.value]
    ap, ap*value
end

@define_boundary DirichletFunction Divergence{Linear} VectorField begin
    flux = -term.flux[fID]
    ap = term.sign*(flux)
    value = bc.value(face.centre, time, i)[component.value]
    0.0, ap*value
end

@define_boundary DirichletFunction Divergence{Upwind} VectorField begin
    flux = -term.flux[fID]
    ap = term.sign*(flux)
    value = bc.value(face.centre, time, i)[component.value]
    0.0, ap*value
end

@define_boundary DirichletFunction Divergence{BoundedUpwind} VectorField begin
    flux = -term.flux[fID]
    ap = term.sign*(flux)
    value = bc.value(face.centre, time, i)[component.value]
    flux, ap*value
end

@define_boundary DirichletFunction Si begin
    0.0, 0.0
end

@define_boundary DirichletFunction Laplacian{Linear} ScalarField begin # JL: modify this to behave like a Neumann function, or create a new Neumann user function. 
# JL: need to define my own type to control what i can return
    J = term.flux[fID]
    (; area, delta, centre) = face 
    flux = J*area/delta
    ap = term.sign*(-flux)
    # bc.value.update!(bc.value, centre, time, i)
    value = bc.value(centre, time, i)
    ap, ap*value
end

@define_boundary DirichletFunction Divergence{Linear} ScalarField begin
    flux = -term.flux[fID]
    ap = term.sign*(flux)
    value = bc.value(face.centre, time, i)
    0.0, ap*value
end

@define_boundary DirichletFunction Divergence{Upwind} ScalarField begin
    flux = -term.flux[fID]
    ap = term.sign*(flux)
    value = bc.value(face.centre, time, i)
    0.0, ap*value
end

@define_boundary DirichletFunction Divergence{BoundedUpwind} ScalarField begin
    flux = -term.flux[fID]
    ap = term.sign*(flux)
    value = bc.value(face.centre, time, i)
    flux, ap*value
end