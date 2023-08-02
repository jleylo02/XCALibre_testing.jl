export Grad
export grad!, source!
export get_scheme

# Define Gradient type and functionality

struct Grad{S<:AbstractScheme,F,R,I,M}
    field::F
    result::R
    correctors::I
    correct::Bool
    mesh::M
end

Grad{S}(phi::ScalarField) where S= begin
    mesh = phi.mesh
    grad = VectorField(mesh)
    F = typeof(phi)
    R = typeof(grad)
    I = eltype(mesh.nodes[1].neighbourCells)
    M = typeof(mesh)
    Grad{S,F,R,I,M}(phi, grad, one(I), false, mesh)
end

Grad{S}(psi::VectorField) where S = begin
    mesh = psi.mesh
    tgrad = TensorField(mesh)
    F = typeof(psi)
    R = typeof(tgrad)
    I = eltype(mesh.nodes[1].neighbourCells)
    M = typeof(mesh)
    Grad{S,F,R,I,M}(psi, tgrad, one(I), false, mesh)
end

# Grad{S}(phi::ScalarField, correctors::Integer) where S = begin 
#     mesh = phi.mesh
#     (; cells) = mesh
#     ncells = length(cells)
#     F = eltype(mesh.nodes[1].coords)
#     I = eltype(mesh.nodes[1].neighbourCells)
#     SF = typeof(phi)
#     M = typeof(mesh)
#     gradx = zeros(F, ncells)
#     grady = zeros(F, ncells)
#     gradz = zeros(F, ncells)
#     Grad{S,I,F,SF,M}(phi, gradx, grady, gradz, correctors, true, mesh)
# end
# get_scheme(term::Grad{S,I,F}) where {S,I,F} = S

Base.getindex(grad::Grad{S,F,R,I,M}, i::Integer) where {S,F,R<:AbstractVector,I,M} = begin
    Tf = eltype(grad.result.x.values)
    SVector{3,Tf}(
        grad.result.x[i], 
        grad.result.y[i], 
        grad.result.z[i]
        )
end

Base.getindex(grad::Grad{S,F,R,I,M}, i::Integer) where {S,F,R<:AbstractTensorField,I,M} = begin
    Tf = eltype(grad.result.xx.values)
    SMatrix{3,3,Tf,9}(
        grad.xx[i],
        grad.xy[i],
        grad.xz[i],
        grad.yx[i],
        grad.yy[i],
        grad.yz[i],
        grad.zx[i],
        grad.zy[i],
        grad.zz[i],
        )
end

# GRADIENT CALCULATION FUNCTIONS

function source!(grad::Grad, phif, phi, BCs; source=true)
    grad!(grad, phif, phi, BCs; source=source)
end

# Mid-point gradient calculation

# function grad!(grad::Grad{Midpoint,TI,TF}, phif, phi, BCs; source=false) where {TI,TF}
#     interpolate!(get_scheme(grad), phif, phi)
#     correct_boundaries!(phif, phi, BCs)
#     green_gauss!(grad, phif; source)
#     for i ∈ 1:2
#         correct_interpolation!(grad, phif, phi)
#         green_gauss!(grad, phif; source)
#     end
# end

# function correct_interpolation!(
#     grad::Grad{Midpoint,TI,TF}, phif::FaceScalarField{TI,TF}, phi::ScalarField{TI,TF}
#     ) where {TI,TF}
#     (; mesh, values) = phif
#     (; faces, cells) = mesh
#     phic = phi.values
#     nbfaces = total_boundary_faces(mesh)
#     start = nbfaces + 1
#     @inbounds @simd for fID ∈ start:length(faces)
#         face = faces[fID]
#         ownerCells = face.ownerCells
#         owner1 = ownerCells[1]
#         owner2 = ownerCells[2]
#         cell1 = cells[owner1]
#         cell2 = cells[owner2]
#         phi1 = phic[owner1]
#         phi2 = phic[owner2]
#         ∇phi1 = grad(owner1)
#         ∇phi2 = grad(owner2)
#         weight = 0.5
#         rf = face.centre 
#         rP = cell1.centre 
#         rN = cell2.centre
#         phifᵖ = weight*(phi1 + phi2)
#         ∇phi = weight*(∇phi1 + ∇phi2)
#         R = rf - weight*(rP + rN)
#         values[fID] = phifᵖ + ∇phi⋅R
#     end
# end

# Linear gradient calculation

function grad!(grad::Grad{Linear,F,R,I,M}, phif, phi, BCs; source=false) where {F,R,I,M}
    interpolate!(phif, phi)
    correct_boundaries!(phif, phi, BCs)
    green_gauss!(grad, phif; source)
    # for i ∈ 1:2
    #     correct_interpolation!(grad, phif, phi)
    #     green_gauss!(grad, phif; source)
    # end
end