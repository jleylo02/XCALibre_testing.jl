########### Flux.jl NN struct ###################
struct NNKWallFunction{U,G,N,K,K1,M,S,C,Y,Yp,Ys,T} <: XCALibreUserFunctor
    Uplus::U # vector to hold network prediction (U+)
    gradient::G # function to calcuate gradient DU+/dy+
    network::N # neural network
    k::K # turbulent kinetic energy
    nu::K1 # kinematic viscosity (user defined)
    data_mean::M # training data mean for normalisation (used to scale back)
    data_std::S # training data standard deviation for normalisation
    cmu::C # empirical constant (0.09) needed for yplus calculation and gradient scaling
    y::Y # mesh y values (wall distance)
    yplus::Yp # calculated y plus values (need to preserve for calculation of nut wall)
    yplus_s::Ys # scaled y plus values (input to the network)
    steady::T # this will need to be false to run at every timestep
end
Adapt.@adapt_structure NNKWallFunction
