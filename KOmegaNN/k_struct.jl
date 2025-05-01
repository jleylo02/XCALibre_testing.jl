struct NNKWallFunction{I,O,G,N,M,S,T} <: XCALibreUserFunctor
    input::I # vector to hold input yplus value
    output::O # vector to hold network prediction
    gradient::G # vector to hold scaled gradient
    network::N # neural network
    data_mean::M # training data mean for normalisation
    data_std::S # training data standard deviation for normalisation
    steady::T # this will need to be false to run at every timestep
end
Adapt.@adapt_structure NNKWallFunction
