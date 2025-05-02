struct NNKWallFunction{I,O,G,N,M,S,T} <: XCALibreUserFunctor
    input::I # vector to hold input y cell values (user space)
    output::O # vector to hold network prediction (U+)
    gradient::G # function to calcuate gradient DU+/dy+
    network::N # neural network
    data_mean::M # training data mean for normalisation (used to scale back)
    data_std::S # training data standard deviation for normalisation
    steady::T # this will need to be false to run at every timestep
end
Adapt.@adapt_structure NNKWallFunction
