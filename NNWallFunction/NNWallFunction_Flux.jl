"""
This Neural Network can be used to predict U+ values for a given y+ on k-ω turbulence model simulations.       
The network takes a specified y+ as the input, and outputs the corresponding U+ value at the cell centre.
The network has been trained using k-ω simulation data, obtained from 2D flat plate simulations in XCALibre.
The network is compatible with any data in .csv format, allowing it to be retrained on another dataset
THE INPUT MUST BE A VECTOR (INSIDE []) 
The network has been constructed using Flux.jl, a popular machine learning library, as well as using a number of other Julia packages (see below).
This network is recommended for less experienced users as Flux.jl offers a more accessible API and structure for inexperienced users.
For users who have prior experience in Machine Learning and Neural Networks, the Lux.jl network is recommended, which can be found 
in the same repository, with the name KOmegaNN_Lux.jl. Lux.jl is recommended for more experienced users, as the structure is more advanced and offers more 
fine tuning of the parameters and network training, which may be difficult to grasp at an introductory level of knowledge. 

The current architecture has been refined for the provided dataset, meaning that this structure may need to be altered if the training data is altered.

Descriptive comments are included throughout the script, to aid in understanding and accessibility. 

Towards the end of the script, gradient calculations are performed to obtain the gradient of the network output using auto differentiation, using Zygote.jl.
The calculation of the training data velocity gradient is also performed, using the finite difference method. 
The network gradient has then been scaled using the turbulent kinetic energy (k) obtained from the simulations, to allow the plotting and comparison of the 
two velocity gradients.

This framework has many potential applications, with this particular architecture being used as a replacement wall function in k-ω turbulence model
simulations within XCALibre.

"""

using CSV, DataFrames, Flux, Plots, Statistics, Zygote
m = DataFrame(CSV.File("training_data_integration.csv")) #import training data from csv File

# Data processing
y_train = Array(m[:, 6]) 
y_train= y_train' # Converts the data into correct format for network input
u_train = Array(m[:, 5])
u_train = u_train'

y_train = Float32.(y_train) # Flux.jl layers require a Float32 input. If not done explicitly, this conversion is done implicitly, which slows down the network


# Normalise data
data_mean = mean(y_train)
data_std = std(y_train)
y_train_n = (y_train .- data_mean) ./ data_std

# Define the network architecture
network = Flux.Chain(
    Flux.Dense(1 => 10, sigmoid), # 1 input feature (y+) mapped to 10 neurons, with sigmoid activation function
    Flux.Dense(10 => 10, tanh), # hidden layer, mapping 10 neurons to 10, using tanh activation function
    Flux.Dense(10 => 1), # output layer, mapping 10 neurons to 1 label (u+) (no activation function)
)

# Loss function

loss(network,x,y) = Flux.mse(network(x), y)

# Optimiser function 

opt = Flux.setup(Flux.Adam(0.001, (0.9, 0.999), 1e-8), network) #standard ADAM parameters, can be changed

# Training loop

loss_history = [] # empty array to write loss history to
epochs = [] # empty array to write epochs to
data = [( y_train_n, u_train)] # defines training data for loop

@time for epoch in 1:15000
    Flux.train!(loss, network, data, opt)
    push!(epochs, epoch)
    push!(loss_history, loss(network, y_train_n, u_train))
    if loss(network, y_train_n, u_train) < 1e-3 # breaks training loop once loss criteria is achieved
        break
    end
end

# plot training loss history 

plot(epochs, loss_history[:], label="Loss History")
xlabel!("Epochs")
ylabel!("Loss")

# Save network to allow it to be called externally
using BSON: @save
@save "WallNormNN_Flux.bson" network
@save "NNmean.bson" data_mean
@save "NNstd.bson" data_std

# Trained network prediction on training data

loss(network, y_train_n, u_train)

u_trained = network(y_train_n)

scatter(y_train[:], u_train[:],    label = "Training data", xscale=:log10)
scatter!(y_train[:], u_trained[:],  label="Prediction")
xlabel!("y+")
ylabel!("U+")
scatter!(legend=:topleft)

############ Calculate gradient of training data using autodiff ####################
# Function to compute gradient dU+/dy+
compute_gradient(y) = Zygote.gradient(x -> network(x)[1], y)[1]
# for loop to calculate gradient for all values in input
gradients = [compute_gradient(y_train_n[:, i]) for i in 1:size(y_train_n, 2)] # use of size rather than eachindex here is not ideal, but doesnt work with the latter
gradients = gradients/data_std # unnormalise network gradients
gradients = hcat(gradients...)
scatter(gradients[:], y_train[:])

u_t = Cmu^0.25 .* k^0.5
nu = 1.5e-5
du_dy_NN = (u_t.^2/nu).* gradients # scaling of network gradients using k
scatter(du_dy_NN[:], y_train[:])

############## Calculate gradient on unseen test data ####################

# Load test data
test = DataFrame(CSV.File("test.csv"))

# Split test data into single velocity profile
u = (test[1:46,1])'
y = (test[1:46,3])'
u = vec(u)
y = vec(y)

du_dy = similar(u)  # Create an array to store the gradients

## Finite difference to calculate REAL velocity gradient
# Forward difference for first point
du_dy[1] = (u[2] - u[1]) / (y[2] - y[1])

# Central difference for interior points
for i in 2:length(y)-1
    du_dy[i] = (u[i+1] - u[i-1]) / (y[i+1] - y[i-1])
end

# Backward difference for last point
du_dy[end] = (u[end] - u[end-1]) / (y[end] - y[end-1])

du_dy = hcat(du_dy...) # convert into format to allow flattening for plotting

# plot REAL velocity gradients against wall distance (delta)
scatter(y[:], du_dy[:], label="Simulation", xlabel="y [-]", ylabel="du/dy [-]")

########### Calculate gradient of NN output ##################
# Data processing
y_test = Array(test[1:46, 6]) 
y_test= y_test' # Converts the data into correct format for network input
y_test = Float32.(y_test)
y_test_n = (y_test .- data_mean) ./ data_std

# Compute gradients - is this calculating Jacobian? off by factor of 100
gradients = [compute_gradient(y_test_n[:, i]) for i in 1:size(y_test_n, 2)]
gradients = gradients/data_std
gradients = hcat(gradients...)
# Scale gradient to du/dy (need to find alternative way to do this, but validates)
Cmu = 0.09
k = Array(test[1:46, 9])'
u_t = Cmu^0.25 .* k.^0.5
nu = 1.5e-5
du_dy_NN = ((u_t.^2/nu).* (gradients)) 

scatter!(y[:], du_dy_NN[:], label = "NN- k scaling")
scatter(y_test[:], k[:], xlabel="y+ [-]", ylabel="k", xscale=:log10)
#scatter(u[:], y[:],)

## Finite difference to calculate SPALDING velocity gradient
Uplus = (test[1:46,5])'
YplusSpal= (test[1:46,7])'
grad_spal = similar(Uplus)
# Forward difference for first point
grad_spal[1] = (Uplus[2] - Uplus[1]) / (YplusSpal[2] - YplusSpal[1])

# Central difference for interior points
for i in 2:length(YplusSpal)-1
    grad_spal[i] = (Uplus[i+1] - Uplus[i-1]) / (YplusSpal[i+1] - YplusSpal[i-1])
end

# Backward difference for last point
grad_spal[end] = (Uplus[end] - Uplus[end-1]) / (YplusSpal[end] - YplusSpal[end-1])
du_dy_Spal = ((u_t.^2/nu).* (grad_spal)) 

scatter!(y[:], du_dy_Spal[:], label = "Spaldings - k scaling")