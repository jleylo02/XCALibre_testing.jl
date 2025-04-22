"""
This Neural Network can be used to predict U+ values for a given y+ on k-ω turbulence model simulations.       
The network takes a specified y+ as the input, and outputs the corresponding U+ value at the cell centre.
The network has been trained using k-ω simulation data, obtained from 2D flat plate simulations in XCALibre.
The network is compatible with any data in .csv format, allowing it to be retrained on another dataset
THE INPUT MUST BE A VECTOR (INSIDE []) 
The network has been constructed using Lux.jl, a popular machine learning library, as well as using a number of other Julia packages (see below).
This network is recommended for users who have prior experience in Machine Learning and Neural Networks, as the structure is more advanced and offers more 
fine tuning of the parameters and network training, which may be difficult to grasp at an introductory level of knowledge. 
For less experienced users, Flux.jl is recommended, which offers a more accessible API and structure for inexperienced users. The Flux.jl variant of this network
os available in the same repository, named KOmegaNN_Flux.jl
The current architecture has been refined for the provided dataset, meaning that this structure may need to be altered if the training data is altered.

Descriptive comments are included throughout the script, to aid in understanding and accessibility. 

Towards the end of the script, gradient calculations are performed to obtain the gradient of the network output using auto differentiation, using Zygote.jl.
The calculation of the training data velocity gradient is also performed, using the finite difference method. 
The network gradient has then been scaled using the turbulent kinetic energy (k) obtained from the simulations, to allow the plotting and comparison of the 
two velocity gradients.

This framework has many potential applications, with this particular architecture being used as a replacement wall function in k-ω turbulence model
simulations within XCALibre.

"""

using Lux, CSV, DataFrames, Plots, Statistics, Random, Distributions, LuxLib, DataFrames, Optimisers, Zygote
m = DataFrame(CSV.File("training_data_integration.csv")) #import training data from csv File
rng=Xoshiro(42) # specifies rng seed used to generate initial parameters, constant seed allows reproducibility 

# Data processing
y_train = Array(m[:, 6]) 
y_train= y_train' # Converts the data into correct format for network input
u_train = Array(m[:, 5])
u_train = u_train'
y_train = Float32.(y_train)

# Normalise data
data_mean = mean(y_train)
data_std = std(y_train)
y_train_n = (y_train .- data_mean) ./ data_std

# Define the network architecture
network = Lux.Chain(
    Lux.Dense(1 => 10, sigmoid), # 1 input feature (y+) mapped to 10 neurons, with sigmoid activation function
    Lux.Dense(10 => 10, tanh), # hidden layer, mapping 10 neurons to 10, using tanh activation function
    Lux.Dense(10 => 1), # output layer, mapping 10 neurons to 1 label (u+) (no activation function)
)

# Initialise parameters and layer states

parameters, layer_states = Lux.setup(rng, network) # creates a NamedTuple with the initial guesses at the parameters (layers with weights and biases) 
# layer states (contains NN layers with empty NamedTuples as no states yet)

# The forward (loss) function (using auto diff)

function loss_fn(p, ls) # takes the parameters and layer states
    u_prediction, new_ls = network(y_train_n, p, ls) # produces new y prediction and layer states by querieing network on stated parameters
    loss = 0.5 * mean((u_prediction .- u_train).^2) # MSE loss
    return loss, new_ls
end

# Define optimiser function 

opt = Optimisers.Adam(0.001) # Custom learning rate to reduce loss spikes- change based on loss history/epochs
opt = Optimisers.OptimiserChain(opt, Optimisers.ClipNorm(5.0)) # Gradient clipping to reduce loss spikes
opt_state = Optimisers.setup(opt, parameters)


# Training loop
loss_history = []
epochs = []
@time for epoch in 1:150000
    (loss, layer_states,), back = pullback(loss_fn, parameters, layer_states) # Performs forward pass, calls loss function of current p and ls, and returns new ones.
    grad, _ = back((1.0, nothing)) # backwards pass to allow transform of parameters and ls based on gradients, 
    # nothing and _ doesnt provide information on ls, as this isnt needed

    opt_state, parameters = Optimisers.update(opt_state, parameters, grad) # update step

    push!(loss_history, loss)
    push!(epochs, epoch) # writes loss history every 100 epochs (allows monitoring of loss)
    if epoch % 100 == 0
        println("Epoch: $epoch, Loss: $loss") 
    end
    if loss < 1e-3 # breaks training loop when loss criteria is achieved
        break
    end
end

# Plot training loss history

plot(loss_history, yscale=:log10)

# Save network to allow it to be called externally. Meand and Standard Deviation must be saved and used to scale any external input data, or networks 
# predictions will be incorrect
using BSON: @save
@save "WallNormNN_Lux.bson" network
@save "WallNormNN_p.bson" parameters
@save "WallNormNN_ls.bson" layer_states
@save "NNmean.bson" data_mean
@save "NNstd.bson" data_std

# Trained network prediction on training data

u_trained, new_layer_states = network(y_train_n, parameters, layer_states) 

loss_train = 0.5 * mean((u_trained .- u_train).^2)

scatter(y_train[:], u_train[:],  label = "Training data", xscale=:log10)
scatter!(y_train[:], u_trained[:],  label="Prediction")
xlabel!("y+")
ylabel!("U+")
scatter!(legend=:topleft)

############ Calculate gradient of training data using autodiff ####################
# Function to compute gradient dU+/dy+
compute_gradient(y) = Zygote.jacobian(x -> network(x, parameters, layer_states)[1], y)[1]

# for loop to calculate gradient for all values in input
gradients = [compute_gradient(y_train_n[:, i]) for i in 1:size(y_train_n, 2)] # use of size instead of eachindex isnt ideal, but will not work with the latter
gradients = gradients/data_std # scales gradient back to unnormalised
gradients = hcat(gradients...) # convert into format to allow flattening for plotting

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

# Compute gradients 
gradients = [compute_gradient(y_test_n[:, i]) for i in 1:size(y_test_n, 2)]
gradients = gradients/data_std
gradients = hcat(gradients...)
# Scale gradient to du/dy using k
Cmu = 0.09
k = Array(test[1:46, 9])'
u_t = Cmu^0.25 .* k.^0.5
nu = 1.5e-5
du_dy_NN = ((u_t.^2/nu).* (gradients)) 

scatter!(y[:], du_dy_NN[:], label = "NN")
scatter(y_test[:], k[:], xlabel="y+", ylabel="k", xscale=:log10)
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

scatter!(y[:], du_dy_Spal[:], label = "Spaldings")