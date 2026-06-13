# This file is a part of RobustNeuralNetworks.jl. License is MIT: https://github.com/acfr/RobustNeuralNetworks.jl/blob/main/LICENSE 

cd(@__DIR__)
using Pkg
Pkg.activate("..")

REN_DIR = @__DIR__
# REN_DIR: scalar string path to the folder containing this REN script
DEFAULT_EXP1_DIR = normpath(joinpath(REN_DIR, "..", ".."))
# DEFAULT_EXP1_DIR: scalar string path to Exp1 when this script is run in this repository
EXP1_DIR = get(ENV, "NFIR_EXP1_DIR", DEFAULT_EXP1_DIR)
# EXP1_DIR: scalar string path to the open-source Exp1 folder
DATA_DIR = joinpath(EXP1_DIR, "Training_data")
# DATA_DIR: scalar string path to the Exp1 MATLAB training data folder
RESULTS_DIR = joinpath(EXP1_DIR, "Results")
# RESULTS_DIR: scalar string path to the Exp1 output results folder

using CairoMakie
# using CUDA
using Flux
using Printf
using Random
using RobustNeuralNetworks
using Statistics
#---------------- Milestone 2: tiny input-only REN start ----------------#
using Test
#---------------- Milestone 2: tiny input-only REN end ----------------#
#---------------- Milestone 3: differentiable REN rollout start ----------------#
using Zygote: Buffer
#---------------- Milestone 3: differentiable REN rollout end ----------------#
#---------------- Exp1 REN save trained model start ----------------#
using BSON
#---------------- Exp1 REN save trained model end ----------------#

"""
A note for the interested reader:

- Change `dev = gpu` and `T = Float32` to train the REN observer on an Nvidia GPU with CUDA
- This example is currently not optimised for the GPU, and runs faster on CPU
- It would be easy to re-write it to be much faster on the GPU
- If you feel like doing this, please go ahead and submit a pull request :)

"""

rng = MersenneTwister(0)
dev = cpu
T = Float64

#---------------- Milestone 3: differentiable REN rollout start ----------------#
function rollout_io(model, zinit, u_seq)
    # model:    explicit callable REN
    # zinit:    (nz, batches)
    # u_seq:    Vector of matrices
    # u_seq[t]: (nu_model, batches)
    batches = size(zinit,2) # scalar, batches 
    model_output_seq = Buffer([zeros(eltype(zinit), model.ny, batches)], length(u_seq)) # length(u_seq) slots, each store (ny_model, batches)

    z_t = zinit # (nz, batches)
    for t in eachindex(u_seq)
        u_t = u_seq[t] # (nu, batches)
        z_tpls1, model_output_t = model(z_t, u_t) # z_tple1 (nz, batches): state at next step.  model_output_t (ny, batches) output at current step
        model_output_seq[t] = model_output_t
        z_t = z_tpls1 # (nz, batches)
    end
    return copy(model_output_seq)
end
#---------------- Milestone 3: differentiable REN rollout end ----------------#
#---------------- Milestone 6: simulation-error loss start ----------------#
function loss_io(model_ps, u_seq, y_seq)
    # model_ps: direct REN parameterization containing trainable parameters
    # u_seq:    Vector of matrices
    # u_seq[t]: (nu_model, batches)
    # y_seq:    Vector of matrices
    # y_seq[t]: (ny_model, batches)
    model = REN(model_ps) 
    batches = size(u_seq[1],2) #scalar , batches
    z0 = init_states(model, batches)            # (nz, batches)
    model_output_seq = rollout_io(model,z0,u_seq) # model_output_seq: length(u_seq); each item: (ny_model, batches)
    total_loss = 0.0                            # scalar: unscaled squared 2-norm error
    for t in eachindex(y_seq)
        error_t = y_seq[t] - model_output_seq[t] # error_t: (ny_model, batches)
        total_loss += sum(abs2, error_t) # scalar
    end
    return total_loss
end
#---------------- Milestone 6: simulation-error loss end ----------------#
#####################################################################
# Problem setup

# System parameters
m = 1                   # Mass (kg)
k = 5                   # Spring constant (N/m)
μ = 0.5                 # Viscous damping coefficient (kg/m)
nx_plant  = 2                  # Number of states

# Continuous and discrete dynamics and measurements
_visc(v) = μ * v .* abs.(v)
f(x,u) = [x[2:2,:]; (u[1:1,:] - k*x[1:1,:] - _visc(x[2:2,:]))/m]
fd(x,u) = x + dt*f(x,u)
#---------------- Milestone Passivity : start ----------------#
gd(x) = x[2:2,:]
#---------------- Milestone Passivity : end ----------------#

#---------------- Exp1 REN MATLAB training data start ----------------#

#---------------- Exp1 REN MATLAB training data end ----------------#
include("REN_load_data.jl")
mat_path = joinpath(
    DATA_DIR,
    "Data_M_NLdamper_500B_OneCart.mat",
)
# mat_path: path to MATLAB data file
loaded_data = load_exp1_ren_mat_data(mat_path; T)
# loaded_data: named tuple containing raw matrices and REN sequences
dt = loaded_data.sample_time # dt: scalar sample time = 0.02 seconds
ts_train = 1:loaded_data.n_time_train # ts_train: length 250 range of training time indices
batches_train = loaded_data.n_batch_train # batches_train: scalar total available training trajectories = 500
u_train = loaded_data.u_train # u_train: length 250; each item (1, batches_train) = (1, 500)
y_train = loaded_data.y_train # y_train: length 250; each item (1, batches_train) = (1, 500)
data_io = [(
    u_train |> dev,
    y_train |> dev,
)] # data_io: length 1 # data_io[1]: tuple containing one complete ordered batched rollout
@test length(ts_train) == 250
@test length(u_train) == length(ts_train)
@test length(y_train) == length(ts_train)
@test batches_train == 500
@test size(u_train[1]) == (1, batches_train)
@test size(y_train[1]) == (1, batches_train)
@test dt == T(0.02)
@test all(isfinite, reduce(vcat, u_train))
@test all(isfinite, reduce(vcat, y_train))
@test u_train[1] != u_train[2]
@test y_train[1] != y_train[end]
@test length(data_io) == 1

println("Exp1 REN MATLAB training-data tests passed.")

batch_to_plot_train = 1 # batch_to_plot_train: scalar selected training trajectory
time_train = loaded_data.time_train # time_train: (250,), seconds
u_train_plot = [
    u_train[t][1, batch_to_plot_train]
    for t in eachindex(u_train)
] # u_train_plot: (250,), input signal
y_train_plot = [
    y_train[t][1, batch_to_plot_train]
    for t in eachindex(y_train)
] # y_train_plot: (250,), measured output signal
@test length(time_train) == length(u_train_plot)
@test length(time_train) == length(y_train_plot)
@test all(isfinite, u_train_plot)
@test all(isfinite, y_train_plot)

fig_training_data = Figure(size=(700, 450))

ax_training_input = Axis(
    fig_training_data[1, 1],
    xlabel="Time (s)",
    ylabel="Input",
    title="MATLAB training input",
)

ax_training_output = Axis(
    fig_training_data[2, 1],
    xlabel="Time (s)",
    ylabel="Measured output",
    title="MATLAB training output",
)

lines!(ax_training_input, time_train, u_train_plot)
lines!(ax_training_output, time_train, y_train_plot)

display(fig_training_data)

println("Exp1 REN MATLAB training-data plot passed.")
# error("Stop after milestone 3")
#---------------- Exp1 REN MATLAB training data end ----------------#

#---------------- Exp1 REN tiny MATLAB debug data start ----------------#

debug_time_indices = 1:3
# debug_time_indices: length 3 ordered time-index range

debug_batch_indices = 1:2
# debug_batch_indices: length 2 training-trajectory range

u_debug_plant = [
    u_t[:, debug_batch_indices]
    for u_t in u_train[debug_time_indices]
]
# u_debug_plant: length 3; each item (1, 2)

y_debug = [
    y_t[:, debug_batch_indices]
    for y_t in y_train[debug_time_indices]
]
# y_debug: length 3; each item (1, 2)

batches_plant_debug = length(debug_batch_indices)
# batches_plant_debug: scalar = 2

@test length(u_debug_plant) == 3
@test length(y_debug) == 3
@test size(u_debug_plant[1]) == (1, batches_plant_debug)
@test size(y_debug[1]) == (1, batches_plant_debug)
@test all(isfinite, reduce(vcat, u_debug_plant))
@test all(isfinite, reduce(vcat, y_debug))

println("Exp1 REN tiny MATLAB debug-data tests passed.")
# error("Stop after milestone 4")

#---------------- Exp1 REN tiny MATLAB debug data end ----------------#

# batches = 200
# u  = fill(zeros(T, 1, batches), length(ts)-1)
# X  = fill(zeros(T, 1, batches), length(ts))
# X[1] = (2*rand(rng, T, nx_plant , batches) .- 1) / 2

# for t in ts[1:end-1]
#     X[t+1] = fd(X[t],u[t])
# end

# Xt = X[1:end-1]
# Xn = X[2:end]
# y = gd.(Xt)

#  Store data for training
# observer_data = [[ut; yt] for (ut,yt) in zip(u, y)]
# indx = shuffle(rng, 1:length(observer_data))
# data = zip(Xn[indx] |> dev, Xt[indx] |> dev, observer_data[indx]|> dev)


#####################################################################
# Train a model
#---------------- Milestone 2: tiny input-only REN start ----------------#

# Define a REN model for the observer
# nv = 20
# nu = size(observer_data[1], 1)
# ny = nx_plant 
# model_ps = ContractingRENParams{Float32}(nu, nx_plant , nv, ny; output_map=false, rng)
# model = DiffREN(model_ps) |> dev

#---------------- Milestone Passivity : start ----------------#
nz = 5  # scalar: latent REN state dimension
nv = 20  # scalar: number of nonlinear REN neurons
nu_model = 1               # scalar: one external force input row u_t
ny_model = 1               # scalar: one predicted output row yhat_t
# model_ps = ContractingRENParams{T}(nu_model, 
#                                          nz, 
#                                          nv,
#                                          ny_model;
#                                          output_map=true,
#                                         rng) # model_ps: direct REN para  containing the trainable parameters
ν_REN = T(1e-6)
ρ_REN = T(0.0)
model_ps = PassiveRENParams{T}(nu_model, nz, nv, ny_model, ν_REN, ρ_REN;rng)
model = REN(model_ps) # model: explicit callable REN used for evaluation
#---------------- Milestone Passivity : end ----------------#

batches_debug = 2 # scalar: two parallel trajectories
z0_debug = init_states(model, batches_debug) # (nz, batches) of zero matrix. state at t = 0 
u0_debug = zeros(T, nu_model, batches_debug) # (nu, batches), input at t=0
z1_debug, model_output0_debug = model(z0_debug, u0_debug) # z1_debug (nz, batches), state at t=1
                                                        # model_output0_debug (ny_model,batches), output at t=0 
@test model.nu == nu_model
@test model.nx == nz     
@test model.nv == nv     
@test model.ny == ny_model     
@test size(z0_debug) == (nz, batches_debug)  
@test size(u0_debug) == (nu_model, batches_debug)          
@test size(z1_debug) == (nz, batches_debug)  
@test size(model_output0_debug) == (ny_model, batches_debug)     
@test all(isfinite, z1_debug)
@test all(isfinite, model_output0_debug)    
println("Milestone 2 passed.") 
# error("Stop after milestone 2")
#---------------- Milestone 2: tiny input-only REN end ----------------#
#---------------- Milestone 3: rollout shape test start ----------------#
u_debug = [
    T[0.1 -0.2],
    T[0.3 -0.2],
    T[0.5 -0.2],
] # time length = 3. For each time step, nu x batches, 
model_output_debug = rollout_io(model, z0_debug, u_debug) # model_output_debug: length 3; each item: (ny_model, batches_debug) = (1, 2)
@test length(model_output_debug) == length(u_debug)
@test size(model_output_debug[1]) == (ny_model, batches_debug)
@test size(model_output_debug[2]) == (ny_model, batches_debug)
@test size(model_output_debug[3]) == (ny_model, batches_debug)
@test all(isfinite, reduce(vcat, model_output_debug))
println("Milestone 3 passed.")
# error("Stop after milestone 3")
#---------------- Milestone 3: rollout shape test end ----------------#
#---------------- Milestone 4: analytical rollout test start ----------------#
z1_expected, model_output1_expected = model(z0_debug, u_debug[1])# z1_expected:            (nz, batches_debug) = (2, 2)
                                                                # model_output1_expected: (ny_model, batches_debug) = (1, 2)
z2_expected, model_output2_expected = model(
    z1_expected,                               # (nz, batches_debug) = (2, 2)
    u_debug[2],                                # (nu_model, batches_debug) = (1, 2)
)  
z3_expected, model_output3_expected = model(
    z2_expected,                               # (nz, batches_debug) = (2, 2)
    u_debug[3],                                # (nu_model, batches_debug) = (1, 2)
)  
@test model_output_debug[1] ≈ model_output1_expected atol=1e-7 rtol=1e-7
@test model_output_debug[2] ≈ model_output2_expected atol=1e-7 rtol=1e-7
@test model_output_debug[3] ≈ model_output3_expected atol=1e-7 rtol=1e-7  

recurrent_debug = Flux.Recur(model,z0_debug) # recurrent_debug: stateful evaluator whose stored state begins at z0_debug
recur_output_debug = [
    recurrent_debug(u_t)
    for u_t in u_debug
] # recur_output_debug: length 3; each item: (ny_model, batches_debug) = (1, 2)
@test length(recur_output_debug) == length(model_output_debug)
@test recur_output_debug[1] ≈ model_output_debug[1] atol=1e-7 rtol=1e-7
@test recur_output_debug[2] ≈ model_output_debug[2] atol=1e-7 rtol=1e-7
@test recur_output_debug[3] ≈ model_output_debug[3] atol=1e-7 rtol=1e-7
@test recur_output_debug[1] ≈ model_output1_expected atol=1e-7 rtol=1e-7
@test recur_output_debug[2] ≈ model_output2_expected atol=1e-7 rtol=1e-7
@test recur_output_debug[3] ≈ model_output3_expected atol=1e-7 rtol=1e-7
println("Milestone 4 passed.")
# error("Stop after milestone 4")
#---------------- Milestone 4: analytical rollout test end ----------------#
#---------------- Milestone 6: loss tests start ----------------#
loss_debug = loss_io(
    model_ps,
    u_debug_plant,                             # length 3; each item: (1, 2)
    y_debug,                                   # length 3; each item: (1, 2)
)# loss_debug: scalar
@test loss_debug isa Number
@test isfinite(loss_debug)

z0_loss_debug = init_states(model, batches_plant_debug) # (nz, batches_plant_debug)
manual_model_output_debug = rollout_io(
    model,
    z0_loss_debug,
    u_debug_plant,
)# manual_model_output_debug: length 3; each item: (ny_model, batches_plant_debug)
manual_loss_debug = (
    sum(abs2, y_debug[1] - manual_model_output_debug[1]) +
    sum(abs2, y_debug[2] - manual_model_output_debug[2]) +
    sum(abs2, y_debug[3] - manual_model_output_debug[3])
)# manual_loss_debug: scalar
@test loss_debug ≈ manual_loss_debug atol=1e-7 rtol=1e-7
println("loss_debug", loss_debug)
println("manual_loss_debug.", manual_loss_debug)
loss_before_update, grads_debug = Flux.withgradient(model_ps) do ps
    loss_io(
        ps,
        u_debug_plant,
        y_debug,
    )
end# loss_before_update: scalar
   # grads_debug[1]: gradient tree with the same trainable structure as model_ps

@test loss_before_update isa Number
@test isfinite(loss_before_update)
@test grads_debug[1] !== nothing
gradient_arrays_debug = Flux.trainables(grads_debug[1])
# gradient_arrays_debug: Vector of gradient arrays
@test !isempty(gradient_arrays_debug)
@test all(all(isfinite, gradient_array) for gradient_array in gradient_arrays_debug)

opt_state_debug = Flux.setup(Adam(1e-3), model_ps) # opt_state_debug: Adam optimizer state matching model_ps

Flux.update!(opt_state_debug, model_ps, grads_debug[1])
loss_after_update = loss_io(
    model_ps,
    u_debug_plant,
    y_debug,
) # loss_after_update: scalar
@test isfinite(loss_after_update)
println("Loss before update.", loss_before_update)
println("Loss after update.", loss_after_update)
println("Milestone 6 passed.")
# error("Stop after milestone 6")
#---------------- Milestone 6: loss tests end ----------------#
# Loss function: one step ahead error (average over time)
function loss(model, xn, xt, inputs)
    xpred = model(xt, inputs)[1]
    return mean(sum((xn - xpred).^2, dims=1))
end

# Train the model
function train_observer!(model, data; epochs=5, lr=1e-3, min_lr=1e-6)

    opt_state = Flux.setup(Adam(lr), model)
    mean_loss = [T(1e5)]
    for epoch in 1:epochs

        batch_loss = []
        for (xn, xt, inputs) in data
            train_loss, ∇J = Flux.withgradient(loss, model, xn, xt, inputs)
            Flux.update!(opt_state, model, ∇J[1])
            push!(batch_loss, train_loss)
        end
        @printf "Epoch: %d, Lr: %.1g, Loss: %.4g\n" epoch lr mean(batch_loss)

        # Drop learning rate if mean loss is stuck or growing
        push!(mean_loss, mean(batch_loss))
        if (mean_loss[end] >= mean_loss[end-1]) && !(lr < min_lr || lr ≈ min_lr)
            lr = 0.1lr
            Flux.adjust!(opt_state, lr)
        end
    end
    return mean_loss
end
#---------------- Milestone 8: input-output training function start ----------------#
# tloss = train_observer!(model, data)
function train_io!(model_ps, data; epochs=1, lr=1e-3, min_lr=1e-6)
    # model_ps: direct REN para containing trainable parameters
    # data: Vector of tuples (u_seq, y_seq)
    # u_seq[t]: (nu_model, batches)
    # y_seq[t]: (ny_model, batches)
    opt_state = Flux.setup(Adam(lr), model_ps) # opt_state: Adam optimizer state matching model_ps

    mean_loss = [T(Inf)] # mean_loss: Vector of scalar losses. The initial Inf allows comparison after the first epoch.
    for epoch in 1:epochs

        batch_loss = Float64[] # batch_loss: Vector of scalar losses, one scalar per tuple in data
        for (u_seq, y_seq) in data
            train_loss, grads = Flux.withgradient(model_ps) do ps
                    loss_io(
                        ps,
                        u_seq,
                        y_seq)
            end # train_loss: scalar  # grads[1]: gradient tree matching model_ps
            Flux.update!(opt_state, model_ps, grads[1])
            push!(batch_loss, train_loss)
        end
        epoch_loss = mean(batch_loss) 
        @printf "Epoch: %d, Lr: %.1g, Loss: %.4g\n" epoch lr epoch_loss
        push!(mean_loss, epoch_loss)

        if (mean_loss[end] >= mean_loss[end - 1]) &&
            !(lr < min_lr || lr ≈ min_lr)
                lr = 0.1lr
                Flux.adjust!(opt_state, lr)
        end
    end
    return mean_loss
end
#---------------- Milestone 8: input-output training function end ----------------#
#---------------- Exp1 REN tiny real-data training test start ----------------#

tiny_time_indices = 1:3
# tiny_time_indices: length 3 ordered time-index range

tiny_batch_indices = 1:2
# tiny_batch_indices: length 2 training-trajectory range

u_train_tiny = [
    u_t[:, tiny_batch_indices]
    for u_t in u_train[tiny_time_indices]
]
# u_train_tiny: length 3; each item (1, 2)

y_train_tiny = [
    y_t[:, tiny_batch_indices]
    for y_t in y_train[tiny_time_indices]
]
# y_train_tiny: length 3; each item (1, 2)

data_io_tiny = [(
    u_train_tiny |> dev,
    y_train_tiny |> dev,
)]
# data_io_tiny: length 1
# data_io_tiny[1]: tuple containing one ordered three-step rollout

tiny_loss_before_training = loss_io(
    model_ps,
    u_train_tiny |> dev,
    y_train_tiny |> dev,
)
# tiny_loss_before_training: scalar unscaled simulation error

@test isfinite(tiny_loss_before_training)

tloss_tiny = train_io!(
    model_ps,
    data_io_tiny;
    epochs=1,
    lr=1e-3,
)
# tloss_tiny: length 2 = [Inf, first_epoch_loss]

tiny_loss_after_training = loss_io(
    model_ps,
    u_train_tiny |> dev,
    y_train_tiny |> dev,
)
# tiny_loss_after_training: scalar unscaled simulation error

@test length(tloss_tiny) == 2
@test isfinite(tloss_tiny[end])
@test isfinite(tiny_loss_after_training)

println("tiny loss before training = ", tiny_loss_before_training)
println("tiny loss after training = ", tiny_loss_after_training)
println("Exp1 REN tiny measured-data training test passed.")
# error("Stop after milestone 5")

#---------------- Exp1 REN tiny real-data training test end ----------------#

#---------------- Milestone 9: configurable ordered training subset start ----------------#
training_horizon = 250                           # scalar: number of ordered time indices
training_batches = 300                           # scalar: number of trajectories
selected_time_indices = 1:training_horizon
selected_batch_indices = 1:training_batches
u_train_selected = [
    u_t[:, selected_batch_indices]
    for u_t in u_train[selected_time_indices]
]
# u_train_selected: length training_horizon; each item: (1, training_batches)

y_train_selected = [
    y_t[:, selected_batch_indices]
    for y_t in y_train[selected_time_indices]
]
# y_train_selected: length training_horizon; each item: (1, training_batches)

data_io_selected = [(
    u_train_selected |> dev,
    y_train_selected |> dev,
)]
# data_io_selected: length 1

selected_loss_before_training = loss_io(
    model_ps,
    u_train_selected |> dev,
    y_train_selected |> dev,
)
# selected_loss_before_training: scalar

@test length(u_train_selected) == training_horizon
@test length(y_train_selected) == training_horizon
@test size(u_train_selected[1]) == (1, training_batches)
@test size(y_train_selected[1]) == (1, training_batches)
@test isfinite(selected_loss_before_training)

@printf "Selected horizon: %d, batches: %d, initial loss: %.4g\n" training_horizon training_batches selected_loss_before_training

tloss = train_io!(
    model_ps,
    data_io_selected;
    epochs=2800,
    lr=1e-3,
)

# data_io_full = [(
#     u_train |> dev,                            # length 999; each item: (1, 200)
#     y_train |> dev,                            # length 999; each item: (1, 200)
# )]

# tloss = train_io!(
#     model_ps,
#     data_io_full;
#     epochs=10,
#     lr=1e-4,
# )
# tloss: length 2 for one training epoch

@test isfinite(tloss[end])

println("Milestone 9 selected-scale test passed.")
println("Exp1 REN selected MATLAB subset training passed.")
# error("Stop after selected Milestone 9 scale")

#---------------- Milestone 9: configurable ordered training subset end ----------------#
#---------------- Exp1 REN save trained model start ----------------#
mkpath(RESULTS_DIR)

artifact_path = joinpath(RESULTS_DIR, "run_Exp1_REN_model.bson")
# artifact_path: path to saved BSON artifact

bson(
    artifact_path,
    Dict(
        "model_ps" => (model_ps |> cpu),
        "training_loss" => tloss,
        "selected_time_indices" => collect(selected_time_indices),
        "selected_batch_indices" => collect(selected_batch_indices),
        "source_mat_path" => loaded_data.source_mat_path,
        "sample_time" => loaded_data.sample_time,
        "ν_REN" => ν_REN,
        "ρ_REN" => ρ_REN,
    ),
)

@test isfile(artifact_path)

loaded_artifact = BSON.load(artifact_path)
# loaded_artifact: Dict containing saved parameters and metadata

loaded_model_ps = loaded_artifact["model_ps"]
# loaded_model_ps: reloaded direct passive-REN parameters

loaded_model = REN(loaded_model_ps)
# loaded_model: explicit callable passive REN reconstructed from disk

reload_batches = 2
# reload_batches: scalar test-trajectory count

reload_z0 = init_states(loaded_model, reload_batches)
# reload_z0: (nz, reload_batches) = (5, 2)

reload_u0 = zeros(T, nu_model, reload_batches)
# reload_u0: (nu_model, reload_batches) = (1, 2)

reload_z1, reload_y0 = loaded_model(reload_z0, reload_u0)
# reload_z1: (nz, reload_batches) = (5, 2)
# reload_y0: (ny_model, reload_batches) = (1, 2)

@test size(reload_z1) == (nz, reload_batches)
@test size(reload_y0) == (ny_model, reload_batches)
@test all(isfinite, reload_z1)
@test all(isfinite, reload_y0)
@test loaded_artifact["selected_time_indices"] == collect(selected_time_indices)
@test loaded_artifact["selected_batch_indices"] == collect(selected_batch_indices)

println("Saved trained passive REN to: ", artifact_path)
println("Exp1 REN BSON save/reload smoke test passed.")
# error("Stop after milestone 7")

#---------------- Exp1 REN save trained model end ----------------#
#---------------- Milestone 10: post-training fit evaluation start ----------------#
model_train_eval = REN(model_ps) # model_train_eval: explicit trained REN constructed from final direct parameters
z0_train_eval = init_states(model_train_eval, training_batches)  # z0_train_eval: (nz, batches_train) = (5, 200), zero latent initial states
model_output_train = rollout_io(
    model_train_eval,
    z0_train_eval,
    u_train_selected,
) # model_output_train: length 999; each item: (ny_model, batches_train) = (1, 200)

train_eval_loss_at_t = Float64[] # train_eval_loss_at_t: Vector of scalar squared errors, one scalar per time index
for t in eachindex(y_train_selected)
    error_t = y_train_selected[t] - model_output_train[t]  # error_t: (ny_model, batches_train) = (1, 200)

    push!(train_eval_loss_at_t, sum(abs2, error_t))
end
train_eval_loss = sum(train_eval_loss_at_t) # train_eval_loss: scalar, fresh post-training unscaled simulation error
@test length(model_output_train) == length(y_train_selected)
@test size(model_output_train[1]) == (ny_model, training_batches)
@test length(train_eval_loss_at_t) == length(y_train_selected)
@test all(isfinite, reduce(vcat, model_output_train))
@test isfinite(train_eval_loss)
@test model_output_train[1] != model_output_train[end]
println("post-training fit loss = ", train_eval_loss)
println("Milestone 10 shapes and finite values passed.")

batch_to_plot_train_fit = 1                   # scalar: selected training trajectory

model_output_train_plot = [
    model_output_train[t][1, batch_to_plot_train_fit]
    for t in eachindex(model_output_train)
] # model_output_train_plot: (length(model_output_train),) = (999,)

time_train = [
    (time_index - 1) * dt
    for time_index in eachindex(u_train_selected)
]# time_train: (length(u_train),) = (999,), seconds
u_train_plot = [
    u_train_selected[time_index][1, batch_to_plot_train_fit] # u_train_selected [time_index] is a nu=1 x batches matrix, then [1, batch_to_plot_train] is accessing the certain batch
    for time_index in eachindex(u_train_selected)
]# u_train_plot: (length(u_train_selected ),) = (999,), force (N)
y_train_plot = [
    y_train_selected[time_index][1, batch_to_plot_train_fit]
    for time_index in eachindex(y_train_selected)
]# y_train_plot: (length(y_train),) = (999,), position (m)


train_fit_error_plot = y_train_plot - model_output_train_plot # train_fit_error_plot: (length(y_train),) = (999,)
@test length(time_train) == length(u_train_plot)
@test length(time_train) == length(y_train_plot)
@test length(time_train) == length(model_output_train_plot)
@test length(time_train) == length(train_fit_error_plot)

fig_training_fit = Figure(resolution=(750, 850))

ax_train_fit_force = Axis(
    fig_training_fit[1, 1],
    xlabel="Time (s)",
    ylabel="Force (N)",
    title="Training input",
)

ax_train_fit_output = Axis(
    fig_training_fit[2, 1],
    xlabel="Time (s)",
    ylabel="Position (m)",
    title="Post-training fit on one training trajectory",
)

ax_train_fit_error = Axis(
    fig_training_fit[3, 1],
    xlabel="Time (s)",
    ylabel="Output error (m)",
    title="True training output - REN output error",
)

ax_train_fit_loss = Axis(
    fig_training_fit[4, 1],
    xlabel="Time (s)",
    ylabel="Squared error",
    title="Training squared error summed over trajectories",
)
lines!(ax_train_fit_force, time_train, u_train_plot)
lines!(ax_train_fit_output, time_train, y_train_plot, label="True training y_t")
lines!(ax_train_fit_output, time_train, model_output_train_plot, label="REN yhat_t")
lines!(ax_train_fit_error, time_train, train_fit_error_plot)
lines!(ax_train_fit_loss, time_train, train_eval_loss_at_t)

axislegend(ax_train_fit_output)
display(fig_training_fit)

println("Milestone 10 passed.")
# error("Stop after milestone 10")
#---------------- Milestone 10: post-training fit evaluation end ----------------#
