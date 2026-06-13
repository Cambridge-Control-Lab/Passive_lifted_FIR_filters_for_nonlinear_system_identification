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
mkpath(RESULTS_DIR)

using BSON
using CairoMakie
using Flux
using MAT
using RobustNeuralNetworks
using Test
using Zygote: Buffer

include("REN_load_data.jl")

T = Float64
# T: numeric element type used during evaluation

noise_on = "--snr10" in ARGS
# noise_on: scalar Bool; true evaluates the SNR10 REN result
if noise_on == true
    artifact_path = joinpath(
        RESULTS_DIR,
        "run_Exp1_REN_SNR10_model.bson",
    )# artifact_path: path to saved BSON model artifact
    mat_path = joinpath(
        DATA_DIR,
        "Data_M_NLdamper_500B_OneCart_SNR10.mat" 
    ) # mat_path: path to MATLAB data file
else
    artifact_path = joinpath(
        RESULTS_DIR,
        "run_Exp1_REN_model.bson",
    )# artifact_path: path to saved BSON model artifact

    mat_path = joinpath(
        DATA_DIR,
        "Data_M_NLdamper_500B_OneCart.mat" 
    ) # mat_path: path to MATLAB data file
end


@test isfile(mat_path)
@test isfile(artifact_path)

loaded_data = load_exp1_ren_mat_data(mat_path; T) # Use function in REN_load_data.jl
loaded_artifact = BSON.load(artifact_path) # loaded_artifact: Dict containing saved model parameters and metadata

model_ps = loaded_artifact["model_ps"] # model_ps: loaded direct passive-REN parameters
model = REN(model_ps) # model: explicit callable passive REN


# Below mirrors the rollout helper used by the REN training scripts.
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

function simulation_error_summary(y_seq,
    model_output_seq)
    # y_seq[t]:            (ny_model, batches)
    # model_output_seq[t]: (ny_model, batches)

    @test length(y_seq) == length(model_output_seq)

    loss_at_t = Float64[]

    # model = REN(model_ps) 
    # batches = size(u_seq[1],2) #scalar , batches
    # z0 = init_states(model, batches)            # (nz, batches)
    # model_output_seq = rollout_io(model,z0,u_seq) # model_output_seq: length(u_seq); each item: (ny_model, batches)


    total_loss = 0.0                            # scalar: unscaled squared 2-norm error
    for t in eachindex(y_seq)
        error_t = y_seq[t] - model_output_seq[t] # error_t: (ny_model, batches)
        total_loss += sum(abs2, error_t) # scalar
        push!(loss_at_t, sum(abs2, error_t))
    end

    n_time = length(y_seq)
    n_batch = size(y_seq[1],2)

    return (
        loss_at_t = loss_at_t,
        total_loss = total_loss,
        loss_per_time = total_loss / n_time,
        loss_per_batch = total_loss / n_batch,
        loss_per_time_per_batch = total_loss / n_time / n_batch
    )
end

function plot_io_batches(
    time_values,
    u_seq,
    y_seq,
    model_output_seq,
    batch_indices;
    figure_title,
    save_path,
)
    # time_values:         (n_time,), seconds
    # u_seq[t]:            (nu_model, batches)
    # y_seq[t]:            (ny_model, batches)
    # model_output_seq[t]: (ny_model, batches)
    # batch_indices:       selected trajectory-column indices

    @test length(time_values) == length(u_seq)
    @test length(time_values) == length(y_seq)
    @test length(time_values) == length(model_output_seq)

    number_of_plots = length(batch_indices)
    # number_of_plots: scalar trajectory-plot count

    fig = Figure(size=(1400, 270 * number_of_plots))

    for (plot_row, batch_index) in enumerate(batch_indices)
        # plot_row:    scalar figure-row index
        # batch_index: scalar trajectory-column index

        u_plot = [
            u_seq[t][1, batch_index]
            for t in eachindex(u_seq)
        ]
        # u_plot: (n_time,), input signal

        y_plot = [
            y_seq[t][1, batch_index]
            for t in eachindex(y_seq)
        ]
        # y_plot: (n_time,), measured output signal

        model_output_plot = [
            model_output_seq[t][1, batch_index]
            for t in eachindex(model_output_seq)
        ]
        # model_output_plot: (n_time,), simulated REN output signal

        error_plot = y_plot - model_output_plot
        # error_plot: (n_time,), measured output minus REN output

        squared_error_plot = abs2.(error_plot)
        # squared_error_plot: (n_time,), squared output error

        @test all(isfinite, u_plot)
        @test all(isfinite, y_plot)
        @test all(isfinite, model_output_plot)
        @test all(isfinite, error_plot)
        @test all(isfinite, squared_error_plot)

        ax_input = Axis(
            fig[plot_row, 1],
            xlabel="Time (s)",
            ylabel="Input",
            title="$(figure_title): batch $(batch_index), input",
        )

        ax_output = Axis(
            fig[plot_row, 2],
            xlabel="Time (s)",
            ylabel="Output",
            title="Measured output and REN output",
        )

        ax_error = Axis(
            fig[plot_row, 3],
            xlabel="Time (s)",
            ylabel="Output error",
            title="Measured output - REN output",
        )

        # ax_squared_error = Axis(
        #     fig[plot_row, 4],
        #     xlabel="Time (s)",
        #     ylabel="Squared error",
        #     title="Per-time-step squared error",
        # )

        lines!(ax_input, time_values, u_plot)
        lines!(ax_output, time_values, y_plot, label="Measured y_t")
        lines!(
            ax_output,
            time_values,
            model_output_plot,
            label="REN yhat_t",
        )
        lines!(ax_error, time_values, error_plot)
        # lines!(ax_squared_error, time_values, squared_error_plot)

        axislegend(ax_output)
    end

    mkpath(dirname(save_path))
    display(fig)
    save(save_path, fig)

    return fig
end

@test loaded_artifact["source_mat_path"] == loaded_data.source_mat_path
@test loaded_artifact["sample_time"] == loaded_data.sample_time

println("Exp1 REN evaluation setup passed.")
println("loaded model type = ", typeof(model_ps))
println("sample time = ", loaded_data.sample_time)
println("available training trajectories = ", loaded_data.n_batch_train)
println("available validation trajectories = ", loaded_data.n_batch_valid)
# error("Stop after milestone 8")

selected_time_indices = loaded_artifact["selected_time_indices"]
# selected_time_indices: length 250 vector of time indices seen by Adam

selected_batch_indices = loaded_artifact["selected_batch_indices"]
# selected_batch_indices: length 20 vector of trajectory indices seen by Adam

u_train_seen = [
    u_t[:, selected_batch_indices]
    for u_t in loaded_data.u_train[selected_time_indices]
]
# u_train_seen: length 250; each item (1, 20)

y_train_seen = [
    y_t[:, selected_batch_indices]
    for y_t in loaded_data.y_train[selected_time_indices]
]
# y_train_seen: length 250; each item (1, 20)

training_batches = length(selected_batch_indices)
# training_batches: scalar = 20

z0_train = init_states(model, training_batches)
# z0_train: (nz, training_batches) = (5, 20)

model_output_train = rollout_io(
    model,
    z0_train,
    u_train_seen
) # model_output_train: length 250; each item (1, 20)

training_summary = simulation_error_summary(
    y_train_seen,
    model_output_train,
) # training_summary: named tuple containing training-fit loss diagnostics

@test length(model_output_train) == length(y_train_seen)
@test size(model_output_train[1]) == (1, training_batches)
@test all(isfinite, reduce(vcat, model_output_train))
@test model_output_train[1] != model_output_train[end]
@test isfinite(training_summary.total_loss)
@test all(isfinite, training_summary.loss_at_t)

println("Training-fit total loss = ", training_summary.total_loss)
println("Training-fit loss per time step = ", training_summary.loss_per_time)
println("Training-fit loss per trajectory = ", training_summary.loss_per_batch)
println(
    "Training-fit loss per time step per trajectory = ",
    training_summary.loss_per_time_per_batch,
)

training_plot_batches = [1, 2, 3]
# training_plot_batches: length 3 local columns from the selected training subset

training_time = loaded_data.time_train[selected_time_indices]
# training_time: (250,), seconds

training_plot_path = joinpath(
    RESULTS_DIR,
    noise_on ? "run_Exp1_REN_SNR10_training_fit.svg" : "run_Exp1_REN_training_fit.svg",
)
# training_plot_path: path to training-fit SVG figure

fig_training_fit = plot_io_batches(
    training_time,
    u_train_seen,
    y_train_seen,
    model_output_train,
    training_plot_batches;
    figure_title="Seen training trajectory",
    save_path=training_plot_path
)# fig_training_fit: CairoMakie figure containing three trajectory rows

println("Saved training-fit plot to: ", training_plot_path)
println("Exp1 REN seen training-data evaluation passed.")


training_loss_history = loaded_artifact["training_loss"]
# training_loss_history: length epochs + 1; first item is Inf from initialization
@test length(training_loss_history) >= 2
@test !isfinite(training_loss_history[1])

cost_value = Float64.(training_loss_history[2:end]) # cost_value: (n_epochs,), unscaled simulation-error cost at each epoch
cost_epoch = collect(1:length(cost_value)) # cost_epoch: (n_epochs,), epoch numbers starting at 1

@test length(cost_epoch) == length(cost_value)
@test all(isfinite, cost_value)
@test all(cost_value .>= 0)
if noise_on == true
    cost_plot_path = joinpath(
        RESULTS_DIR,
        "run_Exp1_REN_SNR10_training_cost_vs_epoch.svg",
    )
    fig_training_cost = Figure(size=(750, 450)) # fig_training_cost: CairoMakie figure for optimization cost history
else
    cost_plot_path = joinpath(
        RESULTS_DIR,
        "run_Exp1_REN_training_cost_vs_epoch.svg",
    )
    fig_training_cost = Figure(size=(750, 450)) # fig_training_cost: CairoMakie figure for optimization cost history
end

ax_training_cost = Axis(
    fig_training_cost[1, 1],
    xlabel="Epoch",
    ylabel="Unscaled simulation-error cost",
    title="Training cost vs epoch",
) # ax_training_cost: one axis showing cost_value against cost_epoch
lines!(ax_training_cost, cost_epoch, cost_value)
scatter!(ax_training_cost, cost_epoch, cost_value; markersize=5)
ylims!(ax_training_cost, 0.0, 0.5e5)
mkpath(dirname(cost_plot_path))
display(fig_training_cost)
save(cost_plot_path, fig_training_cost)

println("Saved training-cost plot to: ", cost_plot_path)

# error("Stop after milestone 9")


# --------- Simulate all -------------------#
println("loaded_data.n_batch_train ",loaded_data.n_batch_train)
println("Simulate all ")
z0_all = init_states(model, loaded_data.n_batch_train) # z0_train: (nz, training_batches) = (nz, 500)
println("loaded_data.n_batch_train ",loaded_data.n_batch_train)
println(" size(zinit,2)  ", size(z0_all,2) )
println(" length(loaded_data.u_train) ", length(loaded_data.u_train) )
# error("Stop")
model_output_all = rollout_io(
    model,
    z0_all,
    loaded_data.u_train
) # model_output_train: length 250; each item (1, 500)
println("Simulate all end ")
@test length(model_output_all) == 250
@test size(model_output_all[1]) == (1, 500)
#= Need 
     y_test_batch   
     y_pre_test_batch
     y_pre_test_batch_cl

    y_pre_train_batch
    y_pre_train_batch_cl
    y_train_batch
=#
batch_idx_train = 1:300
batch_idx_test = 401:500
y_train_batch = reduce(vcat, [
    loaded_data.y_train[t][:, batch_idx_train]
    for t in eachindex(loaded_data.y_train)
])
# y_train_batch: (250, 300)

y_pre_train_batch = reduce(vcat, [
    model_output_all[t][:, batch_idx_train]
    for t in eachindex(model_output_all)
])
# y_pre_train_batch: (250, 300)

y_test_batch = reduce(vcat, [
    loaded_data.y_train[t][:, batch_idx_test]
    for t in eachindex(loaded_data.y_train)
])
# y_test_batch: (250, 100)

y_pre_test_batch = reduce(vcat, [
    model_output_all[t][:, batch_idx_test]
    for t in eachindex(model_output_all)
])
# y_pre_test_batch: (250, 100)
y_pre_test_batch_cl = y_pre_test_batch
y_pre_train_batch_cl = y_pre_train_batch

@test size(y_train_batch) == (250, 300)
@test size(y_pre_train_batch) == (250, 300)
@test size(y_test_batch) == (250, 100)
@test size(y_pre_test_batch) == (250, 100)

# --------- Simulate end -------------------#
# y_batch:       (n_time, n_batches)
# yhat_batch:    (n_time, n_batches)
# batch_indices: selected matrix-column indices
function plot_y_batches(
    time_values,
    y_batch,
    yhat_batch,
    batch_indices;
    figure_title,
    save_path,
)
    # time_values:  (n_time,)
    # y_batch:      (n_time, n_batches)
    # yhat_batch:   (n_time, n_batches)
    # batch_indices: vector of batch-column indices

    @test length(time_values) == size(y_batch, 1)
    @test size(y_batch) == size(yhat_batch)
    @test maximum(batch_indices) <= size(y_batch, 2)

    number_of_plots = length(batch_indices)
    # number_of_plots: scalar

    fig = Figure(size=(1000, 270 * number_of_plots))

    for (plot_row, batch_index) in enumerate(batch_indices)
        # plot_row: scalar figure-row index
        # batch_index: scalar matrix-column index

        y_plot = y_batch[:, batch_index]
        # y_plot: (n_time,)

        yhat_plot = yhat_batch[:, batch_index]
        # yhat_plot: (n_time,)

        error_plot = y_plot - yhat_plot
        # error_plot: (n_time,)

        ax_output = Axis(
            fig[plot_row, 1],
            xlabel="Time (s)",
            ylabel="Output",
            title="$(figure_title): batch $(batch_index)",
        )

        ax_error = Axis(
            fig[plot_row, 2],
            xlabel="Time (s)",
            ylabel="Output error",
            title="Measured y - REN yhat",
        )

        lines!(ax_output, time_values, y_plot, label="Measured y")
        lines!(ax_output, time_values, yhat_plot, label="REN yhat")
        lines!(ax_error, time_values, error_plot)

        axislegend(ax_output)
    end

    mkpath(dirname(save_path))
    display(fig)
    save(save_path, fig)

    return fig
end


training_time = loaded_data.time_train
# training_time: (250,)

training_plot_batches = [1, 2, 3]
# training_plot_batches: columns of y_train_batch and y_pre_train_batch
if noise_on == true
    train_batch_plot_path = joinpath(
        RESULTS_DIR,
        "run_Exp1_REN_SNR10_train_batch_matrix_plot.svg",
    )
    test_batch_plot_path = joinpath(
        RESULTS_DIR,
        "run_Exp1_REN_SNR10_test_batch_matrix_plot.svg",
    )
    save_mat_path = joinpath(
        RESULTS_DIR,
        "run_Exp1_REN_SNR10_train.mat",
    )
    # save_mat_path: path to output .mat file
else
    train_batch_plot_path = joinpath(
        RESULTS_DIR,
        "run_Exp1_REN_train_batch_matrix_plot.svg",
    )
    test_batch_plot_path = joinpath(
        RESULTS_DIR,
        "run_Exp1_REN_test_batch_matrix_plot.svg",
    )
    save_mat_path = joinpath(
        RESULTS_DIR,
        "run_Exp1_REN_train.mat",
    )
    # save_mat_path: path to output .mat file
end

fig_train_batch = plot_y_batches(
    training_time,
    y_train_batch,
    y_pre_train_batch,
    training_plot_batches;
    figure_title="Training matrix trajectory",
    save_path=train_batch_plot_path,
)

test_plot_batches = [1, 2, 3]
# local columns inside y_test_batch and y_pre_test_batch


fig_test_batch = plot_y_batches(
    training_time,
    y_test_batch,
    y_pre_test_batch,
    test_plot_batches;
    figure_title="Test matrix trajectory",
    save_path=test_batch_plot_path,
)

mkpath(dirname(save_mat_path))

@test size(y_test_batch) == size(y_pre_test_batch)
@test size(y_test_batch) == size(y_pre_test_batch_cl)
@test size(y_train_batch) == size(y_pre_train_batch)
@test size(y_train_batch) == size(y_pre_train_batch_cl)

@test all(isfinite, y_test_batch)
@test all(isfinite, y_pre_test_batch)
@test all(isfinite, y_pre_test_batch_cl)
@test all(isfinite, y_train_batch)
@test all(isfinite, y_pre_train_batch)
@test all(isfinite, y_pre_train_batch_cl)

data_to_save = Dict(
    "y_test_batch" => y_test_batch,
    "y_pre_test_batch" => y_pre_test_batch,
    "y_pre_test_batch_cl" => y_pre_test_batch_cl,
    "y_pre_train_batch" => y_pre_train_batch,
    "y_pre_train_batch_cl" => y_pre_train_batch_cl,
    "y_train_batch" => y_train_batch,
)
# data_to_save: MATLAB struct fields; each field is a matrix

if noise_on == true
    matwrite(save_mat_path, Dict(
        "data_REN_noise" => data_to_save,
    ))
else
    matwrite(save_mat_path, Dict(
        "data_REN" => data_to_save,
    ))
end

println("Saved evaluation batch data to: ", save_mat_path)
