using MAT
using Test

function load_exp1_ren_mat_data(
    mat_path::AbstractString;
    T=Float64,
)
    # mat_path: path to one MATLAB .mat file
    # T:        numeric element type used by the REN

    mat_data = matread(mat_path)
    # mat_data: Dict containing the top-level MATLAB variables

    @test haskey(mat_data, "dta")

    dta = mat_data["dta"]
    # dta: Dict containing the MATLAB struct fields

    n_time_train = Int(dta["N_train"])
    # n_time_train: scalar training time-sample count = 250

    n_batch_train = Int(dta["n_train"])
    # n_batch_train: scalar training-trajectory count = 500

    n_time_valid = Int(dta["N_valid"])
    # n_time_valid: scalar validation time-sample count = 250

    n_batch_valid = Int(dta["n_valid"])
    # n_batch_valid: scalar validation-trajectory count = 2

    sample_time = T(dta["Ts"])
    # sample_time: scalar sample time = 0.02 seconds

    u_train_tb = T.(
        dta["ipt_train_mat"][1, 1:n_time_train, 1:n_batch_train],
    )
    # u_train_tb: (n_time_train, n_batch_train) = (250, 500)

    y_train_tb = T.(
        dta["opt_train_mat"][1, 1:n_time_train, 1:n_batch_train],
    )
    # y_train_tb: (n_time_train, n_batch_train) = (250, 500)

    u_valid_tb = T.(
        dta["ipt_valid_mat"][1, 1:n_time_valid, 1:n_batch_valid],
    )
    # u_valid_tb: (n_time_valid, n_batch_valid) = (250, 2)

    y_valid_tb = T.(
        dta["opt_valid_mat"][1, 1:n_time_valid, 1:n_batch_valid],
    )
    # y_valid_tb: (n_time_valid, n_batch_valid) = (250, 2)

    u_train = [
        reshape(u_train_tb[t, :], 1, n_batch_train)
        for t in 1:n_time_train
    ]
    # u_train: length n_time_train = 250
    # u_train[t]: (1, n_batch_train) = (1, 500)

    y_train = [
        reshape(y_train_tb[t, :], 1, n_batch_train)
        for t in 1:n_time_train
    ]
    # y_train: length n_time_train = 250
    # y_train[t]: (1, n_batch_train) = (1, 500)

    u_valid = [
        reshape(u_valid_tb[t, :], 1, n_batch_valid)
        for t in 1:n_time_valid
    ]
    # u_valid: length n_time_valid = 250
    # u_valid[t]: (1, n_batch_valid) = (1, 2)

    y_valid = [
        reshape(y_valid_tb[t, :], 1, n_batch_valid)
        for t in 1:n_time_valid
    ]
    # y_valid: length n_time_valid = 250
    # y_valid[t]: (1, n_batch_valid) = (1, 2)

    time_train = T.(collect(0:n_time_train - 1)) .* sample_time
    # time_train: (n_time_train,) = (250,), seconds

    time_valid = T.(collect(0:n_time_valid - 1)) .* sample_time
    # time_valid: (n_time_valid,) = (250,), seconds

    return (
        u_train_tb=u_train_tb,
        y_train_tb=y_train_tb,
        u_valid_tb=u_valid_tb,
        y_valid_tb=y_valid_tb,
        u_train=u_train,
        y_train=y_train,
        u_valid=u_valid,
        y_valid=y_valid,
        n_time_train=n_time_train,
        n_batch_train=n_batch_train,
        n_time_valid=n_time_valid,
        n_batch_valid=n_batch_valid,
        sample_time=sample_time,
        time_train=time_train,
        time_valid=time_valid,
        source_mat_path=abspath(mat_path),
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    ren_dir = @__DIR__
    # ren_dir: scalar string path to the folder containing this REN script
    default_exp1_dir = normpath(joinpath(ren_dir, "..", ".."))
    # default_exp1_dir: scalar string path to Exp1 when this script is run in this repository
    exp1_dir = get(ENV, "NFIR_EXP1_DIR", default_exp1_dir)
    # exp1_dir: scalar string path to the open-source Exp1 folder
    data_dir = joinpath(exp1_dir, "Training_data")
    # data_dir: scalar string path to the Exp1 MATLAB training data folder

    mat_path = joinpath(
        data_dir,
        "Data_M_NLdamper_500B_OneCart.mat",
    )
    # mat_path: absolute or normalized path to MATLAB data file

    loaded_data = load_exp1_ren_mat_data(mat_path)
    # loaded_data: named tuple containing raw matrices and REN sequences

    @test size(loaded_data.u_train_tb) == (250, 500)
    @test size(loaded_data.y_train_tb) == (250, 500)
    @test size(loaded_data.u_valid_tb) == (250, 2)
    @test size(loaded_data.y_valid_tb) == (250, 2)

    @test length(loaded_data.u_train) == 250
    @test length(loaded_data.y_train) == 250
    @test length(loaded_data.u_valid) == 250
    @test length(loaded_data.y_valid) == 250

    @test size(loaded_data.u_train[1]) == (1, 500)
    @test size(loaded_data.y_train[1]) == (1, 500)
    @test size(loaded_data.u_valid[1]) == (1, 2)
    @test size(loaded_data.y_valid[1]) == (1, 2)

    @test loaded_data.sample_time == 0.02
    @test loaded_data.time_train[1] == 0.0
    @test loaded_data.time_train[end] == 4.98

    @test all(isfinite, loaded_data.u_train_tb)
    @test all(isfinite, loaded_data.y_train_tb)
    @test all(isfinite, loaded_data.u_valid_tb)
    @test all(isfinite, loaded_data.y_valid_tb)

    println("MATLAB loader smoke test passed.")
    println("training u shape = ", size(loaded_data.u_train_tb))
    println("training y shape = ", size(loaded_data.y_train_tb))
    println("validation u shape = ", size(loaded_data.u_valid_tb))
    println("validation y shape = ", size(loaded_data.y_valid_tb))
    println("sample time = ", loaded_data.sample_time)
end
