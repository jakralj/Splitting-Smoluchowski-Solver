using AlgebraOfGraphics
using CairoMakie
using DataFrames
using DelimitedFiles
using CSV
# Include the potentials file
include("../utils/Potentials/potentials.jl")

nx = 100
dx = 2.0 / (nx - 1)

μ1, _, _, _, _ = generate_potential_1(nx, dx)
μ2, _, _, _, _ = generate_potential_2(nx, dx)
μ3, _, _, _, _ = generate_potential_3("realistic_potential/")
x_range = range(-1, 1, length = nx)
y_range = range(-1, 1, length = nx)

# Helper function to convert matrix to DataFrame for AlgebraOfGraphics
function matrix_to_df(x_range, y_range, matrix, potential_name)
    df = DataFrame(
        x = repeat(collect(x_range), outer = length(y_range)),
        y = repeat(collect(y_range), inner = length(x_range)),
        z = vec(matrix'),
        potential = fill(potential_name, length(x_range) * length(y_range)),
    )
    return df
end

# 3D visualisation of potentials
begin
    println("Generating figures of potentials...")
    df1 = matrix_to_df(x_range, y_range, μ1, "Potential 1: x²y + eˣsin(y)")
    df2 = matrix_to_df(x_range, y_range, μ2, "Potential 2: xy² + ln(x²+1) + y³")
    df3 = matrix_to_df(x_range, y_range, μ3, "Potential 3")

    df_all = vcat(df1, df2, df3)

    fig = Figure(; size = (1500, 500))

    plt =
        data(df_all) *
        mapping(:x, :y, :z, col = :potential) *
        visual(Surface, colormap = :magma)
    ag = draw!(
        fig,
        plt,
        axis = (type = Axis3, aspect = (1, 1, 1), limits = (nothing, nothing, (-3.5, 3.5))),
        facet = (linkyaxes = :minimal, linkxaxes = :minimal),
    )

    output_path = "figures/potentials.png"
    println("Saving $output_path")
    save(output_path, fig)
end

#Initial concentration generation
begin
    println("Generating the graph of initial concentration")
    nx = 100
    function generate_u0(nx, sigma)
        x_range = range(-1, 1, length = nx)
        y_range = range(-1, 1, length = nx)
        u0_matrix = zeros(nx, nx)
        norm_factor = 1 / (2 * π * sigma^2)
        x = []
        y = []
        gauss = []
        for i = 1:nx
            for j = 1:nx
                push!(x, x_range[i])
                push!(y, y_range[j])
                push!(
                    gauss,
                    norm_factor * exp(-(x_range[i]^2 + y_range[j]^2) / (2 * sigma^2)),
                )
            end
        end
        return x, y, gauss
    end
    x, y, gauss = generate_u0(100, 1.0)
    plt = mapping(x, y, gauss) * visual(Surface, colormap = :magma)
    fig = Figure(; size = (500, 500))
    draw!(
        fig,
        plt,
        axis = (
            type = Axis3,
            aspect = (1, 1, 1),
            elevation = pi/4,
            title = "Initial concentration",
        ),
    )
    output_path = "figures/initial.png"
    println("Saving $output_path")
    save(output_path, fig)
end

# Method of manufactured solution
begin
    mms1 = CSV.read("data/mms1_t_ex.tsv", DataFrame, delim = '\t')
    mms1[!, :potential] .= "Potential 1"
    mms2 = CSV.read("data/mms1_t_ex.tsv", DataFrame, delim = '\t')
    mms2[!, :potential] .= "Potential 2"
    mms = vcat(mms1, mms2)
    plt =
        data(filter(:method => in(["lie", "strang"]), mms))*mapping(
            :dt => (x -> log2(x)),
            :dx => (x -> log2(x)),
            :error => L"$L^2$ error",
            row = :method,
            col = :potential,
        )*visual(Heatmap)
    dx_values = [-i for i = 3:8]
    dt_values = [-i for i = 3:16]
    figure_options = (;
        title = L"$L^2$ error of spliting methods on 2 different MMS potentials",
        size = (1000, 500),
    )
    figure = draw(
        plt,
        scales(Color = (; colormap = :magma)),
        axis = (
            aspect = 2,
            # Custom ticks for the x-axis (dt)
            xticks = (dt_values, [L"2^{%$(-i)}" for i = 3:16]),
            # Custom ticks for the y-axis (dx)
            yticks = (dx_values, [L"2^{%$(-i)}" for i = 3:8]),
            xlabel = "dt (time step)",
            ylabel = "dx (grid size)",
            xticklabelsize = 12, # <-- Decrease X tick label font size
            yticklabelsize = 12,  # <-- Decrease Y tick label font size
        );
        figure = figure_options
    )
    output_path = "figures/splting_mms.pdf"
    println("Saving $output_path")
    save(output_path, figure)
end

begin
    mms1 = CSV.read("data/mms1_t_ex.tsv", DataFrame, delim = '\t')
    mms1[!, :potential] .= "Potential 1"
    mms2 = CSV.read("data/mms1_t_ex.tsv", DataFrame, delim = '\t')
    mms2[!, :potential] .= "Potential 2"
    mms = vcat(mms1, mms2)
    plt =
        data(filter(:method => in(["adi"]), mms))*mapping(
            :dt => (x -> log2(x)),
            :dx => (x -> log2(x)),
            :error => L"$L^2$ error",
            col = :potential,
        )*visual(Heatmap)
    dx_values = [-i for i = 3:8]
    dt_values = [-i for i = 3:16]
    figure_options = (;
        title = L"$L^2$ error of ADI method on 2 different MMS potentials",
        size = (1000, 300),
    )
    figure = draw(
        plt,
        scales(Color = (; colormap = :magma)),
        axis = (
            aspect = 2,
            # Custom ticks for the x-axis (dt)
            xticks = (dt_values, [L"2^{%$(-i)}" for i = 3:16]),
            # Custom ticks for the y-axis (dx)
            yticks = (dx_values, [L"2^{%$(-i)}" for i = 3:8]),
            xlabel = "dt (time step)",
            ylabel = "dx (grid size)",
            xticklabelsize = 12, # <-- Decrease X tick label font size
            yticklabelsize = 12,  # <-- Decrease Y tick label font size
        );
        figure = figure_options
    )
    output_path = "figures/adi_mms.pdf"
    println("Saving $output_path")
    save(output_path, figure)
end
