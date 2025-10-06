using LinearAlgebra
using SparseArrays
using Plots
using StatsPlots
using DelimitedFiles
using BenchmarkTools
using Statistics
using DataFrames
using CSV

const MARKERSIZE = 4
const LW = 2.5
default(titlefont = font(10), guidefont = font(9), legendfont = font(7))

const D = 0.01
const dx = 1/1000
const kbT = 0.65

include("../../methods.jl")
include("../../utils/Potentials/potentials.jl")

function run_suite(
    which = :lie;
    u0 = undef,
    μ = undef,
    ∇μX = undef,
    ∇μY = undef,
    ∇2μX = undef,
    ∇2μY = undef,
    dt = 0.01,
    num_steps = 1000,
    D = D,
    dx = dx,
    kbT = kbT,
)

    if which === :lie
        b = @benchmarkable lie_splitting(
            $dt,
            $dx,
            $u0,
            $num_steps,
            $μ,
            $∇μX,
            $∇μY,
            $∇2μX,
            $∇2μY,
            $D,
            $kbT,
        )
        res = run(b)
        return res
    elseif which === :strang
        b = @benchmarkable strang_splitting(
            $dt,
            $dx,
            $u0,
            $num_steps,
            $μ,
            $∇μX,
            $∇μY,
            $∇2μX,
            $∇2μY,
            $D,
            $kbT,
        )
        res = run(b)
        return res
    elseif which === :adi
        b = @benchmarkable adi_scheme(
            $dt,
            $dx,
            $u0,
            $num_steps,
            $μ,
            $∇μX,
            $∇μY,
            $∇2μX,
            $∇2μY,
            $D,
            $kbT,
        )
        res = run(b)
        return res
    else # :both
        lie = @benchmarkable lie_splitting(
            $dt,
            $dx,
            $u0,
            $num_steps,
            $μ,
            $∇μX,
            $∇μY,
            $∇2μX,
            $∇2μY,
            $D,
            $kbT,
        )
        strang = @benchmarkable strang_splitting(
            $dt,
            $dx,
            $u0,
            $num_steps,
            $μ,
            $∇μX,
            $∇μY,
            $∇2μX,
            $∇2μY,
            $D,
            $kbT,
        )
        adi = @benchmarkable adi_scheme(
            $dt,
            $dx,
            $u0,
            $num_steps,
            $μ,
            $∇μX,
            $∇μY,
            $∇2μX,
            $∇2μY,
            $D,
            $kbT,
        )
        return (; lie, strang, adi)
    end
end

"""
    bench( ;steps_vec=[100, 250, 500, 1000], nxs=[100],
           potential=:potential_2, dt=0.01)

Executes a full suite of benchmarks for different numerical schemes,
collects run times and calculates medians and standard deviations.
"""
function bench(
    steps_vec = [100, 250, 500, 1000],
    nxs = [100],
    potential = :potential_2,
    dt = 0.01,
)
    #################################################################
    # 1. Pick potential
    #################################################################
    pot =
        potential === :potential_1 ? generate_potential_1 :
        potential === :potential_2 ? generate_potential_2 : generate_potential_2        # default

    # Store all time samples for each run to calculate median and std dev
    all_lie_times_samples = Vector{Float64}[]
    all_strang_times_samples = Vector{Float64}[]
    all_adi_times_samples = Vector{Float64}[]

    # Store the corresponding k and nx_eff values for each set of samples
    collected_ks = Int[]
    collected_nx_effs = Int[]

    for nx_eff in nxs
        μ, ∇μX, ∇μY, ∇2μX, ∇2μY = pot(nx_eff, dx)
        x_center, y_center = nx_eff / 2, nx_eff / 2
        σ = nx_eff / 100
        u0 = [exp(-((x-x_center)^2 + (y-y_center)^2)/(2σ^2)) for y = 1:nx_eff, x = 1:nx_eff]
        u0 ./= sum(u0)

        for k in steps_vec
            println("→ Running $k steps at $nx_eff …")

            # Store the current k and nx_eff for this set of benchmarks
            push!(collected_ks, k)
            push!(collected_nx_effs, nx_eff)

            # ----- Simple CPU version ------------------------------------
            lie_trial = run_suite(
                :lie,
                u0 = u0,
                μ = μ,
                ∇μX = ∇μX,
                ∇μY = ∇μY,
                ∇2μX = ∇2μX,
                ∇2μY = ∇2μY,
                dt = dt,
                num_steps = k,
            )
            push!(all_lie_times_samples, lie_trial.times / 1e9) # ns → sec

            strang_trial = run_suite(
                :strang,
                u0 = u0,
                μ = μ,
                ∇μX = ∇μX,
                ∇μY = ∇μY,
                ∇2μX = ∇2μX,
                ∇2μY = ∇2μY,
                dt = dt,
                num_steps = k,
            )
            push!(all_strang_times_samples, strang_trial.times / 1e9)

            adi_trial = run_suite(
                :adi,
                u0 = u0,
                μ = μ,
                ∇μX = ∇μX,
                ∇μY = ∇μY,
                ∇2μX = ∇2μX,
                ∇2μY = ∇2μY,
                dt = dt,
                num_steps = k,
            )
            push!(all_adi_times_samples, adi_trial.times / 1e9)
        end
    end

    # Calculate medians and standard deviations
    lie_medians = Float64[]
    lie_stds = Float64[]
    strang_medians = Float64[]
    strang_stds = Float64[]
    adi_medians = Float64[]
    adi_stds = Float64[]

    # Iterate through the collected time samples
    for i = 1:length(collected_ks)
        current_lie_times_samples = all_lie_times_samples[i]
        current_strang_times_samples = all_strang_times_samples[i]
        current_adi_times_samples = all_adi_times_samples[i]

        push!(lie_medians, median(current_lie_times_samples))
        push!(lie_stds, std(current_lie_times_samples))
        push!(strang_medians, median(current_strang_times_samples))
        push!(strang_stds, std(current_strang_times_samples))
        push!(adi_medians, median(current_adi_times_samples))
        push!(adi_stds, std(current_adi_times_samples))
    end

    # Prepare DataFrame for LaTeX output
    df_results = DataFrame(
        Steps = collected_ks,
        Nx = collected_nx_effs,
        LIE = [
            "$(round(m, digits=4)) \\pm $(round(s, digits=4))" for
            (m, s) in zip(lie_medians, lie_stds)
        ],
        STRANG = [
            "$(round(m, digits=4)) \\pm $(round(s, digits=4))" for
            (m, s) in zip(strang_medians, strang_stds)
        ],
        ADI = [
            "$(round(m, digits=4)) \\pm $(round(s, digits=4))" for
            (m, s) in zip(adi_medians, adi_stds)
        ],
    )

    return df_results
end

df = bench()
CSV.write("speed_with_std.csv", df)
println(df)

# Function to output DataFrame in LaTeX tabular format
function writetable_latex(df::DataFrame, filename::String)
    open(filename, "w") do io
        println(io, "\\begin{tabular}{c|c|c|c|c}")
        println(io, "\\hline")
        # Header
        println(io, "Steps & Nx & LIE & STRANG & ADI \\\\")
        println(io, "\\hline")

        # Rows
        for row in eachrow(df)
            println(
                io,
                "$(row.Steps) & $(row.Nx) & $(row.LIE) & $(row.STRANG) & $(row.ADI) \\\\",
            )
        end

        println(io, "\\hline")
        println(io, "\\end{tabular}")
    end
end

writetable_latex(df, "benchmark_results.tex")
