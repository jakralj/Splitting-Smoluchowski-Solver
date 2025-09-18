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
#const nx = 100
const dx = 1/1000
const kbT = 0.65



include("../methods.jl")

"""
Generate potential field and derivatives for f(x,y) = x²y + e^x sin(y)
"""
function generate_potential_1(nx, dx)
    # Grid generation
    x_range = range(-1, 1, length=nx)
    y_range = range(-1, 1, length=nx)

    μ = zeros(nx, nx)
    ∇μX = zeros(nx, nx)
    ∇μY = zeros(nx, nx)
    ∇2μX = zeros(nx, nx)
    ∇2μY = zeros(nx, nx)

    for i in 1:nx
        for j in 1:nx
            x = x_range[i]
            y = y_range[j]

            # Potential: f(x,y) = x²y + e^x sin(y)
            μ[i,j] = x^2 * y + exp(x) * sin(y)

            # First derivatives
            ∇μX[i,j] = 2*x*y + exp(x)*sin(y)  # ∂f/∂x
            ∇μY[i,j] = x^2 + exp(x)*cos(y)    # ∂f/∂y

            # Second derivatives
            ∇2μX[i,j] = 2*y + exp(x)*sin(y)   # ∂²f/∂x²
            ∇2μY[i,j] = -exp(x)*sin(y)        # ∂²f/∂y²
        end
    end

    return μ, ∇μX, ∇μY, ∇2μX, ∇2μY
end

"""
Generate potential field and derivatives for f(x,y) = xy² + ln(x² + 1) + y³
"""
function generate_potential_2(nx, dx)
    # Grid generation
    x_range = range(-1, 1, length=nx)
    y_range = range(-1, 1, length=nx)

    μ = zeros(nx, nx)
    ∇μX = zeros(nx, nx)
    ∇μY = zeros(nx, nx)
    ∇2μX = zeros(nx, nx)
    ∇2μY = zeros(nx, nx)

    for i in 1:nx
        for j in 1:nx
            x = x_range[i]
            y = y_range[j]

            # Potential: f(x,y) = xy² + ln(x² + 1) + y³
            μ[i,j] = x*y^2 + log(x^2 + 1) + y^3

            # First derivatives
            ∇μX[i,j] = y^2 + 2*x/(x^2 + 1)    # ∂f/∂x
            ∇μY[i,j] = 2*x*y + 3*y^2          # ∂f/∂y

            # Second derivatives
            ∇2μX[i,j] = 2*(1 - x^2)/(x^2 + 1)^2  # ∂²f/∂x²
            ∇2μY[i,j] = 2*x + 6*y                # ∂²f/∂y²
        end
    end

    return μ, ∇μX, ∇μY, ∇2μX, ∇2μY
end
"""
Generate potential field based on anatomical examples 
"""
function generate_potential_3()
    # Generate potential field and derivatives
    μ = readdlm("pmf.in")

    # X-direction derivatives
    ∇μX = readdlm("damjux.in")
    ∇2μX = readdlm("d2amjux.in")

    # Y-direction derivatives
    ∇μY = readdlm("damjuy.in")
    ∇2μY = readdlm("d2amjuy.in")

    return μ, ∇μX, ∇μY, ∇2μX, ∇2μY
end

function run_suite(which=:gpu;
                   u0=undef,
                   μ=undef,   ∇μX=undef,  ∇μY=undef,
                   ∇2μX=undef, ∇2μY=undef,
                   dt=0.01,
                   num_steps=1000,
                   D=D, dx=dx, kbT=kbT)

    if which === :lie
        b = @benchmarkable lie_splitting($dt,$dx,$u0,$num_steps,
                                         $μ,$∇μX,$∇μY,$∇2μX,$∇2μY,$D,$kbT)
        res = run(b)
        return res
    elseif which === :strang
        b = @benchmarkable strang_splitting($dt,$dx,$u0,$num_steps,
                                         $μ,$∇μX,$∇μY,$∇2μX,$∇2μY,$D,$kbT)
        res = run(b)
        return res
    elseif which === :adi
        b = @benchmarkable adi_scheme($dt,$dx,$u0,$num_steps,
                                         $μ,$∇μX,$∇μY,$∇2μX,$∇2μY,$D,$kbT)
        res = run(b)
        return res
    elseif which === :fd
        b = @benchmarkable fd($dt,$dx,$u0,$num_steps,
                                         $μ,$∇μX,$∇μY,$∇2μX,$∇2μY,$D,$kbT)
        res = run(b)
        return res
    else # :both
        lie = @benchmarkable lie_splitting($dt,$dx,$u0,$num_steps,
                                         $μ,$∇μX,$∇μY,$∇2μX,$∇2μY,$D,$kbT)
        strang = @benchmarkable strang_splitting($dt,$dx,$u0,$num_steps,
                                         $μ,$∇μX,$∇μY,$∇2μX,$∇2μY,$D,$kbT)
        adi = @benchmarkable adi_scheme($dt,$dx,$u0,$num_steps,
                                         $μ,$∇μX,$∇μY,$∇2μX,$∇2μY,$D,$kbT)
        fd = @benchmarkable fd_scheme($dt,$dx,$u0,$num_steps,
                                         $μ,$∇μX,$∇μY,$∇2μX,$∇2μY,$D,$kbT)
        return (; lie, strang, adi, fd)
    end
end

"""
    compare_gpu( ;steps_vec=[50, 100, 250, 500],
                 potential=:potential_2,    # :potential_1, :potential_3
                 dt=0.01,
                 nx_eff=nx)

Main entry point used for the article.  Executes a full suite,
generates CPU/GPU speed-up plots and accuracy/energy scatter plots.
"""

function bench(steps_vec=[100, 250, 500, 1000], nxs=[100], potential = :potential_2, dt = 0.01)
    #################################################################
    # 1. Pick potential
    #################################################################
    pot = potential === :potential_1 ? generate_potential_1 :
      potential === :potential_2 ? generate_potential_2 :
      generate_potential_2        # default

    lie_ts  = Float64[]
    strang_ts  = Float64[]
    adi_ts = Float64[]
    fd_ts = Float64[]
    ks = Float64[]
    nx_effs = Float64[]

    for nx_eff in nxs
        μ, ∇μX, ∇μY, ∇2μX, ∇2μY = pot(nx_eff, dx)
        x_center, y_center = nx_eff / 2, nx_eff / 2
        σ = nx_eff / 100
        u0 = [exp(-((x-x_center)^2 + (y-y_center)^2)/(2σ^2)) for y in 1:nx_eff, x in 1:nx_eff]
        u0 ./= sum(u0)

        for k in steps_vec
            println("→ Running $k steps at $nx_eff …")
            push!(ks, k)
            push!(nx_effs, nx_eff)
            # ----- Simple CPU version ------------------------------------
            lie_trial = run_suite(:lie,
                 u0=u0, μ=μ, ∇μX=∇μX, ∇μY=∇μY, ∇2μX=∇2μX, ∇2μY=∇2μY,
                 dt=dt, num_steps=k)
            push!(lie_ts, median(lie_trial.times) / 1e9) # ns → sec
            strang_trial = run_suite(:strang,
                 u0=u0, μ=μ, ∇μX=∇μX, ∇μY=∇μY, ∇2μX=∇2μX, ∇2μY=∇2μY,
                 dt=dt, num_steps=k)
            push!(strang_ts, median(strang_trial.times) / 1e9)
            adi_trial = run_suite(:adi,
                 u0=u0, μ=μ, ∇μX=∇μX, ∇μY=∇μY, ∇2μX=∇2μX, ∇2μY=∇2μY,
                 dt=dt, num_steps=k)
            push!(adi_ts, median(adi_trial.times) / 1e9)
            fd_trial = run_suite(:fd,
                 u0=u0, μ=μ, ∇μX=∇μX, ∇μY=∇μY, ∇2μX=∇2μX, ∇2μY=∇2μY,
                 dt=dt, num_steps=k)
            push!(fd_ts, median(fd_trial.times) / 1e9)
        end
    end

    return DataFrame(T = ks, LIE=lie_ts, STRANG=strang_ts, ADI=adi_ts, FD=fd_ts)
end
df = bench()
CSV.write("speed.csv", df)
print(df)
