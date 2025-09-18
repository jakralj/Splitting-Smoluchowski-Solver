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
const nx = 100
const dx = 1/1000
const kbT = 0.65



include("../methods.jl")
include("../gpu.jl")

function kernel_dummy(x, v)
    i = (blockIdx().x-1) * blockDim().x + threadIdx().x
    if i ≤ length(x)
        @inbounds x[i] = v
    end
    return
end
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

    if which === :cpu
        b = @benchmarkable lie_splitting($dt,$dx,$u0,$num_steps,
                                         $μ,$∇μX,$∇μY,$∇2μX,$∇2μY,$D,$kbT)
        res = run(b)
        return res
    elseif which === :gpu
        b = @benchmarkable lie_splitting_gpu($dt,$dx,$u0,$num_steps,
                                             $μ,$∇μX,$∇μY,$∇2μX,$∇2μY,$D,$kbT)
        res = run(b)
        return res
    else # :both
        cpu = run(@benchmarkable lie_splitting($dt,$dx,$u0,$num_steps,
                                               $μ,$∇μX,$∇μY,$∇2μX,$∇2μY,$D,$kbT))
        gpu = run(@benchmarkable lie_splitting_gpu($dt,$dx,$u0,$num_steps,
                                                   $μ,$∇μX,$∇μY,$∇2μX,$∇2μY,$D,$kbT))
        return (; cpu, gpu)
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

function bench(steps_vec=[100, 250, 500, 1000], nxs=[10, 100, 500, 1000, 5000], potential = :potential_2, dt = 0.01)
    #################################################################
    # 1. Pick potential
    #################################################################
    pot = potential === :potential_1 ? generate_potential_1 :
      potential === :potential_2 ? generate_potential_2 :
      generate_potential_2        # default

    

    #################################################################
    # 3. Warm-up GPU – single micro-kernel to avoid compilation noise
    #################################################################
    let
        dummy = CUDA.rand(Float64,  10, 10)
        @cuda threads=32 blocks=1 kernel_dummy(dummy, 1.0);
        CUDA.synchronize()
    end
    cpu_ts  = Float64[]
    cpu_ts_avg = Float64[]
    cpu_ts_sigma = Float64[]
    gpu_ts  = Float64[]
    gpu_ts_avg = Float64[]
    gpu_ts_sigma = Float64[]
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
            cpu_trial = run_suite(:cpu,
                 u0=u0, μ=μ, ∇μX=∇μX, ∇μY=∇μY, ∇2μX=∇2μX, ∇2μY=∇2μY,
                 dt=dt, num_steps=k)
            push!(cpu_ts, median(cpu_trial.times) / 1e9) # ns → sec
            push!(cpu_ts_avg, mean(cpu_trial.times) / 1e9) # ns → sec
            push!(cpu_ts_sigma, std(cpu_trial.times) / 1e9) # ns → sec

            # ----- GPU ---------------------------------------------------
            gpu_trial = run_suite(:gpu,
                 u0=u0, μ=μ, ∇μX=∇μX, ∇μY=∇μY, ∇2μX=∇2μX, ∇2μY=∇2μY,
                 dt=dt, num_steps=k)
            push!(gpu_ts, median(gpu_trial.times) / 1e9) # ns → sec
            push!(gpu_ts_avg, mean(gpu_trial.times) / 1e9) # ns → sec
            push!(gpu_ts_sigma, std(gpu_trial.times) / 1e9) # ns → sec
        end
    end

    return DataFrame(T = ks, N = nx_effs, CPU_AVG = cpu_ts_avg, CPU_MEDIAN = cpu_ts, CPU_SIGMA = cpu_ts_sigma, GPU_AVG = gpu_ts_avg, GPU_MEDIAN = gpu_ts, GPU_SIGMA = gpu_ts_sigma)
end

function compare_gpu(;
                     steps_vec   = [50, 100, 250, 500],
                     potential   = :potential_2,
                     dt          = 0.01,
                     nx_eff      = nx)

    #################################################################
    # 1. Pick potential
    #################################################################
    pot = potential === :potential_1 ? generate_potential_1 :
      potential === :potential_2 ? generate_potential_2 :
      potential === :potential_3 ? generate_potential_3 :
      generate_potential_2        # default

μ, ∇μX, ∇μY, ∇2μX, ∇2μY = if potential === :potential_3
    generate_potential_3()            # returns  5 matrices already
else
    generate_potential_2(nx_eff, dx)  # ← pick either 1, 2, 3 here
end

    #################################################################
    # 2. Gaussian initial condition on the same grid
    #################################################################
    x_center, y_center = nx_eff / 2, nx_eff / 2
    σ = nx_eff / 100
    u0 = [exp(-((x-x_center)^2 + (y-y_center)^2)/(2σ^2)) for y in 1:nx_eff, x in 1:nx_eff]
    u0 ./= sum(u0)

    #################################################################
    # 3. Warm-up GPU – single micro-kernel to avoid compilation noise
    #################################################################
    let
        dummy = CUDA.rand(Float64,  10, 10)
        @cuda threads=32 blocks=1 kernel_dummy(dummy, 1.0);
        CUDA.synchronize()
    end

    #################################################################
    # 4. Benchmarks and accuracy checks
    #################################################################
    cpu_ts  = Float64[]
    gpu_ts  = Float64[]
    norms   = Float64[]
    gpu_ref = zero(u0)

    for k in steps_vec
        println("→ Running $k steps  …")
        # ----- Simple CPU version ------------------------------------
        cpu_trial = run_suite(:cpu,
                 u0=u0, μ=μ, ∇μX=∇μX, ∇μY=∇μY, ∇2μX=∇2μX, ∇2μY=∇2μY,
                 dt=dt, num_steps=k)
        push!(cpu_ts,  median(cpu_trial.times) / 1e9) # ns → sec

        # ----- GPU ---------------------------------------------------
        gpu_trial = run_suite(:gpu,
                 u0=u0, μ=μ, ∇μX=∇μX, ∇μY=∇μY, ∇2μX=∇2μX, ∇2μY=∇2μY,
                 dt=dt, num_steps=k)
        push!(gpu_ts, median(gpu_trial.times) / 1e9)  # ns → sec
    end

    #################################################################
    # 5. Reference checked for accuracy (finest grid)
    #################################################################
    u_cpu  = lie_splitting(dt, dx, u0, maximum(steps_vec), μ, ∇μX, ∇μY, ∇2μX, ∇2μY, D, kbT)
    u_gpu  = lie_splitting_gpu(dt, dx, u0, maximum(steps_vec), μ, ∇μX, ∇μY, ∇2μX, ∇2μY, D, kbT)
    err    = norm(u_cpu - u_gpu)
    push!(norms, err)

    #################################################################
    # 6. Figure 1 – runtime vs problem size
    #################################################################
    p1 = plot(steps_vec, cpu_ts,
              markershape=:circle, markersize=MARKERSIZE, lw=LW,
              label="CPU (Lie splitting)",
              yaxis=:log, xguide="time steps", yguide="median time (s)")

    plot!(p1, steps_vec, gpu_ts,
          markershape=:rect, markersize=MARKERSIZE, lw=LW,
          label="GPU (Lie splitting on CUDA)")

    speedup = cpu_ts ./ gpu_ts
    p2 = twinx(p1)
    plot!(p2, steps_vec, speedup,
          linestyle=:dash, lw=LW, color=:black,
          label="Speed-up", yguide="speed-up  (CPU/GPU)",
          legend=true)

    #################################################################
    # 7. Figure 2 – bar chart (simplified)
    #################################################################
    labels = ["CPU", "GPU"]
    times  = [cpu_ts[end], gpu_ts[end]]
    p3 = bar(labels, times,
         legend=false,
         yguide="median execution time (s)",
         title="Runtime comparison ($(steps_vec[end]) steps)",
         color=[:darkblue :orange])
    #################################################################
    # 8. Report to console
    #################################################################
    res_df = (; steps=steps_vec, cpu_time=cpu_ts, gpu_time=gpu_ts,
               speedup, accuracy_error=norms[1])
    display(res_df)

    #################################################################
    # 9. Save plots (optional)
    #################################################################
    savefig(p1, "bench_time_steps_$(potential).png")
    savefig(p3, "bench_bar_$(potential).png")

    #################################################################
    # 10. Return data for interactive investigation
    #################################################################
    return res_df
end

########################################################################
# If the file is evaluated directly, run the default benchmark suite   #
########################################################################
if abspath(PROGRAM_FILE) == @__FILE__
    df = bench() # produces plots in the CWD
    CSV.write("speedup.csv", df)
end
