# ------------------------------------------------------------------
#  convergence_stability.jl
#  Convergence + stability study for the 2-D Smoluchowski PDE solver
# ------------------------------------------------------------------
using LinearAlgebra, Statistics, DataFrames, DelimitedFiles, Printf, CSV    # DataFrames just for niceness
# ------------------------------------------------------------------
#  Integral helpers (mid-point rule on the uniform mesh)
# ------------------------------------------------------------------
include("../methods.jl")      # solvers
include("../../utils/Potentials/potentials.jl")   # potentials

# ------------------------------------------------------------
#  Helper integrals
# ------------------------------------------------------------
∫(u, dx) = sum(u) * dx^2

function total_variation(u, dx)
    nx, ny = size(u)
    tv = 0.0
    @inbounds for j in 1:ny, i in 1:nx-1
        tv += abs(u[i+1,j] - u[i,j])
    end
    @inbounds for j in 1:ny-1, i in 1:nx
        tv += abs(u[i,j+1] - u[i,j])
    end
    tv * dx
end

# ------------------------------------------------------------
#  Problem setup
# ------------------------------------------------------------
const D   = 0.01
const kbT = 0.65
const nx  = 100
const dx  = 2.0 / (nx - 1)
const Tend = 1.0

# grid
x = range(-1.0, 1.0, length=nx)
y = range(-1.0, 1.0, length=nx)
# Gaussian initial condition
σ  = 0.3
u0 = [ exp(-( ((xi+1.0)/nx - 0.5)^2 + ((yi+1.0)/nx - 0.5)^2 )/(2σ^2))
       for yi in 1:nx, xi in 1:nx ]
u0 ./= ∫(u0, dx)

# potential data
μ, ∇μX, ∇μY, ∇2μX, ∇2μY = generate_potential_1(nx, dx)

# ------------------------------------------------------------
#  Reference solution (dt_ref = 1e-8) – cached on disk
# ------------------------------------------------------------
const dt_ref  = 1e-7
const steps_r = round(Int, Tend / dt_ref)

if isfile("reference.csv")
    println("Re-using pre-computed reference")
    u_ref = Matrix{Float64}(readdlm("reference.csv", ','))
else
    println("Computing reference (1e-8 steps) … may take minutes")
    u_ref = lie_splitting(dt_ref, dx, u0, steps_r,
                          μ, ∇μX, ∇μY, ∇2μX, ∇2μY, D, kbT)
    writedlm("reference.csv", u_ref, ',')
end

# ------------------------------------------------------------
#  Study loop
# ------------------------------------------------------------
Δt_seq = [1e-5, 2e-5, 5e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]

df = DataFrame(Δt = Float64[],
               method = String[],
               L2_error = Float64[],
               mass     = Float64[],
               max_u    = Float64[],
               TV       = Float64[])

methods = Dict("Lie"    => (dt,s,u)->lie_splitting(dt,dx,u,s,μ,∇μX,∇μY,∇2μX,∇2μY,D,kbT),
               "Strang" => (dt,s,u)->strang_splitting(dt,dx,u,s,μ,∇μX,∇μY,∇2μX,∇2μY,D,kbT),
               "ADI"    => (dt,s,u)->adi_scheme(dt,dx,u,s,μ,∇μX,∇μY,∇2μX,∇2μY,D,kbT))

for (tag, solver!) in methods
    for dt in Δt_seq
        steps = round(Int, Tend / dt)
        u     = solver!(dt, steps, u0)

        push!(df, (dt, tag,
                   sqrt(∫((u .- u_ref).^2, dx)),
                   ∫(u, dx), maximum(abs.(u)), total_variation(u, dx)))
    end
end

# ------------------------------------------------------------
#  Output
# ------------------------------------------------------------
CSV.write("CS_results.csv", df)
println("Results saved to CS_results.csv")
display(first(df, 5))   # quick peek
