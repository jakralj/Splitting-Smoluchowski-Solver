using LinearAlgebra
using DelimitedFiles
using DataFrames
using CSV

const D = 0.01
const nx = 100
const dx = 2/100
const kbT = 0.65

include("../../methods.jl")
include("../../utils/Potentials/potentials.jl")

μ1, ∇μX1, ∇μY1, ∇2μX1, ∇2μY1 = generate_potential_1(nx, dx)
μ2, ∇μX2, ∇μY2, ∇2μX2, ∇2μY2 = generate_potential_2(nx, dx)
μ3, ∇μX3, ∇μY3, ∇2μX3, ∇2μY3 = generate_potential_3()

helmholtz(u, V) = sum(u .* log.(u) .+ V .* u ./kbT) * dx^2

function lie_splitting_helmholtz!(u, dt::Float64, dx, num_steps, μ, ∇μX, ∇μY, ∇2μX, ∇2μY, D, kbT, temp_arrays)
    u_temp = temp_arrays.u_half  # Reuse the allocated matrix
    h = []
    for step in 1:num_steps
        evolve_x!(u_temp, u, μ, ∇μX, ∇2μX, D, dx, dt, kbT, temp_arrays)
        evolve_y!(u, u_temp, μ, ∇μY, ∇2μY, D, dx, dt, kbT, temp_arrays)
        push!(h, helmholtz(u, μ)) 
    end
    return h
end

function lie_splitting_helmholtz(dt::Float64, dx, u0, num_steps, μ, ∇μX, ∇μY, ∇2μX, ∇2μY, D, kbT)
    nx, ny = size(u0)
    temp_arrays = TempArrays2D(eltype(u0), nx, ny)
    u = copy(u0)  # Only copy needed for API compatibility
    return lie_splitting_helmholtz!(u, dt, dx, num_steps, μ, ∇μX, ∇μY, ∇2μX, ∇2μY, D, kbT, temp_arrays)
end

function strang_splitting_helmholtz!(u, dt::Float64, dx, num_steps, μ, ∇μX, ∇μY, ∇2μX, ∇2μY, D, kbT, temp_arrays)
    u_temp = temp_arrays.u_half  # Reuse the allocated matrix
    h = []
    for step in 1:num_steps
        evolve_x!(u_temp, u, μ, ∇μX, ∇2μX, D, dx, dt/2, kbT, temp_arrays)
        evolve_y!(u, u_temp, μ, ∇μY, ∇2μY, D, dx, dt, kbT, temp_arrays)
        evolve_x!(u_temp, u, μ, ∇μX, ∇2μX, D, dx, dt/2, kbT, temp_arrays)
        # Copy result back to u
        copyto!(u, u_temp)
        push!(h, helmholtz(u, μ)) 
    end
    return h
end

function strang_splitting_helmholtz(dt::Float64, dx, u0, num_steps, μ, ∇μX, ∇μY, ∇2μX, ∇2μY, D, kbT)
    nx, ny = size(u0)
    temp_arrays = TempArrays2D(eltype(u0), nx, ny)
    u = copy(u0)  # Only copy needed for API compatibility
    return strang_splitting_helmholtz!(u, dt, dx, num_steps, μ, ∇μX, ∇μY, ∇2μX, ∇2μY, D, kbT, temp_arrays)
end

function adi_scheme_helmholtz!(u, dt::Float64, dx::Float64, num_steps, μ, ∇μX, ∇μY, ∇2μX, ∇2μY, D, kbT, temp_arrays)
    # Pre-compute coefficients
    αx = D*dt/(2*dx^2)
    αy = D*dt/(2*dx^2)
    βx = D*dt/(4*kbT*dx)
    βy = D*dt/(4*kbT*dx)
    h = []
    # Main time-stepping loop
    for step in 1:num_steps
        # First sweep: x-direction (implicit in x, explicit in y)
        x_direction_sweep!(temp_arrays.u_half, u, αx, αy, βx, βy, dt, ∇μX, ∇μY, ∇2μX, ∇2μY, D, kbT, temp_arrays)

        # Second sweep: y-direction (implicit in y, explicit in x)
        y_direction_sweep!(u, temp_arrays.u_half, αx, αy, βx, βy, dt, ∇μX, ∇μY, ∇2μX, ∇2μY, D, kbT, temp_arrays)
        push!(h, helmholtz(u, μ)) 
    end

    return h
end

function adi_scheme_helmholtz(dt::Float64, dx::Float64, u0, num_steps, μ, ∇μX, ∇μY, ∇2μX, ∇2μY, D, kbT)
    nx, ny = size(u0)
    temp_arrays = ADITempArrays(eltype(u0), nx, ny)
    u = copy(u0)  # Only copy needed for API compatibility
    return adi_scheme_helmholtz!(u, dt, dx, num_steps, μ, ∇μX, ∇μY, ∇2μX, ∇2μY, D, kbT, temp_arrays)
end

function generate_u0(nx, sigma)
    x_range = range(-1, 1, length=nx)
    y_range = range(-1, 1, length=nx)
    u0_matrix = zeros(nx, nx)
    norm_factor = 1 / (2 * π * sigma^2)
    for i in 1:nx
        for j in 1:nx
            x = x_range[i]
            y = y_range[j]
            u0_matrix[i,j] = norm_factor * exp(-(x^2 + y^2) / (2 * sigma^2))
        end
    end
    return u0_matrix
end

u0 = generate_u0(nx, 1.0)
h1 = lie_splitting_helmholtz(1/100, dx, u0, 1000, μ1, ∇μX1, ∇μY1, ∇2μX1, ∇2μY1, D, kbT)  
h2 = strang_splitting_helmholtz(1/100, dx, u0, 1000, μ1, ∇μX1, ∇μY1, ∇2μX1, ∇2μY1, D, kbT)  
h3 = adi_scheme_helmholtz(1/100, dx, u0, 1000, μ1, ∇μX1, ∇μY1, ∇2μX1, ∇2μY1, D, kbT)  
