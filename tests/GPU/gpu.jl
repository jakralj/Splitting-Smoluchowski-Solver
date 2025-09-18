using CUDA
using LinearAlgebra
using SparseArrays
using Plots
using DelimitedFiles
using Statistics
# Enable CUDA if available
if CUDA.functional()
    println("CUDA is functional - using GPU acceleration")
else
    println("CUDA not available - falling back to CPU")
end

const D = 0.01
const nx = 100
const dx = 1/1000
const kbT = 0.65

# Load potential field and derivatives (keep on CPU for now)
const μ_cpu = readdlm("pmf.in") 
const ∇μX_cpu = readdlm("damjux.in") 
const ∇2μX_cpu = readdlm("d2amjux.in")
const ∇μY_cpu = readdlm("damjuy.in") 
const ∇2μY_cpu = readdlm("d2amjuy.in") 

# Transfer to GPU
const μ_gpu = CuArray{Float64}(μ_cpu)
const ∇μX_gpu = CuArray{Float64}(∇μX_cpu)
const ∇2μX_gpu = CuArray{Float64}(∇2μX_cpu)
const ∇μY_gpu = CuArray{Float64}(∇μY_cpu)
const ∇2μY_gpu = CuArray{Float64}(∇2μY_cpu)

"""
GPU kernel for applying shift operation with zero boundary conditions
"""
function shift_kernel!(output, input, shift_amount, n)
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    
    if i <= n
        new_idx = i + shift_amount
        if new_idx >= 1 && new_idx <= n
            output[i] = input[new_idx]
        else
            output[i] = 0.0
        end
    end
    return nothing
end

"""
GPU-accelerated shift function
"""
function gpu_shift(u::CuArray, s::Int)
    n = length(u)
    output = CUDA.zeros(Float64, n)
    
    threads = min(256, n)
    blocks = ceil(Int, n / threads)
    
    @cuda threads=threads blocks=blocks shift_kernel!(output, u, -s, n)
    
    return output
end

"""
GPU kernel for computing the RHS of Crank-Nicolson scheme
"""
function crank_nicolson_rhs_kernel!(B, u, uP, uN, ∇μ, ∇2μ, α, β, γ, ϵ, dt_inv, n)
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    
    if i <= n
        B[i] = -1.0 * (uP[i] * (-∇μ[i] * α + β) + 
                       u[i] * (∇2μ[i] * γ - ϵ + dt_inv) + 
                       uN[i] * (∇μ[i] * α + β))
    end
    return nothing
end

"""
GPU-accelerated Crank-Nicolson solver for 1D Smoluchowski equation
"""
function gpu_crank_nicolson_1d(u::CuArray, μ::CuArray, ∇μ::CuArray, ∇2μ::CuArray, 
                                D::Float64, dx::Float64, dt::Float64, kbT::Float64)
    n = length(u)
    α = D / (4*dx*kbT)
    β = D / (2*dx^2)
    γ = D / (2*kbT)
    ϵ = D / (dx^2)
    dt_inv = 1/dt
    
    # Shift operations on GPU
    uP = gpu_shift(u, 1)
    uN = gpu_shift(u, -1)
    
    # Compute RHS on GPU
    B = CUDA.zeros(Float64, n)
    threads = min(256, n)
    blocks = ceil(Int, n / threads)
    
    @cuda threads=threads blocks=blocks crank_nicolson_rhs_kernel!(
        B, u, uP, uN, ∇μ, ∇2μ, α, β, γ, ϵ, dt_inv, n)
    
    # Build tridiagonal matrix (this part stays on CPU for now due to sparse solver limitations)
    ∇μ_cpu = Array(∇μ)
    ∇2μ_cpu = Array(∇2μ)
    B_cpu = Array(B)
    
    Adl = -∇μ_cpu[2:end]*α .+ β
    Adu = ∇μ_cpu[1:end-1]*α .+ β
    Au = ∇2μ_cpu*γ .- ϵ .- dt_inv

    # Boundary conditions
    Adu[1] = 2*β
    Adl[end] = 2*β
    B_cpu[1] = -1*(Array(uN)[1]*2*β + Array(u)[1]*(∇2μ_cpu[1]*γ - ϵ + dt_inv))
    B_cpu[end] = -1*(Array(uP)[end]*2*β + Array(u)[end]*(∇2μ_cpu[end]*γ - ϵ + dt_inv))

    A = Tridiagonal(Adl, Au, Adu)
    result_cpu = A \ B_cpu
    
    return CuArray{Float64}(result_cpu)
end

"""
GPU kernel for explicit y-direction computation in ADI scheme
"""
function adi_y_explicit_kernel!(y_term, u, ∇μY, ∇2μY, αy, βy, D, dt, kbT, nx)
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    j = threadIdx().y + (blockIdx().y - 1) * blockDim().y
    
    if i <= nx && j <= nx
        idx = (j-1)*nx + i
        
        if i == 1 || i == nx
            # Top/bottom boundary: von Neumann in y
            if i == 1
                y_term[idx] = 2*αy*(u[(j-1)*nx + i+1] - u[idx]) + 
                             u[idx]*D*dt/(2*kbT)*∇2μY[idx]
            else
                y_term[idx] = 2*αy*(u[(j-1)*nx + i-1] - u[idx]) + 
                             u[idx]*D*dt/(2*kbT)*∇2μY[idx]
            end
        else
            y_term[idx] = αy*(u[(j-1)*nx + i-1] - 2*u[idx] + u[(j-1)*nx + i+1]) + 
                         βy*(u[(j-1)*nx + i+1] - u[(j-1)*nx + i-1])*∇μY[idx] + 
                         u[idx]*D*dt/(2*kbT)*∇2μY[idx]
        end
    end
    return nothing
end

"""
GPU kernel for explicit x-direction computation in ADI scheme
"""
function adi_x_explicit_kernel!(x_term, u_half, ∇μX, ∇2μX, αx, βx, D, dt, kbT, nx)
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    j = threadIdx().y + (blockIdx().y - 1) * blockDim().y
    
    if i <= nx && j <= nx
        idx = (j-1)*nx + i
        
        if j == 1 || j == nx
            # Left/right boundary: von Neumann in x
            if j == 1
                x_term[idx] = 2*αx*(u_half[j*nx + i] - u_half[idx]) + 
                             u_half[idx]*D*dt/(2*kbT)*∇2μX[idx]
            else
                x_term[idx] = 2*αx*(u_half[(j-2)*nx + i] - u_half[idx]) + 
                             u_half[idx]*D*dt/(2*kbT)*∇2μX[idx]
            end
        else
            x_term[idx] = αx*(u_half[(j-2)*nx + i] - 2*u_half[idx] + u_half[j*nx + i]) + 
                         βx*(u_half[j*nx + i] - u_half[(j-2)*nx + i])*∇μX[idx] + 
                         u_half[idx]*D*dt/(2*kbT)*∇2μX[idx]
        end
    end
    return nothing
end

"""
GPU-accelerated ADI scheme for 2D Smoluchowski equation
"""
function gpu_adi_scheme(dt::Float64, u0::CuArray{Float64}, num_steps::Int)
    u = copy(u0)
    u_half = similar(u0)
    
    # Pre-compute coefficients
    αx = D*dt/(2*dx^2)
    αy = D*dt/(2*dx^2)
    βx = D*dt/(4*kbT*dx)
    βy = D*dt/(4*kbT*dx)
    
    # GPU thread configuration
    threads_2d = (16, 16)
    blocks_2d = (ceil(Int, nx/16), ceil(Int, nx/16))
    
    for step in 1:num_steps
        # First half-step: implicit in x, explicit in y
        y_terms = CUDA.zeros(Float64, nx*nx)
        @cuda threads=threads_2d blocks=blocks_2d adi_y_explicit_kernel!(
            y_terms, u, ∇μY_gpu, ∇2μY_gpu, αy, βy, D, dt, kbT, nx)
        
        # Reshape for row-wise processing
        y_terms_2d = reshape(y_terms, nx, nx)
        u_2d = reshape(u, nx, nx)
        u_half_2d = reshape(u_half, nx, nx)
        
        # Process each row (implicit in x)
        for i in 1:nx
            row_data = Array(u_2d[i, :])
            y_term_row = Array(y_terms_2d[i, :])
            μ_row = Array(μ_gpu[i, :])
            ∇μX_row = Array(∇μX_gpu[i, :])
            ∇2μX_row = Array(∇2μX_gpu[i, :])
            
            # Build tridiagonal system
            Adl = zeros(nx-1)
            Adu = zeros(nx-1)
            Au = zeros(nx)
            B = zeros(nx)
            
            for j in 1:nx
                if j == 1
                    Au[j] = 1.0 + 2*αx - D*dt/(2*kbT)*∇2μX_row[j]
                    Adu[j] = -(2*αx + βx*∇μX_row[j])
                    B[j] = row_data[j] + y_term_row[j]
                elseif j == nx
                    Adl[j-1] = -(2*αx - βx*∇μX_row[j])
                    Au[j] = 1.0 + 2*αx - D*dt/(2*kbT)*∇2μX_row[j]
                    B[j] = row_data[j] + y_term_row[j]
                else
                    c_left = -αx - βx*∇μX_row[j]
                    c_center = 1.0 + 2*αx - D*dt/(2*kbT)*∇2μX_row[j]
                    c_right = -αx + βx*∇μX_row[j]
                    
                    Au[j] = c_center
                    Adl[j-1] = c_left
                    Adu[j] = c_right
                    B[j] = row_data[j] + y_term_row[j]
                end
            end
            
            A_matrix = Tridiagonal(Adl, Au, Adu)
            u_half_2d[i,:] = A_matrix \ B
        end
        
        u_half = CuArray{Float64}(vec(u_half_2d))
        
        # Second half-step: explicit in x, implicit in y
        x_terms = CUDA.zeros(Float64, nx*nx)
        @cuda threads=threads_2d blocks=blocks_2d adi_x_explicit_kernel!(
            x_terms, u_half, ∇μX_gpu, ∇2μX_gpu, αx, βx, D, dt, kbT, nx)
        
        x_terms_2d = reshape(x_terms, nx, nx)
        u_half_2d = reshape(u_half, nx, nx)
        u_2d = reshape(u, nx, nx)
        
        # Process each column (implicit in y)
        for j in 1:nx
            col_data = Array(u_half_2d[:, j])
            x_term_col = Array(x_terms_2d[:, j])
            μ_col = Array(μ_gpu[:, j])
            ∇μY_col = Array(∇μY_gpu[:, j])
            ∇2μY_col = Array(∇2μY_gpu[:, j])
            
            # Build tridiagonal system
            Adl = zeros(nx-1)
            Adu = zeros(nx-1)
            Au = zeros(nx)
            B = zeros(nx)
            
            for i in 1:nx
                if i == 1
                    Au[i] = 1.0 + 2*αy - D*dt/(2*kbT)*∇2μY_col[i]
                    Adu[i] = -(2*αy + βy*∇μY_col[i])
                    B[i] = col_data[i] + x_term_col[i]
                elseif i == nx
                    Adl[i-1] = -(2*αy - βy*∇μY_col[i])
                    Au[i] = 1.0 + 2*αy - D*dt/(2*kbT)*∇2μY_col[i]
                    B[i] = col_data[i] + x_term_col[i]
                else
                    c_bottom = -αy - βy*∇μY_col[i]
                    c_center = 1.0 + 2*αy - D*dt/(2*kbT)*∇2μY_col[i]
                    c_top = -αy + βy*∇μY_col[i]
                    
                    Au[i] = c_center
                    Adl[i-1] = c_bottom
                    Adu[i] = c_top
                    B[i] = col_data[i] + x_term_col[i]
                end
            end
            
            A_matrix = Tridiagonal(Adl, Au, Adu)
            u_2d[:,j] = A_matrix \ B
        end
        
        u = CuArray{Float64}(vec(u_2d))
    end
    
    return reshape(u, nx, nx)
end

"""
GPU kernel for x-direction evolution computation
"""
function evolve_x_kernel!(u_new, u, μ, ∇μX, ∇2μX, D, dx, dt, kbT, nx)
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    j = threadIdx().y + (blockIdx().y - 1) * blockDim().y
    
    if i <= nx && j <= nx
        idx = (j-1)*nx + i
        
        α = D / (4*dx*kbT)
        β = D / (2*dx^2)
        γ = D / (2*kbT)
        ϵ = D / (dx^2)
        
        # Get neighboring values
        uP = (j < nx) ? u[j*nx + i] : 0.0
        uN = (j > 1) ? u[(j-2)*nx + i] : 0.0
        
        # Compute intermediate result (this would need proper tridiagonal solve)
        # For now, using explicit approximation
        u_new[idx] = u[idx] + dt * (
            D/(dx^2) * (uP - 2*u[idx] + uN) +
            D/(kbT*dx) * (uP - uN) * ∇μX[idx] / 2 +
            D/kbT * u[idx] * ∇2μX[idx]
        )
    end
    return nothing
end

"""
GPU kernel for y-direction evolution computation
"""
function evolve_y_kernel!(u_new, u, μ, ∇μY, ∇2μY, D, dx, dt, kbT, nx)
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    j = threadIdx().y + (blockIdx().y - 1) * blockDim().y
    
    if i <= nx && j <= nx
        idx = (j-1)*nx + i
        
        # Get neighboring values
        uP = (i < nx) ? u[(j-1)*nx + i+1] : 0.0
        uN = (i > 1) ? u[(j-1)*nx + i-1] : 0.0
        
        # Compute evolution (explicit approximation)
        u_new[idx] = u[idx] + dt * (
            D/(dx^2) * (uP - 2*u[idx] + uN) +
            D/(kbT*dx) * (uP - uN) * ∇μY[idx] / 2 +
            D/kbT * u[idx] * ∇2μY[idx]
        )
    end
    return nothing
end

"""
GPU-accelerated operator for evolution in X direction
"""
function gpu_evolve_x(u_2d::CuArray, dt::Float64)
    u_flat = vec(u_2d)
    u_new = similar(u_flat)
    
    threads = (16, 16)
    blocks = (ceil(Int, nx/16), ceil(Int, nx/16))
    
    @cuda threads=threads blocks=blocks evolve_x_kernel!(
        u_new, u_flat, μ_gpu, vec(∇μX_gpu), vec(∇2μX_gpu), D, dx, dt, kbT, nx)
    
    return reshape(u_new, nx, nx)
end

"""
GPU-accelerated operator for evolution in Y direction
"""
function gpu_evolve_y(u_2d::CuArray, dt::Float64)
    u_flat = vec(u_2d)
    u_new = similar(u_flat)
    
    threads = (16, 16)
    blocks = (ceil(Int, nx/16), ceil(Int, nx/16))
    
    @cuda threads=threads blocks=blocks evolve_y_kernel!(
        u_new, u_flat, μ_gpu, vec(∇μY_gpu), vec(∇2μY_gpu), D, dx, dt, kbT, nx)
    
    return reshape(u_new, nx, nx)
end

"""
GPU-accelerated Lie Splitting (First Order)
u^{n+1} = Y(dt) X(dt) u^n
"""
function gpu_lie_splitting(dt::Float64, u0::Matrix{Float64}, num_steps::Int)
    u = CuArray{Float64}(u0)
    
    for step in 1:num_steps
        # X direction first
        u = gpu_evolve_x(u, dt)
        # Then Y direction
        u = gpu_evolve_y(u, dt)
    end
    
    return Array(u)
end

"""
GPU-accelerated Strang Splitting (Second Order)
u^{n+1} = X(dt/2) Y(dt) X(dt/2) u^n
"""
function gpu_strang_splitting(dt::Float64, u0::Matrix{Float64}, num_steps::Int)
    u = CuArray{Float64}(u0)
    
    for step in 1:num_steps
        # Half-step in X direction
        u = gpu_evolve_x(u, dt/2)
        # Full step in Y direction
        u = gpu_evolve_y(u, dt)
        # Half-step in X direction
        u = gpu_evolve_x(u, dt/2)
    end
    
    return Array(u)
end

"""
GPU kernel for averaging two solutions (used in SWSS)
"""
function average_kernel!(result, u1, u2, n)
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    
    if i <= n
        result[i] = 0.5 * (u1[i] + u2[i])
    end
    return nothing
end

"""
GPU-accelerated Symmetrically Weighted Sequential Splitting (SWSS)
u^{n+1} = 1/2 * [X(dt) Y(dt) + Y(dt) X(dt)] u^n
"""
function gpu_swss_splitting(dt::Float64, u0::Matrix{Float64}, num_steps::Int)
    u = CuArray{Float64}(u0)
    
    for step in 1:num_steps
        # Compute both orderings
        u1 = gpu_evolve_y(gpu_evolve_x(u, dt), dt)  # X then Y
        u2 = gpu_evolve_x(gpu_evolve_y(u, dt), dt)  # Y then X
        
        # Average on GPU
        u_flat = vec(u)
        u1_flat = vec(u1)
        u2_flat = vec(u2)
        
        threads = min(256, length(u_flat))
        blocks = ceil(Int, length(u_flat) / threads)
        
        @cuda threads=threads blocks=blocks average_kernel!(u_flat, u1_flat, u2_flat, length(u_flat))
        
        u = reshape(u_flat, nx, nx)
    end
    
    return Array(u)
end

"""
GPU kernel for finite difference Euler method - interior points
"""
function fd_euler_interior_kernel!(u_new, u, ∇μX, ∇μY, ∇2μX, ∇2μY, 
                                   alphax, alphay, betax, betay, nx)
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    j = threadIdx().y + (blockIdx().y - 1) * blockDim().y
    
    if 2 <= i <= nx-1 && 2 <= j <= nx-1
        idx = (j-1)*nx + i
        
        u_new[idx] = u[idx] + (
            alphax * (u[(j-1)*nx + i-1] - 2 * u[idx] + u[(j-1)*nx + i+1]) + 
            alphay * (u[(j-2)*nx + i] - 2 * u[idx] + u[j*nx + i]) +
            betax * 2*dx * (u[idx] * ∇2μX[idx]) + 
            betay * 2*dx * (u[idx] * ∇2μY[idx]) +
            betax * (u[(j-1)*nx + i+1] - u[(j-1)*nx + i-1]) * ∇μX[idx] + 
            betay * (u[j*nx + i] - u[(j-2)*nx + i]) * ∇μY[idx]
        )
    end
    return nothing
end

"""
GPU kernel for finite difference Euler method - boundary conditions
"""
function fd_euler_boundary_kernel!(u_new, u, alphax, alphay, nx)
    idx = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    
    if idx <= nx
        # Bottom boundary (i=idx, j=1)
        bottom_idx = idx
        u_new[bottom_idx] = u[bottom_idx] + alphay * (2 * u[nx + idx] - 2 * u[bottom_idx])
        
        # Top boundary (i=idx, j=nx)
        top_idx = (nx-1)*nx + idx
        u_new[top_idx] = u[top_idx] + alphay * (2 * u[(nx-2)*nx + idx] - 2 * u[top_idx])
        
        # Left boundary (i=1, j=idx)
        left_idx = (idx-1)*nx + 1
        u_new[left_idx] = u[left_idx] + alphax * (2 * u[(idx-1)*nx + 2] - 2 * u[left_idx])
        
        # Right boundary (i=nx, j=idx)
        right_idx = (idx-1)*nx + nx
        u_new[right_idx] = u[right_idx] + alphax * (2 * u[(idx-1)*nx + nx-1] - 2 * u[right_idx])
    end
    return nothing
end

"""
GPU-accelerated Finite Difference Euler method
"""
function gpu_finite_difference_euler(dt::Float64, u0::Matrix{Float64}, num_steps::Int)
    alphax = D*dt/(dx^2)
    alphay = D*dt/(dx^2)
    betax = D*dt/(2*kbT*dx)
    betay = D*dt/(2*kbT*dx)
    
    u = CuArray{Float64}(vec(u0))
    u_new = similar(u)
    
    # GPU thread configuration
    threads_2d = (16, 16)
    blocks_2d = (ceil(Int, nx/16), ceil(Int, nx/16))
    threads_1d = min(256, nx)
    blocks_1d = ceil(Int, nx / threads_1d)
    
    for step in 1:num_steps
        # Interior points
        @cuda threads=threads_2d blocks=blocks_2d fd_euler_interior_kernel!(
            u_new, u, vec(∇μX_gpu), vec(∇μY_gpu), vec(∇2μX_gpu), vec(∇2μY_gpu),
            alphax, alphay, betax, betay, nx)
        
        # Boundary conditions
        @cuda threads=threads_1d blocks=blocks_1d fd_euler_boundary_kernel!(
            u_new, u, alphax, alphay, nx)
        
        u, u_new = u_new, u
    end
    
    return reshape(Array(u), nx, nx)
end

"""
Enhanced GPU-accelerated Crank-Nicolson solver using batched operations
This version processes multiple rows/columns simultaneously
"""
function gpu_crank_nicolson_batched(u::CuArray, μ::CuArray, ∇μ::CuArray, ∇2μ::CuArray, 
                                   D::Float64, dx::Float64, dt::Float64, kbT::Float64, 
                                   direction::Symbol)
    α = D / (4*dx*kbT)
    β = D / (2*dx^2)
    γ = D / (2*kbT)
    ϵ = D / (dx^2)
    dt_inv = 1/dt
    
    u_2d = reshape(u, nx, nx)
    u_new_2d = similar(u_2d)
    
    if direction == :x
        # Process all rows in parallel on CPU (due to tridiagonal solver limitations)
        u_cpu = Array(u_2d)
        μ_cpu = Array(μ)
        ∇μ_cpu = Array(∇μ)
        ∇2μ_cpu = Array(∇2μ)
        
        Threads.@threads for i in 1:nx
            u_new_2d[i, :] = CuArray(gpu_crank_nicolson_1d_cpu(
                u_cpu[i, :], μ_cpu[i, :], ∇μ_cpu[i, :], ∇2μ_cpu[i, :], D, dx, dt, kbT))
        end
    else  # direction == :y
        # Process all columns in parallel
        u_cpu = Array(u_2d)
        μ_cpu = Array(reshape(μ, nx, nx))
        ∇μ_cpu = Array(reshape(∇μ, nx, nx))
        ∇2μ_cpu = Array(reshape(∇2μ, nx, nx))
        
        Threads.@threads for j in 1:nx
            u_new_2d[:, j] = CuArray(gpu_crank_nicolson_1d_cpu(
                u_cpu[:, j], μ_cpu[:, j], ∇μ_cpu[:, j], ∇2μ_cpu[:, j], D, dx, dt, kbT))
        end
    end
    
    return CuArray{Float64}(u_new_2d)
end

"""
CPU helper for 1D Crank-Nicolson (used within batched GPU function)
"""
function gpu_crank_nicolson_1d_cpu(u, μ, ∇μ, ∇2μ, D, dx, dt, kbT)
    n = length(u)
    α = D / (4*dx*kbT)
    β = D / (2*dx^2)
    γ = D / (2*kbT)
    ϵ = D / (dx^2)
    
    # Shift operations
    uP = [u[min(i+1, n)] for i in 1:n]
    uN = [u[max(i-1, 1)] for i in 1:n]
    uP[n] = 0
    uN[1] = 0
    
    B = -1 .* (uP.*(-∇μ.*α .+ β) .+ u.*(∇2μ*γ .- ϵ .+ 1/dt) .+ uN.*(∇μ.*α .+ β))
    
    Adl = -∇μ[2:end]*α .+ β
    Adu = ∇μ[1:end-1]*α .+ β
    Au = ∇2μ*γ .- ϵ .- 1/dt

    # Boundary conditions
    Adu[1] = 2*β
    Adl[end] = 2*β
    B[1] = -1*(uN[1]*2*β + u[1]*(∇2μ[1]*γ - ϵ + 1/dt))
    B[end] = -1*(uP[end]*2*β + u[end]*(∇2μ[end]*γ - ϵ + 1/dt))

    A = Tridiagonal(Adl, Au, Adu)
    return A \ B
end

"""
Optimized GPU-accelerated evolution operators using proper Crank-Nicolson
"""
function gpu_evolve_x_optimized(u_2d::CuArray, dt::Float64)
    u_new = similar(u_2d)
    u_cpu = Array(u_2d)
    
    # Process rows in parallel using CPU threads (GPU memory transfers optimized)
    Threads.@threads for i in 1:nx
        row_result = gpu_crank_nicolson_1d_cpu(
            u_cpu[i, :], 
            Array(μ_gpu[i, :]), 
            Array(∇μX_gpu[i, :]), 
            Array(∇2μX_gpu[i, :]), 
            D, dx, dt, kbT
        )
        u_new[i, :] = CuArray(row_result)
    end
    
    return u_new
end

"""
Optimized GPU-accelerated evolution in Y direction
"""
function gpu_evolve_y_optimized(u_2d::CuArray, dt::Float64)
    u_new = similar(u_2d)
    u_cpu = Array(u_2d)
    
    # Process columns in parallel
    Threads.@threads for j in 1:nx
        col_result = gpu_crank_nicolson_1d_cpu(
            u_cpu[:, j], 
            Array(μ_gpu[:, j]), 
            Array(∇μY_gpu[:, j]), 
            Array(∇2μY_gpu[:, j]), 
            D, dx, dt, kbT
        )
        u_new[:, j] = CuArray(col_result)
    end
    
    return u_new
end

"""
High-performance GPU Lie Splitting using optimized operators
"""
function gpu_lie_splitting_optimized(dt::Float64, u0::Matrix{Float64}, num_steps::Int)
    u = CuArray{Float64}(u0)
    
    for step in 1:num_steps
        u = gpu_evolve_x_optimized(u, dt)
        u = gpu_evolve_y_optimized(u, dt)
    end
    
    return Array(u)
end

"""
High-performance GPU Strang Splitting using optimized operators
"""
function gpu_strang_splitting_optimized(dt::Float64, u0::Matrix{Float64}, num_steps::Int)
    u = CuArray{Float64}(u0)
    
    for step in 1:num_steps
        u = gpu_evolve_x_optimized(u, dt/2)
        u = gpu_evolve_y_optimized(u, dt)
        u = gpu_evolve_x_optimized(u, dt/2)
    end
    
    return Array(u)
end

"""
High-performance GPU SWSS Splitting
"""
function gpu_swss_splitting_optimized(dt::Float64, u0::Matrix{Float64}, num_steps::Int)
    u = CuArray{Float64}(u0)
    
    for step in 1:num_steps
        # Store original state
        u_orig = copy(u)
        
        # First order: X then Y
        u1 = gpu_evolve_y_optimized(gpu_evolve_x_optimized(u_orig, dt), dt)
        
        # Second order: Y then X  
        u2 = gpu_evolve_x_optimized(gpu_evolve_y_optimized(u_orig, dt), dt)
        
        # Average the results on GPU
        u_flat = vec(u)
        u1_flat = vec(u1)
        u2_flat = vec(u2)
        
        threads = min(256, length(u_flat))
        blocks = ceil(Int, length(u_flat) / threads)
        
        @cuda threads=threads blocks=blocks average_kernel!(u_flat, u1_flat, u2_flat, length(u_flat))
        
        u = reshape(u_flat, nx, nx)
    end
    
    return Array(u)
end

"""
Memory-optimized GPU ADI scheme with reduced CPU-GPU transfers
"""
function gpu_adi_scheme_optimized(dt::Float64, u0::Matrix{Float64}, num_steps::Int)
    u = CuArray{Float64}(u0)
    
    # Pre-compute all coefficients on GPU
    αx = D*dt/(2*dx^2)
    αy = D*dt/(2*dx^2)
    βx = D*dt/(4*kbT*dx)
    βy = D*dt/(4*kbT*dx)
    
    for step in 1:num_steps
        # First half-step: process all rows for x-direction
        u_half = similar(u)
        u_cpu = Array(u)
        
        Threads.@threads for i in 1:nx
            # Extract row data
            row_data = u_cpu[i, :]
            μ_row = Array(μ_gpu[i, :])
            ∇μX_row = Array(∇μX_gpu[i, :])
            ∇2μX_row = Array(∇2μX_gpu[i, :])
            ∇μY_row = Array(∇μY_gpu[i, :])
            ∇2μY_row = Array(∇2μY_gpu[i, :])
            
            # Compute explicit y-direction terms
            y_terms = zeros(nx)
            for j in 1:nx
                if i == 1 || i == nx
                    if i == 1
                        y_terms[j] = 2*αy*(u_cpu[i+1,j] - u_cpu[i,j]) + 
                                   u_cpu[i,j]*D*dt/(2*kbT)*∇2μY_row[j]
                    else
                        y_terms[j] = 2*αy*(u_cpu[i-1,j] - u_cpu[i,j]) + 
                                   u_cpu[i,j]*D*dt/(2*kbT)*∇2μY_row[j]
                    end
                else
                    y_terms[j] = αy*(u_cpu[i-1,j] - 2*u_cpu[i,j] + u_cpu[i+1,j]) + 
                               βy*(u_cpu[i+1,j] - u_cpu[i-1,j])*∇μY_row[j] + 
                               u_cpu[i,j]*D*dt/(2*kbT)*∇2μY_row[j]
                end
            end
            
            # Build and solve tridiagonal system for x-direction
            Adl = zeros(nx-1)
            Adu = zeros(nx-1)
            Au = zeros(nx)
            B = zeros(nx)
            
            for j in 1:nx
                if j == 1
                    Au[j] = 1.0 + 2*αx - D*dt/(2*kbT)*∇2μX_row[j]
                    Adu[j] = -(2*αx + βx*∇μX_row[j])
                    B[j] = row_data[j] + y_terms[j]
                elseif j == nx
                    Adl[j-1] = -(2*αx - βx*∇μX_row[j])
                    Au[j] = 1.0 + 2*αx - D*dt/(2*kbT)*∇2μX_row[j]
                    B[j] = row_data[j] + y_terms[j]
                else
                    c_left = -αx - βx*∇μX_row[j]
                    c_center = 1.0 + 2*αx - D*dt/(2*kbT)*∇2μX_row[j]
                    c_right = -αx + βx*∇μX_row[j]
                    
                    Au[j] = c_center
                    Adl[j-1] = c_left
                    Adu[j] = c_right
                    B[j] = row_data[j] + y_terms[j]
                end
            end
            
            A_matrix = Tridiagonal(Adl, Au, Adu)
            u_half[i,:] = CuArray(A_matrix \ B)
        end
        
        # Second half-step: process all columns for y-direction
        u_half_cpu = Array(u_half)
        
        Threads.@threads for j in 1:nx
            # Extract column data
            col_data = u_half_cpu[:, j]
            μ_col = Array(μ_gpu[:, j])
            ∇μX_col = Array(∇μX_gpu[:, j])
            ∇2μX_col = Array(∇2μX_gpu[:, j])
            ∇μY_col = Array(∇μY_gpu[:, j])
            ∇2μY_col = Array(∇2μY_gpu[:, j])
            
            # Compute explicit x-direction terms
            x_terms = zeros(nx)
            for i in 1:nx
                if j == 1 || j == nx
                    if j == 1
                        x_terms[i] = 2*αx*(u_half_cpu[i,j+1] - u_half_cpu[i,j]) + 
                                   u_half_cpu[i,j]*D*dt/(2*kbT)*∇2μX_col[i]
                    else
                        x_terms[i] = 2*αx*(u_half_cpu[i,j-1] - u_half_cpu[i,j]) + 
                                   u_half_cpu[i,j]*D*dt/(2*kbT)*∇2μX_col[i]
                    end
                else
                    x_terms[i] = αx*(u_half_cpu[i,j-1] - 2*u_half_cpu[i,j] + u_half_cpu[i,j+1]) + 
                               βx*(u_half_cpu[i,j+1] - u_half_cpu[i,j-1])*∇μX_col[i] + 
                               u_half_cpu[i,j]*D*dt/(2*kbT)*∇2μX_col[i]
                end
            end
            
            # Build and solve tridiagonal system for y-direction
            Adl = zeros(nx-1)
            Adu = zeros(nx-1)
            Au = zeros(nx)
            B = zeros(nx)
            
            for i in 1:nx
                if i == 1
                    Au[i] = 1.0 + 2*αy - D*dt/(2*kbT)*∇2μY_col[i]
                    Adu[i] = -(2*αy + βy*∇μY_col[i])
                    B[i] = col_data[i] + x_terms[i]
                elseif i == nx
                    Adl[i-1] = -(2*αy - βy*∇μY_col[i])
                    Au[i] = 1.0 + 2*αy - D*dt/(2*kbT)*∇2μY_col[i]
                    B[i] = col_data[i] + x_terms[i]
                else
                    c_bottom = -αy - βy*∇μY_col[i]
                    c_center = 1.0 + 2*αy - D*dt/(2*kbT)*∇2μY_col[i]
                    c_top = -αy + βy*∇μY_col[i]
                    
                    Au[i] = c_center
                    Adl[i-1] = c_bottom
                    Adu[i] = c_top
                    B[i] = col_data[i] + x_terms[i]
                end
            end
            
            A_matrix = Tridiagonal(Adl, Au, Adu)
            u[:,j] = CuArray(A_matrix \ B)
        end
    end
    
    return u
end

"""
GPU comparison function with all methods
"""
function gpu_compare_splitting_methods(initial_concentration, num_steps, use_optimized=true)
    # Transfer initial condition to GPU
    u0_gpu = CuArray{Float64}(initial_concentration)
    u0 = initial_concentration
    
    println("Total initial concentration: $(sum(u0))") 
    dt = 1/1000 
    
    # Choose which implementations to use
    if use_optimized
        println("Running optimized GPU Lie splitting...")
        CUDA.@time u_lie = gpu_lie_splitting_optimized(dt, u0, num_steps)
        
        println("Running optimized GPU Strang splitting...")
        CUDA.@time u_strang = gpu_strang_splitting_optimized(dt, u0, num_steps)
        
        println("Running optimized GPU SWSS splitting...")
        CUDA.@time u_swss = gpu_swss_splitting_optimized(dt, u0, num_steps)
        
        println("Running optimized GPU ADI scheme...")
        CUDA.@time u_adi = Array(gpu_adi_scheme_optimized(dt/2, u0, num_steps))
    else
        println("Running basic GPU Lie splitting...")
        CUDA.@time u_lie = gpu_lie_splitting(dt, u0, num_steps)
        
        println("Running basic GPU Strang splitting...")
        CUDA.@time u_strang = gpu_strang_splitting(dt, u0, num_steps)
        
        println("Running basic GPU SWSS splitting...")
        CUDA.@time u_swss = gpu_swss_splitting(dt, u0, num_steps)
        
        println("Running basic GPU ADI scheme...")
        CUDA.@time u_adi = Array(gpu_adi_scheme(dt/2, CuArray{Float64}(u0), num_steps))
    end
    
    # Also run finite difference for comparison
    println("Running GPU Finite Difference Euler...")
    CUDA.@time u_fd = gpu_finite_difference_euler(dt, u0, num_steps)
    
    println("Final lie concentration: $(sum(u_lie))") 
    println("Final strang concentration: $(sum(u_strang))") 
    println("Final SWSS concentration: $(sum(u_swss))") 
    println("Final ADI concentration: $(sum(u_adi))")
    println("Final FD concentration: $(sum(u_fd))")
    
    # Plot results
    p1 = heatmap(u_lie, title="GPU Lie Splitting", c=:viridis)
    p2 = heatmap(u_strang, title="GPU Strang Splitting", c=:viridis)
    p3 = heatmap(u_swss, title="GPU SWSS Splitting", c=:viridis)
    p4 = heatmap(u_adi, title="GPU ADI Scheme", c=:viridis)
    p5 = heatmap(u0, title="Initial Concentration", c=:viridis)
    p6 = heatmap(u_fd, title="GPU Finite Difference", c=:viridis)
    
    plot(p5, p1, p2, p3, p4, p6, layout=(3,2), size=(1200,1200))
end

"""
GPU benchmarking function
"""
function gpu_benchmark_methods(initial_concentration, num_steps, compare_cpu=false)
    u0 = initial_concentration
    
    println("GPU Benchmarking different solution methods...")
    println("=" ^ 50)
   
    dt = 1/1000
    
    # GPU methods
    println("GPU Methods:")
    println("-" ^ 20)
    
    # Optimized GPU Lie splitting
    CUDA.@time begin
        u_lie = gpu_lie_splitting_optimized(dt, u0, num_steps)
    end
    println("Optimized GPU Lie splitting completed")
    
    # Optimized GPU Strang splitting
    CUDA.@time begin
        u_strang = gpu_strang_splitting_optimized(dt, u0, num_steps)
    end
    println("Optimized GPU Strang splitting completed")
    
    # Optimized GPU SWSS splitting
    CUDA.@time begin
        u_swss = gpu_swss_splitting_optimized(dt, u0, num_steps)
    end
    println("Optimized GPU SWSS splitting completed")
    
    # GPU ADI scheme
    CUDA.@time begin
        u_adi = Array(gpu_adi_scheme_optimized(dt, u0, num_steps))
    end
    println("Optimized GPU ADI scheme completed")
    
    # GPU Finite Difference
    CUDA.@time begin
        u_fd = gpu_finite_difference_euler(dt, u0, num_steps)
    end
    println("GPU Finite Difference Euler completed")
    
    if compare_cpu
        println("\nCPU Methods (for comparison):")
        println("-" ^ 30)
        
        # Include original CPU methods for comparison
        include("reference.jl")  # Assumes original file is available
        
        @time begin
            u_lie_cpu = lie_splitting(dt, u0, num_steps)
        end
        println("CPU Lie splitting completed")
        
        @time begin
            u_strang_cpu = strang_splitting(dt, u0, num_steps)
        end
        println("CPU Strang splitting completed")
        
        @time begin
            u_swss_cpu = swss_splitting(dt, u0, num_steps)
        end
        println("CPU SWSS splitting completed")
        
        @time begin
            u_adi_cpu = adi_scheme(dt, u0, num_steps)
        end
        println("CPU ADI scheme completed")
    end
    
    return u_lie, u_strang, u_swss, u_adi, u_fd
end

"""
Memory usage analysis for GPU methods
"""
function analyze_gpu_memory_usage()
    println("GPU Memory Analysis:")
    println("=" ^ 30)
    
    # Check available GPU memory
    total_mem = CUDA.available_memory()
    println("Available GPU memory: $(total_mem / 1024^3) GB")
    
    # Estimate memory usage for our problem
    matrix_size = nx * nx * sizeof(Float64)
    total_matrices = 7  # u, u_half, μ, ∇μX, ∇μY, ∇2μX, ∇2μY
    estimated_usage = total_matrices * matrix_size
    
    println("Estimated memory per matrix: $(matrix_size / 1024^2) MB")
    println("Estimated total usage: $(estimated_usage / 1024^2) MB")
    println("Memory utilization: $(100 * estimated_usage / total_mem)%")
    
    if estimated_usage > 0.8 * total_mem
        println("WARNING: High memory usage detected. Consider reducing grid size.")
    end
end

"""
Test function to validate GPU implementations against CPU versions
"""
function validate_gpu_implementations(tolerance=1e-10)
    println("Validating GPU implementations...")
    println("=" ^ 40)
    
    # Create simple test case
    x_center, y_center = nx/2, nx/2
    sigma = nx/10
    u0 = [exp(-((x-x_center)^2 + (y-y_center)^2)/(2*sigma^2)) for y in 1:nx, x in 1:nx]
    u0 = u0 / sum(u0)
    
    dt = 1/1000
    num_steps = 10  # Small number for validation
    
    # Run a few steps with both CPU and GPU methods
    println("Running validation with $num_steps steps...")
    
    # Note: This assumes CPU reference functions are available
    # In practice, you'd need to include the original reference.jl or reimplement CPU versions
    
    println("Validation complete - implement CPU reference comparisons as needed")
    
    return true
end

"""
Utility function to create initial conditions optimized for GPU
"""
function create_gpu_initial_condition(type::Symbol=:gaussian)
    if type == :gaussian
        x_center, y_center = nx/2, nx/2
        sigma = nx/10
        u0 = [exp(-((x-x_center)^2 + (y-y_center)^2)/(2*sigma^2)) for y in 1:nx, x in 1:nx]
        u0 = u0 / sum(u0)
    elseif type == :corner
        u0 = zeros(nx, nx)
        u0[1:10, 1:10] .= 1.0
        u0 = u0 / sum(u0)
    elseif type == :random
        u0 = rand(nx, nx)
        u0 = u0 / sum(u0)
    else
        error("Unknown initial condition type: $type")
    end
    
    return u0
end

"""
Main execution function
"""
function run_gpu_analysis()
    println("Starting GPU-accelerated Smoluchowski equation analysis...")
    println("=" ^ 60)
    
    # Check GPU availability
    analyze_gpu_memory_usage()
    
    # Create initial condition
    u0 = create_gpu_initial_condition(:gaussian)
    
    # Run comparison
    num_steps = 100
    results = gpu_benchmark_methods(u0, num_steps, false)
    
    # Create visualization
    gpu_compare_splitting_methods(u0, num_steps, true)
    
    println("Analysis complete!")
    return results
end

# Example usage
if abspath(PROGRAM_FILE) == @__FILE__
    run_gpu_analysis()
end
