using LinearAlgebra
using SparseArrays
using Plots
using DelimitedFiles
include("../../methods.jl")  # Import optimized methods

const D = 0.01
const nx = 100
const dx = 1/1000
const kbT = 0.65

# Grid generation
const x_range = range(-1, 1, length=nx)
const y_range = range(-1, 1, length=nx)

# ============================================================================
# SOURCE TERM EXTENSIONS FOR OPTIMIZED METHODS
# ============================================================================

"""
Extended temporary arrays for ADI with source terms
"""
struct ADISourceTempArrays{T}
    base::ADITempArrays{T}
    source_term::Matrix{T}
end

function ADISourceTempArrays(::Type{T}, nx::Int, ny::Int) where T
    base = ADITempArrays(T, nx, ny)
    source_term = Matrix{T}(undef, nx, ny)
    return ADISourceTempArrays(base, source_term)
end

"""
ADI x-direction sweep with source terms - optimized
"""
function x_direction_sweep_with_source!(u_half, u, αx, αy, βx, βy, dt, ∇μX, ∇μY, ∇2μX, ∇2μY, D, kbT, source_term, temp_arrays)
    nx, ny = size(u)
    
    # Reuse pre-allocated arrays from base
    Adl = temp_arrays.base.Adl
    Adu = temp_arrays.base.Adu
    Au = temp_arrays.base.Au
    B = temp_arrays.base.B
    
    @inbounds for i in 1:nx
        # Resize views to ny elements
        Adl_view = view(Adl, 1:ny-1)
        Adu_view = view(Adu, 1:ny-1)
        Au_view = view(Au, 1:ny)
        B_view = view(B, 1:ny)
        
        fill!(Adl_view, 0)
        fill!(Adu_view, 0)
        fill!(Au_view, 0)
        fill!(B_view, 0)
        
        for j in 1:ny
            if j == 1
                # Left boundary: von Neumann condition du/dx = 0
                if i == 1 || i == nx
                    if i == 1
                        y_term = 2*αy*(u[i+1,j] - u[i,j]) + u[i,j]*D*dt/(2*kbT)*∇2μY[i,j]
                    else
                        y_term = 2*αy*(u[i-1,j] - u[i,j]) + u[i,j]*D*dt/(2*kbT)*∇2μY[i,j]
                    end
                else
                    y_term = αy*(u[i-1,j] - 2*u[i,j] + u[i+1,j]) +
                            βy*(u[i+1,j] - u[i-1,j])*∇μY[i,j] +
                            u[i,j]*D*dt/(2*kbT)*∇2μY[i,j]
                end
                Au_view[j] = 1.0 + 2*αx - D*dt/(2*kbT)*∇2μX[i,j]
                if j < ny
                    Adu_view[j] = -(2*αx + βx*∇μX[i,j])
                end
                B_view[j] = u[i,j] + y_term + dt/2 * source_term[i,j]
            elseif j == ny
                # Right boundary: von Neumann condition du/dx = 0
                if i == 1 || i == nx
                    if i == 1
                        y_term = 2*αy*(u[i+1,j] - u[i,j]) + u[i,j]*D*dt/(2*kbT)*∇2μY[i,j]
                    else
                        y_term = 2*αy*(u[i-1,j] - u[i,j]) + u[i,j]*D*dt/(2*kbT)*∇2μY[i,j]
                    end
                else
                    y_term = αy*(u[i-1,j] - 2*u[i,j] + u[i+1,j]) +
                            βy*(u[i+1,j] - u[i-1,j])*∇μY[i,j] +
                            u[i,j]*D*dt/(2*kbT)*∇2μY[i,j]
                end
                Adl_view[j-1] = -(2*αx - βx*∇μX[i,j])
                Au_view[j] = 1.0 + 2*αx - D*dt/(2*kbT)*∇2μX[i,j]
                B_view[j] = u[i,j] + y_term + dt/2 * source_term[i,j]
            else
                # Interior points
                if i == 1 || i == nx
                    if i == 1
                        y_term = 2*αy*(u[i+1,j] - u[i,j]) + u[i,j]*D*dt/(2*kbT)*∇2μY[i,j]
                    else
                        y_term = 2*αy*(u[i-1,j] - u[i,j]) + u[i,j]*D*dt/(2*kbT)*∇2μY[i,j]
                    end
                else
                    y_term = αy*(u[i-1,j] - 2*u[i,j] + u[i+1,j]) +
                            βy*(u[i+1,j] - u[i-1,j])*∇μY[i,j] +
                            u[i,j]*D*dt/(2*kbT)*∇2μY[i,j]
                end
                Au_view[j] = 1.0 + 2*αx - D*dt/(2*kbT)*∇2μX[i,j]
                Adl_view[j-1] = -αx - βx*∇μX[i,j]
                Adu_view[j] = -αx + βx*∇μX[i,j]
                B_view[j] = u[i,j] + y_term + dt/2 * source_term[i,j]
            end
        end
        
        # Solve and store result
        ldiv!(view(u_half, i, :), Tridiagonal(Adl_view, Au_view, Adu_view), B_view)
    end
    return u_half
end

"""
ADI y-direction sweep with source terms - optimized
"""
function y_direction_sweep_with_source!(u, u_half, αx, αy, βx, βy, dt, ∇μX, ∇μY, ∇2μX, ∇2μY, D, kbT, source_term, temp_arrays)
    nx, ny = size(u_half)
    
    # Reuse pre-allocated arrays from base
    Adl = temp_arrays.base.Adl
    Adu = temp_arrays.base.Adu
    Au = temp_arrays.base.Au
    B = temp_arrays.base.B
    
    @inbounds for j in 1:ny
        # Resize views to nx elements
        Adl_view = view(Adl, 1:nx-1)
        Adu_view = view(Adu, 1:nx-1)
        Au_view = view(Au, 1:nx)
        B_view = view(B, 1:nx)
        
        fill!(Adl_view, 0)
        fill!(Adu_view, 0)
        fill!(Au_view, 0)
        fill!(B_view, 0)
        
        for i in 1:nx
            if i == 1
                # Bottom boundary: von Neumann condition du/dy = 0
                if j == 1 || j == ny
                    if j == 1
                        x_term = 2*αx*(u_half[i,j+1] - u_half[i,j]) + u_half[i,j]*D*dt/(2*kbT)*∇2μX[i,j]
                    else
                        x_term = 2*αx*(u_half[i,j-1] - u_half[i,j]) + u_half[i,j]*D*dt/(2*kbT)*∇2μX[i,j]
                    end
                else
                    x_term = αx*(u_half[i,j-1] - 2*u_half[i,j] + u_half[i,j+1]) +
                            βx*(u_half[i,j+1] - u_half[i,j-1])*∇μX[i,j] +
                            u_half[i,j]*D*dt/(2*kbT)*∇2μX[i,j]
                end
                Au_view[i] = 1.0 + 2*αy - D*dt/(2*kbT)*∇2μY[i,j]
                if i < nx
                    Adu_view[i] = -(2*αy + βy*∇μY[i,j])
                end
                B_view[i] = u_half[i,j] + x_term + dt/2 * source_term[i,j]
            elseif i == nx
                # Top boundary: von Neumann condition du/dy = 0
                if j == 1 || j == ny
                    if j == 1
                        x_term = 2*αx*(u_half[i,j+1] - u_half[i,j]) + u_half[i,j]*D*dt/(2*kbT)*∇2μX[i,j]
                    else
                        x_term = 2*αx*(u_half[i,j-1] - u_half[i,j]) + u_half[i,j]*D*dt/(2*kbT)*∇2μX[i,j]
                    end
                else
                    x_term = αx*(u_half[i,j-1] - 2*u_half[i,j] + u_half[i,j+1]) +
                            βx*(u_half[i,j+1] - u_half[i,j-1])*∇μX[i,j] +
                            u_half[i,j]*D*dt/(2*kbT)*∇2μX[i,j]
                end
                Adl_view[i-1] = -(2*αy - βy*∇μY[i,j])
                Au_view[i] = 1.0 + 2*αy - D*dt/(2*kbT)*∇2μY[i,j]
                B_view[i] = u_half[i,j] + x_term + dt/2 * source_term[i,j]
            else
                # Interior points
                if j == 1 || j == ny
                    if j == 1
                        x_term = 2*αx*(u_half[i,j+1] - u_half[i,j]) + u_half[i,j]*D*dt/(2*kbT)*∇2μX[i,j]
                    else
                        x_term = 2*αx*(u_half[i,j-1] - u_half[i,j]) + u_half[i,j]*D*dt/(2*kbT)*∇2μX[i,j]
                    end
                else
                    x_term = αx*(u_half[i,j-1] - 2*u_half[i,j] + u_half[i,j+1]) +
                            βx*(u_half[i,j+1] - u_half[i,j-1])*∇μX[i,j] +
                            u_half[i,j]*D*dt/(2*kbT)*∇2μX[i,j]
                end
                Au_view[i] = 1.0 + 2*αy - D*dt/(2*kbT)*∇2μY[i,j]
                Adl_view[i-1] = -αy - βy*∇μY[i,j]
                Adu_view[i] = -αy + βy*∇μY[i,j]
                B_view[i] = u_half[i,j] + x_term + dt/2 * source_term[i,j]
            end
        end
        
        # Solve and store result
        ldiv!(view(u, :, j), Tridiagonal(Adl_view, Au_view, Adu_view), B_view)
    end
    return u
end

"""
Direct ADI scheme with integrated source terms - optimized
"""
function adi_scheme_direct_source!(u, dt::Float64, num_steps, μ, ∇μX, ∇μY, ∇2μX, ∇2μY, source_term, temp_arrays)
    # Pre-compute coefficients
    αx = D*dt/(2*dx^2)
    αy = D*dt/(2*dx^2)
    βx = D*dt/(4*kbT*dx)
    βy = D*dt/(4*kbT*dx)
    
    # Main time-stepping loop
    for step in 1:num_steps
        current_time = (step - 1) * dt
        
        # Evaluate source at mid-time for better accuracy
        temp_arrays.source_term .= source_term(current_time + dt/2)
        
        # First sweep: x-direction (implicit in x, explicit in y) with source
        x_direction_sweep_with_source!(temp_arrays.base.u_half, u, αx, αy, βx, βy, dt, ∇μX, ∇μY, ∇2μX, ∇2μY, D, kbT, temp_arrays.source_term, temp_arrays)
        
        # Second sweep: y-direction (implicit in y, explicit in x) with source  
        y_direction_sweep_with_source!(u, temp_arrays.base.u_half, αx, αy, βx, βy, dt, ∇μX, ∇μY, ∇2μX, ∇2μY, D, kbT, temp_arrays.source_term, temp_arrays)
    end
    
    return u
end

"""
Public API for ADI scheme with direct source integration
"""
function adi_scheme_direct_source(dt::Float64, u0, num_steps, μ, ∇μX, ∇μY, ∇2μX, ∇2μY, source_term)
    nx, ny = size(u0)
    temp_arrays = ADISourceTempArrays(eltype(u0), nx, ny)
    u = copy(u0)
    return adi_scheme_direct_source!(u, dt, num_steps, μ, ∇μX, ∇μY, ∇2μX, ∇2μY, source_term, temp_arrays)
end

"""
Enhanced forward Euler with optimized boundary conditions and vectorization
"""
function fd_with_source!(u, u_new, dt::Float64, num_steps, μ, ∇μX, ∇μY, ∇2μX, ∇2μY, source_func)
    nx, ny = size(u)
    alphax = D*dt/(dx^2)
    alphay = D*dt/(dx^2)
    betax = D*dt/(2*kbT*dx)
    betay = D*dt/(2*kbT*dx)
    
    for step in 1:num_steps
        current_time = (step - 1) * dt
        
        # Apply source term first (operator splitting approach)
        source_term = source_func(current_time)
        @. u = u + dt * source_term
        
        # Interior points - fully vectorized
        @inbounds for i in 2:nx-1, j in 2:ny-1
            u_new[i,j] = u[i,j] + (
                alphax * (u[i-1,j] - 2 * u[i,j] + u[i+1,j]) +
                alphay * (u[i,j-1] - 2 * u[i,j] + u[i,j+1]) +
                betax * 2*dx * (u[i,j] * ∇2μX[i,j]) +
                betay * 2*dx * (u[i,j] * ∇2μY[i,j]) +
                betax * (u[i+1,j] - u[i-1,j]) * ∇μX[i,j] +
                betay * (u[i,j+1] - u[i,j-1]) * ∇μY[i,j]
            )
        end
        
        # von Neumann boundary conditions - vectorized
        @inbounds for i in 1:nx
            # Bottom and top boundaries
            u_new[i,1] = u[i,1] + alphay * (2 * u[i,2] - 2 * u[i,1])
            u_new[i,ny] = u[i,ny] + alphay * (2 * u[i,ny-1] - 2 * u[i,ny])
        end
        @inbounds for j in 1:ny
            # Left and right boundaries  
            u_new[1,j] = u[1,j] + alphax * (2 * u[2,j] - 2 * u[1,j])
            u_new[nx,j] = u[nx,j] + alphax * (2 * u[nx-1,j] - 2 * u[nx,j])
        end
        
        # Swap arrays instead of copying
        u, u_new = u_new, u
    end
    
    # Return the correct array based on parity of steps
    return num_steps % 2 == 0 ? u : u_new
end

"""
Public API for forward Euler with source terms
"""
function fd_with_source(dt::Float64, u0, num_steps, μ, ∇μX, ∇μY, ∇2μX, ∇2μY, source_func)
    u = copy(u0)
    u_new = similar(u)
    return fd_with_source!(u, u_new, dt, num_steps, μ, ∇μX, ∇μY, ∇2μX, ∇2μY, source_func)
end

"""
Extended splitting methods with source terms using optimized base methods
"""
struct SplittingSourceTempArrays{T}
    base::TempArrays2D{T}
    source_temp::Matrix{T}
end

function SplittingSourceTempArrays(::Type{T}, nx::Int, ny::Int) where T
    base = TempArrays2D(T, nx, ny)
    source_temp = Matrix{T}(undef, nx, ny)
    return SplittingSourceTempArrays(base, source_temp)
end

"""
Lie Splitting with source term - optimized version
"""
function lie_splitting_with_source!(u, dt::Float64, num_steps, μ, ∇μX, ∇μY, ∇2μX, ∇2μY, source_func, temp_arrays)
    u_temp = temp_arrays.base.u_half
    
    for step in 1:num_steps
        current_time = (step - 1) * dt
        
        evolve_x!(u_temp, u, μ, ∇μX, ∇2μX, D, dx, dt/2, kbT, temp_arrays.base)
        evolve_y!(u, u_temp, μ, ∇μY, ∇2μY, D, dx, dt/2, kbT, temp_arrays.base)
        
        # Apply source term
        temp_arrays.source_temp .= source_func(current_time + dt/2)
        @. u = u + dt * temp_arrays.source_temp
        
        evolve_x!(u_temp, u, μ, ∇μX, ∇2μX, D, dx, dt/2, kbT, temp_arrays.base)
        evolve_y!(u, u_temp, μ, ∇μY, ∇2μY, D, dx, dt/2, kbT, temp_arrays.base)
    end
    return u
end

"""
Strang Splitting with source term - optimized version
"""
function strang_splitting_with_source!(u, dt::Float64, num_steps, μ, ∇μX, ∇μY, ∇2μX, ∇2μY, source_func, temp_arrays)
    u_temp = temp_arrays.base.u_half
    
    for step in 1:num_steps
        current_time = (step - 1) * dt
        
        evolve_x!(u_temp, u, μ, ∇μX, ∇2μX, D, dx, dt/4, kbT, temp_arrays.base)
        evolve_y!(u, u_temp, μ, ∇μY, ∇2μY, D, dx, dt/2, kbT, temp_arrays.base)
        evolve_x!(u_temp, u, μ, ∇μX, ∇2μX, D, dx, dt/4, kbT, temp_arrays.base)
        
        # Apply source term
        temp_arrays.source_temp .= source_func(current_time + dt/2)
        @. u_temp = u_temp + dt * temp_arrays.source_temp
        
        evolve_x!(u, u_temp, μ, ∇μX, ∇2μX, D, dx, dt/4, kbT, temp_arrays.base)
        evolve_y!(u_temp, u, μ, ∇μY, ∇2μY, D, dx, dt/2, kbT, temp_arrays.base)
        evolve_x!(u, u_temp, μ, ∇μX, ∇2μX, D, dx, dt/4, kbT, temp_arrays.base)
    end
    return u
end

"""
Public APIs for splitting methods with source terms
"""
function lie_splitting_with_source(dt::Float64, u0, num_steps, μ, ∇μX, ∇μY, ∇2μX, ∇2μY, source_func)
    nx, ny = size(u0)
    temp_arrays = SplittingSourceTempArrays(eltype(u0), nx, ny)
    u = copy(u0)
    return lie_splitting_with_source!(u, dt, num_steps, μ, ∇μX, ∇μY, ∇2μX, ∇2μY, source_func, temp_arrays)
end

function strang_splitting_with_source(dt::Float64, u0, num_steps, μ, ∇μX, ∇μY, ∇2μX, ∇2μY, source_func)
    nx, ny = size(u0)
    temp_arrays = SplittingSourceTempArrays(eltype(u0), nx, ny)
    u = copy(u0)
    return strang_splitting_with_source!(u, dt, num_steps, μ, ∇μX, ∇μY, ∇2μX, ∇2μY, source_func, temp_arrays)
end

# ============================================================================
# METHOD OF MANUFACTURED SOLUTIONS (MMS)
# ============================================================================

"""
Generate potential field 1: f(x,y) = x²y + e^x sin(y)
"""
function generate_potential_1(nx)
    μ = zeros(nx, nx)
    ∇μX = zeros(nx, nx)
    ∇μY = zeros(nx, nx)
    ∇2μX = zeros(nx, nx)
    ∇2μY = zeros(nx, nx)

    for i in 1:nx
        for j in 1:nx
            x = x_range[i]
            y = y_range[j]

            μ[i,j] = x^2 * y + exp(x) * sin(y)
            ∇μX[i,j] = 2*x*y + exp(x)*sin(y)
            ∇μY[i,j] = x^2 + exp(x)*cos(y)
            ∇2μX[i,j] = 2*y + exp(x)*sin(y)
            ∇2μY[i,j] = -exp(x)*sin(y)
        end
    end

    return μ, ∇μX, ∇μY, ∇2μX, ∇2μY
end

"""
Generate potential field 2: f(x,y) = xy² + ln(x² + 1) + y³
"""
function generate_potential_2(nx)
    μ = zeros(nx, nx)
    ∇μX = zeros(nx, nx)
    ∇μY = zeros(nx, nx)
    ∇2μX = zeros(nx, nx)
    ∇2μY = zeros(nx, nx)

    for i in 1:nx
        for j in 1:nx
            x = x_range[i]
            y = y_range[j]

            μ[i,j] = x*y^2 + log(x^2 + 1) + y^3
            ∇μX[i,j] = y^2 + 2*x/(x^2 + 1)
            ∇μY[i,j] = 2*x*y + 3*y^2
            ∇2μX[i,j] = 2*(1 - x^2)/(x^2 + 1)^2
            ∇2μY[i,j] = 2*x + 6*y
        end
    end

    return μ, ∇μX, ∇μY, ∇2μX, ∇2μY
end

"""
MMS manufactured solution 1: u(x,y,t) = (1 + 0.1*cos(x)*cos(y)) * exp(-0.5*t)
This has zero spatial derivatives at corners, satisfying von Neumann conditions.
"""
function manufactured_solution_1(t)
    u = zeros(nx, nx)
    for i in 1:nx
        for j in 1:nx
            x = x_range[i]
            y = y_range[j]
            u[i,j] = (1 + 0.1*cos(x)*cos(y)) * exp(-0.5*t)
        end
    end
    return u
end

"""
Time derivative of manufactured solution 1
"""
function manufactured_solution_1_dt(t)
    u_dt = zeros(nx, nx)
    for i in 1:nx
        for j in 1:nx
            x = x_range[i]
            y = y_range[j]
            u_dt[i,j] = -0.5 * (1 + 0.1*cos(x)*cos(y)) * exp(-0.5*t)
        end
    end
    return u_dt
end

"""
Compute source term for MMS with potential 1
"""
function compute_source_term_1(t, μ, ∇μX, ∇μY, ∇2μX, ∇2μY)
    u = manufactured_solution_1(t)
    u_dt = manufactured_solution_1_dt(t)

    source_term = zeros(nx, nx)

    for i in 1:nx
        for j in 1:nx
            x = x_range[i]
            y = y_range[j]

            # Spatial derivatives of u
            u_x = -0.1 * sin(x) * cos(y) * exp(-0.5*t)
            u_y = -0.1 * cos(x) * sin(y) * exp(-0.5*t)
            u_xx = -0.1 * cos(x) * cos(y) * exp(-0.5*t)
            u_yy = -0.1 * cos(x) * cos(y) * exp(-0.5*t)

            # Smoluchowski operator: L[u] = D*∇²u + (D/kbT)*(∇u·∇μ + u*∇²μ)
            laplacian_u = u_xx + u_yy
            grad_dot = u_x * ∇μX[i,j] + u_y * ∇μY[i,j]
            potential_term = u[i,j] * (∇2μX[i,j] + ∇2μY[i,j])

            Lu = D * laplacian_u + (D/kbT) * (grad_dot + potential_term)

            # Source term: S = ∂u/∂t - L[u]
            source_term[i,j] = u_dt[i,j] - Lu
        end
    end

    return source_term
end

"""
MMS manufactured solution 2: u(x,y,t) = (1 + 0.1*sin(π*x/2)*sin(π*y/2)) * exp(-0.3*t)
This has zero spatial derivatives at corners x,y = ±1.
"""
function manufactured_solution_2(t)
    u = zeros(nx, nx)
    for i in 1:nx
        for j in 1:nx
            x = x_range[i]
            y = y_range[j]
            u[i,j] = (1 + 0.1*sin(π*x/2)*sin(π*y/2)) * exp(-0.3*t)
        end
    end
    return u
end

"""
Time derivative of manufactured solution 2
"""
function manufactured_solution_2_dt(t)
    u_dt = zeros(nx, nx)
    for i in 1:nx
        for j in 1:nx
            x = x_range[i]
            y = y_range[j]
            u_dt[i,j] = -0.3 * (1 + 0.1*sin(π*x/2)*sin(π*y/2)) * exp(-0.3*t)
        end
    end
    return u_dt
end

"""
Compute source term for MMS with potential 2
"""
function compute_source_term_2(t, μ, ∇μX, ∇μY, ∇2μX, ∇2μY)
    u = manufactured_solution_2(t)
    u_dt = manufactured_solution_2_dt(t)

    source_term = zeros(nx, nx)

    for i in 1:nx
        for j in 1:nx
            x = x_range[i]
            y = y_range[j]

            # Spatial derivatives of u
            u_x = 0.1 * (π/2) * cos(π*x/2) * sin(π*y/2) * exp(-0.3*t)
            u_y = 0.1 * (π/2) * sin(π*x/2) * cos(π*y/2) * exp(-0.3*t)
            u_xx = -0.1 * (π/2)^2 * sin(π*x/2) * sin(π*y/2) * exp(-0.3*t)
            u_yy = -0.1 * (π/2)^2 * sin(π*x/2) * sin(π*y/2) * exp(-0.3*t)

            # Smoluchowski operator
            laplacian_u = u_xx + u_yy
            grad_dot = u_x * ∇μX[i,j] + u_y * ∇μY[i,j]
            potential_term = u[i,j] * (∇2μX[i,j] + ∇2μY[i,j])

            Lu = D * laplacian_u + (D/kbT) * (grad_dot + potential_term)

            # Source term: S = ∂u/∂t - L[u]
            source_term[i,j] = u_dt[i,j] - Lu
        end
    end

    return source_term
end

"""
Create source function for MMS testing - returns a function that takes time as input
"""
function create_mms_source_direct(source_func)
    return t -> source_func(t)
end

"""
Legacy wrapper for compatibility with old splitting approach
"""
function create_mms_source(source_func, t_start)
    current_time = [t_start]  # Use array to make it mutable in closure

    function source_wrapper(u, dt)
        current_time[1] += dt
        source_term = source_func(current_time[1])
        return u .+ dt .* source_term
    end

    return source_wrapper
end

"""
Run MMS verification for potential 1
"""
function run_mms_verification_1_direct()
    println("=== MMS Verification for Potential 1 ===")

    # Generate potential data
    μ, ∇μX, ∇μY, ∇2μX, ∇2μY = generate_potential_1(nx)

    # Time parameters
    t_final = 10.0
    dt_values = [1, 0.5, 0.25, 0.1, 0.05, 0.025, 0.0125, 0.00625, 0.003125]

    errors_lie = Float64[]
    errors_strang = Float64[]
    errors_adi_split = Float64[]
    errors_adi_direct = Float64[]

    for dt in dt_values
        num_steps = round(Int, t_final / dt)
        actual_final_time = dt * num_steps

        # Initial condition
        u0 = manufactured_solution_1(0.0)

        # Exact solution at final time
        u_exact = manufactured_solution_1(actual_final_time)

        # Source function for direct integration
        source_func(t) = compute_source_term_1(t, μ, ∇μX, ∇μY, ∇2μX, ∇2μY)
        source_direct = create_mms_source_direct(source_func)

        # Test Lie splitting
        source_lie = create_mms_source(source_func, 0.0)
        u_lie = lie_splitting_with_source(dt, u0, num_steps, μ, ∇μX, ∇μY, ∇2μX, ∇2μY, source_func)
        error_lie = norm(u_lie - u_exact, 2) * dx
        push!(errors_lie, error_lie)

        # Test Strang splitting
        source_strang = create_mms_source(source_func, 0.0)
        u_strang = strang_splitting_with_source(dt, u0, num_steps, μ, ∇μX, ∇μY, ∇2μX, ∇2μY, source_func)
        error_strang = norm(u_strang - u_exact, 2) * dx
        push!(errors_strang, error_strang)

        # Test ADI scheme with direct integration
        u_adi_direct = adi_scheme_direct_source(dt, u0, num_steps, μ, ∇μX, ∇μY, ∇2μX, ∇2μY, source_func)
        error_adi_direct = norm(u_adi_direct - u_exact, 2) * dx
        push!(errors_adi_direct, error_adi_direct)

        # Test FD scheme

        println("dt = $dt:")
        println("  Lie error = $error_lie")
        println("  Strang error = $error_strang")
        println("  ADI direct error = $error_adi_direct")
        println("  Mass conservation check:")
        println("    ADI direct: $(sum(u_adi_direct))")
    end

    # Plot convergence
    p1 = plot(dt_values, [errors_lie errors_strang errors_adi_direct],
              xlabel="Time step", ylabel="L2 Error",
              label=["Lie" "Strang" "ADI Direct"],
              xscale=:log10, yscale=:log10,
              markershape=[:circle :x :hexagon], linestyles = [:solid :solid :solid], title="MMS Convergence - Potential 1", legend=:outertopright)

    return p1, errors_lie, errors_strang, errors_adi_direct
end

"""
Run MMS verification for potential 2 - comparing direct integration vs splitting
"""
function run_mms_verification_2_direct()
    println("=== MMS Verification for Potential 2 - Direct ADI vs Splitting ===")

    # Generate potential data
    μ, ∇μX, ∇μY, ∇2μX, ∇2μY = generate_potential_2(nx)

    # Time parameters
    t_final = 10.0
    dt_values = [1, 0.5, 0.25, 0.1, 0.05, 0.025, 0.0125, 0.00625, 0.003125]

    errors_lie = Float64[]
    errors_strang = Float64[]
    errors_adi_split = Float64[]
    errors_adi_direct = Float64[]

    for dt in dt_values
        num_steps = round(Int, t_final / dt)
        actual_final_time = dt * num_steps

        # Initial condition
        u0 = manufactured_solution_2(0.0)

        # Exact solution at final time
        u_exact = manufactured_solution_2(actual_final_time)

        # Source function for direct integration
        source_func(t) = compute_source_term_2(t, μ, ∇μX, ∇μY, ∇2μX, ∇2μY)
        source_direct = create_mms_source_direct(source_func)

        # Test Lie splitting
        source_lie = create_mms_source(source_func, 0.0)
        u_lie = lie_splitting_with_source(dt, u0, num_steps, μ, ∇μX, ∇μY, ∇2μX, ∇2μY, source_func)
        error_lie = norm(u_lie - u_exact, 2) * dx
        push!(errors_lie, error_lie)

        # Test Strang splitting
        source_strang = create_mms_source(source_func, 0.0)
        u_strang = strang_splitting_with_source(dt, u0, num_steps, μ, ∇μX, ∇μY, ∇2μX, ∇2μY, source_func)
        error_strang = norm(u_strang - u_exact, 2) * dx
        push!(errors_strang, error_strang)

        # Test ADI scheme with direct integration
        u_adi_direct = adi_scheme_direct_source(dt, u0, num_steps, μ, ∇μX, ∇μY, ∇2μX, ∇2μY, source_func)
        error_adi_direct = norm(u_adi_direct - u_exact, 2) * dx
        push!(errors_adi_direct, error_adi_direct)

        println("dt = $dt:")
        println("  Lie error = $error_lie")
        println("  Strang error = $error_strang")
        println("  ADI direct error = $error_adi_direct")
        println("  Mass conservation check:")
        println("    ADI direct: $(sum(u_adi_direct))")
    end

    # Plot convergence
    p1 = plot(dt_values, [errors_lie errors_strang errors_adi_direct],
              xlabel="Time step", ylabel="L2 Error",
              label=["Lie" "Strang" "ADI Direct"],
              xscale=:log10, yscale=:log10,
              markershape=[:circle :x :hexagon], linestyles = [:solid :solid :solid], title="MMS Convergence - Potential 2", legend=:outertopright)

    return p1, errors_lie, errors_strang, errors_adi_direct
end
