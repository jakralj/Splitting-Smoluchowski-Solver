using LinearAlgebra
using SparseArrays
using Plots
using DelimitedFiles

const D = 0.01
const nx = 100
const dx = 1/1000
const kbT = 0.65

# Grid generation
const x_range = range(-1, 1, length=nx)
const y_range = range(-1, 1, length=nx)

"""
Helper function to shift an array with zero boundary conditions
"""
function shift(u, s)
    n = circshift(u, s)
    if s > 0
        n[1:s] .= 0
    end
    if s < 0
        n[end+s+1:end] .= 0
    end
    return n
end

"""
Crank-Nicolson solver for 1D Smoluchowski equation
"""
function CrankNicolson1D(u, μ, ∇μ, ∇2μ, D, dx, dt, kbT)
    α = D / (4*dx*kbT)
    β = D / (2*dx^2)
    γ = D / (2*kbT)
    ϵ = D / (dx^2)
    
    uP = shift(u, 1)
    uN = shift(u, -1)
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
Evolution operators for splitting methods
"""
function evolve_x(u_2d, μ, ∇μX, ∇2μX, dt)
    u_new = similar(u_2d)
    for i in 1:nx
        u_new[i, :] = CrankNicolson1D(u_2d[i, :], μ[i, :], ∇μX[i, :], ∇2μX[i, :], D, dx, dt, kbT)
    end
    return u_new
end

function evolve_y(u_2d, μ, ∇μY, ∇2μY, dt)
    u_new = similar(u_2d)
    for j in 1:nx
        u_new[:, j] = CrankNicolson1D(u_2d[:, j], μ[:, j], ∇μY[:, j], ∇2μY[:, j], D, dx, dt, kbT)
    end
    return u_new
end

"""
Alternating Direction Implicit (ADI) scheme with source term
"""
function x_direction_sweep(u, αx, αy, βx, βy, dt, ∇μX, ∇μY, ∇2μX, ∇2μY)
    u_half = similar(u)
    for i in 1:nx
        # Build tridiagonal system for row i
        Adl = zeros(nx-1)  # Lower diagonal
        Adu = zeros(nx-1)  # Upper diagonal
        Au = zeros(nx)     # Main diagonal
        B = zeros(nx)      # RHS
        for j in 1:nx
            if j == 1
                # Left boundary: von Neumann condition du/dx = 0
                # This means u[i,0] = u[i,2] (ghost point)
                # Discretization: -αx*u[i,0] + (1 + 2αx - D*dt/(2*kbT)*∇2μX[i,j])*u[i,1] - (αx + βx*∇μX[i,j])*u[i,2] = RHS
                # Substituting u[i,0] = u[i,2]: (1 + 2αx - D*dt/(2*kbT)*∇2μX[i,j])*u[i,1] - (2αx + βx*∇μX[i,j])*u[i,2] = RHS

                # Explicit y-direction terms
                if i == 1 || i == nx
                    # Corner: von Neumann in y as well
                    if i == 1
                        y_term = 2*αy*(u[i+1,j] - u[i,j]) +
                                u[i,j]*D*dt/(2*kbT)*∇2μY[i,j]
                    else
                        y_term = 2*αy*(u[i-1,j] - u[i,j]) +
                                u[i,j]*D*dt/(2*kbT)*∇2μY[i,j]
                    end
                else
                    y_term = αy*(u[i-1,j] - 2*u[i,j] + u[i+1,j]) +
                            βy*(u[i+1,j] - u[i-1,j])*∇μY[i,j] +
                            u[i,j]*D*dt/(2*kbT)*∇2μY[i,j]
                end

                Au[j] = 1.0 + 2*αx - D*dt/(2*kbT)*∇2μX[i,j]
                Adu[j] = -(2*αx + βx*∇μX[i,j])
                B[j] = u[i,j] + y_term

            elseif j == nx
                # Right boundary: von Neumann condition du/dx = 0
                # This means u[i,nx+1] = u[i,nx-1] (ghost point)
                # Discretization: -(αx - βx*∇μX[i,j])*u[i,nx-1] + (1 + 2αx - D*dt/(2*kbT)*∇2μX[i,j])*u[i,nx] - αx*u[i,nx+1] = RHS
                # Substituting u[i,nx+1] = u[i,nx-1]: -(2αx - βx*∇μX[i,j])*u[i,nx-1] + (1 + 2αx - D*dt/(2*kbT)*∇2μX[i,j])*u[i,nx] = RHS

                # Explicit y-direction terms
                if i == 1 || i == nx
                    # Corner: von Neumann in y as well
                    if i == 1
                        y_term = 2*αy*(u[i+1,j] - u[i,j]) +
                                u[i,j]*D*dt/(2*kbT)*∇2μY[i,j]
                    else
                        y_term = 2*αy*(u[i-1,j] - u[i,j]) +
                                u[i,j]*D*dt/(2*kbT)*∇2μY[i,j]
                    end
                else
                    y_term = αy*(u[i-1,j] - 2*u[i,j] + u[i+1,j]) +
                            βy*(u[i+1,j] - u[i-1,j])*∇μY[i,j] +
                            u[i,j]*D*dt/(2*kbT)*∇2μY[i,j]
                end

                Adl[j-1] = -(2*αx - βx*∇μX[i,j])
                Au[j] = 1.0 + 2*αx - D*dt/(2*kbT)*∇2μX[i,j]
                B[j] = u[i,j] + y_term

            else
                # Interior points
                # Explicit y-direction terms
                if i == 1 || i == nx
                    # Top/bottom boundary: von Neumann in y
                    if i == 1
                        y_term = 2*αy*(u[i+1,j] - u[i,j]) +
                                u[i,j]*D*dt/(2*kbT)*∇2μY[i,j]
                    else
                        y_term = 2*αy*(u[i-1,j] - u[i,j]) +
                                u[i,j]*D*dt/(2*kbT)*∇2μY[i,j]
                    end
                else
                    y_term = αy*(u[i-1,j] - 2*u[i,j] + u[i+1,j]) +
                            βy*(u[i+1,j] - u[i-1,j])*∇μY[i,j] +
                            u[i,j]*D*dt/(2*kbT)*∇2μY[i,j]
                end

                # Coefficients for implicit x-direction
                c_left = -αx - βx*∇μX[i,j]
                c_center = 1.0 + 2*αx - D*dt/(2*kbT)*∇2μX[i,j]
                c_right = -αx + βx*∇μX[i,j]

                Au[j] = c_center
                Adl[j-1] = c_left
                Adu[j] = c_right
                B[j] = u[i,j] + y_term
            end
        end

        # Solve tridiagonal system
        A_matrix = Tridiagonal(Adl, Au, Adu)
        u_half[i,:] = A_matrix \ B
    end
    return u_half
end

function y_direction_sweep(u_half, αx, αy, βx, βy, dt, ∇μX, ∇μY, ∇2μX, ∇2μY)
    u = similar(u_half)
    for j in 1:nx
        # Build tridiagonal system for column j
        Adl = zeros(nx-1)  # Lower diagonal
        Adu = zeros(nx-1)  # Upper diagonal
        Au = zeros(nx)     # Main diagonal
        B = zeros(nx)      # RHS

        for i in 1:nx
            if i == 1
                # Bottom boundary: von Neumann condition du/dy = 0
                # This means u[0,j] = u[2,j] (ghost point)

                # Explicit x-direction terms
                if j == 1 || j == nx
                    # Corner: von Neumann in x as well
                    if j == 1
                        x_term = 2*αx*(u_half[i,j+1] - u_half[i,j]) +
                                u_half[i,j]*D*dt/(2*kbT)*∇2μX[i,j]
                    else
                        x_term = 2*αx*(u_half[i,j-1] - u_half[i,j]) +
                                u_half[i,j]*D*dt/(2*kbT)*∇2μX[i,j]
                    end
                else
                    x_term = αx*(u_half[i,j-1] - 2*u_half[i,j] + u_half[i,j+1]) +
                            βx*(u_half[i,j+1] - u_half[i,j-1])*∇μX[i,j] +
                            u_half[i,j]*D*dt/(2*kbT)*∇2μX[i,j]
                end

                Au[i] = 1.0 + 2*αy - D*dt/(2*kbT)*∇2μY[i,j]
                Adu[i] = -(2*αy + βy*∇μY[i,j])
                B[i] = u_half[i,j] + x_term

            elseif i == nx
                # Top boundary: von Neumann condition du/dy = 0
                # This means u[nx+1,j] = u[nx-1,j] (ghost point)

                # Explicit x-direction terms
                if j == 1 || j == nx
                    # Corner: von Neumann in x as well
                    if j == 1
                        x_term = 2*αx*(u_half[i,j+1] - u_half[i,j]) +
                                u_half[i,j]*D*dt/(2*kbT)*∇2μX[i,j]
                    else
                        x_term = 2*αx*(u_half[i,j-1] - u_half[i,j]) +
                                u_half[i,j]*D*dt/(2*kbT)*∇2μX[i,j]
                    end
                else
                    x_term = αx*(u_half[i,j-1] - 2*u_half[i,j] + u_half[i,j+1]) +
                            βx*(u_half[i,j+1] - u_half[i,j-1])*∇μX[i,j] +
                            u_half[i,j]*D*dt/(2*kbT)*∇2μX[i,j]
                end

                Adl[i-1] = -(2*αy - βy*∇μY[i,j])
                Au[i] = 1.0 + 2*αy - D*dt/(2*kbT)*∇2μY[i,j]
                B[i] = u_half[i,j] + x_term

            else
                # Interior points
                # Explicit x-direction terms
                if j == 1 || j == nx
                    # Left/right boundary: von Neumann in x
                    if j == 1
                        x_term = 2*αx*(u_half[i,j+1] - u_half[i,j]) +
                                u_half[i,j]*D*dt/(2*kbT)*∇2μX[i,j]
                    else
                        x_term = 2*αx*(u_half[i,j-1] - u_half[i,j]) +
                                u_half[i,j]*D*dt/(2*kbT)*∇2μX[i,j]
                    end
                else
                    x_term = αx*(u_half[i,j-1] - 2*u_half[i,j] + u_half[i,j+1]) +
                            βx*(u_half[i,j+1] - u_half[i,j-1])*∇μX[i,j] +
                            u_half[i,j]*D*dt/(2*kbT)*∇2μX[i,j]
                end

                # Coefficients for implicit y-direction
                c_bottom = -αy - βy*∇μY[i,j]
                c_center = 1.0 + 2*αy - D*dt/(2*kbT)*∇2μY[i,j]
                c_top = -αy + βy*∇μY[i,j]

                Au[i] = c_center
                Adl[i-1] = c_bottom
                Adu[i] = c_top
                B[i] = u_half[i,j] + x_term
            end
        end

        # Solve tridiagonal system
        A_matrix = Tridiagonal(Adl, Au, Adu)
        u[:,j] = A_matrix \ B
    end
    return u
end

function adi_scheme(dt::Float64, u0, num_steps)
    u = copy(u0)
    u_half = similar(u0)

    # Pre-compute coefficients
    αx = D*dt/(2*dx^2)
    αy = D*dt/(2*dx^2)  # Using dx for both since grid is square
    βx = D*dt/(4*kbT*dx)
    βy = D*dt/(4*kbT*dx)
    
    # Main time-stepping loop
    for step in 1:num_steps
        # First sweep: x-direction (implicit in x, explicit in y)
        u = x_direction_sweep(u, αx, αy, βx, βy, dt, ∇μX, ∇μY, ∇2μX, ∇2μY)
        
        # Second sweep: y-direction (implicit in y, explicit in x) 
        u = y_direction_sweep(u, αx, αy, βx, βy, dt, ∇μX, ∇μY, ∇2μX, ∇2μY)
    end
    
    return u
end


function adi_scheme_with_source(dt::Float64, u0, num_steps, μ, ∇μX, ∇μY, ∇2μX, ∇2μY, source)
    u = copy(u0)
    u_half = similar(u0)

    # Pre-compute coefficients
    αx = D*(dt/2)/(2*dx^2)
    αy = D*(dt/2)/(2*dx^2)  # Using dx for both since grid is square
    βx = D*(dt/2)/(4*kbT*dx)
    βy = D*(dt/2)/(4*kbT*dx)
    
    # Main time-stepping loop
    for step in 1:num_steps
        # First sweep: x-direction (implicit in x, explicit in y)
        u = x_direction_sweep(u, αx, αy, βx, βy, dt/2, ∇μX, ∇μY, ∇2μX, ∇2μY)
        
        # Second sweep: y-direction (implicit in y, explicit in x) 
        u = y_direction_sweep(u, αx, αy, βx, βy, dt/2, ∇μX, ∇μY, ∇2μX, ∇2μY)

        u = source(u, dt)

        # First sweep: x-direction (implicit in x, explicit in y)
        u = x_direction_sweep(u, αx, αy, βx, βy, dt/2, ∇μX, ∇μY, ∇2μX, ∇2μY)
        
        # Second sweep: y-direction (implicit in y, explicit in x) 
        u = y_direction_sweep(u, αx, αy, βx, βy, dt/2, ∇μX, ∇μY, ∇2μX, ∇2μY)
    end
    
    return u
end

"""
Finite Difference Euler explicit method for 2D Smoluchowski equation with sources
"""
function fd_with_source(dt::Float64, u0, num_steps, μ, ∇μX, ∇μY, ∇2μX, ∇2μY, source)
    alphax = D*dt/(dx^2)
    alphay = D*dt/(dx^2)  # Using dx for both since grid is square
    betax = D*dt/(2*kbT*dx)
    betay = D*dt/(2*kbT*dx)  # Using dx for both since grid is square

    u = copy(u0)
    u_new = similar(u)

    i_total = sum(u) - sum(u[1,:]) - sum(u[nx,:]) - sum(u[:,1]) - sum(u[:,nx])  # Initial mass

    for step in 1:num_steps
        # Interior points
        u = source(u, dt)
        for i in 2:nx-1
            for j in 2:nx-1
                u_new[i,j] = u[i,j] + (
                    alphax * (u[i-1,j] - 2 * u[i,j] + u[i+1,j]) +
                    alphay * (u[i,j-1] - 2 * u[i,j] + u[i,j+1]) +
                    betax * 2*dx * (u[i,j] * ∇2μX[i,j]) +
                    betay * 2*dx * (u[i,j] * ∇2μY[i,j]) +
                    betax * (u[i+1,j] - u[i-1,j]) * ∇μX[i,j] +
                    betay * (u[i,j+1] - u[i,j-1]) * ∇μY[i,j]
                )
            end
        end

        # von Neumann boundary conditions
        for i in 1:nx
            # Bottom boundary
            u_new[i,1] = u[i,1] + alphay * (2 * u[i,2] - 2 * u[i,1])
            # Top boundary
            u_new[i,nx] = u[i,nx] + alphay * (2 * u[i,nx-1] - 2 * u[i,nx])
        end

        for j in 1:nx
            # Left boundary
            u_new[1,j] = u[1,j] + alphax * (2 * u[2,j] - 2 * u[1,j])
            # Right boundary
            u_new[nx,j] = u[nx,j] + alphax * (2 * u[nx-1,j] - 2 * u[nx,j])
        end
        u = copy(u_new)
    end

    return u
end
"""
Lie Splitting with source term
"""
function lie_splitting_with_source(dt::Float64, u0, num_steps, μ, ∇μX, ∇μY, ∇2μX, ∇2μY, source)
    u = copy(u0)
    for step in 1:num_steps
        u = evolve_x(u, μ, ∇μX, ∇2μX, dt/2)
        u = evolve_y(u, μ, ∇μY, ∇2μY, dt/2)
        u = source(u, dt)
        u = evolve_x(u, μ, ∇μX, ∇2μX, dt/2)
        u = evolve_y(u, μ, ∇μY, ∇2μY, dt/2)
    end
    return u
end

"""
Strang Splitting with source term
"""
function strang_splitting_with_source(dt::Float64, u0, num_steps, μ, ∇μX, ∇μY, ∇2μX, ∇2μY, source)
    u = copy(u0)
    for step in 1:num_steps
        u = evolve_x(u, μ, ∇μX, ∇2μX, dt/4)
        u = evolve_y(u, μ, ∇μY, ∇2μY, dt/2)
        u = evolve_x(u, μ, ∇μX, ∇2μX, dt/4)
        u = source(u, dt)
        u = evolve_x(u, μ, ∇μX, ∇2μX, dt/4)
        u = evolve_y(u, μ, ∇μY, ∇2μY, dt/2)
        u = evolve_x(u, μ, ∇μX, ∇2μX, dt/4)
    end
    return u
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
Create source function for MMS testing
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
function run_mms_verification_1()
    println("=== MMS Verification for Potential 1 ===")
    
    # Generate potential data
    μ, ∇μX, ∇μY, ∇2μX, ∇2μY = generate_potential_1(nx)
    
    # Time parameters
    t_final = 10.0
    dt_values = [1, 0.5, 0.25, 0.1, 0.05, 0.025, 0.0125, 0.00625, 0.003125]
    
    errors_lie = Float64[]
    errors_strang = Float64[]
    errors_adi = Float64[]
    errors_fd = Float64[]
    
    for dt in dt_values
        num_steps = round(Int, t_final / dt)
        actual_final_time = dt * num_steps
        
        # Initial condition
        u0 = manufactured_solution_1(0.0)
        
        # Exact solution at final time
        u_exact = manufactured_solution_1(actual_final_time)
        
        # Create source function
        source_func(t) = compute_source_term_1(t, μ, ∇μX, ∇μY, ∇2μX, ∇2μY)
        source = create_mms_source(source_func, 0.0)
        
        # Test Lie splitting
        source_lie = create_mms_source(source_func, 0.0)
        u_lie = lie_splitting_with_source(dt, u0, num_steps, μ, ∇μX, ∇μY, ∇2μX, ∇2μY, source_lie)
        error_lie = norm(u_lie - u_exact, 2) * dx
        push!(errors_lie, error_lie)
        
        # Test Strang splitting  
        source_strang = create_mms_source(source_func, 0.0)
        u_strang = strang_splitting_with_source(dt, u0, num_steps, μ, ∇μX, ∇μY, ∇2μX, ∇2μY, source_strang)
        error_strang = norm(u_strang - u_exact, 2) * dx
        push!(errors_strang, error_strang)
        println(sum(u_strang))
        # Test ADI scheme
        source_adi = create_mms_source(source_func, 0.0)
        u_adi = adi_scheme_with_source(dt, u0, num_steps, μ, ∇μX, ∇μY, ∇2μX, ∇2μY, source_adi)
        println(sum(u_adi))
        error_adi = norm(u_adi - u_exact, 2) * dx
        push!(errors_adi, error_adi)
        # Test FD scheme
        source_fd = create_mms_source(source_func, 0.0)
        u_fd = fd_with_source(dt/10000, u0, num_steps, μ, ∇μX, ∇μY, ∇2μX, ∇2μY, source_fd)
        error_fd = norm(u_fd - u_exact, 2) * dx
        push!(errors_fd, error_fd)

        println("dt = $dt: Lie error = $error_lie, Strang error = $error_strang, ADI error = $error_adi, FD error = $error_fd")
    end
    
    # Plot convergence
    p1 = plot(dt_values, [errors_lie errors_strang errors_adi errors_fd], 
              xlabel="Time step", ylabel="L2 Error", 
              label=["Lie" "Strang" "ADI" "FD"], 
              xscale=:log10, yscale=:log10,
              marker=:circle, title="MMS Convergence - Potential 1")
    
    # Add reference slopes
    #y_ref1 = errors_lie[1] * (dt_values / dt_values[1]).^1
    #y_ref2 = errors_strang[1] * (dt_values / dt_values[1]).^2
    #plot!(p1, dt_values, y_ref1, linestyle=:dash, color=:black, label="O(dt)")
    #plot!(p1, dt_values, y_ref2, linestyle=:dash, color=:gray, label="O(dt²)")
    
    return p1, errors_lie, errors_strang, errors_adi, errors_fd
end
function run_mms_verification_2()
    println("=== MMS Verification for Potential 2 ===")
    
    # Generate potential data
    μ, ∇μX, ∇μY, ∇2μX, ∇2μY = generate_potential_2(nx)
    
    # Time parameters
    t_final = 10.0
    dt_values = [1, 0.5, 0.25, 0.1, 0.05, 0.025, 0.0125, 0.00625, 0.003125]

    errors_lie = Float64[]
    errors_strang = Float64[]
    errors_adi = Float64[]
    errors_fd = Float64[]
    
    for dt in dt_values
        num_steps = round(Int, t_final / dt)
        actual_final_time = dt * num_steps
        
        # Initial condition
        u0 = manufactured_solution_2(0.0)
        
        # Exact solution at final time
        u_exact = manufactured_solution_2(actual_final_time)
        
        # Create source function
        source_func(t) = compute_source_term_2(t, μ, ∇μX, ∇μY, ∇2μX, ∇2μY)
        source = create_mms_source(source_func, 0.0)
        
        # Test Lie splitting
        source_lie = create_mms_source(source_func, 0.0)
        u_lie = lie_splitting_with_source(dt, u0, num_steps, μ, ∇μX, ∇μY, ∇2μX, ∇2μY, source_lie)
        error_lie = norm(u_lie - u_exact, 2) * dx
        push!(errors_lie, error_lie)
        
        # Test Strang splitting  
        source_strang = create_mms_source(source_func, 0.0)
        u_strang = strang_splitting_with_source(dt, u0, num_steps, μ, ∇μX, ∇μY, ∇2μX, ∇2μY, source_strang)
        error_strang = norm(u_strang - u_exact, 2) * dx
        push!(errors_strang, error_strang)
        
        # Test ADI scheme
        source_adi = create_mms_source(source_func, 0.0)
        u_adi = adi_scheme_with_source(dt, u0, num_steps, μ, ∇μX, ∇μY, ∇2μX, ∇2μY, source_adi)
        error_adi = norm(u_adi - u_exact, 2) * dx
        push!(errors_adi, error_adi)
        # Test FD scheme
        source_fd = create_mms_source(source_func, 0.0)
        u_fd = fd_with_source(dt/10000, u0, num_steps, μ, ∇μX, ∇μY, ∇2μX, ∇2μY, source_fd)
        error_fd = norm(u_fd - u_exact, 2) * dx
        push!(errors_fd, error_fd)

        println("dt = $dt: Lie error = $error_lie, Strang error = $error_strang, ADI error = $error_adi, FD error = $error_fd")
    end
    
    # Plot convergence
    p1 = plot(dt_values, [errors_lie errors_strang errors_adi errors_fd], 
              xlabel="Time step", ylabel="L2 Error", 
              label=["Lie" "Strang" "ADI" "FD"], 
              xscale=:log10, yscale=:log10,
              marker=:circle, title="MMS Convergence - Potential 2")
    
    # Add reference slopes
    #y_ref1 = errors_lie[1] * (dt_values / dt_values[1]).^1
    #y_ref2 = errors_strang[1] * (dt_values / dt_values[1]).^2
    #plot!(p1, dt_values, y_ref1, linestyle=:dash, color=:black, label="O(dt)")
    #plot!(p1, dt_values, y_ref2, linestyle=:dash, color=:gray, label="O(dt²)")
    
    return p1, errors_lie, errors_strang, errors_adi, errors_fd
end
