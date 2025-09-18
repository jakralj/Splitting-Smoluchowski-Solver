using LinearAlgebra
using SparseArrays
using Plots
using DelimitedFiles
using Statistics

const D = 0.01
const nx = 100
const dx = 1/1000
const kbT = 0.65

# Generate potential field and derivatives
const μ = readdlm("pmf.in") 

# X-direction derivatives
const ∇μX = readdlm("damjux.in") 
const ∇2μX = readdlm("d2amjux.in")

# Y-direction derivatives
const ∇μY = readdlm("damjuy.in") 
const ∇2μY = readdlm("d2amjuy.in") 
fd = false

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
Can be used for either x or y direction
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
Operator for evolution in X direction for one timestep
Treats each row independently
"""
function evolve_x(u_2d, dt)
    u_new = similar(u_2d)
    for i in 1:nx
        u_new[i, :] = CrankNicolson1D(u_2d[i, :], μ[i, :], ∇μX[i, :], ∇2μX[i, :], D, dx, dt, kbT)
    end
    return u_new
end

"""
Operator for evolution in Y direction for one timestep
Treats each column independently
"""
function evolve_y(u_2d, dt)
    u_new = similar(u_2d)
    for j in 1:nx
        u_new[:, j] = CrankNicolson1D(u_2d[:, j], μ[:, j], ∇μY[:, j], ∇2μY[:, j], D, dx, dt, kbT)
    end
    return u_new
end

"""
Alternating Direction Implicit (ADI) scheme for 2D Smoluchowski equation
with von Neumann (isolating) boundary conditions
Step 1: Implicit in x-direction, explicit in y-direction
Step 2: Explicit in x-direction, implicit in y-direction
"""
function adi_scheme(dt::Float64, u0, num_steps)
    u = copy(u0)
    u_half = similar(u0)
    
    # Pre-compute coefficients
    αx = D*dt/(2*dx^2)
    αy = D*dt/(2*dx^2)  # Using dx for both since grid is square
    βx = D*dt/(4*kbT*dx)
    βy = D*dt/(4*kbT*dx)
    
    for step in 1:num_steps
        # First half-step: implicit in x, explicit in y
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
        
        # Second half-step: explicit in x, implicit in y
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
    end
    
    return u
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

function adi_scheme_split(dt::Float64, u0, num_steps)
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

"""
Lie Splitting (First Order)
u^{n+1} = Y(dt) X(dt) u^n
"""
function lie_splitting(dt::Float64, u0, num_steps)
    u = copy(u0)
    for step in 1:num_steps
        # X direction first
        u = evolve_x(u, dt)
        # Then Y direction
        u = evolve_y(u, dt)
    end
    return u
end

"""
Strang Splitting (Second Order)
u^{n+1} = X(dt/2) Y(dt) X(dt/2) u^n
"""
function strang_splitting(dt::Float64, u0, num_steps)
    u = copy(u0)
    for step in 1:num_steps
        # Half-step in X direction
        u = evolve_x(u, dt/2)
        # Full step in Y direction
        u = evolve_y(u, dt)
        # Half-step in X direction
        u = evolve_x(u, dt/2)
    end
    return u
end

"""
Symmetrically Weighted Sequential Splitting (SWSS) (Second Order)
u^{n+1} = 1/2 * [X(dt) Y(dt) + Y(dt) X(dt)] u^n
"""
function swss_splitting(dt::Float64, u0::Matrix{Float64}, num_steps::Int64)
    u = copy(u0)
    for step in 1:num_steps
        # Compute Lie splitting in both orders
        u1 = evolve_y(evolve_x(u, dt), dt)  # X then Y
        u2 = evolve_x(evolve_y(u, dt), dt)  # Y then X
        
        # Average the results
        u = 0.5 * (u1 + u2)
    end
    return u
end

"""
Finite Difference Euler explicit method for 2D Smoluchowski equation
Based on the Python implementation
"""
function finite_difference_euler(dt::Float64, u0::Matrix{Float64}, num_steps::Int64)
    alphax = D*dt/(dx^2)
    alphay = D*dt/(dx^2)  # Using dx for both since grid is square
    betax = D*dt/(2*kbT*dx)
    betay = D*dt/(2*kbT*dx)  # Using dx for both since grid is square
    
    u = copy(u0)
    u_new = similar(u)
    
    i_total = sum(u) - sum(u[1,:]) - sum(u[nx,:]) - sum(u[:,1]) - sum(u[:,nx])  # Initial mass
    
    for step in 1:num_steps
        # Interior points
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

# ============================================================================
# POTENTIAL GENERATION AND BENCHMARKING
# ============================================================================

"""
Run simulations with different splitting methods and compare results
Now includes ADI method
"""
function compare_splitting_methods(initial_concentration, num_steps, potential_data=nothing)
    # Use provided potential or default loaded data
    if potential_data !== nothing
        global μ, ∇μX, ∇μY, ∇2μX, ∇2μY = potential_data
    end
    
    # Initial concentration field
    u0 = initial_concentration
    println("Total initial concentration: $(sum(u0))") 
    dt = 1/1000 
    
    # Run each method
    println("Running Lie splitting...")
    u_lie = lie_splitting(dt, u0, num_steps)
    
    println("Running Strang splitting...")
    u_strang = strang_splitting(dt, u0, num_steps)
    
    println("Running SWSS splitting...")
    u_swss = swss_splitting(dt, u0, num_steps)
    
    println("Running ADI scheme...")
    u_adi = adi_scheme(dt/2, u0, num_steps)
   
    
    println("Final lie concentration: $(sum(u_lie))") 
    println("Final strang concentration: $(sum(u_strang))") 
    println("Final SWSS concentration: $(sum(u_swss))") 
    println("Final ADI concentration: $(sum(u_adi))") 
    # Plot results
    p1 = heatmap(u_lie, title="Lie Splitting", c=:viridis)
    p2 = heatmap(u_strang, title="Strang Splitting", c=:viridis)
    p3 = heatmap(u_swss, title="SWSS Splitting", c=:viridis)
    p4 = heatmap(u_adi, title="ADI Scheme", c=:viridis)
    p5 = heatmap(u0, title="Initial Concentration", c=:viridis)
    
    
    plot(p5, p1, p2, p3, p4, layout=(3,2), size=(1000,1000))
end

"""
Benchmarking function to compare performance of different methods
Now includes ADI method
"""
function benchmark_methods(initial_concentration, num_steps, potential_data=nothing)
    # Use provided potential or default loaded data
    if potential_data !== nothing
        global μ, ∇μX, ∇μY, ∇2μX, ∇2μY = potential_data
    end
    
    u0 = initial_concentration
    
    println("Benchmarking different solution methods...")
    println("----------------------------------------")
   
    dt = 1/1000
    
    # Lie splitting
    t1 = time()
    lie_splitting(dt, u0, num_steps)
    t2 = time()
    lie_time = t2 - t1
    println("Lie splitting: $(lie_time) seconds")
    
    # Strang splitting
    t1 = time()
    strang_splitting(dt, u0, num_steps)
    t2 = time()
    strang_time = t2 - t1
    println("Strang splitting: $(strang_time) seconds")
    
    # SWSS splitting
    t1 = time()
    swss_splitting(dt, u0, num_steps)
    t2 = time()
    swss_time = t2 - t1
    println("SWSS splitting: $(swss_time) seconds")
    
    # ADI scheme
    t1 = time()
    adi_scheme(dt, u0, num_steps)
    t2 = time()
    adi_time = t2 - t1
    println("ADI scheme: $(adi_time) seconds")
    
    # Plot performance comparison
    methods = ["Lie", "Strang", "SWSS", "ADI"]
    times = [lie_time, strang_time, swss_time, adi_time]
    
    bar(methods, times, 
        title="Performance Comparison", 
        ylabel="Time (seconds)", 
        legend=false, 
        rotation=45, 
        size=(700, 400))
end

"""
Analyze convergence for different methods including ADI
"""
function create_reference(potential_data=nothing)
    # Use provided potential or default loaded data
    if potential_data !== nothing
        global μ, ∇μX, ∇μY, ∇2μX, ∇2μY = potential_data
    end
    
    # Create reference solution with very fine time stepping
    x_center, y_center = nx/2, nx/2
    sigma = nx/10
    u0 = [exp(-((x-x_center)^2 + (y-y_center)^2)/(2*sigma^2)) for y in 1:nx, x in 1:nx]
    u0 = u0 / sum(u0)
    
    # Reference solution with tiny steps
    dt_ref = 0.0001
    steps_ref = 10000
    println("Running strang...")
    u_strang = strang_splitting(dt_ref, u0, steps_ref)
    println("End sum: $(sum(u_strang))")
    println("Running lie...")
    u_lie = lie_splitting(dt_ref, u0, steps_ref)
    println("End sum: $(sum(u_lie))")
    println("Running swss...")
    u_swss = swss_splitting(dt_ref, u0, steps_ref)
    println("End sum: $(sum(u_swss))")
    println("Running adi...")
    u_adi = adi_scheme(dt_ref, u0, steps_ref)
    println("End sum: $(sum(u_adi))")
    println("Running adi split...")
    u_adi_split = adi_scheme_split(dt_ref, u0, steps_ref)
    println("End sum: $(sum(u_adi_split))")
    u_comb = (u_strang .+ u_lie .+ u_swss)./3
    println("L2 error = $(mean(sqrt.((u_adi .- u_comb).^2)))")
    println("Split L2 error = $(mean(sqrt.((u_adi_split .- u_comb).^2)))")
    writedlm("strang.ref", u_strang, ' ')
    writedlm("lie.ref", u_lie, ' ')
    writedlm("swss.ref", u_swss, ' ')
    writedlm("adi.ref", u_adi, ' ')
   end


# If run directly, execute all tests
if abspath(PROGRAM_FILE) == @__FILE__
    create_reference() 
end
