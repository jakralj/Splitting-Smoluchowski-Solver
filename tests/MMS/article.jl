using LinearAlgebra
using SparseArrays
using Plots
using DelimitedFiles

const D = 0.01
const nx = 100
const dx = 1/1000
const kbT = 0.65

# Generate potential field and derivatives
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
function adi_scheme_with_source(dtt::Float64, u0, num_steps, source)
    u = copy(u0)
    u_half = similar(u0)
    dt = dtt/2 
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
        # source term 
        u = source(u, dtt)

        # second part
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

function lie_splitting_with_source(dt::Float64, u0, num_steps, source)
    u = copy(u0)
    for step in 1:num_steps
        # X direction first
        u = evolve_x(u, dt/2)
        # Then Y direction
        u = evolve_y(u, dt/2)
        u = source(u, dt)
        u = evolve_x(u, dt/2)
        # Then Y direction
        u = evolve_y(u, dt/2)
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

function strang_splitting(dt::Float64, u0, num_steps, source)
    u = copy(u0)
    for step in 1:num_steps
        # Half-step in X direction
        u = evolve_x(u, dt/4)
        # Full step in Y direction
        u = evolve_y(u, dt/2)
        # Half-step in X direction
        u = evolve_x(u, dt/4)
        u = source(u, dt)
        # Half-step in X direction
        u = evolve_x(u, dt/4)
        # Full step in Y direction
        u = evolve_y(u, dt/2)
        # Half-step in X direction
        u = evolve_x(u, dt/4)
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
    u_adi = adi_scheme(dt, u0, num_steps)
   
    if fd 
        println("Running Finite Difference Euler...")
        u_euler = finite_difference_euler(dt/100, u0, 100*num_steps)
    end 
    
    println("Final lie concentration: $(sum(u_lie))") 
    println("Final strang concentration: $(sum(u_strang))") 
    println("Final SWSS concentration: $(sum(u_swss))") 
    println("Final ADI concentration: $(sum(u_adi))") 
    if fd
        println("Final FD Euler concentration: $(sum(u_euler))") 
    end 
    
    # Plot results
    p1 = heatmap(u_lie, title="Lie Splitting", c=:viridis)
    p2 = heatmap(u_strang, title="Strang Splitting", c=:viridis)
    p3 = heatmap(u_swss, title="SWSS Splitting", c=:viridis)
    p4 = heatmap(u_adi, title="ADI Scheme", c=:viridis)
    p5 = heatmap(u0, title="Initial Concentration", c=:viridis)
    
    if fd
        p6 = heatmap(u_euler, title="FD Euler", c=:viridis)
        plot(p5, p1, p2, p3, p4, p6, layout=(3,2), size=(1000,1200))
    else
        plot(p5, p1, p2, p3, p4, layout=(3,2), size=(1000,1000))
    end
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
    
    # Finite Difference Euler
    if fd
        t1 = time()
        finite_difference_euler(dt, u0, num_steps)
        t2 = time()
        euler_time = t2 - t1
        println("Finite Difference Euler: $(euler_time) seconds")
    else
        euler_time = 10
    end
    
    # Plot performance comparison
    methods = ["Lie", "Strang", "SWSS", "ADI", "FD Euler"]
    times = [lie_time, strang_time, swss_time, adi_time, euler_time]
    
    bar(methods, times, 
        title="Performance Comparison", 
        ylabel="Time (seconds)", 
        legend=false, 
        rotation=45, 
        size=(700, 400))
end

"""
Benchmark different potential functions
"""
function benchmark_potentials()
    # Create a standard initial condition
    x_center, y_center = nx/2, nx/2
    sigma = nx/10
    u0 = [exp(-((x-x_center)^2 + (y-y_center)^2)/(2*sigma^2)) for y in 1:nx, x in 1:nx]
    u0 = u0 / sum(u0)  # Normalize
    
    num_steps = 1000
    
    println("\n=== Benchmarking Potential 1: f(x,y) = x²y + e^x sin(y) ===")
    pot1_data = generate_potential_1(nx, dx)
    println("Running comparison for Potential 1...")
    compare_splitting_methods(u0, num_steps, pot1_data)
    savefig("potential1_comparison.pdf")
    
    println("\nBenchmarking performance for Potential 1...")
    benchmark_methods(u0, num_steps, pot1_data)
    savefig("potential1_benchmark.pdf")
    
    println("\n=== Benchmarking Potential 2: f(x,y) = xy² + ln(x² + 1) + y³ ===")
    pot2_data = generate_potential_2(nx, dx)
    println("Running comparison for Potential 2...")
    compare_splitting_methods(u0, num_steps, pot2_data)
    savefig("potential2_comparison.pdf")
    
    println("\nBenchmarking performance for Potential 2...")
    benchmark_methods(u0, num_steps, pot2_data)
    savefig("potential2_benchmark.pdf")
    
    # Create plots showing the potential landscapes
    p1 = heatmap(pot1_data[1], title="Potential 1: x²y + e^x sin(y)", c=:plasma)
    p2 = heatmap(pot2_data[1], title="Potential 2: xy² + ln(x² + 1) + y³", c=:plasma)
    plot(p1, p2, layout=(1,2), size=(1000, 400))
    savefig("potential_landscapes.pdf")
end

"""
Analyze convergence for different methods including ADI
"""
function analyze_convergence(potential_data=nothing)
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
    u_ref = strang_splitting(dt_ref, u0, steps_ref)
    
    # Test different time steps
    dt_values = [0.1, 0.05, 0.025, 0.0125, 0.00125]
    lie_errors = Float64[]
    strang_errors = Float64[]
    swss_errors = Float64[]
    adi_errors = Float64[]
    #fd_errors = Float64[]
    
    for dt in dt_values
        # Run each method for equivalent total time
        total_time = dt_ref * steps_ref
        steps = round(Int, total_time / dt)
        
        u_lie = lie_splitting(dt, u0, steps)
        u_strang = strang_splitting(dt, u0, steps)
        u_swss = swss_splitting(dt, u0, steps)
        u_adi = adi_scheme(dt, u0, steps)
        #u_fd = finite_difference_euler(dt, u0, steps)
        
        # Calculate errors
        push!(lie_errors, norm(u_lie - u_ref))
        push!(strang_errors, norm(u_strang - u_ref))
        push!(swss_errors, norm(u_swss - u_ref))
        push!(adi_errors, norm(u_adi - u_ref))
        #push!(fd_errors, norm(u_fd - u_ref))
    end
    
    # Plot convergence
    p = plot(dt_values, [lie_errors strang_errors swss_errors adi_errors], 
             xlabel="Time step", ylabel="Error", 
             label=["Lie" "Strang" "SWSS" "ADI"], 
             xscale=:log10, yscale=:log10,
             marker=:circle, legend=:bottomright)
    
    # Add reference slopes
    x_ref = dt_values
    y_ref1 = dt_values.^1 * (lie_errors[1]/dt_values[1])
    y_ref2 = dt_values.^2 * (strang_errors[1]/dt_values[1]^2)
    plot!(p, x_ref, y_ref1, linestyle=:dash, color=:black, label="First order")
    plot!(p, x_ref, y_ref2, linestyle=:dash, color=:gray, label="Second order")
    
    return p
end

# Example usage
function run_example()
    # Create a Gaussian initial concentration field
    x_center, y_center = nx/2, nx/2
    sigma = nx/10
    u0 = [exp(-((x-x_center)^2 + (y-y_center)^2)/(2*sigma^2)) for y in 1:nx, x in 1:nx]
    u0 = u0 / sum(u0)  # Normalize
    
    # Run comparison with specified time steps
    compare_splitting_methods(u0, 10000)
end

# Run all tests including new ADI and potential benchmarks
function run_all_tests()
    # Create a standard initial condition
    x_center, y_center = nx/2, nx/2
    sigma = nx/10
    u0 = [exp(-((x-x_center)^2 + (y-y_center)^2)/(2*sigma^2)) for y in 1:nx, x in 1:nx]
    u0 = u0 / sum(u0)  # Normalize
    
    # Run comparative simulations with original potential
    println("\n=== Running comparison of all methods (Original Potential) ===")
    compare_splitting_methods(u0, 10000)
    savefig("method_comparison_original.pdf")
    
    # Run benchmarking with original potential
    println("\n=== Running benchmarking (Original Potential) ===")
    benchmark_methods(u0, 1000)
    savefig("performance_benchmark_original.pdf")
    
    # Run convergence analysis
    println("\n=== Running convergence analysis ===")
    p = analyze_convergence()
    display(p)
    savefig("convergence_analysis.pdf")
    
    # Run potential benchmarks
    println("\n=== Running potential benchmarks ===")
    benchmark_potentials()
end

# If run directly, execute all tests
if abspath(PROGRAM_FILE) == @__FILE__
    run_all_tests()
end
