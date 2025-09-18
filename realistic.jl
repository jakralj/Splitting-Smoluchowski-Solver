using LinearAlgebra
using SparseArrays
using Plots
using DelimitedFiles

D = 0.01
nx = 100
dx = 1/100
kbT = 0.65
dt = 0.01

# Generate potential field and derivatives
μ = readdlm("pmf.in") 

# X-direction derivatives
∇μX = readdlm("damjux.in") 
∇2μX = readdlm("d2amjux.in")

# Y-direction derivatives
∇μY = readdlm("damjuy.in") 
∇2μY = readdlm("d2amjuy.in") 

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
Lie Splitting (First Order)
u^{n+1} = Y(dt) X(dt) u^n
"""
function lie_splitting(u0, num_steps)
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
function strang_splitting(u0, num_steps)
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
function swss_splitting(u0, num_steps)
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
Run simulations with different splitting methods and compare results
"""
function compare_splitting_methods(initial_concentration, num_steps)
    # Initial concentration field
    u0 = initial_concentration
    println("Total initial concentration: $(sum(u0))") 
    # Run each method
    u_lie = lie_splitting(u0, num_steps)
    u_strang = strang_splitting(u0, num_steps)
    u_swss = swss_splitting(u0, num_steps)
    
    println("Final lie concentration: $(sum(u_lie))") 
    println("Final strang concentration: $(sum(u_strang))") 
    println("Final SWSS concentration: $(sum(u_swss))") 
    # Plot results
    p1 = heatmap(u_lie, title="Lie Splitting", c=:viridis)
    p2 = heatmap(u_strang, title="Strang Splitting", c=:viridis)
    p3 = heatmap(u_swss, title="SWSS Splitting", c=:viridis)
    p4 = heatmap(u0, title="Initial Concentration", c=:viridis)
    
    plot(p4, p1, p2, p3, layout=(2,2), size=(800,800))
end

# Example usage
function run_example()
    # Create a Gaussian initial concentration field
    x_center, y_center = nx/2, nx/2
    sigma = nx/10
    u0 = [exp(-((x-x_center)^2 + (y-y_center)^2)/(2*sigma^2)) for y in 1:nx, x in 1:nx]
    u0 = u0 / sum(u0)  # Normalize
    
    # Run comparison with 100 time steps
    compare_splitting_methods(u0, 10000)
end

# Function to analyze convergence
function analyze_convergence()
    # Create reference solution with very fine time stepping
    x_center, y_center = nx/2, nx/2
    sigma = nx/10
    u0 = [exp(-((x-x_center)^2 + (y-y_center)^2)/(2*sigma^2)) for y in 1:nx, x in 1:nx]
    u0 = u0 / sum(u0)
    
    # Reference solution with tiny steps
    u_ref = strang_splitting(u0, 1000)
    
    # Test different time steps
    dt_values = [0.1, 0.05, 0.025, 0.0125]
    lie_errors = Float64[]
    strang_errors = Float64[]
    swss_errors = Float64[]
    
    for test_dt in dt_values
        # Save the original dt
        global_dt = dt
        # Temporarily change dt
        global dt = test_dt
        
        # Run each method for equivalent total time
        steps = round(Int, global_dt * 100 / test_dt)
        u_lie = lie_splitting(u0, steps)
        u_strang = strang_splitting(u0, steps)
        u_swss = swss_splitting(u0, steps)
        
        # Calculate errors
        push!(lie_errors, norm(u_lie - u_ref))
        push!(strang_errors, norm(u_strang - u_ref))
        push!(swss_errors, norm(u_swss - u_ref))
        
        # Restore original dt
        global dt = global_dt
    end
    
    # Plot convergence
    p = plot(dt_values, [lie_errors strang_errors swss_errors], 
             xlabel="Time step", ylabel="Error", 
             label=["Lie" "Strang" "SWSS"], 
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
