using LinearAlgebra
using SparseArrays
using Plots
using DelimitedFiles

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

"""
Run simulations with different splitting methods and compare results
"""
function compare_splitting_methods(initial_concentration, num_steps)
    # Initial concentration field
    u0 = initial_concentration
    println("Total initial concentration: $(sum(u0))") 
    dt = 1/100 
    # Run each method
    println("Running Lie splitting...")
    u_lie = lie_splitting(dt, u0, num_steps)
    
    println("Running Strang splitting...")
    u_strang = strang_splitting(dt, u0, num_steps)
    
    println("Running SWSS splitting...")
    u_swss = swss_splitting(dt, u0, num_steps)
    if fd 
        println("Running Finite Difference Euler...")
        u_euler = finite_difference_euler(dt/100, u0, 100*num_steps)
    end 
    println("Final lie concentration: $(sum(u_lie))") 
    println("Final strang concentration: $(sum(u_strang))") 
    println("Final SWSS concentration: $(sum(u_swss))") 
    if fd
        println("Final FD Euler concentration: $(sum(u_euler))") 
    end 
    # Plot results
    p1 = heatmap(u_lie, title="Lie Splitting", c=:viridis)
    p2 = heatmap(u_strang, title="Strang Splitting", c=:viridis)
    p3 = heatmap(u_swss, title="SWSS Splitting", c=:viridis)
    if fd
        p4 = heatmap(u_euler, title="FD Euler", c=:viridis)
    end
    p5 = heatmap(u0, title="Initial Concentration", c=:viridis)
    if fd 
        plot(p5, p1, p2, p3, p4, layout=(3,2), size=(800,1000))
    else
        plot(p5, p1, p2, p3, layout=(2,2), size=(1000,1000))
    end
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
    euler_errors = Float64[]
    
    for dt in dt_values
        # Run each method for equivalent total time
        steps = round(Int, dt * 100 / test_dt)
        u_lie = lie_splitting(dt, u0, steps)
        u_strang = strang_splitting(dt, u0, steps)
        u_swss = swss_splitting(dt, u0, steps)
        if fd
            u_euler = finite_difference_euler(100*dt, u0, 100*steps)
        end 
        # Calculate errors
        push!(lie_errors, norm(u_lie - u_ref))
        push!(strang_errors, norm(u_strang - u_ref))
        push!(swss_errors, norm(u_swss - u_ref))
        if fd
            push!(euler_errors, norm(u_euler - u_ref))
        end
        
    end
    
    # Plot convergence
    p = plot(dt_values, [lie_errors strang_errors swss_errors euler_errors], 
             xlabel="Time step", ylabel="Error", 
             label=["Lie" "Strang" "SWSS" "FD Euler"], 
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

"""
Benchmarking function to compare performance of different methods
"""
function benchmark_methods(initial_concentration, num_steps)
    u0 = initial_concentration
    
    println("Benchmarking different solution methods...")
    println("----------------------------------------")
   
    dt = 1/100
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
    
    # Finite Difference Euler
    if fd
        t1 = time()
        finite_difference_euler(dt/1000, u0, 1000*num_steps)
        t2 = time()
        euler_time = t2 - t1
        println("Finite Difference Euler: $(euler_time) seconds")
    else
        euler_time = 10
    end
    # Plot performance comparison
    methods = ["Lie", "Strang", "SWSS", "FD Euler"]
    times = [lie_time, strang_time, swss_time, euler_time]
    
    bar(methods, times, 
        title="Performance Comparison", 
        ylabel="Time (seconds)", 
        legend=false, 
        rotation=45, 
        size=(600, 400))
end

# Run all tests
function run_all_tests()
    # Create a standard initial condition
    x_center, y_center = nx/2, nx/2
    sigma = nx/10
    u0 = [exp(-((x-x_center)^2 + (y-y_center)^2)/(2*sigma^2)) for y in 1:nx, x in 1:nx]
    u0 = u0 / sum(u0)  # Normalize
    
    # Run comparative simulations
    println("\n=== Running comparison of all methods ===")
    compare_splitting_methods(u0, 10000)
    savefig("method_comparison.pdf")
    
    # Run convergence analysis
    # println("\n=== Running convergence analysis ===")
    # p = analyze_convergence()
    # display(p)
    # savefig("convergence_analysis.png")
    
    # Run benchmarking
    println("\n=== Running benchmarking ===")
    benchmark_methods(u0, 1000)
    savefig("performance_benchmark.pdf")
end

# If run directly, execute all tests
if abspath(PROGRAM_FILE) == @__FILE__
    run_all_tests()
end
