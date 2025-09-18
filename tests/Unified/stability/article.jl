using LinearAlgebra
using SparseArrays
using Plots
using DelimitedFiles

const D = 0.01
const nx = 100
const dx = 1/1000
const kbT = 0.65

 

include("methods.jl")
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
    steps_ref = 1000000
    #u_ref = strang_splitting(dt_ref, u0, steps_ref)
    u_lie = readdlm("lie.ref")
    u_strang = readdlm("strang.ref")
    u_swss = readdlm("swss.ref")
    u_adi = readdlm("adi.ref")
    u_ref = (u_lie .+ u_strang .+ u_swss)./3
    println("Lie error: $(sum(abs.(u_ref .- u_lie)))")
    println("Swss error: $(sum(abs.(u_ref .- u_swss)))")
    println("Adi error: $(sum(abs.(u_ref .- u_adi)))")
    println("Strang error: $(sum(abs.(u_ref .- u_strang)))")
    # Test different time steps
    dt_values = exp10.(collect(-3:3))
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
    #plot!(p, x_ref, y_ref1, linestyle=:dash, color=:black, label="First order")
    #plot!(p, x_ref, y_ref2, linestyle=:dash, color=:gray, label="Second order")
    
    return p
end

function analyze_stability(potential_data=nothing)
    # Use provided potential or default loaded data
    if potential_data !== nothing
        global μ, ∇μX, ∇μY, ∇2μX, ∇2μY = potential_data
    end
    
    # Create reference solution with very fine time stepping
    x_center, y_center = nx/2, nx/2
    sigma = nx/10
    u0 = [exp(-((x-x_center)^2 + (y-y_center)^2)/(2*sigma^2)) for y in 1:nx, x in 1:nx]
    u0 = u0 / sum(u0)
    
    dt_ref = 0.0001
    steps_ref = 1000000
    # Test different time steps
    dt_values = exp10.(collect(-3:3))
    lie_sums = Float64[]
    strang_sums = Float64[]
    swss_sums = Float64[]
    adi_sums = Float64[]
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
        push!(lie_sums, sum(u_lie))
        push!(strang_sums, sum(u_strang))
        push!(swss_sums, sum(u_swss))
        push!(adi_sums, sum(u_adi))
        #push!(fd_errors, norm(u_fd - u_ref))
    end
    
    # Plot convergence
    p = plot(dt_values, [lie_sums strang_sums swss_sums adi_sums], 
             xlabel="Time step", ylabel="Sum", 
             label=["Lie" "Strang" "SWSS" "ADI"], 
             xscale=:log10, yscale=:log10,
             marker=:circle, legend=:bottomright)
    
    # Add reference slopes
    #plot!(p, x_ref, y_ref1, linestyle=:dash, color=:black, label="First order")
    #plot!(p, x_ref, y_ref2, linestyle=:dash, color=:gray, label="Second order")
    
    return p
end

