using LinearAlgebra
using DelimitedFiles
using DataFrames
using CSV
using Plots
using Printf

const D = 0.01
const nx = 100
const dx = 2/100
const kbT = 0.65

include("../../methods.jl")
include("../../utils/Potentials/potentials.jl")

# Generate all potentials
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

function run_benchmark()
    # Simulation parameters
    dt = 1/100
    num_steps = 1000
    time_vector = dt * (1:num_steps)
    
    # Initial condition
    u0 = generate_u0(nx, 1.0)
    
    # Store all potentials data
    potentials = [
        (μ1, ∇μX1, ∇μY1, ∇2μX1, ∇2μY1, "Potential 1"),
        (μ2, ∇μX2, ∇μY2, ∇2μX2, ∇2μY2, "Potential 2"),
        (μ3, ∇μX3, ∇μY3, ∇2μX3, ∇2μY3, "Potential 3")
    ]
    
    # Store all results
    results = Dict()
    
    println("Running benchmarks...")
    
    for (i, (μ, ∇μX, ∇μY, ∇2μX, ∇2μY, pot_name)) in enumerate(potentials)
        println("Testing $pot_name...")
        
        # Run all methods
        @time h_lie = lie_splitting_helmholtz(dt, dx, u0, num_steps, μ, ∇μX, ∇μY, ∇2μX, ∇2μY, D, kbT)
        @time h_strang = strang_splitting_helmholtz(dt, dx, u0, num_steps, μ, ∇μX, ∇μY, ∇2μX, ∇2μY, D, kbT)
        @time h_adi = adi_scheme_helmholtz(dt, dx, u0, num_steps, μ, ∇μX, ∇μY, ∇2μX, ∇2μY, D, kbT)
        
        results[pot_name] = (
            lie = h_lie,
            strang = h_strang, 
            adi = h_adi,
            time = time_vector
        )
        
        println("  Initial free energy: $(h_lie[1])")
        println("  Final free energies:")
        println("    Lie: $(h_lie[end])")
        println("    Strang: $(h_strang[end])")
        println("    ADI: $(h_adi[end])")
        println()
    end
    
    return results
end

function create_plots(results)
    # Set up plotting parameters
    plot_colors = [:blue, :red, :green]
    method_names = ["Lie Splitting", "Strang Splitting", "ADI"]
    potential_names = ["Potential 1", "Potential 2", "Potential 3"]
    
    # Create individual plots for each potential
    individual_plots = []
    
    for (pot_name, data) in results
        p = plot(title="Free Energy Evolution - $pot_name", 
                xlabel="Time", ylabel="Free Energy",
                legend=:bottomright, linewidth=2)
        
        plot!(p, data.time, data.lie, label="Lie Splitting", color=plot_colors[1])
        plot!(p, data.time, data.strang, label="Strang Splitting", color=plot_colors[2])
        plot!(p, data.time, data.adi, label="ADI", color=plot_colors[3])
        
        push!(individual_plots, p)
    end
    
    # Create comparison plot (all methods, all potentials)
    comparison_plot = plot(title="Free Energy Comparison - All Methods & Potentials",
                          xlabel="Time", ylabel="Free Energy",
                          legend=:outerright, linewidth=2)
    
    method_data = [:lie, :strang, :adi]
    linestyles = [:solid, :dash, :dot]
    
    for (i, pot_name) in enumerate(potential_names)
        data = results[pot_name]
        for (j, method) in enumerate(method_data)
            method_name = method_names[j]
            plot!(comparison_plot, data.time, getfield(data, method), 
                  label="$method_name - $pot_name",
                  color=plot_colors[i], linestyle=linestyles[j])
        end
    end
    
    # Create decay rate comparison (log scale)
    log_plot = plot(title="Free Energy Decay (Log Scale)",
                   xlabel="Time", ylabel="Log|Free Energy - Final|",
                   legend=:topright, linewidth=2, yscale=:log10)
    
    for (i, pot_name) in enumerate(potential_names)
        data = results[pot_name]
        for (j, method) in enumerate(method_data)
            h_values = getfield(data, method)
            h_final = h_values[end]
            decay = abs.(h_values .- h_final) .+ 1e-12  # Add small constant to avoid log(0)
            plot!(log_plot, data.time, decay,
                  label="$(method_names[j]) - $pot_name",
                  color=plot_colors[i], linestyle=linestyles[j])
        end
    end
    
    # Create summary statistics table
    println("\n" * "="^80)
    println("SUMMARY STATISTICS")
    println("="^80)
    
    for pot_name in potential_names
        data = results[pot_name]
        println("\n$pot_name:")
        println(@sprintf("  %-15s %12s %12s %12s %12s", 
                "Method", "Initial F", "Final F", "Total Decay", "Decay Rate"))
        println("-"^65)
        
        methods = [("Lie", data.lie), ("Strang", data.strang), ("ADI", data.adi)]
        
        for (method_name, h_values) in methods
            initial_f = h_values[1]
            final_f = h_values[end]
            total_decay = initial_f - final_f
            decay_rate = total_decay / (length(h_values) * (data.time[2] - data.time[1]))
            
            println(@sprintf("  %-15s %12.4f %12.4f %12.4f %12.4f", 
                    method_name, initial_f, final_f, total_decay, decay_rate))
        end
    end
    
    return individual_plots, comparison_plot, log_plot
end

function save_results_to_csv(results)
    # Create DataFrame with all results
    all_data = DataFrame()
    
    for (pot_name, data) in results
        pot_df = DataFrame(
            Time = data.time,
            Potential = fill(pot_name, length(data.time)),
            Lie_Splitting = data.lie,
            Strang_Splitting = data.strang,
            ADI = data.adi
        )
        all_data = vcat(all_data, pot_df)
    end
    
    CSV.write("benchmark_results.csv", all_data)
    println("Results saved to benchmark_results.csv")
end

# Main execution
println("Starting comprehensive benchmark...")
results = run_benchmark()

println("Creating plots...")
individual_plots, comparison_plot, log_plot = create_plots(results)

# Display plots
display(plot(individual_plots..., layout=(1,3), size=(1200,400)))
savefig("ind.pdf")
display(comparison_plot)
display(log_plot)

# Save results
save_results_to_csv(results)

println("\nBenchmark completed!")
