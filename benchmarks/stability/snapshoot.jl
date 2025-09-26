using LinearAlgebra
using DelimitedFiles
using Plots
using Printf

const D = 0.01
const nx = 100
const dx = 2/100
const kbT = 0.65

include("../../methods.jl")
include("../../utils/Potentials/potentials.jl")

# Generate potential 3 only
μ3, ∇μX3, ∇μY3, ∇2μX3, ∇2μY3 = generate_potential_3()

function lie_splitting_states!(u, dt::Float64, dx, num_steps, μ, ∇μX, ∇μY, ∇2μX, ∇2μY, D, kbT, temp_arrays, save_steps)
    u_temp = temp_arrays.u_half
    states = []
    
    for step in 1:num_steps
        evolve_x!(u_temp, u, μ, ∇μX, ∇2μX, D, dx, dt, kbT, temp_arrays)
        evolve_y!(u, u_temp, μ, ∇μY, ∇2μY, D, dx, dt, kbT, temp_arrays)
        
        if step in save_steps
            push!(states, copy(u))
        end
    end
    return states
end

function strang_splitting_states!(u, dt::Float64, dx, num_steps, μ, ∇μX, ∇μY, ∇2μX, ∇2μY, D, kbT, temp_arrays, save_steps)
    u_temp = temp_arrays.u_half
    states = []
    
    for step in 1:num_steps
        evolve_x!(u_temp, u, μ, ∇μX, ∇2μX, D, dx, dt/2, kbT, temp_arrays)
        evolve_y!(u, u_temp, μ, ∇μY, ∇2μY, D, dx, dt, kbT, temp_arrays)
        evolve_x!(u_temp, u, μ, ∇μX, ∇2μX, D, dx, dt/2, kbT, temp_arrays)
        copyto!(u, u_temp)
        
        if step in save_steps
            push!(states, copy(u))
        end
    end
    return states
end

function adi_scheme_states!(u, dt::Float64, dx::Float64, num_steps, μ, ∇μX, ∇μY, ∇2μX, ∇2μY, D, kbT, temp_arrays, save_steps)
    αx = D*dt/(2*dx^2)
    αy = D*dt/(2*dx^2)
    βx = D*dt/(4*kbT*dx)
    βy = D*dt/(4*kbT*dx)
    states = []
    
    for step in 1:num_steps
        x_direction_sweep!(temp_arrays.u_half, u, αx, αy, βx, βy, dt, ∇μX, ∇μY, ∇2μX, ∇2μY, D, kbT, temp_arrays)
        y_direction_sweep!(u, temp_arrays.u_half, αx, αy, βx, βy, dt, ∇μX, ∇μY, ∇2μX, ∇2μY, D, kbT, temp_arrays)
        
        if step in save_steps
            push!(states, copy(u))
        end
    end
    return states
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

function create_evolution_plot()
    dt = 1/100
    final_time = 100.0
    num_steps = Int(round(final_time / dt))
    
    # Define time points: start (10%), middle (50%), end (90%)
    save_steps = [Int(round(0.1 * num_steps)), Int(round(0.5 * num_steps)), Int(round(0.9 * num_steps))]
    time_labels = ["Early (t=$(0.1*final_time))", "Middle (t=$(0.5*final_time))", "Late (t=$(0.9*final_time))"]
    
    u0 = generate_u0(nx, 1.0)
    
    # Run all three methods and collect states
    temp_arrays_2d = TempArrays2D(eltype(u0), nx, nx)
    temp_arrays_adi = ADITempArrays(eltype(u0), nx, nx)
    
    u_lie = copy(u0)
    u_strang = copy(u0) 
    u_adi = copy(u0)
    
    lie_states = lie_splitting_states!(u_lie, dt, dx, num_steps, μ3, ∇μX3, ∇μY3, ∇2μX3, ∇2μY3, D, kbT, temp_arrays_2d, save_steps)
    strang_states = strang_splitting_states!(u_strang, dt, dx, num_steps, μ3, ∇μX3, ∇μY3, ∇2μX3, ∇2μY3, D, kbT, temp_arrays_2d, save_steps)
    adi_states = adi_scheme_states!(u_adi, dt, dx, num_steps, μ3, ∇μX3, ∇μY3, ∇2μX3, ∇2μY3, D, kbT, temp_arrays_adi, save_steps)
    
    # No longer using consistent color scale - each plot will use its own range
    
    # Create coordinate ranges for plotting
    x_range = range(-1, 1, length=nx)
    y_range = range(-1, 1, length=nx)
    
    # Create 3x3 subplot layout
    plots = []
    method_names = ["Lie Splitting", "Strang Splitting", "ADI Scheme"]
    all_method_states = [lie_states, strang_states, adi_states]
    
    for (method_idx, (method_name, method_states)) in enumerate(zip(method_names, all_method_states))
        for (time_idx, (state, time_label)) in enumerate(zip(method_states, time_labels))
            p = heatmap(x_range, y_range, state', 
                       title="$method_name\n$time_label",
                       xlabel="x", ylabel="y",
                       color=:viridis, aspect_ratio=:equal)
            push!(plots, p)
        end
    end
    
    # Combine all plots in 3x3 layout
    final_plot = plot(plots..., layout=(3,3), size=(1200, 1200))
    
    return final_plot
end

# Execute and display
println("Creating evolution visualization for Potential μ3...")
evolution_plot = create_evolution_plot()
display(evolution_plot)
savefig(evolution_plot, "mu3_evolution.pdf")
println("Plot saved as mu3_evolution.pdf")
