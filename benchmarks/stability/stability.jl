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

helmholtz(u, V) = sum(u .* log.(u) .+ V .* u ./ kbT) * dx^2

function lie_splitting_helmholtz!(
    u,
    dt::Float64,
    dx,
    num_steps,
    μ,
    ∇μX,
    ∇μY,
    ∇2μX,
    ∇2μY,
    D,
    kbT,
    temp_arrays,
)
    u_temp = temp_arrays.u_half  # Reuse the allocated matrix
    h = []
    for step = 1:num_steps
        evolve_x!(u_temp, u, μ, ∇μX, ∇2μX, D, dx, dt, kbT, temp_arrays)
        evolve_y!(u, u_temp, μ, ∇μY, ∇2μY, D, dx, dt, kbT, temp_arrays)
        push!(h, helmholtz(u, μ))
    end
    return h
end

function lie_splitting_helmholtz(
    dt::Float64,
    dx,
    u0,
    num_steps,
    μ,
    ∇μX,
    ∇μY,
    ∇2μX,
    ∇2μY,
    D,
    kbT,
)
    nx, ny = size(u0)
    temp_arrays = TempArrays2D(eltype(u0), nx, ny)
    u = copy(u0)  # Only copy needed for API compatibility
    return lie_splitting_helmholtz!(
        u,
        dt,
        dx,
        num_steps,
        μ,
        ∇μX,
        ∇μY,
        ∇2μX,
        ∇2μY,
        D,
        kbT,
        temp_arrays,
    )
end

function strang_splitting_helmholtz!(
    u,
    dt::Float64,
    dx,
    num_steps,
    μ,
    ∇μX,
    ∇μY,
    ∇2μX,
    ∇2μY,
    D,
    kbT,
    temp_arrays,
)
    u_temp = temp_arrays.u_half  # Reuse the allocated matrix
    h = []
    for step = 1:num_steps
        evolve_x!(u_temp, u, μ, ∇μX, ∇2μX, D, dx, dt/2, kbT, temp_arrays)
        evolve_y!(u, u_temp, μ, ∇μY, ∇2μY, D, dx, dt, kbT, temp_arrays)
        evolve_x!(u_temp, u, μ, ∇μX, ∇2μX, D, dx, dt/2, kbT, temp_arrays)
        # Copy result back to u
        copyto!(u, u_temp)
        push!(h, helmholtz(u, μ))
    end
    return h
end

function strang_splitting_helmholtz(
    dt::Float64,
    dx,
    u0,
    num_steps,
    μ,
    ∇μX,
    ∇μY,
    ∇2μX,
    ∇2μY,
    D,
    kbT,
)
    nx, ny = size(u0)
    temp_arrays = TempArrays2D(eltype(u0), nx, ny)
    u = copy(u0)  # Only copy needed for API compatibility
    return strang_splitting_helmholtz!(
        u,
        dt,
        dx,
        num_steps,
        μ,
        ∇μX,
        ∇μY,
        ∇2μX,
        ∇2μY,
        D,
        kbT,
        temp_arrays,
    )
end

function adi_scheme_helmholtz!(
    u,
    dt::Float64,
    dx::Float64,
    num_steps,
    μ,
    ∇μX,
    ∇μY,
    ∇2μX,
    ∇2μY,
    D,
    kbT,
    temp_arrays,
)
    # Pre-compute coefficients
    αx = D*dt/(2*dx^2)
    αy = D*dt/(2*dx^2)
    βx = D*dt/(4*kbT*dx)
    βy = D*dt/(4*kbT*dx)
    h = []
    # Main time-stepping loop
    for step = 1:num_steps
        # First sweep: x-direction (implicit in x, explicit in y)
        x_direction_sweep!(
            temp_arrays.u_half,
            u,
            αx,
            αy,
            βx,
            βy,
            dt,
            ∇μX,
            ∇μY,
            ∇2μX,
            ∇2μY,
            D,
            kbT,
            temp_arrays,
        )

        # Second sweep: y-direction (implicit in y, explicit in x)
        y_direction_sweep!(
            u,
            temp_arrays.u_half,
            αx,
            αy,
            βx,
            βy,
            dt,
            ∇μX,
            ∇μY,
            ∇2μX,
            ∇2μY,
            D,
            kbT,
            temp_arrays,
        )
        push!(h, helmholtz(u, μ))
    end

    return h
end

function adi_scheme_helmholtz(
    dt::Float64,
    dx::Float64,
    u0,
    num_steps,
    μ,
    ∇μX,
    ∇μY,
    ∇2μX,
    ∇2μY,
    D,
    kbT,
)
    nx, ny = size(u0)
    temp_arrays = ADITempArrays(eltype(u0), nx, ny)
    u = copy(u0)  # Only copy needed for API compatibility
    return adi_scheme_helmholtz!(
        u,
        dt,
        dx,
        num_steps,
        μ,
        ∇μX,
        ∇μY,
        ∇2μX,
        ∇2μY,
        D,
        kbT,
        temp_arrays,
    )
end

function generate_u0(nx, sigma)
    x_range = range(-1, 1, length = nx)
    y_range = range(-1, 1, length = nx)
    u0_matrix = zeros(nx, nx)
    norm_factor = 1 / (2 * π * sigma^2)
    for i = 1:nx
        for j = 1:nx
            x = x_range[i]
            y = y_range[j]
            u0_matrix[i, j] = norm_factor * exp(-(x^2 + y^2) / (2 * sigma^2))
        end
    end
    return u0_matrix
end

function run_benchmark()
    dt_values = [1/25, 1/50, 1/100, 1/200, 1/400]
    final_time = 10.0

    u0 = generate_u0(nx, 1.0)

    potentials = [
        (μ1, ∇μX1, ∇μY1, ∇2μX1, ∇2μY1, "Potential 1"),
        (μ2, ∇μX2, ∇μY2, ∇2μX2, ∇2μY2, "Potential 2"),
        (μ3, ∇μX3, ∇μY3, ∇2μX3, ∇2μY3, "Potential 3"),
    ]

    results = Dict()

    println("Running benchmarks with multiple time steps...")

    for (i, (μ, ∇μX, ∇μY, ∇2μX, ∇2μY, pot_name)) in enumerate(potentials)
        println("Testing $pot_name...")
        results[pot_name] = Dict()

        for dt in dt_values
            num_steps = Int(round(final_time / dt))
            time_vector = dt * (1:num_steps)

            println("  dt = $dt ($(num_steps) steps)")

            # Run all methods
            @time h_lie = lie_splitting_helmholtz(
                dt,
                dx,
                u0,
                num_steps,
                μ,
                ∇μX,
                ∇μY,
                ∇2μX,
                ∇2μY,
                D,
                kbT,
            )
            @time h_strang = strang_splitting_helmholtz(
                dt,
                dx,
                u0,
                num_steps,
                μ,
                ∇μX,
                ∇μY,
                ∇2μX,
                ∇2μY,
                D,
                kbT,
            )
            @time h_adi =
                adi_scheme_helmholtz(dt, dx, u0, num_steps, μ, ∇μX, ∇μY, ∇2μX, ∇2μY, D, kbT)

            results[pot_name][dt] = (
                lie = h_lie,
                strang = h_strang,
                adi = h_adi,
                time = time_vector,
                num_steps = num_steps,
            )

        end
        println()
    end

    return results
end

function create_plots(results)
    # Set up plotting parameters
    plot_colors = [:blue, :red, :green]
    shapes = [:circle, :diamond, :rect]
    method_names = ["Lie Splitting", "Strang Splitting", "ADI"]
    potential_names = ["Potential 1", "Potential 2", "Potential 3"]
    dt_values = [1/25, 1/50, 1/100, 1/200, 1/400]

    # Create plots for each potential showing different dt values
    dt_comparison_plots = []

    for pot_name in potential_names
        p = plot(
            title = "$pot_name - Time Step Comparison",
            xlabel = "Time",
            ylabel = "Free Energy",
            legend = :bottomright,
            linewidth = 1.5,
        )

        # Plot each method with different dt values
        for (method_idx, method_name) in enumerate(method_names)
            method_sym = [:lie, :strang, :adi][method_idx]

            for (dt_idx, dt) in enumerate(dt_values)
                data = results[pot_name][dt]
                h_values = getfield(data, method_sym)

                # Use different line styles for different dt values
                linestyle = [:solid, :dash, :dot, :dashdot, :dashdotdot][dt_idx]
                alpha = 0.7 + 0.3 * (dt_idx / length(dt_values))  # Darker for smaller dt

                plot!(
                    p,
                    data.time,
                    h_values,
                    label = "$method_name (dt=$dt)",
                    color = plot_colors[method_idx],
                    linestyle = linestyle,
                    alpha = alpha,
                    legend=:outertopright
                )
            end
        end
        push!(dt_comparison_plots, p)
    end

    # Create stability analysis plot (final values vs dt)
    stability_plots = []

    for pot_name in potential_names
        p = plot(
            title = "$pot_name - Final Free Energy vs Time Step",
            xlabel = "Time Step (dt)",
            ylabel = "Final Free Energy",
            xticks = (dt_values,["1/25", "1/50", "1/100", "1/200", "1/400"]),
            linewidth = 2,
            xscale = :log,
            legend=:outertopright
        )

        # Extract final values for each method and dt
        for (method_idx, method_name) in enumerate(method_names)
            method_sym = [:lie, :strang, :adi][method_idx]

            final_values = Float64[]
            for dt in dt_values
                data = results[pot_name][dt]
                h_values = getfield(data, method_sym)
                push!(final_values, h_values[end])
            end

            plot!(
                p,
                dt_values,
                final_values,
                label = method_name,
                color = plot_colors[method_idx],
                marker = shapes[method_idx],
                markersize = 4,
                markeralpha = 0.5
            )
        end
        push!(stability_plots, p)
    end

    return dt_comparison_plots, stability_plots
end

# Main execution
println("Starting comprehensive benchmark with multiple time steps...")
results = run_benchmark()

println("Creating plots...")
dt_plots, stability_plots = create_plots(results)

# Display plots
println("\nDisplaying time step comparison plots...")
display(plot(dt_plots..., layout = (1, 3), size = (1800, 500)))
savefig(dt_plots[1], "p1.pdf")
savefig(dt_plots[2], "p2.pdf")
savefig(dt_plots[3], "p3.pdf")

println("Displaying stability analysis plots...")
display(plot(stability_plots..., layout = (1, 3), size = (1800, 500)))
savefig(stability_plots[1], "p4.pdf")
savefig(stability_plots[2], "p5.pdf")
savefig(stability_plots[3], "p6.pdf")
# Save results
println("\nBenchmark completed!")
