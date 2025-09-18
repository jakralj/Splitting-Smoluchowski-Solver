using Plots, DataFrames, CSV
using LaTeXStrings

# Load your data (assuming it's in a CSV file)
# data = CSV.read("convergence_data.csv", DataFrame)

# Or create DataFrame from your data directly:
data = DataFrame(
    dt = [0.0001, 0.0002, 0.0005, 0.001, 0.003, 0.01, 0.03, 
          0.0001, 0.0002, 0.0005, 0.001, 0.003, 0.01, 0.03,
          0.0001, 0.0002, 0.0005, 0.001, 0.003, 0.01, 0.03],
    method = repeat(["Lie", "Strang", "ADI"], inner=7),
    L2_error = [2.2504374508126314e-8, 4.523688799267581e-8, 1.1344154976862568e-7, 
                2.2714045772631198e-7, 3.193787275141704e-5, 2.2806103151711206e-6, 
                0.0003194807501455828,
                2.2682059522720042e-10, 2.2756541132851327e-10, 2.8059143198592765e-10,
                7.252413273670503e-10, 3.188200660101674e-5, 6.981349173832454e-8,
                0.0003189784435378143,
                0.017360979211303237, 0.01736086335448786, 0.017360515889953687,
                0.01735993714029405, 0.017359679956352528, 0.01734959616475071,
                0.017349611217965275],
    mass = [0.9857837217190826, 0.9857836998413666, 0.9857836342085352, 
            0.9857835248215533, 0.9857969629765053, 0.985781556083627, 
            0.9859160705773138,
            0.985783743596825, 0.9857837435966674, 0.9857837435957177,
            0.9857837435923038, 0.9857976186733467, 0.985783743141395,
            0.9859225699628944,
            1.0005223319706578, 1.0005223283554943, 1.0005223175274824,
            1.0005222995634007, 1.0005227850391907, 1.0005219936994754,
            1.000527027111443],
    max_u = [0.502022055891516, 0.5020220548932675, 0.5020220518977109, 
             0.5020220469025004, 0.5020486964889321, 0.5020219564326287, 
             0.5022885305874262,
             0.5020220568896184, 0.5020220568893703, 0.502022056887756,
             0.5020220568820625, 0.5020487263725419, 0.5020220561296103,
             0.5022888241899779,
             0.5023914682432203, 0.5023914849739753, 0.5023915351602029,
             0.5023916187959476, 0.5024180443495597, 0.5023931224568592,
             0.5026574471020109],
    TV = [1.8260064353941108, 1.8260063922878829, 1.8260062620933408, 
          1.8260060421809583, 1.8262233147611602, 1.8260014593295433, 
          1.8281718945711034,
          1.8260064783112158, 1.8260064781820784, 1.8260064772794682,
          1.82600647405599, 1.8262246279000498, 1.826006048567239,
          1.828187446499704,
          1.8458042216144088, 1.8457985692405738, 1.8457816113457055,
          1.8457534351490237, 1.845848504472183, 1.8452626461368524,
          1.8464457982385118]
)

# Plot 1: L2 Error Convergence (Log-Log scale)
p1 = plot(title="Convergence Analysis: L² Error vs Time Step", 
          xlabel=L"\Delta t", ylabel=L"L^2 Error", 
          xscale=:log10, yscale=:log10,
          legend=:bottomright, dpi=300, size=(800, 600))

methods = unique(data.method)
colors = [:blue, :red, :green]
markers = [:circle, :square, :diamond]

for (i, method) in enumerate(methods)
    method_data = filter(row -> row.method == method, data)
    plot!(p1, method_data.dt, method_data.L2_error, 
          label=method, color=colors[i], marker=markers[i], 
          markersize=6, linewidth=2)
end

# Add reference lines for convergence orders
dt_ref = [1e-4, 1e-2]
first_order = [1e-7, 1e-5]  # Slope = 1
second_order = [1e-7, 1e-3]  # Slope = 2
plot!(p1, dt_ref, first_order, label="1st Order", linestyle=:dash, color=:gray)
plot!(p1, dt_ref, second_order, label="2nd Order", linestyle=:dot, color=:gray)

# Plot 2: Mass Conservation
p2 = plot(title="Mass Conservation", 
          xlabel=L"\Delta t", ylabel="Total Mass", 
          xscale=:log10, legend=:topright, dpi=300, size=(800, 600))

for (i, method) in enumerate(methods)
    method_data = filter(row -> row.method == method, data)
    plot!(p2, method_data.dt, method_data.mass, 
          label=method, color=colors[i], marker=markers[i], 
          markersize=6, linewidth=2)
end

# Add reference line at mass = 1.0
hline!(p2, [1.0], label="Perfect Conservation", linestyle=:dash, color=:black)

# Plot 3: Total Variation (Stability indicator)
p3 = plot(title="Total Variation (Stability Indicator)", 
          xlabel=L"\Delta t", ylabel="Total Variation", 
          xscale=:log10, legend=:topright, dpi=300, size=(800, 600))

for (i, method) in enumerate(methods)
    method_data = filter(row -> row.method == method, data)
    plot!(p3, method_data.dt, method_data.TV, 
          label=method, color=colors[i], marker=markers[i], 
          markersize=6, linewidth=2)
end

# Plot 4: Maximum Concentration
p4 = plot(title="Maximum Concentration", 
          xlabel=L"\Delta t", ylabel=L"max(u)", 
          xscale=:log10, legend=:bottomright, dpi=300, size=(800, 600))

for (i, method) in enumerate(methods)
    method_data = filter(row -> row.method == method, data)
    plot!(p4, method_data.dt, method_data.max_u, 
          label=method, color=colors[i], marker=markers[i], 
          markersize=6, linewidth=2)
end

# Plot 5: Combined Error and Mass Conservation
p5 = plot(title="Error vs Mass Conservation Trade-off", 
          xlabel="Mass Conservation Error |Mass - 1.0|", 
          ylabel=L"L^2 Error", 
          yscale=:log10, xscale=:log10, legend=:topright, dpi=300, size=(800, 600))

for (i, method) in enumerate(methods)
    method_data = filter(row -> row.method == method, data)
    mass_error = abs.(method_data.mass .- 1.0)
    scatter!(p5, mass_error, method_data.L2_error, 
            label=method, color=colors[i], marker=markers[i], 
            markersize=8, alpha=0.7)
end

# Plot 6: Convergence Rate Analysis (for fine time steps only)
p6 = plot(title="Convergence Rate Analysis (Fine Time Steps)", 
          xlabel=L"\Delta t", ylabel=L"L^2 Error", 
          xscale=:log10, yscale=:log10, legend=:bottomright, dpi=300, size=(800, 600))

# Filter for small time steps where convergence is clean
fine_steps = [0.0001, 0.0002, 0.0005, 0.001]

for (i, method) in enumerate(["Lie", "Strang"])  # Exclude ADI for clarity
    method_data = filter(row -> row.method == method && row.dt in fine_steps, data)
    plot!(p6, method_data.dt, method_data.L2_error, 
          label=method, color=colors[i], marker=markers[i], 
          markersize=6, linewidth=2)
    
    # Calculate and display convergence rate
    if length(method_data.dt) >= 2
        log_dt = log10.(method_data.dt)
        log_err = log10.(method_data.L2_error)
        # Linear fit to get slope (convergence order)
        A = [log_dt ones(length(log_dt))]
        coeffs = A \ log_err
        convergence_rate = coeffs[1]
        println("$method convergence rate: $(round(convergence_rate, digits=2))")
    end
end

# Create combined layout
combined_plot = plot(p1, p2, p3, p4, p5, p6, layout=(2,3), size=(1800, 1200))

# Save plots
savefig(p1, "convergence_l2_error.pdf")
savefig(p2, "mass_conservation.pdf")  
savefig(p3, "total_variation.pdf")
savefig(p4, "maximum_concentration.pdf")
savefig(p5, "error_mass_tradeoff.pdf")
savefig(p6, "convergence_rates.pdf")
savefig(combined_plot, "combined_convergence_analysis.pdf")

# Display the plots
display(p1)
display(p2)
display(p3)
display(p4)
display(p5)
display(p6)
