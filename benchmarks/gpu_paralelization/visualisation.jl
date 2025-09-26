using Plots, DataFrames, CSV, StatsPlots

df = DataFrame(CSV.File("speedup.csv"))
df[!, "Factor"] = df[!, "CPU_MEDIAN"] ./ df[!, "GPU_MEDIAN"]

# Create the plot
p1 = plot(xlabel="Time Steps", ylabel="Execution Time (s)",
         yscale=:log10, linewidth=2,
         title="CPU vs GPU Performance Scaling",
         legend=:outertopright, size=(800, 600))

# Get unique grid sizes and create a color palette
grid_sizes = sort(unique(df.N))
colors = palette(:Set1_5, length(grid_sizes))

# Plot CPU (solid lines) and GPU (dashed lines) for each grid size
for (i, n) in enumerate(grid_sizes)
    subset = df[df.N .== n, :]
    
    # Sort by time steps for proper line connection
    sort!(subset, :T)
    
    # CPU line (solid)
    plot!(p1, subset.T, subset.CPU_AVG, 
          color=colors[i], linestyle=:solid, linewidth=2,
          label="CPU N=$n", marker=:circle, markersize=4)
    
    # GPU line (dashed)
    plot!(p1, subset.T, subset.GPU_AVG, 
          color=colors[i], linestyle=:dash, linewidth=2,
          label="GPU N=$n", marker=:square, markersize=4)
end

# Display the plot
savefig("GPUdt.pdf")

p = plot(xlabel="Grid Size (N)", ylabel="Execution Time (s)",
         yscale=:log10, linewidth=2,
         title="CPU vs GPU Performance Scaling",
         legend=:topleft, size=(800, 600))

# Get unique time steps and create a color palette
time_steps = sort(unique(df.T))
colors = palette(:tab10, length(time_steps))

# Plot CPU (solid lines) and GPU (dashed lines) for each time step
for (i, t) in enumerate(time_steps)
    subset = df[df.T .== t, :]
    
    # Sort by grid size for proper line connection
    sort!(subset, :N)
    
    # CPU line (solid)
    plot!(p, subset.N, subset.CPU_AVG, 
          color=colors[i], linestyle=:solid, linewidth=2,
          label="CPU T=$t", marker=:circle, markersize=4)
    
    # GPU line (dashed)
    plot!(p, subset.N, subset.GPU_AVG, 
          color=colors[i], linestyle=:dash, linewidth=2,
          label="GPU T=$t", marker=:square, markersize=4)
end

savefig("GPUdx.pdf")

using Plots, StatsPlots

# Create the speedup plot
p2 = plot(xlabel="Grid Size (N)", ylabel="GPU Speedup Factor",
         linewidth=2, title="GPU Speedup vs Grid Size",
         legend=:topleft, size=(800, 600))

# Get unique time steps and create a color palette
time_steps = sort(unique(df.T))
colors = palette(:tab10, length(time_steps))

# Plot speedup factor for each time step
for (i, t) in enumerate(time_steps)
    subset = df[df.T .== t, :]
    
    # Sort by grid size for proper line connection
    sort!(subset, :N)
    
    # Speedup line
    plot!(p2, subset.N, subset.Factor, 
          color=colors[i], linestyle=:solid, linewidth=2,
          label="T=$t", marker=:circle, markersize=4)
end

# Add reference line at speedup = 1 (break-even point)
hline!([1.0], linestyle=:dash, color=:red, linewidth=1,
       label="Break-even (no speedup)", alpha=0.7)

savefig("speedup.pdf")
