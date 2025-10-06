using CSV, DataFrames, Plots, LaTeXStrings, Printf

# Helper function to format numbers into LaTeX scientific notation
function format_scientific_latex(value)
    # Format to scientific notation with 3 decimal places (e.g., "1.234e-05")
    s = @sprintf("%.3e", value)

    # Split the string at 'e' or 'E'
    parts = split(s, 'e')
    mantissa = parts[1]
    exponent_str = parts[2]

    # Remove leading plus sign for exponent if present, and handle leading zeros for a cleaner look in LaTeX.
    # parse(Int, exponent_str) will correctly handle "05" as 5 and "-05" as -5.
    exponent = parse(Int, exponent_str)

    # Construct the LaTeX string
    return "\$$(mantissa) \\times 10^{$(exponent)}\$"
end

# Read the data
data = CSV.read("mms1_t_ex.tsv", DataFrame, delim = '\t')

# Get unique values for axes
dx_values = sort(unique(data.dx))
dt_values = sort(unique(data.dt))  # Keep in ascending order for heatmap
methods = unique(data.method)

println("Methods found: ", methods)
println("dx values: ", dx_values)
println("dt values: ", dt_values)

# Create a function to generate heatmap for each method
function create_heatmap(method_data, method_name)
    # Create matrix for heatmap
    n_dx = length(dx_values)
    n_dt = length(dt_values)
    error_matrix = fill(NaN, n_dx, n_dt)

    # Fill the matrix
    for row in eachrow(method_data)
        dx_idx = findfirst(x -> x == row.dx, dx_values)
        dt_idx = findfirst(x -> x == row.dt, dt_values)
        error_matrix[dx_idx, dt_idx] = row.error
    end

    # Create heatmap with log2 scales and flipped x-axis
    p = heatmap(
        dt_values,
        dx_values,
        error_matrix,
        title = "L2 error heatmap - $(method_name) - Solution 1",
        xlabel = "dt",
        ylabel = "dx",
        color = :magma,
        aspect_ratio = :auto,
        size = (600, 500),
        xscale = :log2,
        yscale = :log2,
    )

    return p
end

# Create heatmaps for each method
plots = []
for method in methods
    method_data = filter(row -> row.method == method, data)
    p = create_heatmap(method_data, method)
    push!(plots, p)
end

# Display all plots
for (i, p) in enumerate(plots)
    display(p)
    # Optionally save each plot
    savefig(p, "heatmap_$(methods[i]).pdf")
end

# Create a combined plot with all three heatmaps
combined_plot = plot(plots..., layout = (1, 3), size = (1800, 500))
display(combined_plot)
savefig(combined_plot, "combined_heatmaps.pdf")

# Generate LaTeX table for each method
function generate_latex_table(method_data, method_name)
    println("\n" * "="^60)
    println("LaTeX table for method: $method_name")
    println("="^60)

    # Get unique dt values and sort them (largest to smallest for left to right display)
    dt_unique = sort(unique(method_data.dt), rev = true)
    dx_unique = sort(unique(method_data.dx))

    # Print LaTeX table header
    println("\\begin{table}[h!]")
    println("\\centering")
    println("\\caption{Error values for method: $method_name}")
    println("\\begin{tabular}{|c|" * "c|"^length(dt_unique) * "}")
    println("\\hline")

    # Header row
    header = "dx/dt"
    for dt_val in dt_unique
        header *= " & $(format_scientific_latex(dt_val))" # Use the new formatting function
    end
    header *= " \\\\ \\hline"
    println(header)

    # Data rows
    for dx_val in dx_unique
        row_data = "$(format_scientific_latex(dx_val))" # Use the new formatting function
        for dt_val in dt_unique
            error_val = method_data[
                (method_data.dx .== dx_val) .& (method_data.dt .== dt_val),
                :error,
            ]
            if !isempty(error_val)
                row_data *= " & $(format_scientific_latex(error_val[1]))" # Use the new formatting function
            else
                row_data *= " & -"
            end
        end
        row_data *= " \\\\ \\hline"
        println(row_data)
    end

    println("\\end{tabular}")
    println("\\end{table}")
    println()
end

# Generate LaTeX tables for each method
for method in methods
    method_data = filter(row -> row.method == method, data)
    generate_latex_table(method_data, method)
end

println("Heatmaps created successfully!")
println("Individual plots saved as: heatmap_lie.pdf, heatmap_strang.pdf, heatmap_adi.pdf")
println("Combined plot saved as: combined_heatmaps.pdf")
println("LaTeX tables printed above for all methods")
