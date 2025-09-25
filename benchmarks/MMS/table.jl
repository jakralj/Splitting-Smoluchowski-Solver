using CSV, DataFrames, Printf # Removed Plots, LaTeXStrings as they are not explicitly used for table output anymore

# Helper function to format numbers into LaTeX scientific notation (for error values)
function format_scientific_latex(value)
    s = @sprintf("%.2e", value)
    parts = split(s, 'e')
    mantissa = parts[1]
    exponent = parse(Int, parts[2])
    return "\$$(mantissa) \\times 10^{$(exponent)}\$"
end

# Helper function to format numbers into LaTeX power of 2 notation (for dx, dt values)
function format_powerof2_latex(value)
    if value <= 0
        return "\$" * string(value) * "\$" # Handle non-positive values, though dx/dt should be positive
    end
    exponent = log2(value)
    rounded_exponent = round(Int, exponent) # Round to nearest integer for the exponent
    return "\$2^{$(rounded_exponent)}\$"
end

# Read the data
data = CSV.read("mms2_t_ex.tsv", DataFrame, delim='\t')

# Get unique values for axes
dx_values = sort(unique(data.dx))
dt_values = sort(unique(data.dt)) # dt values will be rows, dx values will be columns
methods = unique(data.method)

println("Methods found: ", methods)
println("dx values: ", dx_values)
println("dt values: ", dt_values)


# Generate LaTeX table for each method
function generate_latex_table(method_data, method_name)
    println("\n" * "="^60)
    println("LaTeX table for method: $method_name (dt as rows, dx as columns)")
    println("="^60)
    
    # dt values will be row headers, dx values will be column headers
    dt_unique = sort(unique(method_data.dt), rev=true) # Reverse for typical display (smaller dt at the bottom)
    dx_unique = sort(unique(method_data.dx))
    
    # Print LaTeX table header
    println("\\begin{table}[h!]")
    println("\\centering")
    println("\\caption{Error values for method: $method_name}")
    println("\\begin{tabular}{|c|" * "c|"^length(dx_unique) * "}") # Number of 'c|' based on dx_unique
    println("\\hline")
    
    # Header row for dx values
    header = "dt/dx" # Changed to reflect the new layout
    for dx_val in dx_unique
        header *= " & $(format_powerof2_latex(dx_val))" # Use new formatting function for dx column headers
    end
    header *= " \\\\ \\hline"
    println(header)
    
    # Data rows
    for dt_val in dt_unique
        row_data = "$(format_powerof2_latex(dt_val))" # Use new formatting function for dt row headers
        for dx_val in dx_unique
            # Access data by swapping dx and dt roles
            error_val = method_data[(method_data.dt .== dt_val) .& (method_data.dx .== dx_val), :error]
            if !isempty(error_val)
                row_data *= " & $(format_scientific_latex(error_val[1]))" # Error values still use scientific notation
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

println("LaTeX tables printed above for all methods")
