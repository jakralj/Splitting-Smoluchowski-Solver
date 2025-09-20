using DelimitedFiles
using Printf
using Statistics
using Plots

function view_file(filename::String; region=nothing, stats=true)
    """
    View an output file with optional region selection and statistics
    
    Parameters:
    - filename: name of the file to view
    - region: tuple (row_start:row_end, col_start:col_end) to view specific region
    - stats: whether to show statistics
    """
    
    if !isfile(filename)
        println("Error: File '$filename' not found!")
        return
    end
    
    println("=" ^ 50)
    println("Viewing file: $filename")
    println("=" ^ 50)
    
    # Read the data
    data = readdlm(filename)
    rows, cols = size(data)
    println("File dimensions: $rows × $cols")
    
    # Show statistics if requested
    if stats
        println("\nStatistics:")
        println("  Min value: $(minimum(data))")
        println("  Max value: $(maximum(data))")
        println("  Mean: $(mean(data))")
        println("  Non-zero elements: $(count(!iszero, data))")
        println("  Percentage non-zero: $(@sprintf("%.2f", 100 * count(!iszero, data) / length(data)))%")
    end
    
    # Determine what region to show
    if region === nothing
        # Show a sample from the center and corners
        println("\nSample from corners and center:")
        
        # Top-left corner
        println("\nTop-left (1:5, 1:5):")
        show_region(data, 1:min(5,rows), 1:min(5,cols))
        
        # Center region
        center_r = max(1, rows÷2-2):min(rows, rows÷2+2)
        center_c = max(1, cols÷2-2):min(cols, cols÷2+2)
        println("\nCenter region ($center_r, $center_c):")
        show_region(data, center_r, center_c)
        
        # Bottom-right corner
        end_r = max(1, rows-4):rows
        end_c = max(1, cols-4):cols
        println("\nBottom-right ($end_r, $end_c):")
        show_region(data, end_r, end_c)
        
    else
        # Show specified region
        row_range, col_range = region
        println("\nShowing region ($row_range, $col_range):")
        show_region(data, row_range, col_range)
    end
    
    println("\n" * "=" ^ 50)
end

function show_region(data, row_range, col_range)
    """Helper function to display a region of the data"""
    region_data = data[row_range, col_range]
    
    # Format and display
    for i in 1:size(region_data, 1)
        for j in 1:size(region_data, 2)
            @printf("%10.4e ", region_data[i, j])
        end
        println()
    end
end

function list_output_files()
    """List all available output files"""
    files = filter(f -> endswith(f, ".in"), readdir("."))
    if isempty(files)
        println("No .in files found in current directory")
    else
        println("Available output files:")
        for (i, file) in enumerate(files)
            println("  $i. $file")
        end
    end
    return files
end

# Interactive function
function plot_file(filename::String; region=nothing, colormap=:viridis, title_suffix="")
    """
    Plot an output file as a heatmap
    
    Parameters:
    - filename: name of the file to plot
    - region: tuple (row_start:row_end, col_start:col_end) to plot specific region
    - colormap: color scheme (:viridis, :plasma, :inferno, :magma, :hot, :cool, etc.)
    - title_suffix: additional text for the plot title
    """
    
    if !isfile(filename)
        println("Error: File '$filename' not found!")
        return
    end
    
    # Read the data
    data = readdlm(filename)
    rows, cols = size(data)
    
    # Select region to plot
    if region !== nothing
        row_range, col_range = region
        plot_data = data[row_range, col_range]
        plot_title = "$filename $title_suffix (Region: $row_range, $col_range)"
    else
        plot_data = data
        plot_title = "$filename $title_suffix (Full: $rows×$cols)"
    end
    
    # Create the heatmap
    p = heatmap(plot_data, 
               c=colormap,
               aspect_ratio=:equal,
               title=plot_title,
               xlabel="X (pixels)",
               ylabel="Y (pixels)",
               yflip=true)  # Flip Y to match typical image orientation
    
    # Add statistics to the plot
    stats_text = @sprintf("Min: %.2e\nMax: %.2e\nMean: %.2e", 
                         minimum(plot_data), maximum(plot_data), mean(plot_data))
    
    annotate!(p, size(plot_data, 2) * 0.02, size(plot_data, 1) * 0.98, 
              text(stats_text, :left, 8, :white))
    
    display(p)
    return p
end

function plot_comparison(filenames::Vector{String}; region=nothing, colormap=:viridis)
    """
    Plot multiple files side by side for comparison
    """
    
    plots_array = []
    
    for filename in filenames
        if !isfile(filename)
            println("Warning: File '$filename' not found, skipping...")
            continue
        end
        
        data = readdlm(filename)
        
        if region !== nothing
            row_range, col_range = region
            plot_data = data[row_range, col_range]
            plot_title = "$filename\n(Region: $row_range, $col_range)"
        else
            plot_data = data
            plot_title = filename
        end
        
        p = heatmap(plot_data,
                   c=colormap,
                   aspect_ratio=:equal,
                   title=plot_title,
                   yflip=true,
                   titlefontsize=10)
        
        push!(plots_array, p)
    end
    
    if length(plots_array) > 0
        combined_plot = plot(plots_array..., layout=(1, length(plots_array)), size=(400*length(plots_array), 400))
        display(combined_plot)
        return combined_plot
    else
        println("No valid files to plot")
        return nothing
    end
end

function plot_derivatives_overview()
    """
    Plot all derivative files in a 2x2 grid for overview
    """
    derivative_files = ["pmf.in", "pmf_dx.in", "pmf_dy.in", "pmf_dxx.in", "pmf_dyy.in"]
    titles = ["PMF Values", "∂PMF/∂x", "∂PMF/∂y", "∂²PMF/∂x²", "∂²PMF/∂y²"]
    
    plots_array = []
    
    for (i, filename) in enumerate(derivative_files)
        if isfile(filename)
            data = readdlm(filename)
            
            p = heatmap(data,
                       c=:RdBu_r,  # Red-Blue colormap good for derivatives
                       aspect_ratio=:equal,
                       title=titles[i],
                       yflip=true,
                       titlefontsize=12)
            
            push!(plots_array, p)
        else
            # Create empty plot if file doesn't exist
            p = plot(title=titles[i] * "\n(File not found)", 
                    titlefontsize=12, showaxis=false, grid=false)
            push!(plots_array, p)
        end
    end
    
    combined_plot = plot(plots_array..., layout=(2, 3), size=(1200, 800))
    display(combined_plot)
    return combined_plot
end

function interactive_plotter()
    """
    Interactive plotting interface
    """
    files = list_output_files()
    if isempty(files)
        return
    end
    
    println("\n=== Interactive Plotter ===")
    println("Options:")
    println("  1. Plot single file")
    println("  2. Compare multiple files")
    println("  3. Plot derivatives overview")
    println("  4. Plot specific region")
    
    println("\nEnter option number:")
    option = readline()
    
    if option == "1"
        println("Enter file number or filename:")
        input = readline()
        filename = get_filename(input, files)
        if filename !== nothing
            plot_file(filename)
        end
        
    elseif option == "2"
        println("Enter file numbers or names (comma-separated):")
        input = readline()
        filenames = String[]
        for item in split(input, ",")
            filename = get_filename(strip(item), files)
            if filename !== nothing
                push!(filenames, filename)
            end
        end
        if !isempty(filenames)
            plot_comparison(filenames)
        end
        
    elseif option == "3"
        plot_derivatives_overview()
        
    elseif option == "4"
        println("Enter filename:")
        input = readline()
        filename = get_filename(input, files)
        if filename !== nothing
            println("Enter row range (start:end):")
            row_input = readline()
            println("Enter column range (start:end):")
            col_input = readline()
            
            try
                row_range = eval(Meta.parse(row_input))
                col_range = eval(Meta.parse(col_input))
                plot_file(filename, region=(row_range, col_range))
            catch e
                println("Invalid range format: $e")
            end
        end
    end
end

function get_filename(input, files)
    """Helper function to get filename from user input"""
    try
        file_num = parse(Int, input)
        if 1 <= file_num <= length(files)
            return files[file_num]
        else
            println("Invalid file number")
            return nothing
        end
    catch
        if input in files || isfile(input)
            return input
        else
            println("File not found: $input")
            return nothing
        end
    end
end

# Usage examples:
println("File Viewer and Plotter for Output Files")
println()
println("Text viewing commands:")
println("  list_output_files()     - List all .in files")
println("  view_file(\"filename\")   - View a specific file as text")
println()
println("Plotting commands:")
println("  plot_file(\"filename\")   - Plot file as heatmap")
println("  plot_derivatives_overview() - Plot all derivatives in 2x3 grid")
println("  plot_comparison([\"file1\", \"file2\"]) - Compare multiple files")
println("  interactive_plotter()   - Interactive plotting interface")
println()
println("Examples:")
println("  plot_file(\"pmf.in\")")
println("  plot_file(\"pmf_dx.in\", region=(400:500, 400:500))")
println("  plot_comparison([\"pmf_dx.in\", \"pmf_dy.in\"])")
println()
println("Quick start: Run interactive_plotter() or plot_derivatives_overview()")

# Auto-run the file lister to show available files
interactive_plotter()
