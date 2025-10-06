using Plots
using DelimitedFiles
# Include the potentials file.
# In a real setup, ensure this path is correct relative to where you run the script.
# For demonstration purposes within this environment, the functions are defined directly below.
include("../../utils/Potentials/potentials.jl")

nx = 100
dx = 2.0 / (nx - 1)

μ1, _, _, _, _ = generate_potential_1(nx, dx)
μ2, _, _, _, _ = generate_potential_2(nx, dx)
μ3, _, _, _, _ = generate_potential_3()

x_range = range(-1, 1, length = nx)
y_range = range(-1, 1, length = nx)

p1 = heatmap(
    x_range,
    y_range,
    μ1,
    title = "Potential 1: x²y + eˣsin(y)",
    c = :magma,
    colorbar = false,
    legend = false,
)

p2 = heatmap(
    x_range,
    y_range,
    μ2,
    title = "Potential 2: xy² + ln(x²+1) + y³",
    c = :magma,
    colorbar = false,
    legend = false,
)

p3 = heatmap(
    x_range,
    y_range,
    μ3,
    title = "Potential 3",
    c = :magma,
    colorbar = false,
    legend = false,
)

plot(
    p1,
    p2,
    p3,
    layout = (1, 3),
    size = (1500, 500),
    link = :all,
    plot_title = "Potential Field Visualizations",
    colorbar = :right,
    left_margin = 5*Plots.mm,
    right_margin = 10*Plots.mm,
    bottom_margin = 5*Plots.mm,
)

savefig("potentials.pdf")

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

sigma_val = 1 # Choose a suitable sigma for visualization
u0_data = generate_u0(nx, sigma_val)

# Create a separate plot for u0
p_u0 = heatmap(
    x_range,
    y_range,
    u0_data,
    title = "Initial concentration: Gaussian σ=1.0",
    c = :magma,
    colorbar = true,
    ylimits = (-1, 1),
    xlimits = (-1, 1),
    size = (600, 600),
)

savefig("initial.pdf")
