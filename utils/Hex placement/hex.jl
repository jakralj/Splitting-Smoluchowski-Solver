using Random
using ImageFiltering
using DelimitedFiles
using StatsBase: sample

# Define the size of the grid
grid_width = 100
grid_height = 100
inner_radius = 3
pmf1 = -0.367
outer_radius = 7
pmf2 = -2.224
target_ratio = 0.30
capillary_radius = 2
num_capillary = 10
capillary_k = -0.0171e-9
n_smooth = 10
placement_probability = 0.5

function random_placement(number, hexagon_points, width, big_radius, center_x, center_y)
    random_points = sample(collect(hexagon_points), number, replace = false)

    placed_circles = Set{Tuple{Int,Int}}()
    for random_point in random_points
        alpha = rand([30, 90, 150, 210, 270, 330])
        x = random_point[1] + width / sqrt(3) * cos(deg2rad(Float64(alpha)))
        y = random_point[2] + width / sqrt(3) * sin(deg2rad(Float64(alpha)))
        push!(placed_circles, (round(Int, x), round(Int, y)))
    end

    points_to_remove = []
    for point in placed_circles
        dist = distance(
            Float64(center_x),
            Float64(center_y),
            Float64(point[1]),
            Float64(point[2]),
        )
        if dist >= big_radius
            push!(points_to_remove, point)
        end
    end
    setdiff!(placed_circles, points_to_remove)

    return placed_circles
end

function place_neuron!(grid, center_x, center_y, inner_radius, outer_radius, type1, type2)
    height, width = size(grid)

    y_start = max(1, round(Int, center_y - outer_radius))
    y_end = min(height, round(Int, center_y + outer_radius))
    x_start = max(1, round(Int, center_x - outer_radius))
    x_end = min(width, round(Int, center_x + outer_radius))

    for y = y_start:y_end
        for x = x_start:x_end
            distance_squared = (Float64(x) - center_x)^2 + (Float64(y) - center_y)^2

            if distance_squared < outer_radius^2
                if distance_squared < inner_radius^2
                    grid[y, x] = type1  # Inner circle neuron type
                else
                    grid[y, x] = type2  # Outer mantle neuron type
                end
            end
        end
    end
end

function place_capillary!(grid, center_x, center_y, radius, type1)
    height, width = size(grid)

    y_start = max(1, round(Int, center_y - radius))
    y_end = min(height, round(Int, center_y + radius))
    x_start = max(1, round(Int, center_x - radius))
    x_end = min(width, round(Int, center_x + radius))

    for y = y_start:y_end
        for x = x_start:x_end
            distance_squared = (Float64(x) - center_x)^2 + (Float64(y) - center_y)^2

            if distance_squared < radius^2
                grid[y, x] = type1
            end
        end
    end
end

function distance(x1, y1, x2, y2)
    return sqrt((x1 - x2)^2 + (y1 - y2)^2)
end

function hexagonal_place(center_x, center_y, radius, big_radius, ratio)
    @assert ratio < 0.906 && ratio > 0

    hexagon_width = sqrt(radius^2 * π * 2 / (sqrt(3) * ratio))

    placed_circles = Set{Tuple{Float64,Float64}}()

    function hexagon(x, y, width)
        points = [
            (
                x + width * cos(deg2rad(Float64(alpha))),
                y + width * sin(deg2rad(Float64(alpha))),
            ) for alpha = 0:60:300
        ]
        push!(points, (x, y))
        return points
    end

    col_width = hexagon_width * 3
    row_height = sin(π / 3) * hexagon_width

    for row = (-round(Int, big_radius/row_height)):round(Int, big_radius/row_height)
        for col = (-round(Int, big_radius/col_width)):round(Int, big_radius/col_width)
            x = (Float64(col) + 0.5 * (row % 2)) * col_width
            x += Float64(center_x)
            y = Float64(row) * row_height
            y += Float64(center_y)

            hex_points = hexagon(x, y, hexagon_width)
            union!(placed_circles, hex_points)
        end
    end

    points_to_remove = []
    for point in placed_circles
        dist = distance(Float64(center_x), Float64(center_y), point[1], point[2])
        if dist >= big_radius
            push!(points_to_remove, point)
        end
    end
    setdiff!(placed_circles, points_to_remove)

    return hexagon_width, placed_circles
end

function calculate_partial_derivatives(grid)
    height, width = size(grid)

    # First partial derivatives
    grad_x = zeros(height, width)
    grad_y = zeros(height, width)

    # Second partial derivatives
    grad_xx = zeros(height, width)
    grad_yy = zeros(height, width)

    # Calculate first derivatives using central differences
    for i = 2:(height-1)
        for j = 2:(width-1)
            # First derivatives (central difference)
            grad_x[i, j] = (grid[i, j+1] - grid[i, j-1]) / 2.0
            grad_y[i, j] = (grid[i+1, j] - grid[i-1, j]) / 2.0

            # Second derivatives (central difference of first derivatives)
            grad_xx[i, j] = grid[i, j+1] - 2*grid[i, j] + grid[i, j-1]
            grad_yy[i, j] = grid[i+1, j] - 2*grid[i, j] + grid[i-1, j]
        end
    end

    # Handle boundaries with forward/backward differences
    # First row and last row for y derivatives
    for j = 2:(width-1)
        # First row
        grad_y[1, j] = grid[2, j] - grid[1, j]  # forward difference
        grad_yy[1, j] = grid[3, j] - 2*grid[2, j] + grid[1, j]  # forward difference for second derivative

        # Last row  
        grad_y[height, j] = grid[height, j] - grid[height-1, j]  # backward difference
        grad_yy[height, j] = grid[height, j] - 2*grid[height-1, j] + grid[height-2, j]  # backward difference for second derivative

        # x derivatives for boundary rows
        grad_x[1, j] = (grid[1, j+1] - grid[1, j-1]) / 2.0
        grad_x[height, j] = (grid[height, j+1] - grid[height, j-1]) / 2.0
        grad_xx[1, j] = grid[1, j+1] - 2*grid[1, j] + grid[1, j-1]
        grad_xx[height, j] = grid[height, j+1] - 2*grid[height, j] + grid[height, j-1]
    end

    # First column and last column for x derivatives
    for i = 2:(height-1)
        # First column
        grad_x[i, 1] = grid[i, 2] - grid[i, 1]  # forward difference
        grad_xx[i, 1] = grid[i, 3] - 2*grid[i, 2] + grid[i, 1]  # forward difference for second derivative

        # Last column
        grad_x[i, width] = grid[i, width] - grid[i, width-1]  # backward difference
        grad_xx[i, width] = grid[i, width] - 2*grid[i, width-1] + grid[i, width-2]  # backward difference for second derivative

        # y derivatives for boundary columns
        grad_y[i, 1] = (grid[i+1, 1] - grid[i-1, 1]) / 2.0
        grad_y[i, width] = (grid[i+1, width] - grid[i-1, width]) / 2.0
        grad_yy[i, 1] = grid[i+1, 1] - 2*grid[i, 1] + grid[i-1, 1]
        grad_yy[i, width] = grid[i+1, width] - 2*grid[i, width] + grid[i-1, width]
    end

    # Handle corners with forward/backward differences
    # Top-left corner
    grad_x[1, 1] = grid[1, 2] - grid[1, 1]
    grad_y[1, 1] = grid[2, 1] - grid[1, 1]
    grad_xx[1, 1] = grid[1, 3] - 2*grid[1, 2] + grid[1, 1]
    grad_yy[1, 1] = grid[3, 1] - 2*grid[2, 1] + grid[1, 1]

    # Top-right corner
    grad_x[1, width] = grid[1, width] - grid[1, width-1]
    grad_y[1, width] = grid[2, width] - grid[1, width]
    grad_xx[1, width] = grid[1, width] - 2*grid[1, width-1] + grid[1, width-2]
    grad_yy[1, width] = grid[3, width] - 2*grid[2, width] + grid[1, width]

    # Bottom-left corner
    grad_x[height, 1] = grid[height, 2] - grid[height, 1]
    grad_y[height, 1] = grid[height, 1] - grid[height-1, 1]
    grad_xx[height, 1] = grid[height, 3] - 2*grid[height, 2] + grid[height, 1]
    grad_yy[height, 1] = grid[height, 1] - 2*grid[height-1, 1] + grid[height-2, 1]

    # Bottom-right corner
    grad_x[height, width] = grid[height, width] - grid[height, width-1]
    grad_y[height, width] = grid[height, width] - grid[height-1, width]
    grad_xx[height, width] =
        grid[height, width] - 2*grid[height, width-1] + grid[height, width-2]
    grad_yy[height, width] =
        grid[height, width] - 2*grid[height-1, width] + grid[height-2, width]

    return grad_x, grad_y, grad_xx, grad_yy
end

# Main execution
println("Calculating parameters...")
center_x = grid_width ÷ 2
center_y = grid_height ÷ 2
println("Center: ($center_x, $center_y)")
big_radius = min(center_x, center_y) - 2
println("Big radius: $big_radius")

# Create empty grids
empty_grid = zeros(Float64, grid_height, grid_width)
grid_pmf = copy(empty_grid)
grid_sink = copy(empty_grid)
grid_u = copy(empty_grid)
grid_u[1, :] .= 1

println("Trying to place neurons...")
hexagon_width, neuron_points = hexagonal_place(
    center_x,
    center_y,
    outer_radius,
    big_radius - outer_radius,
    target_ratio,
)
println("Number of neuron points: $(length(neuron_points))")

println("Trying to place capillaries...")
capillary_points = random_placement(
    num_capillary,
    neuron_points,
    hexagon_width,
    big_radius,
    center_x,
    center_y,
)
println("Number of capillary points: $(length(capillary_points))")

println("Placing into grid...")
for (x, y) in neuron_points
    if rand() < placement_probability
        place_neuron!(grid_pmf, x, y, inner_radius, outer_radius, pmf1, pmf2)
    end
end
println("Placed neurons")

for (x, y) in capillary_points
    place_capillary!(grid_sink, x, y, capillary_radius, capillary_k)
end
println("Placed capillaries")

# Smoothing with Gaussian filter
println("Smoothing...")
grid_u[1, :] .= 1
grid_u[2, :] .= 1
grid_pmf = imfilter(grid_pmf, Kernel.gaussian(3))
grid_u = imfilter(grid_u, Kernel.gaussian(3))
println("Done smoothing pmf")

# Calculate partial derivatives
println("Calculating partial derivatives...")
grad_x, grad_y, grad_xx, grad_yy = calculate_partial_derivatives(grid_pmf)
println("Done calculating derivatives")

# Save files
println("Saving files...")
writedlm("pmf.in", grid_pmf, ' ')
writedlm("sink.in", grid_sink, ' ')
writedlm("u.in", grid_u, ' ')
writedlm("damjux.in", grad_x, ' ')
writedlm("damjuy.in", grad_y, ' ')
writedlm("d2amjux.in", grad_xx, ' ')
writedlm("d2amjuy.in", grad_yy, ' ')

println("Saved all output files")
println("Files created:")
println("  - pmf.out: PMF values")
println("  - sink.out: Sink values")
println("  - u.out: U values")
println("  - pmf_dx.out: First partial derivative ∂/∂x")
println("  - pmf_dy.out: First partial derivative ∂/∂y")
println("  - pmf_dxx.out: Second partial derivative ∂²/∂x²")
println("  - pmf_dyy.out: Second partial derivative ∂²/∂y²")
