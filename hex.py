import numpy as np
import math
import random
import json
from scipy.ndimage import gaussian_filter
# Define the size of the grid
grid_width = 1000
grid_height = 1000
inner_radius = 20
pmf1 = -0.367
outer_radius = 10
pmf2 = -2.224
target_ratio = 0.60
capillary_radius = 2
num_capillary = 10
capillary_k = -0.0171e-9
n_smooth = 10


def random_placement(number, hexagon_points, width, big_radius):

    random_points = random.sample(list(hexagon_points), number)

    placed_circles = set()
    for random_point in random_points:
        alpha = random.choice(range(30,360,60))
        x = random_point[0] + width / math.sqrt(3) * math.cos(math.radians(alpha))
        y = random_point[1] + width / math.sqrt(3) * math.sin(math.radians(alpha))
        placed_circles.add((int(x),int(y)))

    points_to_remove=[]
    for point in placed_circles:
        dist = distance(center_x, center_y, point[0], point[1])
        if dist >= big_radius:
            points_to_remove.append(point)
    placed_circles.difference_update(points_to_remove)

    return placed_circles


def place_neuron(grid, center_x, center_y, inner_radius, outer_radius, type1, type2):
    height, width = grid.shape

    dict_entry = {}

    for y in range(int(max(center_y - outer_radius, 0)), int(min(center_y + outer_radius + 1, height))):
        for x in range(int(max(center_x - outer_radius, 0)), int(min(center_x + outer_radius + 1, width))):
            distance_squared = (x - center_x)**2 + (y - center_y)**2

            if distance_squared < outer_radius**2:
                if distance_squared < inner_radius**2:
                    grid[y, x] = type1  # Inner circle neuron type
                    dict_entry.setdefault(type1, []).append((x,y))
                else:
                    grid[y, x] = type2  # Outer mantle neuron type
                    dict_entry.setdefault(type2, []).append((x,y))
    return dict_entry


def place_capillary(grid, center_x, center_y, radius, type1):
    height, width = grid.shape

    dict_entry = {}

    for y in range(int(max(center_y - outer_radius, 0)), int(min(center_y + outer_radius + 1, height))):
        for x in range(int(max(center_x - outer_radius, 0)), int(min(center_x + outer_radius + 1, width))):
            distance_squared = (x - center_x)**2 + (y - center_y)**2

            if distance_squared < radius**2:
                grid[y, x] = type1  # Inner circle neuron type
                dict_entry.setdefault(type1, []).append((x,y))
    return dict_entry


def distance(x1, y1, x2, y2):
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)


def hexagonal_place(center_x, center_y, radius, big_radius, ratio):
    assert ratio < 0.906 and ratio > 0

    hexagon_width = math.sqrt(radius**2*math.pi*2 / (math.sqrt(3) * ratio))

    placed_circles=set()

    def hexagon(x,y,width) -> list:
        points = [(x+width*math.cos(math.radians(alpha)), y+width*math.sin(math.radians(alpha))) for alpha in range(0, 360, 60)]
        points.append((int(x),int(y)))
        return points

    col_width = hexagon_width * 3
    row_height = math.sin(math.pi / 3) * hexagon_width

    for row in range(-int(big_radius/row_height), int(big_radius/row_height)+1):
        for col in range(-int(big_radius/col_width), int(big_radius/col_width)+1):
            x = (col + 0.5 * (row % 2)) * col_width
            x += center_x
            y = row * row_height
            y += center_y

            placed_circles.update(hexagon(x,y,hexagon_width))

    points_to_remove=[]
    for point in placed_circles:
        dist = distance(center_x, center_y, point[0], point[1])
        if dist >= big_radius:
            points_to_remove.append(point)
    placed_circles.difference_update(points_to_remove)

    return hexagon_width, placed_circles


# calculate some parameters
center_x = grid_width // 2
center_y = grid_height // 2
print((center_x, center_y))
big_radius = min(center_x, center_y) - 2
print(big_radius)

# Create an empty grid filled with ECF (Extracellular Fluid) represented by 0
empty_grid = np.zeros((grid_height, grid_width), dtype=float)
# Initialize the grid with the big circle
grid_pmf = np.copy(empty_grid)
grid_sink = np.copy(empty_grid)
grid_u = np.copy(empty_grid)
grid_u[0] = 1
print("trying to place neurons")
hexagon_width, neuron_points = hexagonal_place(center_x, center_y, outer_radius, big_radius-outer_radius, target_ratio)

print(len(neuron_points))

print("trying to place capillaries")
capillary_points = random_placement(num_capillary, neuron_points, hexagon_width, big_radius)
print(len(capillary_points))
print("done")


print("placing into grid")
neurons_list = []
for x,y in neuron_points:
    dict_entry = place_neuron(grid_pmf, x, y, inner_radius, outer_radius, pmf1, pmf2)
    neurons_list.append(dict_entry)
print("placed neurons")
capillary_list = []
for x,y in capillary_points:
    dict_entry = place_capillary(grid_sink, x, y, capillary_radius, capillary_k)
    capillary_list.append(dict_entry)
print("placed capillaries")
#for x in range(20, 40):
#    for y in range(20, 40):
#        grid_u[y,x] = 1

#smoothing Diriclet
grid_u[0, :] = 1;
grid_u[1, :] = 1;
grid_pmf = gaussian_filter(grid_pmf, sigma=3)
grid_u = gaussian_filter(grid_u, sigma=3)
print("done smoothing pmf")

np.savetxt('pmf.in', grid_pmf, fmt="%.4e", delimiter=" ")
np.savetxt('sink.in', grid_sink, fmt="%.4e", delimiter=" ")
np.savetxt('u.in', grid_u, fmt="%.4e", delimiter=" ")

print("saved .in files")

# Combine metadata and circle data
combined_data = {
    "metadata": {
        "grid_x" : grid_height,
        "grid_y" : grid_width,
        "inner_radius" : inner_radius,
        "outer_radius" : outer_radius },
    "neurons_data": neurons_list,
    "capillary_data": capillary_list
}
print(len(neurons_list))
print(len(capillary_list))
# Save the data to a JSON file
output_filename = 'circle_data.json'
with open(output_filename, 'w') as f:
    json.dump(combined_data, f, indent=4)
print("dumped json")
