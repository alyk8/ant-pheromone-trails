import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from configparser import ConfigParser # to read config file
import random

config = ConfigParser()
config.read('config.ini')

grid_size = np.array([int(config.get('trails', 'grid_size_x')), int(config.get('trails', 'grid_size_y'))], dtype=np.uint32)
nest_loc = np.array([(grid_size[0]*int(config.get('trails', 'nest_loc_x')))//100,
                     (grid_size[1]*int(config.get('trails', 'nest_loc_y')))//100], dtype=np.uint32)
ants_int = int(config.get('trails', 'ants_int')) # initial no. of ants

grid_marks = np.zeros(shape=grid_size, dtype=np.uint8) # stores locs + levels of markers
ants_locs = np.full(shape=(ants_int, 2), fill_value=nest_loc, dtype=np.uint32) # start all ants at the nest

grid_food = np.zeros(shape=grid_size, dtype=np.uint8) # 1 = food, 0 = no food
food_int = int(config.get('trails', 'food_int')) # initial number of food locations
food_count = 0
while food_count < food_int: # adds food to random locations
    x = random.randint(0, grid_size[0]-1)
    y = random.randint(0, grid_size[1]-1)
    if grid_food[x, y] == 0 and (x != nest_loc[0] or y != nest_loc[1]): # ensures location has no food and is not the nest
        grid_food[x, y] = 1
        food_count += 1

fig, ax = plt.subplots()
ax.set_xlim(0, grid_size[1]) # matplotlib swaps x and y axes
ax.set_ylim(0, grid_size[0])

food_rows, food_cols = np.where(grid_food == 1) # finds all the food 
ax.scatter(food_cols, food_rows, color='red', marker='x', s=50, label='Food') # plots food locations
ax.scatter(nest_loc[1], nest_loc[0], color='b', marker='x', s=100, linewidth=2) # plots the nest location

marks_plot = ax.imshow(np.zeros(grid_size), cmap='Greens', alpha=0.7, vmin=0, vmax=ants_int//2) # spaces with many markers will be darkest green
def update(frame): # moves ants by a simple random walk
    for ant in range(ants_int): # moves ants 1 step in a random direction but keeps it on the grid
        ants_locs[ant, 0] = min(max(0, ants_locs[ant, 0] + random.randint(-1, 1)), grid_size[0]-1)
        ants_locs[ant, 1] = min(max(0, ants_locs[ant, 1] + random.randint(-1, 1)), grid_size[1]-1)

        grid_marks[ants_locs[ant, 0], ants_locs[ant, 1]] += 1 # adds marker

    marks_plot.set_data(np.ma.masked_where(grid_marks.T == 0, grid_marks.T)) # only updates changes to grid_markers
    return [marks_plot]

ani = animation.FuncAnimation(fig, update, frames=200, interval=50, blit=True) # 50ms between frames

def format_coord(x, y): # changes the coordinate readout to integers when hovering
    return 'x =% 2.0f, y =% 2.0f' % (x, y)
ax.format_coord = format_coord

fig.suptitle('Ant Simulation', fontweight ="bold")
plt.show()