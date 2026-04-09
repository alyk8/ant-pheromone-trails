import numpy as np
import matplotlib.pyplot as plt
import random

grid_size = np.array([100, 100], dtype=np.uint32)
nest_loc = np.array([grid_size[0]//2, grid_size[1]//2], dtype=np.uint32) # puts nest in the middle of the grid
ants_int = np.array(10, dtype=np.uint32) # initial no. of ants

grid_marks = np.zeros(shape=grid_size, dtype=np.uint8) # stores locs + levels of markers
ants_locs = np.full(shape=(ants_int, 2), fill_value=nest_loc, dtype=np.uint32) # start all ants at the nest

grid_food = np.zeros(shape=grid_size, dtype=np.uint8) # 1 = food, 0 = no food
food_int = 5 # initial number of food locations
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

def format_coord(x, y): # changes the coordinate readout to integers when hovering
    return 'x =% 2.0f, y =% 2.0f' % (x, y)
ax.format_coord = format_coord

#fig.suptitle('Grid', fontweight ="bold")
plt.show()