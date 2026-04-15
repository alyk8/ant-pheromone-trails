import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from configparser import ConfigParser

def initialise(directions):
    config = ConfigParser()
    config.read('config.ini') # reads in values from config file

    grid_size = np.array([int(config.get('trails', 'grid_size_x')), int(config.get('trails', 'grid_size_y'))], dtype=np.uint32)
    grid_marks = np.zeros(shape=grid_size, dtype=np.float32) # stores locs + levels of markers
    nest_loc = np.array([(grid_size[0]*int(config.get('trails', 'nest_loc_x')))//100,
                        (grid_size[1]*int(config.get('trails', 'nest_loc_y')))//100], dtype=np.uint32)
    
    food_num = int(config.get('trails', 'food_num')) # no. of food sources
    food_locs = np.zeros(shape = [food_num, 3], dtype = np.uint32) # stores locs of food sources + their level
    for f in range(food_num):
        # ensures the food is not placed too close to the nest (i.e. within 5% of the grid size)
        food_locs[f, 0] = random.choice([random.randint(0, int(nest_loc[0]*0.95)), random.randint(int(nest_loc[0]*1.05), grid_size[0])])
        food_locs[f, 1] = random.choice([random.randint(0, int(nest_loc[1]*0.95)), random.randint(int(nest_loc[1]*1.05), grid_size[1])])
        food_locs[f, 2] = 1 # diminishing supply (scale from 0-1)
    
    ants_int = int(config.get('trails', 'ants_int')) # initial no. of ants
    ants_locs = np.full(shape=(ants_int, 2), fill_value=nest_loc, dtype=np.uint32) # start all ants at the nest
    ants_dirs = np.array([random.choice(directions) for _ in range(ants_int)]) # each ant is given a random starting direction
    ants_sens = np.ones(ants_int, dtype=np.float32) # ant sensitivity to markers (initially set to 1 for all ants)
    ants = np.column_stack((ants_locs, ants_dirs, ants_sens)) # combines ants info into a single array: [x, y, dx, dy, sensitivity]

    alpha = float(config.get('trails', 'alpha')) # persistence parameter (i.e. probability that the ant will change direction)
    decay_rate = float(config.get('trails', 'decay_rate')) # environmental decay applied to all markers
    steps = int(config.get('trails', 'steps')) # no. of steps to simulate
    
    return grid_size, grid_marks, nest_loc, ants_int, ants, alpha, decay_rate, steps, food_num, food_locs

def grid(grid_size, grid_marks, nest_loc, ants_int, ants, alpha, decay_rate, steps, food_num, food_locs, directions):
    fig, ax = plt.subplots(figsize=(8, 6), layout='constrained')
    ax.set_xlim(0, grid_size[1]) # matplotlib swaps x and y axes
    ax.set_ylim(0, grid_size[0])
    ax.scatter(nest_loc[1], nest_loc[0], color='b', marker='x', s=100, linewidth=2, label='Nest') # plots the nest location
    frame_text = ax.text(0.02, 0.95, 'Step 0', transform=ax.transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
    
    for f in range(food_num): # adds food locations to the plot
        if f == 0: # only adds label for the first food source to avoid duplicates in the legend
            ax.scatter(food_locs[f, 1], food_locs[f,0], color='r', marker='o', s=100, linewidth=1, label='Food')
        else:
            ax.scatter(food_locs[f, 1], food_locs[f, 0], color='r', marker='o', s=100, linewidth=1)
    
    def format_coord(x, y): # changes the coordinate readout to integers (shows when hovering over the grid)
        return 'x =% 2.0f, y =% 2.0f' % (x, y)
    ax.format_coord = format_coord

    fig.suptitle('Ant Simulation – Basic Active Walkers', fontweight ="bold")
    fig.legend(loc='upper left')

    marks_plot = ax.imshow(np.zeros(grid_size), cmap='Blues', alpha=0.7, vmin=0, vmax=1) # markers with strength > 1 will be darkest green
    cbar = fig.colorbar(marks_plot, ax=ax) # adds colour scale for markers
    cbar.set_label('Marker Strength')
    def update(frame): # moves ants by a biased random walk
        grid_marks[:] = grid_marks * decay_rate # applies decay rate to all markers
        
        for ant in range(ants_int): # moves all ants one step in a random direction
            if random.random() < alpha: # ant changes direction with probability alpha (persistence parameter)
                temp = ants[ant, 2:3].copy() # [x, y, dx, dy, sensitivity]
                while (ants[ant, 2:3] == temp).all(): # ensures the direction actually changes
                    ants[ant, 2:3] = random.choice(directions) # chooses a random direction
            
            new_x = ants[ant, 0] + ants[ant, 2] # calculates new ant position
            new_y = ants[ant, 1] + ants[ant, 3]

            ants[ant, 0] = max(0, min(new_x, grid_size[0] - 1)) # keeps ant within grid boundaries
            ants[ant, 1] = max(0, min(new_y, grid_size[1] - 1))

            grid_marks[ants[ant, 0], ants[ant, 1]] += 1 # adds marker (may increase strength above 1, but this is capped in the plot)

        marks_plot.set_data(np.ma.masked_where(grid_marks.T < 0.1, grid_marks.T)) # hides markers with strength less than 0.1
        frame_text.set_text(f'Step {frame+1}')
        return marks_plot, frame_text

    ani = animation.FuncAnimation(fig, update, frames=steps, interval=50, blit=False, repeat=False) # 50ms between frames
    plt.show()

def main():
    directions = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)] # the 8 possible movement directions (excluding [0,0])
    grid_size, grid_marks, nest_loc, ants_int, ants, alpha, decay_rate, steps, food_num, food_locs = initialise(directions)

    grid(grid_size, grid_marks, nest_loc, ants_int, ants, alpha, decay_rate, steps, food_num, food_locs, directions)

main()