import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from configparser import ConfigParser

def initialise(directions):
    config = ConfigParser()
    config.read('config.ini') # reads in values from config file

    grid_size = np.array([int(config.get('trails', 'grid_size_x')), int(config.get('trails', 'grid_size_y'))], dtype=np.uint32)
    grid_marks = np.zeros(shape=grid_size, dtype=np.float32) # outbound pheromone
    nest_loc = np.array([(grid_size[0]*int(config.get('trails', 'nest_loc_x')))//100,
                        (grid_size[1]*int(config.get('trails', 'nest_loc_y')))//100], dtype=np.uint32)
    
    food_num = int(config.get('trails', 'food_num')) # no. of food sources
    food_locs = np.zeros(shape = [food_num, 3], dtype = np.uint32) # stores locs of food sources + their level
    for f in range(food_num):
        # ensures the food is not placed too close to the nest (i.e. within 5% of the grid size)
        food_locs[f, 0] = random.choice([random.randint(0, int(nest_loc[0]*0.95)), random.randint(int(nest_loc[0]*1.05), grid_size[0])])
        food_locs[f, 1] = random.choice([random.randint(0, int(nest_loc[1]*0.95)), random.randint(int(nest_loc[1]*1.05), grid_size[1])])
        food_locs[f, 2] = 1 # diminishing supply (scale from 0-1)
    food_step = float(config.get('trails', 'food_step')) # how much food can an ant eat at once
    
    ants_int = int(config.get('trails', 'ants_int')) # initial no. of ants
    ants_locs = np.full(shape=(ants_int, 2), fill_value=nest_loc, dtype=np.uint32) # start all ants at the nest
    ants_dirs = np.array([random.choice(directions) for _ in range(ants_int)]) # each ant is given a random starting direction
    ants_mark = np.zeros(ants_int, dtype=np.int8) # which marker the ant is currently following (0 if not following any marker)
    ants = np.column_stack((ants_locs, ants_dirs, ants_mark)) # combines ants info into a single array: [x, y, dx, dy, sensitivity, marker]

    alpha = float(config.get('trails', 'alpha')) # persistence parameter (i.e. probability that the ant will change direction)
    decay_rate = float(config.get('trails', 'decay_rate')) # environmental decay applied to all markers
    steps = int(config.get('trails', 'steps')) # no. of steps to simulate
    detection_range = int(config.get('trails', 'detection_range')) # how many directions the ant can detect
    
    return grid_size, grid_marks, nest_loc, food_num, food_locs, food_step, ants_int, ants, alpha, decay_rate, steps, detection_range

def getNewDirection(ant, ants, grid_size, temp_grid_marks, directions, alpha, detection_range):
    if ants[ant, 4] == 0: # if the ant is not currently following a marker
        dirs = [] # stores all possible new directions (i.e. those that are within the grid boundaries and not blocked by food or the nest)
        current_dir = ants[ant, 2:4].copy() # current direction for ant
        for dir in directions:
            check_x = int(ants[ant, 0] + dir[0])
            check_y = int(ants[ant, 1] + dir[1])
            if (check_x >= 0) and (check_x < grid_size[0]) and (check_y >= 0) and (check_y < grid_size[1]): # ensures the cell is within the grid boundaries
                dirs.append(dir)

        if random.random() < alpha: # ant changes direction with probability alpha (persistence parameter)
            idx = next((i for i, d in enumerate(dirs) if np.array_equal(d, current_dir)), None) # finds the index of the current direction in the list of possible new directions (if it exists)
            if idx is not None:
                dirs.pop(idx) # removes current direction from the list of possible new directions
            ants[ant, 2:4] = random.choice(dirs) # chooses a random direction from the remaining options
    elif ants[ant, 4] == 1: # if the ant is currently following marker A
        dirs = [] # stores all possible new directions (i.e. those that are within the grid boundaries and not blocked by food or the nest)
        markers = [] # stores the directions and strengths of detected markers in the ant's direction of movement
        current_dir = ants[ant, 2:4].copy() # current direction for ant
        detects = getDetectionDirections(current_dir, directions, detection_range) # gets the directions the ant can detect markers based on its current direction of movement
        for dir in directions:
            check_x = int(ants[ant, 0] + dir[0])
            check_y = int(ants[ant, 1] + dir[1])
            if (check_x >= 0) and (check_x < grid_size[0]) and (check_y >= 0) and (check_y < grid_size[1]): # ensures the cell is within the grid boundaries
                if dir in detects: # if it's in the ant's field of detection
                    if temp_grid_marks[check_x, check_y] > 0: # checks for markers
                        markers.append([dir, temp_grid_marks[check_x, check_y]]) # adds location and strength of marker
                    else: # if no marker
                        dirs.append(dir)
                else: # if it's not in the ant's field of detection
                    dirs.append(dir)

        if len(markers) == 0: # if no markers are detected
            if random.random() < alpha: # ant changes direction with probability alpha (persistence parameter)
                for i, d in enumerate(dirs):
                     if np.array_equal(d, current_dir): # finds the index of the current direction in the list of possible new directions (if it exists)
                         dirs.pop(i) # removes current direction from this list
                         break
                ants[ant, 2:4] = random.choice(dirs) # chooses a random direction from the remaining options
        elif len(markers) == 1: # if one marker is detected
            ants[ant, 2:4] = markers[0][0] # chooses the direction of the detected marker
        else: # if more than one marker is detected
            total = sum([marker[1] for marker in markers]) # total marker strength across all detected directions
            probs = [marker[1]/total for marker in markers] # probability of choosing each direction is proportional to the marker strength in that direction
            ants[ant, 2:4] = random.choices([marker[0] for marker in markers], weights=probs, k=1)[0] # chooses a direction based on the probabilities
    #elif ants[ant, 4] == 2: # if the ant is currently following marker B

def grid(grid_size, grid_marks, nest_loc, food_num, food_locs, food_step, ants_int, ants, alpha, decay_rate, steps, detection_range, directions):
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

    marks_plot = ax.imshow(np.zeros(grid_size), cmap='Greens', alpha=0.7, vmin=0, vmax=1) # markers with strength > 1 will be darkest green
    cbar = fig.colorbar(marks_plot, ax=ax) # adds colour scale for markers
    cbar.set_label('Marker Strength')
    def update(frame): # moves ants by a biased random walk
        temp_grid_marks = grid_marks.copy() # temp grid to ensure concurrent movement, i.e. ants don't react to markers dropped by other ants in the same step
        
        for ant in range(ants_int): # moves all ants one step in a random direction
            found_food = False
            for f in range(food_num): # checks if the ant has found food
                if (ants[ant, 0:2] == food_locs[f, 0:2]).all() and food_locs[f, 2] > 0: # if the ant is on a cell with food
                    print('Ant', ant, 'found food at', food_locs[f, 0:2], 'with level', str(food_locs[f, 2]))
                    food_locs[f, 2] = max(0, food_locs[f, 2] - food_step) # reduces the food level by a fixed amount (can be adjusted)
                    ants[ant, 4] = 1 # ant starts following marker A
                    ants[ant, 2:4] = -ants[ant, 2:4] # reverses the ant's direction to head back towards the nest
                    found_food = True
                    break

            if not found_food:
                getNewDirection(ant, ants, grid_size, temp_grid_marks, directions, alpha, detection_range) # updates the ant's direction based on the current grid markers

            new_x = ants[ant, 0] + ants[ant, 2] # calculates new ant position
            new_y = ants[ant, 1] + ants[ant, 3]

            ants[ant, 0] = max(0, min(new_x, grid_size[0] - 1)) # keeps ant within grid boundaries
            ants[ant, 1] = max(0, min(new_y, grid_size[1] - 1))

            grid_marks[int(ants[ant, 0]), int(ants[ant, 1])] += 1 # adds marker (may increase strength above 1, but this is capped in the plot)

        marks_plot.set_data(np.ma.masked_where(temp_grid_marks.T < 0.1, grid_marks.T)) # hides markers with strength less than 0.1
        frame_text.set_text(f'Step {frame+1}')

        grid_marks[:] = grid_marks*decay_rate # applies decay rate to all markers
        return marks_plot, frame_text

    ani = animation.FuncAnimation(fig, update, frames=steps, interval=50, blit=False, repeat=False) # 50ms between frames
    plt.show()

def getDetectionDirections(current_dir, directions, range=3):
    idx = directions.index(tuple(current_dir)) # gets index of current direction
    low = (idx - (range-1)//2) % len(directions) # index of the direction to the left of the current direction
    high = (idx + (range-1)//2) % len(directions) # index of the direction to the right of the current direction

    return [directions[low], directions[idx], directions[high]] # returns the 3 directions the ant can detect based on its current direction of movement

def main():
    directions = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)] # the 8 possible movement directions (excluding [0,0])
    grid_size, grid_marks, nest_loc, food_num, food_locs, food_step, ants_int, ants, alpha, decay_rate, steps, detection_range = initialise(directions)

    grid(grid_size, grid_marks, nest_loc, food_num, food_locs, food_step, ants_int, ants, alpha, decay_rate, steps, detection_range, directions)

main()