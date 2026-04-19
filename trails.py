import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from configparser import ConfigParser

def initialise(directions):
    config = ConfigParser()
    config.read('config.ini') # reads in values from config file

    grid_size = np.array([int(config.get('trails', 'grid_size_x')), int(config.get('trails', 'grid_size_y'))], dtype=np.uint32)
    grid_marks = np.zeros(shape=(grid_size[0], grid_size[1], 2), dtype=np.float32) # outbound pheromone
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
    
    ants_pop = np.array([int(config.get('trails', 'ants_num')), int(config.get('trails', 'scout_ants')), int(config.get('trails', 'recruitment_rate'))]) # ant population size, initial number of scout ants, recruitment rate
    ants_locs = np.full(shape=(ants_pop[0], 2), fill_value=nest_loc, dtype=np.float32) # start all ants at the nest
    ants_dirs = np.array([random.choice(directions) for _ in range(ants_pop[0])]) # each ant is given a random starting direction
    ants_mark = np.zeros(ants_pop[0], dtype=np.int8) # which marker the ant is currently following (0 if not following any marker)
    ants_drop = np.ones(ants_pop[0], dtype=np.int8) # how much marker the ant will drop at each step
    ants = np.column_stack((ants_locs, ants_dirs, ants_mark, ants_drop)) # combines ants info into a single array: [x, y, dx, dy, marker, drop_rate]

    alpha = float(config.get('trails', 'alpha')) # persistence parameter (i.e. probability that the ant will change direction)
    decay_rates = np.array([1 - float(config.get('trails', 'marker_decay_rate')), float(config.get('trails', 'drop_rate_decay'))], dtype=np.float32)
    steps = int(config.get('trails', 'steps')) # no. of steps to simulate
    detection_range = int(config.get('trails', 'detection_range')) # how many directions the ant can detect
    
    return grid_size, grid_marks, nest_loc, food_num, food_locs, food_step, ants_pop, ants, alpha, decay_rates, steps, detection_range

def getNewDirection(ant, ants, grid_size, temp_grid_marks, directions, alpha, detection_range):
    dirs = [] # stores all possible new directions (i.e. those that are within the grid boundaries and not blocked by food or the nest)
    current_dir = ants[ant, 2:4].copy() # current direction for ant
    markers = [] # stores the directions and strengths of detected markers in the ant's direction of movement
    detects = [] # directions the ant can detect based on its current direction of movement
    if (int(ants[ant, 4]) == 1) or (int(ants[ant, 4]) == 2): # if the ant is currently following marker A or B
        detects = getDetectionDirections(current_dir, directions, detection_range)

    for dir in directions:
        x = int(ants[ant, 0] + dir[0])
        y = int(ants[ant, 1] + dir[1])
        if (x >= 0) and (x < grid_size[0]) and (y >= 0) and (y < grid_size[1]):
            if dir in detects: # if it's in the ant's field of detection
                if temp_grid_marks[x, y, int(ants[ant, 4]) - 1] > 0: # checks for marker in that direction (ants[ant, 4] - 1 gives the marker)
                    markers.append([dir, temp_grid_marks[x, y, int(ants[ant, 4]) - 1]]) # adds location and strength of marker
                else: # if no marker
                    dirs.append(dir)
            else: # if it's not in the ant's field of detection
                dirs.append(dir)
        
    if len(markers) == 0: # if no markers are detected (or ant is moving randomly)
        if random.random() < alpha: # ant changes direction with probability alpha (persistence parameter)
            for i, d in enumerate(dirs):
                if np.array_equal(d, current_dir):
                    dirs.pop(i) # removes current direction from the list of possible new directions (if it exists)
                    break
            ants[ant, 2:4] = random.choice(dirs) # chooses a random direction from the remaining options
    elif len(markers) == 1: # if one marker is detected
        ants[ant, 2:4] = markers[0][0] # chooses the direction of the detected marker
    else: # if more than one marker is detected
        total = sum([marker[1] for marker in markers]) # total marker strength across all detected directions
        probs = [marker[1]/total for marker in markers] # probability of choosing each direction is proportional to the marker strength in that direction
        ants[ant, 2:4] = random.choices([marker[0] for marker in markers], weights=probs, k=1)[0] # chooses a direction based on the probabilities

def grid(grid_size, grid_marks, nest_loc, food_num, food_locs, food_step, ants_pop, ants, alpha, decay_rates, steps, detection_range, directions):
    fig, ax = plt.subplots(figsize=(8, 6), layout='constrained')
    ax.set_xlim(0, grid_size[0])
    ax.set_ylim(0, grid_size[1])
    ax.scatter(nest_loc[0], nest_loc[1], color='b', marker='x', s=100, linewidth=2, label='Nest') # plots the nest location
    step_counter = fig.text(0.01, 0.01, 'Step 0', fontsize=12)
    ant_counter = fig.text(0.88, 0.01, 'Ants: 0', fontsize=12)

    food_plots = np.zeros(food_num, dtype=object) # stores the plot objects for the food sources to allow updating their colour when food is found
    for f in range(food_num): # adds food locations to the plot
        label = ''
        if f == 0: # only adds label for the first food source to avoid duplicates in the legend
            label = 'Food'
        food_plots[f] = ax.scatter(food_locs[f, 0], food_locs[f, 1], color='r', marker='o', s=100, linewidth=1, label=label)
    
    def format_coord(x, y): # changes the coordinate readout to integers (shows when hovering over the grid)
        return 'x =% 2.0f, y =% 2.0f' % (x, y)
    ax.format_coord = format_coord

    fig.suptitle('Ant Simulation – Basic Active Walkers', fontweight ="bold")
    fig.legend(loc='outside lower center', ncol=2)

    marks_plot_B = ax.imshow(np.zeros(grid_size), cmap='Blues', alpha=0.7, vmin=0, vmax=1) # markers with strength > 1 will be darkest blue
    cbar_B = fig.colorbar(marks_plot_B, ax=ax) # adds colour scale for marker B
    cbar_B.set_label('Marker B')

    marks_plot_A = ax.imshow(np.zeros(grid_size), cmap='Greens', alpha=0.7, vmin=0, vmax=1)
    cbar_A = fig.colorbar(marks_plot_A, ax=ax) # adds colour scale for marker A
    cbar_A.set_label('Marker A')

    ants_int = ants_pop[1] # number of active ants (scout + recruited ants)

    def update(frame): # moves ants by a biased random walk
        nonlocal ants_int # allows the function to modify ants_int variable defined in the outer scope
        temp_grid_marks = grid_marks.copy() # temp grid to ensure concurrent movement, i.e. ants don't react to markers dropped in the same step
        
        for ant in range(ants_int): # moves all ants one step
            found = False # if the ant has found food or returned to nest
            if (int(ants[ant, 4]) == 1) and (ants[ant, 0] >= nest_loc[0]*0.99) and (ants[ant, 0] <= nest_loc[0]*1.01) and (ants[ant, 1] >= nest_loc[1]*0.99) and (ants[ant, 1] <= nest_loc[1]*1.01): # if the ant is within 1% of the nest with food
                print('Ant', ant, 'returned to nest with food at step', frame)
                ants[ant, 4] = 2 # ant starts following marker B
                ants[ant, 2:4] = -ants[ant, 2:4]
                ants[ant, 5] = 1
                for _ in range(ants_pop[2]): # recruits more ants when an ant successfully returns to the nest with food
                    if ants_int < ants_pop[0]: # checks if the total number of ants has not been exceeded
                        ants[ants_int, 2:4] = ants[ant, 2:4] # same direction as food
                        ants[ants_int, 4] = 2 # follows marker B
                        ants_int += 1
                found = True
            
            if (int(ants[ant, 4]) != 1) and not found: # if the ant is searching for food (i.e. not currently following marker A)
                for f in range(food_num): # checks if the ant has found food
                    if (ants[ant, 0] >= food_locs[f, 0]*0.99) and (ants[ant, 0] <= food_locs[f, 0]*1.01) and (ants[ant, 1] >= food_locs[f, 1]*0.99) and (ants[ant, 1] <= food_locs[f, 1]*1.01) and food_locs[f, 2] > 0: # if the ant is within 1% of food source
                        print('Ant', ant, 'found food at', food_locs[f, 0:2], 'with level', str(food_locs[f, 2]), 'at step', frame)
                        food_locs[f, 2] = max(0, food_locs[f, 2] - food_step) # reduces the food level by a fixed amount (can be adjusted)
                        food_plots[f].set_facecolor('orange') # mark food as found
                        ants[ant, 4] = 1 # ant starts following marker A
                        ants[ant, 2:4] = -ants[ant, 2:4] # reverses the ant's direction to head back towards the nest
                        ants[ant, 5] = 1 # resets the ant's drop rate
                        found = True
                        break

            if not found: # updates the ant's direction based on the current grid markers
                getNewDirection(ant, ants, grid_size, temp_grid_marks, directions, alpha, detection_range)

            new_x = ants[ant, 0] + ants[ant, 2] # calculates new ant position
            new_y = ants[ant, 1] + ants[ant, 3]

            ants[ant, 0] = max(0, min(new_x, grid_size[0] - 1)) # keeps ant within grid boundaries
            ants[ant, 1] = max(0, min(new_y, grid_size[1] - 1))

            # drops markers, may increase above 1 but this is capped in the plot
            if int(ants[ant, 4]) == 1: # if the ant is following marker A (i.e. it has found food and is heading back to the nest)
                grid_marks[int(ants[ant, 0]), int(ants[ant, 1]), 1] += ants[ant, 5] # marker B
            else: # if the ant is moving randomly or following marker B (i.e. searching for food)
                grid_marks[int(ants[ant, 0]), int(ants[ant, 1]), 0] += ants[ant, 5] # marker A
            ants[ant, 5] = max(decay_rates[1], ants[ant, 5] - decay_rates[1]) # applies drop rate decay

        # updates the marker plots and removes markers with strength < 0.1 to improve visibility (this threshold can be adjusted)
        marks_plot_A.set_data(np.ma.masked_where(temp_grid_marks[:, :, 0].T < 0.1, grid_marks[:, :, 0].T)) # marker A
        marks_plot_B.set_data(np.ma.masked_where(temp_grid_marks[:, :, 1].T < 0.1, grid_marks[:, :, 1].T)) # marker B
        step_counter.set_text(f'Step {frame+1}')
        ant_counter.set_text(f'Ants: {ants_int}')

        grid_marks[:] = grid_marks*decay_rates[0] # applies decay rate to all markers
        return marks_plot_A, marks_plot_B, step_counter, ant_counter

    ani = animation.FuncAnimation(fig, update, frames=steps, interval=50, blit=False, repeat=False) # 50ms between frames
    plt.show()

def getDetectionDirections(current_dir, directions, range=3):
    idx = directions.index(tuple(current_dir)) # gets index of current direction
    low = (idx - (range-1)//2) % len(directions) # index of the direction to the left of the current direction
    high = (idx + (range-1)//2) % len(directions) # index of the direction to the right of the current direction

    return [directions[low], directions[idx], directions[high]] # returns the 3 directions the ant can detect based on its current direction of movement

def main():
    directions = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)] # the 8 possible movement directions (excluding [0,0])
    grid_size, grid_marks, nest_loc, food_num, food_locs, food_step, ants_pop, ants, alpha, decay_rates, steps, detection_range = initialise(directions)

    grid(grid_size, grid_marks, nest_loc, food_num, food_locs, food_step, ants_pop, ants, alpha, decay_rates, steps, detection_range, directions)

main()