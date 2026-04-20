import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LogNorm
import matplotlib.lines as mlines
from configparser import ConfigParser

def initialise(directions):
    config = ConfigParser()
    config.read('config.ini') # reads in values from config file

    grid_size = np.array([int(config.get('trails', 'grid_size_x')), int(config.get('trails', 'grid_size_y'))], dtype=np.uint32)
    grid_marks = np.zeros(shape=(grid_size[0], grid_size[1], 2), dtype=np.float32) # outbound pheromone
    nest_loc = np.array([(grid_size[0]*int(config.get('trails', 'nest_loc_x')))//100,
                        (grid_size[1]*int(config.get('trails', 'nest_loc_y')))//100], dtype=np.uint32)

    food_num = int(config.get('trails', 'food_num')) # no. of food sources
    food_locs = np.zeros(shape = [food_num, 3], dtype = np.float32) # stores locs of food sources + their level
    for f in range(food_num):
        # ensures the food is not placed too close to the nest (i.e. within 5% of the grid size)
        food_locs[f, 0] = random.choice([random.randint(0, int(nest_loc[0]*0.95)), random.randint(int(nest_loc[0]*1.05), grid_size[0])])
        food_locs[f, 1] = random.choice([random.randint(0, int(nest_loc[1]*0.95)), random.randint(int(nest_loc[1]*1.05), grid_size[1])])
        food_locs[f, 2] = 1 # diminishing supply (scale from 0-1)
    food_step = float(config.get('trails', 'food_step')) # how much food can an ant eat at once
    
    ants_pop = np.array([int(config.get('trails', 'ants_num')), int(config.get('trails', 'scout_ants')), int(config.get('trails', 'recruitment_rate'))]) # ant population size, initial number of scout ants, recruitment rate
    ants_locs = np.full(shape=(ants_pop[0], 2), fill_value=nest_loc, dtype=np.float32) # start all ants at the nest
    ants_dirs = np.array([random.choice(directions) for _ in range(ants_pop[0])], dtype=np.float32) # each ant is given a random starting direction
    ants_mark = np.zeros(ants_pop[0], dtype=np.float32) # which marker the ant is currently following (0 if not following any marker)
    ants_drop = np.ones(ants_pop[0], dtype=np.float32) # how much marker the ant will drop at each step
    ants_sens = np.full(ants_pop[0], fill_value=0.99, dtype=np.float32) # how sensitive the ant is to the markers
    ants_act = np.zeros(ants_pop[0], dtype=np.float32) # whether the ant is active
    ants_act[:ants_pop[1]] = 1 # initialise the first 'scout_ants' as active
    ants = np.column_stack((ants_locs, ants_dirs, ants_mark, ants_drop, ants_sens, ants_act)) # [x, y, dx, dy, marker, drop_rate, sensitivity, active]

    alpha = float(config.get('trails', 'alpha')) # persistence parameter (i.e. probability that the ant will change direction)
    decay_rates = np.array([float(config.get('trails', 'marker_decay_rate')), float(config.get('trails', 'drop_rate_decay')), float(config.get('trails', 'sensitivity_decay'))], dtype=np.float32) # marker, drop rate and sensitivity decay rates
    steps = int(config.get('trails', 'steps')) # no. of steps to simulate
    detection_range = int(config.get('trails', 'detection_range')) # how many directions the ant can detect
    
    return grid_size, grid_marks, nest_loc, food_num, food_locs, food_step, ants_pop, ants, alpha, decay_rates, steps, detection_range

def getNewDirection(ant, ants, grid_size, temp_grid_marks, directions, alpha, detection_range):
    dirs = [] # stores all possible new directions (i.e. those that are within the grid boundaries and not blocked by food or the nest)
    forward_dirs = [] # stores valid forward directions
    current_dir = ants[ant, 2:4].copy() # ant's current direction
    markers = [] # stores the directions and strengths of detected markers
    detects = getForwardDirections(current_dir, directions, detection_range)

    for dir in directions:
        x = int(ants[ant, 0] + dir[0])
        y = int(ants[ant, 1] + dir[1])
        if (x >= 0) and (x < grid_size[0]) and (y >= 0) and (y < grid_size[1]):
            if dir in detects: # if it's in the ant's field of detection
                if (temp_grid_marks[x, y, max(0, int(ants[ant, 4]) - 1)] > 0.01): # checks for marker A or B in that direction
                    markers.append([dir, temp_grid_marks[x, y, max(0, int(ants[ant, 4]) - 1)]]) # adds location and strength of marker
                elif dir != tuple(current_dir): # if no marker and not current direction
                    forward_dirs.append(dir)
            else: # if it's not in the ant's field of detection
                dirs.append(dir)
    
    if (len(markers) > 0) and (random.random() < ants[ant, 6]): # if the ant detects markers and is sensitive enough to react to them
        if len(markers) == 1: # if only one marker is detected, the ant will choose that direction
            ants[ant, 2:4] = markers[0][0]
        elif len(markers) >= 2: # if multiple markers are detected, the ant chooses the strongest one
            max_strength = max([marker[1] for marker in markers]) # finds the absolute highest marker strength in the ant's field of view
            best_dirs = [marker[0] for marker in markers if marker[1] == max_strength] # find all directions that share this max strength
            ants[ant, 2:4] = random.choice(best_dirs) # choose randomly among the tied best directions
    elif random.random() < alpha: # with probability alpha, the ant changes direction randomly
        if len(forward_dirs) > 0: # if there are valid forward directions (i.e. within the ant's field of detection)
            ants[ant, 2:4] = random.choice(forward_dirs)
        elif len(dirs) > 0: # if the ant is stuck in a corner
            ants[ant, 2:4] = random.choice(dirs) # chooses a random direction from the remaining options

def grid(grid_size, grid_marks, nest_loc, food_num, food_locs, food_step, ants_pop, ants, alpha, decay_rates, steps, detection_range, directions):
    fig, ax = plt.subplots(figsize=(8, 6), layout='constrained')
    ax.set_xlim(0, grid_size[0])
    ax.set_ylim(0, grid_size[1])
    ax.scatter(nest_loc[0], nest_loc[1], color='magenta', marker='x', s=100, linewidth=2, label='Nest') # plots the nest location
    step_counter = fig.text(0.01, 0.01, 'Step 0', fontsize=12)
    ant_counter = fig.text(0.88, 0.01, 'Ants: 0', fontsize=12)

    food_plots = np.zeros(food_num, dtype=object) # stores the plot objects for the food sources to allow updating their colour when food is found
    for f in range(food_num): # adds food locations to the plot
        label = ''
        if f == 0: # only adds label for the first food source to avoid duplicates in the legend
            label = 'Food'
        food_plots[f] = ax.scatter(food_locs[f, 0], food_locs[f, 1], color='g', marker='o', s=50, linewidth=1, label=label)
    
    def format_coord(x, y): # shows location, marker strength and food level when hovering over the plot
        xi, yi = int(round(x)), int(round(y))
        base = 'x=%d, y=%d' % (xi, yi)
        if 0 <= xi < grid_size[0] and 0 <= yi < grid_size[1]:
            a = grid_marks[xi, yi, 0]
            b = grid_marks[xi, yi, 1]
            if a > 0.01 or b > 0.01:
                base += '  A=%.2f, B=%.2f' % (a, b)
        for f in range(food_num):
            if abs(xi - int(food_locs[f, 0])) <= 1 and abs(yi - int(food_locs[f, 1])) <= 1:
                base += '  Food: %.0f%%' % (food_locs[f, 2] * 100)
                break
        return base
    ax.format_coord = format_coord

    fig.suptitle('Ant Simulation – Basic Active Walkers', fontweight ="bold")
    found_handle = mlines.Line2D([], [], color='gold', marker='o', linestyle='None', markersize=6, label='Discovered')
    depleted_handle = mlines.Line2D([], [], color='red', marker='o', linestyle='None', markersize=6, label='Depleted')
    fig.legend(loc='outside lower center', ncol=4, handles=[*ax.get_legend_handles_labels()[0], found_handle, depleted_handle])

    marks_plot_A = ax.imshow(np.zeros(grid_size), cmap='Blues', alpha=0.6, norm=LogNorm(vmin=0.1, vmax=10)) # logarithmic scale
    marks_plot_A.set_mouseover(False)
    cbar_A = fig.colorbar(marks_plot_A, ax=ax) # adds colour scale for marker A
    cbar_A.set_label('Marker A')

    marks_plot_B = ax.imshow(np.zeros(grid_size), cmap='Oranges', alpha=0.6, norm=LogNorm(vmin=0.1, vmax=10))
    marks_plot_B.set_mouseover(False)
    cbar_B = fig.colorbar(marks_plot_B, ax=ax) # adds colour scale for marker B
    cbar_B.set_label('Marker B')

    ants_act = ants_pop[1].copy() # number of active ants
    food_returned = 0 # amount of food successfully returned to the nest

    def update(frame): # moves ants by a biased random walk
        nonlocal ants_act, food_returned # allows the function to modify these variables defined in the outer scope
        if pause: return # pauses the animation when clicked
        temp_grid_marks = grid_marks.copy() # temp grid to ensure concurrent movement, i.e. ants don't react to markers dropped in the same step
        
        for ant in range(ants_pop[0]): # moves all ants one step
            if ants[ant, 7] == 0: # inactive ant
                continue
            elif ants[ant, 6] < 0.5: # removes ants with low sensitivity
                ants[ant] = [nest_loc[0], nest_loc[1], 0, 0, 0, 1, 0.99, 0] # resets values: [x, y, dx, dy, marker, drop_rate, sensitivity, active]
                ants_act -= 1
                continue

            found = False # if the ant has found food or returned to nest
            if (int(ants[ant, 4]) == 1) and (abs(int(ants[ant, 0]) - int(nest_loc[0])) <= 1) and (abs(int(ants[ant, 1]) - int(nest_loc[1])) <= 1): # if the ant is within 1 grid space of the nest with food
                print('Ant', ant, 'returned to nest with food at step', frame)
                found = True
                food_returned += food_step
                ants[ant, 4] = 2 # ant starts following marker B
                ants[ant, 2:4] = -ants[ant, 2:4] # reverses the ant's direction to head back towards the nest
                ants[ant, 5] = 1 # resets the ant's drop rate
                for _ in range(ants_pop[2]): # recruits more ants when an ant successfully returns to the nest with food
                    if ants_act < ants_pop[0]: # checks if the total number of ants has not been exceeded
                        for x in range(ants_pop[0]): # finds an inactive ant to recruit
                            if ants[x, 7] == 0:
                                ants[x, 2:4] = ants[ant, 2:4] # same direction as food
                                ants[x, 4] = 2 # follows marker B
                                ants[x, 7] = 1 # activates the ant
                                ants_act += 1
                                break
            
            if (int(ants[ant, 4]) != 1) and not found: # if the ant is searching for food (i.e. not currently following marker A)
                for f in range(food_num): # checks if the ant has found food
                    if (abs(int(ants[ant, 0]) - int(food_locs[f, 0])) <= 1) and (abs(int(ants[ant, 1]) - int(food_locs[f, 1])) <= 1) and (food_locs[f, 2] > 0): # if the ant is within 1 grid space of the food source
                        print('Ant', ant, 'found food at', food_locs[f, 0:2], 'with level', str(round(food_locs[f, 2], 2)), 'at step', frame)
                        found = True
                        food_locs[f, 2] = max(0, food_locs[f, 2] - food_step) # reduces the food level by a fixed amount
                        if food_locs[f, 2] == 0:
                            food_plots[f].set(facecolor='red', edgecolor='red') # mark food as empty
                        else:
                            food_plots[f].set(facecolor='gold', edgecolor='gold') # mark food as found
                        ants[ant, 4] = 1 # ant starts following marker A
                        ants[ant, 2:4] = -ants[ant, 2:4]
                        ants[ant, 5] = 1
                        ants[ant, 6] = 0.99 # resets the ant's sensitivity to ensure it can follow the marker back to the nest
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
            
            ants[ant, 5] *= 1 - decay_rates[1] # applies drop rate decay
            if ants[ant, 4] != 1: # if the ant is not currently following marker A back to the nest with food
                ants[ant, 6] = max(0, ants[ant, 6] - decay_rates[2]) # applies sensitivity decay

        # updates the marker plots and removes markers with strength < 0.01 to improve visibility
        marks_plot_A.set_data(np.ma.masked_where(temp_grid_marks[:, :, 0].T < 0.05, grid_marks[:, :, 0].T)) # marker A
        marks_plot_B.set_data(np.ma.masked_where(temp_grid_marks[:, :, 1].T < 0.05, grid_marks[:, :, 1].T)) # marker B
        step_counter.set_text(f'Step {frame+1}')
        ant_counter.set_text(f'Ants: {ants_act}')

        grid_marks[:] = grid_marks*(1 - decay_rates[0]) # applies decay rate to all markers

        if ants_act == 0: # stops animation if ants become extinct :(
            ani_ref[0].event_source.stop()

        return marks_plot_A, marks_plot_B, step_counter, ant_counter

    pause = False
    def onClick(event): # function to pause the animation when the plot is clicked
        nonlocal pause
        pause ^= True
    fig.canvas.mpl_connect('button_press_event', onClick)

    ani_ref = [None]
    ani_ref[0] = animation.FuncAnimation(fig, update, frames=steps, interval=50, blit=False, repeat=False)
    plt.show()

def getForwardDirections(current_dir, directions, detection_range=3): # returns the directions that the ant can detect
    detects = []
    idx = directions.index(tuple(current_dir)) # gets index of current direction
    half_range = (detection_range - 1) // 2

    for d in range(-half_range, half_range + 1): # iterates through the indices of the directions in range
        detects.append(directions[(idx + d) % len(directions)]) # adds direction to the detects array, using modulo for wrap-around

    return detects # returns the directions the ant can detect based on its current direction of movement

def main():
    directions = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)] # the 8 possible movement directions (excluding [0,0])
    grid_size, grid_marks, nest_loc, food_num, food_locs, food_step, ants_pop, ants, alpha, decay_rates, steps, detection_range = initialise(directions)

    grid(grid_size, grid_marks, nest_loc, food_num, food_locs, food_step, ants_pop, ants, alpha, decay_rates, steps, detection_range, directions)

main()