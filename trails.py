import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LogNorm
import matplotlib.lines as mlines
from mpl_toolkits.axes_grid1 import make_axes_locatable
from configparser import ConfigParser
from numba import njit
import time

def initialise(directions):
    config = ConfigParser()
    config.read('config.ini') # reads in values from config file

    grid_size = np.array([int(config.get('trails', 'grid_size_x')), int(config.get('trails', 'grid_size_y'))], dtype=np.uint16)
    grid_marks = np.zeros(shape=(grid_size[0], grid_size[1], 2), dtype=np.float32) # outbound pheromone
    nest_loc = np.array([(grid_size[0]*int(config.get('trails', 'nest_loc_x')))//100,
                        (grid_size[1]*int(config.get('trails', 'nest_loc_y')))//100], dtype=np.uint16)

    food_num = int(config.get('trails', 'food_num')) # no. of food sources
    food_locs = np.array([[34, 88, 1], [70, 87, 1], [13, 37, 1], [50, 15, 1], [88, 40, 1]], dtype=np.float32)
    '''food_locs = np.zeros(shape = [food_num, 3], dtype=np.float32) # stores locs of food sources + their level
    for f in range(food_num):
        # ensures the food is not placed too close to the nest (i.e. within 5% of the grid size)
        food_locs[f, 0] = random.choice([random.randint(0, int(nest_loc[0]*0.95)), random.randint(int(nest_loc[0]*1.05), grid_size[0])])
        food_locs[f, 1] = random.choice([random.randint(0, int(nest_loc[1]*0.95)), random.randint(int(nest_loc[1]*1.05), grid_size[1])])
        food_locs[f, 2] = 1 # diminishing supply (scale from 0-1)'''
    food_step = float(config.get('trails', 'food_step')) # how much food can an ant eat at once
    
    ants_pop = np.array([int(config.get('trails', 'ants_num')), int(config.get('trails', 'scout_ants')), int(config.get('trails', 'recruitment_rate'))]) # ant population size, initial number of scout ants, recruitment rate
    ants_locs = np.full(shape=(ants_pop[0], 2), fill_value=nest_loc, dtype=np.float32) # start all ants at the nest
    ants_dirs = np.zeros((ants_pop[0], 2), dtype=np.float32) # each ant is given a random starting direction
    for i in range(ants_pop[0]):
        ants_dirs[i] = directions[random.randint(0, 7)]
    ants_mark = np.zeros(ants_pop[0], dtype=np.float32) # which marker the ant is currently following (0 if not following any marker)
    ants_drop = np.ones(ants_pop[0], dtype=np.float32) # how much marker the ant will drop at each step
    ants_sens = np.full(ants_pop[0], fill_value=0.99, dtype=np.float32) # how sensitive the ant is to the markers
    ants_act = np.zeros(ants_pop[0], dtype=np.float32) # whether the ant is active
    ants_act[:ants_pop[1]] = 1 # initialise the first 'scout_ants' as active
    ants_food_id = np.full(ants_pop[0], fill_value=-1, dtype=np.float32) # which food source the ant is currently carrying (-1 if none)
    ants = np.column_stack((ants_locs, ants_dirs, ants_mark, ants_drop, ants_sens, ants_act, ants_food_id)) # [x, y, dx, dy, marker, drop_rate, sensitivity, active, food_id]

    alpha = float(config.get('trails', 'alpha')) # persistence parameter (i.e. probability that the ant will change direction)
    decay_rates = np.array([float(config.get('trails', 'marker_decay_rate')), float(config.get('trails', 'drop_rate_decay')), float(config.get('trails', 'sensitivity_decay'))], dtype=np.float32) # marker, drop rate and sensitivity decay rates
    detection_range = int(config.get('trails', 'detection_range')) # how many directions the ant can detect
    steps = int(config.get('trails', 'steps')) # no. of steps to simulate
    steps_per_frame = int(config.get('trails', 'speed')) # animation speed
    visualise = config.getboolean('trails', 'visualise')

    return grid_size, grid_marks, nest_loc, food_num, food_locs, food_step, ants_pop, ants, alpha, decay_rates, detection_range, steps, steps_per_frame, visualise

@njit
def get_new_direction(ant, ants, grid_size, temp_grid_marks, directions, alpha, forward_map):
    ant_x = int(ants[ant, 0]) # ant's current position
    ant_y = int(ants[ant, 1])
    curr_dx = int(ants[ant, 2]) # ant's current direction
    curr_dy = int(ants[ant, 3])

    curr_dir_idx = 0
    for i in range(8):
        if directions[i, 0] == curr_dx and directions[i, 1] == curr_dy:
            curr_dir_idx = i
            break
    detects = forward_map[curr_dir_idx] # gets the forward directions once
    target_marker = max(0, int(ants[ant, 4]) - 1) # which marker the ant is currently following (A or B) 

    marker_dirs = np.zeros(8, dtype=np.int8) # stores the number, directions and strengths of detected markers
    marker_strengths = np.zeros(8, dtype=np.float32)
    num_markers = 0

    forward_dirs = np.zeros(8, dtype=np.int8) # stores valid forward directions
    num_forward = 0

    dirs = np.zeros(8, dtype=np.int8) # stores all other possible new directions
    num_dirs = 0

    for d in range(8):
        dx, dy = directions[d, 0], directions[d, 1]
        x = ant_x + dx # calculates the next space if the ant moves in direction dx, dy
        y = ant_y + dy

        if (0 <= x < grid_size[0]) and (0 <= y < grid_size[1]):
            forward = False # if direction 'd' is in the ant's field of detection
            for j in range(len(detects)):
                if d == detects[j]:
                    forward = True
                    break

            if forward:
                strength = temp_grid_marks[x, y, target_marker] # gets strength of marker in direction d
                if strength > 0.01:
                    marker_dirs[num_markers] = d # stores direction and strength of marker
                    marker_strengths[num_markers] = strength
                    num_markers += 1
                elif d != curr_dir_idx: # if no marker and not current direction
                    forward_dirs[num_forward] = d
                    num_forward += 1
            else: # if it's not in the ant's field of detection
                dirs[num_dirs] = d
                num_dirs += 1
    
    if (num_markers > 0) and (np.random.random() < ants[ant, 6]): # if the ant detects markers and is sensitive enough to react to them
        if num_markers == 1: # if only one marker is detected, the ant will choose that direction
            idx = marker_dirs[0] # gets index of marker in directions array
            ants[ant, 2] = directions[idx, 0]
            ants[ant, 3] = directions[idx, 1]
        else: # if multiple markers are detected, the ant chooses the strongest one
            max_strength = marker_strengths[0] # finds the max strength of a marker
            for m in range(1, num_markers):
                if marker_strengths[m] > max_strength:
                    max_strength = marker_strengths[m]

            best_dirs = np.zeros(8, dtype=np.uint32) # find all directions that share this max strength
            num_best = 0
            for m in range(num_markers):
                if marker_strengths[m] == max_strength:
                    best_dirs[num_best] = marker_dirs[m]
                    num_best += 1

            idx = best_dirs[np.random.randint(0, num_best)] # chooses randomly among the tied best directions
            ants[ant, 2] = directions[idx, 0]
            ants[ant, 3] = directions[idx, 1]
    elif random.random() < alpha: # with probability alpha, the ant changes direction randomly
        if num_forward > 0: # if there are valid forward directions (i.e. within the ant's field of detection)
            idx = forward_dirs[np.random.randint(0, num_forward)]
            ants[ant, 2] = directions[idx, 0]
            ants[ant, 3] = directions[idx, 1]
        elif num_dirs > 0: # if the ant is stuck in a corner
            idx = dirs[np.random.randint(0, num_dirs)] # chooses a random direction from the remaining options
            ants[ant, 2] = directions[idx, 0]
            ants[ant, 3] = directions[idx, 1]

@njit
def simulate_one_step(grid_size, grid_marks, nest_loc, food_num, food_locs, food_step, food_returned, ants_pop, ants_act, ants, alpha, decay_rates, directions, forward_map): # moves all ants by exactly one step
    temp_grid_marks = grid_marks.copy() # temp grid to ensure concurrent movement, i.e. ants don't react to markers dropped in the same step
    
    for ant in range(ants_pop[0]): # moves all ants one step
        if ants[ant, 7] == 0: # inactive ant
            continue
        elif ants[ant, 6] < 0.5: # removes ants with low sensitivity
            ants[ant, :] = np.array([nest_loc[0], nest_loc[1], 0, 0, 0, 1, 0.99, 0, -1], dtype=np.float32) # resets values: [x, y, dx, dy, marker, drop_rate, sensitivity, active, food]
            if ants_act <= ants_pop[1]:
                idx = np.random.randint(0, 8) # assign a random direction before reactivating
                ants[ant, 2] = directions[idx, 0]
                ants[ant, 3] = directions[idx, 1]
                ants[ant, 7] = 1.0 # reactivates the ant if the population is too low
            else:
                ants_act -= 1
            continue

        found = False # if the ant has found food or returned to nest
        if (ants[ant, 4] == 1) and (abs(ants[ant, 0] - nest_loc[0]) <= 1) and (abs(ants[ant, 1] - nest_loc[1]) <= 1): # if the ant is within 1 grid space of the nest with food
            found = True
            food_returned += food_step
            #print('Ant', ant, 'returned to nest with food at step', frame, '- food returned:', '%.1f%%' % (100*food_returned/food_num))
            ants[ant, 4] = 2 # ant starts following marker B
            ants[ant, 2] = -ants[ant, 2] # reverses the ant's direction to head back towards the nest
            ants[ant, 3] = -ants[ant, 3]
            ants[ant, 5] = 1 # resets the ant's drop rate
            for _ in range(ants_pop[2]): # recruits more ants when an ant successfully returns to the nest with food
                if ants_act < ants_pop[0]: # checks if the total number of ants has not been exceeded
                    for x in range(ants_pop[0]): # finds an inactive ant to recruit
                        if ants[x, 7] == 0:
                            ants[x, 2] = ants[ant, 2] # same direction as food
                            ants[x, 3] = ants[ant, 3]
                            ants[x, 4] = 2 # follows marker B
                            ants[x, 7] = 1 # activates the ant
                            ants_act += 1
                            break
        
        if (ants[ant, 4] != 1) and not found: # if the ant is searching for food (i.e. not currently following marker A)
            for f in range(food_num): # checks if the ant has found food
                if (abs(ants[ant, 0] - food_locs[f, 0]) <= 1) and (abs(ants[ant, 1] - food_locs[f, 1]) <= 1) and (food_locs[f, 2] > 0): # if the ant is within 1 grid space of the food source
                    #print('Ant', ant, 'found food at', food_locs[f, 0:2], 'with level', str(round(food_locs[f, 2], 2)), 'at step', frame)
                    found = True
                    food_locs[f, 2] = max(0, food_locs[f, 2] - food_step) # reduces the food level by a fixed amount
                    ants[ant, 2] = -ants[ant, 2]
                    ants[ant, 3] = -ants[ant, 3]
                    ants[ant, 4] = 1 # ant starts following marker A
                    ants[ant, 5] = 1
                    ants[ant, 6] = 0.99 # resets the ant's sensitivity to ensure it can follow the marker back to the nest
                    ants[ant, 8] = f # stores which food source the ant is carrying
                    break

        if not found: # updates the ant's direction based on the current grid markers
            get_new_direction(ant, ants, grid_size, temp_grid_marks, directions, alpha, forward_map)

        new_x = ants[ant, 0] + ants[ant, 2] # calculates new ant position
        new_y = ants[ant, 1] + ants[ant, 3]

        ants[ant, 0] = max(0, min(new_x, grid_size[0] - 1)) # keeps ant within grid boundaries
        ants[ant, 1] = max(0, min(new_y, grid_size[1] - 1))

        # drops markers, capped at 10 in the plot
        if int(ants[ant, 4]) == 1: # if the ant is following marker A (i.e. it has found food and is heading back to the nest)
            grid_marks[int(ants[ant, 0]), int(ants[ant, 1]), 1] += ants[ant, 5] # marker B
        else: # if the ant is moving randomly or following marker B (i.e. searching for food)
            grid_marks[int(ants[ant, 0]), int(ants[ant, 1]), 0] += ants[ant, 5] # marker A

        ants[ant, 5] *= 1 - decay_rates[1] # applies drop rate decay
        if ants[ant, 4] != 1: # if the ant is not currently following marker A back to the nest with food
            ants[ant, 6] = max(0, ants[ant, 6] - decay_rates[2]) # applies sensitivity decay

    grid_marks[:, :, 0] *= (1 - decay_rates[0]) # applies decay rate to all markers
    grid_marks[:, :, 1] *= (1 - decay_rates[0])
    
    return food_returned, ants_act

def grid(grid_size, grid_marks, nest_loc, food_num, food_locs, food_step, ants_pop, ants, alpha, decay_rates, steps, steps_per_frame, directions, forward_map):
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.canvas.manager.set_window_title('Ant Pheromone Trails Simulation')
    divider = make_axes_locatable(ax)
    ax_cbar_B = divider.append_axes('left', size='5%', pad=0.5)
    ax_cbar_A = divider.append_axes('left', size='5%', pad=0.8)
    ax_prog = divider.append_axes('right', size='12%', pad=0.5)

    ax.set_xlim(0, grid_size[0])
    ax.set_ylim(0, grid_size[1])
    ax.scatter(nest_loc[0], nest_loc[1], color='magenta', marker='x', s=100, linewidth=2, label='Nest') # plots the nest location
    step_counter = fig.text(0.01, 0.01, 'Step 0', fontsize=12)
    ant_counter = fig.text(0.9, 0.01, 'Ants: 0', fontsize=12)

    ax_prog.set_xlim(-0.5, 0.5) # creates a bar plot to show the percentage of food returned to the nest
    ax_prog.set_ylim(0, 100)
    ax_prog.set_xticks([])
    ax_prog.yaxis.set_label_position('right')
    ax_prog.yaxis.tick_right()
    ax_prog.set_ylabel('Food returned (%)')
    prog_bar = ax_prog.bar([0], [0], color='limegreen', width=1)[0]
    prog_text = ax_prog.text(0, 2, '0%', ha='center', va='bottom', fontsize=10)

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

    fig.suptitle('Ant Simulation – Basic Active Walkers', fontweight='bold')
    found_handle = mlines.Line2D([], [], color='gold', marker='o', linestyle='None', markersize=6, label='Discovered') # custom legend for food sources
    depleted_handle = mlines.Line2D([], [], color='red', marker='o', linestyle='None', markersize=6, label='Depleted')
    fig.legend(loc='lower center', ncol=4, handles=[*ax.get_legend_handles_labels()[0], found_handle, depleted_handle])
    fig.subplots_adjust(bottom=0.11, top=0.93)

    marks_plot_A = ax.imshow(np.zeros(grid_size), cmap='Blues', alpha=0.6, norm=LogNorm(vmin=0.01, vmax=10)) # logarithmic scale
    marks_plot_A.set_mouseover(False)
    cbar_A = fig.colorbar(marks_plot_A, cax=ax_cbar_A)
    cbar_A.set_label('Marker A')
    ax_cbar_A.yaxis.set_label_position('left')
    ax_cbar_A.yaxis.tick_left()

    marks_plot_B = ax.imshow(np.zeros(grid_size), cmap='Oranges', alpha=0.6, norm=LogNorm(vmin=0.01, vmax=10))
    marks_plot_B.set_mouseover(False)
    cbar_B = fig.colorbar(marks_plot_B, cax=ax_cbar_B)
    cbar_B.set_label('Marker B')
    ax_cbar_B.yaxis.set_label_position('left')
    ax_cbar_B.yaxis.tick_left()

    ants_act = ants_pop[1].copy() # number of active ants
    food_returned = 0 # amount of food successfully returned to the nest
    current_step = 0
    pct = 0 # percentage of food returned to the nest

    def update(frame):
        nonlocal ants_act, food_returned, current_step, pct
        if pause: return # pauses the animation when clicked

        for _ in range(steps_per_frame):
            food_returned, ants_act = simulate_one_step(grid_size, grid_marks, nest_loc, food_num, food_locs, food_step, food_returned, ants_pop, ants_act, ants, alpha, decay_rates, directions, forward_map)
            pct = min(100, 100*food_returned/food_num) # calculates percentage of food returned to the nest
            if (ants_act == 0) or (pct == 100): # breaks out of the loop if the simulation is finished
                break
        current_step += steps_per_frame
        
        for f in range(food_num):
            if food_locs[f, 2] == 0:
                food_plots[f].set(facecolor='red', edgecolor='red') # mark food as empty
            elif food_locs[f, 2] < 1:
                food_plots[f].set(facecolor='gold', edgecolor='gold') # mark food as found

        # updates the marker plots and removes markers with strength < 0.01 to improve visibility
        plot_A = grid_marks[:, :, 0].T.copy()
        plot_A[plot_A < 0.01] = np.nan
        marks_plot_A.set_data(plot_A)

        plot_B = grid_marks[:, :, 1].T.copy()
        plot_B[plot_B < 0.01] = np.nan
        marks_plot_B.set_data(plot_B)

        step_counter.set_text(f'Step {current_step}')
        ant_counter.set_text(f'Ants: {ants_act}')
        
        prog_bar.set_height(pct)
        prog_text.set_text('%.1f%%' % pct)
        prog_text.set_position((0, min(pct + 2, 95)))

        if (ants_act == 0) or (pct == 100): # stops animation if ants become extinct or find all the food
            ani_ref[0].event_source.stop()

        return marks_plot_A, marks_plot_B, step_counter, ant_counter

    pause = False
    def onClick(event): # function to pause the animation when the plot is clicked
        nonlocal pause
        pause ^= True
    fig.canvas.mpl_connect('button_press_event', onClick)

    ani_ref = [None]
    repeats = steps // steps_per_frame - 1
    ani_ref[0] = animation.FuncAnimation(fig, update, frames=repeats, interval=300, blit=False, repeat=False)
    plt.show()

def no_grid(grid_size, grid_marks, nest_loc, food_num, food_locs, food_step, ants_pop, ants, alpha, decay_rates, steps, directions, forward_map):
    ants_act = ants_pop[1]
    food_returned = 0 # amount of food successfully returned to the nest
    active_ants_history = [] # tracks active ants at each step
    remaining_food_history = [] # tracks remaining food percentage at each step
    step_history = [] # tracks step numbers

    for step in range(steps):
        food_returned, ants_act = simulate_one_step(grid_size, grid_marks, nest_loc, food_num, food_locs, food_step, food_returned, ants_pop, ants_act, ants, alpha, decay_rates, directions, forward_map)
        pct = min(100, 100*food_returned/food_num) # calculates percentage of food returned to the nest
        remaining_food_pct = sum(food_locs[:, 2]) * 100 / food_num  # percentage of total remaining food
        
        # Track active ants and remaining food history
        active_ants_history.append(ants_act)
        remaining_food_history.append(remaining_food_pct)
        step_history.append(step + 1)

        if (ants_act == 0) or (pct == 100): # breaks out of the loop if the simulation is finished
            break
    
    # Plot active ants vs time and remaining food percentage after simulation
    fig, ax1 = plt.subplots(figsize=(8, 6))
    ax1.plot(step_history, active_ants_history, color='purple', linewidth=2, label='Active Ants')
    ax1.set_xlabel('Steps')
    ax1.set_ylabel('Active Ants', color='purple')
    ax1.tick_params(axis='y', labelcolor='purple')
    ax1.set_title('Active Ants and Remaining Food Over Time')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, max(step_history))

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.plot(step_history, remaining_food_history, color='green', linewidth=2, label='Remaining Food (%)')
    ax2.set_ylabel('Remaining Food (%)', color='green')
    ax2.tick_params(axis='y', labelcolor='green')

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    plt.show()
    
    return step

@njit
def precompute_forward_dirs(detection_range=5): # returns an array of the indices of the forward directions
    forward_map = np.zeros((8, detection_range), dtype=np.int32)
    half_range = (detection_range - 1) // 2

    for i in range(8):
        col = 0
        for d in range(-half_range, half_range + 1): 
            forward_map[i, col] = (i + d) % 8 
            col += 1
            
    return forward_map

def main():
    directions = np.array([(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)], dtype=np.int32) # the 8 possible directions (exc [0,0])
    grid_size, grid_marks, nest_loc, food_num, food_locs, food_step, ants_pop, ants, alpha, decay_rates, detection_range, steps, steps_per_frame, visualise = initialise(directions)
    forward_map = precompute_forward_dirs(detection_range)

    if visualise:
        grid(grid_size, grid_marks, nest_loc, food_num, food_locs, food_step, ants_pop, ants, alpha, decay_rates, steps, steps_per_frame, directions, forward_map)
    else:
        start = time.time()
        total_steps = no_grid(grid_size, grid_marks, nest_loc, food_num, food_locs, food_step, ants_pop, ants, alpha, decay_rates, steps, directions, forward_map)
        end = time.time()
        print('Steps:', total_steps, 'in', round(end - start, 2), 'seconds')

main()