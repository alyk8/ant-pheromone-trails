# Ant Pheromone Trail Simulation

A Python simulation of emergent ant foraging behaviour using the **Active Random Walker** model (Schweitzer, Lao & Family, 1997). Ants perform a biased random walk from a central nest, depositing pheromone markers that decay over time. Trunk trails between the nest and food sources emerge from the collective dynamics — no explicit trail-following is hard-coded.

---

## Repository Structure

```
.
├── trails.py       # Main simulation script
├── config.ini      # All model parameters (edit this to change behaviour)
└── README.md
```

---

## Requirements

- Python 3.8+
- `numpy`, `matplotlib`, `numba`

```bash
pip install numpy matplotlib numba
```

---

## Usage

```bash
python trails.py
```

An animated plot appears showing the two pheromone fields evolving over `steps` timesteps.

- **Click** anywhere on the plot to pause/resume
- **Hover** over cells to inspect marker A/B strengths and food levels
- The progress bar (right) shows total food returned to the nest as a percentage
- The animation stops automatically when all food is collected or all ants go extinct

**Legend:**

| Colour | Meaning |
|--------|---------|
| Magenta ✕ | Nest |
| Green ● | Food source (undiscovered) |
| Gold ● | Food source (being depleted) |
| Red ● | Food source (depleted) |
| Blue (log scale) | Marker A — outbound pheromone |
| Orange (log scale) | Marker B — return pheromone |

Set `visualise = false` in `config.ini` to run headlessly and print elapsed time.

---

## Model Description

### Pheromone system

Two marker fields are maintained on the grid:

- **Marker A** (blue) — deposited by ants that are searching for food or have been recruited and are heading outbound. Helps return ants navigate back toward the nest.
- **Marker B** (orange) — deposited by ants carrying food back to the nest. Guides recruited ants toward the food source.

Both fields decay multiplicatively each timestep: `grid_marks *= (1 - marker_decay_rate)`. Trails persist only if ants keep refreshing them.

### Ant state

Each ant carries the following state:

| Field | Meaning |
|-------|---------|
| `x, y` | Grid position (float, clamped to boundaries) |
| `dx, dy` | Current direction vector (one of 8 compass directions) |
| `marker` | Mode: `0` = searching, `1` = carrying food (heading to nest), `2` = recruited (heading to food) |
| `drop_rate` | Amount of pheromone deposited per step; starts at 1, decays each step |
| `sensitivity` | Sensitivity to pheromone; decays linearly each step while searching |
| `active` | Whether the ant is currently moving |
| `food_id` | Index of the food source being carried (`-1` if none) |

### Movement logic (per step, per ant)

Each active ant executes the following sequence:

**1. Sensitivity check**

If `sensitivity < 0.5`, the ant has "forgotten" its chemical cues and is teleported back to the nest with reset state (`drop_rate = 1`, `sensitivity = 0.99`, `marker = 0`). If the active population has dropped to or below `scout_ants`, it is immediately reactivated with a random direction; otherwise it becomes inactive.

**2. Event detection (nest and food)**

Before choosing a direction, the ant checks for interactions:

- *At the nest with food* (`marker == 1`, within 1 cell of nest): food is added to the returned total; the ant reverses direction and switches to mode `2` (recruited, heading out). `recruitment_rate` additional inactive ants are activated with the same outbound direction and set to mode `2`.
- *At a food source* (`marker != 1`, within 1 cell of an unempty food source): the ant takes `food_step` units, reverses direction, switches to mode `1` (carrying food), resets `drop_rate` and `sensitivity`.

If either event fires, the direction is already updated and the movement step below is skipped.

**3. Direction update**

Directions are chosen from the 8 compass vectors. A `forward_map` is precomputed: for each of the 8 directions, it stores the `detection_range` nearest directions centred on it (e.g. with `detection_range = 5`, an ant facing East can sense the 2 directions to its left, straight ahead, and the 2 directions to its right).

The ant follows this priority:

1. **Chemotaxis** — scan all forward-facing cells for the target marker (marker A if `marker ∈ {0, 1}`, marker B if `marker == 2`). If any cell has marker strength > 0.01 *and* a uniform random draw is less than the ant's current `sensitivity`, move toward the strongest detected cell. Ties are broken randomly.
2. **Random turn** — otherwise, with probability `alpha`, pick a random direction from the forward arc (excluding the current direction). If no forward directions are available (boundary corner), pick any valid direction.
3. **Persist** — otherwise, continue in the current direction.

**4. Move and deposit**

The ant moves one cell in its chosen direction (clamped to grid boundaries). It then deposits pheromone at the new cell:

- Mode `1` (carrying food) → deposits `drop_rate` units of **marker B**
- All other modes → deposits `drop_rate` units of **marker A**

`drop_rate` is then multiplied by `(1 - drop_rate_decay)`, so early steps of each trip are more heavily marked.

### Emergent trail formation

Trail formation is not explicitly programmed. Instead:

- Ants that find food reverse and deposit marker B along their return path.
- Recruited ants follow marker B outbound, then deposit marker A on return.
- The bidirectional reinforcement concentrates pheromone on shorter, more direct paths (because ants traversing them more frequently outpace the decay).
- Sensitivity decay prevents ants from following stale or weak trails indefinitely.

---

## Configuration

All parameters live in `config.ini`. No code changes are needed.

| Parameter | Default | Description |
|---|---|---|
| `grid_size_x` | `100` | Grid width (cells) |
| `grid_size_y` | `100` | Grid height (cells) |
| `nest_loc_x` | `50` | Nest x position as % of grid width |
| `nest_loc_y` | `50` | Nest y position as % of grid height |
| `ants_num` | `100` | Total ant population |
| `scout_ants` | `20` | Number of ants active at the start |
| `recruitment_rate` | `5` | Ants recruited per successful food return |
| `alpha` | `0.4` | Probability of random direction change per step |
| `detection_range` | `5` | Number of forward directions sensed (must be odd) |
| `marker_decay_rate` | `0.01` | Fractional decay applied to all markers each step |
| `drop_rate_decay` | `0.01` | Per-step multiplicative decay of each ant's drop rate |
| `sensitivity_decay` | `0.001` | Per-step linear decay of each ant's pheromone sensitivity |
| `food_num` | `5` | Number of food sources |
| `food_step` | `0.01` | Amount of food consumed/returned per interaction |
| `steps` | `5000` | Number of simulation timesteps |
| `speed` | `1` | Simulation steps computed per animation frame |
| `visualise` | `true` | Show animated plot (`false` = headless benchmark) |

**Notes:**

- Lower `alpha` → straighter, more persistent walks; higher → more diffuse exploration
- Lower `marker_decay_rate` → longer-lasting trails; too low relative to ant count prevents stable trails from forming
- `detection_range` must be odd to ensure symmetry around the ant's heading
- Food is placed at least 5% of the grid away from the nest on each axis

---

## References

> Schweitzer, F., Lao, K., & Family, F. (1997). Active random walkers simulate trunk trail formation by ants. *BioSystems*, 41, 153–166.

> Boissard, E., Degond, P., & Motsch, S. (2013). Trail formation based on directed pheromone deposition. *Journal of Mathematical Biology*, 66, 1267–1301.
