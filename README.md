# Ant Pheromone Trail Simulation

A Python simulation of ant foraging behaviour using the **Active Random Walker** model, based on Schweitzer, Lao & Family (1997). Ants perform a biased random walk from a central nest, depositing two types of pheromone markers to form directed trunk trails between the nest and food sources.

---

## Model Description

Each ant is an **active random walker** that both modifies and responds to its chemical environment. Two pheromone types are used:

- **Marker A** (blue) — deposited by ants searching for food (outbound)
- **Marker B** (orange) — deposited by ants returning to the nest with food

The key dynamics are:

- **Scout ants** start active; the rest are recruited when food is found and returned to the nest
- **Biased random walk** — an ant continues in its current direction or picks a new one with probability `alpha`, biased toward forward-facing directions
- **Chemotaxis** — ants sense the pheromone field within a `detection_range` of forward-facing directions and move toward the strongest detected marker
- **Marker deposition** — ants deposit markers at each step; drop rate decays over time so early trail segments are more strongly marked
- **Marker decay** — all marker levels decay multiplicatively each step, so trails fade unless continuously refreshed
- **Sensitivity decay** — each ant's sensitivity to pheromones decreases over time, causing it to eventually return to the nest and reset
- **Recruitment** — when an ant successfully returns food to the nest, `recruitment_rate` additional ants are activated and sent toward the food source
- **Food depletion** — food sources have a finite supply; depleted sources are marked red

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
- `numpy`
- `matplotlib`

```bash
pip install numpy matplotlib
```

---

## Usage

```bash
python trails.py
```

An animated plot appears showing the pheromone fields evolving over `steps` timesteps.

- **Click** anywhere on the plot to pause/resume
- **Hover** over cells to inspect marker A/B strengths and food levels
- The progress bar (right) shows total food returned to the nest as a percentage
- The animation stops automatically when all ants go extinct or all food is collected

**Legend:**
| Colour | Meaning |
|--------|---------|
| Magenta ✕ | Nest |
| Green ● | Food source (undiscovered) |
| Gold ● | Food source (discovered) |
| Red ● | Food source (depleted) |
| Blue (log scale) | Marker A — outbound pheromone |
| Orange (log scale) | Marker B — return pheromone |

---

## Configuration

All parameters are set in `config.ini`. No code changes are needed.

| Parameter | Default | Description |
|---|---|---|
| `grid_size_x` | `100` | Grid width (cells) |
| `grid_size_y` | `100` | Grid height (cells) |
| `nest_loc_x` | `50` | Nest x position as % of grid width |
| `nest_loc_y` | `50` | Nest y position as % of grid height |
| `ants_num` | `100` | Total ant population |
| `scout_ants` | `20` | Number of ants active at the start |
| `recruitment_rate` | `5` | Ants recruited per successful food return |
| `alpha` | `0.4` | Probability of random direction change per step [0–1] |
| `detection_range` | `5` | Number of forward directions an ant can sense (odd) |
| `marker_decay_rate` | `0.01` | Fractional decay applied to all markers each step [0–1] |
| `drop_rate_decay` | `0.01` | Multiplicative decay of each ant's drop rate [0–1] |
| `sensitivity_decay` | `0.001` | Linear decay of each ant's pheromone sensitivity [0–1] |
| `food_num` | `5` | Number of food sources |
| `food_step` | `0.01` | Amount of food consumed/returned per interaction [0–1] |
| `steps` | `5000` | Number of simulation timesteps |

**Parameter notes:**

- Lower `alpha` → straighter, more persistent walks; higher → more diffuse exploration
- Lower `marker_decay_rate` → longer-lasting trails; if too low relative to ant count, no stable trails form
- `detection_range` must be an odd number to ensure symmetry around the ant's current direction
- Ants with sensitivity below 0.5 are reset to the nest; if fewer than 5 ants remain active, reset ants are immediately reactivated

---

## Background & References

> Schweitzer, F., Lao, K., & Family, F. (1997). Active random walkers simulate trunk trail formation by ants. *BioSystems*, 41, 153–166.

> Boissard, E., Degond, P., & Motsch, S. (2013). Trail formation based on directed pheromone deposition. *Journal of Mathematical Biology*, 66, 1267–1301.
