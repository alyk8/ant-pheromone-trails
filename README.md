# Ant Pheromone Trail Simulation

A Python simulation of ant foraging behaviour using the **Active Random Walker** model, based on Figure 2 of Schweitzer, Lao & Family (1997). Ants perform a biased random walk from a central nest, depositing pheromone markers that decay over time. The resulting emergent trail patterns reproduce the undirected track networks described in the paper.

This is the first stage of a two-part coursework model. The second stage will extend this to a directed trail system using two distinct pheromone types to simulate realistic foraging between a nest and food sources.

---

## Model Description

Each ant is an **active random walker**: it both modifies the environment (by depositing markers) and responds to it. In this basic implementation only one type of marker is used, and the ants do not yet respond to the chemical field — they perform a biased random walk whose persistence is controlled by `alpha`. Markers decay multiplicatively at each timestep at a rate set by `decay_rate`.

The key dynamics are:

- **Biased random walk** — at each step, an ant continues in its current direction with probability `1 - alpha`, or picks a new direction uniformly at random with probability `alpha`
- **Marker deposition** — every ant deposits a unit of marker at its current grid cell each step
- **Marker decay** — all marker levels are multiplied by `decay_rate` each frame, so marks fade unless continuously refreshed

This reproduces the spontaneous, undirected trail structures shown in Figure 2 of Schweitzer et al. (1997), where the grey scale encodes frequency of use.

---

## Repository Structure

```
.
├── main.py         # Main simulation script
├── config.ini      # All model parameters (edit this to change behaviour)
└── README.md
```

---

## Requirements

- Python 3.8+
- `numpy`
- `matplotlib`

Install dependencies with:

```bash
pip install numpy matplotlib
```

---

## Usage

1. Clone the repository and navigate to it:

```bash
git clone <repo-url>
cd <repo-folder>
```

2. Optionally edit `config.ini` to adjust parameters (see below).

3. Run the simulation:

```bash
python main.py
```

An animated plot will appear showing the pheromone marker field evolving over 1000 steps. The nest is marked with a blue cross. Marker strength is shown on a green colour scale — darker green indicates cells that have been visited more frequently.

---

## Configuration

All parameters are set in `config.ini`. No code changes are needed to adjust the model.

| Parameter | Default | Description |
|---|---|---|
| `grid_size_x` | `100` | Grid width (cells) |
| `grid_size_y` | `100` | Grid height (cells) |
| `nest_loc_x` | `50` | Nest x position as % of grid width |
| `nest_loc_y` | `50` | Nest y position as % of grid height |
| `ants_int` | `10` | Number of ants |
| `alpha` | `0.2` | Probability an ant changes direction each step [0-1] |
| `decay_rate` | `0.97` | Multiplicative decay applied to all markers each frame [0-1] |
| `steps` | `1000` | Number of steps to simulate |

**Notes on parameters:**

- A lower `alpha` produces more persistent, straighter walks; a higher `alpha` produces more diffuse, random movement.
- A `decay_rate` close to 1 means marks are long-lasting; a lower value causes them to fade quickly. If the decay rate is too high relative to the number of ants, no stable trails will form.

---

## Background & References

This simulation is based on the active walker framework described in:

> Schweitzer, F., Lao, K., & Family, F. (1997). Active random walkers simulate trunk trail formation by ants. *BioSystems*, 41, 153–166.

The broader modelling context — including directed pheromone deposition and kinetic descriptions — is discussed in:

> Boissard, E., Degond, P., & Motsch, S. (2013). Trail formation based on directed pheromone deposition. *Journal of Mathematical Biology*, 66, 1267–1301.

---

## Planned Extensions

- **Stage 2:** Two-pheromone system — chemical A deposited on the outward journey, chemical B deposited on return from a food source — to produce directed trunk trails linking the nest to food locations (following Schweitzer et al. Section 5)
- Ant sensitivity to the local chemical gradient
- Food source exhaustion and dynamic trail adaptation
