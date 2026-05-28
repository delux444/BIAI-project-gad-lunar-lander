# Lunar Lander: Perfect landing Neuroevolution

A **neuroevolution** project that trains an autonomous flight pilot for OpenAI Gymnasium's `LunarLander-v3` environment using a **Genetic Algorithm (GA)**. Instead of gradient-based reinforcement learning, the pilot's neural network weights are evolved directly — no backpropagation, no policy gradients, just survival of the fittest.

The pilot is trained for a **Perfect landing** mission: descend, touch down between the flags without crash.

---

## Table of Contents

1. [Installation](#installation)
2. [Project Structure](#project-structure)
3. [The Environment](#the-environment)
4. [The Neural Network](#the-neural-network)
5. [The Genetic Algorithm](#the-genetic-algorithm)
6. [Fitness Function](#fitness-function)
7. [Multithreading](#multithreading)
8. [Code Walkthrough](#code-walkthrough)
9. [Results & Tuning](#results--tuning)

---

## Installation

**1. Clone the repository**
```bash
git clone https://github.com/<your-username>/BIAI-project-gad-lunar-lander.git
cd lunar-lander-neuroevolution (cd [project directory])
```

**2. Create a virtual environment**
```bash
# Linux / macOS
python3 -m venv venv
source venv/bin/activate

# Windows (PowerShell)
python -m venv venv
cd venv\bin
.\Activate.ps1
cd [project directory]
```

**3. Install dependencies**
```bash
pip install --upgrade pip
pip install gymnasium "gymnasium[box2d]" numpy pygad swig
```

**4. Run**
```bash
python code/main.py
```

---

## Project Structure

```
.
├── code/
│   └── main.py          # All training logic (single-file)
├── README.md
└── requirements.txt
```

---

## The Environment

`LunarLander-v3` is a classic continuous control problem from [Gymnasium](https://gymnasium.farama.org/). A lander starts at the top of the screen and must touch down softly between two flags at the centre.

### Observation Space — 8 continuous values

| Index | Variable          | Description                                      |
|-------|-------------------|--------------------------------------------------|
| 0     | `x`               | Horizontal position (0 = pad centre)             |
| 1     | `y`               | Vertical position (0 = ground)                   |
| 2     | `vx`              | Horizontal velocity                              |
| 3     | `vy`              | Vertical velocity (negative = descending)        |
| 4     | `angle`           | Body angle (0 = upright)                         |
| 5     | `angular_vel`     | Angular velocity                                 |
| 6     | `leg_left`        | Left leg ground contact (0.0 or 1.0)             |
| 7     | `leg_right`       | Right leg ground contact (0.0 or 1.0)            |

### Action Space — 4 discrete actions

| Value | Action               |
|-------|----------------------|
| 0     | Do nothing           |
| 1     | Fire left engine     |
| 2     | Fire main engine     |
| 3     | Fire right engine    |

### Built-in Reward Signal

The environment provides a dense reward on every step:

- **+100 to +140** for a successful landing between the flags
- **−100** for crashing
- **+10** for each leg touching the ground
- **−0.3 per frame** for firing the main engine
- **−0.03 per frame** for firing side engines

A score above **200** is considered a solved episode.

---

## The Neural Network

The pilot's "brain" is a fully connected feedforward neural network with one hidden layer.

```
Input (8)  →  Hidden (16, ReLU)  →  Output (4, Argmax)
```

### Architecture detail

```
Inputs (8 values)
      │
   ┌──┴──────────────────────┐
   │   Hidden Layer          │
   │   16 neurons            │
   │   Activation: ReLU      │
   │   f(x) = max(0, x)      │
   └──┬──────────────────────┘
      │
   ┌──┴──────────────────────┐
   │   Output Layer          │
   │   4 neurons (one/action)│
   │   Selection: argmax     │
   └──┬──────────────────────┘
      │
   Action (0–3)
```

**Why ReLU?** It is simple, computationally cheap, and avoids the vanishing gradient problem — important when weights are mutated randomly rather than nudged by gradients.

**Why argmax?** The network outputs a raw score for each action; we pick the highest. This is equivalent to a greedy deterministic policy, which works well when the GA already introduces stochasticity through mutation.

### The Chromosome

Every weight and bias in the network is flattened into a single 1D vector called a **chromosome** (or genome). This is what the Genetic Algorithm manipulates.

```
Chromosome length = (8 × 16 + 16) + (16 × 4 + 4)
                  =      144       +      68
                  =      212  genes
```

Each gene is a floating-point number representing one weight or bias. The GA treats this vector as the "DNA" of the pilot.

---

## The Genetic Algorithm

A **Genetic Algorithm** is a population-based metaheuristic inspired by natural selection. It maintains a pool of candidate solutions (pilots), evaluates how well each one performs (fitness), and iteratively breeds better solutions by combining and mutating the best candidates.

### Core Concepts

**Population**
A set of `N` chromosomes (candidate pilots). At generation 0, all chromosomes are initialised with random values uniformly sampled from `[−1.0, 1.0]`.

**Fitness Evaluation**
Each chromosome is decoded back into network weights, then used to fly the lander for a number of episodes. The average total reward across those episodes becomes the chromosome's fitness score.

**Selection**
Chromosomes with higher fitness are more likely to be chosen as parents for the next generation. This project uses **tournament selection**: `K` random candidates are drawn from the population, and the one with the best fitness wins.

> Tournament size `K=3` balances selection pressure — large K favours elites too aggressively and reduces diversity; small K is nearly random.

**Crossover**
Two parent chromosomes are combined to produce offspring. This project uses **scattered crossover**: for each gene position, a coin is flipped to decide which parent contributes that gene. This preserves the statistical properties of both parents across the whole chromosome.

```
Parent A:  [0.3,  0.7, -0.1,  0.5,  0.2, ...]
Parent B:  [-0.4, 0.1,  0.9, -0.2,  0.8, ...]
Mask:      [  A,    B,    B,    A,    A,  ...]
Offspring: [0.3,  0.1,  0.9,  0.5,  0.2, ...]
```

**Mutation**
After crossover, random genes are perturbed. This introduces new genetic material that was not present in either parent, preventing the population from getting stuck in local optima.

This project uses **adaptive mutation**: weak chromosomes (low fitness) receive a high mutation rate `(0.3)`, while strong ones receive a low rate `(0.05)`. This concentrates exploration where it is needed most.

**Elitism**
The top `E` chromosomes are copied directly into the next generation without modification. This guarantees that the best solution found so far is never lost to crossover or mutation noise.

### Full GA Loop

```
Initialise population (random chromosomes)
│
└─► For each generation:
       │
       ├─ Evaluate fitness of all chromosomes
       │    └─ Run N episodes of LunarLander, average reward
       │
       ├─ Select parents (tournament selection)
       │
       ├─ Produce offspring (scattered crossover)
       │
       ├─ Apply mutation (adaptive rates)
       │
       ├─ Insert elites unchanged
       │
       └─ Replace old population → repeat
```

### Hyperparameters

| Parameter               | Value  | Effect                                               |
|-------------------------|--------|------------------------------------------------------|
| `num_generations`       | 150    | Total number of evolutionary cycles                 |
| `sol_per_pop`           | 60     | Population size — more = more diversity, slower     |
| `num_parents_mating`    | 12     | Parents selected each generation                    |
| `K_tournament`          | 3      | Tournament size for parent selection                |
| `mutation_probability`  | [0.3, 0.05] | Adaptive: high for weak, low for elite        |
| `keep_elitism`          | 3      | Best N solutions carried forward unchanged          |
| `init_range`            | [−1, 1] | Initial gene value range                           |

---

## Fitness Function

The fitness function is the bridge between the pilot's genome and its real-world performance. It answers: *"How good is this chromosome?"*

```python
def fitness_func(ga_instance, solution, solution_idx):
    env = get_env()
    total_reward = 0
    episodes = 3

    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        landed_once = False

        while not done:
            action = predict(obs, solution)
            obs, reward, term, trunc, _ = env.step(action)

            # Touchdown bonus — first contact only
            if (obs[6] == 1.0 or obs[7] == 1.0) and not landed_once:
                landed_once = True
                reward += 100

            # Ascent reward — gain altitude after touching down
            if landed_once and obs[1] > 0.2:
                reward += 2

            total_reward += reward
            done = term or trunc

    return total_reward / episodes
```

### Why average over 3 episodes?

The environment resets with slightly randomised initial conditions (spawn position, velocity). A single episode can be misleadingly lucky or unlucky. Averaging over 3 episodes gives a more reliable fitness signal and reduces the chance of a mediocre pilot being selected by chance.

### Custom reward shaping

Beyond the environment's built-in rewards, two modifications guide the pilot toward the Perfect landing mission:

**Touchdown Bonus (+100)**
Fires exactly once, the first time either leg sensor reads `1.0`. This creates a strong selective pressure toward actually touching the ground — without it, hovering indefinitely can score better than landing.

**Ascent Reward (+2/frame)**
Once the pilot has touched down, every frame spent above altitude `0.2` earns a small bonus.

> Note: the `landed_once` flag ensures the touchdown bonus only fires once per episode and the ascent reward only activates after the first contact.

---

## Multithreading

PyGAD supports parallel fitness evaluation using Python threads via:

```python
parallel_processing=["thread", 16]
```

This evaluates up to 16 chromosomes simultaneously, significantly reducing wall-clock time per generation on multi-core machines.

### The Box2D threading problem

Box2D (the physics engine behind LunarLander) is **not thread-safe**. Sharing a single `gym.Env` instance across threads causes `AssertionError` crashes deep inside the physics simulation.

**Solution: `threading.local()`**

```python
thread_local = threading.local()

def get_env():
    if not hasattr(thread_local, "env"):
        thread_local.env = gym.make("LunarLander-v3")
    return thread_local.env
```

`threading.local()` is a namespace where each attribute is **private to the thread that set it**. The first time a thread calls `get_env()`, it has no `env` attribute and creates a fresh environment. Subsequent calls from the same thread reuse that environment. Different threads get different environments — completely isolated, no shared state, no crashes.

---

## Code Walkthrough

### 1. Architecture constants

```python
INPUT_SIZE = 8    # observation vector length
HIDDEN_SIZE = 16  # neurons in the hidden layer
OUTPUT_SIZE = 4   # one output per action

NUM_GENES = (INPUT_SIZE * HIDDEN_SIZE + HIDDEN_SIZE) \
          + (HIDDEN_SIZE * OUTPUT_SIZE + OUTPUT_SIZE)
# = 144 + 68 = 212
```

### 2. Thread-safe environment

```python
thread_local = threading.local()

def get_env():
    if not hasattr(thread_local, "env"):
        thread_local.env = gym.make("LunarLander-v3")
    return thread_local.env
```

Creates one `gym.Env` per thread and reuses it, avoiding the overhead of creating a new environment for every fitness call.

### 3. Network forward pass

```python
def relu(x):
    return np.maximum(0, x)

def predict(observation, weights):
    idx = 0

    w1 = weights[idx : idx + INPUT_SIZE * HIDDEN_SIZE].reshape((HIDDEN_SIZE, INPUT_SIZE))
    idx += INPUT_SIZE * HIDDEN_SIZE
    b1 = weights[idx : idx + HIDDEN_SIZE]
    idx += HIDDEN_SIZE

    w2 = weights[idx : idx + HIDDEN_SIZE * OUTPUT_SIZE].reshape((OUTPUT_SIZE, HIDDEN_SIZE))
    idx += HIDDEN_SIZE * OUTPUT_SIZE
    b2 = weights[idx : idx + OUTPUT_SIZE]

    hidden = relu(w1 @ observation + b1)
    output = w2 @ hidden + b2
    return np.argmax(output)
```

Unpacks the flat chromosome slice-by-slice into weight matrices `w1`, `w2` and bias vectors `b1`, `b2`, then performs the forward pass in two matrix multiplications.

### 4. Fitness function

See [Fitness Function](#fitness-function) above. The key points:
- Averages over 3 episodes for stability
- Adds a one-time `+100` touchdown bonus
- Adds `+2/frame` ascent reward after first contact

### 5. Dashboard callback

```python
def on_generation(ga_instance):
    ...
    sys.stdout.write("\033[9A")  # move cursor up 9 lines
    # overwrite dashboard lines in-place
```

`on_generation` is called by PyGAD after every generation. The ANSI escape `\033[9A` moves the terminal cursor up 9 lines, allowing the dashboard to be redrawn over itself without scrolling — giving a live-updating display.

### 6. GA instantiation

```python
ga_instance = pygad.GA(
    num_generations=150,
    num_parents_mating=12,
    fitness_func=fitness_func,
    sol_per_pop=60,
    num_genes=NUM_GENES,
    init_range_low=-1.0,
    init_range_high=1.0,
    parent_selection_type="tournament",
    K_tournament=3,
    crossover_type="scattered",
    mutation_type="adaptive",
    mutation_probability=[0.3, 0.05],
    keep_elitism=3,
    on_generation=on_generation,
    parallel_processing=["thread", 16]
)
```

### 7. Training and testing

```python
ga_instance.run()
best_sol, best_fit, _ = ga_instance.best_solution()

test_env = gym.make("LunarLander-v3", render_mode="human")
```

After `run()` completes, `best_solution()` returns the chromosome with the highest fitness across all generations. It is then tested visually in a rendered window.

---

## Results & Tuning

### Expected training progression

| Generation range | Typical behaviour                                     |
|-----------------|--------------------------------------------------------|
| 0–20            | Random crashes, score mostly negative                 |
| 20–60           | Pilots learn to hover, avoid crashing                 |
| 60–100          | First successful landings appear                      |
| 100–150         | Touchdown-and-ascent behaviour emerges in top pilots  |

### Tips for better results

**Increase population size** — `sol_per_pop=100` gives more genetic diversity and often converges to better solutions, at the cost of slower generations.

**More averaging episodes** — raising `episodes` in `fitness_func` from 3 to 5 reduces fitness noise and leads to more consistent selection, especially in later generations.

**Wider hidden layer** — changing `HIDDEN_SIZE` from 16 to 32 or 64 gives the network more representational capacity. The chromosome grows accordingly (`NUM_GENES` must be recalculated).

**Increase generations** — complex behaviours like Touch-and-Go often need 200–300 generations to fully stabilise.

---

## Dependencies

| Package      | Purpose                              |
|-------------|---------------------------------------|
| `gymnasium`  | LunarLander-v3 environment           |
| `gymnasium[box2d]` | Box2D physics backend          |
| `numpy`      | Fast array operations                |
| `pygad`      | Genetic algorithm framework          |
| `swig`       | Required to compile Box2D bindings   |
