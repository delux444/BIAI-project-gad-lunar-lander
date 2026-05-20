"""
Lunar Lander - Genetic Algorithm Trainer
=========================================
Trains a neural network pilot to land precisely between the flags.

OBSERVATION SPACE (8 values):
  [0] x position       (0 = pad center)
  [1] y position       (0 = ground)
  [2] x velocity
  [3] y velocity
  [4] angle            (0 = upright)
  [5] angular velocity
  [6] left leg contact  (bool)
  [7] right leg contact (bool)

ACTIONS:
  0 = do nothing
  1 = fire left engine
  2 = fire main engine
  3 = fire right engine

REWARD (built-in):
  +100..+140 for landing between flags
  -100       for crashing
  -0.3/frame for firing main engine
  -0.03/frame for firing side engines

"""

import gymnasium as gym
import numpy as np
import pygad
import time
import sys
import os

# ══════════════════════════════════════════════════════════════════
# 1. NETWORK ARCHITECTURE
#    8 → 64 → 32 → 4
#    Bigger than original — LunarLander needs it.
# ══════════════════════════════════════════════════════════════════
INPUT_SIZE  = 8
H1          = 64
H2          = 32
OUTPUT_SIZE = 4

NUM_GENES = (
    INPUT_SIZE * H1 + H1 +
    H1         * H2 + H2 +
    H2 * OUTPUT_SIZE + OUTPUT_SIZE
)


def unpack(w):
    """Slice flat gene array into (W, b) pairs for each layer."""
    i = 0
    def take(rows, cols):
        nonlocal i
        mat = w[i:i + rows * cols].reshape(rows, cols)
        i  += rows * cols
        bias = w[i:i + rows]
        i  += rows
        return mat, bias
    W1, b1 = take(H1, INPUT_SIZE)
    W2, b2 = take(H2, H1)
    W3, b3 = take(OUTPUT_SIZE, H2)
    return W1, b1, W2, b2, W3, b3


def forward(obs, w):
    """Forward pass: tanh hidden layers, argmax output."""
    W1, b1, W2, b2, W3, b3 = unpack(w)
    h1 = np.tanh(W1 @ obs + b1)
    h2 = np.tanh(W2 @ h1 + b2)
    return int(np.argmax(W3 @ h2 + b3))


# ══════════════════════════════════════════════════════════════════
# 2. ENVIRONMENT  (one per process — avoids Box2D threading issues)
# ══════════════════════════════════════════════════════════════════
_env = None

def get_env():
    global _env
    if _env is None:
        _env = gym.make("LunarLander-v3")
    return _env


# ══════════════════════════════════════════════════════════════════
# 3. FITNESS FUNCTION
#
#   The environment's built-in reward already heavily penalises:
#     • crashing   (-100)
#     • going off-screen
#     • fuel use
#   and rewards:
#     • landing between flags (+100..+140)
#     • both legs touching    (+10 each)
#
#   We ADD three custom shaped bonuses to guide evolution:
#
#   A) PRECISION BONUS  — rewards being close to x=0 at landing.
#      Gives up to +80 extra for a centred touchdown.
#
#   B) SOFT LANDING BONUS  — rewards low vertical speed at contact.
#      Penalises hard crashes that the env already punishes.
#
#   C) UPRIGHT BONUS  — rewards small angle at touchdown.
#      Prevents landers that arrive sideways.
#
#   We REMOVE the "touch-and-go" hack from the original which
#   taught the agent to tap the ground and fly away instead of
#   actually landing.
# ══════════════════════════════════════════════════════════════════
EVAL_EPISODES = 5    # more episodes = more stable fitness signal
MAX_STEPS     = 600  # hard cap — prevents infinite hover

def fitness_func(ga_instance, solution, solution_idx):
    env   = get_env()
    total = 0.0

    for _ in range(EVAL_EPISODES):
        obs, _ = env.reset()
        episode_reward = 0.0
        steps          = 0
        landed         = False

        while not steps == MAX_STEPS:
            action = forward(obs, solution)
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            steps += 1

            both_legs = obs[6] == 1.0 and obs[7] == 1.0

            # ── A) Precision bonus ───────────────────────────────
            # Fires once when both legs are on the ground.
            # Landing right on x=0 → +80, landing at edge → +0.
            if both_legs and not landed:
                landed = True
                x_error   = abs(obs[0])          # 0 = perfect centre
                precision = max(0.0, 1.0 - x_error / 0.3)   # full bonus within ±0.3
                episode_reward += 80.0 * precision

                # ── B) Soft landing bonus ─────────────────────────
                vy = obs[3]   # negative = descending
                softness = max(0.0, 1.0 - abs(vy) / 1.5)
                episode_reward += 40.0 * softness

                # ── C) Upright bonus ──────────────────────────────
                angle   = abs(obs[4])
                upright = max(0.0, 1.0 - angle / 0.3)
                episode_reward += 30.0 * upright

            done = terminated or truncated
            if done:
                break

        total += episode_reward

    return total / EVAL_EPISODES


# ══════════════════════════════════════════════════════════════════
# 4. LIVE DASHBOARD
# ══════════════════════════════════════════════════════════════════
_t0          = None
_best        = -np.inf
_stagnation  = 0
STAG_LIMIT   = 25   # generations before mutation boost
DASHBOARD_H  = 12   # lines printed each generation

def on_generation(ga_instance):
    global _t0, _best, _stagnation

    gen = ga_instance.generations_completed
    if gen == 1:
        _t0 = time.time()
        print("\n" * DASHBOARD_H)

    elapsed  = time.time() - _t0 if _t0 else 0.0
    avg_gen  = elapsed / gen
    eta      = avg_gen * (ga_instance.num_generations - gen)
    _, best_fit, _ = ga_instance.best_solution()

    # ── Stagnation detection ─────────────────────────────────────
    if best_fit > _best + 1.0:
        _best       = best_fit
        _stagnation = 0
    else:
        _stagnation += 1

    stag_indicator = "▓" * _stagnation + "░" * (STAG_LIMIT - _stagnation)

    if _stagnation >= STAG_LIMIT:
        # Temporarily raise mutation rate to escape local optima
        mp = ga_instance.mutation_probability
        ga_instance.mutation_probability = [min(mp[0] * 1.4, 0.7), mp[1]]
        _stagnation = 0

    # ── Progress bar ─────────────────────────────────────────────
    pct  = gen / ga_instance.num_generations
    bars = int(pct * 30)
    bar  = "█" * bars + "░" * (30 - bars)

    # ── Fitness grade ─────────────────────────────────────────────
    if   best_fit >= 200: grade = "PERFECT LANDING ✓"
    elif best_fit >= 100: grade = "Good landing    ~"
    elif best_fit >=   0: grade = "Learning...      "
    else:                  grade = "Crashing badly  ✗"

    sys.stdout.write(f"\033[{DASHBOARD_H}A")
    lines = [
        "╔══════════════════════════════════════════════╗",
        "║      LUNAR LANDER  ·  PRECISION TRAINER      ║",
        "╠══════════════════════════════════════════════╣",
        f"║  [{bar}] {pct*100:5.1f}%  ║",
        f"║  Generation : {gen:>4} / {ga_instance.num_generations:<4}                    ║",
        f"║  Best Fit   : {best_fit:>10.2f}   {grade}  ║",
        f"║  Stagnation : [{stag_indicator[:20]}] {_stagnation:>2}/{STAG_LIMIT}  ║",
        f"║  Elapsed    : {elapsed:>7.1f}s   avg {avg_gen:.2f}s/gen         ║",
        f"║  ETA        : {eta:>7.1f}s                           ║",
        "╚══════════════════════════════════════════════╝",
        "  Press Ctrl+C to stop early and test best agent ",
        "",
    ]
    for line in lines:
        sys.stdout.write(f"\r{line}\033[K\n")
    sys.stdout.flush()


# ══════════════════════════════════════════════════════════════════
# 5. GENETIC ALGORITHM CONFIG
#
#   Key decisions vs the original script:
#
#   parallel_processing=None
#     PyGAD's thread pool adds GIL overhead — for small nets running
#     inside a single Python process, sequential is faster.
#     If you have many cores: change to ["process", os.cpu_count()]
#     (requires the if __name__=="__main__" guard below).
#
#   sol_per_pop=100, num_parents_mating=25
#     Larger population → more genetic diversity → escapes local
#     optima more easily.  Slower per generation but converges
#     to better solutions.
#
#   crossover_type="two_points"
#     Better than "scattered" for weight vectors — preserves blocks
#     of correlated weights instead of randomly interleaving them.
#
#   mutation_type="adaptive"  with [0.25, 0.03]
#     Weak solutions mutate heavily (0.25), elite ones lightly (0.03).
#     This concentrates exploration where it's needed.
#
#   keep_elitism=6
#     The top 6 solutions are copied unchanged into the next gen.
#     Prevents accidentally losing a great pilot.
# ══════════════════════════════════════════════════════════════════
ga = pygad.GA(
    num_generations       = 300,
    num_parents_mating    = 25,
    fitness_func          = fitness_func,
    sol_per_pop           = 100,
    num_genes             = NUM_GENES,
    gene_type             = float,
    init_range_low        = -2.0,
    init_range_high       =  2.0,

    parent_selection_type = "tournament",
    K_tournament          = 6,

    crossover_type        = "two_points",
    crossover_probability = 0.88,

    mutation_type         = "adaptive",
    mutation_probability  = [0.25, 0.03],

    keep_elitism          = 6,
    on_generation         = on_generation,

    parallel_processing   = None,   # see note above
    random_seed           = 7,
)


# ══════════════════════════════════════════════════════════════════
# 6. MAIN
# ══════════════════════════════════════════════════════════════════
def run_test_flight(weights, render=True):
    """Run one episode with the given weights, return total score."""
    env = gym.make("LunarLander-v3", render_mode="human" if render else None)
    obs, _ = env.reset()
    score, steps, done = 0.0, 0, False
    while not done and steps < MAX_STEPS:
        action = forward(obs, weights)
        obs, reward, term, trunc, _ = env.step(action)
        score += reward
        done   = term or trunc
        steps += 1
    env.close()
    return score, steps


if __name__ == "__main__":
    print("=" * 50)
    print("  LUNAR LANDER  —  PRECISION GA TRAINER")
    print("=" * 50)
    print(f"  Network  : {INPUT_SIZE} → {H1} → {H2} → {OUTPUT_SIZE}")
    print(f"  Genes    : {NUM_GENES}")
    print(f"  Pop size : 100   Generations: 300")
    print(f"  Episodes : {EVAL_EPISODES} per fitness call")
    print("=" * 50)
    print()

    try:
        ga.run()
    except KeyboardInterrupt:
        print("\n\n[!] Training stopped early.")

    best_weights, best_fit, _ = ga.best_solution()
    print(f"\n[✓] Training complete!  Best fitness: {best_fit:.2f}")

    # Save weights so you can reload without retraining
    np.save("best_weights.npy", best_weights)
    print("[✓] Weights saved → best_weights.npy")
    print("    Reload with: weights = np.load('best_weights.npy')")

    # ── Interactive test loop ──────────────────────────────────────
    print("\n[*] Launching visual test flights...\n")
    flight = 1
    try:
        while True:
            score, steps = run_test_flight(best_weights, render=True)

            if   score >= 200: verdict = "PERFECT  ★"
            elif score >= 100: verdict = "LANDED   ✓"
            elif score >=   0: verdict = "SURVIVED ~"
            else:               verdict = "CRASHED  ✗"

            print(f"  Flight #{flight:>3}  |  Score: {score:>8.2f}  |  {verdict}  |  {steps} steps")
            flight += 1

            try:
                ans = input("  [Enter] fly again   [q] quit: ").strip().lower()
            except EOFError:
                break
            if ans == "q":
                break
    except KeyboardInterrupt:
        pass

    print("\n[*] Done. Goodbye!")