"""
Sudoku Solver — Simulated Annealing
====================================

WHAT IS SIMULATED ANNEALING?
------------------------------
SA is a probabilistic optimisation algorithm modelled on controlled cooling
in metallurgy. Slowly cooling a metal lets atoms settle into a low-energy
(ordered) state; cooling too fast traps them in disorder (local minima).

The algorithm:
  1. Start from a random-but-partially-valid solution.
  2. Propose a small random change ("neighbour").
  3. Always accept improvements.
  4. Accept *worse* solutions with probability P = exp(-ΔE / T).
     At high T this is nearly 1.0 — the search is exploratory.
     As T falls, P shrinks — the search becomes increasingly greedy.
  5. Repeat, cooling T each iteration, until solved or budget exhausted.

The magic is step 4: accepting bad moves occasionally lets SA escape local
optima that trap pure hill-climbing algorithms.

KEY SA PARAMETERS (tunable in the CONFIG block below)
------------------------------------------------------
INITIAL_TEMP   : Starting temperature T₀.
                 High  → accepts almost any move; broad but slow exploration.
                 Low   → nearly greedy from the start; fast but gets stuck.
                 Rule of thumb: set so exp(-typical_bad_delta / T₀) ≈ 0.5.

COOLING_RATE   : Geometric decay factor α (0 < α < 1). Each "epoch":
                   T_new = T * α
                 α = 0.999  → very slow cool, thorough, expensive.
                 α = 0.99   → faster cool, risks premature convergence.

STEPS_PER_TEMP : How many neighbour proposals to try at each temperature
                 level before cooling. More steps = better exploration per
                 epoch at the cost of runtime.

MIN_TEMP       : Termination condition. When T < MIN_TEMP the acceptance
                 probability for any bad move is negligible, so we stop.

REHEAT_FACTOR  : If the solver gets stuck (no improvement for REHEAT_PATIENCE
                 steps), multiply T by this to escape local optima.
                 Set REHEAT_FACTOR = None to disable reheating.

ENCODING & WHY BOX CONSTRAINTS COME FREE
------------------------------------------
We fill each 3×3 box with a random permutation of its missing digits upfront.
Every box therefore contains exactly 1–9 from the start, and our only move
(swapping two non-clue cells *within* the same box) preserves that invariant.

The cost function only needs to count row and column conflicts — a huge
simplification. A cost of 0 means rows and columns are also conflict-free,
so the puzzle is solved.
"""

import random
import math
import time
import copy
import matplotlib.pyplot as plt


# ===========================================================================
# CONFIG — edit these to experiment with SA behaviour
# ===========================================================================
INITIAL_TEMP    = 0.7      # T₀: starting temperature
COOLING_RATE    = 0.7     # α:  T *= α after each epoch
STEPS_PER_TEMP  = 200      # proposals per epoch
MIN_TEMP        = 0.0005   # halt when T falls below this
MAX_STEPS       = 1_000_000  # hard cap on total iterations (safety net)
REHEAT_FACTOR   = 1.4      # multiply T by this when stuck; None = disabled
REHEAT_PATIENCE = 3_000    # steps without improvement before reheating
RANDOM_SEED     = None     # set an int for reproducible runs, None = random
# ===========================================================================


# ---------------------------------------------------------------------------
# Puzzle I/O
# ---------------------------------------------------------------------------

def parse_puzzle(flat: str) -> list[list[int]]:
    """81-char string (0 or . for blanks) → 9×9 list of ints."""
    flat = flat.replace(".", "0").replace(" ", "").replace("\n", "")
    if len(flat) != 81:
        raise ValueError(f"Expected 81 characters, got {len(flat)}")
    return [[int(flat[r * 9 + c]) for c in range(9)] for r in range(9)]


def print_grid(grid: list[list[int]], title: str = "") -> None:
    if title:
        print(f"\n{'-' * 25}  {title}  {'-' * 25}")
    border = "+---------+---------+---------+"
    print(border)
    for r in range(9):
        row_str = "|"
        for c in range(9):
            v = grid[r][c]
            row_str += f" {v if v else '.'} "
            if (c + 1) % 3 == 0:
                row_str += "|"
        print(row_str)
        if (r + 1) % 3 == 0:
            print(border)


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

def initialise_solution(puzzle: list[list[int]]) -> list[list[int]]:
    """
    Fill each 3×3 box with a shuffled permutation of its missing digits.
    Box constraints are satisfied by construction and stay satisfied
    throughout the entire run (swaps only ever happen within a box).
    """
    grid = copy.deepcopy(puzzle)
    for br in range(3):
        for bc in range(3):
            cells = [(br * 3 + r, bc * 3 + c) for r in range(3) for c in range(3)]
            given  = {grid[r][c] for r, c in cells if grid[r][c] != 0}
            missing = list(set(range(1, 10)) - given)
            random.shuffle(missing)
            for r, c in cells:
                if grid[r][c] == 0:
                    grid[r][c] = missing.pop()
    return grid


# ---------------------------------------------------------------------------
# Cost function
# ---------------------------------------------------------------------------

def cost(grid: list[list[int]]) -> int:
    """
    Count duplicate digits across all rows and all columns.
    Boxes are always conflict-free by construction, so we skip them.
    Returns 0 if the puzzle is solved.
    """
    total = 0
    for i in range(9):
        row = grid[i]
        col = [grid[r][i] for r in range(9)]
        total += (9 - len(set(row))) + (9 - len(set(col)))
    return total


# ---------------------------------------------------------------------------
# Neighbour generation
# ---------------------------------------------------------------------------

def precompute_free_cells(puzzle: list[list[int]]) -> dict[tuple, list[tuple]]:
    """Return {(box_row, box_col): [(r, c), ...]} for non-clue cells only."""
    free = {}
    for br in range(3):
        for bc in range(3):
            cells = [
                (br * 3 + r, bc * 3 + c)
                for r in range(3) for c in range(3)
                if puzzle[br * 3 + r][bc * 3 + c] == 0
            ]
            if len(cells) >= 2:
                free[(br, bc)] = cells
    return free


def propose_swap(grid: list[list[int]], free: dict) -> tuple[int, int, int, int]:
    """Pick a random box and swap two of its free cells. Return (r1,c1,r2,c2)."""
    box   = random.choice(list(free.keys()))
    cells = free[box]
    (r1, c1), (r2, c2) = random.sample(cells, 2)
    grid[r1][c1], grid[r2][c2] = grid[r2][c2], grid[r1][c1]
    return r1, c1, r2, c2


def undo_swap(grid: list[list[int]], r1: int, c1: int, r2: int, c2: int) -> None:
    grid[r1][c1], grid[r2][c2] = grid[r2][c2], grid[r1][c1]


# ---------------------------------------------------------------------------
# Simulated Annealing
# ---------------------------------------------------------------------------

def plot_cost_history(accepted_history: list, best_history: list) -> None:
    steps_a, costs_a = zip(*accepted_history)
    steps_b, costs_b = zip(*best_history)

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(steps_a, costs_a, alpha=0.35, linewidth=0.6, color="steelblue", label="Accepted cost")
    ax.plot(steps_b, costs_b, linewidth=1.8, color="crimson", label="Best cost")
    ax.set_xlabel("Step")
    ax.set_ylabel("Cost (row + column conflicts)")
    ax.set_title("Simulated Annealing — cost over time")
    ax.legend()
    plt.tight_layout()
    plt.show()


def simulated_annealing(
    puzzle: list[list[int]],
    verbose: bool = True,
    plot_progress: bool = False,
) -> list[list[int]] | None:
    if RANDOM_SEED is not None:
        random.seed(RANDOM_SEED)

    grid         = initialise_solution(puzzle)
    free         = precompute_free_cells(puzzle)
    current_cost = cost(grid)

    best_grid    = copy.deepcopy(grid)
    best_cost    = current_cost

    T                      = INITIAL_TEMP
    total_steps            = 0
    steps_no_improvement   = 0
    accepted_bad           = 0
    reheat_count           = 0
    start                  = time.time()

    accepted_history = [(0, current_cost)]
    best_history     = [(0, best_cost)]

    if verbose:
        print(f"\nSA started  | cost={current_cost} | T0={INITIAL_TEMP} "
              f"| alpha={COOLING_RATE} | steps/epoch={STEPS_PER_TEMP}")
        print(f"{'Step':>9}  {'T':>8}  {'Cost':>6}  {'Best':>6}  {'Accept%':>8}  {'Reheats':>7}")
        print("-" * 60)

    log_interval = max(STEPS_PER_TEMP * 100, 10_000)
    last_log     = 0

    while T > MIN_TEMP and total_steps < MAX_STEPS:
        epoch_accepted = 0

        for _ in range(STEPS_PER_TEMP):
            r1, c1, r2, c2 = propose_swap(grid, free)
            new_cost        = cost(grid)
            delta           = new_cost - current_cost

            # Accept: always if improvement, probabilistically if worse
            if delta <= 0 or random.random() < math.exp(-delta / T):
                current_cost = new_cost
                epoch_accepted += 1
                accepted_history.append((total_steps, current_cost))
                if new_cost < best_cost:
                    best_cost  = new_cost
                    best_grid  = copy.deepcopy(grid)
                    best_history.append((total_steps, best_cost))
                    steps_no_improvement = 0
                    accepted_bad = 0  # reset counter for cleaner logging
            else:
                undo_swap(grid, r1, c1, r2, c2)

            total_steps        += 1
            steps_no_improvement += 1
            if delta > 0:
                accepted_bad += 1

            if best_cost == 0:
                elapsed = time.time() - start
                if verbose:
                    print(f"\nSOLVED in {total_steps:,} steps | "
                          f"T={T:.4f} | time={elapsed:.2f}s | reheats={reheat_count}")
                if plot_progress:
                    plot_cost_history(accepted_history, best_history)
                return best_grid

        # Cool down
        T *= COOLING_RATE

        # Reheat if stuck
        if REHEAT_FACTOR and steps_no_improvement >= REHEAT_PATIENCE:
            T                    = min(T * REHEAT_FACTOR, INITIAL_TEMP * 0.8)
            steps_no_improvement = 0
            reheat_count        += 1

        # Periodic progress log
        if verbose and total_steps - last_log >= log_interval:
            accept_pct = 100 * epoch_accepted / STEPS_PER_TEMP
            print(f"{total_steps:>9,}  {T:>8.4f}  {current_cost:>6}  "
                  f"{best_cost:>6}  {accept_pct:>7.1f}%  {reheat_count:>7}")
            last_log = total_steps

    elapsed = time.time() - start
    if verbose:
        print(f"\nNot solved. Best cost={best_cost} | "
              f"steps={total_steps:,} | time={elapsed:.2f}s | reheats={reheat_count}")
        print("  Try: increase INITIAL_TEMP, lower COOLING_RATE, or raise MAX_STEPS.")
    if plot_progress:
        plot_cost_history(accepted_history, best_history)
    return None


# ---------------------------------------------------------------------------
# Example puzzles
# ---------------------------------------------------------------------------

PUZZLES = {
    # Straightforward — SA solves this quickly
    "easy": (
        "530070000"
        "600195000"
        "098000060"
        "800060003"
        "400803001"
        "700020006"
        "060000280"
        "000419005"
        "000080079"
    ),
    # Fewer clues — harder for SA, good for testing reheating
    "hard": (
        "000000000"
        "000003085"
        "001070400"
        "500100000"
        "070000100"
        "000500002"
        "600080700"
        "003002000"
        "000040000"
    ),
    # Near-minimal clue count — stress test
    "extreme": (
        "800000000"
        "003600000"
        "070090200"
        "060005030"
        "004803001"
        "300010060"
        "008300790"
        "000060800"
        "000000030"
    ),

    "new": (
        "300049000"
        "000600501"
        "752001000"
        "001000700"
        "500396000"
        "008150096"
        "003010060"
        "004000100"
        "000028000"
    )
}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    # Select puzzle: pass "easy", "hard", or "extreme" as a command-line arg
    key = sys.argv[1] if len(sys.argv) > 1 else "easy"
    if key not in PUZZLES:
        print(f"Unknown puzzle '{key}'. Choose from: {', '.join(PUZZLES)}")
        sys.exit(1)

    puzzle = parse_puzzle(PUZZLES[key])
    print_grid(puzzle, title=f"Puzzle ({key})")

    solution = simulated_annealing(puzzle, verbose=True, plot_progress=True)

    if solution:
        print_grid(solution, title="Solution")
    else:
        print("\nSA did not find a solution within budget.")
        print("Tune INITIAL_TEMP, COOLING_RATE, or MAX_STEPS at the top of the file.")
