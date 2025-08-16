# This .py file is my tournament runner.
# Jarren Calizo - Aug 15, 2025

# Small tournament driver for Connect Four agents.
# - Alternates starting player every game
# - Runs N games per pairing (100-200 typical)
# - Emits CSV with per-game stats
# - Prints summary: wins/losses/draws, avg moves, avg sec/move
#
# My design choice reasoning:
# Keep it tiny, fast, and reproducible. Choice functions already share a signature
# across our codebase, so we just wire them up, time them, and record outcomes.

from __future__ import annotations

import argparse
import csv
import os
import time
from typing import Callable, Tuple, List, Dict, Optional

from board import Board
from game import Game
import mcts as mcts_mod
from minmax import MinMax
from q_learning import QLearningAgent, random_agent as q_random_agent  # random agent matches signature

ChoiceFn = Callable[[List[int], Board, str, str], int]


# Agent factories

def make_random() -> Tuple[str, ChoiceFn]:
    return "random", q_random_agent

def make_mcts(iterations: int) -> Tuple[str, ChoiceFn]:
    return f"mcts@{iterations}", mcts_mod.mcts_factory(iterations)

def make_minmax(depth: int) -> Tuple[str, ChoiceFn]:
    ai = MinMax()
    return f"minmax@{depth}", ai.get_choice_function(depth=depth)

def make_q_learning(qtable_path: Optional[str]) -> Tuple[str, ChoiceFn]:
    path = qtable_path or "q_table_connect4.json"
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Q-learning table not found at '{path}'. Train first or pass --qtable."
        )
    agent = QLearningAgent.load(path)
    # Greedy (eval) policy: no exploration during tournament
    return "qlearn", agent.choose


# Helpers

def result_to_agent_label(winner_symbol: str, a_is_one: bool) -> str:
    # Map 'X'/'O'/'D' to 'A'/'B'/'D' for attribution, given who moved first
    if winner_symbol == "D":
        return "D"
    if winner_symbol == "X":
        return "A" if a_is_one else "B"
    if winner_symbol == "O":
        return "B" if a_is_one else "A"
    return "D"


def count_moves(board: Board) -> int:
    # Number of pieces placed on the board = total moves
    return sum(len(col) for col in board.board)


# Core tournament logic 

def run_pairing(
    a_name: str,
    a_choice: ChoiceFn,
    b_name: str,
    b_choice: ChoiceFn,
    games: int,
    writer: csv.writer,
    pairing_label: Optional[str] = None,
    seed: Optional[int] = None,
) -> Dict[str, float]:
    # Run `games` matches between A and B, alternating the starter each game
    # Write per-game rows to CSV and return summary stats
    if seed is not None:
        import random
        random.seed(seed)

    pairing = pairing_label or f"{a_name}_vs_{b_name}"

    a_wins = b_wins = draws = 0
    total_moves = 0
    total_sec = 0.0

    for g in range(games):
        # Alternate: even -> A starts (X), odd -> B starts (X)
        a_is_one = (g % 2 == 0)
        one = a_choice if a_is_one else b_choice
        two = b_choice if a_is_one else a_choice

        start_time = time.perf_counter()
        game = Game(one, two)
        winner_symbol = game.run()  # "X","O","D"
        elapsed = time.perf_counter() - start_time

        winner_agent = result_to_agent_label(winner_symbol, a_is_one)
        if winner_agent == "A":
            a_wins += 1
        elif winner_agent == "B":
            b_wins += 1
        else:
            draws += 1

        moves = count_moves(game.state)
        total_moves += moves
        total_sec += elapsed

        # CSV row
        writer.writerow([
            pairing,
            g + 1,
            "A" if a_is_one else "B",               # who started this game
            a_name, b_name,
            "X" if a_is_one else "O",               # A's symbol this game
            "O" if a_is_one else "X",               # B's symbol this game
            winner_agent,                           # "A"/"B"/"D"
            winner_symbol,                          # "X"/"O"/"D"
            moves,
            f"{elapsed:.6f}",
            f"{(elapsed / max(1, moves)):.6f}",
        ])

    games_played = float(games)
    avg_moves = total_moves / games_played if games_played else 0.0
    avg_spm = (total_sec / total_moves) if total_moves else 0.0  # seconds per move
    return {
        "A_wins": a_wins,
        "B_wins": b_wins,
        "draws": draws,
        "avg_moves": avg_moves,
        "avg_sec_per_move": avg_spm,
    }


def parse_agent(spec: str, qtable_path: Optional[str]) -> Tuple[str, ChoiceFn]:
    # Parse agent spec strings:
    #   - "random"
    #   - "mcts@1000"
    #   - "minmax@5"
    #   - "qlearn"  (loads q_table_connect4.json by default, or --qtable PATH)
    spec = spec.strip().lower()
    if spec == "random":
        return make_random()
    if spec.startswith("mcts@"):
        iters = int(spec.split("@", 1)[1])
        return make_mcts(iters)
    if spec.startswith("minmax@"):
        depth = int(spec.split("@", 1)[1])
        return make_minmax(depth)
    if spec == "qlearn":
        return make_q_learning(qtable_path)
    raise ValueError(f"Unknown agent spec: {spec}")


def main():
    parser = argparse.ArgumentParser(description="Connect Four tournament runner")
    parser.add_argument(
        "--csv", default="tournament_results.csv",
        help="Output CSV file path (default: tournament_results.csv)",
    )
    parser.add_argument(
        "--games", type=int, default=200,
        help="Games per pairing (100-200 typical). Default: 200",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed (optional)"
    )
    parser.add_argument(
        "--qtable", default=None,
        help="Path to q_table_connect4.json when using 'qlearn' agent (optional)",
    )
    parser.add_argument(
        "--pairs", nargs="+", default=[
            # sensible defaults; adjust freely
            "qlearn:random",
            "qlearn:mcts@1000",
            "qlearn:minmax@5",
            "mcts@1000:random",
            "minmax@5:random",
        ],
        help=(
            "List of pairings like 'AGENT1:AGENT2'. "
            "Agents: random | mcts@N | minmax@D | qlearn. "
            "Example: --pairs qlearn:random mcts@2000:minmax@4"
        ),
    )
    args = parser.parse_args()

    # Open CSV and write header
    with open(args.csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "pairing", "game_idx", "starter", "A_name", "B_name",
            "A_symbol", "B_symbol", "winner_agent", "winner_symbol",
            "moves", "seconds", "sec_per_move"
        ])

        grand = []  # collect (label, summary) for printing
        for pair in args.pairs:
            try:
                left, right = pair.split(":", 1)
            except ValueError:
                print(f"Skipping invalid pairing spec: {pair!r}")
                continue

            # Build agents
            try:
                a_name, a_choice = parse_agent(left, args.qtable)
                b_name, b_choice = parse_agent(right, args.qtable)
            except Exception as e:
                print(f"Skipping {pair!r}: {e}")
                continue

            label = f"{a_name}_vs_{b_name}"
            summary = run_pairing(
                a_name, a_choice, b_name, b_choice,
                games=max(1, args.games),
                writer=writer,
                pairing_label=label,
                seed=args.seed,
            )
            grand.append((label, summary))

    # Print summary
    print("\n=== Tournament Summary ===")
    for label, s in grand:
        total = s["A_wins"] + s["B_wins"] + s["draws"]
        print(f"\n{label}  (n={total})")
        print(f"  A_wins: {s['A_wins']}  B_wins: {s['B_wins']}  Draws: {s['draws']}")
        print(f"  Avg moves: {s['avg_moves']:.2f}  Avg sec/move: {s['avg_sec_per_move']:.6f}")

    print("\nCSV saved. Done.")


if __name__ == "__main__":
    main()


# How to use this file:
# ------------------------------------------------------------
# TL;DR - I want to pit any two agents against each other,
# alternate who starts, run ~100–200 games, dump a CSV, and
# print a quick “who actually won more” summary.
#
# Agent spec grammar (left:right):
#   - random
#   - qlearn                    (uses q_table_connect4.json by default, or --qtable PATH)
#   - mcts@N                    (e.g., mcts@1000 → 1,000 rollouts per move)
#   - minmax@D                  (e.g., minmax@5 → depth-5 minimax with alpha-beta)
#
# Quick starts (copy/paste):
#   # 200 games each (default), save to tournament_results.csv
#   python tournament.py --pairs qlearn:random
#
#   # Compare my Q vs search baselines (tweak N/D as you like):
#   python tournament.py --pairs qlearn:mcts@1000 qlearn:minmax@5
#
#   # Multiple pairs + custom game count + explicit CSV path:
#   python tournament.py --games 150 --csv results.csv \
#       --pairs qlearn:random mcts@1000:minmax@5
#
#   # If my Q-table lives somewhere else:
#   python tournament.py --qtable runs/q_table_connect4.json --pairs qlearn:random
#
# What the CSV contains per game:
#   pairing, game_idx, starter(A/B), A_name, B_name, A_symbol(X/O), B_symbol,
#   winner_agent(A/B/D), winner_symbol(X/O/D), moves, seconds, sec_per_move
#
# Summary print after it finishes:
#   For each pairing: A_wins, B_wins, Draws, Avg moves, Avg sec/move
# ------------------------------------------------------------
# Run times per agent: 10 min (could be more by +10-15 depending on CPU)