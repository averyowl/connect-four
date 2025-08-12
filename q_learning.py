# This .py file is my Q-Learning agent tuned to push toward ~0.75 vs random.
# Jarren Calizo - Aug 11, 2025
#
# What I applied for smoother curves and strong play:
# - Longer training: default 200k episodes (patience pays).
# - Slightly stronger shaping: block_bonus=0.12, danger_penalty=0.25.
# - Bigger eval + moving average already in place (1,000 games, MA over last 3 points).
# - New today for robustness and higher ceiling:
#     • Per-(state,action) learning-rate decay (α shrinks as I revisit the same entry)
#     • Mixed-opponent training: self-play most of the time, but sometimes vs RANDOM and
#       sometimes vs a frozen greedy TARGET (my own snapshot). This makes me solid vs random
#       and reduces policy churn late in training.
# - Keep the good stuff: symmetry folding, frozen target (stable bootstrap), tiny center bias,
#   slower ε schedule (epsilon_min=0.08, decay=0.99985) to keep exploration alive.
#
# TL;DR: same architecture, just smarter training and a learning rate that cools down per entry.

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Tuple, List, Callable, Optional
import random
import json
from collections import defaultdict, deque

from board import Board
from game import Game

# Engine-facing choice function signature
ChoiceFn = Callable[[List[int], Board, str, str], int]


def random_agent(actions: List[int], state: Board, me: str, opp: str) -> int:
    """Baseline: uniform random among legal actions."""
    return random.choice(actions)


# Utilities

def _canonicalize_board(board: Board, me: str, opp: str) -> Tuple[str, ...]:
    """
    Represent the state from the current player's POV.
    Map my pieces → 'X', opponent → 'O'. This keeps one Q-table valid for both sides.
    """
    out: List[str] = []
    for col_str in board.board:
        mapped = []
        for ch in col_str:
            if ch == me:
                mapped.append('X')
            elif ch == opp:
                mapped.append('O')
            else:
                mapped.append(ch)
        out.append(''.join(mapped))
    return tuple(out)

def _mirror_view(canon: Tuple[str, ...]) -> Tuple[str, ...]:
    """Left–right mirror (reverse the 7 columns)."""
    return tuple(reversed(canon))

def _canon_fold(canon: Tuple[str, ...]) -> Tuple[str, ...]:
    """Fold symmetry by picking a stable representative between state and its mirror."""
    m = _mirror_view(canon)
    return canon if canon <= m else m


def has_immediate_win(state: Board, player: str) -> bool:
    """One-ply lookahead: can `player` win immediately from this state?"""
    acts = state.get_actions()
    for a in acts:
        nb, won = state.take_action(player, a)
        if won != "":
            return True
    return False


# Agent 

@dataclass
class QLearningAgent:
    # Hyperparameters tuned for stable learning and stronger play
    alpha: float = 0.3
    gamma: float = 0.99
    epsilon: float = 0.3           # start higher to explore broader
    epsilon_min: float = 0.08      # keep meaningful exploration late into training
    epsilon_decay: float = 0.99985 # slower decay → more diverse experience
    step_penalty: float = 0.0      # leave shaping to the terms below

    # Light reward shaping (intentionally small but firm enough to matter)
    center_bias: float = 0.01      # nudge toward center column (3)
    block_bonus: float = 0.12      # reward for removing your 1-ply win
    danger_penalty: float = 0.25   # penalty for handing you a 1-ply win

    # Frozen target for bootstrap stability
    use_target: bool = True

    # Per-(state,action) learning-rate decay
    alpha_min: float = 0.05
    alpha_decay_pow: float = 0.5   # α_t = max(alpha_min, alpha / (1 + n)^{pow})

    # Mixed-opponent training schedule (fractions of episodes)
    p_vs_random: float = 0.25      # 25% of episodes vs random opponent
    p_vs_target: float = 0.35      # 35% vs frozen greedy target; rest is pure self-play

    # Q-tables and visit counts
    q: Dict[Tuple[Tuple[str, ...], int], float] = field(default_factory=dict)
    q_target: Dict[Tuple[Tuple[str, ...], int], float] = field(default_factory=dict)
    visits: Dict[Tuple[Tuple[str, ...], int], int] = field(default_factory=dict)

    # Lookup helpers
    def value(self, key: Tuple[Tuple[str, ...], int]) -> float:
        return self.q.get(key, 0.0)

    def value_target(self, key: Tuple[Tuple[str, ...], int]) -> float:
        if self.use_target and self.q_target:
            return self.q_target.get(key, 0.0)
        return self.q.get(key, 0.0)

    def alpha_for(self, key: Tuple[Tuple[str, ...], int]) -> float:
        n = self.visits.get(key, 0)
        cooled = self.alpha / ((1.0 + n) ** self.alpha_decay_pow)
        return max(self.alpha_min, cooled)

    def best_action(self, state: Board, actions: List[int], me: str, opp: str) -> int:
        """Greedy argmax with random tie-breakers (after symmetry folding)."""
        s_key = _canon_fold(_canonicalize_board(state, me, opp))
        best_val = -float("inf")
        best: List[int] = []
        for a in actions:
            v = self.value((s_key, a))
            if v > best_val:
                best_val = v
                best = [a]
            elif v == best_val:
                best.append(a)
        return random.choice(best) if best else random.choice(actions)

    # Choice function
    def __call__(self, actions: List[int], state: Board, me: str, opp: str, *, train: bool = False) -> int:
        """
        Engine calls me like this. If train=True: ε-greedy. Otherwise: greedy.
        """
        if train and random.random() < self.epsilon:
            return random.choice(actions)
        return self.best_action(state, actions, me, opp)

    # Clean plug-in to Game(one_choice, two_choice)
    def choose(self, actions: List[int], state: Board, me: str, opp: str) -> int:
        return self.__call__(actions, state, me, opp, train=False)

    def sync_target(self) -> None:
        """Periodically copy Q → target to stabilize the bootstrap."""
        self.q_target = dict(self.q)

    # Core update
    def update(self, s: Board, a: int, r: float, s_next: Optional[Board], me: str, opp: str) -> None:
        """
        Tabular Q-learning with the *correct* two-player bootstrap.
        After I act, it's now opponent's turn. In zero-sum, my next value is
        the negative of the opponent's best value.

        Q[s,a] <- (1-α_t)Q[s,a] + α_t * target
        where α_t decays per (s,a): α_t = max(α_min, α / (1+n)^{pow})

          target = r                                     if terminal
                 = r - γ * max_a' Q_opp[s', a']          otherwise

        We also fold left–right symmetry so (state, mirror(state)) share the same entries.
        """
        s_key = _canon_fold(_canonicalize_board(s, me, opp))
        key = (s_key, a)
        old = self.value(key)

        if s_next is None:
            target = r
        else:
            next_actions = s_next.get_actions()
            if not next_actions:
                target = r  # board full → terminal
            else:
                # Opponent-to-move view + symmetry folding
                s2_key_for_opp = _canon_fold(_canonicalize_board(s_next, opp, me))
                opp_best = max(self.value_target((s2_key_for_opp, a2)) for a2 in next_actions)
                target = r - self.gamma * opp_best

        # Per-(s,a) learning rate
        a_t = self.alpha_for(key)
        self.q[key] = (1 - a_t) * old + a_t * target
        self.visits[key] = self.visits.get(key, 0) + 1

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # Persistence
    def save(self, path: str) -> None:
        """Pretty-print JSON so it isn't a single mega-line."""
        serial = [{"s": list(k[0]), "a": k[1], "q": v} for k, v in self.q.items()]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "alpha": self.alpha,
                    "gamma": self.gamma,
                    "epsilon": self.epsilon,
                    "epsilon_min": self.epsilon_min,
                    "epsilon_decay": self.epsilon_decay,
                    "step_penalty": self.step_penalty,
                    "center_bias": self.center_bias,
                    "block_bonus": self.block_bonus,
                    "danger_penalty": self.danger_penalty,
                    "use_target": self.use_target,
                    "alpha_min": self.alpha_min,
                    "alpha_decay_pow": self.alpha_decay_pow,
                    "p_vs_random": self.p_vs_random,
                    "p_vs_target": self.p_vs_target,
                    "table": serial,
                },
                f,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
            )

    @staticmethod
    def load(path: str) -> "QLearningAgent":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        agent = QLearningAgent(
            alpha=data.get("alpha", 0.3),
            gamma=data.get("gamma", 0.99),
            epsilon=data.get("epsilon", 0.3),
            epsilon_min=data.get("epsilon_min", 0.08),
            epsilon_decay=data.get("epsilon_decay", 0.99985),
            step_penalty=data.get("step_penalty", 0.0),
            center_bias=data.get("center_bias", 0.01),
            block_bonus=data.get("block_bonus", 0.12),
            danger_penalty=data.get("danger_penalty", 0.25),
            use_target=data.get("use_target", True),
            alpha_min=data.get("alpha_min", 0.05),
            alpha_decay_pow=data.get("alpha_decay_pow", 0.5),
            p_vs_random=data.get("p_vs_random", 0.25),
            p_vs_target=data.get("p_vs_target", 0.35),
        )
        for entry in data.get("table", []):
            s_tuple = tuple(entry["s"])  # list[str] -> tuple[str]
            a = entry["a"]
            agent.q[(s_tuple, a)] = entry["q"]
        # Target starts synced; visits start empty and will rebuild naturally
        agent.q_target = dict(agent.q)
        return agent


# Training & Evaluation

def _greedy_from_table(actions: List[int], state: Board, me: str, opp: str, table: Dict[Tuple[Tuple[str, ...], int], float]) -> int:
    """Greedy choice using a provided Q-table (used for frozen target opponent)."""
    s_key = _canon_fold(_canonicalize_board(state, me, opp))
    best_val = -float("inf")
    best: List[int] = []
    for a in actions:
        v = table.get((s_key, a), 0.0)
        if v > best_val:
            best_val = v
            best = [a]
        elif v == best_val:
            best.append(a)
    return random.choice(best) if best else random.choice(actions)


def train_self_play(
    agent: QLearningAgent,
    episodes: int = 200_000,
    eval_every: int = 2_000,
    eval_games: int = 1_000,  # bigger eval → smoother printouts
    seed: Optional[int] = None,
    eval_opponents: Optional[List[Tuple[str, ChoiceFn]]] = None,
    target_sync_every: int = 2_000,
    ma_k: int = 3,  # moving average window for printed metrics
) -> List[dict]:
    """
    Self-play with a *mixed* opponent schedule.
      • Most episodes: pure self-play (both sides learn via role relabeling)
      • Some episodes: vs RANDOM (teaches me to exploit random reliably)
      • Some episodes: vs frozen TARGET (teaches stability vs my own greedy snapshot)

    Rewards: +1 win, -1 loss, 0 draw. On non-terminal, tiny shaping:
      +center_bias for column 3, +block_bonus if I remove your 1-ply win, -danger_penalty if I give you a 1-ply win.
    """
    if seed is not None:
        random.seed(seed)

    history: List[dict] = []
    if eval_opponents is None:
        eval_opponents = [("random", random_agent)]

    # Ensure target is initialized
    if agent.use_target and not agent.q_target:
        agent.sync_target()

    # Per-opponent moving average buffers
    ma_buffers: Dict[str, deque] = defaultdict(lambda: deque(maxlen=ma_k))

    for ep in range(1, episodes + 1):
        board = Board()
        # Randomize who starts each episode
        current, other = ("X", "O") if random.random() < 0.5 else ("O", "X")

        # Pick opponent mode this episode
        r = random.random()
        if r < agent.p_vs_random:
            mode = "vs_random"
            opponent_fn: Optional[ChoiceFn] = random_agent
        elif r < agent.p_vs_random + agent.p_vs_target:
            mode = "vs_target"
            # freeze current target table into a closure
            table = dict(agent.q_target) if agent.use_target else dict(agent.q)
            opponent_fn = lambda actions, state, me, opp: _greedy_from_table(actions, state, me, opp, table)
        else:
            mode = "self"
            opponent_fn = None  # not used

        # Track the last (state, action) for the *learning* player only if needed
        last_sa: Dict[str, Optional[Tuple[Board, int]]] = {"X": None, "O": None}

        while True:
            actions = board.get_actions()
            if not actions:
                # Draw: credit previous moves of learning player(s) with 0
                for side in ("X", "O"):
                    # In self-play, both sides are the learner; in vs_* only one side is
                    learner_side = True if mode == "self" else (side == current_learner_start)
                    if learner_side and last_sa[side] is not None:
                        s_prev, a_prev = last_sa[side]
                        # me=side, opp=other_side
                        agent.update(s_prev, a_prev, 0.0, None, side, ("O" if side == "X" else "X"))
                break

            # Decide who is acting this turn and whether they are the learning agent
            if mode == "self":
                actor_is_learner = True
                actor_policy = lambda acts, st, me, opp: agent(acts, st, me, opp, train=True)
            else:
                # Choose which side is the learner at episode start
                if 'current_learner_start' not in locals():
                    current_learner_start = current  # whoever starts is my learner side
                if current == current_learner_start:
                    actor_is_learner = True
                    actor_policy = lambda acts, st, me, opp: agent(acts, st, me, opp, train=True)
                else:
                    actor_is_learner = False
                    actor_policy = opponent_fn  # type: ignore

            # Threat status *before* the move (relative to the *opponent* of the actor)
            opp_symbol = ("O" if current == "X" else "X")
            opp_threat_before = has_immediate_win(board, opp_symbol)

            # Pick and apply move
            a = actor_policy(actions, board, current, opp_symbol)
            s_before = board
            board, won = board.take_action(current, a)

            if won != "":
                # Win by current actor
                if actor_is_learner:
                    agent.update(s_before, a, +1.0, None, current, opp_symbol)
                # Penalize learning opponent's last move only if they are a learner
                if last_sa[opp_symbol] is not None and ((mode == "self") or (opp_symbol == current_learner_start)):
                    s_opp, a_opp = last_sa[opp_symbol]
                    agent.update(s_opp, a_opp, -1.0, None, opp_symbol, current)
                break
            else:
                # Non-terminal: if learner acted, apply shaping + bootstrap
                if actor_is_learner:
                    opp_threat_after = has_immediate_win(board, opp_symbol)
                    shaped = agent.step_penalty
                    if a == 3:
                        shaped += agent.center_bias
                    if opp_threat_before and not opp_threat_after:
                        shaped += agent.block_bonus
                    if opp_threat_after:
                        shaped -= agent.danger_penalty
                    agent.update(s_before, a, shaped, board, current, opp_symbol)

            # Track last move for both sides (only used when they are learners on terminal)
            last_sa[current] = (s_before, a)
            # swap turns
            current, opp_symbol = opp_symbol, current

        # Decay ε each episode
        agent.decay_epsilon()

        # Periodic target sync
        if agent.use_target and (ep % target_sync_every) == 0:
            agent.sync_target()

        # Periodic evaluation with moving average print
        if (ep % eval_every) == 0:
            summary = {"episode": ep}
            if eval_opponents is None:
                eval_list = [("random", random_agent)]
            else:
                eval_list = eval_opponents
            for name, opp in eval_list:
                wr = evaluate(agent, opp, games=eval_games, seed=seed)
                ma_buffers[name].append(wr)
                ma = sum(ma_buffers[name]) / len(ma_buffers[name])
                summary[f"winrate_vs_{name}"] = wr
                summary[f"ma{len(ma_buffers[name])}_vs_{name}"] = ma
            history.append(summary)
            print(summary)

    return history


def evaluate(agent: QLearningAgent, opponent: ChoiceFn, games: int = 200, seed: Optional[int] = None) -> float:
    """
    Greedy evaluation (ε=0). Alternate who starts to cancel first-move bias.
    Score: win = 1.0, draw = 0.5, loss = 0.0 → return average over games.
    """
    if seed is not None:
        random.seed(seed + 12345)

    wins = 0
    draws = 0

    def q_eval(actions: List[int], state: Board, me: str, opp: str) -> int:
        return agent(actions, state, me, opp, train=False)

    for g in range(games):
        if (g % 2) == 0:
            one, two = q_eval, opponent
            my_symbol = "X"
        else:
            one, two = opponent, q_eval
            my_symbol = "O"

        game = Game(one, two)
        result = game.run()
        if not result:
            result = getattr(game, "winner", "")
            if not result and len(game.state.get_actions()) == 0:
                result = "D"

        if result == my_symbol:
            wins += 1
        elif result == "D":
            draws += 1

    return (wins + 0.5 * draws) / games


# Quick CLI demo
if __name__ == "__main__":
    # These defaults aim high. Mixed-opponent training + per-entry α decay should give me a
    # steadier climb and less late regression while staying simple/tabular.
    agent = QLearningAgent(
        alpha=0.3,
        gamma=0.99,
        epsilon=0.3,
        epsilon_min=0.08,
        epsilon_decay=0.99985,
        step_penalty=0.0,
        center_bias=0.01,
        block_bonus=0.12,
        danger_penalty=0.25,
        use_target=True,
        alpha_min=0.05,
        alpha_decay_pow=0.5,
        p_vs_random=0.25,
        p_vs_target=0.35,
    )

    _history = train_self_play(
        agent,
        episodes=200_000,
        eval_every=2_000,
        eval_games=1_000,
        seed=42,
        eval_opponents=[("random", random_agent)],
        target_sync_every=2_000,
        ma_k=3,
    )

    agent.save("q_table_connect4.json")
    print("Saved Q-table to q_table_connect4.json (pretty JSON).")

    # Example head-to-head after training (greedy play):
    # game = Game(agent.choose, random_agent)
    # print("Winner:", game.run())
