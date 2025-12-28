"""Closed-loop NNUE training for tic-tac-toe."""

# pylint: disable=c-extension-no-member
# pylint: disable=too-many-arguments,too-many-positional-arguments
# pylint: disable=too-many-locals

from __future__ import annotations

import argparse
import os
import random
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
import torch

import pyspiel

_Q = 127.0 / 64.0
_FEATURE_DIM = 29
_SIDE_TO_MOVE_OFFSET = 27


@dataclass(frozen=True)
class Sample:
    """Training sample with sparse feature indices and target value."""

    features: List[int]
    target: float


class TttNnueNet(torch.nn.Module):
    """Sparse NNUE network for tic-tac-toe."""

    def __init__(self, feature_dim: int = _FEATURE_DIM, l1: int = 64, l2: int = 32):
        super().__init__()
        self.feature_dim = feature_dim
        self.l1 = l1
        self.l2 = l2
        self.embed = torch.nn.EmbeddingBag(
            feature_dim, l1, mode="sum", include_last_offset=True
        )
        self.b0 = torch.nn.Parameter(torch.zeros(l1))
        self.fc1 = torch.nn.Linear(l1, l2)
        self.fc2 = torch.nn.Linear(l2, 1)

    def forward(self, indices: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        """Runs a sparse NNUE forward pass."""
        x = self.embed(indices, offsets) + self.b0
        x = torch.clamp(x, min=0.0, max=_Q)
        x = self.fc1(x)
        x = torch.clamp(x, min=0.0, max=_Q)
        x = self.fc2(x).squeeze(-1)
        return torch.tanh(x)

    def clip_parameters(self) -> None:
        """Clamps parameters into the quantization range."""
        with torch.no_grad():
            for param in self.parameters():
                if param.requires_grad:
                    param.clamp_(-_Q, _Q)


def _extract_features(state: pyspiel.State) -> List[int]:
    """Returns sparse feature indices for a tic-tac-toe state."""
    obs = np.array(state.observation_tensor(0), dtype=np.float32)
    obs = obs.reshape(3, 9)
    features = []
    for cell in range(9):
        state_idx = int(np.argmax(obs[:, cell]))
        features.append(cell * 3 + state_idx)
    player = state.current_player()
    if player >= 0:
        features.append(_SIDE_TO_MOVE_OFFSET + player)
    else:
        features.append(_SIDE_TO_MOVE_OFFSET)
    return features


def _bagify_features(
    batch: Sequence[Sample],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Builds embedding bag indices, offsets, and targets for a batch."""
    offsets = np.zeros(len(batch) + 1, dtype=np.int64)
    indices: List[int] = []
    targets = np.zeros(len(batch), dtype=np.float32)
    total = 0
    for i, sample in enumerate(batch):
        indices.extend(sample.features)
        total += len(sample.features)
        offsets[i + 1] = total
        targets[i] = sample.target
    return np.asarray(indices, dtype=np.int64), offsets, targets


def _evaluate_state(model: TttNnueNet, state: pyspiel.State, player: int) -> float:
    """Evaluates a state from the maximizing player's perspective."""
    features = _extract_features(state)
    device = next(model.parameters()).device
    indices = torch.tensor(features, dtype=torch.int64, device=device).unsqueeze(0)
    offsets = torch.tensor([0, len(features)], dtype=torch.int64, device=device)
    with torch.no_grad():
        value = model(indices, offsets).item()
    if state.current_player() == player:
        return value
    return -value


def _alpha_beta(
    model: TttNnueNet,
    state: pyspiel.State,
    depth: int,
    alpha: float,
    beta: float,
    maximizing_player: int,
) -> Tuple[float, int]:  # pylint: disable=too-many-arguments,too-many-positional-arguments
    """Alpha-beta search with a depth-limited NNUE value function."""
    if state.is_terminal():
        return state.player_return(maximizing_player), -1
    if depth == 0:
        return _evaluate_state(model, state, maximizing_player), -1

    player = state.current_player()
    if player == maximizing_player:
        best_value = -float("inf")
        best_action = -1
        for action in state.legal_actions():
            child = state.clone()
            child.apply_action(action)
            value, _ = _alpha_beta(
                model, child, depth - 1, alpha, beta, maximizing_player
            )
            if value > best_value:
                best_value = value
                best_action = action
            alpha = max(alpha, best_value)
            if alpha >= beta:
                break
        return best_value, best_action

    best_value = float("inf")
    best_action = -1
    for action in state.legal_actions():
        child = state.clone()
        child.apply_action(action)
        value, _ = _alpha_beta(model, child, depth - 1, alpha, beta, maximizing_player)
        if value < best_value:
            best_value = value
            best_action = action
        beta = min(beta, best_value)
        if alpha >= beta:
            break
    return best_value, best_action


def _select_action(
    model: TttNnueNet | None,
    state: pyspiel.State,
    depth: int,
    rng: random.Random,
    epsilon: float,
) -> int:
    """Chooses an action via alpha-beta or random fallback."""
    if model is None:
        return rng.choice(state.legal_actions())
    if epsilon > 0.0 and rng.random() < epsilon:
        return rng.choice(state.legal_actions())
    if depth <= 0:
        return rng.choice(state.legal_actions())
    _, action = _alpha_beta(
        model,
        state,
        depth,
        alpha=-float("inf"),
        beta=float("inf"),
        maximizing_player=state.current_player(),
    )
    if action < 0:
        return rng.choice(state.legal_actions())
    return action


def _play_game(
    game: pyspiel.Game,
    model_x: TttNnueNet | None,
    model_o: TttNnueNet | None,
    depth: int,
    rng: random.Random,
    epsilon: float,
) -> Tuple[List[Sample], int]:
    """Plays one game and returns training samples and winner."""
    state = game.new_initial_state()
    samples: List[Tuple[List[int], int]] = []
    while not state.is_terminal():
        player = state.current_player()
        model = model_x if player == 0 else model_o
        features = _extract_features(state)
        samples.append((features, player))
        action = _select_action(model, state, depth, rng, epsilon)
        state.apply_action(action)

    returns = state.returns()
    dataset = [
        Sample(features=features, target=float(returns[player]))
        for features, player in samples
    ]
    if returns[0] > returns[1]:
        winner = 0
    elif returns[1] > returns[0]:
        winner = 1
    else:
        winner = -1
    return dataset, winner


def _generate_selfplay(
    game: pyspiel.Game,
    model: TttNnueNet,
    depth: int,
    num_games: int,
    seed: int,
    epsilon: float,
) -> List[Sample]:
    """Generates self-play samples."""
    rng = random.Random(seed)
    dataset: List[Sample] = []
    for _ in range(num_games):
        samples, _ = _play_game(game, model, model, depth, rng, epsilon)
        dataset.extend(samples)
    return dataset


def _train_epoch(
    model: TttNnueNet,
    samples: Sequence[Sample],
    batch_size: int,
    lr: float,
    weight_decay: float,
    device: str,
    seed: int,
) -> float:  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    """Runs one training epoch."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(samples))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.MSELoss()
    model.train()
    total_loss = 0.0
    total_batches = 0

    for start in range(0, len(samples), batch_size):
        batch_ids = order[start : start + batch_size]
        batch = [samples[idx] for idx in batch_ids]
        indices, offsets, targets = _bagify_features(batch)
        indices_t = torch.tensor(indices, dtype=torch.int64, device=device)
        offsets_t = torch.tensor(offsets, dtype=torch.int64, device=device)
        targets_t = torch.tensor(targets, dtype=torch.float32, device=device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(indices_t, offsets_t)
        loss = criterion(outputs, targets_t)
        loss.backward()
        optimizer.step()
        model.clip_parameters()
        total_loss += loss.item()
        total_batches += 1
    return total_loss / max(total_batches, 1)


def _evaluate(
    game: pyspiel.Game,
    model_a: TttNnueNet,
    model_b: TttNnueNet | None,
    depth: int,
    num_games: int,
    seed: int,
) -> Tuple[
    int, int, int, int, int
]:  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    """Evaluates model A against model B or random."""
    rng = random.Random(seed)
    wins_a = 0
    wins_b = 0
    draws = 0
    wins_a_first = 0
    wins_a_second = 0
    for i in range(num_games):
        if i % 2 == 0:
            model_x, model_o = model_a, model_b
            a_as_first = True
        else:
            model_x, model_o = model_b, model_a
            a_as_first = False
        _, winner = _play_game(game, model_x, model_o, depth, rng, 0.0)
        if winner == -1:
            draws += 1
            continue
        if (winner == 0 and a_as_first) or (winner == 1 and not a_as_first):
            wins_a += 1
            if a_as_first:
                wins_a_first += 1
            else:
                wins_a_second += 1
        else:
            wins_b += 1
    return wins_a, wins_b, draws, wins_a_first, wins_a_second


def _format_eval(
    label_a: str,
    label_b: str,
    wins_a: int,
    wins_b: int,
    draws: int,
    wins_a_first: int,
    wins_a_second: int,
    games: int,
) -> str:
    """Formats evaluation summary."""
    def pct(num: int, den: int) -> float:
        return 100.0 * num / den if den > 0 else 0.0

    a_win_pct = pct(wins_a, games)
    b_win_pct = pct(wins_b, games)
    draw_pct = pct(draws, games)
    a_first_share = pct(wins_a_first, wins_a)
    a_second_share = pct(wins_a_second, wins_a)
    return (
        f"Comparing {label_a} vs {label_b}. {label_a} wins {a_win_pct:.1f}% "
        f"(of which {a_first_share:.1f}% first mover, {a_second_share:.1f}% "
        f"second mover), {label_b} wins {b_win_pct:.1f}%, draws {draw_pct:.1f}%"
    )


def _parse_depths(arg: str, default_depth: int) -> List[int]:
    """Parses a comma-separated list of depths."""
    if not arg:
        return [default_depth]
    depths = []
    for part in arg.split(","):
        part = part.strip()
        if part:
            depths.append(int(part))
    return depths or [default_depth]


def main() -> None:  # pylint: disable=too-many-locals
    """Runs the closed-loop training."""
    parser = argparse.ArgumentParser(description="Tic-tac-toe NNUE training")
    parser.add_argument("--output_dir", default="results/tic_tac_toe_nnue")
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--games_per_iter", type=int, default=2000)
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--eval_games", type=int, default=200)
    parser.add_argument("--eval_depths", default="")
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = args.device
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    game = pyspiel.load_game("tic_tac_toe")
    eval_depths = _parse_depths(args.eval_depths, args.depth)

    os.makedirs(args.output_dir, exist_ok=True)
    model = TttNnueNet().to(device)
    prev_model = None

    rng = random.Random(args.seed)
    for iteration in range(args.iterations):
        iter_dir = os.path.join(args.output_dir, f"iter_{iteration:02d}")
        os.makedirs(iter_dir, exist_ok=True)
        seed = rng.randint(0, 1_000_000)
        samples = _generate_selfplay(
            game,
            model,
            args.depth,
            args.games_per_iter,
            seed,
            args.epsilon,
        )
        loss = 0.0
        for epoch in range(args.epochs):
            loss = _train_epoch(
                model,
                samples,
                batch_size=args.batch_size,
                lr=args.lr,
                weight_decay=args.weight_decay,
                device=device,
                seed=args.seed + iteration * 100 + epoch,
            )
        torch.save(model.state_dict(), os.path.join(iter_dir, "nnue.pt"))
        print(
            f"iter {iteration}: games={args.games_per_iter} "
            f"samples={len(samples)} loss={loss:.4f}",
            flush=True,
        )

        model.eval()
        for eval_depth in eval_depths:
            wins_a, wins_b, draws, wins_a_first, wins_a_second = _evaluate(
                game,
                model,
                None,
                eval_depth,
                args.eval_games,
                seed + 1000 + eval_depth,
            )
            print(
                _format_eval(
                    f"iter-{iteration}",
                    "random",
                    wins_a,
                    wins_b,
                    draws,
                    wins_a_first,
                    wins_a_second,
                    args.eval_games,
                ),
                flush=True,
            )

        if prev_model is not None:
            prev_model.eval()
            for eval_depth in eval_depths:
                wins_a, wins_b, draws, wins_a_first, wins_a_second = _evaluate(
                    game,
                    model,
                    prev_model,
                    eval_depth,
                    args.eval_games,
                    seed + 2000 + eval_depth,
                )
                print(
                    _format_eval(
                        f"iter-{iteration}",
                        f"iter-{iteration - 1}",
                        wins_a,
                        wins_b,
                        draws,
                        wins_a_first,
                        wins_a_second,
                        args.eval_games,
                    ),
                    flush=True,
                )
        prev_model = TttNnueNet().to(device)
        prev_model.load_state_dict(model.state_dict())
        prev_model.eval()


if __name__ == "__main__":
    main()
