# Long Narde NNUE: Known Conceptual Mismatches

This document captures the conceptual holes we have identified in the Long Narde
NNUE pipeline. These are not performance bugs; they are modeling inconsistencies
that cap learning no matter how many games you generate. Each issue includes the
exact code locations to review and the reasoning for why it blocks convergence.

## 1) Signed training targets vs non-negative NNUE EV (fundamental)

### Where it happens
- Targets are signed outcomes in LNUE:
  - `open_spiel/games/long_narde/long_narde_selfplay.cc` (see `FinalizeSamples`)
  - `outcome_value` is taken from `Returns()` in { -2, -1, +1, +2 }.
  - `target_value = alpha*search_value + (1 - alpha)*outcome_value`.
- NNUE EV is non-negative by construction:
  - `open_spiel/games/long_narde/long_narde_nnue.cc`
  - `eval.ev = eval.p_win + eval.p_mars`, both in [0, 1], so EV in [0, 2].
- Training regresses EV to the signed target:
  - `open_spiel/python/algorithms/long_narde_nnue/train.py`
  - `loss += ev_weight * mean((ev - target)^2)` where `ev` is `p_win+p_mars`.

### Why it is a hole
For side-to-move positions that are losing, `target_value` is negative. But the
EV head cannot go below 0. This means the EV regression loss always pushes
negative targets toward 0, regardless of how losing the position is.

This creates conflicting objectives:
- The BCE heads try to model win/mars probabilities.
- The EV regression tries to match a signed scalar it cannot represent.

The result is a training plateau that does not improve with more data.

### Consequence
Even with deep self-play, the model will not learn a well-calibrated value
function over [-2, 2] because it cannot represent the negative half of the
target distribution. This is a structural mismatch, not a data issue.

## 2) EV definition ignores losing severity (missing opponent mars)

### Where it happens
- EV is computed as:
  - `eval.ev = p_win + p_mars`
  - `open_spiel/games/long_narde/long_narde_nnue.cc`
- Only two heads exist:
  - `p_win` and `p_mars`
  - `open_spiel/games/long_narde/long_narde_nnue.h`
- Targets are in { -2, -1, +1, +2 }:
  - `open_spiel/games/long_narde/long_narde_selfplay.cc`

### Why it is a hole
The scoring model awards:
- +1 for a win
- +1 extra for a mars
- -1 for a loss
- -1 extra for losing by mars

With only `p_win` and `p_mars`, the model can represent the extra +1 on a win,
but it cannot represent the extra -1 on a loss. There is no head for "opponent
mars" or an explicit signed value head.

### Consequence
The evaluator cannot distinguish between "lose 1" and "lose 2" from the
side-to-move perspective. In practice this means the search will treat all
losing leaves as equally bad (or will compress them near 0), even though
the game rules differentiate them by score.

## 3) EV regression dominates by default (amplifies mismatch)

### Where it happens
- Default EV loss weight:
  - `open_spiel/python/algorithms/long_narde_nnue/closed_loop.py`
  - `--ev_weight` default is 0.5

### Why it is a hole
With a non-trivial `ev_weight`, the mismatched EV target in issue (1) actively
pulls the network away from the probability heads. This makes the conflict
stronger and accelerates plateauing, even if you increase data volume.

### Consequence
Increasing the number of games does not reliably improve the network because
the loss function keeps pulling toward an unreachable target distribution.

## 4) Search uses the same incompatible EV

### Where it happens
- Leaf evaluation is `eval.ev` with a sign flip:
  - `open_spiel/games/long_narde/long_narde_search.cc` (`EvaluatePreRoll`)
  - `eval.ev` from the NNUE is signed only by a player-to-move flip.

### Why it is a hole
The search is also driven by the non-negative EV definition. So the decision
policy used to generate training data is already biased toward a value signal
that cannot represent loss severity. This reduces the quality of self-play and
feeds the same bias back into training.

### Consequence
The self-play targets and the training loss reinforce the same mismatch,
creating a stable but suboptimal equilibrium.

## 5) Tic-tac-toe pipeline is not a 1:1 validation of Long Narde

### Where it happens
- Tic-tac-toe NNUE uses a single signed scalar output and MSE:
  - `open_spiel/python/algorithms/tic_tac_toe_nnue/train.py`
- Long Narde uses dual heads + EV regression:
  - `open_spiel/python/algorithms/long_narde_nnue/train.py`

### Why it matters
The tic-tac-toe loop can converge even if Long Narde cannot, because the output
head is compatible with signed targets. This means the tic-tac-toe success does
not imply the Long Narde NNUE is structurally sound.

## Summary

The key blocker is the mismatch between signed outcomes and a non-negative EV
head. Without fixing that, more (or deeper) data will not lead to consistent
improvement. The fix is architectural, not just "more training."

## Actionable Fix Options

Below are concrete, code-level fixes in increasing order of scope. Any of them
must make the EV head compatible with signed targets.

### Option A: Disable EV regression (minimal change)

If you want to keep the dual-head outputs but avoid the incompatible EV loss:
- Set `--ev_weight 0` in:
  - `open_spiel/python/algorithms/long_narde_nnue/closed_loop.py`
  - `open_spiel/python/algorithms/long_narde_nnue/train.py`

This does not fix the mismatch in search, but it removes a major source of
conflicting gradients.

### Option B: Compute a signed EV from the existing heads

Replace the EV target with a signed proxy computed from `p_win` and `p_mars`:
```
ev_signed = (2 * p_win - 1) + p_mars
```
- This yields [-1, 2] instead of [0, 2].
- It still cannot represent opponent mars directly, but it makes the EV term
  compatible with negative outcomes.

Relevant files:
- `open_spiel/python/algorithms/long_narde_nnue/train.py`
- `open_spiel/games/long_narde/long_narde_nnue.cc` (if you want parity in C++).

### Option C: Add a third head for opponent mars (most faithful)

Add `p_opp_mars` and compute EV as:
```
ev = p_win + p_mars - (1 - p_win) - p_opp_mars
```
- This yields a full [-2, 2] range with explicit loss severity.

This requires:
- Updating NNUE output dimension to 3.
- Updating all IO/export formats (`LNNU` header + payload).
- Updating training loss to include the new head.
- Updating C++ inference to compute EV from 3 heads.

Status: implemented (NNUE v2, 3 heads + signed EV) in:
- `open_spiel/games/long_narde/long_narde_nnue.h`
- `open_spiel/games/long_narde/long_narde_nnue.cc`
- `open_spiel/python/algorithms/long_narde_nnue/train.py`

### Option D: Use a single signed scalar head (simplest mathematically)

Replace win/mars heads with one signed scalar in [-2, 2] (or [-1, 1] scaled).
This is the cleanest representation but loses explicit mars probability
calibration. If mars behavior is critical, Option C is preferred.

## Performance Note (Apple Silicon / NEON dotprod)

This is not a conceptual bug, but it is a major performance limiter on M-series
chips if the NEON kernels do not use the SDOT (dotprod) instruction.

### Recommendation
- Use the dotprod-enabled kernel (`vdotq_s32`) for the int8 layer.
- Ensure the build enables dotprod: `-mcpu=apple-m3` or
  `-march=armv8.2-a+dotprod` (or `-march=native` on a local M3 machine).

### Why it matters
The L1->L2 int8 dot products dominate inference time. Without `SDOT`, the NEON
path falls back to widening multiplications (`vmull`) which are several times
slower. On M3-class cores, `SDOT` can yield multi-x speedups for NNUE eval.

This optimization should live in:
- `open_spiel/games/long_narde/long_narde_nnue_kernels.cc`

It is safe to keep a non-dotprod fallback for older ARM cores.

Status: enabled via CMake flag detection for
`open_spiel/games/long_narde/long_narde_nnue_kernels.cc`.

## Additional Recommendations (external review, partially implemented)

These are optimization/strength recommendations from an external reviewer.
Some are already partially present in the codebase; others are not implemented.
This section is intended to reduce ambiguity for anyone looking at the current
engine and deciding what to prioritize next.

### A) Depth 2 is strategically shallow for Long Narde

Rationale:
- Depth 2 means “I move, then I average your reply after a roll.”
- The engine does not see its *next* turn, so it cannot plan long blockades
  (primes) or longer tactical sequences.
- In practice, Depth 2 often caps win rate vs Random, especially when
  temperature adds noise.

Expected outcome:
- Depth 3 or 4 is typically required to reach >95% vs Random in race/fight
  positions. This is not just data volume; it is tree visibility.

Status: default depth bumped to 3 in:
- `open_spiel/games/long_narde/long_narde_search.h`
- `open_spiel/games/long_narde/long_narde_selfplay_main.cc`
- `open_spiel/games/long_narde/long_narde_eval_utils.h`
- `open_spiel/games/long_narde/long_narde_nats_worker_config.h`

### B) Move pruning (Top‑K / Delta / Beam) — already partially implemented

Current implementation:
- `open_spiel/games/long_narde/long_narde_search.cc`
  - `OrderedActions(...)` computes a shallow score for every legal move.
  - It then keeps:
    - at least `min_k` moves,
    - up to `top_k` moves,
    - and any move within `delta` of the best score.
- This is effectively a “value‑based beam width” rule, which matches the
  reviewer’s “narrow‑to‑wide” suggestion.

Status: root “panic widen” re‑search + root iterative deepening with widening
schedule implemented in:
- `open_spiel/games/long_narde/long_narde_search.h`
- `open_spiel/games/long_narde/long_narde_search.cc`

### C) Chance node pruning — implemented (optional)

Current implementation:
- All 21 dice outcomes are enumerated in chance nodes.
  - `open_spiel/games/long_narde/long_narde_search.cc`
    (`SearchState` at chance nodes).

Recommendation:
- If you add dice pruning, treat it as an *approximation* and validate with
  error audits. Naive “skip rare rolls” changes the expected value operator.
- Alternative: Monte Carlo dice sampling at deeper plies (unbiased estimator).

Status: chance sampling is available via:
- `SearchConfig` (`chance_samples`, `chance_sample_depth`, `chance_seed`)
- Default stays exact (set `chance_samples > 0` to enable sampling).
- Sampling applies when `depth <= chance_sample_depth` (<= 0 means all depths).

### D) Feature hints: pip count and mobility — implemented (NNUE/LNUE v3)

Reviewer’s rationale:
- The NNUE feature set is purely board‑local; it has to *learn* pip counts and
  mobility from scratch, which slows convergence.

Suggested additions:
- **Pip count delta** as a small dense scalar (my_pips - opp_pips).
- **Mobility** as a scalar hint. Option 1: expected number of legal actions
  over all 21 dice outcomes (pre‑roll), computed for both players (or as a
  delta) and bucketed to a small range.

Status: implemented via two sparse hint features:
- **Pip delta** bucket (my_pips - opp_pips).
- **Mobility** buckets for expected legal actions (pre‑roll) for current and
  opponent perspectives.

Changes include:
- Feature layout and IO bump (NNUE v3, LNUE v3).
- Feature computation in `open_spiel/games/long_narde/long_narde_nnue.cc`,
  cache update in `open_spiel/games/long_narde/long_narde_nnue_cache.cc`,
  and new helpers in `open_spiel/games/long_narde/long_narde_nnue_features.cc`.

### E) Training policy noise (temperature scheduling) — implemented

Current behavior:
- Softmax sampling with fixed temperature in self‑play.
  - `open_spiel/games/long_narde/long_narde_selfplay.cc`

Recommendation:
- Decay temperature quickly (or use a lower fixed value) to reduce noisy
  “teacher” moves, especially at low search depth.

Status: implemented with per‑ply temperature scheduling:
- `--temperature_end` sets the final temperature.
- `--temperature_decay_plies` controls the linear decay length.

### F) TD‑lambda / n‑step targets — implemented

Current behavior:
- Targets are mixed as `alpha*search_value + (1‑alpha)*outcome_value`.
  - `open_spiel/games/long_narde/long_narde_selfplay.cc`

Recommendation:
- Use n‑step bootstrapping (predicting search value a few turns ahead) to
  reduce outcome variance in long games.

Note:
- This is a training algorithm change; it is implemented in Python training.

Status: implemented via training-time n-step bootstrapping:
- `--bootstrap_steps N` recomputes EV targets using the search value from
  ply `N` ahead (fallbacks to current value if not found).
- `--bootstrap_alpha A` controls the blend with the final outcome
  (defaults to `--alpha` in closed-loop, 0.5 in standalone training).

### G) Replay buffer / mix-in — implemented

Rationale:
- Training on only the latest shards makes the target distribution drift
  rapidly and can cause catastrophic forgetting.

Status: closed-loop can mix recent shards during training:
- `--replay_iters N` (how many previous iterations to draw from).
- `--replay_frac F` (fraction of replay shards relative to current shards).
- `--replay_max_shards M` (cap on replay pool size).

### H) Root-only depth-3 (top-K full depth) — implemented

Rationale:
- Depth-3 everywhere is expensive; most root moves do not need the full depth.

Status: split-depth root search for move scoring:
- Evaluate all root moves at a reduced depth (defaults to `max_depth - 1`).
- Re-evaluate the top-K root moves at full depth.
- Controlled via `SearchConfig` / CLI flags:
  `root_full_depth_top_k` and `root_reduced_depth`.

### I) Mobility auxiliary head — proposed

Rationale:
- Mobility is a dense, low-noise signal that shortens feedback for blockade
  quality beyond sparse win/loss targets.

Recommendation:
- Add an auxiliary head that predicts expected opponent legal moves (or a
  mobility delta), trained with a small loss weight (e.g., 0.05–0.2 of EV).

Status: not implemented.

### J) Head-debt shaping — proposed

Rationale:
- Head moves are constrained (one per turn), so deferring legal head moves builds
  a backlog that the EV signal captures too late.

Recommendation:
- Add a tiny shaping term to targets or policy scoring: reward reductions in
  own head count; apply a small penalty when a head move is legal but skipped.

Status: not implemented.
