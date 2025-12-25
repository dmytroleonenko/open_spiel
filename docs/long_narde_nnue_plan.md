# Long Narde NNUE + Exact-Chance Expectiminimax Plan

Goal
Build a strong, auditable Long Narde engine by learning a fast incremental
evaluator (NNUE) and using it inside an expectiminimax search that handles dice
exactly (21 outcomes). Improve via an iterative self-play loop with stable
targets, strict legality, and reproducible search.

Locked design decisions
- NNUE outputs: dual heads (win + mars). Scalar EV is derived as EV = P(win) + P(mars).
- Features: base 800 one-hot features plus run/prime connectivity features.
- Value scale: train heads as probabilities; no standalone scalar head needed.
- SIMD: AVX2 baseline with AVX512 runtime dispatch; ISA abstraction allows NEON later.
- Parallelism: focus on coarse-grained self-play workers first; avoid root search
  parallelism until later.

Status
- [x] Confirm scope/constraints and map current long_narde C++ state to this plan
- [~] Design NNUE feature extractor + canonicalization + rule_flags/TT keys
- [x] Implement expectiminimax with exact 21-outcome chance nodes and compound actions
- [x] Build self-play data generator (pre-roll-only samples) and target mixing
- [~] Integrate training/export pipeline (PyTorch -> quantized weights) and C++ NNUE glue
- [~] Add audits/tests and performance benchmarks; document workflow

1) Game and State Specification

1.1 Game model
- Two-player, zero-sum, perfect information, stochastic via dice.
- No hitting/bar. Both players move same direction (canonical orientation).
- Turn:
  1) Roll dice.
  2) Execute a compound action resolving the whole turn under rules:
     - Use both dice if possible.
     - If only one die can be used, must use the higher.
     - If no legal moves, pass.
     - Head rule + opening exception per ruleset.
     - Illegal-prime restriction: a 6-point prime must not fully trap opponent.
     - Bearing off: only when all 15 in home; overkill only from farthest checker.
     - Scoring: oin=1, mars=2 (signed).

1.2 State must include rule-relevant flags
State = (board, side_to_move, rule_flags)
Rule flags include anything affecting legality or terminal conditions, e.g.:
- Opening-exception eligibility (first turn / per-side first move status).
- Any repetition/draw counters (if defined).
- Any variant-specific constraints not derivable from board counts.

Hard constraint: TT keys, feature extraction, and dataset samples must include
all rule flags required for correctness.

2) Value Definition and Search Tree Shape

2.1 Learn only pre-roll value
Define NNUE to approximate:
V(s) = E[final signed score | s is a pre-roll state]
Final signed score is in {-2, -1, +1, +2} from side-to-move perspective.

2.2 Enforce full-turn plies so leaves are always pre-roll
Tree alternation is fixed:
Pre-roll -> Chance (21 dice) -> Decision (compound action) -> Next pre-roll
Depth is measured in full turns. Cutoffs occur only at pre-roll nodes.

Hard constraint: NNUE is never queried on post-roll states. If a post-roll
estimate is needed, compute it as max/min over actions of NNUE(next_preroll).

3) Input Representation (NNUE features)

3.1 Base mass-conserving sparse features
- For each side (STM and opponent):
  - 24 points x bucket(count 0..15) one-hot
  - borne-off x bucket(count 0..15) one-hot
- Total: 800 one-hot features; exactly 50 active (25 per side: 24 + offboard).

Canonicalize to side-to-move orientation so "forward" and adjacency are stable.

3.2 Connectivity / run features (inductive bias, not reward shaping)
Add sparse pattern features computed from a blocked/occupied bitset. For Long
Narde, "blocked" means count >= 1 (a single checker blocks the point).
- For run length L in {2,3,4,5,6} and start index i:
  - Run(L, i, side) is active if side has blocking presence on all points i..i+L-1
Optionally add:
- "max run length" indicators
- "run count >= k" indicators

3.3 Optional dense global scalars (small, high ROI)
Append a tiny dense vector post-accumulator:
- pip counts (both sides)
- bearing-off phase indicators
- max run length / number of runs >= 4
- opponent rearmost checker index (canonical)

4) NNUE Architecture and Output Heads

4.1 Architecture
- Sparse accumulator -> hidden (e.g., 256-512) -> small MLP -> outputs
- Incremental update per move only touches changed point-count buckets and runs.

4.2 Dual-head outputs (recommended)
Predict:
- p_win in [0,1]
- p_mars in [0,1] (probability of winning with mars)

Compose expected signed score:
- For STM: EV = 1 * p_win + 1 * p_mars
- Signed to [-2,2] via perspective; opponent perspective flips sign.

Loss:
- BCE for heads, optionally class-balanced for mars rarity.
- Optional auxiliary loss on composed EV (small weight).

5) Search: Exact Chance + Move-Pruned Expectiminimax

5.1 Chance nodes: enumerate all 21 outcomes always
- Expand all 21 unordered dice outcomes with correct probabilities every time.
- No dice Monte Carlo in main search (training or evaluation).

Hard constraint: chance EV is computed exactly.

5.2 Decision nodes: compound actions only
Generate all legal compound turn outcomes for the rolled dice, including pass.
Deduplicate by hashing the resulting next pre-roll state.

5.3 Decision pruning: prune moves, not dice
To control branching:
1) Generate all compound actions.
2) Pre-score each action by evaluating NNUE on next pre-roll state.
3) Sort; keep:
   - top-K
   - any move within delta of best score
   - a minimum floor Kmin

Use iterative deepening; carry PV move first.

5.4 Determinism and tie-breaking
To prevent "same state, different label" noise:
- Fixed tie-breaking (stable sort by score then by move-id).
- Deterministic TT replacement policy.
- Deterministic RNG only for self-play exploration, not search value.

6) Transposition Table (TT)

6.1 Keys
- Pre-roll key: canonical board + rule_flags + side_to_move
- Decision key: above + unordered dice (a <= b)

6.2 Stored info
- depth (full turns)
- exact value (since chance is exact)
- best move (PV)
- optional bounds if using alpha-beta-like pruning variants

Hard constraint: no TT collisions across differing rule_flags.

7) Move Generation: Correctness First, Then Speed

7.1 "Move generator is the source of truth"
Legality must enforce:
- head rule + opening exception
- must-use-both-dice / else higher
- bearing-off exactness + farthest-checker overkill rule
- illegal-prime rule (as defined)
- pass generation when no moves

7.2 Illegal-prime check as a fast predicate
Implement legality as:
- apply candidate compound action -> resulting position
- compute whether a newly formed 6-prime is "trapping" per rule
- reject illegal outcomes

Optimize with:
- per-side blocked bitsets
- incremental updates localized to changed points
- caching (position, dice) -> list of legal compound actions

7.3 Unit tests (non-negotiable)
- legality regression suite (hand-crafted tricky cases)
- symmetry checks (canonicalization, dice order)
- generator completeness (no missing legal moves)

8) Data Generation (Self-Play) and Targets

8.1 What to log (only at pre-roll states)
For each visited pre-roll state:
- features
- search target (exact-chance expectiminimax EV)
- terminal outcome (signed score: +/-1 or +/-2)

8.2 Exploration policy
During self-play, at the decision node for the rolled dice:
- sample moves from softmax over search scores with temperature schedule
- optional root-only noise in openings

Keep search deterministic; exploration only changes visited states.

8.3 Target mixing for stability
y = alpha * EV_search + (1 - alpha) * EV_outcome

For dual-head training:
- derive head targets from outcome (win/mars)
- or train heads on outcome-only while using scalar EV mix as auxiliary target

Alpha schedule:
- higher when search depth is larger / pruning is conservative
- lower early in training and/or when model is unstable
- keep some outcome grounding throughout

9) Training Loop and Deployment
1) Self-play with current engine -> dataset of pre-roll samples.
2) Train NNUE (GPU) with:
   - BCE heads + calibration/EV auxiliary
   - mars balancing (oversample mars or weighted loss)
3) Export -> quantize (int8/int16) -> serialize with headered NNUE format ->
   integrate into C++ evaluator (raw blobs allowed only for legacy loads).
4) Verification gate before using new net:
   - legality tests pass
   - symmetry tests pass
   - fixed-position EV regression within tolerance
5) Repeat.

10) Evaluation, Audits, and Metrics

10.1 Strength evaluation
- Deterministic move choice (argmax) for benchmarking
- Exact dice expansion everywhere
- Elo / win rate vs baselines and previous versions

10.2 Audits (correctness + stability)
- unordered dice equivalence
- canonicalization invariance checks
- "best-move stability" under pruning parameters (K, delta)
- illegal-prime correctness suite

10.3 Performance metrics
- nodes/sec by node type (chance vs decision)
- average legal compound actions per dice after dedup
- NNUE evals/sec and accumulator update cost
- cache hit rates for TT and move lists

11) LNUE Shard Format (self-play output)

11.1 File header (fixed-size, little-endian)
- magic "LNUE"
- format_version u32
- endianness marker u32 (0x01020304)
- feature_schema_id u32
- total_feature_dim u32
- flags bitset (has_run_features, dual_head, has_game_id, has_ply)
- feature_index_bytes u32 (u16 expected for current dims)
- seed u64
- shard_id u32
- run_block_threshold u32
- reserved u32[5]

11.2 Chunk header (fixed-size, per chunk)
- magic "CHNK"
- uncompressed_bytes u32
- compressed_bytes u32 (0 if uncompressed)
- num_samples u32
- reserved u32[3]

11.3 Chunk payload (columnar)
- feat_offsets u32[num_samples + 1]
- feat_indices u16[feat_offsets[-1]]
- v_search f32[num_samples]
- outcome i8[num_samples] in {-2,-1,+1,+2}
- target f32[num_samples]
- game_id u64[num_samples]
- ply u16[num_samples]

Note: compression is reserved; start uncompressed and add zstd later.

12) Golden Pipeline Harness (correctness gate)

12.1 Determinism (C++ self-play)
- Run self-play twice with identical args and seed (single worker).
- Expect bit-identical LNUE shards (hash match).
  Example: `python -m open_spiel.python.algorithms.long_narde_nnue.golden_pipeline`.

12.2 LNUE reader sanity (Python)
- Validate header fields (magic/version/schema/feature_dim/flags/index_bytes).
- Validate payload integrity (offsets monotonic, indices bounds).
- Validate label ranges (v_search/target in [-2, 2], outcome in {-2,-1,+1,+2}).

12.3 NNUE round-trip (Python -> C++)
- Train one optimizer step on a small batch.
- Export NNUE (LNNU) and reload in C++.
- Compare raw logits for N fixed samples between Python and C++.
  Uses `long_narde_lnue_probe` under `build/games`.

13) Python Training Pipeline

13.1 Reader
- LNUE shard reader yields chunked columnar arrays.
- Feature indices are used directly to avoid feature drift between C++ and Python.

13.2 Trainer
- EmbeddingBag for sparse accumulator, two-headed BCE loss (win/mars).
- Auxiliary EV loss on sigmoid(win)+sigmoid(mars) vs target.
- Quantization-aware clamp + optional quantization penalty.
- Export headered NNUE binary (magic "LNNU") aligned with C++ loader.

Project-critical constraints recap
- NNUE queried only at pre-roll leaves (depth in full turns).
- Chance nodes always exact (21 outcomes).
- Legality enforced only by the move generator (no "mask later" shortcuts).
- TT keys include rule flags (no hidden-state collisions).
- Connectivity features via inputs, not reward shaping.
- Deterministic search value computation; randomness only for self-play exploration.
