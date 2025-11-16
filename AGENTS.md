# Repository Guidelines

## Project Scope & Structure
We extend OpenSpiel with a JAX/Flax EfficientZeroV2 port. Core code lives in `open_spiel/python/algorithms/muzero_jax/`, tests in `open_spiel/python/algorithms/muzero_jax/tests/`, and parity checks reference `EfficientZeroV2/`. Keep `TODO.md`, `TODO_JAX_MUZERO.md`, and related task docs synchronized with every behavioral change.

## Living Documents
- `AGENTS.md` – source of truth for workflow/process. Update whenever expectations change.
- `README.md` (and repo-specific READMEs) – front door for users; reflect any interface or dependency updates.
- `TODO.md` – actionable task list (use `[ ]`/`[*]` states). Never start work without recording scope here.
- `TODO_JAX_MUZERO.md` – deep algorithm/status tracker; consolidate technical findings and coverage snapshots.

## Build, Test, and Run
- `./install.sh` once for toolchains; `python -m pip install -r requirements.txt` inside the project uv env (`source .venv/bin/activate`).
- Primary validation: `python run_tests_with_coverage.py --num-workers 12` to obtain true coverage; rerun immediately after any code edit.
- Broader OpenSpiel checks: `./open_spiel/scripts/build_and_run_tests.sh` when touching bindings/C++.
- MuZero-vs-MCTS evaluation: `python -m open_spiel.python.algorithms.muzero_jax.scripts.eval_vs_mcts --save-path /tmp/muzero_ttt_big64 --games 100 --mcts-simulations 400 --override hydra.job_logging.root.level=WARNING --override +hydra.job_logging.handlers.console.level=WARNING`. Increase `--mcts-simulations` to strengthen the pure MCTS baseline; the script alternates starting players automatically.
- Orchestrator defaults: periodic evaluation runs whenever `evaluation.interval > 0` unless configs explicitly set `evaluation.enabled=false`. Self-play episode metrics log each time `output.log_interval` divides `total_episodes`, while CLI verbosity (`-v`/`-vvv`) controls additional training/actor logs.
- Remote inference service toggle lives under the new `inference` config section. Leave `inference.remote_enabled=false` for single-process runs; flip it (and set `inference.rpc_endpoint`, batching knobs) when routing actors through `services/inference_client.py` + `services/inference_server.py`. Development-only local batching can be enabled with `inference.enable_local_batching=true` to emulate a remote RPC hop without deploying gRPC yet (see `docs/muzero_jax_distributed.md` for the full distributed architecture plan). When testing the gRPC parameter publisher/subscribe path in CI or on macOS/Metal, always publish once before subscribing to avoid transport-level blocking; the code path remains identical in production where publishers emit promptly.
- Distributed replay + parameter plane scaffolding: `services/replay_service.py` and `services/parameter_publisher.py` provide in-memory and gRPC implementations. Validate with `python -m pytest open_spiel/python/algorithms/muzero_jax/tests/services/test_replay_service.py open_spiel/python/algorithms/muzero_jax/tests/services/test_parameter_publisher.py`.
- Localhost distributed preset: `configs/presets/localhost_remote.yaml` routes orchestrator/actors through gRPC endpoints on a single host (set REPLAY_ENDPOINT/INFERENCE_ENDPOINT/PUBLISHER_ENDPOINT). Covered by `tests/self_play/test_distributed_actor_localhost.py` and `tests/test_orchestrator_localhost_remote.py`.
- Quick launcher: `scripts/launch_local_muzero_services.sh --replay 1 --publisher 1 --inference 2 --log-dir /tmp` starts services and prints endpoints/log paths for export.
- Auto-launch: when `remote_enabled=true` and no `rpc_endpoint` is provided, the orchestrator now starts local replay/publisher/inference gRPC servers automatically (CPU/GPU only). On TPU, inference auto-launch is skipped and remote mode is disabled so learner + inference stay in one process.
- Remote replay toggle: set `replay_buffer.remote_enabled=true replay_buffer.rpc_endpoint=HOST:PORT` to route the orchestrator through `RemoteReplayBufferAdapter`; minimal round-trip in `tests/test_replay_buffer_factory.py` and `tests/test_run_muzero_jax.py::test_remote_replay_buffer_factory_toggle`. Learner/actor wiring to these clients is the next milestone in TODO Task 11.
- Parameter publisher: enable with `publisher.remote_enabled=true publisher.rpc_endpoint=HOST:PORT` (optional `publisher.publish_interval` for periodic pushes). Actors/inference clients refresh before self-play episodes; learner publishes during training and on shutdown. Tests: `tests/services/test_parameter_publisher_integration.py`, `tests/test_parameter_publisher_factory.py`, `tests/self_play/test_actor_worker_param_refresh.py`.
- Coverage/CI runs set `JAX_PLATFORM_NAME=cpu` (and the async orchestrator tests enforce it) so that many worker processes don’t fight over the Metal GPU; do the same locally when running `run_tests_with_coverage.py`.

## TDD Workflow (Non-Negotiable)
We build only through tests. Break work into tiny, testable intents, author the tests first, and let failing tests dictate implementation:
- Tests express intent; implementation only changes to satisfy a failing test. Passing tests mean no code edits.
- Every function/class output and edge case receives explicit coverage; remove old tests before replacing behavior.
- Run the relevant tests after each change; never hack behavior or tests just to turn runs green.

## Avoiding Hacks & Temporary Scripts
Implementation code must remain free of hacks or hidden workarounds. If behavior diverges, either adjust production code or revise the intent (tests) transparently. For exploratory debugging, place throwaway helpers under `temporary_scripts/`, prefer inline snippets, and delete scripts once their purpose is met.

## Coding Style & Readability
- Python uses snake_case files/functions (e.g., `can_read_file`, `shift_data_array`); C++ stays CamelCase. Provide docstrings listing role plus typed inputs/outputs, and document any file formats emitted (with tests asserting schema).
- Keep functions under ~50 lines, avoid deep if/else cascades, and split files exceeding ~500 lines. Reuse helpers to prevent duplication.

## Communication & Q&A
Ask for clarification whenever requirements or debugging paths are ambiguous. If repeated attempts still misbehave, pause and request direction. When answering questions, anchor responses in our implementation; clearly label any general guidance (e.g., “Generally…”).

## Version Control & Environments
Commit after tests pass, staging only code/doc assets (no logs, weights, or raw data). Ensure the uv environment tied to this repo exists and is active before running Python commands so dependencies remain in sync.

## TODO Discipline
Open every task by adding unchecked bullets to `TODO.md`, execute each action item alongside its dedicated tests, and mark them checked only after the tests pass.
