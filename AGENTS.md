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
