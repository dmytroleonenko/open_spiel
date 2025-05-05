## Action Plan

### Refactor Long Narde Action Space

1. Analyze current action encoding in long_narde and document all sources of action space inflation.
2. Redesign the action encoding to:
    - Remove die value from the action encoding (use only positions, not dice values).
    - Remove dice order from the action encoding (treat all dice orders as equivalent).
    - Use minimal base for encoding (base-25: 24 points + pass).
    - For doubles, encode only the set of up to 4 moves using a canonical order.
    - Ensure the action space matches the theoretical minimum (target: 1250 unique actions).
3. Refactor the encoding and decoding logic in long_narde to implement the new scheme.
4. Update the action space size constant and all related documentation/comments.
5. Update the Python bindings and any wrappers if they depend on the action space size or encoding.

### Test Suite Review and Update

6. Review every test file matching *test*.cc for long_narde:
    - Identify all tests that depend on the action encoding or action space size.
    - Update tests to use the new encoding and expected action space.
    - Add/adjust tests to verify that the new encoding is correct and minimal.
7. Run the full test suite and verify all tests pass with the new encoding.
8. Document any changes in test coverage or behavior due to the encoding refactor.

---

### HIGH PRIORITY: Action Encoding Bug Prevents Playing Lower Die First

**Issue:** The current action encoding and decoding mechanism prevents the agent from choosing a valid first half-move if it involves using the lower die when the higher die is also playable from the same starting position.

**Cause:**
1.  **Ambiguous Encoding:** The `LegalActions` function (in `open_spiel/games/long_narde/long_narde_legal_actions.cc`) generates Spiel Action IDs based solely on the starting position (`move.pos`) of the first half-move in a valid sequence. If multiple sequences start by moving the same checker but using different dice (e.g., dice 2-1, checker at pos 17 can move via die 2 to 15 OR via die 1 to 16), both generate the *same* Action ID (e.g., `17`). See lines `36-40` in `long_narde_legal_actions.cc`.
2.  **Deterministic Decoding to Higher Die:** The `LongNardeSpielMoveToCheckerMoves` function (in `open_spiel/games/long_narde/long_narde_encoding.cc`), when decoding a non-pass Action ID, iterates through all possible valid half-moves (`LongNardeGenerateAllHalfMoves`). If it finds multiple half-moves matching the Action ID's position (e.g., `pos == 17`), it *always* selects the one using the *higher* die value (`m.die > best_move.die`). See lines `37-48` in `long_narde_encoding.cc`.
3.  **Consequence:** When `DoApplyAction` (in `open_spiel/games/long_narde/long_narde_api.cc`) receives an ambiguous Action ID like `17`, it invariably decodes and applies the move corresponding to the higher die (e.g., `(17, 15, 2)`). The agent is *never* given the option to choose the perfectly legal first half-move using the lower die (e.g., `(17, 16, 1)`).

**Impact:** This fundamentally limits the agent's available actions, contradicting the rules of Long Narde which allow playing either die first if both moves are possible. It prevents exploring potentially advantageous lines of play that start with the lower die. This needs to be addressed by the action space refactoring outlined above to ensure each distinct first half-move corresponds to a unique Spiel Action ID.

**Relevant Files:**
- `open_spiel/games/long_narde/long_narde_api.cc` (specifically `DoApplyAction`)
- `open_spiel/games/long_narde/long_narde_legal_actions.cc` (specifically `LegalActions`, `LongNardeFilterBestMoveSequences`)
- `open_spiel/games/long_narde/long_narde_encoding.cc` (specifically `LongNardeSpielMoveToCheckerMoves`, `LongNardeCheckerMovesToSpielMove`)
