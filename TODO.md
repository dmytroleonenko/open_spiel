# TODO: Long Narde Implementation Plan

## Overview
Implementation of Long Narde rules, based on a copy of "games/backgammon".

**Key Rules:**
1.  **Setup:** White: 15 checkers on point 24. Black: 15 checkers on point 12.
2.  **Movement:** Both move counter-clockwise into home (White: 1–6, Black: 13–18), then bear off.
3.  **Starting:** Each rolls 1 die; higher goes first (White). No doubles on first roll.
4.  **Turns:** Roll 2 dice, move checkers exactly by each value. No landing on opponent. If no moves possible, skip turn. If only one die usable, use the higher value. Doubles grant 4 moves.
5.  **Head Rule:** Only 1 checker may leave the head per turn, except on the first turn if a double 6, 4, or 3 is rolled, allowing 2 checkers from the head.
6.  **Bearing Off:** Allowed once all checkers reach home. Use exact or higher rolls.
7.  **Ending/Scoring:** Mars (2 points) if loser bore off none; Oin (1 point) otherwise. `allow_last_roll_tie_` parameter enables optional tie rule.
8.  **Block (Bridge) Rule:** Forming a contiguous block of 6 checkers is illegal unless at least 1 opponent checker is ahead of the block.

## Current Status & Architecture

*   **Core Implementation:** All the rules are implemented.
*   **Code Structure:** Successfully modularized from monolithic `long_narde.cc` into `_state.cc`, `_moves.cc`, `_encoding.cc`, `_validation.cc`, `_legal_actions.cc`, `_api.cc`, `_utils.cc`, `_game.cc`.

## Plan

## MCTS Tree Reuse Plan

1. **Refactor MCTSBot to persist the search tree between moves**
   - ~~1.1: Add a member variable to MCTSBot to hold the current root SearchNode (e.g., `std::unique_ptr<SearchNode> root_`).~~ ✅ Done
   - ~~1.2: Add a member variable to MCTSBot to hold the current root state (e.g., `std::unique_ptr<State> root_state_`).~~ ✅ Done

2. **Update MCTSBot::Step and MCTSBot::MCTSearch to use the persistent root**
   - ~~2.1: On the first call, if `root_` is null or `root_state_` does not match the input state, create a new root as before.~~ ✅ Done
   - ~~2.2: On subsequent calls, after a move is made, update `root_` and `root_state_` to the child node and state corresponding to the move taken.~~ ✅ Done

3. **Implement root promotion after each move in `MCTSBot::Step`**
   - ~~3.1: After an `action` is selected (from `local_root->BestChild()` or prior sampling), `root_` is set to `local_root` (the result of `MCTSearch` from the current state).~~ ✅ Done
   - ~~3.2: `root_state_` is cloned from the current `state` and advanced by `action`.~~ ✅ Done
   - ~~3.3: Search `root_->children` (which were the children of `local_root`) for the child node corresponding to `action`.~~ ✅ Done
        - ~~If found, `root_`'s direct properties (action, player, prior, stats, outcome) are updated from this child, and the child's `children` vector (its subtree) is *moved* to become `root_->children`. This efficiently promotes the relevant subtree.~~ ✅ Done
        - ~~The original child node within the temporary list of children is effectively consumed/discarded after its subtree is moved.~~ ✅ Done
   - ~~3.4: If the chosen `action` is not found among `root_->children` (e.g., action was sampled from prior, or child was pruned):~~ ✅ Done
        - ~~`root_`'s properties are reset to represent a new frontier at `root_state_` (action is the chosen `action`, player is `root_state_->CurrentPlayer()`, stats/prior are zeroed, and `root_->children` remains empty).~~ ✅ Done

4. **Handle cases where the move taken is not in the current root's children**
   - ~~4.1: If the move taken is not found among the children (e.g., due to exploration or opponent move not previously simulated), discard the old tree and create a new root for the new state.~~ ✅ Done

5. **Ensure correct handling of chance nodes and opponent moves**
   - ~~5.1: For chance nodes, promote the child corresponding to the sampled chance action.~~ ✅ Done
   - ~~5.2: For opponent moves, promote the child corresponding to the opponent's action.~~ ✅ Done

6. **Update all MCTSBot entry points to use the persistent root**
   - 6.1: Ensure that all calls to `Step`, `StepWithPolicy`, and `MCTSearch` use and update the persistent root and state.

7. **Manage `ResetTree` functionality**
   - ~~7.1: `MCTSBot::ResetTree()` method implemented to clear `root_` and `root_state_` (e.g., for starting a new game).~~ ✅ Done
   - ~~7.2: Ensure `ResetTree()` is called appropriately (e.g., by `Restart()` or `RestartAt()` if those are intended to clear the MCTS tree, or by the game-playing loop at the start of new episodes). (This is a TODO for the calling code, not MCTSBot itself).~~ ✅ Done

8. **Test and validate**
   - 8.1: Add unit tests to verify that tree reuse works as intended and produces identical results to the original implementation when not reusing the tree.
   - 8.2: Add performance tests to confirm reduced redundant computation and improved efficiency.
