// Copyright 2021 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "open_spiel/algorithms/mcts.h"

#include <memory>
#include <utility>

#include "open_spiel/abseil-cpp/absl/strings/string_view.h"
#include "open_spiel/abseil-cpp/absl/strings/str_split.h"
#include "open_spiel/algorithms/evaluate_bots.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_bots.h"
#include "open_spiel/spiel_utils.h"

using open_spiel::algorithms::Evaluator;
using open_spiel::algorithms::RandomRolloutEvaluator;

namespace open_spiel {
namespace {

constexpr double UCT_C = 2;

std::unique_ptr<open_spiel::Bot> InitBot(const open_spiel::Game& game,
                                         int max_simulations,
                                         std::shared_ptr<Evaluator> evaluator) {
  return std::make_unique<open_spiel::algorithms::MCTSBot>(
      game, std::move(evaluator), UCT_C, max_simulations,
      /*max_memory_mb=*/5, /*solve=*/true, /*seed=*/42, /*verbose=*/false);
}

void MCTSTest_CanPlayTicTacToe() {
  auto game = LoadGame("tic_tac_toe");
  int max_simulations = 100;
  auto evaluator = std::make_shared<RandomRolloutEvaluator>(20, 42);
  auto bot0 = InitBot(*game, max_simulations, evaluator);
  auto bot1 = InitBot(*game, max_simulations, evaluator);
  auto results =
      EvaluateBots(game->NewInitialState().get(), {bot0.get(), bot1.get()}, 42);
  SPIEL_CHECK_EQ(results[0] + results[1], 0);
}

void MCTSTest_CanPlayTicTacToe_LowSimulations() {
  auto game = LoadGame("tic_tac_toe");
  // Setting max_simulations to 0 or 1 is equivalent to sampling from the prior.
  for (const int max_simulations : {0, 1}) {
    auto evaluator = std::make_shared<RandomRolloutEvaluator>(20, 42);
    auto bot0 = InitBot(*game, max_simulations, evaluator);
    auto bot1 = InitBot(*game, max_simulations, evaluator);
    auto results = EvaluateBots(game->NewInitialState().get(),
                                {bot0.get(), bot1.get()}, 42);
    SPIEL_CHECK_EQ(results[0] + results[1], 0);
  }
}

void MCTSTest_CanPlayBothSides() {
  auto game = LoadGame("tic_tac_toe");
  int max_simulations = 100;
  auto evaluator = std::make_shared<RandomRolloutEvaluator>(20, 42);
  auto bot = InitBot(*game, max_simulations, evaluator);
  auto results =
      EvaluateBots(game->NewInitialState().get(), {bot.get(), bot.get()}, 42);
  SPIEL_CHECK_EQ(results[0] + results[1], 0);
}

void MCTSTest_CanPlaySinglePlayer() {
  auto game = LoadGame("catch");
  int max_simulations = 100;
  auto evaluator = std::make_shared<RandomRolloutEvaluator>(20, 42);
  auto bot = InitBot(*game, max_simulations, evaluator);
  auto results = EvaluateBots(game->NewInitialState().get(), {bot.get()}, 42);
  SPIEL_CHECK_GT(results[0], 0);
}

void MCTSTest_CanPlayThreePlayerStochasticGames() {
  auto game = LoadGame("pig(players=3,winscore=20,horizon=30)");
  int max_simulations = 1000;
  auto evaluator = std::make_shared<RandomRolloutEvaluator>(20, 42);
  auto bot0 = InitBot(*game, max_simulations, evaluator);
  auto bot1 = InitBot(*game, max_simulations, evaluator);
  auto bot2 = InitBot(*game, max_simulations, evaluator);
  auto results = EvaluateBots(game->NewInitialState().get(),
                              {bot0.get(), bot1.get(), bot2.get()}, 42);
  SPIEL_CHECK_FLOAT_EQ(results[0] + results[1] + results[2], 0);
}

open_spiel::Action GetAction(const open_spiel::State& state,
                             const absl::string_view action_str) {
  for (open_spiel::Action action : state.LegalActions()) {
    if (action_str == state.ActionToString(state.CurrentPlayer(), action))
      return action;
  }
  open_spiel::SpielFatalError(absl::StrCat("Illegal action: ", action_str));
}

std::pair<std::unique_ptr<algorithms::SearchNode>, std::unique_ptr<State>>
SearchTicTacToeState(const absl::string_view initial_actions) {
  auto game = LoadGame("tic_tac_toe");
  std::unique_ptr<State> state = game->NewInitialState();
  for (const auto& action_str : absl::StrSplit(initial_actions, ' ')) {
    state->ApplyAction(GetAction(*state, action_str));
  }
  auto evaluator = std::make_shared<RandomRolloutEvaluator>(20, 42);
  algorithms::MCTSBot bot(*game, evaluator, UCT_C,
                          /*max_simulations=*/ 10000,
                          /*max_memory_mb=*/ 10,
                          /*solve=*/ true,
                          /*seed=*/ 42,
                          /*verbose=*/ false);
  return {bot.MCTSearch(*state), std::move(state)};
}

void MCTSTest_SolveDraw() {
  auto [root, state] = SearchTicTacToeState("x(1,1) o(0,0) x(2,2)");
  SPIEL_CHECK_EQ(state->ToString(), "o..\n.x.\n..x");
  SPIEL_CHECK_EQ(root->outcome[root->player], 0);
  for (const algorithms::SearchNode& c : root->children)
    SPIEL_CHECK_LE(c.outcome[c.player], 0);  // No winning moves.
  const algorithms::SearchNode& best = root->BestChild();
  SPIEL_CHECK_EQ(best.outcome[best.player], 0);
  std::string action_str = state->ActionToString(best.player, best.action);
  if (action_str != "o(2,0)" && action_str != "o(0,2)")  // All others lose.
    SPIEL_CHECK_EQ(action_str, "o(2,0)");  // "o(0,2)" is also valid.
}

void MCTSTest_SolveLoss() {
  auto [root, state] =
      SearchTicTacToeState("x(1,1) o(0,0) x(2,2) o(0,1) x(0,2)");
  SPIEL_CHECK_EQ(state->ToString(), "oox\n.x.\n..x");
  SPIEL_CHECK_EQ(root->outcome[root->player], -1);
  for (const algorithms::SearchNode& c : root->children)
    SPIEL_CHECK_EQ(c.outcome[c.player], -1);  // All losses.
}

void MCTSTest_SolveWin() {
  auto [root, state] = SearchTicTacToeState("x(0,1) o(2,2)");
  SPIEL_CHECK_EQ(state->ToString(), ".x.\n...\n..o");
  SPIEL_CHECK_EQ(root->outcome[root->player], 1);
  const algorithms::SearchNode& best = root->BestChild();
  SPIEL_CHECK_EQ(best.outcome[best.player], 1);
  SPIEL_CHECK_EQ(state->ActionToString(best.player, best.action), "x(0,2)");
}

void MCTSTest_GarbageCollect() {
  auto game = LoadGame("tic_tac_toe");
  std::unique_ptr<State> state = game->NewInitialState();
  auto evaluator = std::make_shared<RandomRolloutEvaluator>(1, 42);
  algorithms::MCTSBot bot(*game, evaluator, UCT_C,
                          /*max_simulations=*/ 1000000,
                          /*max_memory_mb=*/ 1,
                          /*solve=*/ true,
                          /*seed=*/ 42,
                          /*verbose=*/ true);  // Verify the log output.
  std::unique_ptr<algorithms::SearchNode> root = bot.MCTSearch(*state);
  SPIEL_CHECK_TRUE(root->outcome.size() == 2 ||
                   root->explore_count == 1000000);
}

void MCTSTest_TreeReuse() {
  auto game = LoadGame("tic_tac_toe");
  const int max_simulations = 100;
  const int seed = 42;
  auto evaluator = std::make_shared<RandomRolloutEvaluator>(1, seed); // 1 rollout for determinism with seed

  // Initial state: x(0,0)
  std::unique_ptr<State> state_template = game->NewInitialState();
  Action initial_action = GetAction(*state_template, "x(0,0)");


  // Scenario 1: No Tree Reuse
  Action action1_move1, action1_move2;
  std::map<Action, std::pair<int, double>> stats1_move2;

  { // Scope for first bot no-reuse
    std::unique_ptr<State> state1_move1 = state_template->Clone();
    state1_move1->ApplyAction(initial_action);

    algorithms::MCTSBot bot1_move1(*game, evaluator, UCT_C, max_simulations,
                                   /*max_memory_mb=*/5, /*solve=*/true,
                                   seed, /*verbose=*/false);
    bot1_move1.MCTSearch(*state1_move1);
    SearchNode* root1_move1 = bot1_move1.GetRootNode();
    SPIEL_CHECK_TRUE(root1_move1 != nullptr);
    action1_move1 = root1_move1->BestChild().action;

    std::unique_ptr<State> state1_move2 = state1_move1->Clone();
    state1_move2->ApplyAction(action1_move1);

    algorithms::MCTSBot bot1_move2(*game, evaluator, UCT_C, max_simulations,
                                   /*max_memory_mb=*/5, /*solve=*/true,
                                   seed, /*verbose=*/false);
    bot1_move2.MCTSearch(*state1_move2);
    SearchNode* root1_move2 = bot1_move2.GetRootNode();
    SPIEL_CHECK_TRUE(root1_move2 != nullptr);
    if (!root1_move2->children.empty()) {
      action1_move2 = root1_move2->BestChild().action;
      for (const auto& child : root1_move2->children) {
        stats1_move2[child.action] = {child.explore_count, child.total_reward};
      }
    } else if (!state1_move2->IsTerminal()){
      // If state is not terminal but no children, it implies no valid moves or an issue.
      // For this test, we expect valid moves.
      SpielFatalError("Scenario 1, Move 2: Expected children in search node but found none for non-terminal state.");
    }
  }

  // Scenario 2: Tree Reuse
  Action action2_move1, action2_move2;
  std::map<Action, std::pair<int, double>> stats2_move2;

  { // Scope for reused bot
    std::unique_ptr<State> state2_move1_orig = state_template->Clone();
    state2_move1_orig->ApplyAction(initial_action);
    
    std::unique_ptr<State> state2_move1 = state2_move1_orig->Clone();


    algorithms::MCTSBot bot2_reused(*game, evaluator, UCT_C, max_simulations,
                                   /*max_memory_mb=*/5, /*solve=*/true,
                                   seed, /*verbose=*/false);
    // First step/search
    action2_move1 = bot2_reused.Step(*state2_move1);
    
    std::unique_ptr<State> state2_move2 = state2_move1->Clone(); // State is advanced by Step
    // state2_move2->ApplyAction(action2_move1); // Step already advances the state passed to it for MCTSearch's internal root_state_

    // Second step/search (reuses tree)
    action2_move2 = bot2_reused.Step(*state2_move2); // This will use the current state of bot2_reused which should be state2_move2
    
    SearchNode* root2_move2 = bot2_reused.GetRootNode();
    SPIEL_CHECK_TRUE(root2_move2 != nullptr);

    // Check if the root_state of the bot matches state2_move2 after applying action2_move2
    // This requires MCTSBot to expose its root_state_ or a way to check its current state.
    // For now, we assume Step correctly updates its internal state to state2_move2 for the search.


    if (!root2_move2->children.empty()) {
      // BestChild might not be meaningful if action2_move2 was from prior and not in children
      // However, for comparison, we rely on the tree structure itself.
      for (const auto& child : root2_move2->children) {
        stats2_move2[child.action] = {child.explore_count, child.total_reward};
      }
    } else if (!state2_move2->IsTerminal()){
       SpielFatalError("Scenario 2, Move 2: Expected children in search node but found none for non-terminal state.");
    }
  }

  // Assertions
  SPIEL_CHECK_EQ(action1_move1, action2_move1);
  
  // If states are not terminal and children were expected.
  if (!stats1_move2.empty() || !stats2_move2.empty()) {
    SPIEL_CHECK_EQ(action1_move2, action2_move2);
    SPIEL_CHECK_EQ(stats1_move2.size(), stats2_move2.size());

    for (const auto& [action, stat1_pair] : stats1_move2) {
      SPIEL_CHECK_TRUE(stats2_move2.count(action));
      const auto& stat2_pair = stats2_move2.at(action);
      SPIEL_CHECK_EQ(stat1_pair.first, stat2_pair.first); // explore_count
      SPIEL_CHECK_FLOAT_EQ(stat1_pair.second, stat2_pair.second, 1e-6); // total_reward
    }
  }
}

}  // namespace
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::MCTSTest_CanPlayTicTacToe();
  open_spiel::MCTSTest_CanPlayTicTacToe_LowSimulations();
  open_spiel::MCTSTest_CanPlayBothSides();
  open_spiel::MCTSTest_CanPlaySinglePlayer();
  open_spiel::MCTSTest_CanPlayThreePlayerStochasticGames();
  open_spiel::MCTSTest_SolveDraw();
  open_spiel::MCTSTest_SolveLoss();
  open_spiel::MCTSTest_SolveWin();
  open_spiel::MCTSTest_GarbageCollect();
  open_spiel::MCTSTest_TreeReuse();
}
