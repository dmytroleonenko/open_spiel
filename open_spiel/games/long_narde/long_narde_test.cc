// Copyright 2019 DeepMind Technologies Limited
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

#include "open_spiel/games/long_narde/long_narde.h"

#include <algorithm>
#include <random>

#include "open_spiel/spiel.h"
#include "open_spiel/tests/basic_tests.h"

namespace open_spiel {
namespace long_narde {
namespace {

namespace testing = open_spiel::testing;

bool ActionsContains(const std::vector<Action>& legal_actions, Action action) {
  return std::find(legal_actions.begin(), legal_actions.end(), action) !=
         legal_actions.end();
}

// Long Narde doesn't have hits, so we check that no hits are returned
void CheckNoHits(const State &state) {
  if (state.IsChanceNode() || state.IsTerminal()) {
    return;
  }
  Player player = state.CurrentPlayer();
  const auto &lnstate = down_cast<const LongNardeState &>(state);
  for (Action action : lnstate.LegalActions()) {
    std::vector<CheckerMove> cmoves = lnstate.AugmentWithHitInfo(
        player, lnstate.SpielMoveToCheckerMoves(player, action));
    for (CheckerMove cmove : cmoves) {
      // Verify that no hit is possible in Long Narde
      SPIEL_CHECK_FALSE(cmove.hit);
    }
  }
}

void BasicLongNardeTestsCheckNoHits() {
  std::shared_ptr<const Game> game = LoadGame("long_narde");
  testing::RandomSimTest(*game, 10, true, true, &CheckNoHits);
}

void BasicLongNardeTestsVaryScoring() {
  for (std::string scoring :
       {"winloss_scoring", "enable_gammons", "full_scoring"}) {
    auto game =
        LoadGame("long_narde", {{"scoring_type", GameParameter(scoring)}});
    testing::ChanceOutcomesTest(*game);
    testing::RandomSimTestWithUndo(*game, 10);
    testing::RandomSimTest(*game, 10);
  }
}

void BasicLongNardeTestsDoNotStartWithDoubles() {
  std::mt19937 rng;
  for (int i = 0; i < 100; ++i) {
    auto game = LoadGame("long_narde");
    std::unique_ptr<State> state = game->NewInitialState();

    while (state->IsChanceNode()) {
      Action outcome =
          SampleAction(state->ChanceOutcomes(),
                       std::uniform_real_distribution<double>(0.0, 1.0)(rng))
              .first;
      state->ApplyAction(outcome);
    }
    LongNardeState* long_narde_state =
        dynamic_cast<LongNardeState*>(state.get());
    // The dice should contain two different numbers,
    // because a tie would not select a starting player.
    SPIEL_CHECK_NE(long_narde_state->dice(0), long_narde_state->dice(1));
  }
}

// Test correct initial board setup for Long Narde
// White's 15 checkers on point 24, Black's 15 on point 12
void InitialBoardSetupTest() {
  std::shared_ptr<const Game> game = LoadGame("long_narde");
  std::unique_ptr<State> state = game->NewInitialState();
  LongNardeState* lnstate = static_cast<LongNardeState*>(state.get());
  
  // Check initial setup for White (kXPlayerId) - all 15 on point 24 (index 23)
  SPIEL_CHECK_EQ(lnstate->board(kXPlayerId, kWhiteHeadPos), kNumCheckersPerPlayer);
  
  // Check initial setup for Black (kOPlayerId) - all 15 on point 12 (index 11)
  SPIEL_CHECK_EQ(lnstate->board(kOPlayerId, kBlackHeadPos), kNumCheckersPerPlayer);
  
  // Verify no checkers anywhere else on the board
  for (int pos = 0; pos < kNumPoints; ++pos) {
    if (pos != kWhiteHeadPos) {
      SPIEL_CHECK_EQ(lnstate->board(kXPlayerId, pos), 0);
    }
    if (pos != kBlackHeadPos) {
      SPIEL_CHECK_EQ(lnstate->board(kOPlayerId, pos), 0);
    }
  }
}

// Test head rule: Only 1 checker can leave the head per turn
void HeadRuleTest() {
  std::shared_ptr<const Game> game = LoadGame("long_narde");
  std::unique_ptr<State> state = game->NewInitialState();
  LongNardeState* lnstate = static_cast<LongNardeState*>(state.get());
  
  // Set up a non-first turn situation with dice 3,4
  lnstate->SetState(
      kXPlayerId, false, {3, 4}, {0, 0}, {0, 0},
      {{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}});
  
  std::vector<Action> legal_actions = lnstate->LegalActions();
  
  // Count actions that move more than one checker from the head
  int multi_head_moves = 0;
  for (Action action : legal_actions) {
    std::vector<CheckerMove> moves = lnstate->SpielMoveToCheckerMoves(kXPlayerId, action);
    int head_moves = 0;
    for (const auto& move : moves) {
      if (move.pos == kWhiteHeadPos) head_moves++;
    }
    if (head_moves > 1) multi_head_moves++;
  }
  
  // Verify no actions allow more than one checker to leave the head
  SPIEL_CHECK_EQ(multi_head_moves, 0);
}

// Test first turn with doubles exception (6-6, 4-4, or 3-3)
void FirstTurnDoublesExceptionTest() {
  std::shared_ptr<const Game> game = LoadGame("long_narde");
  std::unique_ptr<State> state = game->NewInitialState();
  LongNardeState* lnstate = static_cast<LongNardeState*>(state.get());
  
  // Set up a first turn situation with dice 6,6
  lnstate->SetState(
      kXPlayerId, false, {6, 6}, {0, 0}, {0, 0},
      {{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}});
  
  // Mark as first turn
  lnstate->SetState(
      kXPlayerId, true, {6, 6}, {0, 0}, {0, 0},
      {{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}});
  
  std::vector<Action> legal_actions = lnstate->LegalActions();
  
  // Ensure there are actions that move two checkers from the head on special doubles
  int multi_head_moves = 0;
  for (Action action : legal_actions) {
    std::vector<CheckerMove> moves = lnstate->SpielMoveToCheckerMoves(kXPlayerId, action);
    int head_moves = 0;
    for (const auto& move : moves) {
      if (move.pos == kWhiteHeadPos) head_moves++;
    }
    if (head_moves > 1) multi_head_moves++;
  }
  
  // Verify that we have actions allowing two checkers to move from head on double 6
  SPIEL_CHECK_GT(multi_head_moves, 0);
}

// Test blocking bridge rule (cannot form 6 consecutive points that trap opponent)
void BlockingBridgeRuleTest() {
  std::shared_ptr<const Game> game = LoadGame("long_narde");
  std::unique_ptr<State> state = game->NewInitialState();
  LongNardeState* lnstate = static_cast<LongNardeState*>(state.get());
  
  // Set up a situation where White could complete a 6-point prime that would trap Black
  lnstate->SetState(
      kXPlayerId, false, {3, 2}, {0, 0}, {0, 0},
      {{0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 8},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}});
  
  std::vector<Action> legal_actions = lnstate->LegalActions();
  
  // Find the action that would complete the bridge (move from position 23 to position 10)
  bool can_create_bridge = false;
  for (Action action : legal_actions) {
    std::vector<CheckerMove> moves = lnstate->SpielMoveToCheckerMoves(kXPlayerId, action);
    for (const auto& move : moves) {
      if (move.pos == 22 && move.num == 3) { // Would create a 6-point bridge
        can_create_bridge = true;
      }
    }
  }
  
  // Verify that completing the bridge is not allowed
  SPIEL_CHECK_FALSE(can_create_bridge);
}

// Test movement direction - both players must move counter-clockwise
void MovementDirectionTest() {
  std::shared_ptr<const Game> game = LoadGame("long_narde");
  std::unique_ptr<State> state = game->NewInitialState();
  LongNardeState* lnstate = static_cast<LongNardeState*>(state.get());
  
  // Set up a situation with a checker moved out for each player
  lnstate->SetState(
      kXPlayerId, false, {3, 2}, {0, 0}, {0, 0},
      {{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 14},
       {0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 14, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}});
  
  // Check that White's legal moves are counter-clockwise (decreasing position numbers)
  std::vector<Action> white_actions = lnstate->LegalActions();
  bool white_clockwise_move_found = false;
  for (Action action : white_actions) {
    std::vector<CheckerMove> moves = lnstate->SpielMoveToCheckerMoves(kXPlayerId, action);
    for (const auto& move : moves) {
      int to_pos = lnstate->GetToPos(kXPlayerId, move.pos, move.num);
      if (to_pos > move.pos && !lnstate->IsOff(kXPlayerId, to_pos)) {
        white_clockwise_move_found = true;
      }
    }
  }
  SPIEL_CHECK_FALSE(white_clockwise_move_found);
  
  // Set Black as current player
  lnstate->SetState(
      kOPlayerId, false, {3, 2}, {0, 0}, {0, 0},
      {{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 14},
       {0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 14, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}});
  
  // Check that Black's legal moves are counter-clockwise (decreasing position numbers)
  std::vector<Action> black_actions = lnstate->LegalActions();
  bool black_clockwise_move_found = false;
  for (Action action : black_actions) {
    std::vector<CheckerMove> moves = lnstate->SpielMoveToCheckerMoves(kOPlayerId, action);
    for (const auto& move : moves) {
      int to_pos = lnstate->GetToPos(kOPlayerId, move.pos, move.num);
      if (to_pos > move.pos && !lnstate->IsOff(kOPlayerId, to_pos)) {
        black_clockwise_move_found = true;
      }
    }
  }
  SPIEL_CHECK_FALSE(black_clockwise_move_found);
}

// Test home regions: White (1-6) and Black (13-18)
void HomeRegionsTest() {
  std::shared_ptr<const Game> game = LoadGame("long_narde");
  std::unique_ptr<State> state = game->NewInitialState();
  LongNardeState* lnstate = static_cast<LongNardeState*>(state.get());
  
  // Check White's home region (points 1-6)
  for (int pos = 0; pos <= 5; ++pos) {
    SPIEL_CHECK_TRUE(lnstate->IsPosInHome(kXPlayerId, pos));
  }
  for (int pos = 6; pos < kNumPoints; ++pos) {
    SPIEL_CHECK_FALSE(lnstate->IsPosInHome(kXPlayerId, pos));
  }
  
  // Check Black's home region (points 13-18)
  for (int pos = 12; pos <= 17; ++pos) {
    SPIEL_CHECK_TRUE(lnstate->IsPosInHome(kOPlayerId, pos));
  }
  for (int pos = 0; pos < 12; ++pos) {
    SPIEL_CHECK_FALSE(lnstate->IsPosInHome(kOPlayerId, pos));
  }
  for (int pos = 18; pos < kNumPoints; ++pos) {
    SPIEL_CHECK_FALSE(lnstate->IsPosInHome(kOPlayerId, pos));
  }
}

// Test bearing off logic - must use exact or higher rolls when all checkers are in home
void BearingOffLogicTest() {
  std::shared_ptr<const Game> game = LoadGame("long_narde");
  std::unique_ptr<State> state = game->NewInitialState();
  LongNardeState* lnstate = static_cast<LongNardeState*>(state.get());
  
  // Set up a position where White has all checkers in home
  lnstate->SetState(
      kXPlayerId, false, {5, 3}, {0, 0}, {0, 0},
      {{5, 4, 3, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}});
  
  std::vector<Action> legal_actions = lnstate->LegalActions();
  
  // Check that bearing off with exact or higher rolls is allowed
  bool can_bear_off_pos_0 = false;  // Can bear off from position 0 with a 5
  bool can_bear_off_pos_4 = false;  // Can bear off from position 4 with a 5
  
  for (Action action : legal_actions) {
    std::vector<CheckerMove> moves = lnstate->SpielMoveToCheckerMoves(kXPlayerId, action);
    for (const auto& move : moves) {
      if (move.pos == 0 && move.num == 5) {
        can_bear_off_pos_0 = true;
      }
      if (move.pos == 4 && move.num == 5) {
        can_bear_off_pos_4 = true;
      }
    }
  }
  
  // Verify exact and higher bearing off is allowed
  SPIEL_CHECK_TRUE(can_bear_off_pos_0);  // Exact roll
  SPIEL_CHECK_TRUE(can_bear_off_pos_4);  // Higher roll
}

// Test scoring: mars (2 points) and oin (1 point)
void ScoringSystemTest() {
  std::shared_ptr<const Game> game = LoadGame("long_narde", 
                                            {{"scoring_type", GameParameter("enable_gammons")}});
  std::unique_ptr<State> state = game->NewInitialState();
  LongNardeState* lnstate = static_cast<LongNardeState*>(state.get());
  
  // Set up a position where White has borne off all checkers, Black has none
  lnstate->SetState(
      kOPlayerId, false, {5, 3}, {0, 0}, {15, 0},
      {{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}});
  
  // Check that White gets 2 points (mars) for bearing off all checkers when Black has none
  SPIEL_CHECK_TRUE(lnstate->IsTerminal());
  auto returns = lnstate->Returns();
  SPIEL_CHECK_EQ(returns[kXPlayerId], 2);
  SPIEL_CHECK_EQ(returns[kOPlayerId], -2);
  
  // Set up a position where White has borne off all checkers, Black has some
  lnstate->SetState(
      kOPlayerId, false, {5, 3}, {0, 0}, {15, 3},
      {{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}});
  
  // Check that White gets 1 point (oin) for bearing off all checkers when Black has some
  SPIEL_CHECK_TRUE(lnstate->IsTerminal());
  returns = lnstate->Returns();
  SPIEL_CHECK_EQ(returns[kXPlayerId], 1);
  SPIEL_CHECK_EQ(returns[kOPlayerId], -1);
}

// Test last roll tie feature
void LastRollTieTest() {
  std::shared_ptr<const Game> game = LoadGame("long_narde");
  std::unique_ptr<State> state = game->NewInitialState();
  LongNardeState* lnstate = static_cast<LongNardeState*>(state.get());
  
  // Set up a position where White has all checkers off, Black has 14 off and 1 in home
  lnstate->SetState(
      kOPlayerId, false, {5, 3}, {0, 0}, {15, 14},
      {{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}});
  
  // Verify that the game is not terminal, as Black should get one last roll
  SPIEL_CHECK_FALSE(lnstate->IsTerminal());
  
  // Now have Black bear off their last checker
  lnstate->SetState(
      kChancePlayerId, false, {0, 0}, {0, 0}, {15, 15},
      {{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}});
  
  // Now the game should be terminal
  SPIEL_CHECK_TRUE(lnstate->IsTerminal());
  
  // And the returns should indicate a tie
  auto returns = lnstate->Returns();
  SPIEL_CHECK_EQ(returns[kXPlayerId], 0);
  SPIEL_CHECK_EQ(returns[kOPlayerId], 0);
}

}  // namespace
}  // namespace long_narde
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::testing::LoadGameTest("long_narde");
  open_spiel::long_narde::BasicLongNardeTestsCheckNoHits();
  open_spiel::long_narde::BasicLongNardeTestsDoNotStartWithDoubles();
  open_spiel::long_narde::BasicLongNardeTestsVaryScoring();
  open_spiel::long_narde::InitialBoardSetupTest();
  open_spiel::long_narde::HeadRuleTest();
  open_spiel::long_narde::FirstTurnDoublesExceptionTest();
  open_spiel::long_narde::BlockingBridgeRuleTest();
  open_spiel::long_narde::MovementDirectionTest();
  open_spiel::long_narde::HomeRegionsTest();
  open_spiel::long_narde::BearingOffLogicTest();
  open_spiel::long_narde::ScoringSystemTest();
  open_spiel::long_narde::LastRollTieTest();
}
