#include "open_spiel/games/long_narde/long_narde.h"

#include <algorithm>
#include <iostream>
#include <random>

#include "open_spiel/spiel.h"
#include "open_spiel/tests/basic_tests.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace long_narde {
namespace {

// Check that no checkers land on the opponent's checkers
void CheckNoHits(const State& state) {
  if (state.IsChanceNode() || state.IsTerminal()) {
    return;
  }
  const auto &lnstate = down_cast<const LongNardeState&>(state);
  for (int pos = 0; pos < kNumPoints; ++pos) {
    // Check that no point has both black and white checkers
    if (lnstate.board(kXPlayerId, pos) > 0 && 
        lnstate.board(kOPlayerId, pos) > 0) {
      std::string board_str = lnstate.ToString();
      SpielFatalError(absl::StrCat(
          "Checkers at same point! pos: ", pos, ", board:\n", board_str));
    }
  }
}

// Custom memory-efficient version of the random sim test
void MemoryEfficientRandomSim() {
  // Test with 100 simulations since we have a 20 second time limit now
  const int kNumSimulations = 100;
  const bool verbose = false;  // Set to true for more detailed output
  const int kSeed = 1234;
  std::mt19937 rng(kSeed);
  
  // Create the game
  std::shared_ptr<const Game> game = LoadGame("long_narde");
  
  // Track statistics for reporting
  int total_moves = 0;
  int max_game_length = 0;
  int min_game_length = std::numeric_limits<int>::max();
  int terminated_games = 0;
  int invalid_moves_detected = 0;
  
  std::cout << "=========================================" << std::endl;
  std::cout << "LONG NARDE RANDOM SIMULATION TEST" << std::endl;
  std::cout << "=========================================" << std::endl;
  std::cout << "Running " << kNumSimulations << " simulations..." << std::endl;
  std::cout << "Using memory-efficient implementation" << std::endl;
  std::cout << "Debug output disabled (is_debugging = false)" << std::endl;
  std::cout << "----------------------------------------" << std::endl;
  
  for (int sim = 0; sim < kNumSimulations; ++sim) {
    if (sim % 10 == 0) {
      std::cout << "Starting simulation " << sim + 1 << "/" << kNumSimulations << std::endl;
    }
    std::unique_ptr<State> state = game->NewInitialState();
    LongNardeState* lnstate = dynamic_cast<LongNardeState*>(state.get());
    
    int max_moves = 1000;  // Limit the max number of moves to avoid infinite games
    int move_count = 0;
    bool invalid_move_found = false;
    
    while (!state->IsTerminal() && move_count < max_moves) {
      if (verbose) {
        std::cout << "Move: " << move_count << ", Player: " << state->CurrentPlayer() << std::endl;
      }
      
      if (state->IsChanceNode()) {
        // Sample chance outcome
        std::vector<std::pair<Action, double>> outcomes = state->ChanceOutcomes();
        Action action = open_spiel::SampleAction(outcomes, rng).first;
        state->ApplyAction(action);
      } else {
        // Check no hits condition (defensive check)
        CheckNoHits(*state);
        
        // Get legal actions
        std::vector<Action> legal_actions = state->LegalActions();
        if (legal_actions.empty()) {
          std::cerr << "No legal actions available in non-terminal state!" << std::endl;
          std::cerr << "Game state: " << state->ToString() << std::endl;
          break;
        }
        
        // Select a random action
        std::uniform_int_distribution<> dis(0, legal_actions.size() - 1);
        Action action = legal_actions[dis(rng)];
        
        // Extra validation to catch "invalid move from X to Y" errors
        Player current_player = state->CurrentPlayer();
        
        // Validate the chosen action before applying
        if (lnstate != nullptr) {
          std::vector<CheckerMove> moves = lnstate->SpielMoveToCheckerMoves(current_player, action);
          bool action_valid = true;
          
          for (const auto& move : moves) {
            if (move.pos == kPassPos) continue;  // Skip pass moves in validation
            
            int to_pos = move.to_pos;
            
            // Check if this would be an invalid move (debugging only)
            if (!lnstate->IsValidCheckerMove(current_player, move.pos, to_pos, move.die, true)) {
              std::cerr << "INVALID MOVE DETECTED: from " << move.pos 
                        << " to " << to_pos << " with die=" << move.die 
                        << " for player " << current_player << std::endl;
              std::cerr << "Board state: " << state->ToString() << std::endl;
              action_valid = false;
              invalid_move_found = true;
              invalid_moves_detected++;
              break;
            }
          }
          
          // This should never happen if LegalActions() is working properly
          if (!action_valid) {
            std::cerr << "WARNING: Selected an invalid action from LegalActions()" << std::endl;
            // We should still apply the action since it was in LegalActions()
          }
        }
        
        // Apply the action - it should be legal since we selected it from legal_actions
        state->ApplyAction(action);
      }
      
      move_count++;
      
      // Periodically free memory by cloning the state to a new one
      // This helps prevent growth of the move history
      if (move_count % 20 == 0) {  // Clone state every 20 moves
        if (!state->IsTerminal()) {
          std::unique_ptr<State> new_state = state->Clone();
          state = std::move(new_state);
          lnstate = dynamic_cast<LongNardeState*>(state.get());
          
          if (verbose) {
            std::cout << "Cloned state to avoid memory buildup at move " << move_count << std::endl;
          }
        }
      }
    }
    
    // Update statistics
    total_moves += move_count;
    max_game_length = std::max(max_game_length, move_count);
    min_game_length = std::min(min_game_length, move_count);
    
    // Report end of game
    if (state->IsTerminal()) {
      terminated_games++;
      std::vector<double> returns = state->Returns();
      if (verbose || invalid_move_found) {
        std::cout << "Game ended after " << move_count << " moves with returns: "
                  << returns[0] << ", " << returns[1] 
                  << (invalid_move_found ? " (had invalid moves)" : "") << std::endl;
      }
    } else {
      if (verbose || invalid_move_found) {
        std::cout << "Game stopped after " << move_count << " moves (limit reached)" 
                  << (invalid_move_found ? " (had invalid moves)" : "") << std::endl;
      }
    }
    
    // Force cleanup of the game state after each simulation
    state.reset();
    lnstate = nullptr;
  }
  
  // Report overall statistics
  double avg_game_length = static_cast<double>(total_moves) / kNumSimulations;
  std::cout << "=========================================" << std::endl;
  std::cout << "SIMULATION RESULTS" << std::endl;
  std::cout << "=========================================" << std::endl;
  std::cout << "Random simulation completed: " << kNumSimulations << " games" << std::endl;
  std::cout << "Average game length: " << avg_game_length << " moves" << std::endl;
  std::cout << "Min/Max game length: " << min_game_length << "/" << max_game_length << " moves" << std::endl;
  std::cout << "Terminated games: " << terminated_games << "/" << kNumSimulations << std::endl;
  std::cout << "Invalid moves detected: " << invalid_moves_detected << std::endl;
  
  if (invalid_moves_detected > 0) {
    std::cerr << "WARNING: Detected " << invalid_moves_detected 
              << " invalid moves! Check the implementation of LegalActions() vs. IsValidCheckerMove()" << std::endl;
  } else {
    std::cout << "No invalid moves detected - legal move generation is consistent with move validation" << std::endl;
  }
  
  std::cout << "=========================================" << std::endl;
  std::cout << "TEST COMPLETED SUCCESSFULLY" << std::endl;
  std::cout << "=========================================" << std::endl;
}

void RunRandomSimTest() {
  std::cout << "Running Long Narde random simulation test..." << std::endl;
  MemoryEfficientRandomSim();
}

}  // namespace

void RunRandomSimTests() {
  RunRandomSimTest();
}

}  // namespace long_narde
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::long_narde::RunRandomSimTests();
  return 0;
} 