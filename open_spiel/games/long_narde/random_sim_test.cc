#include "open_spiel/games/long_narde/long_narde.h"

#include <algorithm>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "open_spiel/spiel.h"
#include "open_spiel/tests/basic_tests.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace long_narde {
namespace {

// Default values for simulation parameters
constexpr int kDefaultNumSimulations = 5;
constexpr int kDefaultSeed = 1224;

struct DebugStep {
  int move_index;
  Player cur_player;
  std::string state_string;
  std::string action_description;
};

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
void MemoryEfficientRandomSim(int num_simulations = kDefaultNumSimulations, 
                              int seed = kDefaultSeed) {
  const bool verbose = false;  // Set to true for more detailed output
  std::mt19937 rng(seed);
  
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
  std::cout << "Running " << num_simulations << " simulations..." << std::endl;
  std::cout << "Using seed: " << seed << std::endl;
  std::cout << "Using memory-efficient implementation" << std::endl;
  std::cout << "Debug output disabled (is_debugging = false)" << std::endl;
  std::cout << "----------------------------------------" << std::endl;
  
  for (int sim = 0; sim < num_simulations; ++sim) {
    if (sim % 10 == 0) {
      std::cout << "Starting simulation " << sim + 1 << "/" << num_simulations << std::endl;
    }
    std::unique_ptr<State> state = game->NewInitialState();
    LongNardeState* lnstate = dynamic_cast<LongNardeState*>(state.get());
    
    // We will keep a debug log of each step so we can reconstruct the path
    // if/when we see an invalid move.
    std::vector<DebugStep> debug_steps;
    debug_steps.reserve(200);  // arbitrary
    
    int max_moves = 1000;  // Limit the max number of moves to avoid infinite games
    int move_count = 0;
    bool invalid_move_found = false;
    
    while (!state->IsTerminal() && move_count < max_moves) {
      if (verbose) {
        std::cout << "\n--- Move " << move_count
                  << ", Player " << state->CurrentPlayer() << " ---\n";
        std::cout << state->ToString() << "\n";
      }
      
      DebugStep step_info;
      step_info.move_index = move_count;
      step_info.cur_player = state->CurrentPlayer();
      step_info.state_string = state->ToString();  // snapshot before move
      
      if (state->IsChanceNode()) {
        // Sample chance outcome
        std::vector<std::pair<Action, double>> outcomes = state->ChanceOutcomes();
        Action action = open_spiel::SampleAction(outcomes, rng).first;
        step_info.action_description =
            "CHANCE ROLL: " + std::to_string(action) + " => " +
            state->ActionToString(kChancePlayerId, action);
        
        // Apply
        state->ApplyAction(action);
      } else {
        // Check no hits condition (defensive check)
        CheckNoHits(*state);
        
        // Get legal actions
        std::vector<Action> legal_actions = state->LegalActions();
        if (legal_actions.empty()) {
          std::cerr << "No legal actions in non-terminal state!\n";
          std::cerr << "Game state: " << state->ToString() << "\n";
          break;
        }
        
        // Choose a random action from the legal set
        std::uniform_int_distribution<> dis(0, legal_actions.size() - 1);
        Action action = legal_actions[dis(rng)];
        
        // Extra validation to catch "invalid move" from X->Y errors
        Player current_player = state->CurrentPlayer();
        
        if (lnstate) {
          std::vector<CheckerMove> moves =
              lnstate->SpielMoveToCheckerMoves(current_player, action);
          bool action_valid = true;

          // Create a temporary state to apply moves sequentially for validation
          std::unique_ptr<State> temp_state = state->Clone();
          LongNardeState* temp_lnstate = dynamic_cast<LongNardeState*>(temp_state.get());
          // Keep track of moves applied to temp state for debugging (optional)
          std::string applied_moves_str;

          for (const auto& move : moves) {
            if (move.pos == kPassPos) {
                 // Optionally track passes if needed for complex debug:
                 // applied_moves_str += " pass(" + std::to_string(move.die) + ")";
                 continue; // Pass moves don't need validation against state
            }

            int to_pos = move.to_pos;
            std::string current_part_str = absl::StrFormat(
                " %d->%d(%d)", move.pos, to_pos, move.die);

            // Validate the move in the *current* temporary state
            // Use generate_legal_actions=false here, as we only need validity check
            if (!temp_lnstate->IsValidCheckerMove(current_player, move.pos, to_pos,
                                             move.die, /*generate_legal_actions=*/false)) {
              std::cerr << "INVALID MOVE DETECTED in action " << action << ":\n";
              std::cerr << "  Action: " << state->ActionToString(current_player, action) << "\n";
              std::cerr << "  Failed part: " << current_part_str << "\n";
              std::cerr << "  Applied parts to temp state: " << applied_moves_str << "\n";
              std::cerr << "  Original Board state (before action):\n" << state->ToString() << "\n";
              std::cerr << "  Temporary Board state (before this invalid part):\n" << temp_state->ToString() << "\n";

              invalid_move_found = true;
              invalid_moves_detected++;
              action_valid = false;
              break; // Stop checking this action
            }

            // Apply the validated part to the temporary state to check the next part
            // Use ApplyCheckerMove directly on the temporary state object
            temp_lnstate->ApplyCheckerMove(current_player, move);
            applied_moves_str += current_part_str; // Track applied moves (optional)

          } // end for loop over checker moves

          if (!action_valid) {
            // This warning should now only trigger if there's a genuine
            // deeper bug in LegalActionsInternal generation logic.
            std::cerr << "WARNING: Selected an invalid action sequence from LegalActions()!\n";
          }
        }
        
        step_info.action_description =
            "MOVE: " + std::to_string(action) + " => " +
            state->ActionToString(current_player, action);
        
        // Apply the (allegedly) legal action
        state->ApplyAction(action);
      }
      
      debug_steps.push_back(step_info);
      move_count++;
      
      // Periodically clone the state to limit memory
      if (move_count % 20 == 0 && !state->IsTerminal()) {
        std::unique_ptr<State> new_state = state->Clone();
        state = std::move(new_state);
        lnstate = dynamic_cast<LongNardeState*>(state.get());
      }
      
      if (invalid_move_found && !verbose) {
        // If we found an invalid move, let's print the entire sequence that
        // led here. We skip if `verbose` is already true (so we don't double print).
        std::cerr << "\n=========== RECONSTRUCTING STEPS for invalid move ===========\n";
        for (auto &ds : debug_steps) {
          std::cerr << "[Move index: " << ds.move_index
                    << ", Player: " << ds.cur_player << "]\n";
          std::cerr << "STATE BEFORE MOVE:\n" << ds.state_string << "\n";
          std::cerr << ds.action_description << "\n";
          std::cerr << "----------------------------------------\n";
        }
        std::cerr << "=============================================================\n";
      }
      
      if (invalid_move_found) {
        // We can break out once we've logged everything
        break;
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
  double avg_game_length = static_cast<double>(total_moves) / num_simulations;
  std::cout << "=========================================" << std::endl;
  std::cout << "SIMULATION RESULTS" << std::endl;
  std::cout << "=========================================" << std::endl;
  std::cout << "Random simulation completed: " << num_simulations << " games" << std::endl;
  std::cout << "Average game length: " << avg_game_length << " moves" << std::endl;
  std::cout << "Min/Max game length: " << min_game_length << "/" << max_game_length << " moves" << std::endl;
  std::cout << "Terminated games: " << terminated_games << "/" << num_simulations << std::endl;
  std::cout << "Invalid moves detected: " << invalid_moves_detected << std::endl;
  
  if (invalid_moves_detected > 0) {
    std::cerr << "WARNING: Detected " << invalid_moves_detected 
              << " invalid moves! Check LegalActions() vs. IsValidCheckerMove()..." << std::endl;
  } else {
    std::cout << "No invalid moves detected - all good!" << std::endl;
  }
  
  std::cout << "=========================================" << std::endl;
  std::cout << "TEST COMPLETED" << std::endl;
  std::cout << "=========================================" << std::endl;
}

void RunRandomSimTest(int num_simulations = kDefaultNumSimulations,
                      int seed = kDefaultSeed) {
  std::cout << "Running Long Narde random simulation test..." << std::endl;
  MemoryEfficientRandomSim(num_simulations, seed);
}

// Parses command-line arguments
void ParseArguments(int argc, char** argv, int* num_simulations, int* seed) {
  // Default values already set in the parameters
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    
    if (arg == "--num_simulations" || arg == "-n") {
      if (i + 1 < argc) {
        *num_simulations = std::stoi(argv[++i]);
      } else {
        std::cerr << "Missing value for " << arg << std::endl;
      }
    } else if (arg == "--seed" || arg == "-s") {
      if (i + 1 < argc) {
        *seed = std::stoi(argv[++i]);
      } else {
        std::cerr << "Missing value for " << arg << std::endl;
      }
    } else if (arg == "--help" || arg == "-h") {
      std::cout << "Usage: " << argv[0] << " [options]" << std::endl;
      std::cout << "Options:" << std::endl;
      std::cout << "  --num_simulations, -n <value>  Number of games to simulate (default: " 
                << kDefaultNumSimulations << ")" << std::endl;
      std::cout << "  --seed, -s <value>             Random seed (default: " 
                << kDefaultSeed << ")" << std::endl;
      std::cout << "  --help, -h                     Show this help message" << std::endl;
      exit(0);
    }
  }
}

}  // namespace

void RunRandomSimTests(int argc, char** argv) {
  int num_simulations = kDefaultNumSimulations;
  int seed = kDefaultSeed;
  
  ParseArguments(argc, argv, &num_simulations, &seed);
  RunRandomSimTest(num_simulations, seed);
}

}  // namespace long_narde
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::long_narde::RunRandomSimTests(argc, argv);
  return 0;
} 