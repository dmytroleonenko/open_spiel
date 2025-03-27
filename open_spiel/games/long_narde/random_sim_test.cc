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

void MemoryEfficientRandomSim(int num_simulations = kDefaultNumSimulations, 
                              int seed = kDefaultSeed) {
  const bool verbose = false;
  std::mt19937 rng(seed);
  
  std::shared_ptr<const Game> game = LoadGame("long_narde");
  
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
  std::cout << "----------------------------------------" << std::endl;
  
  for (int sim = 0; sim < num_simulations; ++sim) {
    if (sim % 10 == 0) {
      std::cout << "Starting simulation " << sim + 1 << "/" << num_simulations << std::endl;
    }
    std::unique_ptr<State> state = game->NewInitialState();
    LongNardeState* lnstate = dynamic_cast<LongNardeState*>(state.get());
    
    int max_moves = 1000;
    int move_count = 0;
    bool invalid_move_found = false;
    
    while (!state->IsTerminal() && move_count < max_moves) {
      if (state->IsChanceNode()) {
        std::vector<std::pair<Action, double>> outcomes = state->ChanceOutcomes();
        Action action = open_spiel::SampleAction(outcomes, rng).first;
        state->ApplyAction(action);
      } else {
        std::vector<Action> legal_actions = state->LegalActions();
        if (legal_actions.empty()) {
          break;
        }
        
        std::uniform_int_distribution<> dis(0, legal_actions.size() - 1);
        Action action = legal_actions[dis(rng)];
        
        if (lnstate) {
          std::vector<CheckerMove> moves =
              lnstate->SpielMoveToCheckerMoves(state->CurrentPlayer(), action);
          bool action_valid = true;

          std::unique_ptr<State> temp_state = state->Clone();
          LongNardeState* temp_lnstate = dynamic_cast<LongNardeState*>(temp_state.get());

          for (const auto& move : moves) {
            if (move.pos == kPassPos) {
                 continue;
            }

            if (!temp_lnstate->IsValidCheckerMove(state->CurrentPlayer(), move.pos, move.to_pos,
                                             move.die, false)) {
              invalid_move_found = true;
              invalid_moves_detected++;
              action_valid = false;
              break;
            }

            temp_lnstate->ApplyCheckerMove(state->CurrentPlayer(), move);
          }
        }
        
        state->ApplyAction(action);
      }
      
      move_count++;
      
      if (move_count % 20 == 0 && !state->IsTerminal()) {
        std::unique_ptr<State> new_state = state->Clone();
        state = std::move(new_state);
        lnstate = dynamic_cast<LongNardeState*>(state.get());
      }
      
      if (invalid_move_found) {
        break;
      }
    }
    
    total_moves += move_count;
    max_game_length = std::max(max_game_length, move_count);
    min_game_length = std::min(min_game_length, move_count);
    
    if (state->IsTerminal()) {
      terminated_games++;
    }
    
    state.reset();
    lnstate = nullptr;
  }
  
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