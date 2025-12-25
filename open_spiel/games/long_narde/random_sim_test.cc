// Copyright 2025 DeepMind Technologies Limited
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

#include <algorithm>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "open_spiel/games/long_narde/long_narde.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace long_narde {
namespace {

constexpr int kDefaultNumSimulations = 500;
constexpr int kDefaultSeed = 1224;

void MemoryEfficientRandomSim(int num_simulations = kDefaultNumSimulations,
                              int seed = kDefaultSeed) {
  const bool verbose = false;
  std::mt19937 rng(seed);

  std::shared_ptr<const Game> game = LoadGame("long_narde");

  int total_moves = 0;
  int legal_actions_calls = 0;
  int winner_moves = 0;
  int loser_moves = 0;
  int total_player_moves = 0;
  int max_player_moves = 0;
  int min_player_moves = std::numeric_limits<int>::max();
  int max_game_length = 0;
  int min_game_length = std::numeric_limits<int>::max();
  int terminated_games = 0;

  std::cout << "=========================================" << std::endl;
  std::cout << "LONG NARDE RANDOM SIMULATION TEST" << std::endl;
  std::cout << "=========================================" << std::endl;
  std::cout << "Running " << num_simulations << " simulations..." << std::endl;
  std::cout << "Using seed: " << seed << std::endl;
  std::cout << "Using memory-efficient implementation" << std::endl;
  std::cout << "----------------------------------------" << std::endl;

  for (int sim = 0; sim < num_simulations; ++sim) {
    if (sim % 10 == 0) {
      std::cout << "Starting simulation " << sim + 1 << "/"
                << num_simulations << std::endl;
    }
    std::unique_ptr<State> state = game->NewInitialState();

    int x_moves = 0;
    int o_moves = 0;
    int max_moves = 1000;
    int move_count = 0;

    while (!state->IsTerminal() && move_count < max_moves) {
      if (state->IsChanceNode()) {
        std::vector<std::pair<Action, double>> outcomes =
            state->ChanceOutcomes();
        Action action = open_spiel::SampleAction(outcomes, rng).first;
        state->ApplyAction(action);
      } else {
        ++legal_actions_calls;
        if (state->CurrentPlayer() == kXPlayerId) {
          ++x_moves;
        } else if (state->CurrentPlayer() == kOPlayerId) {
          ++o_moves;
        }
        std::vector<Action> legal_actions = state->LegalActions();
        if (legal_actions.empty()) {
          break;
        }
        std::uniform_int_distribution<> dis(0, legal_actions.size() - 1);
        Action action = legal_actions[dis(rng)];
        state->ApplyAction(action);
      }
      ++move_count;
    }

    total_moves += move_count;
    max_game_length = std::max(max_game_length, move_count);
    min_game_length = std::min(min_game_length, move_count);

    int player_moves = x_moves + o_moves;
    total_player_moves += player_moves;
    max_player_moves = std::max(max_player_moves, player_moves);
    min_player_moves = std::min(min_player_moves, player_moves);

    if (state->IsTerminal()) {
      ++terminated_games;
      std::vector<double> rets = state->Returns();
      if (rets[kXPlayerId] > rets[kOPlayerId]) {
        winner_moves += x_moves;
        loser_moves += o_moves;
      } else if (rets[kOPlayerId] > rets[kXPlayerId]) {
        winner_moves += o_moves;
        loser_moves += x_moves;
      }
    }

    if (verbose) {
      std::cout << state->ToString() << std::endl;
    }
  }

  std::cout << "=========================================" << std::endl;
  std::cout << "SIMULATION RESULTS" << std::endl;
  std::cout << "=========================================" << std::endl;
  std::cout << "Random simulation completed: " << num_simulations << " games"
            << std::endl;
  double avg_player_moves =
      static_cast<double>(total_player_moves) / num_simulations;
  std::cout << "Average game length (player moves only): " << avg_player_moves
            << " moves" << std::endl;
  std::cout << "Min/Max game length (player moves only): " << min_player_moves
            << "/" << max_player_moves << " moves" << std::endl;
  std::cout << "Terminated games: " << terminated_games << "/"
            << num_simulations << std::endl;
  std::cout << "Total moves: " << total_moves << std::endl;
  std::cout << "Total LegalActions calls: " << legal_actions_calls << std::endl;
  std::cout << "Moves by winners: " << winner_moves << std::endl;
  std::cout << "Moves by losers: " << loser_moves << std::endl;

  std::cout << "=========================================" << std::endl;
  std::cout << "TEST COMPLETED" << std::endl;
  std::cout << "=========================================" << std::endl;
}

void RunRandomSimTest(int num_simulations = kDefaultNumSimulations,
                      int seed = kDefaultSeed) {
  std::cout << "Running Long Narde random simulation test..." << std::endl;
  MemoryEfficientRandomSim(num_simulations, seed);
}

void ParseArguments(int argc, char** argv, int* num_simulations, int* seed) {
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
      std::cout << "  --num_simulations, -n <value>  Number of games to "
                   "simulate (default: "
                << kDefaultNumSimulations << ")" << std::endl;
      std::cout << "  --seed, -s <value>             Random seed (default: "
                << kDefaultSeed << ")" << std::endl;
      std::cout << "  --help, -h                     Show this help message"
                << std::endl;
      exit(0);
    }
  }
}

void RunRandomSimTests(int argc, char** argv) {
  std::cout << ">>> Starting Long Narde Random Simulation Test "
               "(RunRandomSimTests entry)..."
            << std::endl;
  int num_simulations = kDefaultNumSimulations;
  int seed = kDefaultSeed;

  ParseArguments(argc, argv, &num_simulations, &seed);
  RunRandomSimTest(num_simulations, seed);
}

}  // namespace
}  // namespace long_narde
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::long_narde::RunRandomSimTests(argc, argv);
  return 0;
}
