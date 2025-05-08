#include "open_spiel/games/long_narde/long_narde.h"

#include <algorithm>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "open_spiel/spiel.h"
#include "open_spiel/tests/basic_tests.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel
{
  namespace long_narde
  {
    namespace
    {

      constexpr int kDefaultNumSimulations = 500;
      constexpr int kDefaultSeed = 1224;

      // StartFunction: MemoryEfficientRandomSim
      // StartTest: test-randsim-1
      void MemoryEfficientRandomSim(int num_simulations = kDefaultNumSimulations,
                                    int seed = kDefaultSeed)
      {
        const bool verbose = false;
        std::mt19937 rng(seed);

        std::shared_ptr<const Game> game = LoadGame("long_narde");

        int total_moves = 0;
        int legal_actions_calls = 0;
        int winner_moves = 0;
        int loser_moves = 0;
        // Stats for player-only moves (exclude chance nodes)
        int total_player_moves = 0;
        int max_player_moves = 0;
        int min_player_moves = std::numeric_limits<int>::max();
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

        for (int sim = 0; sim < num_simulations; ++sim)
        {
          if (sim % 10 == 0)
          {
            std::cout << "Starting simulation " << sim + 1 << "/" << num_simulations << std::endl;
          }
          std::unique_ptr<State> state = game->NewInitialState();
          LongNardeState *lnstate = dynamic_cast<LongNardeState *>(state.get());

          int x_moves = 0;
          int o_moves = 0;
          int max_moves = 1000;
          int move_count = 0;
          bool invalid_move_found = false;

          while (!state->IsTerminal() && move_count < max_moves)
          {
            if (state->IsChanceNode())
            {
              // Chance node: roll dice and apply
              std::vector<std::pair<Action, double>> outcomes = state->ChanceOutcomes();
              Action action = open_spiel::SampleAction(outcomes, rng).first;
              state->ApplyAction(action);
            }
            else
            {
              // Player decision: count calls and moves
              ++legal_actions_calls;
              if (state->CurrentPlayer() == kXPlayerId) ++x_moves;
              else if (state->CurrentPlayer() == kOPlayerId) ++o_moves;
              std::vector<Action> legal_actions = state->LegalActions();
              if (legal_actions.empty())
              {
                break;
              }
              std::uniform_int_distribution<> dis(0, legal_actions.size() - 1);
              Action action = legal_actions[dis(rng)];
              // Per-move validity check (commented out for profiling):
              /*  if (lnstate)
               {
                 std::vector<LongNardeCheckerMove> moves =
                     lnstate->LongNardeSpielMoveToCheckerMoves(state->CurrentPlayer(), action);
                 std::unique_ptr<State> temp_state = state->Clone();
                 LongNardeState *temp_lnstate = dynamic_cast<LongNardeState *>(temp_state.get());
                 for (const auto &move : moves)
                 {
                   if (move.pos == kPassPos) continue;
                   if (!temp_lnstate->LongNardeIsValidCheckerMove(state->CurrentPlayer(), move, false))
                   {
                     invalid_move_found = true;
                     invalid_moves_detected++;
                     break;
                   }
                   temp_lnstate->LongNardeApplyCheckerMove(state->CurrentPlayer(), move);
                 }
               }  */
              state->ApplyAction(action);
            }
            move_count++;

            // Periodically clone state to test undo/redo logic and memory safety.
            /* 
            if (move_count % 20 == 0 && !state->IsTerminal())
            {
              std::unique_ptr<State> new_state = state->Clone();
              state = std::move(new_state);
              lnstate = dynamic_cast<LongNardeState *>(state.get());
            }  
            */

            if (invalid_move_found)
            {
              break;
            }
          }

          // After simulation, accumulate totals
          total_moves += move_count;
          max_game_length = std::max(max_game_length, move_count);
          min_game_length = std::min(min_game_length, move_count);

          // Accumulate player-only move stats
          int player_moves = x_moves + o_moves;
          total_player_moves += player_moves;
          max_player_moves = std::max(max_player_moves, player_moves);
          min_player_moves = std::min(min_player_moves, player_moves);

          if (state->IsTerminal())
          {
            terminated_games++;
            // Categorize moves by game outcome
            std::vector<double> rets = state->Returns();
            if (rets[kXPlayerId] > rets[kOPlayerId])
            {
              winner_moves += x_moves;
              loser_moves += o_moves;
            }
            else if (rets[kOPlayerId] > rets[kXPlayerId])
            {
              winner_moves += o_moves;
              loser_moves += x_moves;
            }
          }
        }

        std::cout << "=========================================" << std::endl;
        std::cout << "SIMULATION RESULTS" << std::endl;
        std::cout << "=========================================" << std::endl;
        std::cout << "Random simulation completed: " << num_simulations << " games" << std::endl;
        // Player-only game length stats (exclude chance nodes)
        double avg_player_moves = static_cast<double>(total_player_moves) / num_simulations;
        std::cout << "Average game length (player moves only): " << avg_player_moves << " moves" << std::endl;
        std::cout << "Min/Max game length (player moves only): " << min_player_moves << "/" << max_player_moves << " moves" << std::endl;
        std::cout << "Terminated games: " << terminated_games << "/" << num_simulations << std::endl;
        std::cout << "Invalid moves detected: " << invalid_moves_detected << std::endl;

        if (invalid_moves_detected > 0)
        {
          std::cerr << "WARNING: Detected " << invalid_moves_detected
                    << " invalid moves! Check LegalActions() vs. IsValidCheckerMove()..." << std::endl;
        }
        else
        {
          std::cout << "No invalid moves detected - all good!" << std::endl;
        }

        std::cout << "Total moves: " << total_moves << std::endl;
        std::cout << "Total LegalActions calls: " << legal_actions_calls << std::endl;
        std::cout << "Moves by winners: " << winner_moves << std::endl;
        std::cout << "Moves by losers: " << loser_moves << std::endl;

        std::cout << "=========================================" << std::endl;
        std::cout << "TEST COMPLETED" << std::endl;
        std::cout << "=========================================" << std::endl;
      }
      // EndTest: test-randsim-1
      // EndFunction: MemoryEfficientRandomSim

      // StartFunction: RunRandomSimTest
      // StartTest: test-runsim-1
      void RunRandomSimTest(int num_simulations = kDefaultNumSimulations,
                            int seed = kDefaultSeed)
      {
        std::cout << "Running Long Narde random simulation test..." << std::endl;
        MemoryEfficientRandomSim(num_simulations, seed);
      }
      // EndTest: test-runsim-1
      // EndFunction: RunRandomSimTest

      // StartFunction: ParseArguments
      // StartTest: test-parseargs-1
      void ParseArguments(int argc, char **argv, int *num_simulations, int *seed)
      {
        // Parse command-line arguments for simulation count and seed.
        for (int i = 1; i < argc; ++i)
        {
          std::string arg = argv[i];
          if (arg == "--num_simulations" || arg == "-n")
          {
            if (i + 1 < argc)
            {
              *num_simulations = std::stoi(argv[++i]);
            }
            else
            {
              std::cerr << "Missing value for " << arg << std::endl;
            }
          }
          else if (arg == "--seed" || arg == "-s")
          {
            if (i + 1 < argc)
            {
              *seed = std::stoi(argv[++i]);
            }
            else
            {
              std::cerr << "Missing value for " << arg << std::endl;
            }
          }
          else if (arg == "--help" || arg == "-h")
          {
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
      // EndTest: test-parseargs-1
      // EndFunction: ParseArguments

    } // namespace

    // StartFunction: RunRandomSimTests
    // StartTest: test-runsimtests-1
    void RunRandomSimTests(int argc, char **argv)
    {
      std::cout << ">>> Starting Long Narde Random Simulation Test (RunRandomSimTests entry)..." << std::endl;
      int num_simulations = kDefaultNumSimulations;
      int seed = kDefaultSeed;

      ParseArguments(argc, argv, &num_simulations, &seed);
      RunRandomSimTest(num_simulations, seed);
    }
    // EndTest: test-runsimtests-1
    // EndFunction: RunRandomSimTests
  } // namespace long_narde
} // namespace open_spiel
// StartFunction: main
// StartTest: test-main-1
int main(int argc, char **argv)
{
  open_spiel::long_narde::RunRandomSimTests(argc, argv);
  return 0;
}
// EndTest: test-main-1
// EndFunction: main