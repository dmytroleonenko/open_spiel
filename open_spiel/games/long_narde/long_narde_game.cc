#include "open_spiel/games/long_narde/long_narde.h"

#include <memory>
#include <string>

#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/game_parameters.h"

namespace open_spiel {
namespace long_narde {

namespace { // Anonymous namespace

// Moved from long_narde.cc
const GameType kGameType{
    /*short_name=*/"long_narde",
    /*long_name=*/"Long Narde",
    GameType::Dynamics::kSequential,
    GameType::ChanceMode::kExplicitStochastic,
    GameType::Information::kPerfectInformation,
    GameType::Utility::kZeroSum,
    GameType::RewardModel::kTerminal,
    /*min_num_players=*/2,
    /*max_num_players=*/2,
    /*provides_information_state_string=*/false,
    /*provides_information_state_tensor=*/false,
    /*provides_observation_string=*/true,
    /*provides_observation_tensor=*/true,
    /*parameter_specification=*/{
        {"scoring_type", GameParameter(std::string(kDefaultScoringType))}
    }};

// Moved from long_narde.cc
static std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(new LongNardeGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);

RegisterSingleTensorObserver single_tensor(kGameType.short_name);

} // End anonymous namespace

// ===== Game Class Methods =====

// Moved from long_narde.cc
LongNardeGame::LongNardeGame(const GameParameters& params)
    : Game(kGameType, params),
      scoring_type_(ParseScoringType(
          params.count("scoring_type") > 0 ?
          params.at("scoring_type").string_value() :
          kDefaultScoringType)) {}

// Moved from long_narde.cc
double LongNardeGame::MaxUtility() const {
  return 2.0; // Max score is 2 for mars/gammon
}

// Note: NewInitialState is defined in the header long_narde.h 
// because it's often needed by derived classes or other parts of the framework.
// std::unique_ptr<State> LongNardeGame::NewInitialState() const {
//   return std::unique_ptr<State>(new LongNardeState(shared_from_this(), scoring_type_));
// }

} // namespace long_narde
} // namespace open_spiel