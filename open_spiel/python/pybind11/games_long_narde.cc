// Copyright 2024 DeepMind Technologies Limited
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

#include "open_spiel/python/pybind11/games_long_narde.h"

#include <functional>

#include "open_spiel/games/long_narde/long_narde.h"
#include "open_spiel/python/pybind11/pybind11.h"
#include "open_spiel/spiel.h"

namespace py = ::pybind11;
using open_spiel::Game;
using open_spiel::State;
using open_spiel::long_narde::LongNardeState;
using open_spiel::long_narde::LongNardeCheckerMove;

namespace open_spiel {
namespace long_narde {

void init_pyspiel_games_long_narde(py::module& m) {
  py::class_<LongNardeCheckerMove>(m, "LongNardeCheckerMove")
      .def(py::init<>())
      .def(py::init<int, int, int>())
      .def_readwrite("pos", &LongNardeCheckerMove::pos)
      .def_readwrite("to_pos", &LongNardeCheckerMove::to_pos)
      .def_readwrite("die", &LongNardeCheckerMove::die);

  py::classh<LongNardeState, State> state_class(m, "LongNardeState");
  state_class.def("board", &LongNardeState::BoardToString)
      .def("opponent", &LongNardeState::Opponent)
      .def("is_off", &LongNardeState::IsOff)
      .def("get_to_pos", &LongNardeState::GetToPos)
      .def("count_total_checkers", &LongNardeState::CountTotalCheckers)
      .def("player_turns", (int (LongNardeState::*)() const) & LongNardeState::player_turns)
      .def("score", &LongNardeState::score)
      .def("dice", &LongNardeState::dice)
      .def("double_turn", &LongNardeState::double_turn)
      .def("moved_from_head", &LongNardeState::moved_from_head)
      .def("spiel_move_to_checker_moves", &LongNardeState::LongNardeSpielMoveToCheckerMoves)
      .def("checker_moves_to_spiel_move", &LongNardeState::LongNardeCheckerMovesToSpielMove)
      .def("board_to_string", &LongNardeState::BoardToString)
      .def("is_head_pos", &LongNardeState::IsHeadPos)
      .def("is_first_turn", &LongNardeState::IsFirstTurn)
      .def("is_legal_head_move", &LongNardeState::IsLegalHeadMove)
      .def("would_form_blocking_bridge", &LongNardeState::WouldFormBlockingBridge)
      .def("furthest_checker_in_home", &LongNardeState::FurthestCheckerInHome)
      .def("all_in_home", &LongNardeState::AllInHome)
      .def("dice_to_string", &LongNardeState::DiceToString)
      .def(py::pickle(
          std::function<std::string(const LongNardeState&)>(
            [](const LongNardeState& state) {  // __getstate__
              return SerializeGameAndState(*state.GetGame(), state);
            }
          ),
          std::function<LongNardeState*(const std::string&)>(
            [](const std::string& data) -> LongNardeState* {  // __setstate__
              std::pair<std::shared_ptr<const Game>, std::unique_ptr<State>>
                  game_and_state = DeserializeGameAndState(data);
              return dynamic_cast<LongNardeState*>(game_and_state.second.release());
            }
          )
      ));
}

}  // namespace long_narde
}  // namespace open_spiel 