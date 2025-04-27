# pybind11 Bindings for Long Narde: Conventions and Plan

## Overview
This document summarizes the conventions for implementing pybind11 bindings in OpenSpiel, using Backgammon as a reference, and outlines the plan for creating Python bindings for Long Narde.

---

## Reference: Backgammon pybind11 Bindings

- **Binding files:**
  - `open_spiel/python/pybind11/games_backgammon.cc`
  - `open_spiel/python/pybind11/games_backgammon.h`
- **Initialization function:**
  - `void open_spiel::init_pyspiel_games_backgammon(py::module& m);`
- **Exposed classes/structs:**
  - `BackgammonState` (inherits from `State`)
  - `CheckerMove`
- **Exposed methods:**
  - `augment_with_hit_info`
  - `board`
  - `checker_moves_to_spiel_move`
  - `spiel_move_to_checker_moves`
  - `translate_action`
- **Pickle support:**
  - Provided for `BackgammonState` using custom serialization/deserialization.
- **pybind11 usage:**
  - Use `py::class_` for class/struct bindings.
  - Use `.def` and `.def_readwrite` for methods and fields.
  - Inherit from `State` where appropriate.

---

## Plan for Long Narde pybind11 Bindings

### 1. Identify Classes and Methods to Expose
- **Classes/Structs:**
  - `LongNardeState` (inherits from `State`)
  - `CheckerMove` (struct)
- **Key Methods to Expose:**
  - `std::vector<CheckerMove> SpielMoveToCheckerMoves(Player player, Action spiel_move) const`
  - `Action CheckerMovesToSpielMove(const std::vector<CheckerMove>& moves) const`
  - `int board(int player, int pos) const`
  - `int GetToPos(int player, int from_pos, int pips) const`
  - `bool IsValidCheckerMove(int player, const CheckerMove& move, bool moved_from_head_this_sequence) const`
  - `bool IsHeadPos(int player, int pos) const`
  - `bool IsFirstTurn(int player) const`
  - `bool IsLegalHeadMove(int player, int from_pos, bool moved_from_head_this_sequence) const`
  - `bool WouldFormBlockingBridge(int player, int from_pos, int to_pos) const`
  - `int FurthestCheckerInHome(Player player) const`
  - `bool AllInHome(Player player) const`
  - `std::string DiceToString() const`
  - `std::string BoardToString() const`
  - Accessors: `score(int player)`, `dice(int i)`, `double_turn()`, `moved_from_head()`
  - Serialization (for pickle support)

- **Other Considerations:**
  - Expose constants such as `kNumPoints`, `kXPlayerId`, `kOPlayerId`, `kPassPos`, etc., for Python-side logic.
  - Expose the `LongNardeGame` class if needed for advanced game parameterization.

### 2. File Structure
- `open_spiel/python/pybind11/games_long_narde.cc`
- `open_spiel/python/pybind11/games_long_narde.h`

### 3. Implementation Steps
1. **Draft binding file:**
   - Implement `init_pyspiel_games_long_narde(py::module& m)`.
   - Expose `LongNardeState` and `CheckerMove` struct.
   - Bind the methods listed above.
   - Add pickle support if needed.
   - **Status: Complete**
2. **Update CMake/build scripts:**
   - Ensure new binding files are included in the build.
   - **Status: Complete**
3. **Test Python bindings:**
   - Write Python tests to verify correct exposure and functionality.
   - Tests implemented in `open_spiel/python/tests/games_long_narde_test.py`.
   - **Status: Complete**
4. **Document Python API:**
   - Provide usage examples and API documentation.
   - **Status: Complete**

### 4. Recommendations
- Follow the Backgammon binding pattern closely for consistency.
- Expose only the most useful and stable methods initially; expand as needed.
- Ensure all exposed methods are well-documented in both C++ and Python.
- Consider exposing additional helper methods if they simplify Python-side logic or testing.

---

## Key Lessons Learned

- **Holder Type Consistency:** Pybind11 requires consistent holder types (e.g., `py::classh` vs `py::class_<..., std::shared_ptr<...>>`) across the class hierarchy (base and derived classes). Mismatches can lead to runtime crashes (like SIGSEGV during method dispatch). Following the pattern used by existing, working bindings (like Backgammon using `py::classh`) is crucial.
- **Pickle Support Implementation:** The `__setstate__` lambda for pickle support needs correct handling of pointer ownership (`unique_ptr::release()`) and type casting (`dynamic_cast`) to reconstruct the derived class object properly.
- **Undefined Symbols:** C++ functions declared in headers and bound via pybind11 *must* have a corresponding implementation defined in a compiled `.cc` file. Missing implementations lead to `ImportError: undefined symbol` at runtime.
- **Build Environment:**
  - **Compiler Choice:** In some environments, explicitly setting the C++ compiler (e.g., `export CXX=g++`) might be necessary for successful linking, even if the default (like `clang++`) compiles without error.
  - **Build System:** Use the established build system for the project (here, `pip install .` was the correct method, not direct CMake invocation).
  - **Clean Builds:** Always clean previous build artifacts (`build/`, `*.egg-info`, `dist/`) before rebuilding after C++ or binding changes (`rm -rf build open_spiel.egg-info dist`).
- **Debugging Segmentation Faults:**
  - **Isolate the Issue:** Test simpler, related components (like the `backgammon` game) to determine if the issue is specific or general.
  - **C++ Unit Tests:** Run C++ unit tests first. If they pass, the issue is likely in the C++/Python boundary (bindings, object lifetime).
  - **GDB:** Use GDB attached to the Python process. Set pending breakpoints on C++ constructors and methods (`break Class::Method`). Use `bt` after a crash to get a backtrace and pinpoint the fault location.

---

## Next Steps
- [x] Identify and list all relevant Long Narde C++ classes and methods to expose.
- [x] Draft `games_long_narde.cc` and `games_long_narde.h` following the Backgammon template.
- [x] Update build scripts and test the bindings.
- [x] Document the Python API for Long Narde.
- [x] Implement Python tests in `games_long_narde_test.py`.

---

*This document should be updated as the binding implementation progresses and new features are added.* 