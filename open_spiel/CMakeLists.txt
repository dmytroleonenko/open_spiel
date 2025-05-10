# Version >= 3.12 required for new FindPython module
# https://cmake.org/cmake/help/v3.12/release/3.12.html
# Version >= 3.17 required for CMAKE_CUDA_STANDARD
# https://gitlab.kitware.com/cmake/cmake/-/issues/19123
cmake_minimum_required (VERSION 3.17)
project (open_spiel)

# Define some nice terminal colors.
if(NOT WIN32)
  string(ASCII 27 Esc)
  set(ColourReset "${Esc}[m")
  set(ColourBold  "${Esc}[1m")
  set(Red         "${Esc}[31m")
  set(Green       "${Esc}[32m")
  set(Yellow      "${Esc}[33m")
  set(Blue        "${Esc}[34m")
  set(Magenta     "${Esc}[35m")
  set(Cyan        "${Esc}[36m")
  set(White       "${Esc}[37m")
  set(BoldRed     "${Esc}[1;31m")
  set(BoldGreen   "${Esc}[1;32m")
  set(BoldYellow  "${Esc}[1;33m")
  set(BoldBlue    "${Esc}[1;34m")
  set(BoldMagenta "${Esc}[1;35m")
  set(BoldCyan    "${Esc}[1;36m")
  set(BoldWhite   "${Esc}[1;37m")
endif()

set(CMAKE_CXX_STANDARD 17)

# Conditionally set CUDA variables
if(NOT DEFINED USE_CUDA OR NOT USE_CUDA STREQUAL "OFF")
  # Try to find CUDA toolkit and nvcc
  find_package(CUDAToolkit QUIET)
  if(CUDAToolkit_FOUND)
    set(CMAKE_CUDA_STANDARD 14)
    # Only set native if a GPU is detected, otherwise let it be empty or OFF
    # This part might need a more robust way to check for actual GPU presence for 'native'
    set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "CUDA architectures, e.g., 70;75;80 or native or OFF")
    if(NOT CMAKE_CUDA_COMPILER)
      set(CMAKE_CUDA_COMPILER ${CUDAToolkit_NVCC_EXECUTABLE})
    endif()
    set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)
    message(STATUS "CUDA Toolkit found. Version: ${CUDAToolkit_VERSION}. Compiler: ${CMAKE_CUDA_COMPILER}")
  else()
    message(STATUS "CUDA Toolkit not found. Forcing USE_CUDA to OFF.")
    set(USE_CUDA OFF CACHE BOOL "Disable CUDA due to missing toolkit" FORCE)
    set(CMAKE_CUDA_ARCHITECTURES OFF CACHE STRING "CUDA architectures disabled" FORCE)
  endif()
else()
  message(STATUS "USE_CUDA is explicitly set to OFF. Skipping CUDA configuration and disabling architectures.")
  set(CMAKE_CUDA_ARCHITECTURES OFF CACHE STRING "CUDA architectures explicitly disabled" FORCE)
endif()

set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

# Enable building shared libraries by default
option(BUILD_SHARED_LIBS "Build shared libraries" ON)
message("${BoldYellow}Building shared libraries: ${BUILD_SHARED_LIBS}${ColourReset}")

# Ensure that libraries are built as shared by default
if(BUILD_SHARED_LIBS)
  set(CMAKE_FIND_LIBRARY_SUFFIXES ".so" ".dylib" ".dll" ${CMAKE_FIND_LIBRARY_SUFFIXES})
  # On Linux, make sure the linker adds RPATH so runtime library search works
  if(UNIX AND NOT APPLE)
    set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
    set(CMAKE_SKIP_BUILD_RPATH FALSE)
    set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
    set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
  endif()
endif()

# Shim for CUDA::nvToolsExt -> CUDA::nvtx3 alias (requires CMake >=3.25)
if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.25")
  find_package(CUDAToolkit REQUIRED)
  if(NOT TARGET CUDA::nvToolsExt)
    add_library(CUDA::nvToolsExt INTERFACE IMPORTED)
    target_link_libraries(CUDA::nvToolsExt INTERFACE CUDA::nvtx3)
  endif()
endif()

# Set default build type.
set (BUILD_TYPE $ENV{BUILD_TYPE})
if(NOT BUILD_TYPE)
   set(BUILD_TYPE Testing
       CACHE STRING "Choose the type of build: Debug Release Testing."
       FORCE)
endif()
message("${BoldYellow}Current build type is: ${BUILD_TYPE}${ColourReset}")

# Global optimization flags (can be overridden for specific targets)
if(${BUILD_TYPE} STREQUAL "Debug")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -Og")
elseif(${BUILD_TYPE} STREQUAL "Testing")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")
elseif(${BUILD_TYPE} STREQUAL "Release")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNDEBUG -O3")
endif()

# Specific debug symbols for long_narde code components
set(DEBUG_FLAGS_LONG_NARDE "-O3 -s")
message("${BoldYellow}Debug flags for long_narde: ${DEBUG_FLAGS_LONG_NARDE}${ColourReset}")

if(APPLE)
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-everything -w -undefined dynamic_lookup")
  set (CMAKE_FIND_FRAMEWORK LAST)
elseif(WIN32)
  # No changes for MSVC
else()
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-everything")
endif()

# Position-independent code is needed for Python extension modules.
set (CMAKE_POSITION_INDEPENDENT_CODE ON)

## Optional dependencies
macro(openspiel_optional_dependency DEP_NAME DEP_DEFAULT DEP_DESCRIPTION)
  set (${DEP_NAME} ${DEP_DEFAULT} CACHE BOOL ${DEP_DESCRIPTION})
  if(NOT DEFINED ENV{${DEP_NAME}})
    message("${BoldRed}${DEP_NAME} not set. Defaults to ${DEP_DEFAULT}${ColourReset}")
    set (ENV{${DEP_NAME}} ${DEP_DEFAULT})
  endif()
  set (${DEP_NAME} $ENV{${DEP_NAME}})
  message("${BoldYellow}${DEP_NAME}: ${${DEP_NAME}} ${ColourReset}")
  if (${DEP_NAME})
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D${DEP_NAME}")
  endif()
endmacro()

# List of all optional dependencies:
openspiel_optional_dependency(OPEN_SPIEL_BUILD_WITH_ACPC           OFF "Build against the Universal Poker library.")
openspiel_optional_dependency(OPEN_SPIEL_BUILD_WITH_HANABI         OFF "Build against the Hanabi game.")
openspiel_optional_dependency(OPEN_SPIEL_BUILD_WITH_JULIA          OFF "Build binary for Julia.")
openspiel_optional_dependency(OPEN_SPIEL_BUILD_WITH_LIBNOP         OFF "Build with support for libnop.")
openspiel_optional_dependency(OPEN_SPIEL_BUILD_WITH_LIBTORCH       OFF "Build with support for libtorch.")
openspiel_optional_dependency(OPEN_SPIEL_BUILD_WITH_PYTHON         ON  "Build binary for Python.")
openspiel_optional_dependency(OPEN_SPIEL_BUILD_WITH_XINXIN         OFF "Build against xinxin Hearts program.")
openspiel_optional_dependency(OPEN_SPIEL_BUILD_WITH_ROSHAMBO       OFF "Build against RoShamBo bots.")
openspiel_optional_dependency(OPEN_SPIEL_BUILD_WITH_GAMUT          OFF "Build with GAMUT generator integration.")
openspiel_optional_dependency(OPEN_SPIEL_BUILD_WITH_ORTOOLS        OFF "Build with C++ optimization library OR-Tools.")

if (WIN32)
  openspiel_optional_dependency(OPEN_SPIEL_ENABLE_JAX              OFF "Enable JAX.")
  openspiel_optional_dependency(OPEN_SPIEL_ENABLE_PYTORCH          OFF "Enable PyTorch.")
  openspiel_optional_dependency(OPEN_SPIEL_ENABLE_TENSORFLOW       OFF "Enable Tensorflow.")
else()
  openspiel_optional_dependency(OPEN_SPIEL_ENABLE_JAX             AUTO "Enable JAX.")
  openspiel_optional_dependency(OPEN_SPIEL_ENABLE_PYTORCH         AUTO "Enable PyTorch.")
  openspiel_optional_dependency(OPEN_SPIEL_ENABLE_TENSORFLOW      AUTO "Enable Tensorflow.")
endif()

openspiel_optional_dependency(OPEN_SPIEL_ENABLE_PYTHON_MISC       OFF "Enable miscellaneous Python dependencies.")
openspiel_optional_dependency(OPEN_SPIEL_BUILDING_WHEEL           OFF "Building a Python wheel?")

# Disable Abseil tests.
set (BUILD_TESTING OFF)

enable_testing()

set (OPEN_SPIEL_CORE_FILES
  action_view.h
  action_view.cc
  canonical_game_strings.cc
  canonical_game_strings.h
  game_parameters.cc
  game_parameters.h
  matrix_game.cc
  matrix_game.h
  normal_form_game.h
  observer.cc
  observer.h
  policy.cc
  policy.h
  simultaneous_move_game.cc
  simultaneous_move_game.h
  spiel.cc
  spiel.h
  spiel_bots.cc
  spiel_bots.h
  spiel_globals.h
  spiel_utils.cc
  spiel_utils.h
  tensor_game.cc
  tensor_game.h
  utils/usage_logging.h
  utils/usage_logging.cc
)

set(ABSL_PROPAGATE_CXX_STD ON)
add_subdirectory (abseil-cpp)
include_directories (abseil-cpp)

add_library(open_spiel_core OBJECT ${OPEN_SPIEL_CORE_FILES})
target_include_directories (open_spiel_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} abseil-cpp)
link_libraries(open_spiel_core
  absl::algorithm
  absl::flags
  absl::flags_parse
  absl::flat_hash_map
  absl::optional
  absl::random_random
  absl::str_format
  absl::strings
  absl::time
)

set (OPEN_SPIEL_CORE_OBJECTS $<TARGET_OBJECTS:open_spiel_core>)

set (OPEN_SPIEL_OBJECTS
  $<TARGET_OBJECTS:open_spiel_core>
  $<TARGET_OBJECTS:bots>
  $<TARGET_OBJECTS:games>
  $<TARGET_OBJECTS:game_transforms>
  $<TARGET_OBJECTS:bridge_double_dummy_solver>
  $<TARGET_OBJECTS:algorithms>
  $<TARGET_OBJECTS:evaluation>
  $<TARGET_OBJECTS:utils>
  $<TARGET_OBJECTS:tests>
)

if (OPEN_SPIEL_BUILD_WITH_HANABI)
  list(APPEND OPEN_SPIEL_OBJECTS $<TARGET_OBJECTS:hanabi_learning_environment>)
endif()
if (OPEN_SPIEL_BUILD_WITH_ACPC)
  list(APPEND OPEN_SPIEL_OBJECTS
      $<TARGET_OBJECTS:universal_poker_clib>
      $<TARGET_OBJECTS:universal_poker_lib>)
endif()
if (OPEN_SPIEL_BUILD_WITH_XINXIN)
  list(APPEND OPEN_SPIEL_OBJECTS $<TARGET_OBJECTS:xinxin>)
endif()
if (OPEN_SPIEL_BUILD_WITH_ROSHAMBO)
  list(APPEND OPEN_SPIEL_OBJECTS $<TARGET_OBJECTS:roshambo>)
endif()
if (OPEN_SPIEL_BUILD_WITH_LIBNOP)
  include_directories(libnop/libnop/include)
  add_subdirectory(libnop)
endif()
if (OPEN_SPIEL_BUILD_WITH_LIBTORCH)
  list(APPEND CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/libtorch/libtorch")
  find_package(Torch REQUIRED)
  include_directories(${TORCH_INCLUDE_DIRS})
endif()
if (OPEN_SPIEL_BUILD_WITH_GAMUT)
  list(APPEND OPEN_SPIEL_OBJECTS $<TARGET_OBJECTS:gamut>)
endif()
if (OPEN_SPIEL_BUILD_WITH_ORTOOLS)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSE_BOP -DUSE_GLOP -DUSE_CBC -DUSE_CLP -DUSE_SCIP -pthread")
  set(ORTOOLS_HOME "${CMAKE_CURRENT_SOURCE_DIR}/ortools")
  set(ORTOOLS_INC_DIRS ${ORTOOLS_HOME} ${ORTOOLS_HOME}/include)
  set(ORTOOLS_LIB_DIRS ${ORTOOLS_HOME}/lib ${ORTOOLS_HOME}/lib64)
  set(ORTOOLS_LIBS z rt pthread ortools)
  include_directories(${ORTOOLS_INC_DIRS})
  link_directories(${ORTOOLS_LIB_DIRS})
endif()

include_directories(..)

add_subdirectory (algorithms)
add_subdirectory (bots)
add_subdirectory (evaluation)
add_subdirectory (examples)
add_subdirectory (games)
add_subdirectory (game_transforms)

if (OPEN_SPIEL_BUILD_WITH_PYTHON)
  add_subdirectory (python)
endif()

add_subdirectory (utils)

if (OPEN_SPIEL_BUILD_WITH_JULIA)
  add_subdirectory (julia)
endif()

set (BUILD_SHARED_LIB OFF CACHE BOOL "Build a shared library?")
if(NOT DEFINED ENV{BUILD_SHARED_LIB})
  set (ENV{BUILD_SHARED_LIB} OFF)
endif()
set (BUILD_SHARED_LIB $ENV{BUILD_SHARED_LIB})
if (BUILD_SHARED_LIB)
  if (OPEN_SPIEL_BUILD_WITH_ORTOOLS)
    add_library(open_spiel SHARED ${OPEN_SPIEL_OBJECTS} $<TARGET_OBJECTS:open_spiel_ortools>)
  else()
    add_library(open_spiel SHARED ${OPEN_SPIEL_OBJECTS})
  endif()
  target_include_directories(open_spiel PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} abseil-cpp)
  target_link_libraries(open_spiel PUBLIC
    absl::algorithm
    absl::flat_hash_map
    absl::optional
    absl::random_random
    absl::str_format
    absl::strings
    absl::time
    ${ORTOOLS_LIBS}
  )
endif()

add_subdirectory (tests)
