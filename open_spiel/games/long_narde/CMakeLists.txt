add_executable(long_narde_test
  long_narde_test_main.cc
  long_narde_test_basic.cc
  long_narde_test_movement.cc
  long_narde_test_bridges.cc
  long_narde_test_actions.cc
  long_narde_test_endgame.cc
  long_narde_test_legacy.cc
)

target_include_directories(long_narde_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../)
target_link_libraries(long_narde_test
  games
  tests
)
set_target_properties(long_narde_test PROPERTIES COMPILE_FLAGS ${DEBUG_FLAGS_LONG_NARDE})

add_executable(long_narde_test_minimal
  long_narde_test_minimal.cc
)

target_include_directories(long_narde_test_minimal PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../)
target_link_libraries(long_narde_test_minimal
  games
  tests
)
set_target_properties(long_narde_test_minimal PROPERTIES COMPILE_FLAGS ${DEBUG_FLAGS_LONG_NARDE})

add_executable(random_sim_test
  random_sim_test.cc
)

target_include_directories(random_sim_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../)
target_link_libraries(random_sim_test
  games
  tests
)
set_target_properties(random_sim_test PROPERTIES COMPILE_FLAGS ${DEBUG_FLAGS_LONG_NARDE}) 