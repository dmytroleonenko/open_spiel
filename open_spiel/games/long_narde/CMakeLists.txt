add_executable(long_narde_test_minimal
  long_narde_test_minimal.cc
)

target_include_directories(long_narde_test_minimal PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../)
target_link_libraries(long_narde_test_minimal
  games
  tests
)

add_executable(random_sim_test
  random_sim_test.cc
)

target_include_directories(random_sim_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../)
target_link_libraries(random_sim_test
  games
  tests
) 