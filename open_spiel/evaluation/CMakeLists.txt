add_library (evaluation OBJECT
  soft_condorcet_optimization.cc
  soft_condorcet_optimization.h
)
target_include_directories (evaluation PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(soft_condorcet_optimization_test soft_condorcet_optimization_test.cc
        $<TARGET_OBJECTS:evaluation> ${OPEN_SPIEL_OBJECTS})
add_test(soft_condorcet_optimization_test soft_condorcet_optimization_test)
