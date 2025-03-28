#!/bin/bash
set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Building the entire OpenSpiel project in Release mode with -O3 optimization${NC}"

# Get the absolute path to the script directory (project root)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Create build directory if it doesn't exist
if [ ! -d "build" ]; then
  echo -e "${YELLOW}Creating build directory...${NC}"
  mkdir -p build
  cd build
  # Configure with CMake
  echo -e "${YELLOW}Configuring with CMake...${NC}"
  cmake -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-O3" ../open_spiel
else
  cd build
fi

# Build the entire project
echo -e "${YELLOW}Building the entire project...${NC}"
make -j$(nproc)

# Go back to project root
cd "$SCRIPT_DIR"

echo -e "${GREEN}Build process complete.${NC}"

# --- Long Narde specific tests ---
# Note: This section runs only Long Narde tests even though the entire project is built
echo -e "${YELLOW}Running Long Narde specific tests...${NC}"

# Create a log file for failed tests
FAILED_TESTS_LOG="long_narde_failed_tests.log"
echo "Long Narde Failed Tests Log - $(date)" > $FAILED_TESTS_LOG
echo "====================================" >> $FAILED_TESTS_LOG

# Execute the tests (only Long Narde) and log any failures
echo -e "${YELLOW}Running Long Narde tests and logging failures...${NC}"
"$SCRIPT_DIR/build/games/long_narde_test" --gtest_filter=*LongNarde* 2>&1 | tee long_narde_test_output.log

# Count failures in the log file
failures=$(grep -c "FAILED" long_narde_test_output.log)
if [ $failures -gt 0 ]; then
  echo -e "${RED}$failures tests failed in Long Narde tests.${NC}" | tee -a $FAILED_TESTS_LOG
  grep "FAILED" long_narde_test_output.log | tee -a $FAILED_TESTS_LOG
else
  echo -e "${GREEN}All Long Narde tests passed.${NC}" | tee -a $FAILED_TESTS_LOG
fi

# Run the random simulation test with time and memory limits
if [ -f "$SCRIPT_DIR/build/games/random_sim_test" ]; then
  echo -e "${YELLOW}Running Random Simulation Test with time limit of 20 seconds and memory limit of 1GB...${NC}"
  # Set memory limit to 1GB and redirect output to log file
  (ulimit -v 1048576; timeout 20s "$SCRIPT_DIR/build/games/random_sim_test") > long_narde_random_sim_output.log 2>&1
  
  # Check the exit code to determine if it completed, timed out, or failed
  exit_code=$?
  if [ $exit_code -eq 0 ]; then
    echo -e "${GREEN}Random Simulation Test completed successfully.${NC}" | tee -a $FAILED_TESTS_LOG
    # Show summary of results (last few lines)
    echo "Summary of results:" | tee -a $FAILED_TESTS_LOG
    tail -n 10 long_narde_random_sim_output.log | tee -a $FAILED_TESTS_LOG
  elif [ $exit_code -eq 124 ]; then
    echo -e "${YELLOW}Random Simulation Test timed out after 20 seconds.${NC}" | tee -a $FAILED_TESTS_LOG
    # Show the last few lines to see progress
    echo "Last output before timeout:" | tee -a $FAILED_TESTS_LOG
    tail -n 10 long_narde_random_sim_output.log | tee -a $FAILED_TESTS_LOG
  else
    echo -e "${RED}Random Simulation Test failed with exit code $exit_code.${NC}" | tee -a $FAILED_TESTS_LOG
    echo "Check log for details: long_narde_random_sim_output.log" | tee -a $FAILED_TESTS_LOG
  fi
else
  echo -e "${RED}Failed to build random_sim_test executable.${NC}" | tee -a $FAILED_TESTS_LOG
fi

echo -e "${GREEN}Build and test process complete.${NC}"

# Display summary
if [ -s "$FAILED_TESTS_LOG" ]; then
  echo -e "${RED}Some tests failed. See $FAILED_TESTS_LOG for details.${NC}"
else
  echo -e "${GREEN}All tests passed!${NC}"
fi

echo -e "${GREEN}Build and test logging completed!${NC}" 