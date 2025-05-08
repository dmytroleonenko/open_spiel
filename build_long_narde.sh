#!/bin/bash
set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Default build type
export BUILD_TYPE="Debug"
export CXX_FLAGS="-g -O0 -pg"

# Check for release build argument
if [[ "$1" == "--release" ]]; then
  BUILD_TYPE="Release"
  CXX_FLAGS="-O3 -pg"
  echo -e "${YELLOW}Building in Release mode with -O3 optimization${NC}"
else
  echo -e "${YELLOW}Building in Debug mode with -g -O0 flags${NC}"
fi

# Get the absolute path to the script directory (project root)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BUILD_DIR="$SCRIPT_DIR/open_spiel/build" # Define the new build directory

cd "$SCRIPT_DIR"

# Create a log file for failed tests in the script directory *before* changing to build
FAILED_TESTS_LOG="long_narde_failed_tests.log"
echo "Long Narde Failed Tests Log - $(date)" > $FAILED_TESTS_LOG
echo "====================================" >> $FAILED_TESTS_LOG

# Create the new build directory if it doesn't exist
if [ ! -d "$BUILD_DIR" ]; then
  echo -e "${YELLOW}Creating build directory: $BUILD_DIR...${NC}"
  mkdir -p "$BUILD_DIR"
  cd "$BUILD_DIR"
  # Configure with CMake - Source is now .. relative to open_spiel/build
  echo -e "${YELLOW}Configuring with CMake...${NC}"
  cmake -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=$BUILD_TYPE -DCMAKE_CXX_FLAGS="$CXX_FLAGS" ..
else
  # Ensure we are in the build directory even if it exists
  cd "$BUILD_DIR"
fi

# Build only Long Narde related targets (Run from build dir)
echo -e "${YELLOW}Building Long Narde targets...${NC}"
make -j$(nproc) open_spiel_core games long_narde_test

# Build the random_sim_test executable (Run from build dir)
echo -e "${YELLOW}Building random simulation test...${NC}"
make -j$(nproc) random_sim_test

# Execute the tests (only Long Narde) and log any failures (Run from build dir)
echo -e "${YELLOW}Running Long Narde tests and logging failures...${NC}"
# Execute from build dir, save log to SCRIPT_DIR
./games/long_narde_test --gtest_filter=*LongNarde* 2>&1 | tee "$SCRIPT_DIR/$FAILED_TESTS_LOG"

# Count failures in the log file (referencing log in SCRIPT_DIR)
failures=$(grep -c "FAILED" "$SCRIPT_DIR/$FAILED_TESTS_LOG")
if [ $failures -gt 0 ]; then
  echo -e "${RED}$failures tests failed in Long Narde tests.${NC}" | tee -a "$SCRIPT_DIR/$FAILED_TESTS_LOG"
  grep "FAILED" "$SCRIPT_DIR/long_narde_test_output.log" | tee -a "$SCRIPT_DIR/$FAILED_TESTS_LOG"
else
  echo -e "${GREEN}All Long Narde tests passed.${NC}" | tee -a "$SCRIPT_DIR/$FAILED_TESTS_LOG"
fi

# Run the random simulation test with time and memory limits (Run from build dir)
# Check relative path from build dir
if [ -f "./games/random_sim_test" ]; then
  echo -e "${YELLOW}Running Random Simulation Test with time limit of 20 seconds and memory limit of 1GB...${NC}"
  # Set memory limit to 1GB and redirect output to log file in SCRIPT_DIR
  (ulimit -v 1048576; timeout 20s ./games/random_sim_test) > "$SCRIPT_DIR/long_narde_random_sim_output.log" 2>&1

  # Check the exit code to determine if it completed, timed out, or failed
  exit_code=$?
  if [ $exit_code -eq 0 ]; then
    echo -e "${GREEN}Random Simulation Test completed successfully.${NC}" | tee -a "$SCRIPT_DIR/$FAILED_TESTS_LOG"
    # Show summary of results (last few lines)
    echo "Summary of results:" | tee -a "$SCRIPT_DIR/$FAILED_TESTS_LOG"
    tail -n 10 "$SCRIPT_DIR/long_narde_random_sim_output.log" | tee -a "$SCRIPT_DIR/$FAILED_TESTS_LOG"
  elif [ $exit_code -eq 124 ]; then
    echo -e "${YELLOW}Random Simulation Test timed out after 20 seconds.${NC}" | tee -a "$SCRIPT_DIR/$FAILED_TESTS_LOG"
    # Show the last few lines to see progress
    echo "Last output before timeout:" | tee -a "$SCRIPT_DIR/$FAILED_TESTS_LOG"
    tail -n 10 "$SCRIPT_DIR/long_narde_random_sim_output.log" | tee -a "$SCRIPT_DIR/$FAILED_TESTS_LOG"
  else
    echo -e "${RED}Random Simulation Test failed with exit code $exit_code.${NC}" | tee -a "$SCRIPT_DIR/$FAILED_TESTS_LOG"
    echo "Check log for details: $SCRIPT_DIR/long_narde_random_sim_output.log" | tee -a "$SCRIPT_DIR/$FAILED_TESTS_LOG"
  fi
else
  echo -e "${RED}Failed to build random_sim_test executable.${NC}" | tee -a "$SCRIPT_DIR/$FAILED_TESTS_LOG"
fi

# Go back to project root *after* all build and test execution
cd "$SCRIPT_DIR"

echo -e "${GREEN}Build and test process complete.${NC}"

# Display summary (use absolute path for log file)
if grep -q "FAILED" "$SCRIPT_DIR/$FAILED_TESTS_LOG"; then
  echo -e "${RED}Some tests failed. See $SCRIPT_DIR/$FAILED_TESTS_LOG for details.${NC}"
elif grep -q "timed out" "$SCRIPT_DIR/$FAILED_TESTS_LOG"; then
  echo -e "${YELLOW}Random sim test timed out. See $SCRIPT_DIR/$FAILED_TESTS_LOG for details.${NC}"
else
  echo -e "${GREEN}All tests passed!${NC}"
fi

echo -e "${GREEN}Build and test logging completed!${NC}" 