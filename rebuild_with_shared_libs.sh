#!/bin/bash
set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Building OpenSpiel with shared libraries and debug symbols for long_narde${NC}"

# Clean existing build directory
if [ -d "build" ]; then
  echo -e "${YELLOW}Removing existing build directory...${NC}"
  rm -rf build
fi

# Create build directory
mkdir -p build
cd build

# Configure with CMake
echo -e "${YELLOW}Configuring with CMake...${NC}"
cmake -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Testing ../open_spiel

# Build
echo -e "${YELLOW}Building...${NC}"
make -j$(nproc)

# Run long_narde tests
echo -e "${YELLOW}Running long_narde tests:${NC}"
ctest -V -R "long_narde_.*" || echo "Some tests failed"

# Verify shared libraries
echo -e "${YELLOW}Verifying shared libraries:${NC}"
find . -name "*.so" | wc -l

# Check binary sizes
echo -e "${YELLOW}Checking binary sizes:${NC}"
ls -lh open_spiel/games/long_narde/long_narde_test

echo -e "${GREEN}Build completed successfully!${NC}" 