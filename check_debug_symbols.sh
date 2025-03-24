#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Checking for debug symbols in built files...${NC}"

# Function to check debug symbols in a file
check_debug_symbols() {
  local file=$1
  local name=$2
  
  echo -e "${YELLOW}Checking $name ($file):${NC}"
  
  if [ ! -f "$file" ]; then
    echo -e "${RED}File not found!${NC}"
    return
  fi
  
  # Check if file is a dynamic executable
  file_info=$(file "$file")
  echo "File type: $file_info"
  
  if [[ "$file_info" == *"dynamically linked"* ]]; then
    echo -e "${GREEN}File is dynamically linked.${NC}"
    
    # Check shared library dependencies
    echo -e "${YELLOW}Library dependencies:${NC}"
    ldd "$file" | grep -v linux-vdso
    
    # Check for debug symbols
    debug_info=$(readelf --debug-dump=info "$file" 2>/dev/null | head -20)
    if [ -n "$debug_info" ]; then
      echo -e "${GREEN}Contains debug symbols.${NC}"
      echo -e "${YELLOW}Debug info sample:${NC}"
      echo "$debug_info" | head -5
    else
      echo -e "${RED}No debug symbols found.${NC}"
    fi
  else
    echo -e "${RED}File is not dynamically linked.${NC}"
  fi
  
  echo ""
}

# Check build directory exists
if [ ! -d "build" ]; then
  echo -e "${RED}Build directory not found. Run rebuild_with_shared_libs.sh first.${NC}"
  exit 1
fi

cd build

# List of files to check
check_debug_symbols "examples/example" "Main Example Executable"
check_debug_symbols "games/long_narde_test" "Long Narde Test"
check_debug_symbols "games/random_sim_test" "Random Sim Test" 

# Find all libraries related to long_narde
echo -e "${YELLOW}Checking all files containing 'long_narde':${NC}"
find . -type f -name "*long_narde*" | while read file; do
  check_debug_symbols "$file" "$(basename $file)"
done

echo -e "${GREEN}Check completed.${NC}" 