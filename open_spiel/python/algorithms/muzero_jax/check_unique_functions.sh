#!/bin/bash

# ==============================================================================
# Function Uniqueness Checker for MuZero JAX Test Suite
# ==============================================================================
#
# DESCRIPTION:
#   This script verifies that all test functions across test_trainer_*.py files
#   are unique (no duplicates). It's essential for maintaining clean test 
#   organization after refactoring the massive test_trainer.py file.
#   
#   ENHANCED FEATURE: Also detects functions that were copied from test_trainer.py
#   to other files but still remain in the original test_trainer.py file.
#
# USAGE:
#   ./check_unique_functions.sh   (from any directory)
#   OR
#   bash /path/to/check_unique_functions.sh
#
# REQUIREMENTS:
#   - Requires bash shell with standard Unix tools (grep, sed, sort, uniq)
#   - Script should be located in the muzero_jax directory
#   - Test files should be in tests/training/ subdirectory relative to script location
#
# WHAT IT DOES:
#   1. Auto-detects the muzero_jax directory based on script location
#   2. Scans all test_trainer_*.py files in tests/training/
#   3. Extracts all function names starting with "def test_"
#   4. Identifies any duplicate function names across files
#   5. Shows which files contain each duplicate
#   6. Reports total function count and uniqueness status
#   7. **NEW**: Detects functions in test_trainer.py that also exist in other files
#   8. **NEW**: Suggests removal commands for incomplete migrations
#
# EXPECTED OUTPUT (when no duplicates):
#   === FUNCTION UNIQUENESS CHECK ===
#   Working directory: /path/to/muzero_jax
#   Total functions found: 125
#   ✅ All functions are unique!
#   ✅ No functions need removal from test_trainer.py
#
# EXPECTED OUTPUT (when duplicates exist):
#   === FUNCTION UNIQUENESS CHECK ===
#   Working directory: /path/to/muzero_jax
#   Total functions found: 154
#   ❌ Duplicate functions found:
#     - test_some_function
#       test_some_function|tests/training/test_trainer_file1.py
#       test_some_function|tests/training/test_trainer_file2.py
#   ❌ Functions in test_trainer.py that exist elsewhere (need removal):
#     - test_moved_function (also in test_trainer_other.py)
#
# RECOMMENDED ACTIONS WHEN DUPLICATES FOUND:
#   1. Review the duplicate functions to understand their purpose
#   2. Determine which file should contain the function (based on logical grouping)
#   3. Remove the duplicate from the other file(s) using:
#      mcp_nuanced_remove_functions or manual editing
#   4. Re-run this script to verify duplicates are resolved
#
# RECOMMENDED ACTIONS WHEN INCOMPLETE REMOVALS FOUND:
#   1. Use the suggested mcp_nuanced_remove_functions commands
#   2. Re-run this script to verify removals are complete
#
# MAINTENANCE:
#   - Run this script from any directory - it will auto-detect the correct paths
#   - Include it in CI/CD pipeline to prevent duplicate commits
#   - Update if test file naming convention changes
#
# ==============================================================================

set -e  # Exit on any error

echo "=== FUNCTION UNIQUENESS CHECK ==="

# Auto-detect the script's directory and use it as the base directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$SCRIPT_DIR"
TESTS_DIR="$BASE_DIR/tests/training"

echo "Script location: $SCRIPT_DIR"
echo "Working directory: $BASE_DIR"
echo "Scanning $TESTS_DIR/test_trainer_*.py files..."

# Check if the tests/training directory exists relative to script location
if [ ! -d "$TESTS_DIR" ]; then
    echo "❌ Error: $TESTS_DIR directory not found!"
    echo "   Expected directory structure:"
    echo "     $BASE_DIR/"
    echo "     ├── tests/"
    echo "     │   └── training/"
    echo "     │       ├── test_trainer.py"
    echo "     │       ├── test_trainer_*.py"
    echo "     │       └── ..."
    echo "     └── check_unique_functions.sh  (this script)"
    echo ""
    echo "   Please ensure the script is located in the muzero_jax directory"
    echo "   and that the tests/training/ subdirectory exists."
    exit 1
fi

# Check if test files exist
test_files=$(find "$TESTS_DIR" -name "test_trainer*.py" -type f | wc -l | tr -d ' ')
if [ "$test_files" -eq 0 ]; then
    echo "❌ Error: No test_trainer*.py files found in $TESTS_DIR/"
    echo "   Expected to find files like:"
    echo "     - test_trainer.py"
    echo "     - test_trainer_core.py"
    echo "     - test_trainer_checkpoints.py"
    echo "     - etc."
    exit 1
fi

echo "Found $test_files test_trainer_*.py files"

# Extract all functions with their source files
temp_file=$(mktemp)
temp_trainer_functions=$(mktemp)
temp_other_functions=$(mktemp)

for file in "$TESTS_DIR"/test_trainer*.py; do
    if [ -f "$file" ]; then
        # Use relative path from base directory for cleaner output
        relative_file=$(echo "$file" | sed "s|^$BASE_DIR/||")
        grep "^def test_" "$file" 2>/dev/null | sed 's/def \([^(]*\).*/\1/' | while read func; do
            echo "$func|$relative_file"
        done
    fi
done > "$temp_file"

# Separate test_trainer.py functions from others
grep "|tests/training/test_trainer\.py$" "$temp_file" | cut -d'|' -f1 | sort > "$temp_trainer_functions"
grep -v "|tests/training/test_trainer\.py$" "$temp_file" | cut -d'|' -f1 | sort > "$temp_other_functions"

total_funcs=$(wc -l < "$temp_file")
trainer_funcs=$(wc -l < "$temp_trainer_functions")
other_funcs=$(wc -l < "$temp_other_functions")

echo "Total functions found: $total_funcs"
echo "Functions in test_trainer.py: $trainer_funcs"
echo "Functions in other files: $other_funcs"

if [ "$total_funcs" -eq 0 ]; then
    echo "⚠️  Warning: No test functions found!"
    rm "$temp_file" "$temp_trainer_functions" "$temp_other_functions"
    exit 0
fi

# Check for duplicates across all files
echo "Checking for duplicate function names..."
duplicates=$(cut -d'|' -f1 "$temp_file" | sort | uniq -d)

# Check for functions in test_trainer.py that also exist in other files
echo "Checking for functions in test_trainer.py that exist elsewhere..."
incomplete_removals=$(comm -12 "$temp_trainer_functions" "$temp_other_functions")

# Initialize status flags
has_duplicates=false
has_incomplete_removals=false
overall_status="PASS"

# Report duplicates
if [ -z "$duplicates" ]; then
    echo "✅ All functions are unique across files!"
else
    echo "❌ Duplicate functions found:"
    echo ""
    has_duplicates=true
    overall_status="FAIL"
    
    duplicate_count=0
    for func in $duplicates; do
        echo "  🔴 DUPLICATE: $func"
        echo "     Found in files:"
        grep "^$func|" "$temp_file" | cut -d'|' -f2 | sed 's/^/       - /'
        echo ""
        duplicate_count=$((duplicate_count + 1))
    done
fi

# Report incomplete removals
if [ -z "$incomplete_removals" ]; then
    echo "✅ No functions need removal from test_trainer.py"
else
    echo "❌ Functions in test_trainer.py that exist elsewhere (need removal):"
    echo ""
    has_incomplete_removals=true
    overall_status="FAIL"
    
    removal_count=0
    removal_functions=""
    for func in $incomplete_removals; do
        echo "  🟡 INCOMPLETE REMOVAL: $func"
        echo "     Also found in:"
        grep "^$func|" "$temp_file" | grep -v "test_trainer\.py$" | cut -d'|' -f2 | sed 's/^/       - /'
        echo ""
        
        # Build removal function list
        if [ -z "$removal_functions" ]; then
            removal_functions="\"$func\""
        else
            removal_functions="$removal_functions, \"$func\""
        fi
        
        removal_count=$((removal_count + 1))
    done
fi

echo ""
echo "SUMMARY:"
echo "  - Base directory: $BASE_DIR"
echo "  - Total test files: $test_files"
echo "  - Total test functions: $total_funcs"
echo "  - Functions in test_trainer.py: $trainer_funcs"
echo "  - Functions in other files: $other_funcs"

if [ "$has_duplicates" = true ]; then
    echo "  - Duplicate functions: $duplicate_count"
else
    echo "  - Duplicate functions: 0"
fi

if [ "$has_incomplete_removals" = true ]; then
    echo "  - Incomplete removals: $removal_count"
else
    echo "  - Incomplete removals: 0"
fi

echo "  - Status: $overall_status $([ "$overall_status" = "PASS" ] && echo "✅" || echo "❌")"

if [ "$overall_status" = "FAIL" ]; then
    echo ""
    echo "NEXT STEPS:"
    
    if [ "$has_duplicates" = true ]; then
        echo "  FOR DUPLICATES:"
        echo "    1. Review each duplicate function listed above"
        echo "    2. Decide which file should contain each function"
        echo "    3. Remove duplicates from other files"
    fi
    
    if [ "$has_incomplete_removals" = true ]; then
        echo "  FOR INCOMPLETE REMOVALS:"
        echo "    1. Remove functions from test_trainer.py using the command below:"
        echo "    2. Run this command from a directory where mcp_nuanced tools are available:"
        echo ""
        echo "    REMOVAL COMMAND:"
        echo "    mcp_nuanced_remove_functions(file_path=\"tests/training/test_trainer.py\", function_names=[$removal_functions])"
        echo ""
        echo "    This will remove the following functions from test_trainer.py:"
        for func in $incomplete_removals; do
            echo "      - $func"
        done
    fi
    
    echo ""
    echo "  3. Re-run this script to verify all issues are resolved"
    echo "     (can be run from any directory: $SCRIPT_DIR/$(basename "$0")"
    
    rm "$temp_file" "$temp_trainer_functions" "$temp_other_functions"
    exit 1
fi

rm "$temp_file" "$temp_trainer_functions" "$temp_other_functions"
echo ""
echo "Function uniqueness verification completed successfully! 🎉"
echo "All functions are properly organized with no duplicates or incomplete removals."
echo ""
echo "You can run this script from any directory using:"
echo "  $SCRIPT_DIR/$(basename "$0")" 