#!/usr/bin/env python3
"""
Comprehensive script to fix all unpacking issues after LSTM integration.
The LSTM integration added a 6th return value to initial_inference and recurrent_inference methods.
"""

import re
import os
import glob

def fix_unpacking_in_file(filepath):
    """Fix unpacking issues in a single file."""
    try:
        with open(filepath, 'r') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return False
    
    original_content = content
    changes_made = False
    
    # Fix duplicate reward_hidden first (from previous script runs)
    if ', reward_hidden, reward_hidden' in content:
        content = content.replace(', reward_hidden, reward_hidden', ', reward_hidden')
        changes_made = True
    
    # Pattern 1: Fix initial_inference unpacking (5 values -> 6 values)
    # Look for patterns like: var1, var2, var3, var4, var5 = model.initial_inference(...)
    # But avoid cases that already have 6 values or use _ placeholders
    pattern1 = r'(\w+),\s*(\w+),\s*(\w+),\s*(\w+),\s*(\w+)\s*=\s*(\w+)\.initial_inference\('
    def replacement1(match):
        if 'reward_hidden' not in match.group(0):
            return f'{match.group(1)}, {match.group(2)}, {match.group(3)}, {match.group(4)}, {match.group(5)}, reward_hidden = {match.group(6)}.initial_inference('
        return match.group(0)
    
    new_content = re.sub(pattern1, replacement1, content)
    if new_content != content:
        content = new_content
        changes_made = True
    
    # Pattern 2: Fix recurrent_inference unpacking (5 values -> 6 values)
    pattern2 = r'(\w+),\s*(\w+),\s*(\w+),\s*(\w+),\s*(\w+)\s*=\s*(\w+)\.recurrent_inference\('
    def replacement2(match):
        if 'reward_hidden' not in match.group(0):
            return f'{match.group(1)}, {match.group(2)}, {match.group(3)}, {match.group(4)}, {match.group(5)}, reward_hidden = {match.group(6)}.recurrent_inference('
        return match.group(0)
    
    new_content = re.sub(pattern2, replacement2, content)
    if new_content != content:
        content = new_content
        changes_made = True
    
    # Pattern 3: Fix cases with underscores like: _, _, value, _, _ = model.initial_inference(...)
    pattern3 = r'(_),\s*(_),\s*(\w+),\s*(_),\s*(_)\s*=\s*(\w+)\.initial_inference\('
    def replacement3(match):
        if 'reward_hidden' not in match.group(0):
            return f'_, _, {match.group(3)}, _, _, reward_hidden = {match.group(6)}.initial_inference('
        return match.group(0)
    
    new_content = re.sub(pattern3, replacement3, content)
    if new_content != content:
        content = new_content
        changes_made = True
    
    # Pattern 4: Fix cases with mixed underscores and variables for recurrent_inference
    pattern4 = r'(_),\s*(\w+),\s*(\w+),\s*(_),\s*(_)\s*=\s*(\w+)\.recurrent_inference\('
    def replacement4(match):
        if 'reward_hidden' not in match.group(0):
            return f'_, {match.group(2)}, {match.group(3)}, _, _, reward_hidden = {match.group(6)}.recurrent_inference('
        return match.group(0)
    
    new_content = re.sub(pattern4, replacement4, content)
    if new_content != content:
        content = new_content
        changes_made = True
    
    # Pattern 5: Fix cases like: _, reward_e, value_e, _, _ = model_base_e.initial_inference(...)
    pattern5 = r'(_),\s*(\w+),\s*(\w+),\s*(_),\s*(_)\s*=\s*(\w+)\.initial_inference\('
    def replacement5(match):
        if 'reward_hidden' not in match.group(0):
            return f'_, {match.group(2)}, {match.group(3)}, _, _, reward_hidden = {match.group(6)}.initial_inference('
        return match.group(0)
    
    new_content = re.sub(pattern5, replacement5, content)
    if new_content != content:
        content = new_content
        changes_made = True
    
    # Write back if changes were made
    if changes_made and content != original_content:
        try:
            with open(filepath, 'w') as f:
                f.write(content)
            print(f"✅ Fixed unpacking issues in {filepath}")
            return True
        except Exception as e:
            print(f"❌ Error writing {filepath}: {e}")
            return False
    elif changes_made:
        print(f"⚠️  No net changes in {filepath} (duplicate fixes)")
        return False
    else:
        print(f"ℹ️  No unpacking issues found in {filepath}")
        return False

def main():
    """Main function to fix all files."""
    print("🔧 Fixing unpacking issues across all test files...")
    
    # Find all Python files in the muzero_jax directory
    base_dir = "open_spiel/python/algorithms/muzero_jax"
    
    # Patterns to find relevant files
    patterns = [
        f"{base_dir}/tests/**/*.py",
        f"{base_dir}/self_play/*.py", 
        f"{base_dir}/training/*.py",
        f"{base_dir}/mcts/*.py",
        f"{base_dir}/*.py",
        "test_*.py",  # Root level test files
        "debug_*.py"  # Debug files
    ]
    
    files_to_fix = set()
    for pattern in patterns:
        files_to_fix.update(glob.glob(pattern, recursive=True))
    
    # Remove files that are already fixed (like test_network.py)
    files_to_fix = [f for f in files_to_fix if f.endswith('.py')]
    
    print(f"📁 Found {len(files_to_fix)} Python files to check")
    
    fixed_count = 0
    for filepath in sorted(files_to_fix):
        if fix_unpacking_in_file(filepath):
            fixed_count += 1
    
    print(f"\n🎉 Fixed unpacking issues in {fixed_count} files")
    print("✅ All files processed!")

if __name__ == "__main__":
    main() 