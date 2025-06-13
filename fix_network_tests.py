#!/usr/bin/env python3
"""
Script to fix network test unpacking issues after LSTM integration.
The LSTM integration added a 6th return value to initial_inference and recurrent_inference methods.
"""

import re
import sys

def fix_unpacking_in_file(filepath):
    """Fix unpacking issues in a single file."""
    with open(filepath, 'r') as f:
        content = f.read()
    
    original_content = content
    
    # Pattern 1: Fix initial_inference unpacking (5 values -> 6 values)
    # Look for patterns like: var1, var2, var3, var4, var5 = model.initial_inference(...)
    pattern1 = r'(\w+),\s*(\w+),\s*(\w+),\s*(\w+),\s*(\w+)\s*=\s*(\w+)\.initial_inference\('
    replacement1 = r'\1, \2, \3, \4, \5, reward_hidden = \6.initial_inference('
    content = re.sub(pattern1, replacement1, content)
    
    # Pattern 2: Fix recurrent_inference unpacking (5 values -> 6 values)
    pattern2 = r'(\w+),\s*(\w+),\s*(\w+),\s*(\w+),\s*(\w+)\s*=\s*(\w+)\.recurrent_inference\('
    replacement2 = r'\1, \2, \3, \4, \5, reward_hidden = \6.recurrent_inference('
    content = re.sub(pattern2, replacement2, content)
    
    # Pattern 3: Handle cases with underscores (ignoring return values)
    # _, _, value_symlog, _, _ = model_symlog.initial_inference(...)
    pattern3 = r'(_),\s*(_),\s*(\w+),\s*(_),\s*(_)\s*=\s*(\w+)\.initial_inference\('
    replacement3 = r'\1, \2, \3, \4, \5, _ = \6.initial_inference('
    content = re.sub(pattern3, replacement3, content)
    
    pattern4 = r'(_),\s*(_),\s*(\w+),\s*(_),\s*(_)\s*=\s*(\w+)\.recurrent_inference\('
    replacement4 = r'\1, \2, \3, \4, \5, _ = \6.recurrent_inference('
    content = re.sub(pattern4, replacement4, content)
    
    # Pattern 5: Handle mixed cases with some underscores
    # hidden_state, reward, value, policy_logits, _ = model.initial_inference(...)
    pattern5 = r'(\w+),\s*(\w+),\s*(\w+),\s*(\w+),\s*(_)\s*=\s*(\w+)\.initial_inference\('
    replacement5 = r'\1, \2, \3, \4, \5, reward_hidden = \6.initial_inference('
    content = re.sub(pattern5, replacement5, content)
    
    pattern6 = r'(\w+),\s*(\w+),\s*(\w+),\s*(\w+),\s*(_)\s*=\s*(\w+)\.recurrent_inference\('
    replacement6 = r'\1, \2, \3, \4, \5, reward_hidden = \6.recurrent_inference('
    content = re.sub(pattern6, replacement6, content)
    
    if content != original_content:
        with open(filepath, 'w') as f:
            f.write(content)
        print(f"Fixed unpacking issues in {filepath}")
        return True
    else:
        print(f"No changes needed in {filepath}")
        return False

def main():
    """Main function to fix network test files."""
    files_to_fix = [
        'open_spiel/python/algorithms/muzero_jax/tests/models/test_network.py'
    ]
    
    total_fixed = 0
    for filepath in files_to_fix:
        try:
            if fix_unpacking_in_file(filepath):
                total_fixed += 1
        except Exception as e:
            print(f"Error fixing {filepath}: {e}")
    
    print(f"\nFixed {total_fixed} files total.")

if __name__ == "__main__":
    main() 