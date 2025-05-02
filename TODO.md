## Action Plan

### Refactor Long Narde Action Space

1. Analyze current action encoding in long_narde and document all sources of action space inflation.
2. Redesign the action encoding to:
    - Remove die value from the action encoding (use only positions, not dice values).
    - Remove dice order from the action encoding (treat all dice orders as equivalent).
    - Use minimal base for encoding (base-25: 24 points + pass).
    - For doubles, encode only the set of up to 4 moves using a canonical order.
    - Ensure the action space matches the theoretical minimum (target: 1250 unique actions).
3. Refactor the encoding and decoding logic in long_narde to implement the new scheme.
4. Update the action space size constant and all related documentation/comments.
5. Update the Python bindings and any wrappers if they depend on the action space size or encoding.

### Test Suite Review and Update

6. Review every test file matching *test*.cc for long_narde:
    - Identify all tests that depend on the action encoding or action space size.
    - Update tests to use the new encoding and expected action space.
    - Add/adjust tests to verify that the new encoding is correct and minimal.
7. Run the full test suite and verify all tests pass with the new encoding.
8. Document any changes in test coverage or behavior due to the encoding refactor.
