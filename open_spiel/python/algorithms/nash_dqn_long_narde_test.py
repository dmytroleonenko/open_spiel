# Copyright 2024 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for open_spiel.python.algorithms.nash_dqn_long_narde."""

import unittest
import numpy as np

from absl.testing import absltest
# Import the function(s) to test
from open_spiel.python.algorithms.nash_dqn_long_narde import saddle_point, _rot12, canonical_state, convert_action_canonical, convert_legal_actions_canonical

class NashDqnLongNardeTest(absltest.TestCase):
    """Test suite for NashDQNLongNarde components."""

    def test_saddle_point_identity(self):
        """Test saddle point on an identity matrix."""
        M = np.eye(3)
        p, q, v = saddle_point(M)
        self.assertAlmostEqual(v, 1.0)
        np.testing.assert_allclose(p, [1.0, 0.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(q, [1.0, 0.0, 0.0], atol=1e-6)
        # Implementation will be added next
        # pass

    def test_saddle_point_neg_identity(self):
        """Test saddle point on a negative identity matrix."""
        M = -np.eye(3)
        p, q, v = saddle_point(M)
        self.assertAlmostEqual(v, -1.0)
        # For -I, the minimax strategy is typically uniform if multiple saddle points exist,
        # but the LP might pick a corner. Let's check if it picks the last action.
        # Check if the result indicates the last action as optimal
        # Allow for potential floating point inaccuracies, check if the max is at index 2
        self.assertEqual(np.argmax(p), 2, msg=f"p strategy {p} unexpected for -I")
        self.assertEqual(np.argmax(q), 2, msg=f"q strategy {q} unexpected for -I")
        # Or check if it's close to uniform (less likely with LP)
        # np.testing.assert_allclose(p, [1/3, 1/3, 1/3], atol=1e-6)
        # np.testing.assert_allclose(q, [1/3, 1/3, 1/3], atol=1e-6)
        # Implementation will be added next
        # pass

    def test_saddle_point_random(self):
        """Test saddle point on a random zero-sum matrix."""
        np.random.seed(42) # for reproducibility
        M = np.random.rand(4, 5)
        M = M - M.T[:4,:4].mean() # Make it approx zero-sum-ish, not perfectly
        # Create a specifically zero-sum matrix
        M_rand = np.random.rand(4, 5)
        # We need a square matrix to subtract transpose easily
        M_sq = np.random.rand(4, 4)
        M_zero_sum = M_sq - M_sq.T # Ensure M[i,j] == -M[j,i]

        p, q, v = saddle_point(M_zero_sum)

        # Basic sanity checks
        self.assertEqual(len(p), 4)
        self.assertEqual(len(q), 4)
        self.assertAlmostEqual(np.sum(p), 1.0, places=5)
        self.assertAlmostEqual(np.sum(q), 1.0, places=5)
        self.assertTrue(np.all(p >= -1e-6)) # Allow for small floating point errors
        self.assertTrue(np.all(q >= -1e-6))

        # Check von Neumann minimax theorem implication: max_p min_q pTQq = min_q max_p pTQq
        # Value for row player playing p against q should be v
        self.assertAlmostEqual(p @ M_zero_sum @ q, v, places=5)
        # Value for row player playing p against any column j should be >= v
        min_val_for_p = np.min(p @ M_zero_sum)
        self.assertTrue(min_val_for_p >= v - 1e-6)
        # Value for col player playing q against any row i should be <= v
        max_val_for_q = np.max(M_zero_sum @ q)
        self.assertTrue(max_val_for_q <= v + 1e-6)

        # Implementation will be added next
        # pass

    def test_saddle_point_masked(self):
        """Test saddle point on a matrix with masked actions."""
        MASK_VAL = -1e9
        # Case 1: One dominated masked action for row player
        M1 = np.array([[1, 0], [MASK_VAL, MASK_VAL]])
        p1, q1, v1 = saddle_point(M1)
        self.assertAlmostEqual(v1, 0.0)
        np.testing.assert_allclose(p1, [1.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(q1, [0.0, 1.0], atol=1e-6)

        # Case 2: One dominated masked action for col player
        M2 = np.array([[1, MASK_VAL], [0, MASK_VAL]])
        p2, q2, v2 = saddle_point(M2)
        self.assertAlmostEqual(v2, 1.0)
        np.testing.assert_allclose(p2, [1.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(q2, [1.0, 0.0], atol=1e-6)

        # Case 3: Fully masked row (should trigger pre-check)
        M3 = np.array([[1, 0], [MASK_VAL, MASK_VAL], [2, 3]])
        p3, q3, v3 = saddle_point(M3)
        self.assertAlmostEqual(v3, 0.0) # Pre-check returns 0 value
        # Row player should play uniform over valid rows (0 and 2)
        np.testing.assert_allclose(p3, [0.5, 0.0, 0.5], atol=1e-6)
        # Col player should play uniform over valid cols (0 and 1)
        np.testing.assert_allclose(q3, [0.5, 0.5], atol=1e-6)

        # Case 4: Fully masked column (should trigger pre-check)
        M4 = np.array([[1, MASK_VAL, 0], [2, MASK_VAL, 3]])
        p4, q4, v4 = saddle_point(M4)
        self.assertAlmostEqual(v4, 0.0)
        # Row player should play uniform over valid rows (0 and 1)
        np.testing.assert_allclose(p4, [0.5, 0.5], atol=1e-6)
        # Col player should play uniform over valid cols (0 and 2)
        np.testing.assert_allclose(q4, [0.5, 0.0, 0.5], atol=1e-6)

        # Case 5: All masked (should trigger pre-check)
        M5 = np.array([[MASK_VAL, MASK_VAL], [MASK_VAL, MASK_VAL]])
        p5, q5, v5 = saddle_point(M5)
        self.assertAlmostEqual(v5, 0.0)
        np.testing.assert_allclose(p5, [0.5, 0.5], atol=1e-6)
        np.testing.assert_allclose(q5, [0.5, 0.5], atol=1e-6)

        # Implementation will be added next
        # pass

    def test_double_rotation_is_identity(self):
        """Test that rotating by +12 twice is identity mod 24."""
        for i in range(24):
            self.assertEqual(_rot12(_rot12(i)), i)

    def test_head_and_home_rotation(self):
        """Test rotation maps black's head/home to canonical positions."""
        # Black head (11) becomes canonical head (23)
        self.assertEqual(_rot12(11), 23)
        # Black home (17-12, moving backwards) maps to canonical 5-0
        self.assertListEqual([ _rot12(i) for i in range(17, 11, -1) ], [5, 4, 3, 2, 1, 0])
        # Test individual home points
        self.assertEqual(_rot12(17), 5) # Last home point
        self.assertEqual(_rot12(12), 0) # First home point

    def test_action_mapping_conversion(self):
        """Test that converting an action view works correctly for both players."""
        # Example Black action: Move from head (11) to point 8, using a 3-pip die
        action_p1_real = (11, 8, 3)
        action_p1_canonical = convert_action_canonical(action_p1_real, player_id=1)
        # Expected canonical: rot12(11)=23, rot12(8)=20, die=3
        self.assertEqual(action_p1_canonical, (23, 20, 3))
        # Convert back to real view (should be identical operation for player 1)
        action_p1_retrieved = convert_action_canonical(action_p1_canonical, player_id=1)
        self.assertEqual(action_p1_retrieved, action_p1_real)

        # Example White action (should remain unchanged)
        action_p0_real = (23, 20, 3)
        action_p0_canonical = convert_action_canonical(action_p0_real, player_id=0)
        self.assertEqual(action_p0_canonical, action_p0_real)
        action_p0_retrieved = convert_action_canonical(action_p0_canonical, player_id=0)
        self.assertEqual(action_p0_retrieved, action_p0_real)

    def test_legal_actions_conversion(self):
        """Test converting a list of legal actions view."""
        legal_p1_real = [(11, 8, 3), (11, 7, 4)] # Black's real actions
        legal_p1_canonical = convert_legal_actions_canonical(legal_p1_real, player_id=1)
        expected_canonical = [(23, 20, 3), (23, 19, 4)]
        self.assertListEqual(legal_p1_canonical, expected_canonical)

        # Convert back to real view
        legal_p1_retrieved = convert_legal_actions_canonical(legal_p1_canonical, player_id=1)
        self.assertListEqual(legal_p1_retrieved, legal_p1_real)

        # Test for player 0 (should do nothing)
        legal_p0_real = [(23, 20, 3), (23, 19, 4)]
        legal_p0_canonical = convert_legal_actions_canonical(legal_p0_real, player_id=0)
        self.assertListEqual(legal_p0_canonical, legal_p0_real)
        legal_p0_retrieved = convert_legal_actions_canonical(legal_p0_canonical, player_id=0)
        self.assertListEqual(legal_p0_retrieved, legal_p0_real)

    def test_canonical_state_mapping(self):
        """Test the canonical_state function for player 1."""
        # Create a sample observation (54-dim)
        obs = np.zeros(54, dtype=np.float32)
        # White pieces (player 0 perspective)
        obs[23] = 1 # White head
        obs[5] = 1 # White home
        # Black pieces (player 0 perspective)
        obs[24 + 11] = 1 # Black head
        obs[24 + 17] = 1 # Black home

        # Get canonical view for player 1 (black)
        canonical_obs_p1 = canonical_state(obs, player_id=1)

        # Check canonical white pieces (should be black's rotated pieces)
        # Black head (11) rotated is 23
        self.assertEqual(canonical_obs_p1[23], 1)
        # Black home (17) rotated is 5
        self.assertEqual(canonical_obs_p1[5], 1)
        # Check other white positions are 0
        self.assertEqual(np.sum(canonical_obs_p1[:24]) - canonical_obs_p1[23] - canonical_obs_p1[5], 0)

        # Check canonical black pieces (should be white's rotated pieces)
        # White head (23) rotated is 11
        self.assertEqual(canonical_obs_p1[24 + 11], 1)
        # White home (5) rotated is 17
        self.assertEqual(canonical_obs_p1[24 + 17], 1)
         # Check other black positions are 0
        self.assertEqual(np.sum(canonical_obs_p1[24:48]) - canonical_obs_p1[24+11] - canonical_obs_p1[24+17], 0)

        # Check canonical view for player 0 (should be identity)
        canonical_obs_p0 = canonical_state(obs, player_id=0)
        np.testing.assert_array_equal(canonical_obs_p0, obs)

if __name__ == "__main__":
    absltest.main() 