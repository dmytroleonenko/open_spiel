
import numpy as np
import pytest

def compute_n_step_targets(rewards, search_values, n_step, gamma):
    targets = []
    length = len(rewards)
    for i in range(length):
        g_val = 0.0
        for k in range(n_step):
            if i + k < length:
                g_val += (gamma ** k) * rewards[i + k]

        # Bootstrap
        if i + n_step < len(search_values):
            g_val += (gamma ** n_step) * search_values[i + n_step]

        targets.append(g_val)
    return targets

def test_sarsa_computation():
    """Verify n-step return logic matches expected behavior."""
    rewards = [1.0, 1.0, 1.0, 1.0, 1.0]
    # search_values represents network values for bootstrapping
    search_values = [0.5, 0.5, 0.5, 0.5, 0.5]

    n_step = 2
    gamma = 0.9

    targets = compute_n_step_targets(rewards, search_values, n_step, gamma)

    # t=0: r0 + g*r1 + g^2*v2
    # 1.0 + 0.9*1.0 + 0.81*0.5 = 1.9 + 0.405 = 2.305
    assert np.isclose(targets[0], 2.305)

    # t=3: r3 + g*r4 + g^2*v5(out) -> r3 + g*r4 (no bootstrap as index 5 out of bounds)
    # 1.0 + 0.9*1.0 = 1.9
    assert np.isclose(targets[3], 1.9)

    # t=4: r4 + g*v5(out) -> r4
    # 1.0
    assert np.isclose(targets[4], 1.0)

if __name__ == "__main__":
    test_sarsa_computation()
