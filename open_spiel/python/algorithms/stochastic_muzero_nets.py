import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List, Tuple, Optional

# Helper MLP class
class MLP(nn.Module):
    def __init__(self, input_size: int, layer_sizes: List[int], output_size: int, activation=nn.ReLU):
        super().__init__()
        layers = []
        current_size = input_size
        for size in layer_sizes:
            layers.append(nn.Linear(current_size, size))
            layers.append(activation())
            current_size = size
        layers.append(nn.Linear(current_size, output_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# --- Stochastic MuZero Network Components (PyTorch) ---

class RepresentationNetwork(nn.Module):
    """Encodes observation to latent state 's'. (h function)"""
    def __init__(self, observation_size: int, encoding_size: int, hidden_layers: List[int] = [128, 128]):
        super().__init__()
        self.net = MLP(observation_size, hidden_layers, encoding_size)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        # observation shape: [B, *observation_shape]
        # Flatten if necessary
        if observation.ndim > 2:
            observation = observation.view(observation.size(0), -1)
        latent_state = self.net(observation)
        # Output latent_state shape: [B, encoding_size]
        return latent_state

class PredictionNetwork(nn.Module):
    """Predicts policy logits and value from latent state 's'. (f function)"""
    def __init__(self, encoding_size: int, action_space_size: int, hidden_layers: List[int] = [128]):
        super().__init__()
        self.policy_head = MLP(encoding_size, hidden_layers, action_space_size)
        self.value_head = MLP(encoding_size, hidden_layers, 1)

    def forward(self, latent_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # latent_state shape: [B, encoding_size]
        policy_logits = self.policy_head(latent_state)
        value = self.value_head(latent_state)
        # policy_logits shape: [B, action_space_size]
        # value shape: [B, 1]
        return policy_logits, value

class AfterstateDynamicsNetwork(nn.Module):
    """Predicts afterstate 'as' from latent state 's' and action 'a'. (phi function)"""
    def __init__(self, encoding_size: int, action_space_size: int, hidden_layers: List[int] = [128, 128]):
        super().__init__()
        # Input is concatenation of latent state and one-hot encoded action
        input_size = encoding_size + action_space_size
        self.net = MLP(input_size, hidden_layers, encoding_size)
        self.action_space_size = action_space_size

    def forward(self, latent_state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        # latent_state shape: [B, encoding_size]
        # action shape: [B] (indices)
        action_one_hot = F.one_hot(action, num_classes=self.action_space_size).float()
        # action_one_hot shape: [B, action_space_size]
        combined = torch.cat((latent_state, action_one_hot), dim=1)
        afterstate = self.net(combined)
        # afterstate shape: [B, encoding_size]
        return afterstate

class AfterstatePredictionNetwork(nn.Module):
    """Predicts chance outcome logits and afterstate value Q(s,a) from afterstate 'as'. (psi function)"""
    def __init__(self, encoding_size: int, codebook_size: int, hidden_layers: List[int] = [128]):
        super().__init__()
        self.chance_logits_head = MLP(encoding_size, hidden_layers, codebook_size)
        self.afterstate_value_head = MLP(encoding_size, hidden_layers, 1) # Q(s, a)

    def forward(self, afterstate: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # afterstate shape: [B, encoding_size]
        chance_logits = self.chance_logits_head(afterstate)
        afterstate_value = self.afterstate_value_head(afterstate)
        # chance_logits shape: [B, codebook_size]
        # afterstate_value shape: [B, 1]
        return chance_logits, afterstate_value

class DynamicsNetwork(nn.Module):
    """Predicts next latent state 's' and reward 'r' from afterstate 'as' and chance outcome 'c'. (g function)"""
    def __init__(self, encoding_size: int, codebook_size: int, hidden_layers: List[int] = [128, 128]):
        super().__init__()
        # Input is concatenation of afterstate and one-hot encoded chance outcome
        input_size = encoding_size + codebook_size
        self.state_head = MLP(input_size, hidden_layers, encoding_size)
        self.reward_head = MLP(input_size, hidden_layers, 1)
        self.codebook_size = codebook_size

    def forward(self, afterstate: torch.Tensor, chance_outcome: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # afterstate shape: [B, encoding_size]
        # chance_outcome shape: [B] (indices)
        chance_one_hot = F.one_hot(chance_outcome, num_classes=self.codebook_size).float()
        # chance_one_hot shape: [B, codebook_size]
        combined = torch.cat((afterstate, chance_one_hot), dim=1)
        next_latent_state = self.state_head(combined)
        reward = self.reward_head(combined)
        # next_latent_state shape: [B, encoding_size]
        # reward shape: [B, 1]
        return next_latent_state, reward

# --- Main Network Wrapper ---

class StochasticMuZeroNetwork(nn.Module):
    """Combines all Stochastic MuZero components into a single network."""
    def __init__(self, config: 'StochasticMuZeroConfig'): # Use forward reference
        super().__init__()
        self.config = config

        obs_size = config.state_representation_size
        act_size = config.action_space_size
        enc_size = config.encoding_size
        code_size = config.codebook_size

        # TODO: Make hidden layer sizes configurable
        hidden_layers_rep = [256, 256]
        hidden_layers_pred = [128]
        hidden_layers_after_dyn = [256, 256]
        hidden_layers_after_pred = [128]
        hidden_layers_dyn = [256, 256]

        self.representation_net = RepresentationNetwork(obs_size, enc_size, hidden_layers_rep)
        self.prediction_net = PredictionNetwork(enc_size, act_size, hidden_layers_pred)
        self.afterstate_dynamics_net = AfterstateDynamicsNetwork(enc_size, act_size, hidden_layers_after_dyn)
        self.afterstate_prediction_net = AfterstatePredictionNetwork(enc_size, code_size, hidden_layers_after_pred)
        self.dynamics_net = DynamicsNetwork(enc_size, code_size, hidden_layers_dyn)

        # TODO: Implement VQ-VAE encoder if needed (e function)
        # self.encoder_net = ...

    # --- Main Interface Methods ---

    def initial_inference(self, observation: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Representation + Prediction. Used at the root of MCTS."""
        latent_state = self.representation_net(observation)
        policy_logits, value = self.prediction_net(latent_state)
        return {
            "latent_state": latent_state,
            "policy_logits": policy_logits,
            "value": value
        }

    def recurrent_inference(self, latent_state: torch.Tensor, action: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Afterstate Dynamics + Afterstate Prediction + Dynamics + Prediction.
           Used during MCTS simulation after selecting an action and sampling a chance outcome.
           Requires a sampled chance_outcome corresponding to the action.
        """
        afterstate = self.afterstate_dynamics_net(latent_state, action)
        chance_logits, afterstate_value = self.afterstate_prediction_net(afterstate)

        # --- Crucial Step: Sampling Chance Outcome ---
        # In a real implementation, the chance_outcome would be sampled based on chance_logits
        # or provided from the MCTS node expansion if using an explicit chance node.
        # For now, we need a placeholder way to get the chance outcome.
        # Let's assume it's provided externally for this recurrent step.
        # We'll handle the actual sampling/node logic in the MCTS part.
        # Placeholder: Sample a random chance outcome for testing the forward pass.
        # In practice, this should come from the MCTS process.
        if self.training:
            # During training, we might get the true outcome from replay buffer
            # Or sample based on the predicted logits
             _, sampled_chance_outcome = torch.max(chance_logits, dim=1) # Greedy sample for placeholder
        else:
             # During inference/MCTS, sampling depends on search strategy
             _, sampled_chance_outcome = torch.max(chance_logits, dim=1) # Greedy sample for placeholder

        next_latent_state, reward = self.dynamics_net(afterstate, sampled_chance_outcome)
        next_policy_logits, next_value = self.prediction_net(next_latent_state)

        return {
            "afterstate": afterstate,
            "chance_logits": chance_logits,
            "afterstate_value": afterstate_value, # Q(s,a)
            "reward": reward,
            "next_latent_state": next_latent_state,
            "next_policy_logits": next_policy_logits,
            "next_value": next_value
            # Note: sampled_chance_outcome is handled outside this function during MCTS
        }

    # --- Convenience wrappers for individual network components ---
    def representation(self, observation: torch.Tensor) -> torch.Tensor:
        return self.representation_net(observation)

    def prediction(self, latent_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.prediction_net(latent_state)

    def afterstate_dynamics(self, latent_state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.afterstate_dynamics_net(latent_state, action)

    def afterstate_prediction(self, afterstate: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.afterstate_prediction_net(afterstate)

    def dynamics(self, afterstate: torch.Tensor, chance_outcome: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.dynamics_net(afterstate, chance_outcome)

# --- Network Factory Function ---
def muzero_network_factory(config: 'StochasticMuZeroConfig') -> StochasticMuZeroNetwork:
    """Creates the StochasticMuZeroNetwork based on the config."""
    return StochasticMuZeroNetwork(config) 