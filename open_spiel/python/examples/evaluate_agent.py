"""Evaluate a trained agent against different opponents in an OpenSpiel game."""

import pyspiel
from open_spiel.python import rl_environment
from open_spiel.python.algorithms import random_agent # For random opponent
from open_spiel.python.algorithms import tabular_qlearner_long_narde # For Long Narde/Backgammon forced pass Q-learner
from open_spiel.python.utils import agent_serialization # Added
import csv # For output
import os # For path checks
import sys # For error messages
from absl import app
from absl import flags
import yaml # Added for config file loading
import json # Added for config file loading
from tensorboardX import SummaryWriter # Added for TensorBoard logging
import math # Added for Elo calculation

FLAGS = flags.FLAGS

# Evaluation parameters
flags.DEFINE_string("game_name", "tic_tac_toe", "Name of the game to evaluate.")
flags.DEFINE_integer("num_eval_episodes", 100, "Number of episodes to run for evaluation.")
flags.DEFINE_string("agent_path", None, "Path to load the primary trained agent from (directory containing checkpoints).")
flags.DEFINE_string("opponent_type", "random", "Type of opponent ('random' or 'trained').")
flags.DEFINE_string("opponent_path", None, "Path to load a trained opponent agent (if opponent_type='trained').")
flags.DEFINE_string("output_file", "evaluation_results.csv", "Path to save evaluation metrics (CSV).")
# TODO: Add flag for primary agent's player ID if it might not be 0?
# TODO: Add flag for opponent agent's player ID if it might not be 1?

# --- Configuration File ---
flags.DEFINE_string("config_file", None, "Path to a YAML/JSON configuration file. Flags set in the file will override defaults but be overridden by command-line flags.")

# --- TensorBoard Logging ---
flags.DEFINE_string("tensorboard_logdir", None, "Optional directory to save TensorBoard logs for evaluation metrics.")

# --- Elo Rating Parameters --- (Task 14)
flags.DEFINE_float("initial_elo", 1200.0, "Initial Elo rating for both agents.")
flags.DEFINE_float("elo_k_factor", 32.0, "K-factor used in Elo rating updates.")

# Ensure required flags are provided
flags.mark_flag_as_required("agent_path")

# --- Elo Calculation Functions --- (Task 14)
def calculate_expected_outcome(rating1, rating2):
    """Calculate the expected outcome (probability of winning) for player 1."""
    return 1.0 / (1.0 + math.pow(10.0, (rating2 - rating1) / 400.0))

def update_elo(rating1, rating2, score1, k_factor):
    """Update Elo ratings based on the outcome.

    Args:
        rating1: Current Elo rating of player 1.
        rating2: Current Elo rating of player 2.
        score1: Outcome for player 1 (1.0 for win, 0.5 for tie, 0.0 for loss).
        k_factor: The K-factor determining the magnitude of rating change.

    Returns:
        A tuple (new_rating1, new_rating2).
    """
    expected1 = calculate_expected_outcome(rating1, rating2)
    # expected2 = 1.0 - expected1 # Not needed directly

    new_rating1 = rating1 + k_factor * (score1 - expected1)
    # Player 2's score is the inverse of player 1's score in zero-sum games
    score2 = 1.0 - score1
    expected2 = calculate_expected_outcome(rating2, rating1) # Calculate expected for p2 vs p1
    new_rating2 = rating2 + k_factor * (score2 - expected2)

    # Alternative calculation for player 2 (should yield the same result for zero-sum):
    # new_rating2 = rating2 - k_factor * (score1 - expected1)

    return new_rating1, new_rating2

# --- Game Specific Metrics --- (Task 15)
def calculate_pip_counts(state: pyspiel.LongNardeState):
    """Calculates the pip counts for both players in a Long Narde state.

    Args:
        state: The pyspiel.LongNardeState object.

    Returns:
        A list [player0_pips, player1_pips].
    """
    pip_counts = [0, 0]
    num_checkers = [0, 0] # Sanity check
    for player in range(2):
        for pos in range(pyspiel.long_narde.kNumPoints):
            count = state.board(player, pos)
            if count > 0:
                num_checkers[player] += count
                if player == pyspiel.long_narde.kXPlayerId: # White (Player 0)
                    # Pip count is distance to bear off (from point 24)
                    pips_for_point = (pyspiel.long_narde.kNumPoints - pos)
                    pip_counts[player] += count * pips_for_point
                else: # Black (Player 1)
                    # Pip count is distance to bear off (from point 1)
                    pips_for_point = (pos + 1)
                    pip_counts[player] += count * pips_for_point

        # Add pips for checkers already borne off (0 pips)
        # Sanity check the total number of checkers
        # borne_off = state.count_borne_off(player) # Assuming this method exists
        # total_checkers = num_checkers[player] + borne_off
        # if total_checkers != pyspiel.long_narde.kNumCheckersPerPlayer:
        #      print(f"Warning: Player {player} has {total_checkers} checkers, expected {pyspiel.long_narde.kNumCheckersPerPlayer}")
        # Let's assume count_total_checkers includes borne off checkers implicitly or is not needed here

    return pip_counts

# Helper function (could be shared in a utils module)
def _load_config_from_file(config_path: str) -> dict:
    """Loads configuration from a YAML or JSON file."""
    if not config_path or not os.path.exists(config_path):
        return {}
    try:
        with open(config_path, 'r') as f:
            if config_path.lower().endswith('.yaml') or config_path.lower().endswith('.yml'):
                return yaml.safe_load(f) or {}
            elif config_path.lower().endswith('.json'):
                return json.load(f) or {}
            else:
                print(f"Warning: Unknown config file format for '{config_path}'. Only .yaml, .yml, or .json are supported. Ignoring.", file=sys.stderr)
                return {}
    except Exception as e:
        print(f"Warning: Error loading config file '{config_path}': {e}. Ignoring.", file=sys.stderr)
        return {}

# Helper function (could be shared in a utils module)
def _apply_config_to_flags(config: dict, flags_obj):
    """Applies loaded configuration values to flags, respecting CLI precedence.

    Iterates through the loaded config dictionary. For each key that corresponds
    to a defined flag, if that flag was *not* set explicitly on the command line,
    its value is updated from the config file.

    Args:
        config: The dictionary loaded from the config file.
        flags_obj: The Abseil FLAGS object.
    """
    print("Applying config values to flags (respecting CLI precedence)...")
    flags_dict = flags_obj.flag_values_dict()
    applied_count = 0
    ignored_count = 0

    for key, value in config.items():
        if key in flags_dict:
            flag = flags_obj[key]
            # Check if the flag was set from the command line or code
            is_multi_string = isinstance(flag.value, list) and flag.value != flag.default

            if not flag.present and not is_multi_string:
                try:
                    if isinstance(flags_obj.find_flag_values_object(key).value, list) and isinstance(value, list):
                        print(f"  Overriding multi_string '{key}' from config: {value}")
                        flags_obj.__setattr__(key, value)
                        applied_count += 1
                    elif not isinstance(flags_obj.find_flag_values_object(key).value, list):
                        print(f"  Setting flag '{key}' from config: {value}")
                        flags_obj.__setattr__(key, value)
                        applied_count += 1
                    else:
                        print(f"  Ignoring config value for multi_string '{key}' because config value is not a list.")
                        ignored_count += 1
                except Exception as e:
                    print(f"  Warning: Failed to set flag '{key}' from config value '{value}'. Error: {e}", file=sys.stderr)
                    ignored_count += 1
            elif flag.present:
                print(f"  Flag '{key}' was set via command line. Ignoring config value '{value}'.")
                ignored_count += 1
            elif is_multi_string:
                 print(f"  Flag '{key}' is a multi_string set by default/code. Ignoring config value '{value}'.")
                 ignored_count += 1
        else:
            print(f"  Warning: Key '{key}' from config file does not match any defined flag. Ignoring.", file=sys.stderr)
            ignored_count += 1
    print(f"Config application complete. Applied: {applied_count}, Ignored/Skipped: {ignored_count}")

def main(_):
    """Main evaluation function."""

    # --- Load Config File ---
    config_from_file = {}
    if FLAGS.config_file:
        print(f"Loading configuration from: {FLAGS.config_file}")
        config_from_file = _load_config_from_file(FLAGS.config_file)
        print(f"  Config loaded: {config_from_file}")

    # --- Apply Config to Flags ---
    _apply_config_to_flags(config_from_file, FLAGS)

    # --- Setup TensorBoard Writer ---
    tb_writer = None
    if FLAGS.tensorboard_logdir:
        # Append "_eval" to distinguish from training logs if using the same parent dir
        eval_logdir = os.path.join(FLAGS.tensorboard_logdir, "eval")
        try:
            tb_writer = SummaryWriter(log_dir=eval_logdir)
            print(f"TensorBoard logging for evaluation enabled. Log directory: {eval_logdir}")
        except Exception as e:
            print(f"Warning: Could not initialize TensorBoard SummaryWriter at '{eval_logdir}'. Error: {e}", file=sys.stderr)
            tb_writer = None

    print(f"Starting evaluation for {FLAGS.game_name}...")
    # Print effective flag values *after* potential override
    print(f"  Primary Agent Path: {FLAGS.agent_path}")
    print(f"  Opponent Type: {FLAGS.opponent_type}")
    if FLAGS.opponent_type == 'trained':
        if not FLAGS.opponent_path:
            raise ValueError("opponent_path must be specified when opponent_type is 'trained'")
        print(f"  Opponent Agent Path: {FLAGS.opponent_path}")
    print(f"  Num Episodes: {FLAGS.num_eval_episodes}")
    print(f"  Output File: {FLAGS.output_file}")

    # 1. Load game
    print(f"Loading game: {FLAGS.game_name}...")
    game = pyspiel.load_game(FLAGS.game_name)
    print("Game loaded.")

    # 2. Create environment
    print("Creating environment...")
    env = rl_environment.Environment(game)
    num_players = env.num_players
    if num_players != 2:
        # For now, assume 2 players for agent vs opponent setup
        print(f"Warning: This script currently assumes 2 players, but game has {num_players}. Adapting might be needed.", file=sys.stderr)

    num_actions = env.action_spec()["num_actions"]
    print(f"Environment created: num_players={num_players}, num_actions={num_actions}")

    # 3. Load primary agent (assuming player 0) using the universal loader
    print(f"Loading primary agent (Player 0) from: {FLAGS.agent_path}...")
    try:
        # load_agent needs the path to the *directory* containing metadata.yaml etc.
        primary_agent = agent_serialization.load_agent(FLAGS.agent_path, env=env)
        # Verify player ID matches expectation (optional but good practice)
        if primary_agent.player_id != 0:
             print(f"Warning: Loaded primary agent has player_id {primary_agent.player_id}, expected 0.", file=sys.stderr)
             # Re-assign player_id if necessary? Or handle based on loaded ID?
             # For now, we'll proceed assuming the ID in the object is correct.

        print(f"Primary agent loaded (Type: {type(primary_agent).__name__}).")
    except Exception as e:
        print(f"Error loading primary agent from {FLAGS.agent_path}: {e}", file=sys.stderr)
        return # Exit if primary agent fails to load

    # 4. Create/load opponent agent (assuming player 1)
    print(f"Creating opponent agent (Player 1, type: {FLAGS.opponent_type})...")
    opponent_agent = None
    opponent_player_id = 1 # Assume opponent is player 1

    if FLAGS.opponent_type == 'random':
        # Task 24: Instantiate RandomAgent
        opponent_agent = random_agent.RandomAgent(player_id=opponent_player_id, num_actions=num_actions)
        print("Instantiated RandomAgent for opponent.")
    elif FLAGS.opponent_type == 'trained':
        # Task 25: Load trained agent
        if not FLAGS.opponent_path:
            raise ValueError("opponent_path is required for opponent_type='trained'")
        print(f"Loading trained opponent from: {FLAGS.opponent_path}")
        try:
            opponent_agent = agent_serialization.load_agent(FLAGS.opponent_path, env=env)
            # Verify player ID
            if opponent_agent.player_id != opponent_player_id:
                 print(f"Warning: Loaded opponent agent has player_id {opponent_agent.player_id}, expected {opponent_player_id}.", file=sys.stderr)
                 # Assign expected player_id if needed by the evaluation loop structure
                 # opponent_agent.player_id = opponent_player_id
            print(f"Trained opponent agent loaded (Type: {type(opponent_agent).__name__}).")
        except Exception as e:
            print(f"Error loading trained opponent from {FLAGS.opponent_path}: {e}", file=sys.stderr)
            return # Exit if opponent fails to load
    else:
        raise ValueError(f"Unsupported opponent_type: {FLAGS.opponent_type}")

    print("Opponent agent created/loaded.")

    agents = [None] * num_players
    if primary_agent.player_id < num_players:
        agents[primary_agent.player_id] = primary_agent
    else:
         raise ValueError(f"Primary agent player ID {primary_agent.player_id} is out of bounds for {num_players} players.")
    if opponent_agent.player_id < num_players:
        agents[opponent_agent.player_id] = opponent_agent
    else:
        raise ValueError(f"Opponent agent player ID {opponent_agent.player_id} is out of bounds for {num_players} players.")
        
    if not agents[0] or not agents[1]:
        raise ValueError("Failed to assign agents to player IDs 0 and 1.")


    # --- Initialize Metrics and Elo --- (Task 14)
    print(f"Starting evaluation loop for {FLAGS.num_eval_episodes} episodes...")
    total_rewards = {p: 0 for p in range(num_players)}
    wins = {p: 0 for p in range(num_players)}
    ties = 0
    losses = {p: 0 for p in range(num_players)} # Add losses tracking
    # Initialize Elo ratings
    elo_ratings = [FLAGS.initial_elo] * num_players
    print(f"Initial Elo ratings: {elo_ratings}")
    # Initialize game-specific metrics storage (Task 15)
    pip_differences = []

    # 5. Run evaluation loop
    for ep in range(FLAGS.num_eval_episodes):
        # Placeholder for playing one episode
        time_step = env.reset()
        while not time_step.last():
            player_id = time_step.observations["current_player"]
            agent = agents[player_id]
            if agent is None:
                 raise ValueError(f"Agent for player {player_id} is None.")
                 
            # Task 27: Use is_evaluation=True in agent.step()
            agent_output = agent.step(time_step, is_evaluation=True)
            action_list = [agent_output.action]
            time_step = env.step(action_list)
        
        # Episode finished, record results
        # Task 28: Track evaluation metrics accurately
        episode_returns = time_step.rewards
        is_tie = all(r == 0 for r in episode_returns) # Check for tie first

        if is_tie:
            ties += 1
            # Elo update for tie
            score1 = 0.5
        else:
            # Determine winner/loser for Elo
            if episode_returns[primary_agent.player_id] > 0:
                wins[primary_agent.player_id] += 1
                losses[opponent_agent.player_id] += 1
                score1 = 1.0 # Primary agent won
            elif episode_returns[opponent_agent.player_id] > 0:
                wins[opponent_agent.player_id] += 1
                losses[primary_agent.player_id] += 1
                score1 = 0.0 # Primary agent lost
            else:
                # This case might occur in non-zero-sum or complex reward scenarios
                # For standard win/loss/tie, this shouldn't happen if not a tie.
                print(f"Warning: Non-tie episode resulted in zero reward for both players? Rewards: {episode_returns}")
                # How to handle Elo? Treat as tie? Skip update? For now, treat as tie:
                ties += 1 # Or handle differently
                score1 = 0.5

        # Update Elo Ratings (Task 14)
        p0_id = primary_agent.player_id
        p1_id = opponent_agent.player_id
        elo_ratings[p0_id], elo_ratings[p1_id] = update_elo(
            elo_ratings[p0_id],
            elo_ratings[p1_id],
            score1, # Score from primary agent's perspective
            FLAGS.elo_k_factor
        )

        # Accumulate rewards regardless of win/loss/tie for average calculation
        for p in range(num_players):
            total_rewards[p] += episode_returns[p] # Accumulate reward here

        # Calculate and store game-specific metrics if applicable (Task 15)
        if FLAGS.game_name == "long_narde":
             try:
                 # Get the final raw game state
                 final_state = env.get_state
                 if isinstance(final_state, pyspiel.LongNardeState):
                     # Calculate pip counts
                     pip_counts = calculate_pip_counts(final_state)
                     # Store pip difference (P0 - P1)
                     pip_differences.append(pip_counts[0] - pip_counts[1])
                 else:
                     print(f"Warning: Expected LongNardeState but got {type(final_state)}. Skipping pip count.", file=sys.stderr)
             except Exception as e:
                 print(f"Warning: Failed to calculate pip count for episode {ep+1}. Error: {e}", file=sys.stderr)

        if (ep + 1) % 10 == 0 or ep == FLAGS.num_eval_episodes - 1:
            print(f"Episode {ep + 1}/{FLAGS.num_eval_episodes} completed...")

    print("Evaluation loop finished.")

    # 6. Calculate and Print Metrics (Refined for Task 28)
    print("\n--- Evaluation Results ---")
    print(f"Episodes run: {FLAGS.num_eval_episodes}")
    avg_returns = {p: total_rewards[p] / FLAGS.num_eval_episodes if FLAGS.num_eval_episodes > 0 else 0 for p in range(num_players)}
    print(f"Average returns per episode: {avg_returns}")
    print(f"Total wins: {wins}")
    print(f"Total losses: {losses}") # Print losses
    print(f"Total ties: {ties}")
    # Verify counts add up (optional sanity check)
    # Check if wins[0] + wins[1] + ties == num_episodes for 2-player zero-sum games
    # This check is simplified for common 2-player zero-sum scenarios
    if num_players == 2 and wins.get(0,0) + wins.get(1,0) + ties != FLAGS.num_eval_episodes:
        print(f"Warning: Outcome count mismatch. Wins P0={wins.get(0,0)}, Wins P1={wins.get(1,0)}, Ties={ties}. Total recorded outcomes: {wins.get(0,0) + wins.get(1,0) + ties}. Expected episodes: {FLAGS.num_eval_episodes}")

    # Task 29: Write metrics to file
    print(f"\nWriting metrics to {FLAGS.output_file}...")
    try:
        with open(FLAGS.output_file, 'w', newline='') as csvfile:
            fieldnames = ['metric'] + [f'player_{p}' for p in range(num_players)] + ['total']
            # Add game-specific fields if needed (Task 15)
            if FLAGS.game_name == "long_narde":
                 fieldnames.append('avg_pip_difference') # Add new field name

            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            
            # Write average returns
            avg_return_row = {'metric': 'average_return'}
            for p in range(num_players):
                avg_return_row[f'player_{p}'] = avg_returns.get(p, 0)
            avg_return_row['total'] = 'N/A' # Total avg return doesn't make much sense
            writer.writerow(avg_return_row)

            # Write wins
            win_row = {'metric': 'wins'}
            for p in range(num_players):
                win_row[f'player_{p}'] = wins.get(p, 0)
            win_row['total'] = sum(wins.values()) # Sum of wins across players
            writer.writerow(win_row)

            # Write losses
            loss_row = {'metric': 'losses'}
            for p in range(num_players):
                loss_row[f'player_{p}'] = losses.get(p, 0)
            loss_row['total'] = sum(losses.values()) # Sum of losses across players
            writer.writerow(loss_row)

            # Write ties
            tie_row = {'metric': 'ties'}
            for p in range(num_players):
                tie_row[f'player_{p}'] = 'N/A' # Ties are usually game-wide
            tie_row['total'] = ties
            writer.writerow(tie_row)

            # Write Final Elo Ratings (Task 14)
            elo_row = {'metric': 'final_elo_rating'}
            for p in range(num_players):
                elo_row[f'player_{p}'] = elo_ratings[p]
            # Fill remaining columns with N/A or specific value
            elo_row.setdefault('total', 'N/A')
            if FLAGS.game_name == "long_narde":
                elo_row.setdefault('avg_pip_difference', 'N/A')
            writer.writerow(elo_row)

            # Write Game-Specific Metrics (Task 15)
            if FLAGS.game_name == "long_narde":
                avg_pip_diff = sum(pip_differences) / len(pip_differences) if pip_differences else 0
                pip_diff_row = {'metric': 'avg_pip_difference_p0_minus_p1'}
                # Fill player/total columns appropriately
                for p in range(num_players):
                    pip_diff_row[f'player_{p}'] = 'N/A'
                pip_diff_row['total'] = 'N/A' # Or maybe store the raw average here?
                pip_diff_row['avg_pip_difference'] = avg_pip_diff # Store in the dedicated column
                writer.writerow(pip_diff_row)

        print(f"Metrics successfully written to {FLAGS.output_file}")
    except IOError as e:
        print(f"Error writing metrics to file: {e}", file=sys.stderr)

    # Log final evaluation metrics to TensorBoard if enabled
    if tb_writer:
        print("Logging summary metrics to TensorBoard...")
        # Log average returns
        for p, avg_ret in avg_returns.items():
            tb_writer.add_scalar(f'Evaluation/AverageReturn_P{p}', avg_ret, FLAGS.num_eval_episodes)
        # Log win/loss/tie percentages (more informative than counts)
        if FLAGS.num_eval_episodes > 0:
            for p in range(num_players):
                win_rate = wins.get(p, 0) / FLAGS.num_eval_episodes
                loss_rate = losses.get(p, 0) / FLAGS.num_eval_episodes
                tb_writer.add_scalar(f'Evaluation/WinRate_P{p}', win_rate, FLAGS.num_eval_episodes)
                tb_writer.add_scalar(f'Evaluation/LossRate_P{p}', loss_rate, FLAGS.num_eval_episodes)
            tie_rate = ties / FLAGS.num_eval_episodes
            tb_writer.add_scalar('Evaluation/TieRate', tie_rate, FLAGS.num_eval_episodes)
            # Log Elo Rating over episodes (Task 14)
            for p, elo in enumerate(elo_ratings):
                tb_writer.add_scalar(f'Evaluation/EloRating_P{p}', elo, FLAGS.num_eval_episodes) # Log final Elo
            # To log Elo per episode, add inside the loop:
            # if tb_writer:
            #     for p, elo in enumerate(elo_ratings):
            #         tb_writer.add_scalar(f'Elo/Player_{p}', elo, ep + 1)
            # Log game-specific metrics (Task 15)
            if FLAGS.game_name == "long_narde" and pip_differences:
                avg_pip_diff = sum(pip_differences) / len(pip_differences)
                tb_writer.add_scalar('Evaluation/AvgPipDifference_P0_minus_P1', avg_pip_diff, FLAGS.num_eval_episodes)
            print("Finished logging to TensorBoard.")
        else:
             # Handle case where num_eval_episodes is 0 (to avoid division by zero)
             # Log Elo even if no episodes run (shows initial Elo)
             for p, elo in enumerate(elo_ratings):
                 tb_writer.add_scalar(f'Evaluation/EloRating_P{p}', elo, 0)
             print("Logged initial Elo ratings to TensorBoard (0 episodes run).")

    # Close the TensorBoard writer
    if tb_writer:
        tb_writer.close()

if __name__ == "__main__":
    app.run(main) 