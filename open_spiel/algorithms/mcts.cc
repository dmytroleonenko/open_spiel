// Copyright 2021 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "open_spiel/algorithms/mcts.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <random>
#include <vector>

#include "open_spiel/abseil-cpp/absl/algorithm/container.h"
#include "open_spiel/abseil-cpp/absl/random/distributions.h"
#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/abseil-cpp/absl/strings/str_format.h"
#include "open_spiel/abseil-cpp/absl/time/clock.h"
#include "open_spiel/abseil-cpp/absl/time/time.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_globals.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace algorithms {

int MIN_GC_LIMIT = 5;

int MemoryUsedMb(int nodes) {
  return nodes * sizeof(SearchNode) / (1 << 20);
}

std::vector<double> RandomRolloutEvaluator::Evaluate(const State& state) {
  std::vector<double> result;
  for (int i = 0; i < n_rollouts_; ++i) {
    std::unique_ptr<State> working_state = state.Clone();
    while (!working_state->IsTerminal()) {
      if (working_state->IsChanceNode()) {
        ActionsAndProbs outcomes = working_state->ChanceOutcomes();
        working_state->ApplyAction(SampleAction(outcomes, rng_).first);
      } else {
        std::vector<Action> actions = working_state->LegalActions();
        working_state->ApplyAction(
            actions[absl::Uniform(rng_, 0u, actions.size())]);
      }
    }

    std::vector<double> returns = working_state->Returns();
    if (result.empty()) {
      result.swap(returns);
    } else {
      SPIEL_CHECK_EQ(returns.size(), result.size());
      for (int i = 0; i < result.size(); ++i) {
        result[i] += returns[i];
      }
    }
  }
  for (int i = 0; i < result.size(); ++i) {
    result[i] /= n_rollouts_;
  }
  return result;
}

ActionsAndProbs RandomRolloutEvaluator::Prior(const State& state) {
  // Returns equal probability for all actions.
  if (state.IsChanceNode()) {
    return state.ChanceOutcomes();
  } else {
    std::vector<Action> legal_actions = state.LegalActions();
    ActionsAndProbs prior;
    prior.reserve(legal_actions.size());
    for (const Action& action : legal_actions) {
      prior.emplace_back(action, 1.0 / legal_actions.size());
    }
    return prior;
  }
}

// UCT value of given child
double SearchNode::UCTValue(int parent_explore_count, double uct_c) const {
  if (!outcome.empty()) {
    return outcome[player];
  }

  if (explore_count == 0) return std::numeric_limits<double>::infinity();

  // The "greedy-value" of choosing a given child is always with respect to
  // the current player for this node.
  return total_reward / explore_count +
         uct_c * std::sqrt(std::log(parent_explore_count) / explore_count);
}

double SearchNode::PUCTValue(int parent_explore_count, double uct_c) const {
  // Returns the PUCT value of this node.
  if (!outcome.empty()) {
    return outcome[player];
  }

  return ((explore_count != 0 ? total_reward / explore_count : 0) +
          uct_c * prior * std::sqrt(parent_explore_count) /
              (explore_count + 1));
}

bool SearchNode::CompareFinal(const SearchNode& b) const {
  double out = (outcome.empty() ? 0 : outcome[player]);
  double out_b = (b.outcome.empty() ? 0 : b.outcome[b.player]);
  if (out != out_b) {
    return out < out_b;
  }
  if (explore_count != b.explore_count) {
    return explore_count < b.explore_count;
  }
  return total_reward < b.total_reward;
}

const SearchNode& SearchNode::BestChild() const {
  // Returns the best action from this node, either proven or most visited.
  //
  // This ordering leads to choosing:
  // - Highest proven score > 0 over anything else, including a promising but
  //   unproven action.
  // - A proven draw only if it has higher exploration than others that are
  //   uncertain, or the others are losses.
  // - Uncertain action with most exploration over loss of any difficulty
  // - Hardest loss if everything is a loss
  // - Highest expected reward if explore counts are equal (unlikely).
  // - Longest win, if multiple are proven (unlikely due to early stopping).
  return *std::max_element(children.begin(), children.end(),
                           [](const SearchNode& a, const SearchNode& b) {
                             return a.CompareFinal(b);
                           });
}

std::string SearchNode::ChildrenStr(const State& state) const {
  std::string out;
  if (!children.empty()) {
    std::vector<const SearchNode*> refs;  // Sort a list of refs, not a copy.
    refs.reserve(children.size());
    for (const SearchNode& child : children) {
      refs.push_back(&child);
    }
    std::sort(refs.begin(), refs.end(),
              [](const SearchNode* a, const SearchNode* b) {
                return b->CompareFinal(*a);
              });
    for (const SearchNode* child : refs) {
      absl::StrAppend(&out, child->ToString(state), "\n");
    }
  }
  return out;
}

std::string SearchNode::ToString(const State& state) const {
  return absl::StrFormat(
      "%6s: player: %d, prior: %5.3f, value: %6.3f, sims: %5d, outcome: %s, "
      "%3d children",
      (action != kInvalidAction ? state.ActionToString(player, action)
                                : "none"),
      player, prior, (explore_count ? total_reward / explore_count : 0.),
      explore_count,
      (outcome.empty()
           ? "none"
           : absl::StrFormat("%4.1f",
                             outcome[player == kChancePlayerId ? 0 : player])),
      children.size());
}

Action SearchNode::SampleFromPrior(const State& state,
                                   Evaluator* evaluator,
                                   std::mt19937* rng) const {
  std::unique_ptr<State> working_state = state.Clone();
  ActionsAndProbs prior = evaluator->Prior(*working_state);
  Action chosen_action = SampleAction(prior, *rng).first;
  return chosen_action;
}

std::vector<double> dirichlet_noise(int count, double alpha,
                                    std::mt19937* rng) {
  std::vector<double> noise;
  noise.reserve(count);

  std::gamma_distribution<double> gamma(alpha, 1.0);
  for (int i = 0; i < count; ++i) {
    noise.emplace_back(gamma(*rng));
  }

  double sum = absl::c_accumulate(noise, 0.0);
  for (double& v : noise) {
    v /= sum;
  }
  return noise;
}

MCTSBot::MCTSBot(const Game& game, std::shared_ptr<Evaluator> evaluator,
                 double uct_c, int max_simulations, int64_t max_memory_mb,
                 bool solve, int seed, bool verbose,
                 ChildSelectionPolicy child_selection_policy,
                 double dirichlet_alpha, double dirichlet_epsilon,
                 bool dont_return_chance_node)
    : uct_c_{uct_c},
      max_simulations_{max_simulations},
      max_nodes_((max_memory_mb << 20) / sizeof(SearchNode) + 1),
      nodes_(0),
      gc_limit_(MIN_GC_LIMIT),
      verbose_(verbose),
      solve_(solve),
      max_utility_(game.MaxUtility()),
      dirichlet_alpha_(dirichlet_alpha),
      dirichlet_epsilon_(dirichlet_epsilon),
      dont_return_chance_node_(dont_return_chance_node),
      rng_(seed),
      child_selection_policy_(child_selection_policy),
      evaluator_(evaluator) {
  GameType game_type = game.GetType();
  if (game_type.reward_model != GameType::RewardModel::kTerminal)
    SpielFatalError("Game must have terminal rewards.");
  if (game_type.dynamics != GameType::Dynamics::kSequential)
    SpielFatalError("Game must have sequential turns.");
}

Action MCTSBot::Step(const State& state) {
  absl::Time start = absl::Now();
  MCTSearch(state); // New way: MCTSearch updates this->root_
  // Now, this->root_ is the root of the search tree for 'state'.

  Action action;
  if (this->root_ == nullptr || max_simulations_ <= 1) { // Check this->root_ for safety
    // sample from prior
    if (this->root_) { // Ensure root_ is valid before sampling
      action = this->root_->SampleFromPrior(state, evaluator_.get(), &rng_);
    } else {
      // Fallback if root_ is somehow null: sample from raw prior of the state
      // This case should ideally not be hit if MCTSearch correctly initializes root_.
      ActionsAndProbs prior = evaluator_->Prior(state);
      if (prior.empty()) SpielFatalError("Cannot sample from empty prior after MCTSearch failed to set root.");
      // Use the SampleAction(const ActionsAndProbs& outcomes, double random_value) overload
      double random_sample = absl::uniform_real_distribution<double>(0.0, 1.0)(rng_);
      action = open_spiel::SampleAction(prior, random_sample).first;
    }
  } else {
    // return best action
    const SearchNode& best = this->root_->BestChild(); // New way
    if (verbose_) {
      double seconds = absl::ToDoubleSeconds(absl::Now() - start);
      std::cerr << absl::StrFormat(
                       ("Finished %d sims in %.3f secs, %.1f sims/s, "
                        "tree size: %d nodes / %d mb."),
                       this->root_->explore_count, seconds,
                       (this->root_->explore_count / seconds), nodes_,
                       MemoryUsedMb(nodes_))
                << std::endl;
      std::cerr << "Root:" << std::endl;
      std::cerr << this->root_->ToString(state) << std::endl;
      std::cerr << "Children:" << std::endl;
      std::cerr << this->root_->ChildrenStr(state) << std::endl;
      if (!best.children.empty()) {
        std::unique_ptr<State> chosen_state = state.Clone();
        chosen_state->ApplyAction(best.action);
        std::cerr << "Children of chosen:" << std::endl;
        std::cerr << best.ChildrenStr(*chosen_state) << std::endl;
      }
    }
    action = best.action;
  }

  // Persist the tree and state
  // root_ = std::move(local_root); // Old way: local_root was the result of old MCTSearch
                                // New way: this->root_ is already set by the new MCTSearch for 'state'.
                                // Now we need to update it for the state *after* 'action'.
  
  // root_state_ is currently equivalent to 'state' because MCTSearch updated it.
  // SPIEL_CHECK_TRUE(root_state_ != nullptr && *root_state_ == state);

  // Apply the chosen action to our persistent root_state_
  // This was previously done implicitly when local_root was moved and then its state cloned.
  // Now root_state_ is already state, so we just advance it.
  // SPIEL_CHECK_TRUE(root_ != nullptr); // Should be guaranteed by MCTSearch
  // root_state_ = state.Clone(); // This was the old logic line, not quite right for persistence
  // The root_state_ should already be current_real_state from MCTSearch call.
  // Now, advance it by the chosen action.
  if(root_state_) { // Ensure root_state_ is valid
    root_state_->ApplyAction(action);
  } else {
    // This should not happen if MCTSearch works correctly.
    // If it does, we need a valid root_state_ to proceed.
    // For robustness, clone and apply, but this indicates a logic flaw earlier.
    root_state_ = state.Clone();
    root_state_->ApplyAction(action);
    // And ensure root_ matches this new state by resetting if it doesn't (though it should be post-MCTSearch)
    if(root_ && root_->action != action) { // A loose check. Proper check would be state comparison.
        // If root_ is not already representing the state *after* the action (which it shouldn't be yet),
        // or if the root_ is for a different path, then clear its children and update action property.
        // The current promotion logic below handles this.
    }
  }
  

  // Promote the actual chosen child to be the new root_ for the next turn's search
  bool found_child_to_promote = false;
  if (this->root_) { // Check if root_ is valid before accessing children
    std::vector<SearchNode> potential_children = std::move(this->root_->children);
    this->root_->children.clear(); // Clear children of the current root as we are selecting one to become the new root

    for (SearchNode& child_candidate : potential_children) {
      if (child_candidate.action == action) {
        // This child_candidate is the one that corresponds to the chosen action.
        // We want its properties and its children to become the new root.
        this->root_->action = child_candidate.action;       // Action that *led* to this new state.
        this->root_->player = child_candidate.player;       // Player who *made* that action.
        this->root_->prior = child_candidate.prior;
        this->root_->explore_count = child_candidate.explore_count;
        this->root_->total_reward = child_candidate.total_reward;
        this->root_->outcome = std::move(child_candidate.outcome); // Move outcome vector
        this->root_->children = std::move(child_candidate.children); // Crucially, move its children

        found_child_to_promote = true;
        break;
      }
    }
  } else {
    // If this->root_ is null, then found_child_to_promote remains false.
    // This can happen if MCTSearch had an issue or if it was the very first step and sampling from prior occurred
    // before root_ was fully established by the main logic. The !found_child_to_promote block will handle it.
  }

  if (!found_child_to_promote) {
    // The action taken was not among the direct children explored by the MCTSearch from the current state,
    // or root_ was null to begin with.
    // In this case, the new root_ (which represents the state *after* the action)
    // will not have a pre-populated subtree from the previous search pass.
    // Its own properties (action, player, prior) should reflect that it's a new frontier.

    // If root_ is null, create it. This is a safety net.
    if (!this->root_) {
        // We need a current player for the new node. If root_state_ is valid, use its player.
        // Otherwise, use state.CurrentPlayer() (player who made the action to get to 'state').
        Player player_for_new_node = root_state_ ? root_state_->CurrentPlayer() : state.CurrentPlayer();
        this->root_ = std::make_unique<SearchNode>(action, player_for_new_node, 0.0);
    } else {
        // root_ exists, just update its properties.
        this->root_->action = action; // The action that led to this state
    }

    // Ensure root_state_ is valid and represents the state *after* the action.
    // If root_state_ was not advanced before, or was null, fix it.
    if (!root_state_ || (root_state_ && !(*root_state_ == state) && !state.History().empty() && 
                         (state.History().back() != action))) { // Assuming .back() is an Action
        // This is complex. If root_state_ is stale or represents something before 'action',
        // we need to ensure it's correct. A simple clone and apply is safest if unsure.
        // The existing root_state_->ApplyAction(action) should have handled it if root_state_ was initially 'state'.
        // This block is more of a safeguard or handles cases where root_state_ was null.
        // If MCTSearch set root_state_ to 'state', and then we did root_state_->ApplyAction(action), it should be fine.
        // The main concern is if root_state_ was null initially for this block.
        if(!root_state_) root_state_ = state.Clone();
        if(root_state_->History().empty() || 
           (root_state_->History().back() != action)) { // Assuming .back() is an Action
            // If the last action isn't what we just took, state needs to be properly set.
            // This indicates root_state_ was not properly managed before this block.
            // For safety, let's assume state represents the state *before* action, and root_state_ needs to be after.
            root_state_ = state.Clone();
            root_state_->ApplyAction(action);
        }
    }
    
    // Player whose turn it is AT the new root_state_
    // If root_state_ is null, this would crash. Need to ensure root_state_ is valid.
    this->root_->player = root_state_ ? root_state_->CurrentPlayer() : state.CurrentPlayer(); 
    this->root_->prior = 0; // Prior is unknown without re-evaluation for this specific new root.
    this->root_->explore_count = 0;
    this->root_->total_reward = 0;
    this->root_->outcome.clear();
    this->root_->children.clear(); // Ensure children are empty for a new frontier node.
  }

  return action;
}

std::pair<ActionsAndProbs, Action> MCTSBot::StepWithPolicy(const State& state) {
  Action action = Step(state);
  return {{{action, 1.}}, action};
}

std::unique_ptr<State> MCTSBot::ApplyTreePolicy(
    SearchNode* root, const State& state,
    std::vector<SearchNode*>* visit_path) {
  visit_path->push_back(root);
  std::unique_ptr<State> working_state = state.Clone();
  SearchNode* current_node = root;
  while ((!working_state->IsTerminal() && current_node->explore_count > 0) ||
         (working_state->IsChanceNode() && dont_return_chance_node_)) {
    if (current_node->children.empty()) {
      // For a new node, initialize its state, then choose a child as normal.
      ActionsAndProbs legal_actions = evaluator_->Prior(*working_state);
      if (current_node == root && dirichlet_alpha_ > 0) {
        std::vector<double> noise =
            dirichlet_noise(legal_actions.size(), dirichlet_alpha_, &rng_);
        for (int i = 0; i < legal_actions.size(); i++) {
          legal_actions[i].second =
              (1 - dirichlet_epsilon_) * legal_actions[i].second +
              dirichlet_epsilon_ * noise[i];
        }
      }
      // Reduce bias from move generation order.
      std::shuffle(legal_actions.begin(), legal_actions.end(), rng_);
      Player player = working_state->CurrentPlayer();
      current_node->children.reserve(legal_actions.size());
      for (auto [action, prior] : legal_actions) {
        current_node->children.emplace_back(action, player, prior);
      }
      nodes_ += current_node->children.capacity();
    }

    Action selected_action;
    if (current_node->children.empty()) {
      // no children, sample from prior
      selected_action = current_node->SampleFromPrior(state, evaluator_.get(),
                                                      &rng_);
    } else {
      // look at children
      SearchNode* chosen_child = nullptr;
      if (working_state->IsChanceNode()) {
        // For chance nodes, rollout according to chance node's probability
        // distribution
        Action chosen_action =
            SampleAction(working_state->ChanceOutcomes(), rng_).first;

        for (SearchNode& child : current_node->children) {
          if (child.action == chosen_action) {
            chosen_child = &child;
            break;
          }
        }
      } else {
        // Otherwise choose node with largest UCT value.
        double max_value = -std::numeric_limits<double>::infinity();
        for (SearchNode& child : current_node->children) {
          double val;
          switch (child_selection_policy_) {
            case ChildSelectionPolicy::UCT:
              val = child.UCTValue(current_node->explore_count, uct_c_);
              break;
            case ChildSelectionPolicy::PUCT:
              val = child.PUCTValue(current_node->explore_count, uct_c_);
              break;
          }
          if (val > max_value) {
            max_value = val;
            chosen_child = &child;
          }
        }
      }
      selected_action = chosen_child->action;
      current_node = chosen_child;
    }

    working_state->ApplyAction(selected_action);
    visit_path->push_back(current_node);
  }

  return working_state;
}

void MCTSBot::MCTSearch(const State& current_real_state) {
  // Tree Nurturing & Initialization Logic (Tasks 2.1, 4.1, 5.1, 5.2)
  if (root_ == nullptr || root_state_ == nullptr) {
    // First call, or after ResetTree(). Initialize for current_real_state.
    root_ = std::make_unique<SearchNode>(kInvalidAction, current_real_state.CurrentPlayer(), 1.0);
    root_state_ = current_real_state.Clone();
    nodes_ = 1; // Reset node count for the new tree
    gc_limit_ = MIN_GC_LIMIT; // Reset GC limit
  } else if (*(root_state_) == current_real_state) {
    // State matches, root_ and root_state_ are current.
    // MCTSearch will continue to build on this existing root_.
    // nodes_ should reflect the current size of root_ structure if needed for GC,
    // but ApplyTreePolicy increments nodes_ for *new* expansions, which is the primary concern for max_nodes.
    // For now, we can assume nodes_ is implicitly managed correctly if ApplyTreePolicy only adds.
    // If MCTSearch always resets nodes_ = 1 here, it implies simulations are only counted for the current call.
    // Let's reset nodes_ and gc_limit_ similar to how a new tree was created by old MCTSearch.
    // This means 'nodes_' tracks nodes relevant to the current MCTSearch pass (newly explored or part of simulations).
    nodes_ = 1; // Represents the root node itself for this search pass.
                // ApplyTreePolicy will add to this for new nodes.
    gc_limit_ = MIN_GC_LIMIT;
  } else {
    // States do not match. An opponent move or chance event likely occurred.
    // Try to find current_real_state among the children of this->root_
    bool found_matching_child_path = false;
    if (root_ && !root_->children.empty()) { // Check root_ as it might have been reset by a previous failed attempt
      std::vector<SearchNode> potential_next_roots = std::move(root_->children);
      // root_->children is now empty. If promotion fails, new root will have no children from old tree.

      for (SearchNode& child_candidate : potential_next_roots) {
        std::unique_ptr<State> state_after_candidate_action = root_state_->Clone();
        state_after_candidate_action->ApplyAction(child_candidate.action);
        if (*state_after_candidate_action == current_real_state) {
          // Found the path! Promote child_candidate to be the new this->root_
          // Create a new SearchNode and move contents.
          auto new_root_node = std::make_unique<SearchNode>();
          new_root_node->action = child_candidate.action;
          new_root_node->player = child_candidate.player;
          new_root_node->prior = child_candidate.prior;
          new_root_node->explore_count = child_candidate.explore_count;
          new_root_node->total_reward = child_candidate.total_reward;
          new_root_node->outcome = std::move(child_candidate.outcome);
          new_root_node->children = std::move(child_candidate.children);
          
          root_ = std::move(new_root_node);
          root_state_ = current_real_state.Clone();
          found_matching_child_path = true;
          // Reset nodes_ and gc_limit_ for the new search starting from this promoted root
          nodes_ = 1; // Or count nodes in promoted subtree: CountNodes(root_.get());
          gc_limit_ = MIN_GC_LIMIT;
          break;
        }
      }
    }

    if (!found_matching_child_path) {
      // Cannot find current_real_state in children. Discard old tree.
      this->ResetTree(); // Clears root_ and root_state_
      root_ = std::make_unique<SearchNode>(kInvalidAction, current_real_state.CurrentPlayer(), 1.0);
      root_state_ = current_real_state.Clone();
      nodes_ = 1;
      gc_limit_ = MIN_GC_LIMIT;
    }
  }

  // Ensure root_ is not null after the above logic
  SPIEL_CHECK_TRUE(root_ != nullptr);
  SPIEL_CHECK_TRUE(root_state_ != nullptr);

  SearchNode* search_root_ptr = root_.get(); // Use the persistent root

  std::vector<SearchNode*> visit_path;
  std::vector<double> returns;
  visit_path.reserve(64);
  for (int i = 0; i < max_simulations_; ++i) {
    visit_path.clear();
    returns.clear();

    std::unique_ptr<State> working_state =
        // ApplyTreePolicy(root.get(), state, &visit_path); // Old way
        ApplyTreePolicy(search_root_ptr, *root_state_, &visit_path); // New way, use *root_state_ as basis for policy application

    bool solved;
    if (working_state->IsTerminal()) {
      returns = working_state->Returns();
      visit_path[visit_path.size() - 1]->outcome = returns;
      solved = solve_;
    } else {
      returns = evaluator_->Evaluate(*working_state);
      solved = false;
    }

    // Propagate values back.
    while (!visit_path.empty()) {
      int decision_node_idx = visit_path.size() - 1;
      SearchNode* node = visit_path[decision_node_idx];

      // If it's a chance node, find the parent player id.
      while (visit_path[decision_node_idx]->player == kChancePlayerId) {
        decision_node_idx--;
      }

      node->total_reward += returns[visit_path[decision_node_idx]->player];
      node->explore_count += 1;
      visit_path.pop_back();

      // Back up solved results as well.
      if (solved && !node->children.empty()) {
        Player player = node->children[0].player;
        if (player == kChancePlayerId) {
          // Only back up chance nodes if all have the same outcome.
          // An alternative would be to back up the weighted average of
          // outcomes if all children are solved, but that is less clear.
          const std::vector<double>& outcome = node->children[0].outcome;
          if (!outcome.empty() &&
              std::all_of(node->children.begin() + 1, node->children.end(),
                          [&outcome](const SearchNode& c) {
                            return c.outcome == outcome;
                          })) {
            node->outcome = outcome;
          } else {
            solved = false;
          }
        } else {
          // If any have max utility (won?), or all children are solved,
          // choose the one best for the player choosing.
          const SearchNode* best = nullptr;
          bool all_solved = true;
          for (const SearchNode& child : node->children) {
            if (child.outcome.empty()) {
              all_solved = false;
            } else if (best == nullptr ||
                       child.outcome[player] > best->outcome[player]) {
              best = &child;
            }
          }
          if (best != nullptr &&
              (all_solved || best->outcome[player] == max_utility_)) {
            node->outcome = best->outcome;
          } else {
            solved = false;
          }
        }
      }
    }

    if (!search_root_ptr->outcome.empty() ||  // Full game tree is solved.
        search_root_ptr->children.size() == 1) {
      break;
    }
    if (max_nodes_ > 1 && nodes_ >= max_nodes_) {
      // Note that actual memory used as counted by ps/top might exceed the
      // counted value here, possibly by a significant margin (1.5x even!). Part
      // of that is not counting the outcome array, but most of that is due to
      // memory fragmentation and is out of our control without writing our own
      // memory manager.
      if (verbose_) {
        std::cerr << absl::StrFormat(
            ("Approx %d mb in %d nodes after %d sims, garbage collecting with "
             "limit %d ... "),
            MemoryUsedMb(nodes_), nodes_, i, gc_limit_);
      }
      GarbageCollect(search_root_ptr);

      // Slowly increase or decrease to target releasing half the memory.
      gc_limit_ *= (nodes_ > max_nodes_ / 2 ? 1.25 : 0.9);
      gc_limit_ = std::max(MIN_GC_LIMIT, gc_limit_);
      if (verbose_) {
        std::cerr << absl::StrFormat(
            "%d mb in %d nodes remaining\n",
            MemoryUsedMb(nodes_), nodes_);
      }
    }
  }
}

void MCTSBot::GarbageCollect(SearchNode* node) {
  if (node->children.empty()) {
    return;
  }
  bool clear_children = node->explore_count < gc_limit_;
  for (SearchNode& child : node->children) {
    GarbageCollect(&child);
  }
  if (clear_children) {
    nodes_ -= node->children.capacity();
    node->children.clear();
    node->children.shrink_to_fit();  // release the memory
  }
}

// FIRST_EDIT: Declare ResetTree to clear the persistent search tree and state
void MCTSBot::ResetTree() {
  root_.reset();
  root_state_.reset();
}

}  // namespace algorithms
}  // namespace open_spiel
