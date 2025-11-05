# Search Algorithms for Vegas Solitaire: Deep Dive

A comprehensive technical reference for implementing various search and optimization algorithms for Vegas Solitaire.

## Table of Contents

1. [Algorithm Overview](#algorithm-overview)
2. [Random Baseline](#random-baseline)
3. [Heuristic Search](#heuristic-search)
4. [Monte Carlo Tree Search (MCTS)](#monte-carlo-tree-search)
5. [Expectimax](#expectimax)
6. [Beam Search](#beam-search)
7. [Minimax & Alpha-Beta](#minimax-and-alpha-beta)
8. [Reinforcement Learning](#reinforcement-learning)
9. [Hybrid Approaches](#hybrid-approaches)

---

## Algorithm Overview

### The Challenge

Vegas Solitaire presents several algorithmic challenges:
- **Stochastic**: Draw order is random
- **Partial observability**: Stock cards are hidden
- **Large branching factor**: 10-30 valid moves per state
- **Long horizon**: 100+ moves per game
- **Sparse rewards**: Win/lose at end, or incremental scoring

### Algorithm Categories

**Search-based methods:**
- Explore game tree systematically
- No learning required
- Computational cost per move

**Learning-based methods:**
- Train from experience
- Fast inference after training
- Requires training data/time

**Hybrid methods:**
- Combine search and learning
- Often achieves best performance
- More complex to implement

---

## Random Baseline

The simplest possible player - choose moves uniformly at random.

### Purpose
- Establish performance floor
- Sanity check for more complex algorithms
- Fast baseline for comparison

### Implementation

```python
class RandomPlayer:
    def __init__(self, seed=None):
        self.rng = random.Random(seed)
    
    def choose_move(self, game_state):
        valid_moves = game_state.get_valid_moves()
        if not valid_moves:
            return None
        return self.rng.choice(valid_moves)
    
    def play_game(self, game):
        while not game.is_over():
            move = self.choose_move(game)
            if move:
                game.make_move(move)
        return game.score
```

### Expected Performance
- Win rate: ~2-8% (estimated)
- Average score: Negative (typically -$20 to -$30)
- Games/second: >10,000

### Use Cases
- Performance baseline
- Rollout policy for MCTS
- Sanity testing game implementation

---

## Heuristic Search

Rule-based decision making using domain knowledge.

### Core Principles

1. **Foundation priority**: Move to foundations when safe
2. **Expose cards**: Prefer moves that reveal face-down cards
3. **Build sequences**: Create long descending sequences
4. **Empty columns**: Maintain empty columns for maneuverability
5. **Avoid blocking**: Don't bury cards you'll need

### Example Heuristic Function

```python
def evaluate_move(move, game_state):
    score = 0
    
    # Foundation moves are usually good
    if move.destination_type == 'foundation':
        score += 10
        
    # Revealing cards is valuable
    if move.reveals_card:
        score += 8
        
    # Creating empty columns
    if move.empties_column:
        score += 15
        
    # Building long sequences
    score += move.sequence_length * 2
    
    # Penalty for blocking useful cards
    if blocks_needed_card(move, game_state):
        score -= 20
    
    # Moving from waste is often good
    if move.source_type == 'waste':
        score += 3
    
    return score

class HeuristicPlayer:
    def choose_move(self, game_state):
        valid_moves = game_state.get_valid_moves()
        if not valid_moves:
            return None
        
        # Score all moves
        scored_moves = [(evaluate_move(m, game_state), m) 
                       for m in valid_moves]
        
        # Return highest scoring move
        return max(scored_moves)[1]
```

### Advanced Heuristics

**Card tracking:**
```python
def track_needed_cards(game_state):
    """Track which cards we still need and where they might be."""
    needed = set()
    for foundation in game_state.foundations:
        next_rank = foundation.top_card.rank + 1
        if next_rank <= 13:
            needed.add((foundation.suit, next_rank))
    return needed
```

**Tableau flexibility:**
```python
def tableau_flexibility(game_state):
    """Measure how many options we have in tableau."""
    score = 0
    empty_columns = sum(1 for col in game_state.tableau if not col)
    score += empty_columns * 10
    
    # Count movable sequences
    for col in game_state.tableau:
        score += count_movable_sequences(col)
    
    return score
```

### Expected Performance
- Win rate: ~10-20% (with good heuristics)
- Average score: Slightly negative to break-even
- Games/second: ~1,000-5,000

### Strengths
- Fast and interpretable
- No training required
- Good for bootstrapping learning methods

### Weaknesses
- Hard to tune optimally
- May miss counter-intuitive strategies
- Doesn't adapt to specific game situations

---

## Monte Carlo Tree Search

Iteratively build search tree by simulating games from promising states.

### Algorithm Overview

MCTS has four phases repeated in each iteration:

1. **Selection**: Traverse tree using selection policy (UCB1)
2. **Expansion**: Add new child node to tree
3. **Simulation**: Play out game randomly from new node
4. **Backpropagation**: Update statistics along path

### Implementation

```python
import math

class MCTSNode:
    def __init__(self, state, parent=None, move=None):
        self.state = state
        self.parent = parent
        self.move = move
        self.children = []
        self.visits = 0
        self.total_reward = 0
        self.untried_moves = state.get_valid_moves()
    
    def ucb1(self, exploration_weight=1.41):
        """Upper Confidence Bound for trees."""
        if self.visits == 0:
            return float('inf')
        
        exploitation = self.total_reward / self.visits
        exploration = exploration_weight * math.sqrt(
            math.log(self.parent.visits) / self.visits
        )
        return exploitation + exploration
    
    def select_child(self):
        """Select child with highest UCB1 value."""
        return max(self.children, key=lambda c: c.ucb1())
    
    def expand(self):
        """Add a new child node."""
        move = self.untried_moves.pop()
        next_state = self.state.apply_move(move)
        child = MCTSNode(next_state, parent=self, move=move)
        self.children.append(child)
        return child
    
    def simulate(self):
        """Random rollout from this state."""
        state = self.state.copy()
        while not state.is_terminal():
            moves = state.get_valid_moves()
            if not moves:
                break
            move = random.choice(moves)
            state.apply_move(move)
        return state.get_reward()
    
    def backpropagate(self, reward):
        """Update statistics up the tree."""
        self.visits += 1
        self.total_reward += reward
        if self.parent:
            self.parent.backpropagate(reward)

class MCTSPlayer:
    def __init__(self, num_simulations=1000):
        self.num_simulations = num_simulations
    
    def choose_move(self, game_state):
        root = MCTSNode(game_state)
        
        for _ in range(self.num_simulations):
            node = root
            
            # Selection
            while node.untried_moves == [] and node.children != []:
                node = node.select_child()
            
            # Expansion
            if node.untried_moves != []:
                node = node.expand()
            
            # Simulation
            reward = node.simulate()
            
            # Backpropagation
            node.backpropagate(reward)
        
        # Return most visited child's move
        best_child = max(root.children, key=lambda c: c.visits)
        return best_child.move
```

### Optimizations

**Transposition tables:**
```python
class MCTSWithCache:
    def __init__(self):
        self.transposition_table = {}
    
    def get_or_create_node(self, state):
        state_hash = hash(state)
        if state_hash not in self.transposition_table:
            self.transposition_table[state_hash] = MCTSNode(state)
        return self.transposition_table[state_hash]
```

**Heuristic rollouts:**
```python
def guided_simulate(self, heuristic_fn):
    """Use heuristic to guide rollout."""
    state = self.state.copy()
    while not state.is_terminal():
        moves = state.get_valid_moves()
        if not moves:
            break
        
        # Weight moves by heuristic
        move = weighted_random_choice(moves, heuristic_fn)
        state.apply_move(move)
    return state.get_reward()
```

**Progressive widening:**
```python
def expand_progressive(self, alpha=1.0, k=1.0):
    """Only expand a limited number of children."""
    max_children = int(k * self.visits ** alpha)
    if len(self.children) < max_children:
        return self.expand()
    return None
```

### Expected Performance
- Win rate: ~25-40% (with sufficient simulations)
- Average score: Positive expected value
- Time per move: 100ms - 10s depending on budget

### Strengths
- No training required
- Anytime algorithm
- Handles stochasticity naturally
- Proven effective for many games

### Weaknesses
- Slow compared to heuristics
- Memory intensive
- Doesn't generalize across games

---

## Expectimax

Optimal search algorithm for stochastic games with chance nodes.

### Algorithm Structure

```
Decision nodes (max): Choose best move
Chance nodes (expected value): Weight by probability
```

### Core Implementation

```python
def expectimax(state, depth, eval_fn):
    """
    Expectimax search with limited depth.
    
    Args:
        state: Current game state
        depth: Remaining search depth
        eval_fn: Heuristic evaluation function
    
    Returns:
        Expected value of this state
    """
    # Base case: terminal state or depth limit
    if state.is_terminal() or depth == 0:
        return eval_fn(state)
    
    # Decision node: maximize over actions
    if state.is_decision_point():
        values = []
        for move in state.get_valid_moves():
            next_state = state.apply_move(move)
            values.append(expectimax(next_state, depth - 1, eval_fn))
        return max(values) if values else eval_fn(state)
    
    # Chance node: compute expected value
    else:
        expected_value = 0
        outcomes = state.get_chance_outcomes()  # [(state, probability)]
        for next_state, probability in outcomes:
            value = expectimax(next_state, depth - 1, eval_fn)
            expected_value += probability * value
        return expected_value

class ExpectimaxPlayer:
    def __init__(self, depth=3, eval_fn=None):
        self.depth = depth
        self.eval_fn = eval_fn or default_eval
    
    def choose_move(self, game_state):
        valid_moves = game_state.get_valid_moves()
        if not valid_moves:
            return None
        
        # Evaluate each move
        move_values = []
        for move in valid_moves:
            next_state = game_state.apply_move(move)
            value = expectimax(next_state, self.depth - 1, self.eval_fn)
            move_values.append((value, move))
        
        # Return best move
        return max(move_values)[1]
```

### Handling Chance Nodes in Solitaire

**Drawing from stock:**
```python
def get_chance_outcomes_for_draw(state):
    """
    Returns all possible draws from stock with probabilities.
    This is expensive - 52! possible orderings!
    """
    unknown_cards = state.get_unknown_stock_cards()
    n = len(unknown_cards)
    
    # If drawing 3 cards, there are many combinations
    outcomes = []
    for card_combo in combinations(unknown_cards, 3):
        prob = 1.0 / comb(n, 3)  # Assuming uniform
        new_state = state.apply_draw(card_combo)
        outcomes.append((new_state, prob))
    
    return outcomes
```

**Simplification strategies:**
```python
def get_sampled_outcomes(state, num_samples=10):
    """Sample possible draws instead of enumerating all."""
    outcomes = []
    for _ in range(num_samples):
        sampled_state = state.sample_draw()
        outcomes.append((sampled_state, 1.0 / num_samples))
    return outcomes
```

### Pruning Strategies

**Alpha-beta for decision nodes:**
```python
def expectimax_with_pruning(state, depth, alpha, beta, eval_fn):
    if state.is_terminal() or depth == 0:
        return eval_fn(state)
    
    if state.is_decision_point():
        value = -infinity
        for move in state.get_valid_moves():
            next_state = state.apply_move(move)
            value = max(value, expectimax_with_pruning(
                next_state, depth-1, alpha, beta, eval_fn))
            alpha = max(alpha, value)
            if beta <= alpha:
                break  # Prune
        return value
    else:
        # Chance node - must evaluate all
        expected = 0
        for next_state, prob in state.get_chance_outcomes():
            expected += prob * expectimax_with_pruning(
                next_state, depth-1, alpha, beta, eval_fn)
        return expected
```

**Low-probability pruning:**
```python
def prune_unlikely_outcomes(outcomes, threshold=0.01):
    """Ignore outcomes with very low probability."""
    significant = [(s, p) for s, p in outcomes if p >= threshold]
    
    # Renormalize probabilities
    total_prob = sum(p for _, p in significant)
    return [(s, p/total_prob) for s, p in significant]
```

### Expected Performance
- Win rate: ~20-35% (limited depth)
- Average score: Moderate positive
- Time per move: 50ms - 5s (depends heavily on depth)

### Strengths
- Theoretically sound for stochastic games
- Provides expected value estimates
- Can be combined with good heuristics

### Weaknesses
- Exponential explosion on chance nodes
- Requires strong evaluation function
- Deep search computationally prohibitive

---

## Beam Search

Keep only the top-k most promising states at each depth level.

### Core Algorithm

```python
class BeamSearch:
    def __init__(self, beam_width=10, max_depth=5, eval_fn=None):
        self.beam_width = beam_width
        self.max_depth = max_depth
        self.eval_fn = eval_fn or default_eval
    
    def search(self, initial_state):
        # Initialize beam with starting state
        beam = [(initial_state, 0, [])]  # (state, score, move_history)
        
        for depth in range(self.max_depth):
            candidates = []
            
            # Expand all states in current beam
            for state, score, history in beam:
                if state.is_terminal():
                    candidates.append((state, score, history))
                    continue
                
                for move in state.get_valid_moves():
                    next_state = state.apply_move(move)
                    new_score = self.eval_fn(next_state)
                    new_history = history + [move]
                    candidates.append((next_state, new_score, new_history))
            
            # Keep only top beam_width candidates
            candidates.sort(key=lambda x: x[1], reverse=True)
            beam = candidates[:self.beam_width]
            
            # Check for terminal states
            for state, score, history in beam:
                if state.is_terminal() and state.is_winning():
                    return history
        
        # Return best path found
        return beam[0][2]  # move history of best state

class BeamSearchPlayer:
    def __init__(self, beam_width=10, depth=5):
        self.searcher = BeamSearch(beam_width, depth)
    
    def choose_move(self, game_state):
        moves = self.searcher.search(game_state)
        return moves[0] if moves else None
```

### Variations

**Diverse beam search:**
```python
def diverse_beam_expand(beam, diversity_penalty=0.1):
    """Encourage exploration of diverse states."""
    candidates = []
    
    for state, score, history in beam:
        for move in state.get_valid_moves():
            next_state = state.apply_move(move)
            base_score = eval_fn(next_state)
            
            # Penalize similarity to existing beam states
            similarity = sum(state_distance(next_state, s) 
                           for s, _, _ in beam)
            diverse_score = base_score - diversity_penalty * similarity
            
            candidates.append((next_state, diverse_score, history + [move]))
    
    return sorted(candidates, key=lambda x: x[1])[-beam_width:]
```

**Stochastic beam search:**
```python
def stochastic_beam_select(candidates, beam_width, temperature=1.0):
    """Sample from top candidates instead of taking argmax."""
    scores = np.array([score for _, score, _ in candidates])
    probs = softmax(scores / temperature)
    
    indices = np.random.choice(
        len(candidates), 
        size=min(beam_width, len(candidates)),
        replace=False,
        p=probs
    )
    
    return [candidates[i] for i in indices]
```

**Adaptive beam width:**
```python
class AdaptiveBeamSearch:
    def adapt_width(self, state, base_width):
        """Adjust beam width based on branching factor."""
        num_moves = len(state.get_valid_moves())
        
        if num_moves < 5:
            return base_width // 2  # Narrow beam
        elif num_moves > 20:
            return base_width * 2  # Wide beam
        else:
            return base_width
```

### Expected Performance
- Win rate: ~15-30% (depends on beam width and eval function)
- Average score: Slightly positive
- Time per move: 10-500ms

### Strengths
- Fast and memory efficient
- Explores diverse strategies
- Easy to parallelize
- Scales gracefully with beam width

### Weaknesses
- May discard optimal path early
- No optimality guarantees
- Heavily dependent on evaluation function

---

## Minimax and Alpha-Beta

**Note:** These are primarily included for educational comparison, as they're not well-suited for solitaire.

### Why Minimax Doesn't Fit

Minimax assumes:
1. Two players
2. Perfect information
3. Adversarial opponent
4. Zero-sum game

Solitaire has:
1. One player
2. Partial information (hidden stock)
3. Random "opponent" (draw deck)
4. Not zero-sum

### Educational Implementation

```python
def minimax(state, depth, is_maximizing):
    """
    Standard minimax - NOT APPROPRIATE for solitaire!
    Included only for educational comparison.
    """
    if depth == 0 or state.is_terminal():
        return evaluate(state)
    
    if is_maximizing:
        max_eval = -infinity
        for move in state.get_valid_moves():
            next_state = state.apply_move(move)
            eval = minimax(next_state, depth - 1, False)
            max_eval = max(max_eval, eval)
        return max_eval
    else:
        # This is the problem: treats chance as adversary!
        min_eval = infinity
        for move in state.get_valid_moves():
            next_state = state.apply_move(move)
            eval = minimax(next_state, depth - 1, True)
            min_eval = min(min_eval, eval)
        return min_eval

def alpha_beta(state, depth, alpha, beta, is_maximizing):
    """Alpha-beta pruning - still wrong for solitaire."""
    if depth == 0 or state.is_terminal():
        return evaluate(state)
    
    if is_maximizing:
        max_eval = -infinity
        for move in state.get_valid_moves():
            next_state = state.apply_move(move)
            eval = alpha_beta(next_state, depth - 1, alpha, beta, False)
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break  # Beta cutoff
        return max_eval
    else:
        min_eval = infinity
        for move in state.get_valid_moves():
            next_state = state.apply_move(move)
            eval = alpha_beta(next_state, depth - 1, alpha, beta, True)
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break  # Alpha cutoff
        return min_eval
```

### Why This Performs Poorly

1. **Pessimistic**: Assumes worst-case draws, underestimates winnable positions
2. **Wrong model**: Chance is not adversarial
3. **Missed opportunities**: Doesn't account for probability

### Expected Performance
- Win rate: ~5-12% (worse than heuristics!)
- Average score: Very negative
- Demonstrates importance of algorithm selection

---

## Reinforcement Learning

Learn optimal policy through experience and neural network function approximation.

### Approach Overview

Instead of searching, train a neural network to:
- **Policy network**: π(a|s) - probability of action given state
- **Value network**: V(s) - expected return from state

### Basic Policy Gradient

```python
import torch
import torch.nn as nn

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, state):
        return self.network(state)

class RLAgent:
    def __init__(self, state_dim, action_dim):
        self.policy = PolicyNetwork(state_dim, action_dim)
        self.optimizer = torch.optim.Adam(self.policy.parameters())
    
    def choose_move(self, state, valid_moves):
        state_tensor = torch.FloatTensor(state.to_vector())
        action_probs = self.policy(state_tensor)
        
        # Mask invalid moves
        mask = torch.zeros_like(action_probs)
        for move_idx in valid_moves:
            mask[move_idx] = 1
        masked_probs = action_probs * mask
        masked_probs = masked_probs / masked_probs.sum()
        
        # Sample action
        action = torch.multinomial(masked_probs, 1).item()
        return action
    
    def train_episode(self, states, actions, rewards):
        """Train on one episode using REINFORCE."""
        # Compute returns
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + 0.99 * G  # discount factor
            returns.insert(0, G)
        
        returns = torch.FloatTensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Compute loss
        loss = 0
        for state, action, G in zip(states, actions, returns):
            state_tensor = torch.FloatTensor(state)
            action_probs = self.policy(state_tensor)
            log_prob = torch.log(action_probs[action])
            loss -= log_prob * G
        
        # Update policy
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
```

### Actor-Critic

```python
class ValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim=256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state):
        return self.network(state)

class ActorCritic:
    def __init__(self, state_dim, action_dim):
        self.actor = PolicyNetwork(state_dim, action_dim)
        self.critic = ValueNetwork(state_dim)
        self.actor_opt = torch.optim.Adam(self.actor.parameters())
        self.critic_opt = torch.optim.Adam(self.critic.parameters())
    
    def train_step(self, state, action, reward, next_state, done):
        """One-step Actor-Critic update."""
        state_t = torch.FloatTensor(state)
        next_state_t = torch.FloatTensor(next_state)
        
        # Compute TD error
        value = self.critic(state_t)
        next_value = 0 if done else self.critic(next_state_t)
        td_error = reward + 0.99 * next_value - value
        
        # Update critic
        critic_loss = td_error ** 2
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()
        
        # Update actor
        action_probs = self.actor(state_t)
        log_prob = torch.log(action_probs[action])
        actor_loss = -log_prob * td_error.detach()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()
```

### Expected Performance
- Win rate: ~30-50% (after substantial training)
- Average score: Strong positive expected value
- Inference time: <10ms per move
- Training time: Hours to days on GPU

---

## Hybrid Approaches

### AlphaZero Style

Combine MCTS with neural networks:

```python
class AlphaZeroNode(MCTSNode):
    def __init__(self, state, policy_net, value_net, parent=None):
        super().__init__(state, parent)
        self.policy_net = policy_net
        self.value_net = value_net
        
        # Get initial policy and value from networks
        self.prior_policy = self.policy_net(state)
        self.init_value = self.value_net(state)
    
    def select_child(self):
        """Use PUCT (Polynomial Upper Confidence Trees)."""
        return max(self.children, key=lambda c: self.puct_value(c))
    
    def puct_value(self, child):
        """PUCT selection formula from AlphaGo Zero."""
        if child.visits == 0:
            U = child.prior
        else:
            U = child.prior * math.sqrt(self.visits) / (1 + child.visits)
        
        Q = child.total_reward / child.visits if child.visits > 0 else 0
        return Q + U
    
    def simulate(self):
        """Use value network instead of rollout."""
        return self.value_net(self.state)
```

### Beam Search + MCTS

Use beam search for fast initial plan, refine with MCTS:

```python
class HybridPlayer:
    def __init__(self):
        self.beam = BeamSearch(beam_width=5, depth=3)
        self.mcts = MCTSPlayer(num_simulations=500)
    
    def choose_move(self, state):
        # Quick beam search to identify promising region
        beam_moves = self.beam.search(state)
        
        if not beam_moves:
            return self.mcts.choose_move(state)
        
        # MCTS on most promising path
        best_state = state.copy()
        for move in beam_moves[:2]:  # First 2 moves
            best_state.apply_move(move)
        
        # Refine with MCTS
        return self.mcts.choose_move(best_state)
```

---

## Implementation Tips

### Optimization Techniques

1. **State caching**: Hash states to avoid recomputation
2. **Move ordering**: Try best moves first (for pruning)
3. **Iterative deepening**: Gradually increase search depth
4. **Parallel search**: Multiple threads/processes
5. **GPU acceleration**: For neural network inference

### Debugging Strategies

1. **Unit test each algorithm** on simple positions
2. **Compare to random baseline** (should always be better)
3. **Visualize search trees** to understand behavior
4. **Profile performance** to find bottlenecks
5. **Test on known positions** with correct expected outcomes

### Common Pitfalls

- **Off-by-one errors** in depth limiting
- **Invalid move generation** leading to illegal positions
- **Memory leaks** in tree search (not freeing nodes)
- **Incorrect probability normalization** in chance nodes
- **Overfitting in RL** (train on diverse game states)

---

## Conclusion

Vegas Solitaire offers a rich testbed for comparing search and learning algorithms. The "right" algorithm depends on your constraints:

- **Need speed?** → Heuristic or RL (after training)
- **Need optimality?** → MCTS or Expectimax
- **Limited compute?** → Beam search or shallow search
- **Educational?** → Implement multiple and compare!

The most exciting direction is hybrid approaches that combine the best of search (optimality, interpretability) with learning (speed, generalization).

Happy algorithm hunting! 🔍🃏
