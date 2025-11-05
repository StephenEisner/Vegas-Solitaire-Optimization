# Advanced Algorithms for Vegas Solitaire

## Overview
Expand beyond current approaches (Random, Heuristic, MCTS) to include classic AI search algorithms and reinforcement learning methods that produce **learned policies**.

---

## Current State

### Implemented Solvers
1. **Random** - Baseline, chooses moves uniformly at random
2. **Heuristic** - Hand-crafted scoring function (recently improved)
3. **MCTS** - Monte Carlo Tree Search with UCB1

### Limitations
- **Heuristic**: Hand-tuned weights, no learning, domain-specific
- **MCTS**: Computationally expensive (100 sims × 500 moves = 50K simulations/game)
- **No Policy Learning**: Can't improve from experience across multiple games

---

## Proposed Algorithm Categories

### 1. Classic Search Algorithms
These are **online** algorithms that search during gameplay.

#### A. Best-First Search / A*
- **Concept**: Prioritize exploring states with best estimated value
- **Heuristic**: Use improved heuristic as h(s) function
- **Advantage**: Optimal if heuristic is admissible
- **Challenge**: Memory - need to store open/closed sets
- **Adaptation**: Beam search to limit memory

#### B. Beam Search
- **Concept**: Keep only top-K states at each depth
- **Parameters**: Beam width K (e.g., K=100)
- **Advantage**: Memory efficient, parallelizable
- **Implementation**: Expand all K states, keep best K children

#### C. Iterative Deepening
- **Concept**: Depth-first search with increasing depth limit
- **Advantage**: Memory efficient (O(depth))
- **Challenge**: Repeated work at shallow depths
- **Vegas Adaptation**: Set max depth = max moves allowed

#### D. Limited-Width Alpha-Beta
- **Concept**: Treat as single-player game, prune unlikely branches
- **Adaptation**: Use heuristic to evaluate terminal positions
- **Note**: Classical alpha-beta is for adversarial games, but can adapt

---

### 2. Reinforcement Learning (Policy-Based) ⭐
These produce **learned policies** that can be trained once and reused.

#### A. TD(λ) - Temporal Difference Learning
**Type**: Value-based → derive policy from value function

**Approach**:
- Learn V(s) or Q(s,a) across many games
- Use experience replay to update values
- Derive greedy policy: π(s) = argmax_a Q(s,a)

**Advantages**:
- Learns from actual game outcomes
- Can improve heuristic automatically
- Bootstrapping (learn from incomplete episodes)

**State Representation**:
- Foundation counts (4 values)
- Tableau visible cards (7 × max_cards features)
- Stock/waste sizes (2 values)
- Feature engineering crucial for performance

**Training**:
```python
# TD(0) update rule
V(s) ← V(s) + α[r + γV(s') - V(s)]

# TD(λ) with eligibility traces
e(s) ← e(s) + 1
For all states:
    V(s) ← V(s) + α δ e(s)
    e(s) ← γλ e(s)
```

#### B. Q-Learning (Off-Policy TD)
**Type**: Action-value learning

**Approach**:
- Learn Q(s,a) for every state-action pair
- Update: Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]
- Exploration: ε-greedy (random with probability ε)

**Advantages**:
- Off-policy (can learn from random exploration)
- Proven convergence to optimal policy

**Challenges**:
- Large state-action space (need function approximation)
- Solitaire has variable action sets (different valid moves per state)

#### C. SARSA (On-Policy TD)
**Type**: Action-value learning (on-policy)

**Difference from Q-Learning**:
- Q-Learning: learns optimal policy (uses max)
- SARSA: learns policy being followed (uses actual next action)

**Update**:
```python
Q(s,a) ← Q(s,a) + α[r + γ Q(s',a') - Q(s,a)]
# where a' is actually taken (not max)
```

**Better for**: Risk-sensitive policies

#### D. Policy Gradient (REINFORCE)
**Type**: Direct policy learning

**Approach**:
- Learn π(a|s; θ) directly (parameterized policy)
- Use gradient ascent on expected reward
- Update: θ ← θ + α ∇θ log π(a|s; θ) G_t

**Advantages**:
- Can learn stochastic policies
- Works with continuous action spaces (not needed here)
- Can incorporate domain knowledge in network architecture

**Challenges**:
- High variance (use baseline)
- Requires many episodes

#### E. Actor-Critic
**Type**: Combines value and policy learning

**Components**:
- **Actor**: Policy π(a|s; θ)
- **Critic**: Value function V(s; w)

**Advantages**:
- Lower variance than pure policy gradient
- More sample efficient

---

### 3. Hybrid Approaches

#### A. MCTS + Learned Value Function
- Use learned V(s) for MCTS rollouts instead of random playouts
- Reduces simulation depth needed
- Can use TD-learned values

#### B. Heuristic + RL Fine-Tuning
- Start with hand-crafted heuristic weights
- Use RL to fine-tune weights
- Combines domain knowledge with learning

#### C. Neural Network Policy
- Deep learning approach
- CNN or Transformer to process game state
- Large training requirement but potentially best performance

---

## Implementation Plan

### Phase 1: Classic Search (1-2 days)
Implement memory-efficient search algorithms that run fast.

**Priority**:
1. ✅ **Beam Search** - Most promising for Vegas Solitaire
   - Memory bounded
   - Parallelizable
   - Can use improved heuristic for scoring

2. **Iterative Deepening A\***
   - Memory efficient
   - Completeness guarantee

### Phase 2: Value-Based RL (2-3 days)
Learn value functions from experience.

**Priority**:
1. ✅ **TD(0) Learning** - Simplest, good starting point
   - State representation design
   - Feature extraction
   - Training loop (1000-10000 games)

2. **Q-Learning with Function Approximation**
   - Linear function approximation first
   - Neural network later if needed

### Phase 3: Policy-Based RL (3-5 days)
Direct policy learning.

**Priority**:
1. **REINFORCE** - Classic policy gradient
2. **Actor-Critic** - More sample efficient

### Phase 4: Hybrid & Advanced (Ongoing)
Combine approaches for best results.

---

## Framework Design

### Policy Interface
All learned algorithms should produce a reusable policy:

```python
class Policy(ABC):
    """Base class for learned policies."""

    @abstractmethod
    def choose_action(self, state: GameState, valid_moves: List[Move]) -> Move:
        """Choose action given state and valid moves."""
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """Save policy to disk."""
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """Load policy from disk."""
        pass
```

### Trainer Interface
Separate training from execution:

```python
class PolicyTrainer(ABC):
    """Base class for training policies."""

    @abstractmethod
    def train(self, num_episodes: int) -> Policy:
        """Train and return a policy."""
        pass

    @abstractmethod
    def get_training_stats(self) -> Dict[str, Any]:
        """Return training statistics."""
        pass
```

### Feature Extraction
Common features for all learning algorithms:

```python
class StateFeatures:
    """Extract features from game state for learning."""

    @staticmethod
    def extract_features(state: GameState) -> np.ndarray:
        """
        Extract feature vector from state.

        Features:
        - Foundation counts (4 values)
        - Max tableau depth (7 values)
        - Hidden card counts (7 values)
        - Stock size (1 value)
        - Waste size (1 value)
        - Passes through deck (1 value)
        - Empty tableau columns (1 value)
        - Total: 22 features
        """
        pass
```

---

## Experimental Setup

### Training Protocol
1. **Training Set**: Seeds 0-9999 (10K games)
2. **Validation Set**: Seeds 10000-10999 (1K games)
3. **Test Set**: Seeds 11000-11999 (1K games)

### Hyperparameters to Tune
- Learning rate α (try: 0.1, 0.01, 0.001)
- Discount factor γ (try: 0.95, 0.99, 0.999)
- Exploration ε (try: 0.1, 0.2 with decay)
- TD(λ): λ (try: 0.0, 0.5, 0.9)

### Evaluation Metrics
- Win rate (most important)
- Average score
- Average foundation cards
- Training time
- Inference time per move
- Sample efficiency (performance vs training games)

---

## Expected Results

### Beam Search
- **Win Rate**: 5-15% (depends on beam width)
- **Speed**: Fast (< 1s per game)
- **Memory**: O(beam_width)

### TD Learning
- **Win Rate**: 10-20% (after sufficient training)
- **Training Time**: Hours (10K games)
- **Inference**: Very fast (< 0.01s per move)
- **Advantage**: Policy improves with more training

### Q-Learning
- **Win Rate**: 15-25% (with good features)
- **Training Time**: Longer than TD (more parameters)
- **Advantage**: Can discover novel strategies

### Policy Gradient
- **Win Rate**: 20-30% (potentially best)
- **Training Time**: Longest (high variance)
- **Advantage**: Direct optimization of win rate

---

## Key Research Questions

1. **What features matter most?**
   - Foundation counts? Hidden cards? Empty columns?
   - Feature selection experiments

2. **How much training data is needed?**
   - Plot learning curves
   - Diminishing returns point

3. **Can we transfer learning?**
   - Train on Draw-1, transfer to Draw-3?
   - Train on unlimited passes, transfer to 3-pass?

4. **Hybrid vs Pure?**
   - Is MCTS + learned value better than pure RL?
   - What's the optimal combination?

5. **State abstraction**:
   - Can we group similar states?
   - Use card counting (ranks/suits remaining)?

---

## File Structure

```
optimization/
├── solvers/
│   ├── search/
│   │   ├── beam_search.py
│   │   ├── iterative_deepening.py
│   │   └── astar.py
│   ├── rl/
│   │   ├── td_learning.py
│   │   ├── q_learning.py
│   │   ├── sarsa.py
│   │   ├── policy_gradient.py
│   │   └── actor_critic.py
│   └── hybrid/
│       └── mcts_learned_value.py
├── policies/
│   ├── base.py              # Policy interface
│   ├── value_policy.py      # Derive policy from value function
│   └── learned_policy.py    # Direct learned policy
├── training/
│   ├── trainer.py           # Base trainer class
│   ├── td_trainer.py
│   ├── q_trainer.py
│   └── pg_trainer.py
└── features/
    ├── state_features.py    # Feature extraction
    └── feature_engineering.py
```

---

## Next Steps

1. ✅ Implement `Policy` base class
2. ✅ Implement `StateFeatures` for feature extraction
3. ✅ Implement **Beam Search** solver
4. ✅ Implement **TD(0) Learning** trainer and policy
5. Run experiments comparing all algorithms
6. Iterate on features and hyperparameters
7. Publish results in README

---

## Success Criteria

**Minimum Viable**:
- At least one RL algorithm implemented and trained
- Win rate > 10% on test set
- Faster inference than MCTS

**Stretch Goals**:
- Win rate > 20% (approaching human expert level)
- Multiple RL approaches compared
- Learned policy outperforms hand-tuned heuristic
- Transferable insights to related card games

---

## References

### Classic Search
- Russell & Norvig, "Artificial Intelligence: A Modern Approach"
- Hart, Nilsson, Raphael (1968), "A Formal Basis for the Heuristic Determination of Minimum Cost Paths"

### Reinforcement Learning
- Sutton & Barto, "Reinforcement Learning: An Introduction" (2nd ed)
- Mnih et al. (2015), "Human-level control through deep RL" (DQN)
- Silver et al. (2016), "Mastering the game of Go with deep neural networks" (AlphaGo)

### Game-Specific
- Yan et al. (2005), "Using Monte-Carlo evaluations in a game tree search"
- Browne et al. (2012), "A Survey of Monte Carlo Tree Search Methods"
