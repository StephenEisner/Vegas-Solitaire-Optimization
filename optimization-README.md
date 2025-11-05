# Optimization Algorithms

This directory contains the strategy optimization and analysis tools for Vegas Solitaire.

## Goals

- Develop algorithms to find optimal or near-optimal play strategies
- Compare different approaches (search, learning, heuristics)
- Generate training data for neural networks
- Analyze game complexity and decision points

## Architecture

```
optimization/
├── README.md                  # This file
├── solvers/
│   ├── random.py             # Random baseline
│   ├── heuristic.py          # Hand-crafted heuristics
│   ├── mcts.py               # Monte Carlo Tree Search
│   ├── expectimax.py         # Expectimax for stochastic games
│   ├── beam_search.py        # Beam search with pruning
│   ├── minimax.py            # Minimax (limited applicability)
│   ├── alpha_beta.py         # Alpha-beta pruning
│   └── rl_agent.py           # Reinforcement Learning agent
├── training/
│   ├── train_policy.py       # Policy network training
│   ├── train_value.py        # Value network training
│   └── self_play.py          # Self-play data generation
├── evaluation/
│   ├── benchmark.py          # Compare strategies
│   ├── analyze.py            # Game analysis tools
│   └── statistics.py         # Win rate, expected value, etc.
├── models/
│   ├── networks.py           # Neural network architectures
│   └── features.py           # State feature extraction
├── utils/
│   ├── replay_buffer.py      # Experience replay
│   └── visualization.py      # Training curves, etc.
└── requirements.txt
```

## Optimization Approaches

### 1. Monte Carlo Tree Search (MCTS)
Pure search-based approach without learning.

**Pros:**
- No training required
- Theoretically optimal given enough time
- Works with perfect game simulation

**Cons:**
- Computationally expensive
- May be too slow for real-time play
- Doesn't generalize across games

**Implementation:**
- UCT (Upper Confidence bounds applied to Trees)
- Rollout policy (random or heuristic-guided)
- State caching/transposition tables

### 2. Reinforcement Learning
Learn a policy through self-play and experience.

**Approaches:**
- **Policy Gradient** (REINFORCE, A2C, PPO)
- **Value-based** (DQN, Double DQN)
- **Actor-Critic** (A3C, SAC)

**Training pipeline:**
1. Self-play to generate experience
2. Train policy and value networks
3. Evaluate against baselines
4. Iterate

### 3. Hybrid: AlphaZero Style
Combine MCTS with learned policy and value networks.

**Advantages:**
- MCTS provides strong play for training data
- Neural networks improve MCTS efficiency
- State-of-the-art for many games

**Requirements:**
- GPU for network training
- Substantial compute for self-play

### 4. Heuristic Approaches
Hand-crafted rules and strategies.

**Use cases:**
- Baseline for comparison
- Initialization for learning
- Understanding human-like play

**Examples:**
- Prefer foundation moves
- Avoid blocking useful cards
- Maximize tableau flexibility

### 5. Expectimax Search
Tree search algorithm designed for stochastic games with chance nodes.

**Why Expectimax for Solitaire:**
- Vegas Solitaire has **chance nodes** (drawing from stock)
- Unlike minimax (adversarial), expectimax handles **random events**
- Computes expected value over possible draws

**How it works:**
1. Build game tree with alternating decision and chance nodes
2. Decision nodes: Choose move that maximizes expected value
3. Chance nodes: Weight outcomes by probability
4. Propagate expected values up the tree

**Pros:**
- Theoretically sound for stochastic games
- Accounts for probability of different draws
- Can provide optimal play given perfect evaluation

**Cons:**
- Exponential branching on chance nodes
- Requires evaluating many possible futures
- Deep search is computationally prohibitive

**Implementation considerations:**
- Limit search depth (e.g., 3-5 moves ahead)
- Use heuristic evaluation function at leaf nodes
- Prune unlikely branches
- Cache repeated states

**Pseudocode:**
```python
def expectimax(state, depth):
    if depth == 0 or state.is_terminal():
        return evaluate(state)
    
    if state.is_decision_node():
        # Maximize over actions
        return max(expectimax(child, depth-1) 
                   for child in state.get_children())
    else:
        # Chance node - expected value
        return sum(prob * expectimax(child, depth-1) 
                   for child, prob in state.get_chance_outcomes())
```

### 6. Beam Search
Limited-width breadth-first search that keeps only the top-k most promising states.

**Why Beam Search for Solitaire:**
- Explores multiple paths simultaneously
- Memory-efficient compared to full breadth-first search
- Can look ahead several moves without explosion

**How it works:**
1. Start with initial game state
2. Generate all possible next states
3. Evaluate states with heuristic function
4. Keep only top-k states (the "beam")
5. Repeat for desired depth

**Pros:**
- Computationally tractable for large branching factors
- Explores diverse strategies in parallel
- Can be very fast with good heuristics

**Cons:**
- Not guaranteed to find optimal solution
- Beam width is a critical hyperparameter
- May discard the optimal path early

**Implementation:**
```python
def beam_search(initial_state, beam_width, max_depth):
    beam = [(initial_state, 0)]  # (state, score)
    
    for depth in range(max_depth):
        candidates = []
        for state, score in beam:
            for move in state.get_valid_moves():
                next_state = state.apply_move(move)
                candidates.append((next_state, evaluate(next_state)))
        
        # Keep top beam_width candidates
        beam = sorted(candidates, key=lambda x: x[1])[-beam_width:]
    
    return beam[0][0]  # Best state found
```

**Variations:**
- **Diverse beam search**: Penalize similar states to encourage exploration
- **Stochastic beam search**: Add randomness to avoid getting stuck
- **Adaptive beam width**: Expand/contract beam based on state space

### 7. Minimax & Alpha-Beta Pruning
Classic adversarial search algorithms (limited applicability to solitaire).

**Important note:** Solitaire is NOT an adversarial game, so minimax is not directly applicable. However, we include these for completeness and educational purposes.

**Possible applications:**
- **Adversarial framing**: Treat stock/chance as an "adversary"
- **Two-player variant**: Competitive solitaire racing
- **Educational comparison**: Show why wrong algorithm matters

**Minimax:**
- Assumes opponent plays optimally to minimize your score
- Alternates between maximizing and minimizing layers
- Provides worst-case guarantees

**Alpha-Beta Pruning:**
- Optimized minimax that prunes branches that can't affect outcome
- Same result as minimax but faster
- Can search deeper in the same time

**Why they don't fit well:**
```
Solitaire:          Adversarial games:
- Stochastic        - Deterministic
- Single player     - Two players  
- Chance nodes      - Adversarial nodes
- Goal: maximize    - Goal: win/lose
```

**Educational value:**
- Demonstrate algorithm selection matters
- Show performance of "wrong" algorithm
- Contrast with expectimax (right for stochastic games)

**If implemented (for comparison):**
```python
def minimax(state, depth, is_maximizing):
    if depth == 0 or state.is_terminal():
        return evaluate(state)
    
    if is_maximizing:
        return max(minimax(child, depth-1, False)
                   for child in state.get_children())
    else:
        # Wrong for solitaire! Assumes adversary
        return min(minimax(child, depth-1, True)
                   for child in state.get_children())
```

**Alpha-beta optimization:**
```python
def alpha_beta(state, depth, alpha, beta, is_maximizing):
    if depth == 0 or state.is_terminal():
        return evaluate(state)
    
    if is_maximizing:
        value = -infinity
        for child in state.get_children():
            value = max(value, alpha_beta(child, depth-1, alpha, beta, False))
            alpha = max(alpha, value)
            if beta <= alpha:
                break  # Beta cutoff
        return value
    else:
        value = infinity
        for child in state.get_children():
            value = min(value, alpha_beta(child, depth-1, alpha, beta, True))
            beta = min(beta, value)
            if beta <= alpha:
                break  # Alpha cutoff
        return value
```

### 8. Hybrid & Advanced Methods

**MCTS + Heuristics:**
- Use heuristic evaluation for rollouts
- Guides search toward promising areas
- Faster convergence than random rollouts

**Beam Search + MCTS:**
- Use beam search for fast approximate plan
- Refine with MCTS on promising branches
- Balances speed and quality

**Genetic Algorithms:**
- Evolve sequences of moves
- Crossover between successful game strategies
- Potentially interesting for exploring strategy space

**Imitation Learning:**
- Learn from expert human play
- Bootstrap RL training
- Warm-start for faster convergence

## Algorithm Selection Guide

### When to use each algorithm:

| Algorithm | Best for | Avoid when |
|-----------|----------|------------|
| **Random** | Baseline testing | Actual play |
| **Heuristic** | Fast decisions, baseline | Need optimality |
| **MCTS** | Strong play, no training | Time-constrained |
| **Expectimax** | Short-term planning | Deep search needed |
| **Beam Search** | Fast lookahead | Need guarantees |
| **RL** | Long-term strategy | Limited data/compute |
| **Hybrid** | Best performance | Simplicity needed |
| **Minimax/AB** | Educational only | Solitaire play |

### Computational complexity:

| Algorithm | Time per move | Space | Anytime? |
|-----------|---------------|-------|----------|
| Random | O(1) | O(1) | Yes |
| Heuristic | O(n) moves | O(1) | Yes |
| MCTS | O(budget) | O(tree) | Yes |
| Expectimax | O(b^d * c^d) | O(d) | No |
| Beam Search | O(k * b * d) | O(k) | Yes |
| RL | O(inference) | O(model) | Yes |

Where:
- b = branching factor (~10-30 for solitaire)
- d = depth
- c = chance outcomes (52 for full stock)
- k = beam width

## State Representation for ML

Key question: How do we represent the game state for neural networks?

### Option A: Structured Representation
- One-hot encode each card position
- Separate channels for suit, rank, visibility
- Fixed-size tensor (e.g., 52 positions × features)

### Option B: Set-Based Representation
- Treat tableau columns as sets
- Use attention mechanisms or set encoders
- More flexible for varying pile sizes

### Option C: Graph Representation
- Cards as nodes, moves as edges
- Graph neural networks
- Capture card dependencies

## Evaluation Metrics

### Performance Metrics
- **Win rate**: % of games that clear all foundations
- **Average score**: Expected profit in Vegas mode
- **Median score**: Robust to outliers
- **Score distribution**: Full picture of performance

### Efficiency Metrics
- **Moves per second**: Computation speed
- **Time per game**: Wall-clock time
- **Convergence speed**: Training efficiency

### Analysis Metrics
- **Decision quality**: Compare to optimal (if known)
- **Branching factor**: Average # of valid moves
- **Game length**: Average # of moves to completion
- **Critical decisions**: Identify high-impact moments

## Training Infrastructure

### Local Development
- Small-scale experiments
- Algorithm prototyping
- Testing on CPU

### Cloud GPU Training
- Large-scale self-play
- Network training
- Hyperparameter sweeps

**Platforms:**
- Google Colab (free GPU, limited time)
- Kaggle Notebooks (free GPU, 30h/week)
- AWS/GCP/Azure (paid, scalable)

## Experiment Tracking

Track experiments systematically:
- Hyperparameters
- Training curves
- Evaluation results
- Model checkpoints

**Tools:**
- TensorBoard
- Weights & Biases
- MLflow
- Simple CSV logs

## Research Questions

### Strategic Questions
1. What is the theoretical win rate of Vegas Solitaire?
2. What are the most important decision points?
3. How much does draw order (randomness) matter?
4. Can we identify "unwinnable" games early?
5. What heuristics correlate most with winning?

### Algorithmic Questions
1. How does MCTS compare to expectimax and learned policies?
2. What's the optimal search depth for expectimax/beam search?
3. Does minimax perform poorly (as expected) on stochastic games?
4. What beam width gives best speed/quality trade-off?
5. What network architecture works best for RL?
6. How much training data is needed for RL convergence?
7. Does transfer learning help across rule variants?
8. Can hybrid approaches (beam+MCTS) outperform pure methods?

### Practical Questions
1. Can we achieve near-optimal play on commodity hardware?
2. How fast can we make each solver?
3. What's the minimal model size for good performance?
4. Which algorithm is best for real-time play vs offline analysis?
5. How do algorithms scale with longer time budgets?

## Detailed Algorithm Comparison

### Comparison Matrix

| Feature | Random | Heuristic | Expectimax | Beam | MCTS | RL | Minimax/AB |
|---------|--------|-----------|------------|------|------|----|----|
| **Training needed** | No | No | No | No | No | Yes | No |
| **Theoretical optimality** | No | No | Yes* | No | Yes* | No | No** |
| **Handles stochasticity** | N/A | N/A | Yes | Yes | Yes | Yes | No |
| **Anytime algorithm** | Yes | Yes | No | Yes | Yes | Yes | No |
| **Memory usage** | Low | Low | Medium | Low | High | Medium | Medium |
| **Computational cost** | Very Low | Low | High | Medium | High | Low*** | High |
| **Generalizes across games** | Yes | No | Yes | No | Yes | Yes | Yes |
| **Interpretability** | High | High | Medium | Medium | Low | Low | High |

\* Given sufficient computation time and perfect evaluation  
\*\* Not optimal for stochastic games  
\*\*\* After training; training itself is expensive

### Expected Performance Ranking

Based on theoretical suitability (to be validated empirically):

**Tier 1: Strong Performance**
- AlphaZero-style (MCTS + RL)
- Pure MCTS (with sufficient time)
- Deep RL (with good training)

**Tier 2: Good Performance**
- Expectimax (limited depth)
- Beam search (with good heuristics)
- Heuristic-guided MCTS

**Tier 3: Baseline Performance**
- Pure heuristic
- Shallow search methods

**Tier 4: Reference Baseline**
- Random play

**Tier 5: Educational Only**
- Minimax/Alpha-beta (wrong algorithm for the task)

### Algorithm Characteristics

#### Deterministic vs Stochastic
- **Solitaire is stochastic** (random draw order)
- Best algorithms: Expectimax, MCTS, RL
- Poor fit: Minimax (assumes deterministic/adversarial)

#### Complete vs Incomplete Information
- **Partial observability** (stock cards hidden)
- RL can learn to handle uncertainty
- Search methods need to consider all possibilities

#### Short-term vs Long-term Planning
- **Long horizon** (100+ moves per game)
- MCTS and RL excel at long-term planning
- Expectimax/Beam limited by computational budget

### Time Budget Analysis

How algorithms perform at different time budgets:

**Instant (<10ms):**
1. Heuristic
2. RL (inference)
3. Beam (small width)
4. Random

**Fast (10-100ms):**
1. RL
2. Beam search (medium width)
3. Shallow MCTS
4. Expectimax (depth 2-3)

**Medium (100ms-1s):**
1. MCTS
2. Expectimax (depth 4-5)
3. Beam search (large width)

**Slow (>1s):**
1. Deep MCTS
2. Deep expectimax
3. Hybrid methods

**Offline (unlimited):**
1. AlphaZero training
2. Exhaustive MCTS
3. Large-scale RL training

### Strengths & Weaknesses Summary

**Expectimax:**
- ✅ Theoretically correct for stochastic games
- ✅ Provides expected value estimates
- ❌ Exponential growth with chance nodes
- ❌ Requires good heuristic evaluation
- **Best for:** Short-term tactical decisions

**Beam Search:**
- ✅ Fast and memory-efficient
- ✅ Explores multiple strategies in parallel
- ❌ Can miss optimal path
- ❌ Beam width is critical parameter
- **Best for:** Quick lookahead with diverse exploration

**MCTS:**
- ✅ Handles stochasticity naturally
- ✅ Anytime algorithm (improves with time)
- ✅ No need for heuristic evaluation
- ❌ Can be slow to converge
- ❌ High memory usage for tree
- **Best for:** Strong play when time permits

**Reinforcement Learning:**
- ✅ Fast inference after training
- ✅ Can learn complex long-term strategies
- ✅ Generalizes across different scenarios
- ❌ Requires substantial training time/data
- ❌ No optimality guarantees
- **Best for:** Deployment after offline training

**Heuristic:**
- ✅ Fast and interpretable
- ✅ Encodes human knowledge
- ❌ Hard to tune for optimality
- ❌ May miss non-obvious strategies
- **Best for:** Baselines and rollout policies

**Minimax/Alpha-Beta:**
- ✅ Optimal for adversarial games
- ✅ Well-understood theory
- ❌ Wrong model for solitaire
- ❌ Pessimistic for stochastic games
- **Best for:** Educational comparison only

### Recommended Implementation Order

1. **Random** (1 hour) - Establish floor performance
2. **Heuristic** (1-2 days) - Understand game dynamics
3. **Beam Search** (2-3 days) - Fast practical solver
4. **Expectimax** (3-5 days) - Theoretically grounded search
5. **MCTS** (1 week) - Strong general solver
6. **RL** (2-3 weeks) - Learnable policy
7. **Minimax** (optional, 1 day) - Educational contrast
8. **Hybrid** (1-2 weeks) - Push state-of-the-art

Total: ~6-8 weeks for complete implementation

## Benchmarking Protocol

Standard evaluation procedure:
1. Test on fixed set of seeds (e.g., 1000 games)
2. Report mean and confidence intervals
3. Compare to baselines (random, heuristic, MCTS)
4. Measure computation time

## Data Generation

### Self-Play Data
- Generate games using current policy
- Store (state, action, outcome) tuples
- Balance dataset (winnin/losing games)

### Expert Data (if available)
- Human expert games
- Games from strong solvers
- Use for imitation learning

### Synthetic Data
- Random games for coverage
- Targeted scenarios (e.g., endgames)
- Adversarial examples

## Model Deployment

After training:
1. Export model weights
2. Optimize for inference (quantization, pruning)
3. Package for web deployment (ONNX, TensorFlow.js)
4. Integrate with website/visualization

## Dependencies

```
numpy
torch / jax / tensorflow
gym (custom Vegas Solitaire environment)
matplotlib / seaborn (visualization)
tensorboard (experiment tracking)
```

## Development Roadmap

### Phase 1: Baselines
- [ ] Random player (baseline for comparison)
- [ ] Simple heuristic player (rule-based)
- [ ] Benchmark on test set

### Phase 2: Classical Search
- [ ] Expectimax (proper stochastic game algorithm)
- [ ] Beam search with heuristics
- [ ] Minimax/Alpha-beta (educational comparison)
- [ ] Evaluate depth vs performance trade-offs

### Phase 3: MCTS
- [ ] Basic MCTS implementation
- [ ] Optimize with caching and transposition tables
- [ ] Heuristic-guided rollouts
- [ ] Evaluate strength vs baselines

### Phase 4: RL Training
- [ ] Define state/action spaces
- [ ] Implement gym environment
- [ ] Train simple policy network
- [ ] Evaluate and iterate

### Phase 5: Advanced Methods
- [ ] AlphaZero-style hybrid (MCTS + NN)
- [ ] Train on cloud GPU
- [ ] Hyperparameter optimization
- [ ] Hybrid algorithms (beam + MCTS, etc.)
- [ ] Final policy generation

### Phase 6: Analysis & Comparison
- [ ] Comprehensive benchmark suite
- [ ] Algorithm performance comparison
- [ ] Analyze learned strategies
- [ ] Identify strengths/weaknesses of each approach
- [ ] Document findings
- [ ] Prepare for website integration

## Performance Goals

Target metrics (to be refined):
- Win rate: >30% (random baseline: ~5-10%?)
- Speed: >100 moves/sec evaluation
- Training: Converge in <24 hours on single GPU

## Notes on Vegas Solitaire Complexity

- **State space**: Enormous (52! permutations × hidden cards)
- **Decision complexity**: Varies from trivial to dozens of options
- **Horizon**: Games can last 100+ moves
- **Partial observability**: Stock cards are hidden until drawn
- **Stochasticity**: Draw order affects winability

This makes it a challenging but tractable problem for modern RL.

## Open Questions

- Should we model the problem as MDP or POMDP?
- Is the game learnable end-to-end, or do we need structure?
- What's the best balance of exploration vs exploitation?
- Can we learn transferable card game knowledge?

## Future Directions

- Multi-agent learning (vs different variants)
- Curriculum learning (easy to hard games)
- Explainable AI (interpret learned strategies)
- Human-AI collaboration (hint systems)
