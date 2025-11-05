# Multi-Deal Mode with AI Optimization

This document describes the multi-deal mode feature, which simulates realistic casino play with bankroll management, reroll decisions, and AI-powered deal selection.

## Overview

Multi-deal mode implements a realistic casino scenario where:
- You start with a bankroll (e.g., $500)
- Each game costs $52 to play
- You earn $5 per card placed in foundations
- You can reroll deals (limited number of times, costs $5 per reroll)
- Rerolls reset when you accept and play a deal
- Goal: Maximize profit over multiple deals

## Key Features

### 1. Q-Learning Implementation

**Location**: `optimization/solvers/rl/q_learning.py`

Q-Learning learns the action-value function Q(s,a) directly, unlike TD Learning which learns V(s).

**Key components**:
- `QPolicy`: Policy that acts greedily w.r.t learned Q-function
- `QLearner`: Training algorithm with linear function approximation
- `train_q_policy()`: Convenience function for training

**Update rule**:
```
Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]
```

**Usage**:
```python
from optimization.solvers.rl.q_learning import train_q_policy

# Train a Q-Learning policy
policy = train_q_policy(
    num_episodes=1000,
    alpha=0.005,
    gamma=0.95,
    epsilon=0.15,
    save_path="policies/q_policy.json"
)
```

### 2. Multi-Deal Mode Framework

**Location**: `optimization/meta/multideal_mode.py`

Main components:
- `MultidealManager`: Manages sessions with bankroll and reroll tracking
- `DealSelector`: Chooses which deal to play from available rerolls
- `DealAcceptor`: Decides whether to accept or reroll a deal

**Features**:
- Bankroll management
- Reroll tracking (resets after each accepted deal)
- Deal evaluation and selection
- Session statistics and reporting

**Usage**:
```python
from optimization.meta.multideal_mode import MultidealManager, DealSelector, DealAcceptor
from optimization.solvers.heuristic_solver import HeuristicSolver

# Create components
solver = HeuristicSolver()
selector = DealSelector(strategy="secretary")
acceptor = DealAcceptor(strategy="threshold")

# Create manager
manager = MultidealManager(
    starting_bankroll=500.0,
    max_rerolls=3,
    reroll_cost=5.0,
    solver=solver,
    selector=selector,
    acceptor=acceptor
)

# Run session
session = manager.run_session(max_deals=10, start_seed=42, verbose=True)

print(f"Final profit: ${session.profit:+.2f}")
print(f"Win rate: {session.win_rate*100:.1f}%")
```

### 3. AI-Powered Deal Selection

**Location**: `optimization/meta/ai_deal_strategy.py`

Integrates learned policies (Q-Learning, TD Learning) with heuristic strategies for intelligent deal decisions.

**Components**:
- `LearnedDealSelector`: Uses learned value functions to evaluate deals
- `LearnedDealAcceptor`: Makes acceptance decisions using learned policies
- `HybridDealStrategy`: Ensemble of multiple strategies
- `create_ai_deal_strategy()`: Factory for creating strategies

**Strategies available**:
1. **Heuristic**: Pure heuristic-based evaluation (baseline)
2. **Q-Learning**: Uses Q-Learning policy for evaluation
3. **TD-Learning**: Uses TD Learning policy for evaluation
4. **Hybrid**: Ensemble of multiple approaches

**Usage**:
```python
from optimization.meta.ai_deal_strategy import create_ai_deal_strategy

# Create Q-Learning strategy
selector, acceptor = create_ai_deal_strategy(
    strategy_type="q_learning",
    policy_path="policies/q_policy.json"
)

# Use with MultidealManager
manager = MultidealManager(
    starting_bankroll=500.0,
    max_rerolls=3,
    solver=solver,
    selector=selector,
    acceptor=acceptor
)
```

### 4. Deal Selection Algorithms

**Location**: `optimization/meta/deal_selector.py`

Implements optimal stopping algorithms for choosing which deal to play:

1. **Secretary Problem (37% rule)**:
   - Observe first 37% of deals
   - Take next deal better than best observed
   - Theoretical optimal for unknown distribution

2. **Threshold-based**:
   - Calibrate quality threshold on sample deals
   - Accept first deal above threshold
   - Practical and effective

3. **Best (Oracle)**:
   - See all deals, pick best
   - Upper bound on performance

### 5. Deal Evaluation

**Location**: `optimization/meta/deal_evaluator.py`

Evaluates deal quality based on:
1. **Immediate playability** (most important - user insight!)
2. **Stack configuration** (short vs long stacks)
3. **Hidden card potential**
4. **Sequence quality**

**User insights integrated**:
- "Don't play deals without many immediately playable cards"
- "Sometimes prioritize short stacks (for Kings)"
- "Sometimes prioritize long stacks (for hidden cards)"

## Demo Script

**Location**: `demo_multideal_ai.py`

Comprehensive demonstration of all features:

```bash
# Run basic demos
python demo_multideal_ai.py

# Train Q-Learning and TD Learning policies
python demo_multideal_ai.py --train

# Play with Q-Learning
python demo_multideal_ai.py --play

# Compare AI strategies
python demo_multideal_ai.py --compare

# Run all demos
python demo_multideal_ai.py --all
```

## Architecture

```
Multi-Deal Session
    ├── Bankroll Management
    │   ├── Starting bankroll
    │   ├── Cost per game ($52)
    │   ├── Reward per card ($5)
    │   └── Reroll cost ($5)
    │
    ├── Deal Selection
    │   ├── Secretary Problem
    │   ├── Threshold-based
    │   ├── Learned Q-values
    │   └── Learned TD values
    │
    ├── Deal Acceptance
    │   ├── Heuristic quality
    │   ├── Learned value function
    │   ├── Risk tolerance
    │   └── Bankroll constraints
    │
    └── Game Play
        ├── Heuristic Solver
        ├── Q-Learning Policy
        ├── TD Learning Policy
        └── Strategic Solver
```

## Performance Metrics

The system tracks:
- **Total profit/loss**: Final bankroll - starting bankroll
- **Win rate**: Percentage of deals won (52/52 cards)
- **Average cards played**: Cards per deal
- **Rerolls used**: Total rerolls across session
- **Deals evaluated**: Total deals seen (including rerolls)

## Configuration Options

### MultidealManager

```python
MultidealManager(
    starting_bankroll=500.0,     # Starting money
    cost_per_game=52.0,          # Cost to play one game
    reward_per_card=5.0,         # Reward per foundation card
    max_rerolls=3,               # Rerolls available
    reroll_cost=5.0,             # Cost to reroll
    solver=None,                 # Solver for playing deals
    selector=None,               # Deal selection strategy
    acceptor=None                # Deal acceptance strategy
)
```

### DealSelector

```python
DealSelector(
    strategy="secretary"  # "secretary", "threshold", "best", "learned"
)
```

### DealAcceptor

```python
DealAcceptor(
    strategy="threshold",         # "threshold", "value", "conservative", "aggressive"
    quality_threshold=None,       # Custom threshold (auto-calibrated if None)
    value_function=None           # Learned value function
)
```

## Examples

### Example 1: Basic Multi-Deal Session

```python
from optimization.meta.multideal_mode import MultidealManager
from optimization.solvers.heuristic_solver import HeuristicSolver

# Create manager with default settings
manager = MultidealManager(
    starting_bankroll=500.0,
    max_rerolls=3,
    solver=HeuristicSolver()
)

# Run session
session = manager.run_session(max_deals=10, verbose=True)

# Results
print(f"Profit: ${session.profit:+.2f}")
print(f"Win rate: {session.win_rate*100:.1f}%")
```

### Example 2: AI-Powered Strategy

```python
from optimization.meta.ai_deal_strategy import create_ai_deal_strategy
from optimization.meta.multideal_mode import MultidealManager

# Create AI strategy
selector, acceptor = create_ai_deal_strategy(
    strategy_type="q_learning",
    policy_path="policies/q_policy.json"
)

# Use with manager
manager = MultidealManager(
    starting_bankroll=500.0,
    solver=HeuristicSolver(),
    selector=selector,
    acceptor=acceptor
)

session = manager.run_session(max_deals=10)
```

### Example 3: Strategy Comparison

```python
from optimization.meta.ai_deal_strategy import compare_deal_strategies

# Compare different strategies
compare_deal_strategies(
    num_sessions=5,
    deals_per_session=10,
    start_seed=1000
)
```

## Training Policies

### Q-Learning

```python
from optimization.solvers.rl.q_learning import train_q_policy

policy = train_q_policy(
    num_episodes=1000,
    alpha=0.005,        # Learning rate
    gamma=0.95,         # Discount factor
    epsilon=0.15,       # Exploration rate
    save_path="policies/q_policy.json"
)
```

### TD Learning

```python
from optimization.solvers.rl.td_learning import train_td_policy

policy = train_td_policy(
    num_episodes=1000,
    alpha=0.01,         # Learning rate
    gamma=0.95,         # Discount factor
    lambda_=0.7,        # Eligibility trace decay
    save_path="policies/td_policy.json"
)
```

## Future Enhancements

Potential improvements:
1. **Policy-based solver**: Direct integration of learned policies for gameplay
2. **Deep Q-Learning**: Neural network function approximation
3. **MCTS for deal selection**: Use tree search for deal evaluation
4. **Risk-sensitive strategies**: Adaptive strategy based on bankroll
5. **Multi-armed bandits**: Explore/exploit trade-off for deal selection
6. **Transfer learning**: Pre-train on similar card games

## References

- Watkins & Dayan (1992), "Q-learning"
- Sutton & Barto, "Reinforcement Learning: An Introduction"
- Ferguson (1989), "Who Solved the Secretary Problem?"
- User insights from domain expertise

## See Also

- `docs/algorithms.md` - Deep dive on search algorithms
- `optimization/meta/strategic_solver.py` - Strategic insights
- `optimization/meta/deal_evaluator.py` - Deal quality evaluation
- `optimization/solvers/rl/td_learning.py` - TD Learning implementation
