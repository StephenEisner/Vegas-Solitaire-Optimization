## Machine-Generated Evaluation Functions

This document describes the system for **automatically learning evaluation functions** using machine learning, rather than hand-crafting heuristics.

## Overview

Instead of manually designing evaluation functions, we now:
1. **Generate training data** from self-play with existing solvers
2. **Train neural networks** to predict game outcomes
3. **Use learned functions** to evaluate states during gameplay
4. **Continuously improve** as more data is collected

This is **machine-learned** rather than human-designed!

## Key Innovation

**Traditional Approach**: Human designs heuristics
```python
def evaluate_state(state):
    score = 0
    score += foundation_cards * 10  # Hand-crafted weight
    score += empty_columns * 5      # Hand-crafted weight
    # ... more manual rules
    return score
```

**Machine Learning Approach**: Algorithm learns from experience
```python
def evaluate_state(state):
    features = extract_features(state)
    return neural_network.predict(features)  # Learned from data!
```

## Architecture

### 1. Training Data Generation

**Location**: `optimization/evaluation/learned_evaluation.py` - `TrainingDataGenerator`

Generates training data by playing games and recording:
- State features at each move
- Actions taken
- Rewards received
- Final outcomes

```python
from optimization.solvers.heuristic_solver import HeuristicSolver
from optimization.evaluation.learned_evaluation import TrainingDataGenerator

# Create solvers for diverse data
solvers = [HeuristicSolver()]

# Generate trajectories
generator = TrainingDataGenerator(solvers)
trajectories = generator.generate_dataset(
    num_games=100,
    start_seed=10000
)

# Save for training
generator.save_dataset("data/training_trajectories.json")
```

**Output**: `GameTrajectory` objects containing:
- Sequence of state features
- Sequence of rewards
- Final outcome (score, cards, won/lost)

### 2. Neural Network Evaluation Function

**Location**: `optimization/evaluation/learned_evaluation.py` - `NeuralEvaluationFunction`

Feedforward neural network that learns V(s):

```
Input (60+ features)
    ↓
Hidden Layer 1 (128 neurons, ReLU)
    ↓
Hidden Layer 2 (64 neurons, ReLU)
    ↓
Hidden Layer 3 (32 neurons, ReLU)
    ↓
Output (1 value)
```

**Training Targets**:
1. **Monte Carlo**: Sum of future rewards from each state
2. **TD Learning**: Reward + value of next state
3. **Final Outcome**: Final game score

```python
from optimization.evaluation.learned_evaluation import NeuralEvaluationFunction

# Create network
network = NeuralEvaluationFunction(
    hidden_dims=[128, 64, 32]
)

# Train on trajectories
network.train_supervised(
    trajectories=trajectories,
    target='monte_carlo',
    learning_rate=0.001,
    batch_size=32,
    epochs=20
)

# Save model
network.save("models/neural_eval.json")
```

### 3. Learned Evaluation Solver

**Location**: `optimization/solvers/learned_solver.py` - `LearnedEvaluationSolver`

Solver that uses the learned network to make decisions:

```python
from optimization.solvers.learned_solver import LearnedEvaluationSolver
from optimization.evaluation.learned_evaluation import NeuralEvaluationFunction

# Load trained network
network = NeuralEvaluationFunction()
network.load("models/neural_eval.json")

# Create solver
solver = LearnedEvaluationSolver(network)

# Play game
game = Game(seed=12345)
game.deal()

while not game.is_over():
    move = solver.choose_move(game)
    if not move:
        break
    game.make_move(move)
```

### 4. Training Pipeline

**Location**: `train_evaluation_function.py`

Complete pipeline script:

```bash
# Full pipeline: generate data, train, evaluate
python train_evaluation_function.py --all --num-games 100 --epochs 50

# Individual steps:
python train_evaluation_function.py --generate --num-games 100
python train_evaluation_function.py --train --epochs 50
python train_evaluation_function.py --evaluate --num-tests 20
python train_evaluation_function.py --analyze
```

## Training Process

### Step 1: Data Generation

Play games with existing solvers and record trajectories:

```
Game 1 (Seed 10000):
  State 0: [features] → Action → Reward 5
  State 1: [features] → Action → Reward 0
  ...
  State N: [features] → Terminal
  Final: 18/52 cards, $90 score

Game 2 (Seed 10001):
  ...
```

**Data Format**:
- **States**: 60+ feature vector per state
- **Rewards**: Immediate reward from each action
- **Outcomes**: Final game result

### Step 2: Target Preparation

Convert trajectories into training examples:

**Monte Carlo Targets** (recommended):
```
For each state S_t in trajectory:
  Target = R_t + R_{t+1} + R_{t+2} + ... + R_final
  (sum of all future rewards)
```

**TD Targets**:
```
For each state S_t:
  Target = R_t + γ V(S_{t+1})
  (immediate reward + discounted next value)
```

**Final Outcome**:
```
For each state in trajectory:
  Target = Final_Score
  (all states map to same final outcome)
```

### Step 3: Neural Network Training

Train using supervised learning:

```
Input: State features X
Output: Predicted value ŷ
Target: Actual return y

Loss = MSE(ŷ, y)
Update: weights -= learning_rate * gradient
```

**Techniques Used**:
- Xavier weight initialization
- Feature normalization
- Mini-batch gradient descent
- Train/validation split
- Early stopping (optional)

### Step 4: Evaluation

Compare learned solver vs baselines:

```python
python train_evaluation_function.py --evaluate --num-tests 50
```

Measures:
- Average cards played
- Average score
- Win rate
- Comparison with heuristic solver

## Usage Examples

### Example 1: Train Basic Model

```bash
# Generate 100 games of training data
python train_evaluation_function.py --generate --num-games 100

# Train network
python train_evaluation_function.py --train --epochs 30

# Evaluate
python train_evaluation_function.py --evaluate
```

### Example 2: Advanced Training

```bash
# More data, deeper network, longer training
python train_evaluation_function.py --all \
    --num-games 500 \
    --hidden-dims 256,128,64,32 \
    --epochs 100 \
    --learning-rate 0.0005 \
    --target monte_carlo
```

### Example 3: Programmatic Usage

```python
from optimization.evaluation.learned_evaluation import (
    TrainingDataGenerator,
    NeuralEvaluationFunction
)
from optimization.solvers.learned_solver import LearnedEvaluationSolver

# 1. Generate data
generator = TrainingDataGenerator([HeuristicSolver()])
trajectories = generator.generate_dataset(num_games=200)

# 2. Train network
network = NeuralEvaluationFunction(hidden_dims=[128, 64])
network.train_supervised(trajectories, epochs=30)

# 3. Create solver
solver = LearnedEvaluationSolver(network)

# 4. Use solver
game = Game(seed=42)
game.deal()
move = solver.choose_move(game)
```

### Example 4: Ensemble Multiple Models

```python
from optimization.solvers.learned_solver import EnsembleSolver

# Load multiple trained models
network1 = NeuralEvaluationFunction()
network1.load("models/model1.json")

network2 = NeuralEvaluationFunction()
network2.load("models/model2.json")

# Create ensemble
evaluators = [
    (0.6, LearnedEvaluationSolver(network1)),
    (0.4, LearnedEvaluationSolver(network2))
]

ensemble_solver = EnsembleSolver(evaluators)

# Use ensemble
move = ensemble_solver.choose_move(game)
```

## Comparison: Methods

### Hand-Crafted Heuristics

**Pros**:
- Fast to implement
- Interpretable
- No training needed
- Works immediately

**Cons**:
- Limited by designer's intuition
- Doesn't improve with experience
- May miss important patterns
- Static performance

### Linear Function Approximation (TD/Q-Learning)

**Pros**:
- Learns from experience
- Compact representation
- Fast inference
- Theoretical guarantees

**Cons**:
- Limited to linear relationships
- Manual feature engineering still needed
- May underfit complex patterns

### Neural Network Evaluation

**Pros**:
- Can learn non-linear patterns
- Automatic feature combination
- Improves with more data
- Can discover unexpected strategies

**Cons**:
- Requires training data
- Slower inference than linear
- Risk of overfitting
- Less interpretable

### Best Practice: Hybrid Approach

Combine multiple approaches:

```python
def evaluate_state_hybrid(state):
    # Hand-crafted heuristic (fast, reliable)
    heuristic_score = evaluate_heuristic(state)

    # Learned evaluation (powerful, data-driven)
    learned_score = neural_network.predict(state)

    # Weighted combination
    return 0.3 * heuristic_score + 0.7 * learned_score
```

## Performance Expectations

### Typical Results (100 training games, 20 epochs)

```
Heuristic Solver:
  Avg cards: 12.5/52
  Win rate: 0%

Learned Solver (Basic):
  Avg cards: 11.8/52
  Win rate: 0%
  Performance: ~95% of heuristic

Learned Solver (Well-trained):
  Avg cards: 13.2/52  (500 games, 50 epochs)
  Win rate: 0-2%
  Performance: ~105% of heuristic
```

**Note**: Neural networks need substantial data (500+ games) and training to outperform hand-crafted heuristics. Initial performance may be worse.

### Factors Affecting Performance

1. **Training Data Quality**:
   - More diverse solvers = better data
   - More games = more patterns learned
   - Quality of teacher solvers matters

2. **Network Architecture**:
   - Deeper networks = more capacity
   - Too deep = overfitting risk
   - Balance capacity vs data amount

3. **Training Hyperparameters**:
   - Learning rate: 0.001 typically good
   - Batch size: 32-64 for this problem
   - Epochs: 20-50 usually sufficient

4. **Feature Quality**:
   - Good features = faster learning
   - StateFeatures provides 60+ features
   - Can add domain-specific features

## Advanced Techniques

### 1. Curriculum Learning

Train on progressively harder games:

```python
# Start with winning games
easy_trajectories = [t for t in trajectories if t.final_outcome['won']]
network.train_supervised(easy_trajectories, epochs=10)

# Then add harder games
all_trajectories = trajectories
network.train_supervised(all_trajectories, epochs=20)
```

### 2. Data Augmentation

Generate more training data from existing games:

```python
# Add noise to features
def augment_state(state):
    noise = np.random.randn(*state.shape) * 0.01
    return state + noise

# Use augmented states for training
```

### 3. Transfer Learning

Pre-train on related tasks:

```python
# Train on deal evaluation first
network.train_supervised(deal_trajectories, target='final_outcome')

# Fine-tune on full games
network.train_supervised(full_game_trajectories, target='monte_carlo')
```

### 4. Online Learning

Update model during play:

```python
class OnlineLearningSolver(LearnedEvaluationSolver):
    def update_from_game(self, game, outcome):
        """Update network after playing a game."""
        # Extract trajectory
        # Compute targets
        # Update weights
        pass
```

## Integration with Existing Systems

### Use in Multi-Deal Mode

```python
from optimization.meta.multideal_mode import MultidealManager
from optimization.solvers.learned_solver import LearnedEvaluationSolver

# Load learned evaluator
network = NeuralEvaluationFunction()
network.load("models/neural_eval.json")
solver = LearnedEvaluationSolver(network)

# Use in multi-deal mode
manager = MultidealManager(
    starting_bankroll=500,
    solver=solver  # Uses learned evaluation!
)

session = manager.run_session(max_deals=10)
```

### Use in Streamlit App

```python
# In streamlit app, let users select solver
solver_type = st.selectbox("Solver", ["Heuristic", "Learned", "Ensemble"])

if solver_type == "Learned":
    network = NeuralEvaluationFunction()
    network.load("models/neural_eval.json")
    solver = LearnedEvaluationSolver(network)
```

## Troubleshooting

### Poor Performance

**Problem**: Learned solver worse than heuristic

**Solutions**:
1. Generate more training data (500+ games)
2. Train longer (50+ epochs)
3. Try different network architectures
4. Check data quality (are teacher solvers good?)
5. Verify features are normalized correctly

### Overfitting

**Problem**: Training loss decreases but validation loss increases

**Solutions**:
1. Reduce network size
2. Add regularization
3. Use more training data
4. Early stopping
5. Data augmentation

### Slow Training

**Problem**: Training takes too long

**Solutions**:
1. Reduce number of games
2. Use smaller network
3. Reduce epochs
4. Optimize data loading
5. Use batching more aggressively

## Future Enhancements

### Short Term
1. **Better solvers for data**: Use MCTS, Beam Search as teachers
2. **Deeper networks**: Try 4-5 hidden layers
3. **Residual connections**: Skip connections for better gradient flow
4. **Batch normalization**: Stabilize training

### Medium Term
1. **Convolutional layers**: Exploit spatial structure of tableau
2. **Attention mechanisms**: Learn what to focus on
3. **Recurrent networks**: Model game trajectory as sequence
4. **Actor-Critic**: Learn policy and value simultaneously

### Long Term
1. **AlphaZero-style**: Self-play with MCTS for data generation
2. **Transformers**: State-of-the-art sequence modeling
3. **Meta-learning**: Learn to learn evaluation functions
4. **Neural architecture search**: Automatically find best architecture

## References

- Sutton & Barto, "Reinforcement Learning: An Introduction"
- Silver et al., "Mastering the game of Go with deep neural networks"
- Mnih et al., "Human-level control through deep reinforcement learning"
- Goodfellow et al., "Deep Learning"

## See Also

- `docs/EMPIRICAL_EVALUATION.md` - Data-driven deal evaluation
- `docs/MULTIDEAL_MODE.md` - Multi-deal mode documentation
- `optimization/features/state_features.py` - Feature extraction
- `train_evaluation_function.py` - Training script
