# Trained TD Learning Policies

This directory contains trained TD Learning policies for Vegas Solitaire.

## Available Policies

### 1. `td0_standard.json`
**Configuration**:
- Algorithm: TD(0) - Basic temporal difference learning
- Learning rate (α): 0.01
- Discount (γ): 0.95
- Eligibility (λ): 0.0
- Exploration (ε): 0.1
- Training episodes: 1000

**Performance**:
- Training (last 100 eps): -$9.0, 8.6/52 cards
- Test set (100 games): -$29.9, 4.4/52 cards
- Best test game: $38, 18/52 cards

**Use when**: You want a stable, well-tested baseline policy.

---

### 2. `td_lambda_standard.json` ⭐ **RECOMMENDED**
**Configuration**:
- Algorithm: TD(λ) - With eligibility traces
- Learning rate (α): 0.01
- Discount (γ): 0.95
- Eligibility (λ): 0.7
- Exploration (ε): 0.1
- Training episodes: 1000

**Performance**:
- Training (last 100 eps): -$6.5, 9.1/52 cards ← Best training performance!
- Test set (100 games): -$26.9, 5.0/52 cards
- Best test game: $33, 17/52 cards

**Use when**: You want the best-performing policy from standard hyperparameters.

**Why it's better**: Eligibility traces allow better credit assignment - actions that led to good outcomes earlier in the game get appropriate credit.

---

### 3. `td0_fast_learning.json`
**Configuration**:
- Algorithm: TD(0) - With faster learning
- Learning rate (α): 0.05  ← 5x higher!
- Discount (γ): 0.95
- Eligibility (λ): 0.0
- Exploration (ε): 0.2  ← More exploration
- Training episodes: 1000

**Performance**:
- Training (last 100 eps): TBD (training in progress)
- Test set (100 games): TBD
- Peak episodes: +$9.1, 12.2/52 cards

**Use when**: You want aggressive learning with higher variance.

**Trade-off**: Higher learning rate = faster adaptation but more instability. Some episodes achieve 10-12 cards, others regress.

---

## How to Use

### Load a Policy

```python
from optimization.solvers.rl.td_learning import TDPolicy

# Load the recommended policy
policy = TDPolicy()
policy.load("trained_policies/td_lambda_standard.json")
```

### Play a Game

```python
from game.core.game import Game

game = Game(seed=42)
game.deal()

moves_made = 0
while moves_made < 500:
    valid_moves = game.get_valid_moves()
    if not valid_moves:
        break

    # Use policy to choose move
    move = policy.choose_action(game.state, valid_moves)
    if not move or not game.make_move(move):
        break

    moves_made += 1

print(f"Final: ${game.state.score}, {game.state.get_foundation_count()}/52 cards")
```

### Integrate with Streamlit

```python
# In your Streamlit app
import streamlit as st
from optimization.solvers.rl.td_learning import TDPolicy

# Load policy once
@st.cache_resource
def load_policy():
    policy = TDPolicy()
    policy.load("trained_policies/td_lambda_standard.json")
    return policy

policy = load_policy()

# Use in game
if st.button("Make AI Move"):
    move = policy.choose_action(st.session_state.game.state,
                                st.session_state.game.get_valid_moves())
    st.session_state.game.make_move(move)
    st.rerun()
```

---

## Policy Comparison

| Metric | TD(0) Standard | TD(λ) Standard | Fast Learning |
|--------|---------------|----------------|---------------|
| **Training Avg Score** | -$9.0 | **-$6.5** ⭐ | TBD |
| **Training Avg Foundation** | 8.6/52 | **9.1/52** ⭐ | ~10/52 |
| **Test Avg Score** | -$29.9 | **-$26.9** ⭐ | TBD |
| **Test Avg Foundation** | 4.4/52 | **5.0/52** ⭐ | TBD |
| **Best Test Game** | $38, 18/52 | $33, 17/52 | TBD |
| **Learning Rate** | Low | Low | High |
| **Stability** | Stable | Stable | Variable |

**Conclusion**: TD(λ) Standard offers the best balance of performance and stability.

---

## Training Details

All policies were trained on:
- Seeds 0-999 (1000 training games)
- Evaluated on seeds 10000-10099 (100 test games)
- Maximum 500 moves per episode
- Features: 24-dimensional state representation
- Function approximation: Linear (w^T φ(s))

Training time:
- ~10-15 minutes per configuration on modern CPU
- ~30-45 minutes total for all 3 policies

---

## Improving Policies

### Train Longer
```python
from optimization.solvers.rl.td_learning import train_td_policy

# Train for 5000 episodes instead of 1000
policy = train_td_policy(
    num_episodes=5000,
    alpha=0.01,
    gamma=0.95,
    lambda_=0.7,
    save_path="trained_policies/td_lambda_long.json"
)
```

### Tune Hyperparameters
```python
# Try different learning rates
for alpha in [0.001, 0.01, 0.05, 0.1]:
    policy = train_td_policy(
        num_episodes=1000,
        alpha=alpha,
        save_path=f"trained_policies/td_alpha_{alpha}.json"
    )
```

### Use Extended Features
```python
from optimization.solvers.rl.td_learning import TDLearner

learner = TDLearner(
    use_extended_features=True  # 33 features instead of 24
)
policy = learner.train(num_episodes=2000)
```

---

## Understanding the Numbers

### Why are test scores lower than training?
**Overfitting** - The policy learns patterns specific to the training seed range. This is normal in RL.

**Solution**: Train on more diverse seeds (10K+ episodes).

### Why no wins?
Vegas Solitaire Draw-3 is **very hard**. Even expert human players only win ~15% of deals.

Our policies are getting 4-5 cards on average, which is reasonable progress. With more training and better algorithms (Q-Learning, Deep RL), we can improve.

### What's a good score?
- **Breaking even ($0)**: Excellent! (~10 foundation cards)
- **Positive score**: Very good!
- **-$10 to -$20**: Decent progress (7-8 cards)
- **-$30 to -$40**: Learning but not optimal

### Can policies improve over time?
Yes! Continue training from a saved policy:

```python
policy = TDPolicy()
policy.load("trained_policies/td_lambda_standard.json")

learner = TDLearner()
learner.weights = policy.feature_weights  # Resume from saved weights

# Train more
learner.train(num_episodes=2000, start_seed=1000)
new_policy = learner.get_policy()
new_policy.save("trained_policies/td_lambda_extended.json")
```

---

## Troubleshooting

### Policy plays poorly
**Possible causes**:
1. Policy overfitted to training seeds
2. Not enough training episodes
3. Features don't capture important information

**Solutions**:
- Train longer (5000+ episodes)
- Use extended features
- Try different hyperparameters

### Policies seem random
**Possible causes**:
1. Feature weights are all near zero
2. Learning rate too low
3. Not enough exploration during training

**Solutions**:
- Increase learning rate
- Check policy.feature_weights - should have varying magnitudes
- Increase training episodes

### How to compare policies?
Run benchmark:

```python
from optimization.evaluation.benchmark import Benchmark

policies = ["td0_standard", "td_lambda_standard", "td0_fast_learning"]

for name in policies:
    policy = TDPolicy()
    policy.load(f"trained_policies/{name}.json")

    benchmark = Benchmark(test_set_name=name)
    result = benchmark.evaluate_solver(policy, num_games=100, start_seed=20000)

    print(f"{name}: {result.average_score}, {result.average_foundation_count}/52")
```

---

## Next Steps

1. **Use the best policy** (td_lambda_standard.json) in your Streamlit app
2. **Train longer** (5000+ episodes) for better performance
3. **Try advanced algorithms** (Q-Learning, Deep RL)
4. **Experiment with features** (add/remove features, see impact)
5. **Ensemble policies** (combine multiple for robustness)

See `TRAINING_GUIDE.md` for comprehensive training instructions.

See `ADVANCED_ALGORITHMS.md` for other algorithms to try.
