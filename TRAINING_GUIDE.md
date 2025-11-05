# Training Guide for Vegas Solitaire Policies

This guide explains how to train reinforcement learning policies for Vegas Solitaire.

---

## Quick Start

### Option 1: Use Pre-Trained Policies (Easiest)

Pre-trained policies are saved in `trained_policies/` after running the training script:

```python
from optimization.solvers.rl.td_learning import TDPolicy

# Load a trained policy
policy = TDPolicy()
policy.load("trained_policies/td0_standard.json")

# Use it to play games
from game.core.game import Game

game = Game(seed=42)
game.deal()

while True:
    valid_moves = game.get_valid_moves()
    if not valid_moves:
        break

    move = policy.choose_action(game.state, valid_moves)
    if not move or not game.make_move(move):
        break

print(f"Final score: ${game.state.score}")
print(f"Foundation: {game.state.get_foundation_count()}/52")
```

### Option 2: Run Full Training Pipeline (Recommended)

Run the automated training script that trains multiple configurations:

```bash
python train_policies.py
```

This will:
- Train 3 different TD Learning configurations
- Each for 1000 episodes
- Evaluate on 100 test games
- Save all policies to `trained_policies/`
- Print comparison results

**Time**: ~10-30 minutes depending on your machine

### Option 3: Custom Training (Advanced)

Train with specific hyperparameters:

```python
from optimization.solvers.rl.td_learning import train_td_policy

# Train custom policy
policy = train_td_policy(
    num_episodes=2000,      # More training
    alpha=0.01,             # Learning rate
    gamma=0.95,             # Discount factor
    lambda_=0.0,            # TD(0) vs TD(λ)
    save_path="my_policy.json"
)
```

---

## Where to Run Training

### 1. Your Local Machine

**Setup**:
```bash
git clone <repo-url>
cd Vegas-Solitaire-Optimization
pip install -r requirements.txt  # If you have one
```

**Run**:
```bash
python train_policies.py
```

**Pros**:
- Full control
- Can run overnight for long training
- Keep all trained models locally

**Cons**:
- Uses your CPU
- Need Python environment set up

---

### 2. Google Colab (For Free GPU/TPU)

Even though we don't need GPU for this simple linear model, Colab is convenient:

**Create a new Colab notebook**:

```python
# Cell 1: Clone repo
!git clone <your-repo-url>
%cd Vegas-Solitaire-Optimization

# Cell 2: Install dependencies (if any)
# !pip install -r requirements.txt

# Cell 3: Train policies
!python train_policies.py

# Cell 4: Download trained policies
from google.colab import files
files.download('trained_policies/td0_standard.json')
```

**Pros**:
- Free compute
- Can leave it running
- Easy to share notebooks

**Cons**:
- Session timeout after inactivity
- Need to download results

---

### 3. GitHub Actions (Automated Training)

Create `.github/workflows/train_policies.yml`:

```yaml
name: Train Policies

on:
  workflow_dispatch:  # Manual trigger
  schedule:
    - cron: '0 0 * * 0'  # Weekly on Sunday

jobs:
  train:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'

      - name: Install dependencies
        run: |
          pip install numpy

      - name: Train policies
        run: python train_policies.py

      - name: Upload trained policies
        uses: actions/upload-artifact@v3
        with:
          name: trained-policies
          path: trained_policies/
```

**Pros**:
- Fully automated
- No local resources used
- Policies stored as artifacts

**Cons**:
- Limited to GitHub Actions time limits
- Need GitHub repo

---

### 4. In Streamlit App (Interactive)

We could add a training page to the Streamlit app:

```python
# In streamlit_app/pages/training_mode.py

import streamlit as st
from optimization.solvers.rl.td_learning import TDLearner

st.title("Train TD Learning Policy")

# Training parameters
num_episodes = st.slider("Training Episodes", 100, 5000, 1000)
alpha = st.slider("Learning Rate (α)", 0.001, 0.1, 0.01)
gamma = st.slider("Discount Factor (γ)", 0.9, 0.99, 0.95)

if st.button("Start Training"):
    learner = TDLearner(alpha=alpha, gamma=gamma)

    progress_bar = st.progress(0)
    status_text = st.empty()

    for ep in range(num_episodes):
        game = Game(seed=ep)
        game.deal()
        learner.train_episode(game)

        progress_bar.progress((ep + 1) / num_episodes)
        status_text.text(f"Episode {ep+1}/{num_episodes}")

    st.success("Training complete!")
    st.download_button("Download Policy",
                      policy_json,
                      "trained_policy.json")
```

**Pros**:
- Visual feedback
- Interactive parameter tuning
- Download policies directly

**Cons**:
- Ties up browser
- Streamlit session limits

---

## Training Parameters

### Learning Rate (α)

Controls how much we update weights each step.

- **Low (0.001-0.01)**: Slow, stable learning
- **Medium (0.01-0.05)**: Good default
- **High (0.05-0.1)**: Fast learning, might oscillate

**Rule of thumb**: Start with 0.01, increase if learning is too slow.

### Discount Factor (γ)

How much we value future rewards.

- **Low (0.9)**: Myopic, focuses on immediate rewards
- **Medium (0.95)**: Balanced
- **High (0.99)**: Far-sighted, plans ahead

**For Vegas Solitaire**: 0.95 works well (game has clear end goal).

### Eligibility Trace (λ)

How much credit to give to earlier states.

- **λ=0 (TD(0))**: Only update most recent state
- **λ=0.5-0.7**: Balance between recent and old states
- **λ=1.0 (Monte Carlo)**: Credit all states equally

**Trade-off**: Higher λ = better credit assignment but higher variance.

### Exploration Rate (ε)

Probability of random move during training.

- **Low (0.05-0.1)**: Mostly exploit learned policy
- **Medium (0.1-0.2)**: Balanced exploration
- **High (0.3-0.5)**: Explore more

**Note**: During evaluation/deployment, set ε=0 (pure exploitation).

### Number of Episodes

How many games to train on.

- **100**: Quick test, won't learn much
- **1000**: Good starting point
- **5000**: Solid training
- **10000+**: Diminishing returns

**Rule of thumb**: Plot learning curves, stop when performance plateaus.

---

## Monitoring Training

### Track These Metrics

1. **Average Score** (per 100 episodes)
   - Should increase over time
   - Plateaus indicate convergence

2. **Average Foundation Cards**
   - More direct measure of progress
   - Target: 10+ cards

3. **Win Rate** (on test set)
   - Ultimate goal
   - Even 5-10% is good for Vegas Draw-3

4. **Weight Magnitude**
   - Check `learner.weights` periodically
   - Should stabilize over time

### Example Training Curve Analysis

```python
import matplotlib.pyplot as plt

# During training, save statistics
scores = []
for episode in range(num_episodes):
    score, _ = learner.train_episode(game)
    scores.append(score)

# Plot learning curve
plt.figure(figsize=(10, 6))
window = 100
moving_avg = [np.mean(scores[max(0, i-window):i+1])
              for i in range(len(scores))]
plt.plot(moving_avg)
plt.xlabel('Episode')
plt.ylabel('Average Score (100-episode window)')
plt.title('TD Learning Training Progress')
plt.savefig('training_curve.png')
```

---

## Saving and Loading Policies

### Save

```python
policy = learner.get_policy()
policy.save("my_trained_policy.json")
```

### Load

```python
from optimization.solvers.rl.td_learning import TDPolicy

policy = TDPolicy()
policy.load("my_trained_policy.json")
```

### Policy File Format

JSON file containing:
```json
{
  "name": "TD-Learned",
  "feature_weights": [0.123, -0.045, ...],
  "use_extended_features": false,
  "metadata": {
    "training_episodes": 1000,
    "alpha": 0.01,
    "gamma": 0.95
  }
}
```

---

## Comparing Multiple Policies

### Benchmark Script

```python
from optimization.evaluation.benchmark import Benchmark
from optimization.solvers.rl.td_learning import TDPolicy

# Load multiple policies
policies = []
for name in ["td0_standard", "td_lambda", "td0_fast"]:
    policy = TDPolicy()
    policy.load(f"trained_policies/{name}.json")
    policies.append(policy)

# Benchmark each
benchmark = Benchmark(test_set_name="policy_comparison")

for policy in policies:
    result = benchmark.evaluate_solver(
        solver=policy,
        num_games=100,
        start_seed=10000
    )
    print(f"{policy.name}: Win Rate {result.win_rate:.1%}, "
          f"Avg Score ${result.average_score:.0f}")
```

---

## Advanced Training Techniques

### 1. Learning Rate Decay

Reduce learning rate over time:

```python
for episode in range(num_episodes):
    # Decay learning rate
    learner.alpha = initial_alpha * (0.99 ** episode)

    learner.train_episode(game)
```

### 2. Curriculum Learning

Start with easier games:

```python
# Train on Draw-1 first, then Draw-3
for episode in range(500):
    game = Game(seed=episode, draw_count=1)  # Easier
    learner.train_episode(game)

for episode in range(500, 1500):
    game = Game(seed=episode, draw_count=3)  # Harder
    learner.train_episode(game)
```

### 3. Experience Replay

Store and replay important games:

```python
replay_buffer = []

for episode in range(num_episodes):
    game = Game(seed=episode)
    score, history = learner.train_episode(game, save_history=True)

    # Save good games
    if score > 0:
        replay_buffer.append(history)

    # Replay occasionally
    if episode % 10 == 0 and replay_buffer:
        replay_history = random.choice(replay_buffer)
        learner.replay_episode(replay_history)
```

### 4. Ensemble Policies

Combine multiple trained policies:

```python
from optimization.policies.base import HybridPolicy

policies = [load_policy(name) for name in policy_names]
ensemble = HybridPolicy("Ensemble", policies, weights=[0.3, 0.3, 0.4])
```

---

## Troubleshooting

### Training isn't improving

**Symptoms**: Score stays flat, no improvement after 500+ episodes

**Solutions**:
1. Increase learning rate (α) from 0.01 → 0.05
2. Increase exploration (ε) from 0.1 → 0.3
3. Check feature engineering - are features informative?
4. Try TD(λ) with λ=0.7 instead of TD(0)

### Scores are getting worse

**Symptoms**: Performance degrades after initial improvement

**Solutions**:
1. Reduce learning rate (α) - it's too high
2. Reduce epsilon (ε) - too much random exploration
3. Use learning rate decay
4. Check for bugs in feature extraction

### Training is too slow

**Symptoms**: Takes hours for 1000 episodes

**Solutions**:
1. Profile code to find bottlenecks
2. Use vectorized numpy operations
3. Reduce max_moves_per_episode
4. Train on fewer features

### Policies aren't saving/loading

**Symptoms**: File errors, JSON decode errors

**Solutions**:
1. Check file permissions
2. Ensure parent directory exists
3. Validate JSON structure
4. Check for NaN values in weights

---

## Next Steps After Training

1. **Evaluate on test set** (different seeds than training)
2. **Compare to baselines** (Random, Heuristic, Beam Search)
3. **Integrate into Streamlit app** (show policy playing)
4. **Analyze learned weights** (which features matter most?)
5. **Try more advanced algorithms** (Q-Learning, Policy Gradient)
6. **Experiment with features** (add/remove features, see impact)

---

## Resources

- **Sutton & Barto**: "Reinforcement Learning: An Introduction" (free online)
- **David Silver's RL Course**: deepmind.com/learning-resources
- **Spinning Up in Deep RL**: openai.com/spinning-up

---

## Questions?

Check `ADVANCED_ALGORITHMS.md` for more details on algorithms and theory.

See `test_advanced_algorithms.py` for working examples.

Run `python train_policies.py --help` for command-line options.
