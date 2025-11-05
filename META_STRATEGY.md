# Meta-Strategy: Multi-Deal Optimization for Vegas Solitaire

## The Real Problem

In actual Vegas Solitaire, you **don't just play one deal**. The real strategic problem is:

> **Given N rerolls, which deal should you choose to play, and how should you play it?**

This is fundamentally different from single-deal optimization!

---

## Problem Formulation

### Scenario
- You have **R rerolls** (e.g., 5 chances to see different deals)
- You can "peek" at each deal's initial configuration
- You must **commit to one deal** and play it out
- Goal: **Maximize expected profit** across your session

### Optimal Stopping Problem
This is a classic **optimal stopping** problem:
- Each deal has an unknown value (depends on hidden cards)
- You observe some information (visible cards)
- You must decide: **"Play this deal" vs "See next deal"**
- Trade-off: Current deal might be good, but next could be better

---

## User Insights (Critical!)

### 1. Immediate Playability
> "You shouldn't play a deal without that many cards that can be immediately, or quickly used"

**Operationalize**:
- Count **immediately playable cards** at deal start
- Aces visible → can start foundations
- Cards that can move to tableau immediately
- Empty columns available for Kings

**Heuristic**: Reject deals with < 3 immediately playable moves

### 2. Short Stack Strategy
> "Sometimes you want to take cards from shorter stacks to open up spaces to place Kings"

**When to prioritize short stacks**:
- You have Kings visible or likely in stock
- Need empty columns for reorganization
- Short stacks = fewer moves to clear

**Implementation**: Bonus for revealing cards in short columns when Kings available

### 3. Long Stack Strategy
> "Sometimes you want to take from bigger stacks to work on getting out the deeper hidden cards"

**When to prioritize long stacks**:
- Early in game (information gathering)
- Long stacks have more hidden potential
- Could unlock key cards (Aces, needed ranks)

**Implementation**: Early-game bonus for revealing deeply hidden cards

---

## Deal Evaluation Framework

### Fast Features (Observable at Deal Time)

#### Immediate Playability (0-5 seconds of lookahead)
```python
def evaluate_immediate_playability(state: GameState) -> float:
    """
    How many moves can be made immediately?
    High score = deal has immediate options.
    """
    score = 0

    # 1. Visible Aces (can start foundations immediately)
    for col in range(7):
        visible = state.get_tableau_visible_cards(col)
        for card in visible:
            if card.rank == Rank.ACE:
                score += 10  # Very valuable!

    # 2. Tableau-to-tableau moves available
    valid_moves = get_valid_moves(state)
    tableau_moves = [m for m in valid_moves
                     if m.move_type == MoveType.TABLEAU_TO_TABLEAU]
    score += len(tableau_moves) * 2

    # 3. Empty columns available
    empty_cols = sum(1 for col in state.tableau if len(col) == 0)
    score += empty_cols * 5

    # 4. Kings visible (can use empty columns)
    kings_visible = 0
    for col in range(7):
        visible = state.get_tableau_visible_cards(col)
        for card in visible:
            if card.rank == Rank.KING:
                kings_visible += 1
    if empty_cols > 0 and kings_visible > 0:
        score += 10  # Synergy!

    return score
```

#### Stack Configuration
```python
def evaluate_stack_configuration(state: GameState) -> dict:
    """
    Analyze stack depths and hidden card distribution.
    """
    stack_depths = []
    hidden_counts = []

    for col in range(7):
        visible = state.get_tableau_visible_cards(col)
        hidden = state.tableau_hidden[col]

        stack_depths.append(len(visible))
        hidden_counts.append(hidden)

    return {
        'avg_visible': np.mean(stack_depths),
        'max_hidden': max(hidden_counts),
        'total_hidden': sum(hidden_counts),
        'short_stacks': sum(1 for d in stack_depths if d <= 2),
        'long_stacks': sum(1 for d in stack_depths if d >= 5),
        'empty_cols': sum(1 for col in state.tableau if len(col) == 0)
    }
```

#### Hidden Card Potential
```python
def estimate_hidden_potential(state: GameState) -> float:
    """
    Estimate value of hidden cards (probabilistic).

    User insight: Long stacks might have valuable hidden cards.
    """
    score = 0

    # More hidden cards = more uncertainty = higher variance
    total_hidden = sum(state.tableau_hidden)

    # Expected number of Aces in hidden cards
    cards_seen = 28 - total_hidden  # Initial 28 cards dealt
    aces_seen = count_aces_visible(state)
    expected_hidden_aces = (4 - aces_seen) * (total_hidden / (52 - cards_seen))

    score += expected_hidden_aces * 15  # Aces are very valuable

    # Bonus for deep stacks (more potential)
    for col in range(7):
        hidden = state.tableau_hidden[col]
        if hidden >= 4:  # Deep stack
            score += hidden * 2  # More chances for good cards

    return score
```

### Deal Quality Score

Combine all factors:

```python
def evaluate_deal_quality(state: GameState) -> float:
    """
    Overall deal quality score.
    Higher = better deal to play.

    Incorporates user insights about immediate playability
    and stack configuration.
    """
    score = 0

    # 1. Immediate playability (critical!)
    immediate = evaluate_immediate_playability(state)
    score += immediate * 10  # Weight heavily

    # 2. Stack configuration
    config = evaluate_stack_configuration(state)

    # Prefer some empty columns (for Kings)
    score += config['empty_cols'] * 8

    # Prefer mix of short and long stacks (flexibility)
    if config['short_stacks'] >= 2 and config['long_stacks'] >= 2:
        score += 15  # Good mix!

    # 3. Hidden potential
    potential = estimate_hidden_potential(state)
    score += potential * 0.5  # Lower weight (speculative)

    # 4. Foundation head start
    foundation_cards = state.get_foundation_count()
    score += foundation_cards * 20  # Already made progress!

    return score
```

---

## Optimal Stopping Strategy

### Secretary Problem Approach

Classic result: With N candidates, look at first N/e ≈ 0.37N, then pick the next one better than all seen.

**Applied to Vegas Solitaire**:
- With 5 rerolls, look at first ~2 deals
- Track the **best deal quality seen**
- Starting from deal 3, play first deal that beats the best of first 2

### Threshold Strategy

Set a **quality threshold** based on experience:

```python
class DealSelector:
    """
    Optimal stopping for deal selection.
    """

    def __init__(self, num_rerolls: int = 5):
        self.num_rerolls = num_rerolls
        self.threshold_percentile = 70  # Play deals in top 30%

        # Calibrate threshold from historical data
        self.quality_threshold = self._calibrate_threshold()

    def _calibrate_threshold(self) -> float:
        """
        Run many random deals, find 70th percentile quality.
        """
        qualities = []
        for seed in range(1000):
            game = Game(seed=seed)
            game.deal()
            quality = evaluate_deal_quality(game.state)
            qualities.append(quality)

        return np.percentile(qualities, self.threshold_percentile)

    def select_deal(self, max_deals: int = 5) -> tuple[int, float]:
        """
        Examine up to max_deals, select best using optimal stopping.

        Returns:
            (chosen_seed, quality_score)
        """
        deals_seen = []

        # Phase 1: Observation (first 37% of deals)
        observe_count = int(max_deals * 0.37)

        for i in range(observe_count):
            game = Game(seed=i)
            game.deal()
            quality = evaluate_deal_quality(game.state)
            deals_seen.append((i, quality))

        best_observed = max(deals_seen, key=lambda x: x[1])[1]

        # Phase 2: Selection (remaining deals)
        for i in range(observe_count, max_deals):
            game = Game(seed=i)
            game.deal()
            quality = evaluate_deal_quality(game.state)

            # Take first deal better than best observed
            if quality > best_observed:
                return (i, quality)

            deals_seen.append((i, quality))

        # Fallback: Pick best of all seen
        return max(deals_seen, key=lambda x: x[1])
```

---

## Strategic Insights Integration

### Context-Dependent Stack Selection

Based on user insights, stack selection depends on **game phase** and **available resources**:

```python
def prioritize_stack_moves(state: GameState, valid_moves: List[Move]) -> List[Move]:
    """
    Rank tableau moves based on strategic context.

    Incorporates user insights:
    - Short stacks when need empty columns
    - Long stacks when need information
    """
    # Context analysis
    empty_cols = sum(1 for col in state.tableau if len(col) == 0)
    kings_available = count_kings_available(state)
    game_phase = estimate_game_phase(state)  # Early/Mid/Late

    scored_moves = []

    for move in valid_moves:
        if move.move_type != MoveType.TABLEAU_TO_TABLEAU:
            continue

        score = 0
        source_col = state.tableau[move.source]
        source_hidden = state.tableau_hidden[move.source]
        source_visible = len(source_col) - source_hidden

        # User insight 1: Prioritize short stacks when need empty columns
        if empty_cols == 0 and kings_available > 0:
            if source_visible <= 3:  # Short stack
                score += 20
                if source_visible == 1:  # Will empty the column!
                    score += 30

        # User insight 2: Prioritize long stacks early game
        if game_phase == 'early':
            if source_hidden >= 4:  # Deep hidden cards
                score += 15

        # Always value revealing hidden cards
        if source_hidden > 0:
            score += 10

        scored_moves.append((move, score))

    # Return moves sorted by score (best first)
    scored_moves.sort(key=lambda x: x[1], reverse=True)
    return [m for m, s in scored_moves]
```

### Game Phase Detection

```python
def estimate_game_phase(state: GameState) -> str:
    """
    Determine if we're in early/mid/late game.
    """
    foundation_count = state.get_foundation_count()
    passes = state.passes_through_deck

    if foundation_count < 10 and passes == 0:
        return 'early'  # Still exploring
    elif foundation_count < 25 and passes <= 2:
        return 'mid'    # Building foundations
    else:
        return 'late'   # Finishing or stuck
```

---

## Learning from Experience

### Reinforcement Learning for Deal Selection

Train a **value function** V(deal_features) to predict expected score:

```python
class DealValueLearner:
    """
    Learn to predict deal value from initial configuration.
    """

    def __init__(self):
        # Features: immediate playability, stack config, etc.
        self.weights = np.zeros(15)

    def extract_features(self, state: GameState) -> np.ndarray:
        """Extract features from initial deal."""
        features = []

        # Immediate playability features
        features.append(evaluate_immediate_playability(state) / 50.0)

        # Stack configuration
        config = evaluate_stack_configuration(state)
        features.append(config['empty_cols'] / 7.0)
        features.append(config['short_stacks'] / 7.0)
        features.append(config['long_stacks'] / 7.0)
        features.append(config['total_hidden'] / 28.0)

        # ... more features

        return np.array(features)

    def predict_value(self, state: GameState) -> float:
        """Predict expected score for this deal."""
        features = self.extract_features(state)
        return np.dot(self.weights, features)

    def train(self, num_deals: int = 5000):
        """
        Train by playing many deals and observing outcomes.

        For each deal:
        1. Extract initial features
        2. Play it out (using best solver)
        3. Observe final score
        4. Update weights to predict better
        """
        for seed in range(num_deals):
            game = Game(seed=seed)
            game.deal()

            initial_features = self.extract_features(game.state)

            # Play deal (use Beam Search or TD policy)
            solver = BeamSearch(beam_width=50, max_depth=30)
            while True:
                move = solver.choose_move(game)
                if not move or not game.make_move(move):
                    break

            final_score = game.state.score

            # TD update: V(s) ← V(s) + α[r - V(s)]
            predicted = self.predict_value(game.state)
            error = final_score - predicted
            self.weights += 0.01 * error * initial_features

            if seed % 100 == 0:
                print(f"Trained on {seed} deals...")
```

---

## Expected Value Calculation

### Multi-Deal Expected Value

With R rerolls and selection strategy:

```python
def expected_value_multi_deal(selector: DealSelector,
                              solver: Solver,
                              num_rerolls: int = 5,
                              num_trials: int = 1000) -> float:
    """
    Calculate expected value of multi-deal strategy.

    Returns average profit across many sessions.
    """
    total_profit = 0

    for trial in range(num_trials):
        # Generate candidate deals
        base_seed = trial * num_rerolls

        # Select best deal
        chosen_seed, quality = selector.select_deal_from_seeds(
            seeds=[base_seed + i for i in range(num_rerolls)]
        )

        # Play chosen deal
        game = Game(seed=chosen_seed)
        game.deal()

        while True:
            move = solver.choose_move(game)
            if not move or not game.make_move(move):
                break

        profit = game.state.score + 52  # Normalize (paid $52 to play)
        total_profit += profit

    return total_profit / num_trials
```

---

## Implementation Roadmap

### Phase 1: Deal Evaluation (This Session)
1. ✅ Implement `evaluate_immediate_playability()`
2. ✅ Implement `evaluate_stack_configuration()`
3. ✅ Implement `evaluate_deal_quality()`
4. ✅ Test on 1000 random deals

### Phase 2: Optimal Stopping
1. ✅ Implement `DealSelector` with secretary problem strategy
2. ✅ Calibrate quality thresholds
3. ✅ Benchmark against random selection

### Phase 3: Strategic Insights
1. ✅ Implement `prioritize_stack_moves()` with user insights
2. ✅ Implement game phase detection
3. ✅ Integrate into existing solvers

### Phase 4: Learning
1. Train `DealValueLearner` on 5000 deals
2. Compare learned predictor vs hand-crafted evaluator
3. Use learned values for deal selection

### Phase 5: Integration
1. Add "Multi-Deal Mode" to Streamlit app
2. Show deal comparisons side-by-side
3. Visualize why certain deals are better

---

## Experimental Questions

1. **How many rerolls are worth it?**
   - Measure expected value vs number of rerolls
   - Find diminishing returns point

2. **Is immediate playability predictive?**
   - Correlation between initial playability and final score
   - Can we predict winners from initial state?

3. **Short vs long stack strategy**
   - When does each strategy perform better?
   - Can we learn optimal switching point?

4. **How much does deal selection matter?**
   - Compare: random deal vs best of 5
   - Measure profit improvement

---

## Success Metrics

**Deal Selection**:
- Choose deals in top 30% of quality distribution
- Beat random selection by >20% expected value

**Strategic Integration**:
- Solvers using insights should beat baseline
- Measurable improvement in games with strategic decisions

**Learning**:
- Learned deal evaluator should predict final score (R² > 0.3)
- Deal selector using learned values beats hand-crafted

---

## References

- **Optimal Stopping**: Ferguson (1989), "Who Solved the Secretary Problem?"
- **Vegas Solitaire**: Klondike with draw-3, 3-pass, -$52 buy-in, +$5 per card
- **User Insights**: Critical domain knowledge from actual play experience!
