# Heuristic Solver Improvements

## Current Problems

### 1. **Saddle Points / Cycling**
The solver gets stuck moving cards back and forth (especially Kings).

**Example:**
```
Move 1: K♥ from Col 1 to empty Col 5 → Score: 60
Move 2: K♥ from Col 5 to empty Col 1 → Score: 60
Repeat forever...
```

**Root Cause:** No state history or cycle detection.

### 2. **Myopic Strategy**
Purely greedy evaluation without lookahead or understanding of game progression.

### 3. **Poor Empty Column Usage**
Treats all empty columns equally, doesn't understand strategic value.

---

## Proposed Solutions

### Phase 1: Add Cycle Detection (Immediate Fix)

```python
class HeuristicSolver(Solver):
    def __init__(self):
        super().__init__(name="Heuristic")
        self.state_history = []  # Track recent states
        self.max_history = 10    # Last N states

    def choose_move(self, game: Game):
        # ... existing code ...

        # Filter out moves that create cycles
        non_cycling_moves = []
        for move, score in scored_moves:
            # Simulate move
            next_state = apply_move(game.state, move)

            # Check if we've seen this state recently
            if not self._is_recent_state(next_state):
                non_cycling_moves.append((move, score))

        # If all moves cycle, take the best one anyway
        if not non_cycling_moves:
            return scored_moves[0][0]

        return non_cycling_moves[0][0]
```

### Phase 2: Improve Heuristic Weights

**Problems with current weights:**
- Moving to empty columns scores same regardless of context
- No penalty for non-progressive moves
- No bonus for progress toward win

**Better strategy:**
```python
self.weights = {
    # Direct progress
    'foundation_move': 100,
    'foundation_progress': 20,  # NEW: How many cards already on foundation

    # Information gain
    'reveals_hidden_card': 50,
    'reveals_from_large_stack': 10,  # NEW: Bonus for big stacks

    # Space management
    'empties_column': 80,
    'uses_empty_column_for_king': 100,  # NEW: Only if placing King
    'uses_empty_column_other': -20,     # NEW: Penalty otherwise

    # Sequencing
    'creates_sequence': 10,
    'sequence_length': 3,
    'consolidates_colors': 5,  # NEW: Group by color alternation

    # Waste management
    'moves_from_waste': 15,
    'clears_blocking_waste': 25,  # NEW: If waste blocks foundation

    # Movement penalties
    'tableau_to_tableau_no_reveal': -5,  # NEW: Discourage shuffling
    'moves_to_occupied_empty': -30,      # NEW: Don't waste empty space
}
```

### Phase 3: Add Game Phase Awareness

```python
def get_game_phase(self, state: GameState) -> str:
    """Determine what phase of game we're in."""
    foundation_pct = state.foundation_count / 52

    if foundation_pct < 0.15:
        return 'OPENING'    # Focus: reveal cards, build sequences
    elif foundation_pct < 0.50:
        return 'MIDGAME'    # Focus: foundation moves, create space
    elif foundation_pct < 0.90:
        return 'ENDGAME'    # Focus: forced sequences, planning
    else:
        return 'FINISHING'  # Focus: just finish
```

Adjust weights based on phase:
- **Opening**: Prioritize reveals and exploration
- **Midgame**: Balance foundation moves with setup
- **Endgame**: Be more careful, avoid traps
- **Finishing**: Just complete remaining moves

---

## Phase 4: Multi-Level Strategy (Future)

### The Real Vegas Solitaire Problem

You're absolutely right - the real optimization is:

**Problem:** You pay $52 to see a deal. You can "window shop" multiple deals before deciding which to play. How do you:
1. **Evaluate** if a deal is worth playing?
2. **Decide** when to commit vs. keep looking?
3. **Optimize** expected value across multiple deals?

### This is Actually TWO Problems:

#### Problem A: Single Deal Optimization
*"Given I'm playing this deal, what moves maximize EV?"*
- This is what we're currently solving
- Heuristic, MCTS, etc.

#### Problem B: Deal Selection Meta-Strategy
*"Which deals should I play, and how many should I evaluate?"*
- This is portfolio optimization!
- Needs different techniques

### Proposed Approach for Problem B:

```
Meta-Strategy:
1. See Deal 1 → Evaluate "winnability score" → Record
2. See Deal 2 → Evaluate "winnability score" → Record
3. See Deal N → Decide: Play best seen so far, or keep looking?

Decision criteria:
- Cost: $52 per deal evaluated
- Benefit: Better deals have higher EV
- Tradeoff: Exploration vs. Exploitation

This is a STOPPING PROBLEM (optimal stopping theory!)
```

#### Mathematical Framework:

```
Let:
- C = $52 (cost to see a deal)
- EV(deal) = Expected value of playing a deal
- W(deal) = Winnability score (0-1, estimated quickly)

Goal: Maximize:
  E[Profit] = EV(chosen_deal) - C * num_deals_seen

Strategy:
1. Quick winnability estimator (fast, lightweight)
2. Threshold-based stopping rule
3. Play the best deal seen so far
```

### Quick Winnability Features:

```python
def quick_winnability_score(state: GameState) -> float:
    """
    Fast evaluation of deal quality without playing it.

    Features:
    - Hidden cards in tableau (fewer = better)
    - Kings not in columns (blocking potential)
    - Aces location (accessible = better)
    - Suit distribution in tableau
    - Initial valid moves count
    """
    score = 0.0

    # Feature 1: Hidden card penalty
    total_hidden = sum(state.tableau_hidden)
    score -= total_hidden * 2

    # Feature 2: Accessible aces bonus
    accessible_aces = count_accessible_aces(state)
    score += accessible_aces * 10

    # Feature 3: Kings blocking
    buried_kings = count_buried_kings(state)
    score -= buried_kings * 5

    # Feature 4: Initial options
    initial_moves = len(get_valid_moves(state))
    score += initial_moves * 2

    return sigmoid(score)  # Normalize to 0-1
```

---

## Implementation Plan

### Phase 1 (Now): Fix Heuristic Cycling
- [ ] Add state history tracking
- [ ] Add cycle detection
- [ ] Test with games that were cycling

### Phase 2 (Next): Better Heuristics
- [ ] Revise weight system
- [ ] Add game phase awareness
- [ ] Add negative rewards for non-progress
- [ ] Benchmark improvement

### Phase 3 (Later): Meta-Strategy Planning
- [ ] Design winnability estimator
- [ ] Research optimal stopping theory
- [ ] Simulate multi-deal scenarios
- [ ] Determine optimal exploration strategy

### Phase 4 (Future): Implementation
- [ ] Implement quick evaluator
- [ ] Implement deal selection strategy
- [ ] Build meta-game simulator
- [ ] Benchmark full system

---

## Expected Performance

### Current Heuristic: ~0-5% win rate
*(Gets stuck in cycles)*

### With Cycle Detection: ~10-15% win rate
*(No more saddle points)*

### With Better Heuristics: ~15-25% win rate
*(Smarter strategy)*

### With Meta-Strategy: ???
*(This is the interesting research question!)*

The meta-strategy doesn't improve single-game win rate, but improves **overall EV** by:
- Only playing favorable deals
- Skipping unfavorable deals
- Optimizing exploration cost

---

## Questions for Further Research

1. **How many deals should we evaluate on average?**
   - Depends on variance in deal quality
   - Depends on cost vs. benefit ratio

2. **Can we learn a winnability function?**
   - Train on thousands of games
   - Predict P(win | initial_state)

3. **Is there a closed-form optimal stopping rule?**
   - Secretary problem variant?
   - Threshold-based?

4. **How does this interact with solver quality?**
   - Better solver → can win harder deals → changes threshold

---

## References

- Optimal Stopping Theory (Secretary Problem)
- Exploration-Exploitation Tradeoffs
- Portfolio Selection (Markowitz)
- Multi-Armed Bandits (if learning winnability)
