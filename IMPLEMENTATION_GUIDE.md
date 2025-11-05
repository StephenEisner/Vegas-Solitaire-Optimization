# Vegas Solitaire: Implementation Guide

A detailed, step-by-step guide to implementing the Vegas Solitaire project from scratch.

## Table of Contents

1. [Phase 1: Game Core (Week 1)](#phase-1-game-core)
2. [Phase 2: Basic Solvers (Week 2)](#phase-2-basic-solvers)
3. [Phase 3: Advanced Search (Week 3-4)](#phase-3-advanced-search)
4. [Phase 4: Machine Learning (Week 5-6)](#phase-4-machine-learning)
5. [Phase 5: Evaluation & Analysis (Week 7)](#phase-5-evaluation--analysis)
6. [Phase 6: Visualization (Week 8+)](#phase-6-visualization)

---

## Phase 1: Game Core

**Goal**: Implement a fully functional, well-tested Vegas Solitaire game engine.

### Day 1: Card and Deck

**File**: `game/core/card.py`

**What to build**:
1. Card class with suit, rank, color
2. Deck class with shuffling
3. Basic card operations

**Implementation checklist**:
```python
□ Define Suit enum (♠♥♦♣)
□ Define Rank enum (A-K, values 1-13)
□ Create Card dataclass (frozen=True)
□ Implement Card.color property
□ Implement Card.__str__() for display
□ Create Deck class
□ Implement Deck.shuffle() with seeded RNG
□ Implement Deck.draw(n)
□ Add __repr__ methods for debugging
```

**Testing**:
```python
# test_card.py
def test_card_color():
    assert Card(Suit.HEARTS, Rank.ACE).color == 'red'
    assert Card(Suit.SPADES, Rank.ACE).color == 'black'

def test_deck_has_52_cards():
    deck = Deck()
    assert len(deck.cards) == 52

def test_deck_shuffle_is_deterministic():
    deck1 = Deck(seed=42)
    deck1.shuffle()
    deck2 = Deck(seed=42)
    deck2.shuffle()
    assert deck1.cards == deck2.cards
```

**Time estimate**: 2-3 hours

---

### Day 2: Game State

**File**: `game/core/state.py`

**What to build**:
1. GameState class representing complete game state
2. State copying and hashing
3. Vector representation for ML

**Implementation checklist**:
```python
□ Create GameState dataclass with:
  □ stock: List[Card]
  □ waste: List[Card]
  □ tableau: List[List[Card]]
  □ tableau_hidden: List[int]
  □ foundations: Dict[Suit, List[Card]]
  □ score: int
  □ move_count: int
□ Implement GameState.copy() (deep copy)
□ Implement GameState.__hash__() for caching
□ Implement GameState.is_winning()
□ Implement GameState.get_foundation_count()
□ Stub out GameState.to_vector() (implement later)
□ Add __str__ for pretty printing
```

**Key decisions**:
- **Immutability**: States should be copyable without side effects
- **Hashing**: Only hash visible state (not RNG seed) for efficiency
- **Foundation tracking**: Use dict keyed by Suit

**Testing**:
```python
def test_state_copy_is_independent():
    state1 = GameState()
    state1.waste.append(Card(Suit.HEARTS, Rank.ACE))
    state2 = state1.copy()
    state2.waste.append(Card(Suit.SPADES, Rank.TWO))
    assert len(state1.waste) == 1
    assert len(state2.waste) == 2

def test_winning_state():
    state = GameState()
    for suit in Suit:
        state.foundations[suit] = [Card(suit, rank) for rank in Rank]
    assert state.is_winning()
```

**Time estimate**: 3-4 hours

---

### Day 3: Move Representation

**File**: `game/core/moves.py`

**What to build**:
1. MoveType enum
2. Move dataclass
3. String representation

**Implementation checklist**:
```python
□ Define MoveType enum:
  □ TABLEAU_TO_FOUNDATION
  □ TABLEAU_TO_TABLEAU
  □ WASTE_TO_FOUNDATION
  □ WASTE_TO_TABLEAU
  □ DRAW
  □ RECYCLE
□ Create Move dataclass with:
  □ move_type: MoveType
  □ source: Optional[int]
  □ destination: Optional[int]
  □ card_count: int
  □ card: Optional[Card]
□ Make Move immutable (frozen=True)
□ Implement Move.__str__()
□ Implement Move.__hash__()
```

**Design note**: Move should contain all information needed to:
1. Apply the move
2. Display the move to user
3. Validate the move

**Time estimate**: 1-2 hours

---

### Day 4-5: Rules Engine

**File**: `game/core/rules.py`

**What to build**:
1. Move validation functions
2. Move generation
3. Move application

**Implementation checklist**:
```python
□ Implement can_move_to_foundation(card, state)
□ Implement can_move_to_tableau(card, tableau_col)
□ Implement is_valid_sequence(cards)
□ Implement get_valid_moves(state) -> List[Move]
  □ Generate draw/recycle moves
  □ Generate waste-to-foundation moves
  □ Generate waste-to-tableau moves
  □ Generate tableau-to-foundation moves
  □ Generate tableau-to-tableau moves (sequences!)
□ Implement apply_move(state, move) -> GameState
  □ Handle each MoveType
  □ Update tableau_hidden when revealing cards
  □ Return NEW state (functional approach)
□ Add helper: get_visible_cards(tableau_col, hidden_count)
```

**Critical function**: `get_valid_moves()`
This will be called millions of times during search, so it must be efficient:
- No unnecessary allocations
- Early termination when possible
- Cache results if beneficial

**Testing**:
```python
def test_ace_to_empty_foundation():
    state = GameState()
    ace = Card(Suit.HEARTS, Rank.ACE)
    assert can_move_to_foundation(ace, state)

def test_two_needs_ace_first():
    state = GameState()
    two = Card(Suit.HEARTS, Rank.TWO)
    assert not can_move_to_foundation(two, state)

def test_kings_only_to_empty_tableau():
    king = Card(Suit.HEARTS, Rank.KING)
    queen = Card(Suit.HEARTS, Rank.QUEEN)
    assert can_move_to_tableau(king, [])
    assert not can_move_to_tableau(queen, [])

def test_alternating_colors():
    red_king = Card(Suit.HEARTS, Rank.KING)
    black_queen = Card(Suit.SPADES, Rank.QUEEN)
    red_queen = Card(Suit.DIAMONDS, Rank.QUEEN)
    
    assert can_move_to_tableau(black_queen, [red_king])
    assert not can_move_to_tableau(red_queen, [red_king])

def test_reveal_card_when_tableau_emptied():
    state = GameState()
    state.tableau[0] = [
        Card(Suit.HEARTS, Rank.KING),
        Card(Suit.SPADES, Rank.QUEEN)
    ]
    state.tableau_hidden[0] = 1  # King is face down
    
    move = Move(
        MoveType.TABLEAU_TO_FOUNDATION,
        source=0,
        card=Card(Suit.SPADES, Rank.QUEEN)
    )
    
    new_state = apply_move(state, move)
    assert new_state.tableau_hidden[0] == 0  # King should be revealed
```

**Time estimate**: 6-8 hours (this is complex!)

---

### Day 6: Game Controller

**File**: `game/core/game.py`

**What to build**:
1. Game class that orchestrates everything
2. Deal functionality
3. Move making and validation

**Implementation checklist**:
```python
□ Create Game class with:
  □ state: GameState
  □ deck: Deck
  □ move_history: List[Move]
□ Implement Game.deal()
  □ Shuffle deck
  □ Deal to tableau (1, 2, 3... 7 cards)
  □ Set tableau_hidden correctly
  □ Remaining cards to stock
□ Implement Game.get_valid_moves()
□ Implement Game.make_move(move)
  □ Validate move
  □ Apply move
  □ Update score (+$5 per foundation card)
  □ Record in history
□ Implement Game.is_over()
□ Implement Game.get_state_copy()
□ Add Game.undo() (optional, useful for debugging)
```

**Testing**:
```python
def test_deal_creates_proper_tableau():
    game = Game(seed=42)
    game.deal()
    
    for i in range(7):
        assert len(game.state.tableau[i]) == i + 1
        assert game.state.tableau_hidden[i] == i

def test_cannot_make_invalid_move():
    game = Game(seed=42)
    game.deal()
    
    # Create obviously invalid move
    bad_move = Move(
        MoveType.WASTE_TO_FOUNDATION,
        card=Card(Suit.HEARTS, Rank.KING)
    )
    
    result = game.make_move(bad_move)
    assert not result

def test_game_tracks_score():
    game = Game(seed=42)
    game.deal()
    
    initial_score = game.state.score
    assert initial_score == -52  # Vegas entry cost
    
    # Make a foundation move (if possible)
    moves = game.get_valid_moves()
    foundation_moves = [m for m in moves 
                       if 'FOUNDATION' in m.move_type.name]
    
    if foundation_moves:
        game.make_move(foundation_moves[0])
        assert game.state.score == initial_score + 5
```

**Time estimate**: 4-5 hours

---

### Day 7: CLI and Testing

**File**: `game/ui/cli.py`

**What to build**:
1. Simple command-line interface for playing
2. Text rendering of game state
3. Interactive move selection

**Implementation checklist**:
```python
□ Create simple ASCII art renderer
□ Display tableau, foundations, waste, stock
□ Number moves for user selection
□ Handle user input
□ Display game statistics
□ Allow replay of games by seed
```

**Example output**:
```
Stock: [24]  Waste: 3♥ 7♦ K♠

Foundations:  ♠[A]  ♥[--]  ♦[--]  ♣[--]

Tableau:
  1     2     3     4     5     6     7
 [▓]   [▓]   [▓]   [▓]   [▓]   [▓]   [▓]
       [▓]   [▓]   [▓]   [▓]   [▓]   [▓]
              8♦   [▓]   [▓]   [▓]   [▓]
                    4♣   [▓]   [▓]   [▓]
                          9♠   [▓]   [▓]
                                J♥   [▓]
                                      7♣

Valid moves:
1. Draw from stock
2. Move K♠ from waste to column 0
3. Move 8♦ from column 2 to column 4
...

Choose move: _
```

**Testing**:
Play through several complete games manually to ensure everything works.

**Time estimate**: 3-4 hours

---

### End of Phase 1 Checklist

□ All card/deck operations work correctly
□ Game state can be created, copied, hashed
□ All move types can be generated and applied
□ Rules are validated by comprehensive tests
□ Full games can be played via CLI
□ Code is well-documented
□ ~95% test coverage for core game logic

**Deliverable**: Fully functional Vegas Solitaire game engine

---

## Phase 2: Basic Solvers

**Goal**: Implement baseline solvers (random, heuristic) and evaluation framework.

### Day 8: Solver Base Class

**File**: `optimization/solvers/base.py`

**What to build**:
```python
□ Create abstract Solver class
□ Define choose_move() interface
□ Implement play_game() method
□ Add statistics tracking
□ Implement get_statistics()
```

**Time estimate**: 2 hours

---

### Day 9: Random Solver

**File**: `optimization/solvers/random_solver.py`

**What to build**:
```python
□ Extend Solver base class
□ Implement choose_move() - random selection
□ Use seeded RNG for reproducibility
□ Test on various games
```

**Validation**:
- Run 1000 games
- Expected win rate: ~2-8%
- Expected avg score: -$20 to -$30

**Time estimate**: 2 hours

---

### Day 10-11: Heuristic Solver

**File**: `optimization/solvers/heuristic_solver.py`

**What to build**:
```python
□ Define heuristic evaluation function
  □ Foundation moves: +10
  □ Reveal card: +8
  □ Empty column: +15
  □ Sequence length: +2 per card
  □ Block important card: -20
  □ Waste to tableau: +3
□ Implement score_move(move, state)
□ Implement choose_move() - argmax heuristic
□ Test and tune weights
```

**Heuristic ideas**:
1. **Always prefer foundation moves** (usually safe)
2. **Prioritize revealing face-down cards**
3. **Keep empty columns** (very valuable)
4. **Avoid blocking needed cards** (look at foundations)
5. **Move from waste when possible** (cycles through stock)

**Time estimate**: 4-6 hours (tuning takes time)

---

### Day 12: Benchmark Framework

**File**: `optimization/evaluation/benchmark.py`

**What to build**:
```python
□ Create Benchmark class
□ Implement evaluate_solver(solver, num_games)
□ Generate fixed test set (e.g., seeds 1-1000)
□ Collect statistics:
  □ Win rate
  □ Average score
  □ Median score
  □ Score distribution
  □ Average moves per game
□ Implement BenchmarkResults class
□ Add to_dataframe() for analysis
□ Add to_json() for saving
```

**Time estimate**: 3-4 hours

---

### Day 13: Analysis Tools

**File**: `optimization/evaluation/analyze.py`

**What to build**:
```python
□ Functions to analyze game traces
□ Identify critical decision points
□ Compute branching factor statistics
□ Analyze which moves matter most
□ Compare games (won vs lost)
```

**Time estimate**: 3-4 hours

---

### Day 14: Baseline Comparison

**Goal**: Get baseline numbers for all future comparison.

**Tasks**:
1. Run Random solver on test set (1000 games)
2. Run Heuristic solver on same test set
3. Generate comparison report
4. Save results for future reference

**Create reference document**:
```markdown
# Baseline Results (Date: YYYY-MM-DD)

## Random Solver
- Games: 1000
- Win rate: 5.2%
- Avg score: -$25.30
- Median score: -$27.00
- Time per game: 0.05s

## Heuristic Solver
- Games: 1000
- Win rate: 18.4%
- Avg score: -$8.20
- Median score: -$12.00
- Time per game: 0.12s

## Analysis
Heuristic solver achieves 3.5x better win rate...
```

**Time estimate**: 2-3 hours

---

### End of Phase 2 Checklist

□ Random solver implemented and tested
□ Heuristic solver implemented and tuned
□ Benchmark framework working
□ Baseline results documented
□ Can compare solver performance easily

**Deliverable**: Working solvers with evaluation framework

---

## Phase 3: Advanced Search

**Goal**: Implement MCTS, Expectimax, and Beam Search.

### Day 15-17: MCTS Implementation

**File**: `optimization/solvers/mcts_solver.py`

**Day 15: Core MCTS**
```python
□ Create MCTSNode class
  □ State, parent, children
  □ visits, total_reward
  □ untried_moves
□ Implement UCB1 selection
□ Implement expand()
□ Implement simulate() (random rollout)
□ Implement backpropagate()
```

**Day 16: MCTS Solver**
```python
□ Create MCTSSolver class
□ Implement main MCTS loop
□ Add time budget support
□ Add simulation budget support
□ Test on simple positions
```

**Day 17: MCTS Optimizations**
```python
□ Add transposition table
□ Implement heuristic rollouts
□ Add progressive widening
□ Tune exploration parameter
□ Benchmark vs baselines
```

**Expected performance**:
- Win rate: ~25-35% (1000 simulations)
- Time per move: 0.5-2 seconds

**Time estimate**: 12-15 hours total

---

### Day 18-19: Expectimax

**File**: `optimization/solvers/expectimax_solver.py`

**Day 18: Core Algorithm**
```python
□ Implement expectimax(state, depth, eval_fn)
□ Handle decision nodes (max)
□ Handle chance nodes (expected value)
□ Implement depth limiting
□ Create default evaluation function
```

**Day 19: Optimizations**
```python
□ Add alpha-beta pruning for decision nodes
□ Implement low-probability pruning
□ Add transposition table
□ Sampling-based chance nodes
□ Tune depth limit
```

**Challenges**:
- Chance nodes are expensive (many possible draws)
- Need good evaluation function
- Depth limit is critical

**Time estimate**: 8-10 hours

---

### Day 20-21: Beam Search

**File**: `optimization/solvers/beam_search_solver.py`

**Day 20: Basic Beam Search**
```python
□ Implement BeamSearch class
□ Implement search(initial_state, beam_width, depth)
□ Beam expansion and pruning
□ Evaluation function
```

**Day 21: Variations**
```python
□ Implement diverse beam search
□ Implement stochastic beam search
□ Implement adaptive beam width
□ Tune hyperparameters
```

**Expected performance**:
- Win rate: ~15-25% (beam=10, depth=5)
- Time per move: 0.05-0.3 seconds

**Time estimate**: 8-10 hours

---

### Day 22: Minimax (Educational)

**File**: `optimization/solvers/minimax_solver.py`

**What to build**:
```python
□ Implement minimax() - basic version
□ Implement alpha_beta() - with pruning
□ Add evaluation function
□ Run experiments showing it performs poorly
□ Document why it's wrong for solitaire
```

**Goal**: Demonstrate importance of algorithm selection

**Time estimate**: 3-4 hours

---

### Day 23-24: Search Algorithm Comparison

**Tasks**:
1. Run all search algorithms on test set
2. Compare with different hyperparameters
3. Generate comprehensive comparison
4. Document findings

**Create comparison table**:
```
| Algorithm | Win Rate | Avg Score | Time/Move | Notes |
|-----------|----------|-----------|-----------|-------|
| Random | 5.2% | -$25 | 0.05s | Baseline |
| Heuristic | 18.4% | -$8 | 0.12s | Fast & decent |
| Beam(10,5) | 22.1% | -$2 | 0.18s | Good speed/quality |
| Expectimax(4) | 26.8% | +$3 | 1.2s | Slow but strong |
| MCTS(1000) | 31.5% | +$8 | 1.8s | Best search method |
```

**Time estimate**: 6-8 hours

---

### End of Phase 3 Checklist

□ MCTS implemented and optimized
□ Expectimax working with pruning
□ Beam search with variations
□ Minimax for comparison
□ Comprehensive search algorithm comparison
□ Documentation of findings

**Deliverable**: Suite of search algorithms with analysis

---

## Phase 4: Machine Learning

**Goal**: Implement RL training pipeline and neural networks.

### Day 25-26: State Representation for ML

**File**: `game/core/state.py` (expand), `optimization/models/features.py`

**What to build**:
```python
□ Implement GameState.to_vector()
  □ Encode each card position (52 positions)
  □ One-hot encode suits and ranks
  □ Encode visibility (face up/down)
  □ Encode foundations
  □ Additional features (empty cols, etc.)
□ Test encoding/decoding
□ Verify shape is consistent
```

**Design decisions**:
- **Fixed-size tensor**: Pad to maximum possible size
- **Channel-based**: Separate channels for different info
- **Normalized**: Scale values to [0, 1] range

**Example encoding**:
```
State vector shape: (416,)
- 52 * 4 = 208 (card suit/rank encoding)
- 52 * 1 = 52 (visibility mask)
- 52 * 1 = 52 (position encoding)
- 7 * 1 = 7 (tableau hidden counts)
- ...
```

**Time estimate**: 6-8 hours

---

### Day 27-28: Neural Networks

**File**: `optimization/models/networks.py`

**What to build**:
```python
□ Implement PolicyNetwork
  □ Input: state vector
  □ Output: action logits
  □ Architecture: MLP with 2-3 hidden layers
□ Implement ValueNetwork
  □ Input: state vector
  □ Output: scalar value estimate
  □ Similar architecture to policy
□ Implement PolicyValueNetwork (combined)
□ Add masking for invalid actions
□ Test forward passes
```

**Architecture ideas**:
```python
PolicyNetwork(
  Linear(416 -> 256),
  ReLU,
  LayerNorm,
  Linear(256 -> 256),
  ReLU,
  LayerNorm,
  Linear(256 -> 200),  # Max actions
)
```

**Time estimate**: 6-8 hours

---

### Day 29-30: Training Infrastructure

**File**: `optimization/training/train_policy.py`

**What to build**:
```python
□ Implement ReplayBuffer
  □ Add episodes
  □ Sample batches
  □ Priority sampling (optional)
□ Implement PolicyTrainer
  □ Self-play data generation
  □ Policy gradient loss
  □ Optimization loop
  □ Checkpointing
□ Implement training loop
□ Add TensorBoard logging
```

**Time estimate**: 8-10 hours

---

### Day 31-32: RL Algorithm

**File**: `optimization/solvers/rl_solver.py`

**What to build**:
```python
□ Implement RLSolver
  □ Load trained policy network
  □ Implement choose_move()
  □ Handle action masking
  □ Support temperature-based sampling
□ Implement REINFORCE training
□ Or implement Actor-Critic
□ Or implement PPO (more advanced)
```

**Time estimate**: 8-10 hours

---

### Day 33-35: Training Run

**Goal**: Train a policy network end-to-end.

**Day 33**: Small-scale experiment
- Train on 100 epochs, 50 episodes each
- Verify learning is happening
- Check loss curves

**Day 34**: Medium-scale training
- Train on 1000 epochs
- Monitor performance on validation set
- Tune hyperparameters

**Day 35**: Final training run
- Train with best hyperparameters
- Use cloud GPU if available
- Save best checkpoint

**Expected results**:
- After training: 30-45% win rate
- Inference time: <10ms per move

**Time estimate**: 15-20 hours (mostly waiting for training)

---

### End of Phase 4 Checklist

□ State representation for ML working
□ Neural networks implemented
□ Training pipeline functional
□ At least one trained policy
□ RL solver can load and use trained model
□ Training documented with curves

**Deliverable**: Trained RL agent

---

## Phase 5: Evaluation & Analysis

**Goal**: Comprehensive comparison and analysis of all methods.

### Day 36-37: Tournament Framework

**File**: `optimization/evaluation/tournament.py`

**What to build**:
```python
□ Run all solvers on same test set
□ Generate comparison tables
□ Statistical significance tests
□ Performance vs time trade-offs
□ Create visualizations
```

**Time estimate**: 6-8 hours

---

### Day 38: Strategy Analysis

**File**: `optimization/evaluation/analyze.py`

**What to build**:
```python
□ Analyze what each algorithm learned
□ Identify common patterns in winning games
□ Analyze where algorithms make mistakes
□ Compare decision-making styles
□ Generate insights report
```

**Questions to answer**:
- Which moves matter most?
- Do algorithms agree on critical decisions?
- What separates winning from losing games?
- How does randomness affect outcomes?

**Time estimate**: 6-8 hours

---

### Day 39-40: Documentation

**Tasks**:
1. Write comprehensive results document
2. Create visualizations
3. Document interesting findings
4. Write algorithm recommendations

**Create**:
- `docs/results.md` - Full results
- `docs/insights.md` - Key insights
- `docs/recommendations.md` - When to use each algorithm

**Time estimate**: 8-10 hours

---

### End of Phase 5 Checklist

□ All algorithms benchmarked consistently
□ Statistical comparisons complete
□ Strategy analysis done
□ Results fully documented
□ Visualizations created

**Deliverable**: Complete analysis of all approaches

---

## Phase 6: Visualization

**Goal**: Create interactive website or Jupyter notebook.

### Week 8+: Choose Path

**Option A: Jupyter Notebook**
- Faster to create
- Good for technical audience
- Interactive plots with Plotly
- Embedded game demonstrations
- **Estimated time**: 20-30 hours

**Option B: React Website**
- More polished
- Better user experience
- Mobile-friendly
- Interactive game board
- **Estimated time**: 40-60 hours

### Jupyter Notebook Implementation

**Notebook sections**:
1. Introduction & Rules
2. Interactive game demo (ipywidgets)
3. Heuristic explanation with examples
4. Search algorithm visualization
5. RL training curves
6. Final comparison
7. Conclusions

**Time estimate**: 20-30 hours

---

### React Website Implementation

**Week 8**: Setup and game board
- Set up React project
- Create card components
- Implement game board rendering
- Add animation

**Week 9**: Interactivity
- Interactive move selection
- Solver integration
- Real-time policy visualization
- Compare player vs AI

**Week 10**: Content
- Write all sections
- Add explanations
- Create visualizations
- Polish UI

**Week 11**: Deployment
- Optimize performance
- Deploy to web
- Test on various devices
- Documentation

**Time estimate**: 40-60 hours

---

## End of Phase 6 Checklist

□ Website or notebook completed
□ All sections with content
□ Interactive elements working
□ Visualizations included
□ Deployed and accessible

**Deliverable**: Interactive educational experience

---

## Project Management Tips

### Version Control Strategy

```bash
# Main branches
main          # Production-ready code
develop       # Integration branch

# Feature branches
feature/game-core
feature/mcts
feature/rl-training
feature/website
```

### Commit Guidelines

- Commit frequently (every 1-2 hours)
- Write descriptive messages
- Include tests with features
- Keep commits atomic

### Testing Strategy

- Write tests BEFORE implementation (TDD)
- Aim for >90% coverage on core code
- Run tests before each commit
- Use pytest with coverage reporting

### Documentation

- Document as you go
- Add docstrings to all functions
- Keep README up to date
- Create examples

### Time Management

- Track time spent on each component
- Take breaks every 2 hours
- Review progress weekly
- Adjust estimates as needed

---

## Milestones

| Date | Milestone | Status |
|------|-----------|--------|
| End Week 1 | Game core complete | □ |
| End Week 2 | Baselines working | □ |
| End Week 4 | Search algorithms done | □ |
| End Week 6 | RL agent trained | □ |
| End Week 7 | Analysis complete | □ |
| End Week 11 | Website deployed | □ |

---

## Troubleshooting

### Common Issues

**Game hangs during random play**:
- Add move limit (e.g., 1000 moves max)
- Check for infinite loops in move generation
- Verify game-over detection

**Solver performs worse than expected**:
- Check move validation is correct
- Verify evaluation function makes sense
- Test on known positions
- Profile for performance bottlenecks

**Training doesn't converge**:
- Reduce learning rate
- Check gradient flow
- Verify loss function
- Increase batch size
- Simplify network architecture

**Tests are slow**:
- Use smaller test sets for quick iteration
- Profile tests to find bottlenecks
- Parallelize where possible
- Mock expensive operations

---

## Summary

This guide provides a complete roadmap from empty directory to deployed project. The key is to:

1. **Build incrementally** - Each phase builds on previous
2. **Test thoroughly** - Catch bugs early
3. **Document everything** - Future you will thank present you
4. **Stay flexible** - Adjust timeline based on progress

Estimated total time: **250-350 hours** over 2-3 months.

Good luck and enjoy the journey! 🎮🤖
