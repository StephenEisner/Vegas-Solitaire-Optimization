# Vegas Solitaire: Coding Standards & Best Practices

Guidelines for writing clean, maintainable, and efficient code for this project.

## Python Style Guide

### General Principles

1. **Follow PEP 8** with minor exceptions
2. **Clarity over cleverness** - readable code is better than compact code
3. **Type hints everywhere** - helps catch bugs and documents code
4. **Docstrings for all public functions** - explain what, not how

### Code Formatting

**Use Black** for automatic formatting:
```bash
black game/ optimization/
```

**Line length**: 100 characters (Black default is 88, we extend slightly)

**Import organization**:
```python
# Standard library
import random
from typing import List, Optional

# Third-party
import numpy as np
import torch

# Local
from game.core.card import Card, Suit, Rank
from game.core.state import GameState
```

### Type Hints

**Always use type hints**:
```python
# Good
def evaluate_move(move: Move, state: GameState) -> float:
    """Calculate heuristic score for a move."""
    ...

# Bad
def evaluate_move(move, state):
    ...
```

**Use Union, Optional, List, Dict**:
```python
from typing import Optional, List, Dict, Union

def get_valid_moves(state: GameState) -> List[Move]:
    ...

def find_card(cards: List[Card], rank: Rank) -> Optional[Card]:
    ...

def create_mapping() -> Dict[Suit, List[Card]]:
    ...
```

### Docstrings

**Use Google-style docstrings**:
```python
def can_move_to_tableau(card: Card, tableau_column: List[Card]) -> bool:
    """Check if card can be placed on tableau column.
    
    A card can be placed on a tableau column if:
    1. Column is empty and card is a King, OR
    2. Card is one rank lower and opposite color from column's top card
    
    Args:
        card: The card to place
        tableau_column: The tableau column (list of cards)
    
    Returns:
        True if move is valid, False otherwise
    
    Examples:
        >>> can_move_to_tableau(Card(Suit.SPADES, Rank.QUEEN), 
        ...                     [Card(Suit.HEARTS, Rank.KING)])
        True
        >>> can_move_to_tableau(Card(Suit.HEARTS, Rank.QUEEN),
        ...                     [Card(Suit.DIAMONDS, Rank.KING)])
        False  # Same color
    """
    if not tableau_column:
        return card.rank == Rank.KING
    
    top_card = tableau_column[-1]
    return (card.rank.value == top_card.rank.value - 1 and 
            card.color != top_card.color)
```

### Naming Conventions

**Classes**: PascalCase
```python
class GameState:
class MCTSSolver:
class PolicyNetwork:
```

**Functions/methods**: snake_case
```python
def get_valid_moves():
def apply_move():
def train_policy():
```

**Constants**: UPPER_SNAKE_CASE
```python
MAX_MOVES = 1000
DEFAULT_BEAM_WIDTH = 10
VEGAS_ENTRY_COST = -52
```

**Private methods/attributes**: prefix with `_`
```python
class Game:
    def _update_score(self):
        """Private method."""
        ...
    
    def _initialize_state(self):
        """Private method."""
        ...
```

## Code Organization

### File Structure

**One class per file** (generally):
```
card.py        -> Card, Suit, Rank, Deck
state.py       -> GameState
game.py        -> Game
moves.py       -> Move, MoveType
```

**Exception**: Related small classes can share a file
```python
# moves.py
class MoveType(Enum):
    ...

@dataclass(frozen=True)
class Move:
    ...
```

### Function Length

**Keep functions short** - aim for <50 lines, ideally <20

**Long function** (refactor this):
```python
def get_valid_moves(state: GameState) -> List[Move]:
    moves = []
    
    # 100 lines of move generation...
    if state.stock:
        moves.append(...)
    
    for col in range(7):
        # 30 lines
        ...
    
    for col in range(7):
        for other_col in range(7):
            # 40 lines
            ...
    
    return moves
```

**Better** (split into functions):
```python
def get_valid_moves(state: GameState) -> List[Move]:
    """Get all valid moves from current state."""
    moves = []
    moves.extend(_get_draw_moves(state))
    moves.extend(_get_waste_moves(state))
    moves.extend(_get_tableau_moves(state))
    return moves

def _get_draw_moves(state: GameState) -> List[Move]:
    """Get drawing and recycling moves."""
    ...

def _get_waste_moves(state: GameState) -> List[Move]:
    """Get moves from waste pile."""
    ...

def _get_tableau_moves(state: GameState) -> List[Move]:
    """Get moves between tableau columns."""
    ...
```

### Class Design

**Use dataclasses for simple data containers**:
```python
from dataclasses import dataclass

@dataclass(frozen=True)
class Card:
    suit: Suit
    rank: Rank
```

**Use regular classes for behavior-heavy objects**:
```python
class Game:
    def __init__(self, seed: Optional[int] = None):
        self.state = GameState(seed=seed)
        self.deck = Deck(seed=seed)
    
    def deal(self):
        ...
    
    def make_move(self, move: Move) -> bool:
        ...
```

**Prefer composition over inheritance**:
```python
# Good
class Game:
    def __init__(self):
        self.state = GameState()  # Composition
        self.rules = RulesEngine()

# Avoid (unless there's a good reason)
class Game(GameState):  # Inheritance
    ...
```

## Performance Guidelines

### Critical Paths

**Functions called millions of times**:
- `get_valid_moves()`
- `apply_move()`
- `GameState.copy()`
- `GameState.__hash__()`

These MUST be fast. Profile and optimize carefully.

### Optimization Techniques

**Use list comprehensions** (faster than loops):
```python
# Good
valid_moves = [m for m in all_moves if is_valid(m)]

# Slower
valid_moves = []
for m in all_moves:
    if is_valid(m):
        valid_moves.append(m)
```

**Cache expensive computations**:
```python
from functools import lru_cache

@lru_cache(maxsize=10000)
def compute_state_value(state_hash: int) -> float:
    """Expensive computation that should be cached."""
    ...
```

**Avoid unnecessary allocations**:
```python
# Bad - creates many temporary lists
def get_top_cards(tableau: List[List[Card]]) -> List[Card]:
    return [col[-1] if col else None for col in tableau]

# Better - generator
def get_top_cards(tableau: List[List[Card]]):
    for col in tableau:
        yield col[-1] if col else None
```

**Use NumPy for numerical operations**:
```python
# Good
import numpy as np
values = np.array([score(m) for m in moves])
best_idx = np.argmax(values)

# Slower for large arrays
best_idx = max(range(len(moves)), key=lambda i: score(moves[i]))
```

### When NOT to Optimize

**Don't optimize prematurely**:
- UI code (called infrequently)
- Testing utilities
- Logging and debugging code

**Profile first**:
```python
import cProfile

profiler = cProfile.Profile()
profiler.enable()
# ... code to profile ...
profiler.disable()
profiler.print_stats(sort='cumulative')
```

## Testing Best Practices

### Test Organization

**Mirror source structure**:
```
game/
  core/
    card.py
    state.py
  tests/
    test_card.py
    test_state.py
```

### Test Naming

**Descriptive test names**:
```python
# Good
def test_ace_can_move_to_empty_foundation():
    ...

def test_two_cannot_move_to_empty_foundation():
    ...

def test_king_can_move_to_empty_tableau():
    ...

# Bad
def test_foundation_1():
    ...

def test_tableau():
    ...
```

### Test Structure

**Use Arrange-Act-Assert**:
```python
def test_card_color():
    # Arrange
    red_card = Card(Suit.HEARTS, Rank.ACE)
    black_card = Card(Suit.SPADES, Rank.ACE)
    
    # Act
    red_color = red_card.color
    black_color = black_card.color
    
    # Assert
    assert red_color == 'red'
    assert black_color == 'black'
```

### Fixtures

**Use pytest fixtures for common setup**:
```python
import pytest

@pytest.fixture
def empty_game():
    """Game with no cards dealt."""
    return Game(seed=42)

@pytest.fixture
def dealt_game():
    """Game ready to play."""
    game = Game(seed=42)
    game.deal()
    return game

def test_initial_score(dealt_game):
    assert dealt_game.state.score == -52

def test_has_stock(dealt_game):
    assert len(dealt_game.state.stock) == 24
```

### Test Coverage

**Aim for >90% coverage** on core logic:
```bash
pytest --cov=game --cov=optimization --cov-report=html
```

**Test edge cases**:
```python
def test_empty_tableau_column():
    """Test behavior with empty column."""
    ...

def test_single_card_in_column():
    """Test moving single card."""
    ...

def test_full_sequence_move():
    """Test moving entire sequence."""
    ...

def test_all_cards_face_down():
    """Test column with all cards hidden."""
    ...
```

### Parametrized Tests

**Test multiple cases efficiently**:
```python
import pytest

@pytest.mark.parametrize("card,expected_color", [
    (Card(Suit.HEARTS, Rank.ACE), 'red'),
    (Card(Suit.DIAMONDS, Rank.KING), 'red'),
    (Card(Suit.SPADES, Rank.QUEEN), 'black'),
    (Card(Suit.CLUBS, Rank.JACK), 'black'),
])
def test_card_colors(card, expected_color):
    assert card.color == expected_color
```

## Error Handling

### Validation

**Validate inputs early**:
```python
def apply_move(state: GameState, move: Move) -> GameState:
    """Apply move to state, returning new state.
    
    Args:
        state: Current game state
        move: Move to apply
    
    Returns:
        New game state after move
    
    Raises:
        ValueError: If move is invalid
    """
    if not is_valid_move(state, move):
        raise ValueError(f"Invalid move: {move}")
    
    # ... apply move ...
```

### Exceptions vs Return Values

**Use exceptions for exceptional conditions**:
```python
# Good - invalid input is exceptional
def get_card_at(column: List[Card], index: int) -> Card:
    if index < 0 or index >= len(column):
        raise IndexError(f"Invalid index {index} for column of length {len(column)}")
    return column[index]

# Good - no valid moves is expected
def choose_move(state: GameState) -> Optional[Move]:
    moves = get_valid_moves(state)
    return random.choice(moves) if moves else None
```

### Custom Exceptions

**Create custom exceptions for domain errors**:
```python
class InvalidMoveError(Exception):
    """Raised when attempting an invalid move."""
    pass

class GameNotInitializedError(Exception):
    """Raised when operating on uninitialized game."""
    pass

# Usage
def make_move(self, move: Move) -> bool:
    if not self._initialized:
        raise GameNotInitializedError("Must call deal() before making moves")
    
    if not self.is_valid_move(move):
        raise InvalidMoveError(f"Move {move} is not valid in current state")
    
    # ... apply move ...
```

## Comments and Documentation

### When to Comment

**Comment the WHY, not the WHAT**:
```python
# Bad - obvious
# Increment counter
counter += 1

# Good - explains reasoning
# Use beam width of 10 as empirically this gives best speed/quality tradeoff
beam_width = 10

# Good - explains non-obvious algorithm
# We use UCB1 with exploration parameter sqrt(2) per Kocsis & Szepesvári 2006
ucb_value = exploit + math.sqrt(2) * explore
```

### TODO Comments

**Use TODO for future work**:
```python
# TODO(username): Add support for custom scoring systems
# TODO(username): Optimize this function - profiling shows it's a bottleneck
# TODO(username): Consider using sets instead of lists for O(1) lookup
```

### Function Documentation

**Document complex functions thoroughly**:
```python
def expectimax(state: GameState, depth: int, eval_fn: Callable) -> float:
    """Expectimax search algorithm for stochastic games.
    
    Computes expected value of state by alternating between:
    - Decision nodes: maximize over available actions
    - Chance nodes: compute weighted average over random outcomes
    
    Args:
        state: Current game state to evaluate
        depth: Maximum search depth (0 = evaluate immediately)
        eval_fn: Heuristic evaluation function for leaf nodes
    
    Returns:
        Expected value of the state
    
    Notes:
        Time complexity: O(b^d * c^d) where b=branching factor,
        d=depth, c=chance outcomes. This can be very expensive!
        
        Space complexity: O(d) due to recursion depth.
    
    References:
        Expectimax is described in Russell & Norvig, "Artificial Intelligence:
        A Modern Approach", Section 5.5.
    """
    ...
```

## Git Workflow

### Commit Messages

**Follow conventional commits**:
```
feat: add MCTS solver implementation
fix: correct move validation for tableau sequences  
docs: update README with installation instructions
test: add tests for edge cases in move generation
refactor: extract move scoring into separate function
perf: optimize state hashing for better cache hits
```

**Detailed body when needed**:
```
feat: implement expectimax solver

- Add expectimax algorithm with depth limiting
- Include alpha-beta pruning for decision nodes
- Add low-probability outcome pruning for efficiency
- Benchmark shows 28% win rate at depth 4

Closes #42
```

### Branch Naming

```
main              # Production code
develop           # Integration branch

feature/mcts      # New feature
fix/move-bug      # Bug fix
refactor/state    # Refactoring
docs/architecture # Documentation
```

### Pull Request Guidelines

1. **Small, focused PRs** - easier to review
2. **Pass all tests** before requesting review
3. **Update documentation** if needed
4. **Add tests** for new features
5. **Link related issues**

## Configuration Management

### Use Config Files

**Don't hardcode hyperparameters**:
```python
# Bad
class MCTSSolver:
    def __init__(self):
        self.num_simulations = 1000  # Hardcoded
        self.exploration = 1.41      # Hardcoded

# Good
from dataclasses import dataclass

@dataclass
class MCTSConfig:
    num_simulations: int = 1000
    exploration: float = 1.41
    use_transposition_table: bool = True
    max_depth: Optional[int] = None

class MCTSSolver:
    def __init__(self, config: MCTSConfig):
        self.config = config
```

**Use YAML or JSON for configs**:
```yaml
# configs/mcts_fast.yaml
num_simulations: 500
exploration: 1.41
use_transposition_table: true

# configs/mcts_strong.yaml
num_simulations: 5000
exploration: 1.0
use_transposition_table: true
```

```python
import yaml

with open('configs/mcts_fast.yaml') as f:
    config_dict = yaml.safe_load(f)
    config = MCTSConfig(**config_dict)

solver = MCTSSolver(config)
```

## Logging

### Use Python's logging module

**Set up logging properly**:
```python
import logging

# In main script
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('vegas_solitaire.log'),
        logging.StreamHandler()
    ]
)

# In modules
logger = logging.getLogger(__name__)

logger.info("Starting MCTS search with %d simulations", num_simulations)
logger.debug("State hash: %s", hash(state))
logger.warning("Slow convergence detected")
logger.error("Failed to load checkpoint: %s", error)
```

### Logging Levels

- **DEBUG**: Detailed information for debugging
- **INFO**: General information about program execution
- **WARNING**: Something unexpected but not critical
- **ERROR**: An error that prevented operation
- **CRITICAL**: Serious error, program may terminate

## Code Review Checklist

**Before submitting code**:

□ All tests pass
□ Code is formatted (Black)
□ Type hints are present
□ Docstrings are complete
□ No commented-out code
□ No debugging print statements
□ No TODOs without issue numbers
□ Performance-critical code is profiled
□ New features have tests
□ Documentation is updated

**During review, check**:

□ Code is readable and maintainable
□ Naming is clear and consistent
□ No unnecessary complexity
□ Edge cases are handled
□ Error messages are helpful
□ Commits are logical and well-described
□ No security issues (RNG seeds, etc.)

## Common Pitfalls to Avoid

### Mutating Shared State

**Bad**:
```python
def apply_move(state: GameState, move: Move):
    # Mutates the input state!
    state.waste.append(card)
    return state
```

**Good**:
```python
def apply_move(state: GameState, move: Move) -> GameState:
    # Creates new state
    new_state = state.copy()
    new_state.waste.append(card)
    return new_state
```

### Comparing Floating Point Numbers

**Bad**:
```python
if score == 0.0:  # Floating point comparison
    ...
```

**Good**:
```python
import math

if math.isclose(score, 0.0, abs_tol=1e-9):
    ...
```

### Ignoring Exceptions

**Bad**:
```python
try:
    result = dangerous_operation()
except Exception:
    pass  # Silent failure!
```

**Good**:
```python
try:
    result = dangerous_operation()
except SpecificException as e:
    logger.error("Operation failed: %s", e)
    # Handle or re-raise
    raise
```

### Using Mutable Default Arguments

**Bad**:
```python
def add_move(moves=[]):  # Mutable default!
    moves.append(Move(...))
    return moves
```

**Good**:
```python
def add_move(moves: Optional[List[Move]] = None) -> List[Move]:
    if moves is None:
        moves = []
    moves.append(Move(...))
    return moves
```

## Resources

### Python Style
- [PEP 8](https://pep8.org/)
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
- [Black Code Formatter](https://black.readthedocs.io/)

### Type Hints
- [mypy](http://mypy-lang.org/)
- [Python Type Checking Guide](https://realpython.com/python-type-checking/)

### Testing
- [pytest documentation](https://docs.pytest.org/)
- [pytest best practices](https://docs.pytest.org/en/stable/goodpractices.html)

### Performance
- [Python Performance Tips](https://wiki.python.org/moin/PythonSpeed/PerformanceTips)
- [cProfile](https://docs.python.org/3/library/profile.html)

---

## Summary

Good code is:
1. **Correct** - does what it should
2. **Clear** - easy to understand
3. **Maintainable** - easy to modify
4. **Tested** - proven to work
5. **Documented** - explains itself

Follow these guidelines and your code will be a joy to work with! 🎉
