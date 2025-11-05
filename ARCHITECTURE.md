# Vegas Solitaire: Software Architecture

This document details the software architecture, code organization, and design decisions for the Vegas Solitaire project.

## Table of Contents

1. [Overall Architecture](#overall-architecture)
2. [Game Module Design](#game-module-design)
3. [Optimization Module Design](#optimization-module-design)
4. [Data Flow & Interfaces](#data-flow--interfaces)
5. [Testing Strategy](#testing-strategy)
6. [Performance Considerations](#performance-considerations)

---

## Overall Architecture

### High-Level Design

```
┌─────────────────────────────────────────────────────────────┐
│                        User Interface                        │
│  (CLI, GUI, Web Frontend, Jupyter Notebooks)                │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│                    Controller Layer                          │
│  (Game Manager, Tournament Runner, Experiment Coordinator)   │
└────────┬────────────────────────────────┬───────────────────┘
         │                                │
         ▼                                ▼
┌────────────────────┐          ┌────────────────────────────┐
│   Game Engine      │          │   Optimization Engine      │
│                    │◄────────►│                            │
│  - State mgmt      │          │  - Solvers (MCTS, etc)     │
│  - Rules engine    │          │  - RL Agents               │
│  - Move validation │          │  - Training pipeline       │
│  - Scoring         │          │  - Evaluation tools        │
└────────────────────┘          └────────────────────────────┘
         │                                │
         ▼                                ▼
┌─────────────────────────────────────────────────────────────┐
│                      Data Layer                              │
│  (Game logs, Training data, Model checkpoints, Stats)        │
└─────────────────────────────────────────────────────────────┘
```

### Key Design Principles

1. **Separation of Concerns**: Game logic separate from solvers
2. **Interface-Driven**: Clear contracts between modules
3. **Extensibility**: Easy to add new algorithms
4. **Testability**: Each component independently testable
5. **Performance**: Optimized for high-throughput simulations

---

## Game Module Design

### Directory Structure

```
game/
├── __init__.py
├── core/
│   ├── __init__.py
│   ├── card.py              # Card and Deck classes
│   ├── state.py             # Game state representation
│   ├── game.py              # Main game controller
│   ├── rules.py             # Move validation logic
│   └── moves.py             # Move representations
├── ui/
│   ├── __init__.py
│   ├── cli.py               # Command-line interface
│   ├── renderer.py          # Text-based rendering
│   └── gui.py               # (Optional) graphical interface
├── utils/
│   ├── __init__.py
│   ├── serialization.py     # Save/load game states
│   └── replay.py            # Game replay functionality
└── tests/
    ├── __init__.py
    ├── test_card.py
    ├── test_state.py
    ├── test_game.py
    └── test_rules.py
```

### Core Classes

#### Card (`core/card.py`)

```python
from enum import Enum
from dataclasses import dataclass

class Suit(Enum):
    SPADES = '♠'
    HEARTS = '♥'
    DIAMONDS = '♦'
    CLUBS = '♣'

class Rank(Enum):
    ACE = 1
    TWO = 2
    # ... through KING = 13

@dataclass(frozen=True)
class Card:
    """Immutable card representation."""
    suit: Suit
    rank: Rank
    
    @property
    def color(self) -> str:
        """Returns 'red' or 'black'."""
        return 'red' if self.suit in (Suit.HEARTS, Suit.DIAMONDS) else 'black'
    
    @property
    def value(self) -> int:
        """Returns numeric value (1-13)."""
        return self.rank.value
    
    def __str__(self) -> str:
        return f"{self.rank.name[0]}{self.suit.value}"
    
    def __hash__(self) -> int:
        return hash((self.suit, self.rank))

class Deck:
    """Standard 52-card deck with shuffling."""
    
    def __init__(self, seed: Optional[int] = None):
        self.cards: List[Card] = []
        self.rng = random.Random(seed)
        self._initialize()
    
    def _initialize(self):
        """Create all 52 cards."""
        for suit in Suit:
            for rank in Rank:
                self.cards.append(Card(suit, rank))
    
    def shuffle(self):
        """Shuffle the deck in place."""
        self.rng.shuffle(self.cards)
    
    def draw(self, n: int = 1) -> List[Card]:
        """Draw n cards from top of deck."""
        drawn = self.cards[:n]
        self.cards = self.cards[n:]
        return drawn
```

**Design decisions:**
- Cards are **immutable** (frozen dataclass) - safer for state management
- Deck uses **seeded RNG** - reproducible games for testing
- Clean **enum-based** suit/rank representation

#### GameState (`core/state.py`)

```python
from typing import List, Optional, Tuple
from dataclasses import dataclass, field

@dataclass
class GameState:
    """Complete representation of game state."""
    
    # Stock and waste
    stock: List[Card] = field(default_factory=list)
    waste: List[Card] = field(default_factory=list)
    
    # Tableau (7 columns)
    tableau: List[List[Card]] = field(default_factory=lambda: [[] for _ in range(7)])
    tableau_hidden: List[int] = field(default_factory=lambda: [0] * 7)  # Cards face down per column
    
    # Foundations (4 piles, one per suit)
    foundations: Dict[Suit, List[Card]] = field(default_factory=dict)
    
    # Game metadata
    move_count: int = 0
    draw_count: int = 0
    score: int = -52  # Vegas scoring: start at -$52
    seed: Optional[int] = None
    
    def __post_init__(self):
        """Initialize foundations."""
        if not self.foundations:
            for suit in Suit:
                self.foundations[suit] = []
    
    def copy(self) -> 'GameState':
        """Deep copy of game state."""
        return GameState(
            stock=self.stock.copy(),
            waste=self.waste.copy(),
            tableau=[col.copy() for col in self.tableau],
            tableau_hidden=self.tableau_hidden.copy(),
            foundations={suit: cards.copy() for suit, cards in self.foundations.items()},
            move_count=self.move_count,
            draw_count=self.draw_count,
            score=self.score,
            seed=self.seed
        )
    
    def __hash__(self) -> int:
        """Hash for state caching (transposition tables)."""
        # Only hash visible state (not hidden cards, for efficiency)
        return hash((
            tuple(self.waste),
            tuple(tuple(col) for col in self.tableau),
            tuple(self.tableau_hidden),
            tuple((s, tuple(c)) for s, c in self.foundations.items())
        ))
    
    def to_vector(self) -> np.ndarray:
        """Convert state to fixed-size vector for ML."""
        # For neural network input
        # Returns shape (n,) array encoding entire state
        pass  # Detailed implementation in state.py
    
    def get_foundation_count(self) -> int:
        """Total cards in foundations."""
        return sum(len(cards) for cards in self.foundations.values())
    
    def is_winning(self) -> bool:
        """Check if game is won (all 52 cards in foundations)."""
        return self.get_foundation_count() == 52
```

**Design decisions:**
- Uses **dataclasses** for clean syntax
- **Deep copy** method for state exploration
- **Hashing** for transposition tables in search
- **to_vector()** for ML integration
- Separate tracking of **hidden vs visible** cards

#### Move (`core/moves.py`)

```python
from enum import Enum
from dataclasses import dataclass

class MoveType(Enum):
    TABLEAU_TO_FOUNDATION = 'tableau_to_foundation'
    TABLEAU_TO_TABLEAU = 'tableau_to_tableau'
    WASTE_TO_FOUNDATION = 'waste_to_foundation'
    WASTE_TO_TABLEAU = 'waste_to_tableau'
    FOUNDATION_TO_TABLEAU = 'foundation_to_tableau'  # If allowed
    DRAW = 'draw'
    RECYCLE = 'recycle'  # Turn waste back into stock

@dataclass(frozen=True)
class Move:
    """Represents a single game move."""
    move_type: MoveType
    source: Optional[int] = None  # Column index or None
    destination: Optional[int] = None  # Column index or None
    card_count: int = 1  # For moving multiple cards
    card: Optional[Card] = None  # The card being moved (for validation)
    
    def __str__(self) -> str:
        if self.move_type == MoveType.DRAW:
            return "Draw from stock"
        elif self.move_type == MoveType.RECYCLE:
            return "Recycle waste to stock"
        else:
            src = f"col{self.source}" if self.source is not None else "waste"
            dst = f"col{self.destination}" if self.destination is not None else "foundation"
            return f"Move {self.card} from {src} to {dst}"
    
    def __hash__(self) -> int:
        return hash((self.move_type, self.source, self.destination, self.card_count))
```

**Design decisions:**
- **Immutable** moves (frozen dataclass)
- **Type-safe** with enum
- Encodes **all move information** needed for replay
- **Human-readable** string representation

#### Game (`core/game.py`)

```python
class Game:
    """Main game controller."""
    
    def __init__(self, seed: Optional[int] = None):
        self.state = GameState(seed=seed)
        self.deck = Deck(seed=seed)
        self.move_history: List[Move] = []
        self._initialized = False
    
    def deal(self):
        """Deal initial tableau and stock."""
        self.deck.shuffle()
        
        # Deal to tableau
        for col_idx in range(7):
            for row in range(col_idx + 1):
                card = self.deck.draw(1)[0]
                self.state.tableau[col_idx].append(card)
                if row < col_idx:  # All but last card are face down
                    self.state.tableau_hidden[col_idx] += 1
        
        # Remaining cards go to stock
        self.state.stock = self.deck.cards.copy()
        self._initialized = True
    
    def get_valid_moves(self) -> List[Move]:
        """Get all valid moves from current state."""
        from .rules import get_valid_moves
        return get_valid_moves(self.state)
    
    def make_move(self, move: Move) -> bool:
        """
        Apply move to game state.
        Returns True if successful, False if invalid.
        """
        if not self.is_valid_move(move):
            return False
        
        # Apply move (delegates to rules module)
        from .rules import apply_move
        self.state = apply_move(self.state, move)
        self.move_history.append(move)
        
        # Update scoring
        self._update_score(move)
        
        return True
    
    def is_valid_move(self, move: Move) -> bool:
        """Check if move is legal in current state."""
        from .rules import is_valid_move
        return is_valid_move(self.state, move)
    
    def is_over(self) -> bool:
        """Check if game is complete (won or no more moves)."""
        if self.state.is_winning():
            return True
        return len(self.get_valid_moves()) == 0
    
    def _update_score(self, move: Move):
        """Update Vegas score (+$5 per card to foundation)."""
        if move.move_type in (MoveType.TABLEAU_TO_FOUNDATION, 
                               MoveType.WASTE_TO_FOUNDATION):
            self.state.score += 5
    
    def get_state_copy(self) -> GameState:
        """Get copy of current state (for solvers)."""
        return self.state.copy()
    
    def set_state(self, state: GameState):
        """Set game to specific state (for replay, testing)."""
        self.state = state.copy()
```

**Design decisions:**
- **Separation**: Game controller separate from state
- **Move validation** delegated to rules module
- **Immutability**: get_state_copy() prevents accidental modification
- **Move history** for replay and analysis

#### Rules (`core/rules.py`)

```python
def get_valid_moves(state: GameState) -> List[Move]:
    """
    Generate all valid moves from current state.
    This is called frequently by solvers, so must be efficient.
    """
    moves = []
    
    # Check if we can draw
    if state.stock:
        moves.append(Move(MoveType.DRAW))
    elif state.waste and not state.stock:
        moves.append(Move(MoveType.RECYCLE))
    
    # Waste to foundation
    if state.waste:
        top_card = state.waste[-1]
        if can_move_to_foundation(top_card, state):
            moves.append(Move(
                MoveType.WASTE_TO_FOUNDATION,
                card=top_card
            ))
    
    # Waste to tableau
    if state.waste:
        top_card = state.waste[-1]
        for col_idx in range(7):
            if can_move_to_tableau(top_card, state.tableau[col_idx]):
                moves.append(Move(
                    MoveType.WASTE_TO_TABLEAU,
                    destination=col_idx,
                    card=top_card
                ))
    
    # Tableau to foundation
    for col_idx in range(7):
        if state.tableau[col_idx]:
            # Only consider moving the top card
            visible_start = len(state.tableau[col_idx]) - \
                           (len(state.tableau[col_idx]) - state.tableau_hidden[col_idx])
            if visible_start < len(state.tableau[col_idx]):
                top_card = state.tableau[col_idx][-1]
                if can_move_to_foundation(top_card, state):
                    moves.append(Move(
                        MoveType.TABLEAU_TO_FOUNDATION,
                        source=col_idx,
                        card=top_card
                    ))
    
    # Tableau to tableau (can move sequences)
    for src_col in range(7):
        if not state.tableau[src_col]:
            continue
        
        # Find start of visible sequence
        visible_start = state.tableau_hidden[src_col]
        
        for dst_col in range(7):
            if src_col == dst_col:
                continue
            
            # Try moving sequences of different lengths
            for seq_start in range(visible_start, len(state.tableau[src_col])):
                if is_valid_sequence(state.tableau[src_col][seq_start:]):
                    card = state.tableau[src_col][seq_start]
                    if can_move_to_tableau(card, state.tableau[dst_col]):
                        moves.append(Move(
                            MoveType.TABLEAU_TO_TABLEAU,
                            source=src_col,
                            destination=dst_col,
                            card_count=len(state.tableau[src_col]) - seq_start,
                            card=card
                        ))
    
    return moves

def can_move_to_foundation(card: Card, state: GameState) -> bool:
    """Check if card can be moved to its foundation."""
    foundation = state.foundations[card.suit]
    
    if not foundation:
        return card.rank == Rank.ACE
    
    top_card = foundation[-1]
    return card.rank.value == top_card.rank.value + 1

def can_move_to_tableau(card: Card, tableau_column: List[Card]) -> bool:
    """Check if card can be moved to tableau column."""
    if not tableau_column:
        # Empty column - only King can go here
        return card.rank == Rank.KING
    
    top_card = tableau_column[-1]
    
    # Must be descending rank
    if card.rank.value != top_card.rank.value - 1:
        return False
    
    # Must be opposite color
    return card.color != top_card.color

def is_valid_sequence(cards: List[Card]) -> bool:
    """Check if cards form valid descending alternating-color sequence."""
    if len(cards) <= 1:
        return True
    
    for i in range(len(cards) - 1):
        if cards[i].rank.value != cards[i+1].rank.value + 1:
            return False
        if cards[i].color == cards[i+1].color:
            return False
    
    return True

def apply_move(state: GameState, move: Move) -> GameState:
    """
    Apply move to state, returning new state.
    Does NOT modify original state (functional approach).
    """
    new_state = state.copy()
    new_state.move_count += 1
    
    if move.move_type == MoveType.DRAW:
        # Draw 3 cards (or fewer if stock is small)
        drawn = new_state.stock[:3]
        new_state.stock = new_state.stock[3:]
        new_state.waste.extend(drawn)
        new_state.draw_count += 1
    
    elif move.move_type == MoveType.RECYCLE:
        new_state.stock = new_state.waste[::-1]  # Reverse order
        new_state.waste = []
    
    elif move.move_type == MoveType.WASTE_TO_FOUNDATION:
        card = new_state.waste.pop()
        new_state.foundations[card.suit].append(card)
    
    elif move.move_type == MoveType.WASTE_TO_TABLEAU:
        card = new_state.waste.pop()
        new_state.tableau[move.destination].append(card)
    
    elif move.move_type == MoveType.TABLEAU_TO_FOUNDATION:
        card = new_state.tableau[move.source].pop()
        new_state.foundations[card.suit].append(card)
        # Reveal card if column now has face-down cards exposed
        if (new_state.tableau_hidden[move.source] > 0 and 
            len(new_state.tableau[move.source]) == new_state.tableau_hidden[move.source]):
            new_state.tableau_hidden[move.source] -= 1
    
    elif move.move_type == MoveType.TABLEAU_TO_TABLEAU:
        # Move sequence of cards
        cards = new_state.tableau[move.source][-move.card_count:]
        new_state.tableau[move.source] = new_state.tableau[move.source][:-move.card_count]
        new_state.tableau[move.destination].extend(cards)
        # Reveal card if needed
        if (new_state.tableau_hidden[move.source] > 0 and 
            len(new_state.tableau[move.source]) == new_state.tableau_hidden[move.source]):
            new_state.tableau_hidden[move.source] -= 1
    
    return new_state
```

**Design decisions:**
- **Functional approach**: apply_move returns new state, doesn't modify original
- **Efficient move generation**: Critical for solver performance
- **Sequence validation**: Properly handles moving multiple cards
- **Face-down card management**: Automatically reveals when exposed

---

## Optimization Module Design

### Directory Structure

```
optimization/
├── __init__.py
├── solvers/
│   ├── __init__.py
│   ├── base.py              # Abstract base class for all solvers
│   ├── random_solver.py
│   ├── heuristic_solver.py
│   ├── mcts_solver.py
│   ├── expectimax_solver.py
│   ├── beam_search_solver.py
│   ├── minimax_solver.py    # Educational
│   └── rl_solver.py
├── training/
│   ├── __init__.py
│   ├── train_policy.py      # Policy network training
│   ├── train_value.py       # Value network training
│   ├── self_play.py         # Generate training data
│   └── config.py            # Training hyperparameters
├── models/
│   ├── __init__.py
│   ├── networks.py          # Neural network architectures
│   ├── features.py          # State featurization for ML
│   └── policy_value.py      # Combined policy-value network
├── evaluation/
│   ├── __init__.py
│   ├── benchmark.py         # Run standardized benchmarks
│   ├── tournament.py        # Compare multiple solvers
│   ├── analyze.py           # Analyze game statistics
│   └── visualize.py         # Plot results
├── utils/
│   ├── __init__.py
│   ├── replay_buffer.py     # Experience replay for RL
│   ├── logger.py            # Experiment logging
│   └── parallel.py          # Parallel game execution
└── tests/
    ├── __init__.py
    ├── test_solvers.py
    └── test_training.py
```

### Base Solver Interface

```python
from abc import ABC, abstractmethod
from typing import Optional

class Solver(ABC):
    """Abstract base class for all solvers."""
    
    def __init__(self, name: str):
        self.name = name
        self.games_played = 0
        self.total_score = 0
        self.wins = 0
    
    @abstractmethod
    def choose_move(self, game_state: GameState) -> Optional[Move]:
        """
        Given a game state, choose the best move.
        Returns None if no valid moves.
        """
        pass
    
    def play_game(self, game: Game) -> dict:
        """
        Play a complete game and return statistics.
        """
        game.deal()
        moves_made = []
        
        while not game.is_over():
            move = self.choose_move(game.get_state_copy())
            if move is None:
                break
            game.make_move(move)
            moves_made.append(move)
        
        # Update statistics
        self.games_played += 1
        self.total_score += game.state.score
        if game.state.is_winning():
            self.wins += 1
        
        return {
            'score': game.state.score,
            'won': game.state.is_winning(),
            'moves': len(moves_made),
            'foundation_count': game.state.get_foundation_count()
        }
    
    def get_statistics(self) -> dict:
        """Return solver performance statistics."""
        return {
            'name': self.name,
            'games_played': self.games_played,
            'win_rate': self.wins / self.games_played if self.games_played > 0 else 0,
            'avg_score': self.total_score / self.games_played if self.games_played > 0 else 0,
            'total_wins': self.wins
        }
    
    def reset_statistics(self):
        """Reset performance counters."""
        self.games_played = 0
        self.total_score = 0
        self.wins = 0
```

**Design decisions:**
- **Abstract base class** enforces interface
- **Statistics tracking** built-in to all solvers
- **Consistent interface** makes benchmarking easy
- **play_game()** method standardizes evaluation

### Solver Implementations

Each solver extends the base class:

```python
# solvers/random_solver.py
class RandomSolver(Solver):
    def __init__(self, seed=None):
        super().__init__("Random")
        self.rng = random.Random(seed)
    
    def choose_move(self, game_state: GameState) -> Optional[Move]:
        from game.core.rules import get_valid_moves
        moves = get_valid_moves(game_state)
        return self.rng.choice(moves) if moves else None

# solvers/heuristic_solver.py
class HeuristicSolver(Solver):
    def __init__(self, heuristic_fn=None):
        super().__init__("Heuristic")
        self.heuristic_fn = heuristic_fn or default_heuristic
    
    def choose_move(self, game_state: GameState) -> Optional[Move]:
        from game.core.rules import get_valid_moves
        moves = get_valid_moves(game_state)
        if not moves:
            return None
        
        # Score moves and return best
        scored = [(self.heuristic_fn(m, game_state), m) for m in moves]
        return max(scored)[0][1]

# solvers/mcts_solver.py
class MCTSSolver(Solver):
    def __init__(self, num_simulations=1000, exploration=1.41):
        super().__init__(f"MCTS-{num_simulations}")
        self.num_simulations = num_simulations
        self.exploration = exploration
        self.transposition_table = {}
    
    def choose_move(self, game_state: GameState) -> Optional[Move]:
        root = MCTSNode(game_state, exploration=self.exploration)
        
        for _ in range(self.num_simulations):
            # ... MCTS algorithm implementation
            pass
        
        # Return best move
        return root.get_best_move()
```

**Design decisions:**
- Each solver is **self-contained**
- **Parameterized** constructors (num_simulations, etc.)
- **Named** solvers for easy identification in benchmarks

### Training Pipeline

```python
# training/train_policy.py

class PolicyTrainer:
    """Trains a policy network via self-play."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.policy_net = PolicyNetwork(
            state_dim=config.state_dim,
            action_dim=config.action_dim,
            hidden_dim=config.hidden_dim
        )
        self.optimizer = torch.optim.Adam(
            self.policy_net.parameters(),
            lr=config.learning_rate
        )
        self.replay_buffer = ReplayBuffer(config.buffer_size)
    
    def train(self, num_epochs: int):
        """Main training loop."""
        for epoch in range(num_epochs):
            # Generate training data via self-play
            episodes = self.generate_episodes(self.config.episodes_per_epoch)
            
            # Add to replay buffer
            for episode in episodes:
                self.replay_buffer.add_episode(episode)
            
            # Train on batch
            loss = self.train_epoch()
            
            # Evaluate
            if epoch % self.config.eval_interval == 0:
                stats = self.evaluate()
                self.log_progress(epoch, loss, stats)
                self.save_checkpoint(epoch)
    
    def generate_episodes(self, num_episodes: int) -> List[Episode]:
        """Generate training data via self-play."""
        episodes = []
        
        for _ in range(num_episodes):
            game = Game(seed=None)  # Random seed each time
            game.deal()
            
            states, actions, rewards = [], [], []
            
            while not game.is_over():
                state = game.get_state_copy()
                
                # Get action from policy
                action = self.policy_net.sample_action(state)
                
                # Apply action
                move = state.get_valid_moves()[action]
                game.make_move(move)
                
                states.append(state.to_vector())
                actions.append(action)
                rewards.append(0)  # Intermediate reward
            
            # Terminal reward
            rewards[-1] = game.state.score
            
            episodes.append(Episode(states, actions, rewards))
        
        return episodes
    
    def train_epoch(self) -> float:
        """Train on batch from replay buffer."""
        batch = self.replay_buffer.sample(self.config.batch_size)
        
        # Compute policy gradient loss
        loss = self.compute_loss(batch)
        
        # Update network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
```

**Design decisions:**
- **Config-driven** training (easy to tune hyperparameters)
- **Self-play** data generation
- **Replay buffer** for experience replay
- **Checkpointing** for recovery and evaluation

---

## Data Flow & Interfaces

### Game → Solver Interface

```
┌─────────────┐
│    Game     │
│             │
│  .deal()    │
└──────┬──────┘
       │
       │ get_state_copy()
       ▼
┌─────────────────┐
│   GameState     │
│   (immutable)   │
└──────┬──────────┘
       │
       │ passed to solver
       ▼
┌─────────────────┐
│     Solver      │
│                 │
│ .choose_move()  │
└──────┬──────────┘
       │
       │ returns Move
       ▼
┌─────────────────┐
│      Game       │
│                 │
│  .make_move()   │
└─────────────────┘
```

**Key invariant**: Solvers receive **immutable copies** of game state, cannot accidentally modify game.

### Solver → Evaluation Interface

```python
# evaluation/benchmark.py

class Benchmark:
    """Standardized evaluation suite."""
    
    def __init__(self, test_seeds: List[int]):
        self.test_seeds = test_seeds
    
    def evaluate_solver(self, solver: Solver, verbose=True) -> BenchmarkResults:
        """
        Evaluate solver on fixed test set.
        Returns comprehensive statistics.
        """
        results = []
        
        for seed in self.test_seeds:
            game = Game(seed=seed)
            result = solver.play_game(game)
            result['seed'] = seed
            results.append(result)
            
            if verbose:
                print(f"Seed {seed}: Score={result['score']}, Won={result['won']}")
        
        return BenchmarkResults(results)

class BenchmarkResults:
    """Container for benchmark statistics."""
    
    def __init__(self, results: List[dict]):
        self.results = results
    
    @property
    def win_rate(self) -> float:
        return sum(r['won'] for r in self.results) / len(self.results)
    
    @property
    def avg_score(self) -> float:
        return sum(r['score'] for r in self.results) / len(self.results)
    
    @property
    def median_score(self) -> float:
        scores = sorted(r['score'] for r in self.results)
        return scores[len(scores) // 2]
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame for analysis."""
        return pd.DataFrame(self.results)
```

### Training → Model Interface

```python
# models/networks.py

class PolicyNetwork(nn.Module):
    """Neural network for policy (action probabilities)."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, action_dim),
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Returns logits for each action."""
        return self.network(state)
    
    def get_action_probs(self, state: GameState, valid_moves: List[Move]) -> np.ndarray:
        """Get probabilities for valid moves only."""
        state_vec = torch.FloatTensor(state.to_vector())
        logits = self.forward(state_vec)
        
        # Mask invalid moves
        mask = torch.full_like(logits, float('-inf'))
        for move_idx in valid_moves:
            mask[move_idx] = 0
        
        masked_logits = logits + mask
        probs = torch.softmax(masked_logits, dim=0)
        
        return probs.detach().numpy()
```

---

## Testing Strategy

### Unit Tests

Each module has comprehensive unit tests:

```python
# game/tests/test_rules.py

class TestRules(unittest.TestCase):
    
    def setUp(self):
        """Create test game state."""
        self.state = GameState()
        # Set up specific configuration
    
    def test_ace_to_empty_foundation(self):
        """Test that Ace can move to empty foundation."""
        ace_spades = Card(Suit.SPADES, Rank.ACE)
        self.assertTrue(can_move_to_foundation(ace_spades, self.state))
    
    def test_two_requires_ace(self):
        """Test that 2 cannot move to empty foundation."""
        two_spades = Card(Suit.SPADES, Rank.TWO)
        self.assertFalse(can_move_to_foundation(two_spades, self.state))
    
    def test_alternating_colors(self):
        """Test tableau building requires alternating colors."""
        red_king = Card(Suit.HEARTS, Rank.KING)
        red_queen = Card(Suit.DIAMONDS, Rank.QUEEN)
        black_queen = Card(Suit.SPADES, Rank.QUEEN)
        
        self.state.tableau[0] = [red_king]
        
        # Black queen should be valid
        self.assertTrue(can_move_to_tableau(black_queen, self.state.tableau[0]))
        
        # Red queen should be invalid
        self.assertFalse(can_move_to_tableau(red_queen, self.state.tableau[0]))
```

### Integration Tests

Test full game scenarios:

```python
# game/tests/test_game.py

class TestGameIntegration(unittest.TestCase):
    
    def test_full_game_simulation(self):
        """Test that a game can be played to completion."""
        game = Game(seed=42)
        game.deal()
        
        moves_made = 0
        while not game.is_over() and moves_made < 1000:  # Prevent infinite loop
            moves = game.get_valid_moves()
            if moves:
                game.make_move(random.choice(moves))
                moves_made += 1
        
        # Game should terminate
        self.assertTrue(game.is_over() or moves_made < 1000)
    
    def test_deterministic_replay(self):
        """Test that same seed produces same game."""
        game1 = Game(seed=123)
        game1.deal()
        
        game2 = Game(seed=123)
        game2.deal()
        
        # Initial states should be identical
        self.assertEqual(game1.state, game2.state)
```

### Solver Tests

Verify solver behavior:

```python
# optimization/tests/test_solvers.py

class TestSolvers(unittest.TestCase):
    
    def test_random_solver_makes_valid_moves(self):
        """Test that random solver only makes legal moves."""
        solver = RandomSolver(seed=42)
        game = Game(seed=42)
        game.deal()
        
        for _ in range(50):  # Try 50 moves
            move = solver.choose_move(game.get_state_copy())
            if move is None:
                break
            self.assertTrue(game.is_valid_move(move))
            game.make_move(move)
    
    def test_heuristic_better_than_random(self):
        """Test that heuristic solver outperforms random."""
        random_solver = RandomSolver(seed=1)
        heuristic_solver = HeuristicSolver()
        
        benchmark = Benchmark(test_seeds=range(100))
        
        random_results = benchmark.evaluate_solver(random_solver, verbose=False)
        heuristic_results = benchmark.evaluate_solver(heuristic_solver, verbose=False)
        
        # Heuristic should have better average score
        self.assertGreater(heuristic_results.avg_score, random_results.avg_score)
```

---

## Performance Considerations

### Optimization Hotspots

1. **Move generation** (`get_valid_moves()`)
   - Called millions of times during search
   - Must be extremely efficient
   - Profile and optimize carefully

2. **State copying** (`GameState.copy()`)
   - Frequent in tree search
   - Consider copy-on-write or state diffs

3. **State hashing** (`GameState.__hash__()`)
   - For transposition tables
   - Balance completeness vs speed

### Performance Targets

| Operation | Target Time | Notes |
|-----------|-------------|-------|
| get_valid_moves() | <100μs | Critical path |
| apply_move() | <50μs | Called frequently |
| State.copy() | <200μs | For search algorithms |
| State.to_vector() | <500μs | For ML inference |
| NN forward pass | <5ms | Batched inference |

### Profiling Strategy

```python
# utils/profiling.py

import cProfile
import pstats

def profile_solver(solver: Solver, num_games: int = 10):
    """Profile solver performance."""
    profiler = cProfile.Profile()
    
    profiler.enable()
    for _ in range(num_games):
        game = Game()
        solver.play_game(game)
    profiler.disable()
    
    # Print stats
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(20)
```

### Parallelization

```python
# utils/parallel.py

from multiprocessing import Pool

def evaluate_parallel(solver: Solver, seeds: List[int], num_workers: int = 4) -> List[dict]:
    """
    Evaluate solver on multiple games in parallel.
    """
    def play_one_game(seed):
        game = Game(seed=seed)
        return solver.play_game(game)
    
    with Pool(num_workers) as pool:
        results = pool.map(play_one_game, seeds)
    
    return results
```

---

## Configuration Management

### Centralized Config

```python
# optimization/training/config.py

from dataclasses import dataclass

@dataclass
class TrainingConfig:
    """Configuration for training runs."""
    
    # Model architecture
    state_dim: int = 416  # Computed from state representation
    action_dim: int = 200  # Max possible moves
    hidden_dim: int = 256
    
    # Training hyperparameters
    learning_rate: float = 3e-4
    batch_size: int = 128
    buffer_size: int = 100000
    episodes_per_epoch: int = 100
    num_epochs: int = 1000
    
    # Evaluation
    eval_interval: int = 10
    eval_games: int = 50
    
    # System
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_workers: int = 4
    seed: int = 42
    
    # Checkpointing
    checkpoint_dir: str = 'checkpoints/'
    save_interval: int = 50

# Usage:
config = TrainingConfig(
    learning_rate=1e-3,
    hidden_dim=512
)
```

---

## Summary

This architecture provides:

1. **Clear separation of concerns**: Game engine, solvers, training, evaluation
2. **Extensibility**: Easy to add new algorithms via base classes
3. **Testability**: Comprehensive test coverage at all levels
4. **Performance**: Optimized for high-throughput simulations
5. **Reproducibility**: Seeded RNG, deterministic behavior
6. **Modularity**: Each component can be developed/tested independently

The design supports both rapid prototyping (simple solvers) and production deployment (trained neural networks), making it suitable for research, education, and practical applications.
