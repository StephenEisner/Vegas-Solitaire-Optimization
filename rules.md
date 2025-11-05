# Vegas Solitaire: Complete Rules

## Overview

Vegas Solitaire, also known as Vegas Klondike, is a variant of the classic Klondike solitaire with a gambling-style scoring system. The game simulates playing solitaire in a casino where you pay to play and earn money for each card you successfully place in the foundations.

## Objective

Move all 52 cards to the four foundation piles, building them up by suit from Ace to King. In Vegas mode, the goal is to maximize profit.

## Setup

1. **Deck**: One standard 52-card deck
2. **Shuffle**: Randomize the deck order
3. **Deal**:
   - Create 7 tableau columns
   - Column 1: 1 card (face up)
   - Column 2: 2 cards (1 face down, 1 face up)
   - Column 3: 3 cards (2 face down, 1 face up)
   - Continue pattern to Column 7: 7 cards (6 face down, 1 face up)
   - Remaining cards form the stock (face down)

Initial layout:
```
Stock: [24 cards]  Waste: []  Foundations: [♠] [♥] [♦] [♣]

Tableau:
  1    2    3    4    5    6    7
[A♠] [▓] [▓] [▓] [▓] [▓] [▓]
     [2♥] [▓] [▓] [▓] [▓] [▓]
          [3♦] [▓] [▓] [▓] [▓]
               [4♣] [▓] [▓] [▓]
                    [5♠] [▓] [▓]
                         [6♥] [▓]
                              [7♦]
```

## Foundations

- **Four foundation piles**, one for each suit (♠ ♥ ♦ ♣)
- **Building rule**: Must start with Ace, then build up sequentially (A, 2, 3, ..., Q, K)
- **Suit rule**: Only cards of the same suit can be placed in each foundation
- Cards in foundations are typically not moved back to the tableau (though some variants allow this)

## Tableau

- **Seven columns** of varying lengths
- **Building rule**: Cards are placed in descending rank (K, Q, J, 10, ..., 2, A)
- **Color rule**: Cards must alternate colors (red ♥♦, black ♠♣)
- **Face-up rule**: Only face-up cards can be moved
- **Revealing**: When a face-down card is exposed (becomes the top card of a column), it is immediately turned face up
- **Moving sequences**: Multiple cards can be moved together if they form a valid descending, alternating-color sequence
- **Empty spaces**: Can only be filled by a King (or a sequence starting with a King)

### Tableau Example
```
Valid build:
[K♠]
[Q♥] ← can place on K♠ (descending, alternating color)
[J♣] ← can place on Q♥
[10♦] ← can place on J♣

Invalid:
[K♠]
[Q♣] ← cannot place (same color)
[J♥] ← cannot place (not descending from K)
```

## Stock & Waste

### Draw 3 Rule (Standard Vegas)
- **Drawing**: Click the stock to draw 3 cards at a time
- **Waste pile**: Drawn cards are placed face-up in the waste pile, fanned so all three are visible
- **Top card**: Only the top card of the waste pile can be played
- **Cycling**: When the stock is empty, the waste pile can be turned over to form a new stock
- **Redeal limit**: In Vegas mode, unlimited redeals (some variants allow only 1 or 3 passes)

### Draw 1 Rule (Easier Variant)
- Draw one card at a time from stock
- Higher win rate, sometimes used for practice

## Valid Moves

### 1. Tableau to Foundation
Move the top card of a tableau pile to a foundation if:
- It's an Ace and the foundation is empty, OR
- It's the next sequential rank in the correct suit

### 2. Waste to Foundation
Move the top waste card to a foundation (same rules as above)

### 3. Tableau to Tableau
Move card(s) from one tableau pile to another if:
- The moved card(s) form a descending, alternating-color sequence
- The destination card is one rank higher than the moved card
- The moved card is opposite color from the destination card

### 4. Waste to Tableau
Move the top waste card to a tableau pile (same rules as above)

### 5. Foundation to Tableau
(Not allowed in strict Vegas rules, but some variants permit this)

### 6. Draw from Stock
Draw the next 3 cards (or 1 card in draw-1 variant)

### 7. Recycle Waste to Stock
When stock is empty, turn the waste pile over to form a new stock

## Winning & Losing

### Win Condition
All 52 cards are successfully placed in the four foundation piles (13 cards each, A through K)

### Loss Condition
No valid moves remain and not all cards are in foundations

### Determining Winnability
Not all deals are winnable. The initial shuffle determines if a perfect game exists. However, determining winnability without playing is computationally complex.

## Vegas Scoring System

### The Gambling Model
- **Entry cost**: You pay $52 to play one game (representing $1 per card in the deck)
- **Earnings**: You earn $5 for each card successfully placed in a foundation
- **Net profit/loss**: Final score = ($5 × cards in foundations) - $52

### Scoring Examples

**Perfect game (win):**
- 52 cards in foundations
- $5 × 52 = $260 earned
- $260 - $52 = **$208 profit**

**Partial game:**
- 24 cards in foundations
- $5 × 24 = $120 earned
- $120 - $52 = **$68 profit**

**Poor game:**
- 8 cards in foundations
- $5 × 8 = $40 earned
- $40 - $52 = **-$12 loss**

**Break-even threshold:**
- Need 11 cards in foundations to break even
- $5 × 11 = $55 (slight profit)

### Expected Value
- Random play: Typically negative expected value (house edge)
- Optimal play: Unknown, but likely still negative for most deals
- Goal of optimization: Maximize expected profit per game

## Strategy Considerations

### Basic Principles
1. **Prioritize foundation moves**: Generally safe and opens up cards
2. **Create empty tableau spaces**: Kings are powerful
3. **Expose face-down cards**: More options = better decisions
4. **Avoid blocking**: Don't bury cards you'll need later
5. **Plan ahead**: Consider which cards you're trying to expose

### Common Pitfalls
- Moving cards to foundations too early (may need them in tableau)
- Filling empty spaces with non-Kings
- Not thinking several moves ahead
- Ignoring which cards are still hidden in stock

### Advanced Tactics
- Track which cards remain in stock
- Calculate probability of drawing needed cards
- Recognize dead-end positions early
- Balance short-term gains vs long-term position

## Rule Variations

### Redeals
- **Unlimited**: Can cycle through stock infinitely (standard Vegas)
- **3 passes**: Can go through stock up to 3 times
- **1 pass**: No recycling (hardest)

### Draw Count
- **Draw 3**: Standard, more difficult
- **Draw 1**: Each card visible, easier

### Foundation Building
- **Strict**: Cannot move cards from foundations
- **Flexible**: Can move foundation cards back to tableau

### Empty Tableau Spaces
- **Kings only**: Standard rule
- **Any card**: Some variants allow any card

### Aces
- **Auto-foundation**: Aces automatically go to foundations
- **Player choice**: Player decides when to move aces

## This Project's Focus

We'll primarily implement:
- **Draw 3** (standard Vegas difficulty)
- **Unlimited redeals** (gives more strategic depth)
- **Kings-only empty spaces** (standard rule)
- **No foundation reversals** (cards stay in foundations)

This represents the "classic" Vegas Solitaire as played in casinos and provides an interesting optimization challenge.

## References & Resources

- Klondike Solitaire: Classic card game foundation
- Vegas rules: Gambling-inspired scoring system
- Computational complexity: Proven to be NP-complete in general case

## Terminology

- **Stock**: Face-down draw pile
- **Waste**: Discarded cards from stock, face-up
- **Tableau**: The 7 main columns where building happens
- **Foundations**: The 4 goal piles (one per suit)
- **Build**: Place cards in descending, alternating-color order
- **Sequence**: Multiple cards that can move together as a unit
- **Reveal/Flip**: Turn a face-down card face-up

---

Understanding these rules deeply is essential for both playing the game and building optimization algorithms. The interplay between hidden information (stock), revealed information (waste, face-up tableau), and strategic choice creates the game's complexity.
