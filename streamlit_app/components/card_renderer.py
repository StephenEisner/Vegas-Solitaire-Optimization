"""Card rendering utilities for Streamlit UI."""

from typing import Optional, List
from game.core.card import Card, Suit, Rank


# Unicode suit symbols
SUIT_SYMBOLS = {
    Suit.HEARTS: '♥',
    Suit.DIAMONDS: '♦',
    Suit.CLUBS: '♣',
    Suit.SPADES: '♠'
}

# Rank display
RANK_DISPLAY = {
    Rank.ACE: 'A',
    Rank.TWO: '2',
    Rank.THREE: '3',
    Rank.FOUR: '4',
    Rank.FIVE: '5',
    Rank.SIX: '6',
    Rank.SEVEN: '7',
    Rank.EIGHT: '8',
    Rank.NINE: '9',
    Rank.TEN: '10',
    Rank.JACK: 'J',
    Rank.QUEEN: 'Q',
    Rank.KING: 'K'
}


def get_card_color(card: Card) -> str:
    """Get the color class for a card."""
    return "card-red" if card.suit in [Suit.HEARTS, Suit.DIAMONDS] else "card-black"


def render_card(card: Optional[Card], face_up: bool = True, empty: bool = False) -> str:
    """
    Render a single card as HTML.

    Args:
        card: The card to render (None for empty slot)
        face_up: Whether the card is face up
        empty: Whether to render as an empty slot

    Returns:
        HTML string for the card
    """
    if empty or card is None:
        return '<div class="card card-empty"></div>'

    if not face_up:
        return '<div class="card card-back">🂠</div>'

    color_class = get_card_color(card)
    rank = RANK_DISPLAY[card.rank]
    suit = SUIT_SYMBOLS[card.suit]

    return f'<div class="card {color_class}" title="{rank}{suit}">{rank}{suit}</div>'


def render_card_stack(cards: List[Card], face_down_count: int = 0,
                      vertical_offset: int = 20, show_empty: bool = True) -> str:
    """
    Render a stack of cards with vertical offset.

    Args:
        cards: List of face-up cards (top of stack first)
        face_down_count: Number of face-down cards at bottom
        vertical_offset: Vertical offset in pixels between cards
        show_empty: Whether to show empty slot if no cards

    Returns:
        HTML string for the card stack
    """
    if not cards and face_down_count == 0:
        if show_empty:
            return render_card(None, empty=True)
        return ''

    html_parts = []
    html_parts.append('<div style="position: relative; display: inline-block; width: 64px;">')

    # Render face-down cards
    for i in range(face_down_count):
        top = i * vertical_offset
        html_parts.append(
            f'<div style="position: absolute; top: {top}px; left: 0;">'
            f'{render_card(None, face_up=False)}'
            f'</div>'
        )

    # Render face-up cards
    start_offset = face_down_count * vertical_offset
    for i, card in enumerate(cards):
        top = start_offset + i * vertical_offset
        html_parts.append(
            f'<div style="position: absolute; top: {top}px; left: 0;">'
            f'{render_card(card, face_up=True)}'
            f'</div>'
        )

    # Add spacer for height
    total_height = start_offset + len(cards) * vertical_offset + 84
    html_parts.append(f'<div style="height: {total_height}px;"></div>')

    html_parts.append('</div>')

    return ''.join(html_parts)


def render_foundation_pile(cards: List[Card], suit: Suit) -> str:
    """
    Render a foundation pile.

    Args:
        cards: List of cards in the foundation (bottom to top)
        suit: The suit for this foundation

    Returns:
        HTML string for the foundation pile
    """
    if not cards:
        # Show empty foundation with suit indicator
        suit_symbol = SUIT_SYMBOLS[suit]
        color = "card-red" if suit in [Suit.HEARTS, Suit.DIAMONDS] else "card-black"
        return f'<div class="card card-empty"><small class="{color}">{suit_symbol}</small></div>'

    # Show top card
    return render_card(cards[-1], face_up=True)


def render_stock_and_waste(stock_count: int, waste: List[Card]) -> str:
    """
    Render the stock and waste piles.

    Args:
        stock_count: Number of cards in stock
        waste: List of cards in waste (top card last)

    Returns:
        HTML string for stock and waste
    """
    html_parts = []

    # Stock pile
    if stock_count > 0:
        html_parts.append(render_card(None, face_up=False))
        html_parts.append(f'<div style="display: inline-block; margin-left: -40px; font-size: 0.8rem; color: #666;">({stock_count})</div>')
    else:
        html_parts.append(render_card(None, empty=True))

    html_parts.append('<div style="display: inline-block; width: 20px;"></div>')

    # Waste pile - show top 3 cards with offset
    if waste:
        visible_waste = waste[-3:] if len(waste) >= 3 else waste
        html_parts.append('<div style="position: relative; display: inline-block; width: 140px; height: 84px;">')

        for i, card in enumerate(visible_waste):
            left = i * 30
            html_parts.append(
                f'<div style="position: absolute; top: 0; left: {left}px;">'
                f'{render_card(card, face_up=True)}'
                f'</div>'
            )

        html_parts.append('</div>')
    else:
        html_parts.append(render_card(None, empty=True))

    return ''.join(html_parts)
