"""Game board rendering for Streamlit UI."""

import streamlit as st
from typing import Optional
from game.core.game import Game
from game.core.card import Suit
from .card_renderer import (
    render_card_stack,
    render_foundation_pile,
    render_stock_and_waste
)


def render_game_board(game: Game, show_score: bool = True) -> None:
    """
    Render the complete game board.

    Args:
        game: The game to render
        show_score: Whether to show the score display
    """
    state = game.state

    # Score display
    if show_score:
        score_color = "🟢" if state.score >= 0 else "🔴"
        profit = state.score
        st.markdown(
            f'<div class="score-display">'
            f'{score_color} Score: ${profit} | '
            f'💰 Profit: ${profit + 52} | '
            f'📊 Foundation: {state.foundation_count}/52'
            f'</div>',
            unsafe_allow_html=True
        )

    # Top row: Stock/Waste and Foundations
    st.markdown("### Stock & Foundations")

    top_row = st.columns([1.5, 0.5, 2])

    with top_row[0]:
        st.markdown("**Stock & Waste**")
        stock_html = render_stock_and_waste(
            stock_count=len(state.stock),
            waste=state.waste
        )
        st.markdown(stock_html, unsafe_allow_html=True)
        st.caption(f"Stock: {len(state.stock)} | Waste: {len(state.waste)} | Pass: {state.passes_through_deck}/3")

    with top_row[2]:
        st.markdown("**Foundations**")
        foundations_html = []
        for suit in [Suit.HEARTS, Suit.DIAMONDS, Suit.CLUBS, Suit.SPADES]:
            foundations_html.append(
                render_foundation_pile(state.foundations[suit], suit)
            )
        st.markdown(
            '<div style="display: flex; gap: 10px;">' +
            ''.join(foundations_html) +
            '</div>',
            unsafe_allow_html=True
        )

    st.markdown("---")

    # Tableau
    st.markdown("### Tableau")

    tableau_cols = st.columns(7)

    for col_idx, col in enumerate(tableau_cols):
        with col:
            column = state.tableau[col_idx]

            # Count face-down cards
            face_down_count = 0
            face_up_cards = []

            for i, (card, is_face_up) in enumerate(column):
                if is_face_up:
                    face_up_cards = [c for c, _ in column[i:]]
                    break
                face_down_count += 1

            # Render column
            st.markdown(f"**Col {col_idx + 1}**")
            if not column:
                column_html = '<div class="card card-empty"></div>'
            else:
                column_html = render_card_stack(
                    cards=face_up_cards,
                    face_down_count=face_down_count,
                    vertical_offset=25
                )

            st.markdown(column_html, unsafe_allow_html=True)
            st.caption(f"{len(column)} cards")

    # Game status
    st.markdown("---")

    # Check win condition
    if state.foundation_count == 52:
        st.success(f"🎉 **WINNER!** All 52 cards on foundation! Final Score: ${state.score}")
    elif not game.get_valid_moves() and len(state.stock) == 0:
        st.warning(f"😔 **Game Over** - No more valid moves. Final Score: ${state.score}")


def render_compact_board(game: Game) -> str:
    """
    Render a compact text representation of the board.

    Args:
        game: The game to render

    Returns:
        Text representation
    """
    state = game.state
    lines = []

    lines.append(f"Score: ${state.score} | Foundation: {state.foundation_count}/52 | Pass: {state.passes_through_deck}/3")
    lines.append(f"Stock: {len(state.stock)} | Waste: {len(state.waste)}")

    # Foundations
    foundation_counts = []
    for suit in [Suit.HEARTS, Suit.DIAMONDS, Suit.CLUBS, Suit.SPADES]:
        foundation_counts.append(f"{suit.name[0]}: {len(state.foundations[suit])}")
    lines.append("Foundations: " + " | ".join(foundation_counts))

    # Tableau
    tableau_info = []
    for i, column in enumerate(state.tableau):
        face_up = sum(1 for _, is_up in column if is_up)
        face_down = len(column) - face_up
        tableau_info.append(f"C{i+1}: {face_down}↓{face_up}↑")

    lines.append("Tableau: " + " | ".join(tableau_info))

    return "\n".join(lines)
