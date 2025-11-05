"""Interactive Play Mode for Vegas Solitaire."""

import streamlit as st
from game.core.game import Game
from game.core.moves import MoveType
from streamlit_app.components import render_game_board


def initialize_game():
    """Initialize a new game in session state."""
    seed = st.session_state.get('play_seed', 42)
    game = Game(seed=seed)
    game.deal()  # Deal the cards!
    st.session_state.play_game = game
    st.session_state.play_move_history = []
    st.session_state.play_move_count = 0


def show():
    """Display the Play Mode page."""
    st.title("🎮 Play Mode")
    st.markdown("Play Vegas Solitaire yourself! Make moves and try to beat the AI.")

    # Initialize game if not exists
    if 'play_game' not in st.session_state:
        initialize_game()

    game = st.session_state.play_game

    # Control panel
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        new_seed = st.number_input("Seed", min_value=0, max_value=10000, value=42, key='play_seed_input')
        if st.button("🎲 New Game", use_container_width=True):
            st.session_state.play_seed = new_seed
            initialize_game()
            st.rerun()

    with col2:
        if st.button("↩️ Undo", use_container_width=True, disabled=len(st.session_state.play_move_history) == 0):
            if st.session_state.play_move_history:
                st.session_state.play_move_history.pop()
                initialize_game()
                # Replay moves
                for move in st.session_state.play_move_history:
                    game.apply_move(move)
                st.session_state.play_move_count = len(st.session_state.play_move_history)
                st.rerun()

    with col3:
        if st.button("🔄 Reset", use_container_width=True):
            initialize_game()
            st.rerun()

    with col4:
        st.metric("Moves Made", st.session_state.play_move_count)

    st.markdown("---")

    # Render the game board
    render_game_board(game, show_score=True)

    # Move controls
    st.markdown("---")
    st.subheader("🎯 Make a Move")

    # Get valid moves
    valid_moves = game.get_valid_moves()

    if not valid_moves:
        st.warning("⚠️ No valid moves available!")
        if len(game.state.stock) == 0:
            st.error("Game Over! No more moves and stock is empty.")
        else:
            st.info("Try drawing from the stock pile.")
    else:
        st.info(f"✓ {len(valid_moves)} valid moves available")

        # Group moves by type
        move_groups = {
            MoveType.DRAW: [],
            MoveType.RECYCLE: [],
            MoveType.WASTE_TO_FOUNDATION: [],
            MoveType.WASTE_TO_TABLEAU: [],
            MoveType.TABLEAU_TO_FOUNDATION: [],
            MoveType.TABLEAU_TO_TABLEAU: [],
        }

        for move in valid_moves:
            move_groups[move.move_type].append(move)

        # Create tabs for different move types
        move_tabs = []
        move_tab_names = []

        if move_groups[MoveType.DRAW]:
            move_tab_names.append("📤 Draw")
            move_tabs.append(move_groups[MoveType.DRAW])

        if move_groups[MoveType.RECYCLE]:
            move_tab_names.append("🔄 Recycle")
            move_tabs.append(move_groups[MoveType.RECYCLE])

        if move_groups[MoveType.WASTE_TO_FOUNDATION]:
            move_tab_names.append("💰 Waste→Foundation")
            move_tabs.append(move_groups[MoveType.WASTE_TO_FOUNDATION])

        if move_groups[MoveType.WASTE_TO_TABLEAU]:
            move_tab_names.append("📋 Waste→Tableau")
            move_tabs.append(move_groups[MoveType.WASTE_TO_TABLEAU])

        if move_groups[MoveType.TABLEAU_TO_FOUNDATION]:
            move_tab_names.append("💎 Tableau→Foundation")
            move_tabs.append(move_groups[MoveType.TABLEAU_TO_FOUNDATION])

        if move_groups[MoveType.TABLEAU_TO_TABLEAU]:
            move_tab_names.append("↔️ Tableau→Tableau")
            move_tabs.append(move_groups[MoveType.TABLEAU_TO_TABLEAU])

        if move_tab_names:
            tabs = st.tabs(move_tab_names)

            for tab_idx, tab in enumerate(tabs):
                with tab:
                    moves_in_tab = move_tabs[tab_idx]

                    for move in moves_in_tab:
                        move_desc = describe_move(move, game)

                        if st.button(move_desc, key=f"move_{id(move)}", use_container_width=True):
                            # Apply the move
                            game.apply_move(move)
                            st.session_state.play_move_history.append(move)
                            st.session_state.play_move_count += 1
                            st.rerun()

    # Game statistics in sidebar
    with st.sidebar:
        st.markdown("### 📊 Game Statistics")
        st.metric("Current Score", f"${game.state.score}")
        st.metric("Foundation Cards", f"{game.state.foundation_count}/52")
        st.metric("Stock Remaining", len(game.state.stock))
        st.metric("Waste Cards", len(game.state.waste))
        st.metric("Pass Count", f"{game.state.passes_through_deck}/3")

        # Progress bar
        progress = game.state.foundation_count / 52
        st.progress(progress, text=f"Progress: {game.state.foundation_count}/52 cards")


def describe_move(move, game):
    """Generate a human-readable description of a move."""
    state = game.state

    if move.move_type == MoveType.DRAW:
        return "📤 Draw 3 cards from stock"

    elif move.move_type == MoveType.RECYCLE:
        return "🔄 Recycle waste back to stock"

    elif move.move_type == MoveType.WASTE_TO_FOUNDATION:
        card = state.waste[-1] if state.waste else None
        if card:
            return f"💰 {card} to Foundation"
        return "💰 Waste to Foundation"

    elif move.move_type == MoveType.WASTE_TO_TABLEAU:
        card = state.waste[-1] if state.waste else None
        if card:
            return f"📋 {card} to Tableau Column {move.to_position + 1}"
        return f"📋 Waste to Tableau Column {move.to_position + 1}"

    elif move.move_type == MoveType.TABLEAU_TO_FOUNDATION:
        column = state.tableau[move.from_position]
        if column:
            card = column[-1]  # Cards are just Card objects, not tuples
            return f"💎 {card} from Column {move.from_position + 1} to Foundation"
        return f"💎 Column {move.from_position + 1} to Foundation"

    elif move.move_type == MoveType.TABLEAU_TO_TABLEAU:
        column = state.tableau[move.from_position]
        if column and move.num_cards:
            if move.num_cards == 1:
                card = column[-1]  # Cards are just Card objects, not tuples
                return f"↔️ {card} from Column {move.from_position + 1} to Column {move.to_position + 1}"
            else:
                return f"↔️ {move.num_cards} cards from Column {move.from_position + 1} to Column {move.to_position + 1}"
        return f"↔️ Column {move.from_position + 1} to Column {move.to_position + 1}"

    return str(move)
