"""Watch Mode - Observe AI solvers playing Vegas Solitaire."""

import streamlit as st
import time
from game.core.game import Game
from optimization.solvers.random_solver import RandomSolver
from optimization.solvers.heuristic_solver import HeuristicSolver
from optimization.solvers.mcts_solver import MCTSSolver
from streamlit_app.components import render_game_board


def initialize_watch_game():
    """Initialize a new game for watching."""
    seed = st.session_state.get('watch_seed', 42)
    st.session_state.watch_game = Game(seed=seed)
    st.session_state.watch_move_history = []
    st.session_state.watch_is_playing = False
    st.session_state.watch_game_over = False


def show():
    """Display the Watch Mode page."""
    st.title("👀 Watch Mode")
    st.markdown("Watch AI solvers play and learn from their strategies!")

    # Initialize game if not exists
    if 'watch_game' not in st.session_state:
        initialize_watch_game()

    game = st.session_state.watch_game

    # Controls
    st.markdown("### ⚙️ Settings")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        solver_name = st.selectbox(
            "Solver",
            ["Random", "Heuristic", "MCTS (100 sims)", "MCTS (1000 sims)"],
            key='watch_solver_select'
        )

    with col2:
        seed = st.number_input("Seed", min_value=0, max_value=10000, value=42, key='watch_seed_input')

    with col3:
        speed = st.select_slider(
            "Speed",
            options=["Slow (2s)", "Normal (1s)", "Fast (0.5s)", "Very Fast (0.1s)"],
            value="Normal (1s)",
            key='watch_speed'
        )

    with col4:
        max_moves = st.number_input("Max Moves", min_value=10, max_value=2000, value=500, step=50, key='watch_max_moves')

    # Speed mapping
    speed_map = {
        "Slow (2s)": 2.0,
        "Normal (1s)": 1.0,
        "Fast (0.5s)": 0.5,
        "Very Fast (0.1s)": 0.1
    }
    move_delay = speed_map[speed]

    # Action buttons
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("🎬 New Game", use_container_width=True):
            st.session_state.watch_seed = seed
            initialize_watch_game()
            st.rerun()

    with col2:
        start_button = st.button(
            "▶️ Play" if not st.session_state.get('watch_is_playing', False) else "⏸️ Pause",
            use_container_width=True,
            disabled=st.session_state.get('watch_game_over', False)
        )
        if start_button:
            st.session_state.watch_is_playing = not st.session_state.get('watch_is_playing', False)
            st.rerun()

    with col3:
        if st.button("⏭️ Step", use_container_width=True, disabled=st.session_state.get('watch_game_over', False)):
            make_solver_move(game, solver_name)
            st.rerun()

    with col4:
        if st.button("⏩ Run to End", use_container_width=True, disabled=st.session_state.get('watch_game_over', False)):
            # Run until game over or max moves
            solver = create_solver(solver_name)
            moves_made = 0

            placeholder = st.empty()

            while moves_made < max_moves:
                valid_moves = game.get_valid_moves()

                if not valid_moves:
                    st.session_state.watch_game_over = True
                    break

                move = solver.choose_move(game)
                if not move:
                    st.session_state.watch_game_over = True
                    break

                game.apply_move(move)
                st.session_state.watch_move_history.append(move)
                moves_made += 1

                # Update every 10 moves
                if moves_made % 10 == 0:
                    placeholder.info(f"🎮 Played {moves_made} moves... Score: ${game.state.score}")

            placeholder.success(f"✅ Completed! Played {moves_made} moves. Final Score: ${game.state.score}")
            st.rerun()

    st.markdown("---")

    # Move counter and stats
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Moves Played", len(st.session_state.watch_move_history))
    with col2:
        st.metric("Current Score", f"${game.state.score}")
    with col3:
        if st.session_state.get('watch_game_over', False):
            st.metric("Status", "Game Over", delta="Complete")
        else:
            valid_moves = len(game.get_valid_moves())
            st.metric("Valid Moves", valid_moves)

    # Render game board
    render_game_board(game, show_score=True)

    # Auto-play logic
    if st.session_state.get('watch_is_playing', False) and not st.session_state.get('watch_game_over', False):
        time.sleep(move_delay)
        if make_solver_move(game, solver_name):
            st.rerun()
        else:
            st.session_state.watch_is_playing = False
            st.session_state.watch_game_over = True
            st.rerun()

    # Move history (last 10 moves)
    if st.session_state.watch_move_history:
        with st.expander("📜 Move History (Last 10)"):
            recent_moves = st.session_state.watch_move_history[-10:]
            for i, move in enumerate(reversed(recent_moves)):
                move_num = len(st.session_state.watch_move_history) - i
                st.text(f"{move_num}. {move}")


def create_solver(solver_name):
    """Create a solver instance based on name."""
    if solver_name == "Random":
        return RandomSolver(seed=None)  # Use game's randomness
    elif solver_name == "Heuristic":
        return HeuristicSolver()
    elif solver_name == "MCTS (100 sims)":
        return MCTSSolver(simulations_per_move=100, seed=None)
    elif solver_name == "MCTS (1000 sims)":
        return MCTSSolver(simulations_per_move=1000, seed=None)
    else:
        return RandomSolver(seed=None)


def make_solver_move(game, solver_name):
    """
    Make a single move with the selected solver.

    Returns:
        bool: True if move was made, False if game is over
    """
    valid_moves = game.get_valid_moves()

    if not valid_moves:
        return False

    solver = create_solver(solver_name)
    move = solver.choose_move(game)

    if not move:
        return False

    game.apply_move(move)
    st.session_state.watch_move_history.append(move)

    return True
