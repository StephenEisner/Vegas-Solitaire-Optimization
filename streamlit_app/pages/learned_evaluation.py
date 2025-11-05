"""
Machine-Generated Evaluation Functions page for Streamlit app.

Train neural networks to automatically learn evaluation functions
from self-play, rather than hand-crafting heuristics.
"""

import streamlit as st
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from game.core.game import Game
from optimization.solvers.heuristic_solver import HeuristicSolver
from optimization.evaluation.learned_evaluation import (
    TrainingDataGenerator,
    NeuralEvaluationFunction
)
from optimization.solvers.learned_solver import LearnedEvaluationSolver


def render_learned_evaluation():
    """Render the learned evaluation page."""
    st.title("🧠 Machine-Generated Evaluation")

    st.markdown("""
    Train neural networks to **automatically learn** evaluation functions
    from gameplay, rather than hand-crafting heuristics.

    This is true **machine learning** - the algorithm discovers patterns on its own!
    """)

    # Sidebar
    st.sidebar.header("⚙️ Configuration")

    mode = st.sidebar.radio(
        "Mode",
        ["📊 Generate Training Data", "🎓 Train Network", "🧪 Test & Compare", "📈 Analyze Model"]
    )

    # Main content based on mode
    if mode == "📊 Generate Training Data":
        render_generate_data_mode()
    elif mode == "🎓 Train Network":
        render_train_network_mode()
    elif mode == "🧪 Test & Compare":
        render_test_compare_mode()
    else:
        render_analyze_model_mode()


def render_generate_data_mode():
    """Generate training data from self-play."""
    st.subheader("📊 Generate Training Data")

    st.markdown("""
    Generate training data by playing games with existing solvers
    and recording state-action-reward trajectories.
    """)

    col1, col2 = st.columns(2)

    with col1:
        num_games = st.number_input(
            "Number of Games",
            min_value=10,
            max_value=1000,
            value=100,
            step=10
        )

        start_seed = st.number_input(
            "Starting Seed",
            min_value=0,
            max_value=100000,
            value=10000
        )

    with col2:
        st.info("""
        **Time Estimates:**
        - 10 games: ~30 seconds
        - 100 games: ~5 minutes
        - 1000 games: ~50 minutes

        More data = better learning!
        """)

    if st.button("🚀 Generate Training Data", type="primary"):
        with st.spinner("Playing games and recording trajectories..."):
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Create solvers
            solvers = [HeuristicSolver()]

            # Generate data
            generator = TrainingDataGenerator(solvers)

            # Simulate progress (simplified - full version would update per game)
            for i in range(num_games):
                if i % 10 == 0:
                    progress = i / num_games
                    progress_bar.progress(progress)
                    status_text.text(f"Generating game {i+1}/{num_games}...")

                # Generate one trajectory
                game = Game(seed=start_seed + i)
                game.deal()
                trajectory = generator.generate_trajectory(game, solvers[0])

            progress_bar.progress(1.0)
            status_text.text(f"✓ Generated {num_games} trajectories!")

            # Save dataset
            Path("data").mkdir(exist_ok=True)
            generator.save_dataset("data/training_trajectories.json")

            st.session_state.trajectories = generator.trajectories
            st.success(f"✓ Generated and saved {len(generator.trajectories)} game trajectories!")

            # Show summary
            display_trajectory_summary(generator.trajectories)


def render_train_network_mode():
    """Train neural network evaluation function."""
    st.subheader("🎓 Train Neural Network")

    st.markdown("""
    Train a deep neural network to predict game outcomes from state features.
    The network learns automatically from the training data!
    """)

    # Check for training data
    if not Path("data/training_trajectories.json").exists():
        st.warning("⚠️ No training data found. Please generate training data first!")
        return

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Network Architecture")

        hidden_layers = st.text_input(
            "Hidden Layers (comma-separated)",
            value="128,64,32"
        )

        target_type = st.selectbox(
            "Training Target",
            ["monte_carlo", "td", "final_outcome"],
            index=0,
            help="monte_carlo: sum of future rewards (recommended)"
        )

    with col2:
        st.markdown("### Training Parameters")

        learning_rate = st.number_input(
            "Learning Rate",
            min_value=0.0001,
            max_value=0.1,
            value=0.001,
            format="%.4f"
        )

        epochs = st.slider(
            "Epochs",
            min_value=5,
            max_value=100,
            value=20
        )

        batch_size = st.selectbox(
            "Batch Size",
            [16, 32, 64, 128],
            index=1
        )

    if st.button("🎓 Start Training", type="primary"):
        with st.spinner("Training neural network..."):
            # Load training data
            generator = TrainingDataGenerator([])
            trajectories = generator.load_dataset("data/training_trajectories.json")

            # Parse hidden dims
            hidden_dims = [int(x.strip()) for x in hidden_layers.split(',')]

            # Create network
            network = NeuralEvaluationFunction(hidden_dims=hidden_dims)

            # Train
            history = network.train_supervised(
                trajectories=trajectories,
                target=target_type,
                learning_rate=learning_rate,
                batch_size=batch_size,
                epochs=epochs,
                verbose=False
            )

            # Save model
            Path("models").mkdir(exist_ok=True)
            network.save("models/neural_eval.json")

            st.session_state.trained_network = network
            st.session_state.training_history = history

            st.success("✓ Training complete! Model saved.")

            # Show results
            display_training_results(history)


def render_test_compare_mode():
    """Test learned evaluation vs baselines."""
    st.subheader("🧪 Test & Compare")

    st.markdown("""
    Compare the learned evaluation function against baseline solvers
    to see if it's actually better!
    """)

    # Check for model
    if not Path("models/neural_eval.json").exists():
        st.warning("⚠️ No trained model found. Please train a network first!")
        return

    num_test_games = st.slider(
        "Number of Test Games",
        min_value=5,
        max_value=50,
        value=20
    )

    if st.button("🧪 Run Comparison", type="primary"):
        with st.spinner("Testing solvers..."):
            # Load network
            network = NeuralEvaluationFunction()
            network.load("models/neural_eval.json")

            learned_solver = LearnedEvaluationSolver(network)
            heuristic_solver = HeuristicSolver()

            # Test both solvers
            learned_results = test_solver(learned_solver, num_test_games, start_seed=2000)
            heuristic_results = test_solver(heuristic_solver, num_test_games, start_seed=2000)

            st.session_state.learned_results = learned_results
            st.session_state.heuristic_results = heuristic_results

            st.success("✓ Comparison complete!")

            # Display results
            display_comparison_results(learned_results, heuristic_results)


def render_analyze_model_mode():
    """Analyze what the model learned."""
    st.subheader("📈 Analyze Model")

    st.markdown("""
    Understand what patterns the neural network has learned.
    """)

    # Check for model
    if not Path("models/neural_eval.json").exists():
        st.warning("⚠️ No trained model found. Please train a network first!")
        return

    # Load network
    network = NeuralEvaluationFunction()
    network.load("models/neural_eval.json")

    st.markdown("### Model Architecture")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Input Features", network.input_dim)

    with col2:
        st.metric("Hidden Layers", len(network.hidden_dims))

    with col3:
        total_params = sum(
            layer['W'].size + layer['b'].size
            for layer in network.layers
        )
        st.metric("Total Parameters", f"{total_params:,}")

    st.markdown("### Test Predictions")

    st.markdown("See what the model predicts for different game states:")

    test_seed = st.number_input("Test Seed", min_value=0, max_value=100000, value=42)

    if st.button("🔮 Predict"):
        game = Game(seed=test_seed)
        game.deal()

        from optimization.features.state_features import StateFeatures
        features = StateFeatures.extract_features(game.state)

        # Predict
        features_norm = (features - network.input_mean) / network.input_std
        predicted_value = network.predict(features_norm)
        predicted_value = predicted_value * network.target_std + network.target_mean

        # Also get heuristic
        from optimization.meta.deal_evaluator import evaluate_deal_quality
        heuristic_quality = evaluate_deal_quality(game.state)

        col_a, col_b = st.columns(2)

        with col_a:
            st.metric("Neural Network Prediction", f"{predicted_value:.2f}")
            st.caption("Predicted expected reward")

        with col_b:
            st.metric("Heuristic Quality", f"{heuristic_quality:.1f}")
            st.caption("Hand-crafted score")

        # Show deal
        st.markdown("#### Deal Preview")
        st.text(f"Foundation cards: {game.state.get_foundation_count()}/52")
        st.text(f"Current score: ${game.state.score}")


def display_trajectory_summary(trajectories):
    """Display summary of generated trajectories."""
    st.markdown("### 📊 Dataset Summary")

    total_states = sum(len(t) for t in trajectories)
    avg_length = np.mean([len(t) for t in trajectories])

    scores = [t.final_outcome['score'] for t in trajectories]
    cards = [t.final_outcome['cards'] for t in trajectories]
    wins = [t.final_outcome['won'] for t in trajectories]

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Games", len(trajectories))

    with col2:
        st.metric("Total States", f"{total_states:,}")

    with col3:
        st.metric("Avg Length", f"{avg_length:.1f} moves")

    with col4:
        st.metric("Win Rate", f"{np.mean(wins)*100:.1f}%")

    st.markdown("#### Outcome Distribution")

    col_a, col_b = st.columns(2)

    with col_a:
        st.metric("Avg Score", f"${np.mean(scores):.1f}")
        st.caption(f"Range: ${np.min(scores):.0f} - ${np.max(scores):.0f}")

    with col_b:
        st.metric("Avg Cards", f"{np.mean(cards):.1f}/52")
        st.caption(f"Range: {np.min(cards)}-{np.max(cards)}")


def display_training_results(history):
    """Display training results."""
    st.markdown("### 📈 Training Results")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Final Training Loss", f"{history['loss'][-1]:.4f}")

    with col2:
        st.metric("Final Validation Loss", f"{history['val_loss'][-1]:.4f}")

    # Plot training curves
    st.markdown("#### Loss Curves")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(history['loss'], label='Training Loss', linewidth=2)
    ax.plot(history['val_loss'], label='Validation Loss', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (MSE)')
    ax.set_title('Training Progress')
    ax.legend()
    ax.grid(alpha=0.3)

    st.pyplot(fig)
    plt.close()


def display_comparison_results(learned_results, heuristic_results):
    """Display comparison between learned and heuristic solvers."""
    st.markdown("### 🏆 Performance Comparison")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🧠 Learned Evaluation")
        st.metric("Avg Cards", f"{learned_results['avg_cards']:.1f}/52")
        st.metric("Avg Score", f"${learned_results['avg_score']:.1f}")
        st.metric("Win Rate", f"{learned_results['win_rate']*100:.1f}%")

    with col2:
        st.markdown("#### 🧠 Heuristic Solver")
        st.metric("Avg Cards", f"{heuristic_results['avg_cards']:.1f}/52")
        st.metric("Avg Score", f"${heuristic_results['avg_score']:.1f}")
        st.metric("Win Rate", f"{heuristic_results['win_rate']*100:.1f}%")

    # Difference
    st.markdown("#### 📊 Difference")

    cards_diff = learned_results['avg_cards'] - heuristic_results['avg_cards']
    score_diff = learned_results['avg_score'] - heuristic_results['avg_score']

    col_a, col_b = st.columns(2)

    with col_a:
        st.metric(
            "Cards Difference",
            f"{cards_diff:+.1f}",
            delta=f"{cards_diff:+.1f} cards",
            delta_color="normal" if cards_diff > 0 else "inverse"
        )

    with col_b:
        st.metric(
            "Score Difference",
            f"${score_diff:+.1f}",
            delta=f"${score_diff:+.1f}",
            delta_color="normal" if score_diff > 0 else "inverse"
        )

    if cards_diff > 0:
        st.success(f"✓ Learned evaluation is better by {cards_diff:.1f} cards!")
    elif cards_diff < 0:
        st.warning(f"⚠️ Heuristic is better by {-cards_diff:.1f} cards. Try training on more data.")
    else:
        st.info("→ Both perform equally")


def test_solver(solver, num_games, start_seed):
    """Test a solver on multiple games."""
    scores = []
    cards = []
    wins = []

    for seed in range(start_seed, start_seed + num_games):
        game = Game(seed=seed)
        game.deal()

        moves = 0
        while moves < 500:
            move = solver.choose_move(game)
            if not move or not game.make_move(move):
                break
            moves += 1

        scores.append(game.state.score)
        cards.append(game.state.get_foundation_count())
        wins.append(game.state.get_foundation_count() == 52)

    return {
        'avg_score': np.mean(scores),
        'avg_cards': np.mean(cards),
        'win_rate': np.mean(wins),
        'scores': scores,
        'cards': cards
    }


if __name__ == "__main__":
    render_learned_evaluation()
